"""
train.py ─ Training & Evaluation Pipeline
==========================================
Features:
  • Chronological 80/10/10 split (no data leakage)
  • Automatic class-weight detection & application
  • AdamW + CosineAnnealingLR scheduler
  • Gradient clipping (max_norm=1.0)
  • Early stopping on validation loss
  • Best-checkpoint saving
  • Full test evaluation with classification report + confusion matrix

Usage:
  python train.py --symbol RELIANCE --backbone resnet18 --epochs 30
  python train.py --symbol HDFCBANK --backbone efficientnet_b0 --resolution structural
"""

import os
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from data_utils import build_datasets, compute_class_weights, CLASS_NAMES
from model import build_model

# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RGB Market Fingerprint CNN Trainer")
    p.add_argument("--data_dir",    default="minute",          help="Directory with CSV files")
    p.add_argument("--symbol",      default="RELIANCE",         help="Stock symbol to train on")
    p.add_argument("--backbone",    default="resnet18",
                   choices=["resnet18", "efficientnet_b0"])
    p.add_argument("--resolution",  default="micro",
                   choices=["micro", "structural"],
                   help="micro=64-min window | structural=240-min + PAA-60")
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--dropout",     type=float, default=0.3)
    p.add_argument("--workers",     type=int,   default=4,
                   help="DataLoader worker processes")
    p.add_argument("--patience",    type=int,   default=5,
                   help="Early stopping patience (epochs)")
    p.add_argument("--checkpoint",  default="checkpoints/best_model.pt")
    p.add_argument("--no_pretrain", action="store_true",
                   help="Train from scratch (no ImageNet weights)")
    p.add_argument("--no_weights",  action="store_true",
                   help="Disable automatic class-weight balancing")
    return p.parse_args()


# ─── Device ──────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    print(f"Device: {dev}")
    return dev


# ─── One Epoch ───────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device, epoch, total):
    model.train()
    running_loss, correct, n = 0.0, 0, 0
    t0 = time.time()

    for step, (imgs, labels) in enumerate(loader, 1):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs, return_logits=True)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs           = imgs.size(0)
        running_loss += loss.item() * bs
        correct      += (logits.argmax(1) == labels).sum().item()
        n            += bs

        if step % 200 == 0 or step == len(loader):
            print(f"  Epoch {epoch}/{total} | step {step:>5}/{len(loader)} "
                  f"| loss {running_loss/n:.4f} | acc {correct/n:.4f} "
                  f"| {time.time()-t0:.0f}s")

    return running_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, n = 0.0, 0, 0
    all_preds, all_labels    = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(imgs, return_logits=True)
        loss   = criterion(logits, labels)

        bs           = imgs.size(0)
        running_loss += loss.item() * bs
        preds         = logits.argmax(1)
        correct      += (preds == labels).sum().item()
        n            += bs

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return (running_loss / n, correct / n,
            np.array(all_preds), np.array(all_labels))


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = get_device()

    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)

    # ── Banner ────────────────────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print(f"  RGB Market Fingerprint CNN")
    print(f"  Symbol     : {args.symbol}")
    print(f"  Backbone   : {args.backbone}")
    print(f"  Resolution : {args.resolution}")
    print(f"  Epochs     : {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    print(f"{'═'*62}\n")

    # ── 1. Datasets ───────────────────────────────────────────────────────
    print("Building datasets…")
    train_ds, val_ds, test_ds = build_datasets(
        data_dir   = args.data_dir,
        symbol     = args.symbol,
        resolution = args.resolution,
    )

    # ── 2. Class weights (self-correcting imbalance) ──────────────────────
    if args.no_weights:
        weight_tensor = None
        print("  Class weighting disabled.")
    else:
        print("Computing class weights…")
        cw            = compute_class_weights(train_ds)
        weight_tensor = torch.tensor(cw, dtype=torch.float32, device=device)

    # ── 3. DataLoaders ────────────────────────────────────────────────────
    pw = args.workers > 0
    loader_kw = dict(
        batch_size       = args.batch_size,
        num_workers      = args.workers,
        pin_memory       = True,
        persistent_workers = pw,
        prefetch_factor  = 2 if pw else None,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kw)

    # ── 4. Model ──────────────────────────────────────────────────────────
    print("Building model…")
    model = build_model(
        backbone   = args.backbone,
        pretrained = not args.no_pretrain,
        dropout    = args.dropout,
        device     = device,
    )
    model.summary()

    # ── 5. Loss / Optimiser / Scheduler ───────────────────────────────────
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── 6. Training Loop ──────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    print("Starting training…\n")
    for epoch in range(1, args.epochs + 1):
        t_loss, t_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )
        v_loss, v_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  ▶ Epoch {epoch:>3}  "
              f"train_loss={t_loss:.4f}  train_acc={t_acc:.4f}  "
              f"val_loss={v_loss:.4f}  val_acc={v_acc:.4f}  "
              f"lr={lr_now:.2e}\n")

        history.append(dict(epoch=epoch,
                            train_loss=t_loss, train_acc=t_acc,
                            val_loss=v_loss,   val_acc=v_acc))

        # Checkpoint
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            patience_counter = 0
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "optim_state":  optimizer.state_dict(),
                "val_loss":     v_loss,
                "val_acc":      v_acc,
                "args":         vars(args),
            }, args.checkpoint)
            print(f"  ✔ Checkpoint saved  (val_loss={v_loss:.4f})\n")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping triggered after {epoch} epochs "
                      f"(patience={args.patience}).")
                break

    # ── 7. Test Evaluation ────────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print("Loading best checkpoint for test evaluation…")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])

    _, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)
    target_names = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]

    print(f"\nTest Accuracy : {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=target_names, digits=4))
    print("Confusion Matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(labels, preds)
    header = f"{'':10}" + "".join(f"{n:>10}" for n in target_names)
    print(header)
    for row_label, row in zip(target_names, cm):
        print(f"{row_label:10}" + "".join(f"{v:>10}" for v in row))

    # ── 8. Save training history ──────────────────────────────────────────
    import json
    hist_path = Path(args.checkpoint).parent / "history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {hist_path}")
    print(f"Best val_loss  : {best_val_loss:.4f}")
    print(f"Checkpoint     : {args.checkpoint}")
    print("Done. ✓")


if __name__ == "__main__":
    main()
