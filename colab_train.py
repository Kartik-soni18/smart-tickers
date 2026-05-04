import os
import time
import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from scipy.ndimage import zoom
from sklearn.metrics import classification_report, confusion_matrix
from torchvision.models import (
    resnet18, ResNet18_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
)

# ─── Constants ───────────────────────────────────────────────────────────────
MICRO_WINDOW    = 240       # Increased to 240 for more context
STRUCT_WINDOW   = 240       # 4 hours of context
PAA_BINS        = 60        # PAA output bins for structural resolution
IMAGE_SIZE      = 64        # final CNN image size (H = W)
LABEL_HORIZON   = 1800      # 5 trading days (assuming 375 min/day)
LABEL_THRESHOLD = 0.03      # 2% threshold for swing signals
CLASS_NAMES     = {0: "Sell", 1: "Hold", 2: "Buy"}

# ─── DATA UTILS (Merged from data_utils.py) ──────────────────────────────────

def load_ohlcv(path: str) -> pd.DataFrame:
    """Load a single symbol's 1-min OHLCV CSV, handles corrupt first rows and headers."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No data file at '{path}'")
    
    expected_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    
    # Read the file and filter for rows that look like data
    valid_lines = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) in [5, 6]:
                # Skip if it's a header or contains non-numeric price data in col 1
                if "date" in parts[0].lower() or parts[1].strip().isalpha():
                    continue
                valid_lines.append(line)
    
    if not valid_lines:
        raise ValueError(f"No valid data rows found in {path}")

    # Create DataFrame from valid lines, explicitly avoiding mixed dtype warnings
    from io import StringIO
    df = pd.read_csv(StringIO("".join(valid_lines)), header=None, low_memory=False)
    
    # Assign column names based on count
    if len(df.columns) == 6:
        df.columns = expected_cols
    elif len(df.columns) == 5:
        df.columns = expected_cols[:5]
        df['volume'] = 0.0 # Fill missing volume
    
    # Robust date parsing
    try:
        # Try to parse with standard ISO format first for speed
        df["date"] = pd.to_datetime(df[df.columns[0]], utc=True, errors='coerce').dt.tz_localize(None)
    except:
        # Fallback to slower inference if needed
        df["date"] = pd.to_datetime(df[df.columns[0]], errors='coerce').dt.tz_localize(None)
    
    # Drop rows where date or close price failed to parse
    df = df.dropna(subset=['date', 'close'])
    
    # Ensure numeric columns are actually numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.sort_values("date").reset_index(drop=True)
    return df

def paa(series: np.ndarray, n_bins: int) -> np.ndarray:
    n = len(series)
    if n == n_bins: return series.copy()
    if n % n_bins == 0: return series.reshape(n_bins, n // n_bins).mean(axis=1)
    cs  = np.concatenate([[0.0], np.cumsum(series)])
    idx = np.linspace(0, n, n_bins + 1)
    lo, hi = np.floor(idx).astype(int), np.ceil(idx).astype(int)
    frac = idx - lo
    hi   = np.minimum(hi, n)
    interp = cs[lo] + frac * (cs[hi] - cs[lo])
    return np.diff(interp) / np.diff(idx)

def zscore(series: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    z = (series - series.mean()) / (series.std() + eps)
    return np.clip(z, -3.0, 3.0)

class RGBMarketFingerprint:
    def __init__(self, image_size: int = IMAGE_SIZE, n_bins: int = 8, max_n: int = 1000):
        """
        Args:
            image_size: Final output H/W for the CNN.
            n_bins: Number of quantiles for MTF.
            max_n: Memory guard to prevent O(n^2) blowup on long series.
        """
        self.image_size = image_size
        self.n_bins     = n_bins
        self.max_n      = max_n

    @staticmethod
    def _minmax(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        span = hi - lo
        if span < 1e-9: return np.zeros_like(arr)
        return (arr - lo) / span

    @staticmethod
    def _rescale_to_minus1_plus1(series: np.ndarray) -> np.ndarray:
        lo, hi = series.min(), series.max()
        span = hi - lo
        if span < 1e-9: return np.zeros_like(series)
        return 2.0 * (series - lo) / span - 1.0

    def _gasf(self, x: np.ndarray) -> np.ndarray:
        """Optimized GASF using trig identities: cos(A+B) = cosA*cosB - sinA*sinB"""
        x = np.clip(x, -1.0, 1.0)
        sin_phi = np.sqrt(np.clip(1.0 - x**2, 0, 1))
        # Reuse outer products for performance
        return np.outer(x, x) - np.outer(sin_phi, sin_phi)

    def _gadf(self, x: np.ndarray) -> np.ndarray:
        """Optimized GADF using trig identities: sin(A-B) = sinA*cosB - cosA*sinB"""
        x = np.clip(x, -1.0, 1.0)
        sin_phi = np.sqrt(np.clip(1.0 - x**2, 0, 1))
        return np.outer(sin_phi, x) - np.outer(x, sin_phi)

    def _mtf(self, series: np.ndarray) -> np.ndarray:
        n = len(series)
        k = self.n_bins
        
        # 1. Handle bin collapse: Use unique quantiles
        quantiles = np.percentile(series, np.linspace(0, 100, k + 1))
        unique_q = np.unique(quantiles)
        
        # Fallback for constant or near-constant series
        if len(unique_q) < 2:
            return np.zeros((n, n))
            
        actual_k = len(unique_q) - 1
        
        # 2. Assign bins using side='left' (consistent with digitize)
        bins = np.digitize(series, unique_q[1:-1])
        bins = np.clip(bins, 0, actual_k - 1)
        
        # 3. Transition matrix Q
        Q = np.zeros((actual_k, actual_k), dtype=np.float64)
        np.add.at(Q, (bins[:-1], bins[1:]), 1.0)
        
        # 4. Normalize rows. Handle 'dead ends' (states only visited at the end)
        row_sums = Q.sum(axis=1, keepdims=True)
        zero_rows = (row_sums.flatten() == 0)
        if np.any(zero_rows):
            # Assign self-transition to avoid silent zero rows or NaNs
            for idx in np.where(zero_rows)[0]:
                Q[idx, idx] = 1.0
            row_sums[zero_rows] = 1.0
            
        W = Q / row_sums
        
        # 5. Map back to time-domain matrix
        return W[bins[:, None], bins[None, :]]

    def _resize(self, mat: np.ndarray, is_mtf: bool = False) -> np.ndarray:
        n = mat.shape[0]
        if n == self.image_size: return mat
        # MTF uses order=0 (nearest) to preserve Markov block structure
        # GASF/GADF use order=1 (bilinear) for smoothness
        return zoom(mat, self.image_size / n, order=0 if is_mtf else 1)

    def encode(self, price: np.ndarray, volume: np.ndarray) -> np.ndarray:
        # Memory Guard
        if len(price) > self.max_n:
            price = price[-self.max_n:]
            volume = volume[-self.max_n:]
            
        # Price matrices (R and G)
        xp = self._rescale_to_minus1_plus1(price)
        gasf_p = self._gasf(xp)
        gadf_p = self._gadf(xp)
        
        # Volume matrix (B) - Using MTF to capture volume regime transitions
        mtf_v  = self._mtf(volume)
        
        # Resize and combine
        rgb = np.stack([
            self._minmax(self._resize(gasf_p, is_mtf=False)),
            self._minmax(self._resize(gadf_p, is_mtf=False)),
            self._minmax(self._resize(mtf_v,  is_mtf=True)),
        ], axis=0)
        
        return rgb.astype(np.float32)

def compute_labels(close: np.ndarray, horizon: int = LABEL_HORIZON, threshold: float = LABEL_THRESHOLD) -> np.ndarray:
    n       = len(close)
    current = close[: n - horizon]
    future  = close[horizon:]
    pct     = (future - current) / (current + 1e-8)
    labels = np.zeros(len(pct), dtype=np.int64)
    labels[pct >  threshold] =  1
    labels[pct < -threshold] = -1
    return labels

class MarketFingerprintDataset(Dataset):
    def __init__(self, close, volume, labels, indices, resolution="micro", image_size=IMAGE_SIZE, transform=None):
        self.close      = close.astype(np.float64)
        self.volume     = volume.astype(np.float64)
        self.labels     = labels
        self.indices    = indices
        self.resolution = resolution
        self.transform  = transform
        self.window     = MICRO_WINDOW if resolution == "micro" else STRUCT_WINDOW
        self.encoder    = RGBMarketFingerprint(image_size=image_size)

    def __len__(self) -> int: return len(self.indices)

    def __getitem__(self, idx: int):
        start = int(self.indices[idx])
        
        # 1. Price Context (Rolling Z-Score)
        p_lookback = self.close[max(0, start - 200) : start]
        p_mean, p_std = p_lookback.mean(), p_lookback.std()
        p_window = (self.close[start : start + self.window] - p_mean) / (p_std + 1e-8)
        p_window = np.clip(p_window, -3.0, 3.0)
        
        # 2. Volume Context (Log-transformed Z-Score)
        v_lookback = np.log1p(self.volume[max(0, start - 200) : start])
        v_mean, v_std = v_lookback.mean(), v_lookback.std()
        v_window = (np.log1p(self.volume[start : start + self.window]) - v_mean) / (v_std + 1e-8)
        v_window = np.clip(v_window, -3.0, 3.0)
        
        if self.resolution == "structural":
            p_window = paa(p_window, PAA_BINS)
            v_window = paa(v_window, PAA_BINS)
            
        image = self.encoder.encode(p_window, v_window)
        label = int(self.labels[start]) + 1
        img_tensor = torch.from_numpy(image)
        if self.transform: img_tensor = self.transform(img_tensor)
        return img_tensor, torch.tensor(label, dtype=torch.long)

def build_datasets(csv_path, resolution="micro", train_ratio=0.80, val_ratio=0.10, image_size=IMAGE_SIZE):
    window = MICRO_WINDOW if resolution == "micro" else STRUCT_WINDOW
    df     = load_ohlcv(csv_path)
    close  = df["close"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    labels = compute_labels(close)
    
    lookback_pad = 200
    max_idx = min(len(labels), len(close) - window)
    indices = np.arange(lookback_pad, max_idx)
    
    if len(indices) <= 0:
        raise ValueError(f"Not enough data in {csv_path} for window+lookback")

    n = len(indices)
    train_idx = indices[:int(n * train_ratio)]
    val_idx   = indices[int(n * train_ratio):int(n * (train_ratio + val_ratio))]
    test_idx  = indices[int(n * (train_ratio + val_ratio)):]
    
    train_ds = MarketFingerprintDataset(close, volume, labels, train_idx, resolution, image_size)
    val_ds   = MarketFingerprintDataset(close, volume, labels, val_idx,   resolution, image_size)
    test_ds  = MarketFingerprintDataset(close, volume, labels, test_idx,  resolution, image_size)
    return train_ds, val_ds, test_ds

def build_multi_stock_datasets(data_path, resolution="micro", train_ratio=0.8, val_ratio=0.1, image_size=IMAGE_SIZE):
    path = Path(data_path)
    if path.is_file():
        print(f"Loading single symbol: {path.name}")
        return build_datasets(str(path), resolution, train_ratio, val_ratio, image_size)
    
    csv_files = sorted(list(path.glob("*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_path}")
        
    train_datasets, val_datasets, test_datasets = [], [], []
    
    print(f"Loading {len(csv_files)} symbols from {data_path}...")
    for f in csv_files:
        try:
            t_ds, v_ds, te_ds = build_datasets(str(f), resolution, train_ratio, val_ratio, image_size)
            train_datasets.append(t_ds)
            val_datasets.append(v_ds)
            test_datasets.append(te_ds)
        except Exception as e:
            print(f"  [Skip] {f.name}: {e}")
            
    return (
        ConcatDataset(train_datasets),
        ConcatDataset(val_datasets),
        ConcatDataset(test_datasets)
    )

def compute_class_weights(dataset) -> np.ndarray:
    if isinstance(dataset, ConcatDataset):
        all_counts = np.zeros(3, dtype=np.float64)
        for ds in dataset.datasets:
            raw = ds.labels[ds.indices]
            all_counts += np.bincount((raw + 1).astype(np.int64), minlength=3)
        counts = all_counts
    else:
        raw     = dataset.labels[dataset.indices]
        counts  = np.bincount((raw + 1).astype(np.int64), minlength=3).astype(np.float64)
    
    counts  = np.maximum(counts, 1.0)
    weights = counts.sum() / (3.0 * counts)
    return weights.astype(np.float32)

# ─── MODEL (Merged from model.py) ────────────────────────────────────────────

class MarketFingerprintCNN(nn.Module):
    def __init__(self, backbone: Literal["resnet18", "efficientnet_b0"] = "resnet18", num_classes: int = 3, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        if backbone == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            base    = resnet18(weights=weights)
            base.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            base.maxpool = nn.Identity()
            base.fc  = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(base.fc.in_features, num_classes))
            self.model = base
        elif backbone == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base    = efficientnet_b0(weights=weights)
            base.classifier = nn.Sequential(nn.Dropout(p=dropout, inplace=True), nn.Linear(base.classifier[1].in_features, num_classes))
            self.model = base
        self._softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor, return_logits: bool = True) -> torch.Tensor:
        logits = self.model(x)
        return logits if return_logits else self._softmax(logits)

def build_model(backbone="resnet18", pretrained=True, dropout=0.3, device=None):
    model = MarketFingerprintCNN(backbone=backbone, pretrained=pretrained, dropout=dropout)
    if device: model = model.to(device)
    return model

# ─── TRAINING LOOP (Merged from train.py) ────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device, epoch, total):
    model.train()
    running_loss, correct, n = 0.0, 0, 0
    t0 = time.time()
    for step, (imgs, labels) in enumerate(loader, 1):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        bs = imgs.size(0)
        running_loss += loss.item() * bs
        correct += (logits.argmax(1) == labels).sum().item()
        n += bs
        if step % 100 == 0 or step == len(loader):
            print(f"  Epoch {epoch}/{total} | step {step}/{len(loader)} | loss {running_loss/n:.4f} | acc {correct/n:.4f} | {time.time()-t0:.0f}s")
    return running_loss / n, correct / n

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss, correct, n = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   += criterion(logits, labels).item() * imgs.size(0)
        preds   = logits.argmax(1)
        correct += (preds == labels).sum().item()
        n       += imgs.size(0)
        all_preds.extend(preds.cpu().numpy()); all_labels.extend(labels.cpu().numpy())
    return loss/n, correct/n, np.array(all_preds), np.array(all_labels)

@torch.no_grad()
def backtest_engine(model, dataset, device, confidence_threshold=0.6, commission=0.0005):
    model.eval()
    
    # Handle ConcatDataset by recursing or processing sub-datasets
    if isinstance(dataset, ConcatDataset):
        print(f"Running multi-symbol backtest on {len(dataset.datasets)} symbols...")
        total_pnl = []
        total_trades = 0
        for ds in dataset.datasets:
            pnl_arr, trades = _run_sim(model, ds, device, confidence_threshold, commission)
            total_pnl.extend(pnl_arr)
            total_trades += trades
    else:
        pnl_arr, total_trades = _run_sim(model, dataset, device, confidence_threshold, commission)
        total_pnl = pnl_arr

    total_pnl = np.array(total_pnl)
    cum_pnl = np.cumsum(total_pnl)
    win_rate = (total_pnl > 0).sum() / total_trades if total_trades > 0 else 0
    
    print(f"\n{'='*40}")
    print(f"       AGGREGATE BACKTEST (Threshold: {confidence_threshold})")
    print(f"{'='*40}")
    print(f"  Total Trades:   {total_trades:,}")
    print(f"  Win Rate:       {win_rate:.2%}")
    print(f"  Final PnL:      {cum_pnl[-1]:.2%}" if len(cum_pnl)>0 else "  Final PnL: 0.0%")
    print(f"  Avg PnL/Trade:  {total_pnl[total_pnl != 0].mean():.4%}" if total_trades > 0 else "  Avg PnL/Trade: 0.0%")
    print(f"{'='*40}\n")
    return cum_pnl

def _run_sim(model, ds, device, threshold, commission):
    loader = DataLoader(ds, batch_size=128, shuffle=False)
    all_probs = []
    for imgs, _ in loader:
        logits = model(imgs.to(device))
        all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    
    probs = np.concatenate(all_probs, axis=0)
    close = ds.close
    indices = ds.indices
    window = ds.window
    horizon = LABEL_HORIZON
    
    pnl = []
    trades = 0
    for i, start_idx in enumerate(indices):
        p = probs[i]
        entry_price = close[start_idx + window - 1]
        exit_idx = start_idx + window - 1 + horizon
        exit_price = close[min(exit_idx, len(close)-1)]
        ret = (exit_price - entry_price) / entry_price
        
        if p[2] > threshold: # Buy
            pnl.append(ret - commission); trades += 1
        elif p[0] > threshold: # Sell
            pnl.append(-ret - commission); trades += 1
        else:
            pnl.append(0.0)
    return pnl, trades

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",  default="CDSL.csv", help="Path to single CSV or directory of CSVs")
    p.add_argument("--backbone",   default="resnet18", choices=["resnet18", "efficientnet_b0"])
    p.add_argument("--resolution", default="micro", choices=["micro", "structural"])
    p.add_argument("--epochs",     type=int,   default=10)
    p.add_argument("--batch_size", type=int,   default=128)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--workers",    type=int,   default=4)
    p.add_argument("--checkpoint", default="best_model.pt")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Building multi-stock datasets...")
    train_ds, val_ds, test_ds = build_multi_stock_datasets(args.data_path, args.resolution)
    print(f"Total samples | train: {len(train_ds):,}, val: {len(val_ds):,}, test: {len(test_ds):,}")
    weights = compute_class_weights(train_ds)
    
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader   = DataLoader(val_ds, shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds, shuffle=False, **loader_kw)

    print("Building model...")
    model = build_model(args.backbone, device=device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs)
        v_loss, v_acc, _, _ = evaluate(model, val_loader, criterion, device)
        print(f"  ▶ Epoch {epoch} | train_loss: {t_loss:.4f} | val_loss: {v_loss:.4f} | val_acc: {v_acc:.4f}\n")
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), args.checkpoint)
            print(f"  ✔ Saved best model (val_loss: {v_loss:.4f})")

    print("\nFinal Test Evaluation...")
    model.load_state_dict(torch.load(args.checkpoint))
    _, t_acc, preds, labels = evaluate(model, test_loader, criterion, device)
    print(f"Test Accuracy: {t_acc:.4f}")
    print(classification_report(labels, preds, target_names=["Sell", "Hold", "Buy"]))

    print("\nRunning Backtest Simulation on Test Set...")
    # For backtesting, we usually check individual symbols or the whole test pool
    backtest_engine(model, test_ds, device, confidence_threshold=0.6)

if __name__ == "__main__":
    main()
