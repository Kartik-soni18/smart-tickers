# RGB Market Fingerprint — CNN-Based Market State Classifier

> **"Every market moment leaves a fingerprint."**
>
> A research-grade pipeline that encodes 1-minute OHLCV data into 3-channel RGB
> images and classifies 15-minute price movements using a CNN.

---

## 1. The RGB-Market-Fingerprint Concept

Traditional technical indicators (RSI, MACD, Bollinger Bands) are scalar signals
that a trader reads one at a time. The **RGB-Market-Fingerprint** approach instead
encodes the **full temporal geometry** of a price window as a single image, letting
a CNN learn patterns that no hand-crafted indicator can capture.

### Why Images?

A raw 1-D time series is a sequence of numbers. A Gramian Angular Field or Markov
Transition Field converts that sequence into a 2-D matrix where:

- **Spatial position** encodes temporal relationships between every pair of time steps.
- **Pixel intensity** encodes the magnitude and direction of those relationships.
- **The CNN** can detect multi-scale patterns — from micro-structure noise to
  macro-trend structure — simultaneously, just as a human eye reads a chart.

### Channel Encoding

| Channel | Transform | What it captures |
|---------|-----------|-----------------|
| **R — Red** | **GASF** (Gramian Angular Summation Field) | Trend direction, momentum, symmetry of price movements |
| **G — Green** | **GADF** (Gramian Angular Difference Field) | Volatility, mean-reversion, oscillation patterns |
| **B — Blue** | **MTF** (Markov Transition Field) | Probabilistic transitions between price quantiles (regime changes) |

Stacking these three channels gives a single **64×64 RGB image** that is a
"fingerprint" of the market's state at a given minute — no information is
discarded, no indicator is applied.

---

## 2. Architecture

```
1-min OHLCV CSV
       │
       ▼
┌─────────────────────────────────────────┐
│  Sliding Window  (stride = 1 min)       │
│  ├─ Micro       : 64-min  raw window    │
│  └─ Structural  : 240-min → PAA-60 bins │
└─────────────────────────────────────────┘
       │
       ▼  Z-score normalise per window
       │
       ▼
┌─────────────────────────────────────────┐
│  RGBMarketFingerprint Encoder           │
│  R: GASF  │  G: GADF  │  B: MTF        │
│  Output: float32 (3, 64, 64) ∈ [0,1]   │
└─────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  MarketFingerprintCNN                   │
│  ├─ ResNet-18   (default, fast)         │
│  │   conv1: 3×3 stride-1 (64×64 safe)  │
│  │   maxpool removed                   │
│  └─ EfficientNet-B0  (higher accuracy) │
│  Output: 3-class logits                 │
└─────────────────────────────────────────┘
       │
       ▼
  {0=Sell, 1=Hold, 2=Buy}
```

---

## 3. Labeling Logic (T+15)

At each minute `t`, we look 15 candles into the future:

```
pct_change = (close[t+15] - close[t]) / close[t]

if pct_change >  +0.05%  →  Buy  (+1  →  class 2)
if pct_change <  −0.05%  →  Sell (−1  →  class 0)
else                      →  Hold ( 0  →  class 1)
```

The ±0.05% dead-zone accounts for realistic round-trip transaction costs
(brokerage, slippage, impact). Labels below this threshold are noise.

---

## 4. Data Pipeline (Scalability)

The dataset for a single liquid stock spans **~635,000 minutes** (~5 years at
1-min resolution). With a stride of 1 minute, this yields **~635,000 training
windows**. Across 1,000+ symbols, the theoretical pool exceeds **1.8 billion**
windows.

**Solution:** `MarketFingerprintDataset` is a `torch.utils.data.Dataset` that
generates images **on-the-fly** inside `__getitem__`. No images are saved to disk.

```python
# Zero disk footprint — images computed at training time
dataset = MarketFingerprintDataset(close, labels, indices, resolution='micro')
loader  = DataLoader(dataset, batch_size=64, num_workers=4)
```

**Z-score normalisation** is applied per window *before* encoding, making the
fingerprint scale-invariant (₹50 stock vs ₹5000 stock look the same if their
*patterns* are the same).

---

## 5. Resolutions

| Resolution | Window | Preprocessing | Image |
|------------|--------|--------------|-------|
| **Micro** | 64 minutes | None (raw) | 64×64 |
| **Structural** | 240 minutes | PAA → 60 bins | 64×64 |

**Micro** captures short-term momentum and noise structure.  
**Structural** captures intraday regime patterns (e.g., opening range, lunch lull, closing rush).

---

## 6. Class Imbalance Self-Correction

Financial markets have far more *Hold* periods than clear directional moves.
The pipeline **automatically detects** imbalance and applies inverse-frequency
weights to the `CrossEntropyLoss`:

```
w_class = N_total / (n_classes × N_class)
```

This prevents the model from degenerating to a "predict Hold always" strategy.

---

## 7. Project Structure

```
stonks/
├── minute/              # 1,215 CSV files (1-min OHLCV)
├── data_utils.py        # Data pipeline, encoding, labeling, splits
├── model.py             # ResNet-18 / EfficientNet-B0 CNN
├── train.py             # Training loop, evaluation, checkpointing
├── image_encoder.py     # Original encoding utilities (pyts wrappers)
├── interface.py         # StockDataInterface (data loading API)
├── demo_analysis.py     # SMA, RSI, resampling demo
├── demo_encoding.py     # Image encoding demo
├── checkpoints/         # Saved model weights (created at training time)
│   ├── best_model.pt
│   └── history.json
└── README.md
```

---

## 8. Quick Start

### Install dependencies

```bash
# Activate venv first
source venv/bin/activate

# Already installed: pyts, numpy, pandas, scikit-learn, torch, torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Train on a single symbol

```bash
# Default: ResNet-18, micro resolution, RELIANCE
python train.py --symbol RELIANCE

# EfficientNet-B0, structural resolution
python train.py --symbol HDFCBANK --backbone efficientnet_b0 --resolution structural

# Longer training, more workers
python train.py --symbol TCS --epochs 50 --batch_size 128 --workers 8
```

### Available CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--symbol` | `RELIANCE` | Stock symbol (must match CSV filename) |
| `--backbone` | `resnet18` | `resnet18` or `efficientnet_b0` |
| `--resolution` | `micro` | `micro` (64-min) or `structural` (240-min + PAA) |
| `--epochs` | `30` | Training epochs |
| `--batch_size` | `64` | Batch size |
| `--lr` | `3e-4` | Initial learning rate (AdamW) |
| `--patience` | `5` | Early stopping patience |
| `--workers` | `4` | DataLoader parallel workers |
| `--no_pretrain` | `False` | Train from scratch |
| `--no_weights` | `False` | Disable class-weight balancing |
| `--checkpoint` | `checkpoints/best_model.pt` | Output checkpoint path |

---

## 9. Training Details

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Optimiser | AdamW | Weight decay prevents overfitting on small financial datasets |
| Scheduler | CosineAnnealingLR | Smooth LR decay; avoids sharp restarts |
| Loss | CrossEntropyLoss + class weights | Handles label imbalance automatically |
| Regularisation | Dropout (0.3) + gradient clipping (1.0) | Prevents co-adaptation and exploding gradients |
| Split | Chronological 80/10/10 | **No data leakage** — test data is always the most recent |

---

## 10. Extending to Multi-Symbol Training

```python
from data_utils import load_multi_symbol, MarketFingerprintDataset, compute_labels
import numpy as np

# Load and concatenate multiple symbols
df = load_multi_symbol("minute", max_symbols=50)

# Build one large dataset across all symbols
frames = []
for sym, grp in df.groupby("symbol"):
    close  = grp["close"].values
    labels = compute_labels(close)
    n      = min(len(labels), len(close) - 64)
    frames.append((close, labels, np.arange(n)))

# Each symbol's indices are independent — no cross-contamination
```

---

## 11. Theoretical Foundation

| Transform | Formula | Interpretation |
|-----------|---------|----------------|
| **GASF** | `G[i,j] = cos(φᵢ + φⱼ)` | Summation of angular projections |
| **GADF** | `G[i,j] = sin(φᵢ − φⱼ)` | Difference of angular projections |
| **MTF**  | `M[i,j] = Q_{i→j}` along diagonal | Markov transition probability between quantile bins at (i,j) |

Where `φ = arccos(x̃)` is the angular encoding of the Z-score normalised series `x̃`.

**Key property:** GAF matrices are symmetric, encoding *pairwise* temporal
correlations rather than just sequential dependencies. MTF encodes the
*probabilistic regime* structure — capturing phenomena like mean-reversion
and momentum that are invisible to pure price-based indicators.

---

*Built with PyTorch · pyts · scikit-learn · pandas · numpy*
