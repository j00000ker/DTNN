# DTNN — Differential Transformer Neural Network

**English** | [中文](README_zh.md)

Code for the paper **"Predicting High-Frequency Stock Movement with Differential Transformer Neural Network"** ([Electronics 2023, 12(13), 2943](https://doi.org/10.3390/electronics12132943)).

See [PAPER.md](PAPER.md) for the full paper summary (architecture details, experimental results, ablation study).

## Architecture

```
LOB Input: (batch, T=100, features=40→60)
  |
  |-- TCN branch ──────────┐
  |-- TABL branch ─────────┤── concat ── linear embed ── Differential Layer
  |-- Identity passthrough ─┘                                    |
                                                                  v
                                                   CLS token + position embed
                                                                  |
                                                            Transformer
                                                                  |
                                                               MLP head
                                                                  |
                                                       3-class logits output
```

Three parallel encoders (TCN, TABL, identity) are concatenated, projected, and sequentially differenced along the time axis. A CLS token and positional embedding are added, processed by a Transformer encoder, and the CLS output goes through an MLP head for 3-class classification. The **differential layer** is the core contribution — it computes first-order differences between adjacent time slices so the Transformer attends to state *changes* rather than absolute positions.

## Installation

```bash
git clone https://github.com/j00000ker/DTNN.git
cd DTNN
pip install -e .
```

Requires Python >= 3.8, PyTorch >= 1.9.0, NumPy, Pandas, SciPy, scikit-learn, einops, Matplotlib, tqdm.

## Quick Start

```python
import torch
from dtnn import DTNN

model = DTNN(time_slices=100, num_classes=3, dim=60, depth=3, heads=32)

x = torch.randn(4, 100, 60)   # (batch=4, time=100, features=60)
output = model(x)              # (4, 3) — raw logits (use with CrossEntropyLoss)
```

Baseline models from the paper are also included:

```python
from dtnn import LSTM, CNN, CNN_LSTM, MLP, SVM, C_TABL
```

## Training

```bash
python train.py \
    --data-path ./data/ \
    --epochs 150 \
    --batch-size 64 \
    --lr 1e-4 \
    --depth 3 \
    --heads 32 \
    --model-name dtnn_experiment
```

| Argument | Default | Description |
| --- | --- | --- |
| `--data-path` | `''` | Directory containing `.txt` data files |
| `--batch-size` | 64 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--epochs` | 150 | Training epochs |
| `--depth` | 3 | Transformer layers |
| `--heads` | 32 | Attention heads |
| `--k` | 1 | Label column index |
| `--T` | 100 | Time window length |
| `--seed` | 42 | Random seed |
| `--use-sampler` | off | Weighted sampling for class imbalance |

## Data Format

The input is LOB data with 40 features per time slice (10 levels × {ask_price, ask_volume, bid_price, bid_volume}). Labels are 3-class (up / unchanged / down), defined by whether the average return rate over horizon k exceeds a threshold.

The data pipeline: raw `.txt` → `prepare_x` (40 features + 20 pairwise interaction features = 60) → sliding windows of length T → `StockDataset` (converts 1-indexed labels to 0-indexed).

## TCN Benchmarks

The `TCN/` directory contains standard TCN sequence modeling benchmarks (Adding Problem, Copy Memory, Sequential/Permuted MNIST, character/word-level language modeling, polyphonic music, LAMBADA).

## Citation

```bibtex
@article{Lai2023DTNN,
  title   = {Predicting High-Frequency Stock Movement with Differential Transformer Neural Network},
  author  = {Lai, Shijie and Wang, Mingxian and Zhao, Shengjie and Arce, Gonzalo R.},
  journal = {Electronics},
  volume  = {12},
  number  = {13},
  pages   = {2943},
  year    = {2023},
  doi     = {10.3390/electronics12132943},
}
```

## License

MIT — see [LICENSE](LICENSE).
