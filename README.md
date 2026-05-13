# DTNN — Differential Transformer Neural Network

**English** | [中文](README_zh.md)

Code for the paper **"Predicting High-Frequency Stock Movement with Differential Transformer Neural Network"** ([Electronics 2023, 12(13), 2943](https://doi.org/10.3390/electronics12132943)).

DTNN is a hybrid deep learning architecture for high-frequency stock market prediction. It combines three parallel encoders — a Temporal Convolutional Network (TCN), a Temporal Attention-Based Layer (TABL), and a raw identity passthrough — whose outputs are concatenated, sequentially differenced along the time axis, and fed through a Transformer encoder with a class token for classification.

## Architecture

```
Input: (batch, time_steps, features)
  |
  |-- TCN branch ──────────┐
  |-- TABL branch ─────────┤── concat ── linear embed ── sequential differencing
  |-- Identity passthrough ─┘                                |
                                                             v
                                              CLS token + positional embed
                                                             |
                                                       Transformer
                                                             |
                                                          MLP head
                                                             |
                                                  3-class softmax output
```

## Installation

```bash
# Clone the repository
git clone https://github.com/j00000ker/DTNN.git
cd DTNN

# Install in editable mode
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

### Dependencies

- Python >= 3.8
- PyTorch >= 1.9.0
- NumPy, Pandas, SciPy, scikit-learn
- einops
- Matplotlib, tqdm

## Quick Start

```python
import torch
from dtnn import DTNN

# Create model: 100 time steps, 60 features, 3 output classes
model = DTNN(
    time_slices=100,
    num_classes=3,
    dim=60,
    depth=3,
    heads=32,
)

# Forward pass
x = torch.randn(4, 100, 60)  # (batch=4, time=100, features=60)
output = model(x)             # (4, 3) — probability distribution
```

### Baselines

The package also includes baseline models used in the paper:

```python
from dtnn import LSTM, CNN, CNN_LSTM, MLP, SVM, C_TABL

model = LSTM(time_slices=100, dim=60, num_classes=3)
```

## Training

The `train.py` script reproduces the experiments from the paper:

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

Arguments:

| Argument | Default | Description |
|---|---|---|
| `--data-path` | `''` | Directory containing `.txt` data files |
| `--batch-size` | 64 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--epochs` | 150 | Number of training epochs |
| `--depth` | 3 | Transformer depth |
| `--heads` | 32 | Attention heads |
| `--k` | 1 | Label column index |
| `--T` | 100 | Time window length |
| `--seed` | 42 | Random seed |
| `--use-sampler` | off | Enable weighted sampling for class imbalance |

## Data Format

The training script expects text files where each column is a time series feature and rows represent time steps. The data loading pipeline:

1. Takes the first 40 rows as base features
2. Computes pairwise products of adjacent columns (20 additional features)
3. Uses the last 5 rows as label basis
4. Creates sliding windows of length `T` for training

## TCN Benchmarks

The `TCN/` directory contains standard sequence modeling benchmarks using the Temporal Convolutional Network, including:

- Adding problem
- Copy memory
- Sequential / permuted MNIST
- Character-level language modeling (Penn Treebank)
- Word-level language modeling (Penn Treebank / WikiText-103)
- Polyphonic music (JSB Chorales, Nottingham, MuseData, Piano-midi)
- LAMBADA textual understanding

## Citation

If you use this code in your research, please cite:

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

MIT License — see [LICENSE](LICENSE) for details.
