# DTNN Paper Summary

**English** | [中文](PAPER_zh.md)

> Lai, S., Wang, M., Zhao, S., & Arce, G. R. (2023). Predicting High-Frequency Stock Movement with Differential Transformer Neural Network. *Electronics, 12*(13), 2943. https://doi.org/10.3390/electronics12132943

## Problem

Predicting stock price movement from high-frequency Limit Order Book (LOB) data. LOB data is extremely noisy with a very low signal-to-noise ratio — tiny differences between adjacent time slices are easily overwhelmed by the magnitude of the state vectors. Most existing methods let models learn patterns directly from raw (or normalized) data, which is difficult due to the non-stationarity and low SNR of financial time series.

## Architecture

DTNN consists of three modules:

### 1. Feature Extractor

Two parallel denoising branches with a residual identity connection:

- **TABL** (Temporal Attention-augmented Bilinear Layer): `X̃ = β(X ⊙ A) + (1−β)X`, `Y = ReLU(X̃ W₂ + B)`. Learns bilinear projections with 2D attention over both time and feature axes, capturing global sequence patterns.
- **TCN** (Temporal Convolutional Network): Causal dilated convolutions (dilation = 2ⁱ), extracting local features and suppressing noise. Kernel size k=2, three layers with channels = [120, 120, 120] (when dim=60).
- **Identity passthrough**: Preserves the raw signal.

The three outputs are concatenated and projected to a unified dimension via a linear layer (weight initialized with `eye_` so the raw signal passes through unchanged at the start of training).

### 2. Differential Layer (core contribution)

Reformulates the feature sequence from absolute states `(s₁, s₂, ..., sₙ)` to differential form `(s₁, Δ₂, Δ₃, ..., Δₙ)`, where `Δᵢ = sᵢ − sᵢ₋₁`.

**Why it works:** In high-frequency LOB data, the difference `Δᵢ` between adjacent state vectors is negligible compared to `sᵢ`. In a standard Transformer, the attention coefficient `c(sᵢ + pᵢ)` is dominated by the position embedding, causing the model to learn position patterns rather than state changes. After differencing, the attention becomes `c(Δᵢ + pᵢ − pᵢ₋₁)`, where `Δᵢ` cannot be neglected — forcing the model to capture actual state transitions.

**Unit root tests:** Only 6% of raw data passes stationarity tests → 82.2% after feature extraction → **98.3%** after the differential layer.

### 3. Prediction Transformer Module

Learnable CLS token + positional embeddings → multi-head self-attention (depth=3, heads=32, dim_head=64) → extract CLS output only → MLP (LayerNorm → Linear → activation) → 3-class classification.

## Experimental Setup

### Datasets

**FI-2010**: Public LOB benchmark, 5 Nasdaq Nordic stocks, 10 days, ~4M events. 40 features per time slice (10 levels × {ask_price, ask_volume, bid_price, bid_volume}). T=100 slices, prediction horizons k=10, 20, 50, 100. First 7 days for training/validation, last 3 days for testing.

**Chinese real stock data**: Two groups — Group 1 uses 10 stocks, 7 days training / 3 days testing; Group 2 uses 100 stocks for 10 days training, another 5 stocks for 5 days testing.

### Label Definition

`pₜ = (pₐ₍₁₎(t) + p_b(1)(t)) / 2` (mid-price), `m₊(t) = (1/k) Σ_{i=1}^{k} p_{t+i}` (average price over next k steps), `lₜ = (m₊(t) − pₜ) / pₜ` (average return rate). With threshold α: `lₜ > α` → up; `−α < lₜ < α` → unchanged; `lₜ < −α` → down.

### Baselines

SVM, MLP, CNN, LSTM, CNN-LSTM, TABL, DeepLOB, DeepLOB-Attention.

## Results

### FI-2010 (F1 score, %)

| Model | k=10 | k=20 | k=50 | k=100 |
|---|---|---|---|---|
| SVM | 59.07 | 49.57 | 47.79 | 38.05 |
| MLP | 66.26 | 57.23 | 51.74 | 47.03 |
| CNN | 73.84 | 64.97 | 66.65 | 65.22 |
| LSTM | 67.90 | 57.68 | 57.25 | 55.64 |
| CNN-LSTM | 74.50 | 64.78 | 66.45 | 66.41 |
| TABL | 77.63 | 66.93 | 78.44 | — |
| DeepLOB | 83.40 | 72.82 | 80.35 | 76.76 |
| DeepLOB-Attention | 82.37 | 73.73 | 79.38 | 81.49 |
| **DTNN** | **86.92** | **77.14** | **87.94** | **92.53** |

DTNN outperforms all baselines across all horizons. At k=100, F1 is 11.04% higher than DeepLOB-Attention.

### Ablation Study (F1, %)

Removing the differential layer causes significant F1 drops, with larger impact at longer horizons:

| Horizon | Full DTNN | Without Diff Layer | Drop |
|---|---|---|---|
| k=10 | 86.92 | 78.88 | −8.04 |
| k=20 | 77.14 | 67.00 | −10.41 |
| k=50 | 87.94 | 71.05 | −16.89 |
| k=100 | 92.53 | 70.39 | −22.14 |

Removing the feature extractor or Transformer module also causes significant drops.

### Real Stock Data

Group 1 (same distribution): F1 85.79% / 77.62% / 66.09% / 62.08% (k=10/20/50/100).
Group 2 (cross-stock generalization): F1 82.53% / 73.98% / 66.60% / 59.10%.

## Implementation Details

Training uses Adam optimizer, learning rate 1e-4, CrossEntropyLoss, batch_size=64, 150 epochs. The model returns raw logits.

The code's `prepare_x` function additionally generates 20 pairwise interaction features (40→60 dimensions), an enhancement not described in the paper that may bring minor performance gains.
