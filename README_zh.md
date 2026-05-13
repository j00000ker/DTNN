# DTNN — Differential Transformer Neural Network

[English](README.md) | **中文**

Code for the paper **"Predicting High-Frequency Stock Movement with Differential Transformer Neural Network"**（[Electronics 2023, 12(13), 2943](https://doi.org/10.3390/electronics12132943)）。

DTNN 是一种用于高频股票市场预测的混合深度学习架构。它将三条并行编码器——时序卷积网络（TCN）、时序注意力层（TABL）和原始恒等直通——的输出拼接后沿时间轴做差分，再送入带有 class token 的 Transformer 编码器进行分类。

## 架构

```
输入: (batch, time_steps, features)
  |
  |-- TCN 分支 ────────────┐
  |-- TABL 分支 ───────────┤── 拼接 ── 线性嵌入 ── 时序差分
  |-- 恒等直通 ─────────────┘                              |
                                                          v
                                           CLS token + 位置嵌入
                                                          |
                                                      Transformer
                                                          |
                                                       MLP 分类头
                                                          |
                                                   3 类 softmax 输出
```

## 安装

```bash
# 克隆仓库
git clone https://github.com/j00000ker/DTNN.git
cd DTNN

# 可编辑模式安装
pip install -e .

# 或仅安装依赖
pip install -r requirements.txt
```

### 依赖

- Python >= 3.8
- PyTorch >= 1.9.0
- NumPy、Pandas、SciPy、scikit-learn
- einops
- Matplotlib、tqdm

## 快速开始

```python
import torch
from dtnn import DTNN

# 创建模型：100 个时间步，60 维特征，3 个输出类别
model = DTNN(
    time_slices=100,
    num_classes=3,
    dim=60,
    depth=3,
    heads=32,
)

# 前向传播
x = torch.randn(4, 100, 60)  # (batch=4, time=100, features=60)
output = model(x)             # (4, 3) — 概率分布
```

### 基线模型

本包也包含了论文中用于对比的基线模型：

```python
from dtnn import LSTM, CNN, CNN_LSTM, MLP, SVM, C_TABL

model = LSTM(time_slices=100, dim=60, num_classes=3)
```

## 训练

`train.py` 脚本用于复现论文中的实验：

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

参数说明：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data-path` | `''` | 存放 `.txt` 数据文件的目录 |
| `--batch-size` | 64 | 批大小 |
| `--lr` | 1e-4 | 学习率 |
| `--epochs` | 150 | 训练轮数 |
| `--depth` | 3 | Transformer 深度 |
| `--heads` | 32 | 注意力头数 |
| `--k` | 1 | 标签列索引 |
| `--T` | 100 | 时间窗口长度 |
| `--seed` | 42 | 随机种子 |
| `--use-sampler` | 关闭 | 启用加权采样以处理类别不平衡 |

## 数据格式

训练脚本期望的文本文件中，每列为一个时间序列特征，每行为一个时间步。数据加载流程：

1. 取前 40 行作为基础特征
2. 计算相邻列的逐对乘积（额外 20 个特征）
3. 取最后 5 行作为标签基
4. 创建长度为 `T` 的滑动窗口用于训练

## TCN 基准测试

`TCN/` 目录包含了使用时序卷积网络的标准序列建模基准测试，包括：

- Adding Problem
- Copy Memory
- Sequential / Permuted MNIST
- 字符级语言模型（Penn Treebank）
- 词级语言模型（Penn Treebank / WikiText-103）
- 复调音乐（JSB Chorales、Nottingham、MuseData、Piano-midi）
- LAMBADA 文本理解

## 引用

如果你在研究中使用了本代码，请引用：

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

## 许可证

MIT License — 详见 [LICENSE](LICENSE)。
