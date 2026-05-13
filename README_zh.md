# DTNN — Differential Transformer Neural Network

[English](README.md) | **中文**

论文 **"Predicting High-Frequency Stock Movement with Differential Transformer Neural Network"** 的官方代码（[Electronics 2023, 12(13), 2943](https://doi.org/10.3390/electronics12132943)）。

完整的论文内容总结（架构细节、实验结果、消融实验）见 [PAPER_zh.md](PAPER_zh.md)。

## 架构

```
LOB 输入: (batch, T=100, features=40→60)
  |
  |-- TCN 分支 ────────────┐
  |-- TABL 分支 ───────────┤── 拼接 ── 线性嵌入 ── 差分层
  |-- 恒等直通 ─────────────┘                            |
                                                        v
                                             CLS token + 位置嵌入
                                                        |
                                                    Transformer
                                                        |
                                                     MLP 分类头
                                                        |
                                                 3 类 logits 输出
```

三条并行编码器（TCN、TABL、恒等映射）的输出拼接后投影到统一维度，沿时间轴做一阶差分。随后加入 CLS token 和位置嵌入，经 Transformer 编码器处理，提取 CLS 输出通过 MLP 分类头做 3 类预测。**差分层**是核心贡献——它让 Transformer 关注相邻时间片之间的*变化*而非绝对位置。

## 安装

```bash
git clone https://github.com/j00000ker/DTNN.git
cd DTNN
pip install -e .
```

需要 Python >= 3.8、PyTorch >= 1.9.0、NumPy、Pandas、SciPy、scikit-learn、einops、Matplotlib、tqdm。

## 快速开始

```python
import torch
from dtnn import DTNN

model = DTNN(time_slices=100, num_classes=3, dim=60, depth=3, heads=32)

x = torch.randn(4, 100, 60)   # (batch=4, time=100, features=60)
output = model(x)              # (4, 3) — 原始 logits（配合 CrossEntropyLoss 使用）
```

论文中的基线模型也包含在内：

```python
from dtnn import LSTM, CNN, CNN_LSTM, MLP, SVM, C_TABL
```

## 训练

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

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--data-path` | `''` | 存放 `.txt` 数据文件的目录 |
| `--batch-size` | 64 | 批大小 |
| `--lr` | 1e-4 | 学习率 |
| `--epochs` | 150 | 训练轮数 |
| `--depth` | 3 | Transformer 层数 |
| `--heads` | 32 | 注意力头数 |
| `--k` | 1 | 标签列索引 |
| `--T` | 100 | 时间窗口长度 |
| `--seed` | 42 | 随机种子 |
| `--use-sampler` | 关闭 | 加权采样处理类别不平衡 |

## 数据格式

输入为每时间片 40 维 LOB 特征（10 档 × {卖价, 卖量, 买价, 买量}）。标签为 3 类（上涨 / 不变 / 下跌），由预测周期 k 内的平均收益率是否超过阈值定义。

数据处理流程：原始 `.txt` → `prepare_x`（40 特征 + 20 个两两交互特征 = 60 维）→ 长度 T 的滑动窗口 → `StockDataset`（1 索引标签转 0 索引）。

## TCN 基准测试

`TCN/` 目录包含标准 TCN 序列建模基准（Adding Problem、Copy Memory、Sequential/Permuted MNIST、字符级/词级语言模型、复调音乐、LAMBADA）。

## 引用

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

MIT — 详见 [LICENSE](LICENSE)。
