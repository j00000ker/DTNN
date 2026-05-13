# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 常用命令

```bash
pip install -e .              # 以可编辑模式安装 dtnn 包
python train.py --help         # 查看训练参数
python train.py --data-path ./data/ --epochs 150
```

没有测试框架，也没有 CI。TCN 基准测试各自独立运行，例如 `python TCN/adding_problem/add_test.py`。

## 架构概览

### 核心模型 DTNN（`dtnn/models/dtnn.py`）

三条并行编码器分支 → 拼接 → 线性嵌入 → 时序差分 → Transformer → MLP 分类头：

1. **TCN 分支**（`dtnn/modules/tcn.py`）：因果膨胀卷积，保持时序长度不变
2. **TABL 分支**（`dtnn/modules/tabl.py`）：三层双线性注意力，同时变换时间和特征维度
3. **恒等旁路**：原始输入直接传入

三条分支的输出在第 106 行按 `(x, x1, x2)` 顺序拼接。线性嵌入 `self.emb` 用 `nn.init.eye_` 初始化（`[dim, dim*3]` 权重矩阵），只有前 `dim` 列为 1，其余为 0——这样训练初期恒等旁路可以直接通过。**这个初始化依赖于拼接顺序，不能随意调整。**

时序差分（第 113 行）使用 `torch.diff` 沿时间轴求一阶差分，是论文的核心贡献。

之后加上可学习的 CLS token 和位置嵌入，送入标准 Transformer encoder，最后用 `pool='cls'` 取 CLS token 或 `pool='mean'` 取均值池化，经 MLP head 输出原始 logits。

**所有模型（DTNN 和 baselines）都返回原始 logits**，训练时配合 `CrossEntropyLoss` 使用。

### 包结构

- `dtnn/models/` — 最终模型（DTNN + baselines）
- `dtnn/modules/` — 可复用的网络组件（Transformer、TCN、TABL）
- `dtnn/data_utils.py` — 数据加载与预处理
- `dtnn/train_utils.py` — `train_model()` 和 `evaluate()`
- `TCN/` — 旧版 TCN 基准测试套件，与 `dtnn/` 包独立。`TCN/tcn.py` 是从 `dtnn.modules.tcn` 的轻量重导出

### 数据管线

原始 `.txt` 文件（列=特征，行=时间步）→ `prepare_x`（取前 40 行转置，计算相邻列的逐对乘积，40+20=60 特征）→ `get_label`（取最后 5 行作为标签基）→ `data_classification`（滑动窗口长度 T=100）→ `StockDataset`（标签列 k 的标签减 1，将 1 索引转为 0 索引）

## 注意事项

- TABL 的 `torch.bmm` 在 `forward` 中使用广播机制——不要加回 `.repeat()`，那会造成不必要的内存分配
- 原始数据标签是 1 索引的；`StockDataset.__init__` 中的 `y[:, self.k] - 1` 将其转为 0 索引
- `prepare_x` 中的硬编码 40/20 对该论文的数据集具有特异性；不同的数据需要对这部分做适配
- TCN 基准测试中的 `TCN` 类与 `dtnn.modules.tcn.TCN` 是不同的（基准测试各有一个轻量封装，用于各自的任务）
