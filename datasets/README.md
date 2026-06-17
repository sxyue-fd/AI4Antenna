# datasets 数据预处理说明

`datasets` 目录是本仓库唯一的数据预处理入口。CNN 项目和 Current 项目的 `train.py`、`test.py`、`infer.py` 不再做 reshape、统计量计算或在线标准化，只读取这里生成好的 `.preprocessed.standardized.h5`。

## 文件作用

### `preprocess.py`

统一预处理入口。它串联两步：

1. 调用 `build_preprocessed_h5.py`，把原始 MATLAB HDF5 reshape 成 PyTorch 友好的 `.preprocessed.h5`。
2. 计算统一标准化统计量，生成 `.preprocessed.standardized.h5` 和 `.standardizer.pt`。

当前默认配置写在文件顶部的 `DEFAULT_PREPROCESS_CONFIG` 中，默认处理：

```text
datasets/antenna_dataset_20260614_203654.h5
```

直接运行：

```powershell
python datasets\preprocess.py
```

也可以临时覆盖默认文件：

```powershell
python datasets\preprocess.py datasets\other_dataset.h5
```

常用参数：

```powershell
python datasets\preprocess.py --force
python datasets\preprocess.py --input_keys X,current --feed_sigma 2 --seed 106
```

参数含义：

- `--input_keys`：要预处理并标准化的输入键，默认 `X,current`。
- `--feed_sigma`：把 `X` 中的馈电点 one-hot 转成高斯热图时使用的 sigma。
- `--compression`：输出 HDF5 的压缩方式，默认 `lzf`。
- `--force`：强制重建 `.preprocessed.h5`、`.preprocessed.standardized.h5` 和 `.standardizer.pt`。
- `--train_ratio`、`--val_ratio`、`--test_ratio`、`--seed`：划分训练/验证/测试集。标准化统计量只由训练集计算。

### `build_preprocessed_h5.py`

只负责从原始 `.h5` 生成 `.preprocessed.h5`，不做标准化。

主要函数：

- `ensure_preprocessed_h5(...)`：检查缓存是否可复用；如果缺失、shape 不匹配、源文件更新或指定 `force=True`，则重建。
- `build_preprocessed_h5(...)`：实际逐样本读取原始 HDF5，做 reshape，并写入目标 HDF5。
- `get_preprocessed_h5_path(...)`：根据源文件名得到 `.preprocessed.h5` 路径。

一般不需要手动运行这个脚本；推荐通过 `preprocess.py` 间接调用。

### `h5_dataset.py`

HDF5 读取和单样本格式转换层。

主要职责：

- 自动识别 HDF5 layout：`matlab_raw`、`preprocessed`、`standardized`。
- 对原始 MATLAB HDF5 的单样本做 reshape。
- 对标准化 HDF5，读取已经标准化后的 `X/current/Y/pattern`。
- 在评估、测试、推理时，如果文件中有 `Y_raw` 和 `pattern_raw`，通过 `meta["y_raw"]`、`meta["p_raw"]` 返回原始尺度标签。

常用类和函数：

- `H5AntennaDataset`：PyTorch `Dataset`。
- `read_h5_dataset_info(...)`：读取样本数、输入 shape、输出 shape、pattern shape 等信息。
- `normalize_x(...)`、`normalize_current(...)`、`normalize_y(...)`、`normalize_pattern(...)`：名字沿用历史代码，实际作用是从 MATLAB layout 转成训练 layout，不是统计标准化。

### `datamodule.py`

训练项目侧的数据加载器构建层。

主要职责：

- 只接受 `.preprocessed.standardized.h5`。
- 构建 train/val/test split。
- 创建 PyTorch `DataLoader`。
- 读取统一的 `.standardizer.pt`，供评估和可视化时反标准化模型输出。

项目中的 `train.py` 会调用：

```python
dataloaders, dataset_info, standardizer = build_dataloaders(cfg)
```

注意：这里不会再计算标准化统计量，也不会在 DataLoader 中在线标准化。

### `split_loader.py`

只负责生成 train/val/test 的样本索引。

规则：

- 使用 `numpy.random.default_rng(seed)` 打乱样本索引。
- 按 `train_ratio`、`val_ratio`、`test_ratio` 划分。
- ratio 会自动归一化，因此三者总和不必严格等于 1。
- 对极小数据集做保护，避免出现完全不可用的划分。

### `__init__.py`

使 `datasets` 成为可导入的 Python 包。

## 预处理产物

假设原始文件为：

```text
antenna_dataset_20260614_203654.h5
```

运行：

```powershell
python datasets\preprocess.py
```

会生成：

```text
antenna_dataset_20260614_203654.preprocessed.h5
antenna_dataset_20260614_203654.preprocessed.standardized.h5
antenna_dataset_20260614_203654.standardizer.pt
```

其中：

- `.preprocessed.h5`：只做 reshape 和 layout 转换，不做统计标准化。
- `.preprocessed.standardized.h5`：训练、测试、推理直接读取的文件，主要数组都已经标准化。
- `.standardizer.pt`：统一标准化器，CNN 和 Current 项目共用。

## 原始 HDF5 到 `.preprocessed.h5` 的 reshape

### `X`

原始 `X` 支持以下单样本形状：

```text
(H, W)
(H, W, 1)
(H, W, 2)
```

处理步骤：

1. 从原始编码中提取结构通道 `structure`。
2. 从原始编码中提取馈电点 one-hot 图 `feed_map`。
3. 检查馈电点必须且只能有一个。
4. 用 `feed_sigma` 生成馈电点高斯热图 `feed_gaussian`。
5. 拼成两个通道：

```text
(H, W, 2) = [structure, feed_gaussian]
```

6. 转置为 PyTorch 输入布局：

```text
(H, W, 2) -> (2, W, H)
```

整个文件写入后，`X` 的形状为：

```text
(N, 2, W, H)
```

### `current`

原始单样本形状：

```text
(H, W, C, F)
```

处理步骤：

```text
(H, W, C, F)
-> transpose(F, C, W, H)
-> reshape(F*C, W, H)
```

整个文件写入后，`current` 的形状为：

```text
(N, F*C, W, H)
```

### `Y`

原始数据按 MATLAB 风格存储为：

```text
(F, N)
```

单样本读取后转成：

```text
(F,)
```

整个文件写入后，`Y` 的形状为：

```text
(N, F)
```

### `pattern`

原始单样本形状：

```text
(T, P, Fp)
```

处理步骤：

```text
(T, P, Fp) -> transpose(Fp, P, T)
```

整个文件写入后，`pattern` 的形状为：

```text
(N, Fp, P, T)
```

## `.preprocessed.h5` 到 `.preprocessed.standardized.h5` 的标准化

标准化在 `preprocess.py` 的 `build_standardized_h5(...)` 中完成。

### 统计量如何计算

1. 先用 `split_loader.build_split_indices(...)` 按固定 seed 划分 train/val/test。
2. 只遍历训练集索引计算统计量，避免验证集和测试集信息泄漏。
3. 分别计算每个数据分量的逐元素均值和标准差。

均值和标准差的 shape 与对应单样本 shape 相同：

```text
X mean/std:       (2, W, H)
current mean/std: (F*C, W, H)
Y mean/std:       (F,)
pattern mean/std: (Fp, P, T)
```

代码中会用 `min=1e-8` 保护方差开根号，避免浮点误差导致负方差。这不是对某类数据的额外处理，只是数值稳定保护。

### `X` 标准化

如果 `input_keys` 包含 `X`，则对 `.preprocessed.h5` 中的 `X` 做普通 z-score：

```text
X_standardized = (X - X_mean) / X_std
```

结果写入 `.preprocessed.standardized.h5` 的 `/X`。

### `current` 标准化

如果 `input_keys` 包含 `current`，则对 `.preprocessed.h5` 中的 `current` 做普通 z-score，和 `X`、`Y`、`pattern` 一致：

```text
current_standardized = (current - current_mean) / current_std
```

结果写入 `.preprocessed.standardized.h5` 的 `/current`。

### `Y` 标准化

```text
Y_standardized = (Y - Y_mean) / Y_std
```

结果写入 `.preprocessed.standardized.h5` 的 `/Y`。

原始值同时写入：

```text
/Y_raw
```

### `pattern` 标准化

```text
pattern_standardized = (pattern - pattern_mean) / pattern_std
```

结果写入 `.preprocessed.standardized.h5` 的 `/pattern`。

原始值同时写入：

```text
/pattern_raw
```

## `standardizer.pt` 保存什么

`.standardizer.pt` 中保存统一标准化器，字段包括：

```text
y_mean, y_std
p_mean, p_std
input_stats["X"]["mean"], input_stats["X"]["std"]
input_stats["current"]["mean"], input_stats["current"]["std"]
```

其中：

- `y_mean/y_std` 用于反标准化 S 参数输出。
- `p_mean/p_std` 用于反标准化 pattern 输出。
- `input_stats` 记录输入标准化参数，主要用于追踪和复现预处理过程。

训练时模型直接使用标准化后的输入和标签；测试、验证、推理时模型输出仍在标准化空间，需要用 `standardizer.denormalize_y(...)` 和 `standardizer.denormalize_p(...)` 回到原始尺度。

## 项目如何使用预处理结果

CNN 项目配置：

```python
cfg["data"]["input_key"] = "X"
```

Current 项目配置：

```python
cfg["data"]["input_key"] = "current"
```

两个项目都应该指向同一个：

```text
*.preprocessed.standardized.h5
```

并共用同一个：

```text
*.standardizer.pt
```

如果训练时报错提示不是 `standardized` layout，说明当前配置仍指向原始 `.h5` 或 `.preprocessed.h5`，需要先运行：

```powershell
python datasets\preprocess.py
```

然后把项目配置中的 `h5_path` 指向生成的 `.preprocessed.standardized.h5`。

当前 CNN 和 Current 的 `train.py` 会在训练开始前调用 `ensure_standardized_dataset(...)`：

- 如果配置指向的 `.preprocessed.standardized.h5` 已存在，直接复用。
- 如果 `.preprocessed.standardized.h5` 不存在，但 `.preprocessed.h5` 已存在，复用 `.preprocessed.h5`，只重新生成标准化文件和 `.standardizer.pt`。
- 如果 `.preprocessed.h5` 和 `.preprocessed.standardized.h5` 都不存在，但原始 `.h5` 存在，先生成 `.preprocessed.h5`，再生成 `.preprocessed.standardized.h5`。
- 训练入口不会强制覆盖已有产物；如果想用新预处理逻辑重建已有文件，仍然需要手动运行 `python datasets\preprocess.py --force`。

## 推荐工作流

1. 修改 `datasets/preprocess.py` 顶部的 `DEFAULT_PREPROCESS_CONFIG`，确认默认数据文件和参数。
2. 运行：

```powershell
python datasets\preprocess.py
```

3. 确认生成：

```text
*.preprocessed.h5
*.preprocessed.standardized.h5
*.standardizer.pt
```

4. 在 CNN 或 Current 项目中运行训练、测试或推理。

如果原始数据文件发生变化，建议使用：

```powershell
python datasets\preprocess.py --force
```
