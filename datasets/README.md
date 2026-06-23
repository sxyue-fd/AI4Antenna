# 数据集预处理

`datasets/` 是本仓库统一的数据预处理入口。运行 `preprocess.py` 会先把 MATLAB HDF5 转为训练布局的 `.preprocessed.h5`，再仅以训练集样本计算统计量并生成 `.preprocessed.standardized.h5` 与 `.standardizer.pt`。

```powershell
python datasets\preprocess.py
python datasets\preprocess.py datasets\antenna_dataset_20260614_203654.h5 --force
```

默认数据集为 `antenna_dataset_20260614_203654.h5`。`--force` 会在预处理算法更新后重建缓存；旧版标准化缓存会因 schema 不匹配自动重建。

## 数据布局

原始 MATLAB 布局会转换为下列布局：

| 数据 | `.preprocessed.h5` / `.standardized.h5` 布局 |
| --- | --- |
| 天线结构 `X` | `(N, 2, 16, 16)` |
| 电流 `current` | `(N, F=5, C=4, 32, 32)` |
| S11 `Y` | `(N, 41)` |
| 方向图 `pattern` | `(N, F=5, C=4, 120)` |

`X` 的两个通道分别为结构和馈电通道。源数据的馈电 one-hot 会按 `--feed_sigma` 生成为高斯馈电图，两个通道的取值均在 `[0, 1]`。

为兼容现有 `Conv2d` 主干，`H5AntennaDataset(input_key="current")` 在读取时把电流临时视为 `(N, 20, 32, 32)`；HDF5 文件本身始终保留显式的频点和分量轴，统计也按 `(F, C)` 计算。

## 标准化规则

所有统计量均只用由 `train_ratio`、`val_ratio`、`test_ratio` 和 `seed` 决定的训练集计算，避免验证/测试信息泄漏。

### 天线结构

不做数据集统计。对结构和馈电两个通道都执行：

```text
X_standardized = 2 * X - 1
```

输出范围为 `[-1, 1]`。

### 电流

电流使用按频点和分量独立的 signed-log、截断和 z-score：

```text
alpha[f,c] = p95(abs(J_train[:, f, c, :, :]))
Z = sign(J) * log(1 + abs(J) / alpha[f,c])
Z_clip = clip(Z, -4.0, 4.0)
mu[f,c], sigma[f,c] = mean/std(Z_clip_train[:, f, c, :, :])
J_standardized = (Z_clip - mu[f,c]) / sigma[f,c]
```

`alpha`、`mu` 和 `sigma` 保存于 `.standardizer.pt` 的 `input_stats["current"]`，其形状为 `(5, 4)`。

### S11

S11 的 41 个频点共享一组标量统计量：

```text
mu, sigma = mean/std(Y_train[:, :])  # 在 N*41 个值上统计
Y_standardized = (Y - mu) / sigma
```

因此 `y_mean` 和 `y_std` 的形状为 `(1,)`，反标准化时自动广播到所有频点。

### 方向图

方向图对每个频点与分量独立统计，但合并样本轴与 theta 轴：

```text
mu[f,c], sigma[f,c] = mean/std(pattern_train[:, f, c, :])
pattern_standardized = (pattern - mu[f,c]) / sigma[f,c]
```

保存的 `p_mean`、`p_std` 形状为 `(5, 4, 1)`，可以广播到 120 个 theta 采样点。

## 产物与使用

```text
dataset.h5
dataset.preprocessed.h5
dataset.preprocessed.standardized.h5
dataset.standardizer.pt
```

- `.preprocessed.h5`：仅布局转换，保留原始数值。
- `.preprocessed.standardized.h5`：训练、验证、测试使用的标准化数据；另存有 `/Y_raw`、`/pattern_raw` 供评估。
- `.standardizer.pt`：目标反标准化以及输入变换参数。

CNN 项目使用 `input_key="X"`，Current 项目使用 `input_key="current"`，两者均应指向同一个 `.preprocessed.standardized.h5`。
