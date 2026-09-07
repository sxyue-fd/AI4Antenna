# AntGen：端到端天线逆向设计

AntGen 将条件扩散、电流到结构 U-Net、CNN 正向代理和 MATLAB Antenna Toolbox 串成一条两级筛选流水线：

```text
测试集 S11 + 方向图条件（m 个）
        ↓ 每个条件扩散采样 n 次
电流图 (m,n,20,32,32)
        ↓ Current to structure U-Net + 结构后处理
拓扑与馈点 (m,n)
        ↓ CNN-based forward surrogate model
代理 S11/方向图 → 对每个条件的 n 个候选排序
        ↓ 每个条件只选择代理 MAE 最小的 1 个
m 个候选 → MATLAB 全波仿真 m 次
        ↓
最终全波 MAE/MSE 与跨 m 个条件的平均值
```

CNN 会评估全部 `m×n` 个候选，但 MATLAB 只仿真 `m` 个代理最优结果。需要注意：代理最优不保证等于全波意义下的真实最优，因此输出始终把 `surrogate_*` 排序指标和 `fullwave_*` 最终验证指标分开记录。

## 配置与运行

所有日常参数位于 `AntGen/config.py`。通常修改：

```python
"sampling": {
    "num_conditions": 10,  # m
    "num_candidates": 8,   # n
    "selection": "sequential",
    "test_start": 0,
    "seed": 106,
},
"surrogate": {
    "batch_size": 256,
},
"matlab": {
    "command": "matlab",
    "workers": 16,
},
```

然后从 `AI4Ant_demo` 根目录运行：

```powershell
python AntGen\run.py
```

三个 checkpoint 路径保持 `None` 时，会自动选择对应项目中最新的 `best_model.pt`。配置还可控制数据集、GPU、扩散 batch、CFG scale、金属阈值、拓扑后处理、代理 batch、排序权重和可视化数量。命令行参数仍可作为一次性覆盖。

MATLAB 需要 Antenna Toolbox；并行模式还需要 Parallel Computing Toolbox。`workers=0` 使用 MATLAB 默认并行池，`1` 串行，`>1` 指定 worker 数。

## 结构后处理

U-Net 输出一个金属 logits 图和一个馈点 logits 图。AntGen 当前严格按以下顺序处理：

1. 对金属 logits 做 `sigmoid`，再用 `metal_threshold`（默认 `0.5`）二值化。
2. 在全部馈点 logits 上做全局 `argmax`，得到唯一馈点 `(row,col)`。
3. 无论该处是否超过金属阈值，都强制把馈点像素置为金属。
4. 默认 `topology_postprocess="feed-component"`：从馈点出发做四邻域搜索，只保留与馈点四连通的金属分量，删除不可馈电的孤立金属岛；对角接触不算连通。
5. 若设为 `none`，跳过第 4 步，但仍执行馈点强制置金属。

当前没有额外执行开闭运算、孔洞填充、最小线宽、平滑、对称性或最小面积约束。CNN 代理和 MATLAB 使用的是同一份后处理拓扑。预处理阶段已经恢复物理空间轴，导出 MATLAB 时不再交换行列：

```text
MATLAB metal = Python metal
MATLAB feed_rc = Python feed_rc + 1
```

预处理后的 Python 拓扑与 MATLAB 都使用物理 `(row, col)` 轴。这里不能再次转置，
否则会把天线关于 `x=y` 交换并导致 XOZ/YOZ 两组方向图互换。MATLAB 方向图固定为
线性增益，通道顺序为 `XOZ_Gtheta, XOZ_Gphi, YOZ_Gtheta, YOZ_Gphi`。

## CNN 代理排序

代理输入与训练数据完全一致：

- 通道 0：后处理后的 Python 坐标二值金属图。
- 通道 1：以预测馈点为中心的二维 Gaussian feed map，`sigma` 从 HDF5 的 `feed_sigma` 属性读取。
- 两个通道按 checkpoint 的 `input_stats["X"]` 从 `[0,1]` 映射到 `[-1,1]`。
- CNN 输出用其自身 checkpoint 的标准化统计量反标准化为线性 S11 幅值和线性 gain。

每个候选分别计算 S11 和方向图 MAE/MSE。默认排序量为：

```text
surrogate_combined_mae =
    (w_s11 * surrogate_s11_mae + w_pattern * surrogate_pattern_mae)
    / (w_s11 + w_pattern)
```

两个权重默认均为 `1`，让两个响应族等权，而不是让采样点更多的方向图自然占优。排序权重决定了送入 MATLAB 的结构；修改权重后必须重新执行 `infer` 和全波仿真。

## 分阶段恢复

完整流程默认 `run.stage="all"`。也可在 `config.py` 中设置：

- `infer`：生成 `m×n` 候选、CNN 排序、选出 `m` 个结构并准备 MATLAB 输入。
- `simulate`：对已有运行目录中的 `m` 个选中结构做全波仿真，并自动分析。
- `analyze`：重新分析已有 MATLAB 输出。

`simulate/analyze` 时需把 `run.run_dir` 指向已有运行目录。

## 分类输出

每次运行位于 `AntGen/outputs/run_YYYYMMDD_HHMMSS/`：

```text
artifacts/
  manifest.json                  完整配置、checkpoint、索引及文件清单
  generated_candidates.npz      全部 m*n 电流、拓扑、馈点和目标条件
surrogate/
  predictions.npz               全部候选的 CNN S11/方向图
  candidate_ranking.csv         每条件 n 个候选的 surrogate_* 排序
  best_per_condition.csv        送入全波仿真的 m 个代理最优候选
  summary.json                  代理排序汇总
fullwave/
  selected_candidates.npz       被选中的 m 个候选及原始 candidate index
  matlab_input.mat              仅包含 m 个结构
  matlab_output.mat             m 次全波结果
  results.csv                   每个选中候选的 fullwave_* 指标
  summary.json                  全波成功率与平均指标
reports/
  summary.json                  代理排序 + 最终全波验证总报告
figures/
  summary_metrics.png           各条件代理/全波 combined MAE 对比
  condition_XXX/
    candidate_topologies*.png   按代理排名排列的候选拓扑和馈点
    selected_current.png        选中候选的首/中/末频点电流幅值
    selected_surrogate_response.png
    selected_fullwave_comparison.png
```

候选拓扑图只展示生成结果和代理排序，不把测试集原始拓扑当作唯一“真值”；逆向设计的一对多合理性最终由目标响应与全波响应的误差验证。

## 测试

```powershell
python -m unittest discover -s AntGen\tests -v
python -m compileall -q AntGen
```
