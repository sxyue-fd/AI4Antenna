"""AntGen 的统一运行配置。

通常只需修改本文件中的 CONFIG，然后在 AI4Ant_demo 根目录运行：

    python AntGen/run.py

值为 None 的数据集和 checkpoint 会从扩散 checkpoint 配置或现有 outputs
目录中自动发现。Path 可以使用绝对路径；下面的默认输出路径不依赖启动目录。
"""

from pathlib import Path


ANTGEN_DIR = Path(__file__).resolve().parent


CONFIG = {
    # 运行阶段：all=完整流水线；infer/simulate/analyze 用于断点恢复。
    "run": {
        "stage": "all",
        # all/infer 时保持 None 会自动创建带时间戳的目录；
        # simulate/analyze 时改为已有运行目录。
        "run_dir": None,
        "output_root": ANTGEN_DIR / "outputs",
    },

    # 测试条件与一对多采样。这是日常最常修改的一组参数。
    "sampling": {
        "num_conditions": 16,       # m：从测试集取多少个条件
        "num_candidates": 16,       # n：每个条件生成多少个候选
        "selection": "sequential", # sequential 或 random
        "test_start": 0,           # sequential 时在测试 split 内的起始偏移
        "seed": 106,
    },

    # 模型与数据。None 表示自动使用最新 best_model.pt / checkpoint 中的数据集。
    "paths": {
        "h5_path": None,
        "diffusion_checkpoint": None,
        "structure_checkpoint": None,
        "cnn_surrogate_checkpoint": None,
    },

    # 扩散推理和结构后处理。
    "inference": {
        "device": "auto",          # auto、cpu、cuda 或 cuda:<id>
        "generation_batch_size": 256,
        "diffusion_progress": False,
        # None 表示沿用扩散 checkpoint 中保存的 CFG scale。
        "s11_cfg_scale": None,
        "pattern_cfg_scale": None,
        "metal_threshold": 0.5,
        # feed-component：仅保留包含馈点的四连通金属；none：不处理。
        "topology_postprocess": "feed-component",
    },

    # CNN 正向代理对全部 m*n 个候选批量预测并排序。
    "surrogate": {
        "batch_size": 256,
    },

    # MATLAB 全波仿真。workers=0 使用默认并行池，1 为串行，>1 指定池大小。
    "matlab": {
        "command": "matlab",
        "workers": 0,
    },

    # 候选排序使用两个 MAE 的加权平均；两项默认等权。
    "metrics": {
        "s11_weight": 1.0,
        "pattern_weight": 1.0,
    },

    # 可视化最多绘制前 max_conditions 个条件，数值结果仍统计全部 m 个条件。
    "visualization": {
        "enabled": True,
        "max_conditions": 20,
        "candidate_page_size": 16,
        "pattern_frequency_index": 2,  # 默认展示中间方向图频点（当前为 10 GHz）
        "dpi": 150,
    },
}
