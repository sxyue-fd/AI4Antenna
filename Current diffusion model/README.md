# Current diffusion model

条件扩散模型项目：以标准化后的 S11 `(41,)` 和方向图 `(5,4,120)` 为条件，生成标准化电流分布 `(20,32,32)`。

## 结构

- `configs/default_config.py`：数据路径、正向代理 checkpoint 路径、DDPM 和训练超参。
- `models/condition_encoder.py`：S11 MLP、方向图 Conv1d+MLP、512 维条件融合。
- `models/unet.py`：FiLM 条件注入 U-Net。
- `models/gaussian_diffusion.py`：DDPM 前向加噪、反向采样和 loss。
- `engine/trainer.py`：step-based 训练、checkpoint、CSV 日志。
- `engine/evaluator.py`：按 `sample_every` 采样电流，并调用正向代理计算 S11/方向图 MAE。

## 训练

```bash
cd "Current diffusion model"
python train.py
```

常用覆盖项：

```bash
python train.py --train_num_steps 100000 --batch_size 64 --sample_every 1000 --num_samples 16
python train.py --surrogate_checkpoint "../Current forward surrogate/outputs/train/train_20260626_161012/checkpoints/best_model.pt"
```

训练输出位于 `outputs/train/train_YYYYMMDD_HHMMSS/`：

- `logs/train.log`
- `logs/train_log.csv`
- `figures/training_metrics.png`
- `checkpoints/best_model.pt`
- `checkpoints/last_model.pt`
- `samples/sample_step_*.pt`

## 测试

```bash
python test.py --checkpoint outputs/train/<run_id>/checkpoints/best_model.pt --num_samples 64
```

测试会在 test split 上采样，并通过配置中的正向代理模型计算生成电流对应的 S11/方向图与真实标签之间的 MAE/MSE。

