# -*- coding: utf-8 -*-
"""
infer.py
===============================================================================
用途：
    对单个样本执行前向代理模型推理。

功能：
    1. 读取 HDF5 数据集
    2. 从 checkpoint 中恢复模型权重与 standardizer_stats
    3. 自动检查并生成预处理后的 HDF5（如需要）
    4. 对指定样本执行前向预测
    5. 将预测结果从标准化空间反变换回原始空间
    6. 打印并可选保存推理结果

说明：
    本脚本不重新计算标准化统计量，
    而是直接使用训练时保存在 checkpoint 中的 standardizer_stats。

    若已将 H5AntennaDataset 改为读取“预处理后 HDF5”的版本，
    则本脚本会在推理前自动调用 ensure_preprocessed_h5(...)，
    保证读取的是 *.preprocessed.h5。
===============================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, _WORKSPACE_DIR)

from configs.default_config import get_default_config, update_config_from_args
from datasets.build_preprocessed_h5 import ensure_preprocessed_h5
from datasets.h5_dataset import H5AntennaDataset
from datasets.preprocess import TargetStandardizer
from models.forward_surrogate_net import build_forward_surrogate
from utils.visualize import save_prediction_example
from utils.run_dirs import configure_eval_output_dir, find_latest_train_checkpoint
from utils.model_compat import validate_checkpoint_dataset_shapes

def parse_args():
    parser = argparse.ArgumentParser(description="Inference for CNN forward surrogate model")

    parser.add_argument("--h5_path", type=str, help="HDF5 dataset path")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path")
    parser.add_argument("--index", type=int, default=None, help="Sample index in HDF5")
    parser.add_argument("--device", type=str, default=None, help="auto / cuda / cpu")
    parser.add_argument("--save_dir", type=str, default=None, help="Optional output directory")
    parser.add_argument("--save_numpy", action="store_true", help="Save prediction arrays as .npz")

    return parser.parse_args()


def build_standardizer_from_checkpoint(checkpoint, device):
    if "standardizer_stats" not in checkpoint:
        raise KeyError("checkpoint 中缺少 standardizer_stats")

    stats = checkpoint["standardizer_stats"]
    required_keys = ["y_mean", "y_std", "p_mean", "p_std"]
    for k in required_keys:
        if k not in stats:
            raise KeyError(f"standardizer_stats 中缺少字段: {k}")

    standardizer = TargetStandardizer(
        y_mean=stats["y_mean"],
        y_std=stats["y_std"],
        p_mean=stats["p_mean"],
        p_std=stats["p_std"],
        x_mean=stats.get("x_mean", None),
        x_std=stats.get("x_std", None),
    )

    # 逐个移动到目标设备，兼容你当前的 TargetStandardizer 实现
    standardizer.y_mean = standardizer.y_mean.to(device)
    standardizer.y_std = standardizer.y_std.to(device)
    standardizer.p_mean = standardizer.p_mean.to(device)
    standardizer.p_std = standardizer.p_std.to(device)
    if standardizer.x_mean is not None:
        standardizer.x_mean = standardizer.x_mean.to(device)
    if standardizer.x_std is not None:
        standardizer.x_std = standardizer.x_std.to(device)

    return standardizer


@torch.no_grad()
def main():
    args = parse_args()

    cfg = get_default_config()
    cfg = update_config_from_args(cfg, args)
    infer_cfg = cfg["inference"]
    h5_path = args.h5_path or infer_cfg["h5_path"]
    checkpoint_path = args.checkpoint or infer_cfg["checkpoint"]
    index = args.index if args.index is not None else infer_cfg["index"]
    device_name = args.device or infer_cfg["device"]
    save_numpy = args.save_numpy or infer_cfg["save_numpy"]
    warmup_iters = infer_cfg["warmup_iters"]
    benchmark_iters = infer_cfg["benchmark_iters"]
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
    if args.checkpoint is None and not os.path.isfile(checkpoint_path):
        latest_checkpoint = find_latest_train_checkpoint(
            cfg["paths"]["output_dir"],
            checkpoint_name="best_model.pt",
        )
        if latest_checkpoint is not None:
            checkpoint_path = latest_checkpoint

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError("checkpoint 中缺少 model_state_dict")

    save_dir = configure_eval_output_dir(
        cfg=cfg,
        kind="infer",
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        explicit_output_dir=args.save_dir,
    )

    standardizer = build_standardizer_from_checkpoint(checkpoint, device)

    pre_cfg = cfg.get("preprocess", {})

    if pre_cfg.get("enable", True):
        print("Preparing preprocessed dataset for inference...")
        infer_h5_path = ensure_preprocessed_h5(
            src_h5_path=h5_path,
            feed_sigma=pre_cfg.get("feed_sigma",2),
            compression=pre_cfg.get("compression", "lzf"),
            force=pre_cfg.get("force_rebuild", False),
            input_key=cfg.get("data", {}).get("input_key", "X"),
            input_keys=pre_cfg.get("input_keys"),
        )
        print(f"Using preprocessed h5: {infer_h5_path}")
    else:
        infer_h5_path = h5_path
        print("Preprocess disabled, using raw h5 directly.")

    # 这里读取的是“用于推理的数据集”，不做 standardizer 标准化，
    # 便于直接拿到原始标签值与反标准化后的预测结果进行对比。
    infer_dataset = H5AntennaDataset(
        infer_h5_path,
        standardizer=None,
        input_key=cfg.get("data", {}).get("input_key", "X"),
    )

    if len(infer_dataset) == 0:
        raise RuntimeError("数据集为空，无法执行推理。")

    if index < 0 or index >= len(infer_dataset):
        raise IndexError(f"index={index} 超出范围，合法区间为 [0, {len(infer_dataset)-1}]")

    x0, y0, p0, _ = infer_dataset[0]
    dataset_info = {
        "x_shape": tuple(x0.shape),
        "y_shape": tuple(y0.shape),
        "pattern_shape": tuple(p0.shape),
    }
    standardizer.validate_shapes(
        dataset_info["y_shape"],
        dataset_info["pattern_shape"],
        x_shape=dataset_info["x_shape"] if cfg.get("data", {}).get("normalize_input", False) else None,
    )
    validate_checkpoint_dataset_shapes(checkpoint, dataset_info, context="Inference checkpoint")

    model = build_forward_surrogate(cfg=cfg, dataset_info=dataset_info).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x, y_true_raw, p_true_raw, _ = infer_dataset[index]
    x = x.unsqueeze(0).to(device)
    x = standardizer.normalize_x(x)

    # 模型输出仍处于标准化空间
    #y_pred_norm, p_pred_norm = model(x)
    # =========================
    # inference timing
    # =========================

    #if device.type == "cuda":
       # torch.cuda.synchronize()

    #start_time = time.time()

    #y_pred_norm, p_pred_norm = model(x)

    #if device.type == "cuda":
     #   torch.cuda.synchronize()

    #infer_time = time.time() - start_time
    # ==========================================
    # benchmark inference latency
    # ==========================================

    model.eval()

    # -------------------------
    # warmup
    # -------------------------
    for _ in range(warmup_iters):
        _ = model(x)

    # CUDA synchronize
    if device.type == "cuda":
        torch.cuda.synchronize()

    # -------------------------
    # benchmark
    # -------------------------
    start_time = time.time()

    for _ in range(benchmark_iters):
        y_pred_norm, p_pred_norm = model(x)

    # CUDA synchronize
    if device.type == "cuda":
        torch.cuda.synchronize()

    avg_infer_time = (time.time() - start_time) / benchmark_iters

    # 反标准化回原始空间
    y_pred_raw = standardizer.denormalize_y(y_pred_norm).squeeze(0).cpu()
    p_pred_raw = standardizer.denormalize_p(p_pred_norm).squeeze(0).cpu()

    y_true_raw = y_true_raw.cpu()
    p_true_raw = p_true_raw.cpu()

    y_mae = torch.mean(torch.abs(y_pred_raw - y_true_raw)).item()
    p_mae = torch.mean(torch.abs(p_pred_raw - p_true_raw)).item()

    print("=== Inference ===")
    print(f"device: {device}")
    print(f"sample index: {index}")
    print(f"input_h5_path: {infer_h5_path}")
    print(f"x shape: {tuple(x.shape)}")
    print(f"y_pred_raw shape: {tuple(y_pred_raw.shape)}")
    print(f"p_pred_raw shape: {tuple(p_pred_raw.shape)}")
    print(f"y_raw MAE: {y_mae:.6f}")
    print(f"pattern_raw MAE: {p_mae:.6f}")
    #print(f"inference time: {infer_time*1000:.3f} ms")
    print(f"average inference time: {avg_infer_time*1000:.3f} ms")
    #print("\nFirst 10 values of y_true_raw:")
    #print(y_true_raw[:10].numpy())

    #print("\nFirst 10 values of y_pred_raw:")
    #print(y_pred_raw[:10].numpy())

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_prediction_example(
            output_dir=save_dir,
            sample_idx=index,
            y_true=y_true_raw.numpy(),
            y_pred=y_pred_raw.numpy(),
            p_true=p_true_raw.numpy(),
            p_pred=p_pred_raw.numpy(),
        )

        summary = {
            "index": index,
            "device": str(device),
            "input_h5_path": infer_h5_path,
            "x_shape": list(x.shape),
            "y_true_raw_shape": list(y_true_raw.shape),
            "y_pred_raw_shape": list(y_pred_raw.shape),
            "p_true_raw_shape": list(p_true_raw.shape),
            "p_pred_raw_shape": list(p_pred_raw.shape),
            "y_raw_mae": y_mae,
            "pattern_raw_mae": p_mae,
            #"inference_time_ms": infer_time * 1000,
            "average_inference_time_ms": avg_infer_time * 1000,
        }

        with open(os.path.join(save_dir, "infer_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        if save_numpy:
            np.savez(
                os.path.join(save_dir, f"infer_sample_{index}.npz"),
                x=x.squeeze(0).cpu().numpy(),
                y_true_raw=y_true_raw.numpy(),
                y_pred_raw=y_pred_raw.numpy(),
                p_true_raw=p_true_raw.numpy(),
                p_pred_raw=p_pred_raw.numpy(),
            )

        print(f"\nSaved inference results to: {save_dir}")


if __name__ == "__main__":
    main()
