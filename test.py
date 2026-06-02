# -*- coding: utf-8 -*-
"""
test.py
===============================================================================
用途：
    使用训练好的模型在测试集上进行评估，并保存部分测试样本的可视化结果。

本文件职责：
    1. 读取配置与测试参数
    2. 加载数据集和测试集划分
    3. 从 checkpoint 恢复模型权重和 standardizer_stats
    4. 调用 evaluator 计算测试指标
    5. 保存测试结果
    6. 额外保存 10 个测试样本的预测对比图：
       - S11 真值 vs 预测
       - Pattern 真值 vs 预测

说明：
    test.py 默认读取：
    - HDF5 数据集
    - MATLAB 切分文件
    - 已训练好的 checkpoint

    其中测试数据由 split 文件中的 test_idx 决定。
===============================================================================
"""

from __future__ import annotations

import argparse
import os
import pprint

import torch

from data.build_preprocessed_h5 import ensure_preprocessed_h5
from configs.default_config import get_default_config, update_config_from_args
from data.datamodule import build_dataloaders
from data.preprocess import TargetStandardizer
from models.forward_proxy_net import build_forward_proxy_model
from models.losses import build_loss_function
from engine.evaluator import evaluate_model
from utils.seed import set_seed
from utils.io import ensure_dir, save_json
from utils.logger import create_logger
from utils.visualize import save_prediction_example


def parse_args():
    parser = argparse.ArgumentParser(description="Test CNN forward proxy model")

    parser.add_argument("--h5_path", type=str, default=None, help="HDF5 dataset path")
    parser.add_argument("--split_mat", type=str, default=None, help="MATLAB split indices path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

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
    )

    standardizer.y_mean = standardizer.y_mean.to(device)
    standardizer.y_std = standardizer.y_std.to(device)
    standardizer.p_mean = standardizer.p_mean.to(device)
    standardizer.p_std = standardizer.p_std.to(device)

    return standardizer


def save_test_visualizations(
    model,
    loader,
    device,
    figure_dir,
    standardizer,
    start_sample=0,
    max_samples=10,
):
    """
    从 test loader 中取样本，跳过前 start_sample 个后，
    连续保存 max_samples 个样本的预测对比图。
    """
    ensure_dir(figure_dir)

    model.eval()
    global_idx = 0
    saved_count = 0

    with torch.no_grad():
        for x, y, p, meta in loader:
            x = x.to(device)

            y_pred, p_pred = model(x)

            # 先在同一设备上反标准化，再转到 CPU
            y_pred_raw = standardizer.denormalize_y(y_pred).cpu()
            p_pred_raw = standardizer.denormalize_p(p_pred).cpu()

            batch_size = x.size(0)
            for j in range(batch_size):
                # 跳过前 start_sample 个样本
                if global_idx < start_sample:
                    global_idx += 1
                    continue

                save_prediction_example(
                    output_dir=figure_dir,
                    sample_idx=global_idx,   # 文件名直接用真实样本编号
                    y_true=meta["y_raw"][j].cpu().numpy(),
                    y_pred=y_pred_raw[j].numpy(),
                    p_true=meta["p_raw"][j].cpu().numpy(),
                    p_pred=p_pred_raw[j].numpy(),
                )

                global_idx += 1
                saved_count += 1

                if saved_count >= max_samples:
                    return


def main():
    args = parse_args()

    cfg = get_default_config()
    cfg = update_config_from_args(cfg, args)

    ensure_dir(cfg["paths"]["output_dir"])
    ensure_dir(cfg["paths"]["log_dir"])
    ensure_dir(os.path.join(cfg["paths"]["output_dir"], "figures"))

    logger = create_logger(
        name="test",
        log_dir=cfg["paths"]["log_dir"],
        log_filename="test.log",
    )

    set_seed(cfg["train"]["seed"])

    device_str = cfg["train"]["device"]
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    logger.info("Using device: %s", device)
    logger.info("Configuration:")
    logger.info("\n%s", pprint.pformat(cfg, sort_dicts=False))

    checkpoint_path = cfg["test"]["checkpoint"]
    if checkpoint_path is None:
        checkpoint_path = os.path.join(cfg["paths"]["checkpoint_dir"], "best_model.pt")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info("Loading checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError("checkpoint 中缺少 model_state_dict")

    # 测试时统一使用 checkpoint 中保存的 standardizer
    standardizer = build_standardizer_from_checkpoint(checkpoint, device)
    
    raw_h5_path = cfg["paths"]["h5_path"]

    pre_cfg = cfg.get("preprocess", {})

    if pre_cfg.get("enable", True):
        logger.info("Preparing preprocessed dataset for testing...")
        preprocessed_h5_path = ensure_preprocessed_h5(
            src_h5_path=raw_h5_path,
            feed_sigma=pre_cfg.get("feed_sigma", 1.5),
            compression=pre_cfg.get("compression", "lzf"),
            force=pre_cfg.get("force_rebuild", False),
        )
        cfg["paths"]["h5_path"] = preprocessed_h5_path
        logger.info("Using preprocessed h5: %s", preprocessed_h5_path)
    else:
        logger.info("Preprocess disabled, using raw h5 directly.")
    logger.info("Loading dataloaders...")
    dataloaders, dataset_info, _ = build_dataloaders(cfg, standardizer=standardizer)

    logger.info("Dataset info: %s", dataset_info)

    logger.info("Building model...")
    model = build_forward_proxy_model(
        cfg=cfg,
        dataset_info=dataset_info,
    )
    model = model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    logger.info("Building loss function...")
    criterion = build_loss_function(cfg)

    logger.info("Running evaluation on test set...")
    metrics = evaluate_model(
        model=model,
        loader=dataloaders["test"],
        criterion=criterion,
        standardizer=standardizer,
        device=device,
        cfg=cfg,
        logger=logger,
        split_name="test",
    )

    logger.info("Test metrics:")
    for k, v in metrics.items():
        logger.info("%s: %.6f", k, v)

    save_json(
        os.path.join(cfg["paths"]["output_dir"], "test_metrics.json"),
        metrics,
    )
    logger.info("Saved test metrics to output directory.")

    figure_dir = os.path.join(cfg["paths"]["output_dir"], "figures")
    logger.info("Saving 10 test prediction figures to: %s", figure_dir)

    save_test_visualizations(
        model=model,
        loader=dataloaders["test"],
        device=device,
        figure_dir=figure_dir,
        standardizer=standardizer,
        start_sample=0,
        max_samples=5,
    )

    logger.info("Saved 10 test sample figures.")


if __name__ == "__main__":
    main()