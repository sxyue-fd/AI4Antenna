# -*- coding: utf-8 -*-
"""
train.py
===============================================================================
用途：
    训练基于 CNN 的前向代理模型。

任务：
    输入：像素化天线结构 + 馈电位置
    输出：S11 频响 + 方向图

本文件职责：
    1. 解析命令行参数
    2. 读取默认配置
    3. 构建数据集与 DataLoader
    4. 构建模型、损失函数、优化器、学习率调度器
    5. 调用训练器完成训练
    6. 保存最佳模型与最终模型

注意：
    本文件只作为项目入口，不承载底层实现细节。
    具体的数据读取、模型定义、训练循环、日志记录等功能，
    由项目内其他模块分别负责。
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
from models.forward_proxy_net import build_forward_proxy_model
from models.losses import build_loss_function
from engine.trainer import train_model
from utils.seed import set_seed
from utils.io import ensure_dir, save_json
from utils.logger import create_logger
from utils.checkpoint import save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN forward proxy model")

    parser.add_argument("--h5_path", type=str, default=None, help="HDF5 dataset path")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Checkpoint directory")

    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument("--train_ratio", type=float, default=None, help="Train split ratio")
    parser.add_argument("--val_ratio", type=float, default=None, help="Validation split ratio")
    parser.add_argument("--test_ratio", type=float, default=None, help="Test split ratio")

    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay")

    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu")
    parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint path")
    parser.add_argument("--plot_loss", action="store_true", help="Plot loss curves after training")
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=None,
        help="Early stopping patience",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = get_default_config()
    cfg = update_config_from_args(cfg, args)

    ensure_dir(cfg["paths"]["output_dir"])
    ensure_dir(cfg["paths"]["checkpoint_dir"])
    ensure_dir(cfg["paths"]["log_dir"])

    logger = create_logger(
        name="train",
        log_dir=cfg["paths"]["log_dir"],
        log_filename="train.log",
    )

    set_seed(cfg["train"]["seed"])

    device_str = cfg["train"]["device"]
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    if device.type == "cuda" and cfg["train"].get("cudnn_benchmark", True):
        torch.backends.cudnn.benchmark = True

    use_amp = cfg["train"].get("amp", True) and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    logger.info("Using device: %s", device)
    logger.info("AMP enabled: %s", use_amp)
    logger.info("cuDNN benchmark: %s", torch.backends.cudnn.benchmark)
    logger.info("Configuration:")
    logger.info("\n%s", pprint.pformat(cfg, sort_dicts=False))

    save_json(
        os.path.join(cfg["paths"]["output_dir"], "train_config.json"),
        cfg,
    )

    raw_h5_path = cfg["paths"]["h5_path"]
    if not os.path.isfile(raw_h5_path):
        raise FileNotFoundError(f"HDF5 dataset not found: {raw_h5_path}")

    pre_cfg = cfg.get("preprocess", {})
    dataset_h5_path = raw_h5_path

    if pre_cfg.get("enable", True):
        logger.info("Preparing preprocessed dataset...")
        dataset_h5_path = ensure_preprocessed_h5(
            src_h5_path=raw_h5_path,
            feed_sigma=pre_cfg.get("feed_sigma", 1.5),
            compression=pre_cfg.get("compression", "lzf"),
            force=pre_cfg.get("force_rebuild", False),
        )
        logger.info("Using preprocessed h5: %s", dataset_h5_path)
    else:
        logger.info("Preprocess disabled, using raw h5 directly: %s", dataset_h5_path)

    cfg["paths"]["h5_path"] = dataset_h5_path

    logger.info("Building dataloaders...")
    dataloaders, dataset_info, standardizer = build_dataloaders(cfg)

    logger.info("Dataset info: %s", dataset_info)

    logger.info("Building model...")
    model = build_forward_proxy_model(
        cfg=cfg,
        dataset_info=dataset_info,
    )
    model = model.to(device)

    logger.info("Building loss function...")
    criterion = build_loss_function(cfg)

    logger.info("Building optimizer and scheduler...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg["scheduler"]["factor"],
        patience=cfg["scheduler"]["patience"],
    )

    start_epoch = 0
    best_val_loss = float("inf")

    if cfg["train"]["resume"] is not None:
        resume_path = cfg["train"]["resume"]
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        logger.info("Resuming from checkpoint: %s", resume_path)
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        scaler_state_dict = checkpoint.get("scaler_state_dict", None)
        if scaler_state_dict is not None:
            scaler.load_state_dict(scaler_state_dict)

        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)

    logger.info("Start training...")
    results = train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloaders=dataloaders,
        standardizer=standardizer,
        device=device,
        cfg=cfg,
        scaler=scaler,
        logger=logger,
        start_epoch=start_epoch,
        best_val_loss=best_val_loss,
        dataset_info=dataset_info,
    )

    logger.info("Training finished.")
    logger.info("Best validation loss: %.6f", results["best_val_loss"])

    final_ckpt_path = os.path.join(cfg["paths"]["checkpoint_dir"], "last_model.pt")
    save_checkpoint(
        path=final_ckpt_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=results["last_epoch"],
        best_val_loss=results["best_val_loss"],
        extra={
            "config": cfg,
            "dataset_info": dataset_info,
            "scaler_state_dict": scaler.state_dict(),
            "standardizer_stats": {
                "y_mean": standardizer.y_mean.detach().cpu(),
                "y_std": standardizer.y_std.detach().cpu(),
                "p_mean": standardizer.p_mean.detach().cpu(),
                "p_std": standardizer.p_std.detach().cpu(),
            },
        },
    )
    logger.info("Saved final checkpoint to: %s", final_ckpt_path)
    if args.plot_loss:
        logger.info("Plotting loss curves...")
        from utils.plot_loss import main as plot_loss_main
        plot_loss_main()
        logger.info("Loss curves saved to outputs/figures/")


if __name__ == "__main__":
    main()
