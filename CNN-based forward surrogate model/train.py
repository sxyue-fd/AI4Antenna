# -*- coding: utf-8 -*-
"""
train.py
===============================================================================
Train the CNN forward surrogate model.
===============================================================================
"""

from __future__ import annotations

import argparse
import os
import pprint
import sys

import torch

_WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, _WORKSPACE_DIR)

from configs.default_config import get_default_config, update_config_from_args
from datasets.datamodule import build_dataloaders
from datasets.preprocess import ensure_standardized_dataset
from models.forward_surrogate_net import build_forward_surrogate
from models.losses import build_loss_function
from engine.trainer import train_model
from utils.seed import set_seed
from utils.io import ensure_dir, save_json
from utils.logger import create_logger
from utils.checkpoint import save_checkpoint
from utils.run_dirs import configure_train_run_dirs
from utils.model_compat import validate_checkpoint_dataset_shapes


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN forward surrogate model")

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
    parser.add_argument("--no_plot", action="store_true", help="Disable training metric plot output")
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

    resume_checkpoint = None
    if cfg["train"]["resume"] is not None:
        resume_path = cfg["train"]["resume"]
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        resume_checkpoint = torch.load(resume_path, map_location="cpu")

    cfg = configure_train_run_dirs(
        cfg=cfg,
        resume_checkpoint=resume_checkpoint,
        resume_path=cfg["train"]["resume"],
    )

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

    dataset_h5_path = cfg["paths"]["h5_path"]
    split_cfg = cfg.get("split", {})
    dataset_h5_path = ensure_standardized_dataset(
        h5_path=dataset_h5_path,
        input_keys=["X", "current"],
        feed_sigma=2.0,
        compression="lzf",
        train_ratio=split_cfg.get("train_ratio", 0.8),
        val_ratio=split_cfg.get("val_ratio", 0.1),
        test_ratio=split_cfg.get("test_ratio", 0.1),
        seed=split_cfg.get("seed", cfg.get("train", {}).get("seed", 106)),
    )
    cfg["paths"]["h5_path"] = dataset_h5_path
    logger.info("Using standardized h5: %s", dataset_h5_path)

    logger.info("Building dataloaders...")
    dataloaders, dataset_info, standardizer = build_dataloaders(cfg)

    logger.info("Dataset info: %s", dataset_info)

    logger.info("Building model...")
    model = build_forward_surrogate(
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

    scheduler_cfg = cfg.get("scheduler", {})
    t_max = scheduler_cfg.get("t_max")
    if t_max is None:
        t_max = cfg["train"]["epochs"]

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=t_max,
        eta_min=scheduler_cfg.get("eta_min", 0.0),
    )

    start_epoch = 0
    best_val_loss = float("inf")

    if cfg["train"]["resume"] is not None:
        resume_path = cfg["train"]["resume"]

        logger.info("Resuming from checkpoint: %s", resume_path)
        checkpoint = torch.load(resume_path, map_location=device)
        validate_checkpoint_dataset_shapes(checkpoint, dataset_info, context="Resume checkpoint")
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
            "standardizer_stats": standardizer.state_dict(),
            "run_id": cfg["paths"].get("run_id"),
            "run_dir": cfg["paths"].get("run_dir"),
            "output_root": cfg["paths"].get("output_root"),
        },
    )
    logger.info("Saved final checkpoint to: %s", final_ckpt_path)

    if not args.no_plot:
        try:
            from utils.plot_loss import plot_training_metrics

            csv_path = os.path.join(cfg["paths"]["log_dir"], "train_log.csv")
            figure_path = os.path.join(cfg["paths"]["output_dir"], "figures", "training_metrics.png")
            plot_training_metrics(csv_path=csv_path, save_path=figure_path)
            logger.info("Saved training metric plot to: %s", figure_path)
        except Exception as exc:
            logger.warning("Failed to plot training metrics: %s", exc)


if __name__ == "__main__":
    main()
