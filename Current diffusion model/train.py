# -*- coding: utf-8 -*-
"""
Train the conditional current diffusion model.
"""

from __future__ import annotations

import argparse
import os
import pprint
import sys

import torch

_PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
_WORKSPACE_DIR = os.path.abspath(os.path.join(_PROJECT_DIR, ".."))
for _path in (_PROJECT_DIR, _WORKSPACE_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from configs.default_config import get_default_config, update_config_from_args
from data.datamodule import build_diffusion_dataloaders
from datasets.preprocess import ensure_standardized_dataset
from engine.forward_surrogate import load_forward_surrogate
from engine.trainer import train_model
from models.current_diffusion import build_current_diffusion
from utils.checkpoint import save_checkpoint
from utils.io import ensure_dir, save_json
from utils.logger import create_logger
from utils.model_compat import (
    validate_diffusion_checkpoint_dataset_shapes,
    validate_forward_surrogate_shapes,
)
from utils.run_dirs import configure_train_run_dirs
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train conditional DDPM for current distribution")

    parser.add_argument("--h5_path", type=str, default=None, help="HDF5 dataset path")
    parser.add_argument("--output_dir", type=str, default=None, help="Output root directory")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Checkpoint directory")
    parser.add_argument("--surrogate_checkpoint", type=str, default=None, help="Forward surrogate checkpoint path")

    parser.add_argument("--train_num_steps", type=int, default=None, help="Number of optimizer steps")
    parser.add_argument("--batch_size", type=int, default=None, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=None, help="Evaluation batch size")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument("--timesteps", type=int, default=None, help="Diffusion timesteps")
    parser.add_argument("--num_samples", type=int, default=None, help="Samples for surrogate MAE evaluation")
    parser.add_argument("--cfg_scale", type=float, default=None, help="Legacy shared classifier-free guidance scale")
    parser.add_argument("--s11_cfg_scale", type=float, default=None, help="S11 classifier-free guidance scale for sampling")
    parser.add_argument("--pattern_cfg_scale", type=float, default=None, help="Pattern classifier-free guidance scale for sampling")

    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay")

    parser.add_argument("--sample_every", type=int, default=None, help="Sample/evaluate frequency in steps")
    parser.add_argument("--save_every", type=int, default=None, help="Checkpoint frequency in steps")
    parser.add_argument("--log_every", type=int, default=None, help="CSV log frequency in steps")

    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="auto / cuda / cpu")
    parser.add_argument("--resume", type=str, default=None, help="Resume diffusion checkpoint path")
    parser.add_argument("--no_plot", action="store_true", help="Disable training metric plot output")

    return parser.parse_args()


def _device_from_cfg(cfg):
    device_str = cfg["train"]["device"]
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


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
        name="current_diffusion_train",
        log_dir=cfg["paths"]["log_dir"],
        log_filename="train.log",
    )

    set_seed(cfg["train"]["seed"])
    device = _device_from_cfg(cfg)

    if device.type == "cuda" and cfg["train"].get("cudnn_benchmark", True):
        torch.backends.cudnn.benchmark = True

    use_amp = cfg["train"].get("amp", True) and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    logger.info("Using device: %s", device)
    logger.info("AMP enabled: %s", use_amp)
    logger.info("Configuration:")
    logger.info("\n%s", pprint.pformat(cfg, sort_dicts=False))

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
    dataloaders, dataset_info, standardizer = build_diffusion_dataloaders(cfg)
    logger.info("Dataset info: %s", dataset_info)

    logger.info("Building diffusion model...")
    diffusion_model = build_current_diffusion(cfg=cfg, dataset_info=dataset_info)
    diffusion_model = diffusion_model.to(device)

    logger.info("Loading forward surrogate...")
    surrogate_model, surrogate_checkpoint = load_forward_surrogate(
        project_dir=cfg["surrogate"]["project_dir"],
        checkpoint_path=cfg["surrogate"]["checkpoint"],
        device=device,
    )
    validate_forward_surrogate_shapes(surrogate_checkpoint, dataset_info)
    logger.info("Forward surrogate checkpoint: %s", cfg["surrogate"]["checkpoint"])

    save_json(os.path.join(cfg["paths"]["output_dir"], "train_config.json"), cfg)

    logger.info("Building optimizer and scheduler...")
    adam_betas = tuple(cfg["optim"].get("adam_betas", (0.9, 0.99)))
    optimizer = torch.optim.AdamW(
        diffusion_model.parameters(),
        lr=cfg["optim"]["lr"],
        betas=adam_betas,
        weight_decay=cfg["optim"]["weight_decay"],
    )

    scheduler_cfg = cfg.get("scheduler", {})
    t_max = scheduler_cfg.get("t_max") or cfg["train"]["train_num_steps"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=t_max,
        eta_min=scheduler_cfg.get("eta_min", 0.0),
    )

    start_step = 0
    best_sample_mae = float("inf")

    if cfg["train"]["resume"] is not None:
        resume_path = cfg["train"]["resume"]
        logger.info("Resuming from checkpoint: %s", resume_path)
        checkpoint = torch.load(resume_path, map_location=device)
        validate_diffusion_checkpoint_dataset_shapes(checkpoint, dataset_info, context="Resume checkpoint")
        diffusion_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        scaler_state_dict = checkpoint.get("scaler_state_dict", None)
        if scaler_state_dict:
            scaler.load_state_dict(scaler_state_dict)

        start_step = checkpoint.get("step", 0)
        best_sample_mae = checkpoint.get("best_sample_mae", best_sample_mae)

    logger.info("Start diffusion training...")
    results = train_model(
        diffusion_model=diffusion_model,
        surrogate_model=surrogate_model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloaders=dataloaders,
        standardizer=standardizer,
        device=device,
        cfg=cfg,
        scaler=scaler,
        logger=logger,
        start_step=start_step,
        best_sample_mae=best_sample_mae,
        dataset_info=dataset_info,
        surrogate_checkpoint=surrogate_checkpoint,
    )

    logger.info("Training finished.")
    logger.info("Last loss: %s", results["last_loss"])
    logger.info("Best sample MAE: %s", results["best_sample_mae"])

    final_ckpt_path = os.path.join(cfg["paths"]["checkpoint_dir"], "last_model.pt")
    save_checkpoint(
        path=final_ckpt_path,
        model=diffusion_model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=results["last_step"],
        best_sample_mae=results["best_sample_mae"],
        extra={
            "config": cfg,
            "dataset_info": dataset_info,
            "standardizer_stats": standardizer.state_dict(),
            "surrogate_dataset_info": surrogate_checkpoint.get("dataset_info"),
            "surrogate_checkpoint": cfg["surrogate"]["checkpoint"],
            "scaler_state_dict": scaler.state_dict(),
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
