# -*- coding: utf-8 -*-
from __future__ import annotations

import os


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PARENT_PROJECT_DIR = os.path.abspath(os.path.join(PROJECT_DIR, ".."))


def project_path(*parts):
    return os.path.join(PROJECT_DIR, *parts)


def get_default_config():
    default_h5_path = os.path.join(
        PARENT_PROJECT_DIR,
        "datasets",
        "antenna_dataset_20260614_203654.preprocessed.standardized.h5",
    )

    return {
        "paths": {
            "h5_path": default_h5_path,
            "output_dir": project_path("outputs"),
        },
        "split": {
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "seed": 106,
        },
        "data": {
            "current_key": "current",
            "x_key": "X",
            "input_key": "current",
            "batch_size": 128,
            "num_workers": 4,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 2,
        },
        "task": {
            "name": "current_to_structure",
            "input_key": "current",
            "x_key": "X",
        },
        "model": {
            "base_channels": 8,
            "dropout": 0.25,
            "target_size": 16,
        },
        "train": {
            "seed": 106,
            "device": "auto",
            "epochs": 100,
            "amp": True,
            "early_stop_patience": 6,
            "early_stop_min_delta": 1e-5,
            "grad_clip_norm": 1.0,
            "progress_log_interval": 1,
            "iter_log_interval": 0,
            "iter_val_interval": 0,
            "iter_val_batches": 20,
        },
        "optim": {
            "lr": 3e-4,
            "weight_decay": 1e-3,
        },
        "scheduler": {
            "eta_min": 1e-6,
        },
        "loss": {
            "metal_bce_weight": 1.0,
            "metal_dice_weight": 1.0,
            "feed_ce_weight": 2.0,
            "feed_label_smoothing": 0.01,
            #"feed_on_air_weight": 0.1,
        },
    }


def update_config_from_args(cfg, args):
    if getattr(args, "h5_path", None):
        cfg["paths"]["h5_path"] = args.h5_path
    if getattr(args, "output_dir", None):
        cfg["paths"]["output_dir"] = args.output_dir
    if getattr(args, "epochs", None) is not None:
        cfg["train"]["epochs"] = args.epochs
    if getattr(args, "batch_size", None) is not None:
        cfg["data"]["batch_size"] = args.batch_size
    if getattr(args, "num_workers", None) is not None:
        cfg["data"]["num_workers"] = args.num_workers
    if getattr(args, "lr", None) is not None:
        cfg["optim"]["lr"] = args.lr
    if getattr(args, "weight_decay", None) is not None:
        cfg["optim"]["weight_decay"] = args.weight_decay
    if getattr(args, "seed", None) is not None:
        cfg["train"]["seed"] = args.seed
        cfg["split"]["seed"] = args.seed
    if getattr(args, "device", None):
        cfg["train"]["device"] = args.device
    if getattr(args, "base_channels", None) is not None:
        cfg["model"]["base_channels"] = args.base_channels
    if getattr(args, "dropout", None) is not None:
        cfg["model"]["dropout"] = args.dropout
    if getattr(args, "early_stop_patience", None) is not None:
        cfg["train"]["early_stop_patience"] = args.early_stop_patience
    if getattr(args, "log_interval", None) is not None:
        cfg["train"]["progress_log_interval"] = args.log_interval
    if getattr(args, "iter_log_interval", None) is not None:
        cfg["train"]["iter_log_interval"] = args.iter_log_interval
    if getattr(args, "iter_val_interval", None) is not None:
        cfg["train"]["iter_val_interval"] = args.iter_val_interval
    if getattr(args, "iter_val_batches", None) is not None:
        cfg["train"]["iter_val_batches"] = args.iter_val_batches
    return cfg
