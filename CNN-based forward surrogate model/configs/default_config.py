# -*- coding: utf-8 -*-
import os


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def project_path(*parts):
    return os.path.join(PROJECT_DIR, *parts)


def resolve_project_relative_path(path):
    if path is None or os.path.isabs(path) or os.path.exists(path):
        return path

    candidate = os.path.join(PROJECT_DIR, path)
    if os.path.exists(candidate):
        return candidate

    return path


def get_default_config():
    cfg = {
        "paths": {
            #"h5_path": "./datasets/antenna_dataset_forward_proxy_20260415_232242.h5",
            "h5_path": project_path("datasets", "antenna_dataset_20260606_161522.h5"),
            "output_dir": project_path("outputs"),
            "checkpoint_dir": project_path("checkpoints"),
            "log_dir": project_path("outputs", "logs"),
        },

        "train": {
            "seed": 106,
            "device": "auto",   # auto / cuda / cpu
            "epochs": 200,
            "resume": None,
            "early_stop_patience": 20,
            "amp": True,
            "cudnn_benchmark": True,
        },

        "split": {
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "seed": 106,
        },

        "data": {
            "batch_size": 256,
            "num_workers": 12,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 12,
        },

        "preprocess": {
            "enable": True,
            "feed_sigma": 2,
            "compression": "lzf",
            "force_rebuild": False,
            "force_recompute_stats": True,
        },

        "optim": {
            "lr": 7e-4,
            "weight_decay": 1e-4,
        },

        "scheduler": {
            "factor": 0.5,
            "patience": 4,
        },

        "loss": {
            "y_weight": 1.0,
            "p_weight": 1.0,
        },

        "test": {
            "checkpoint": None,
        },

        #增加推理选项
        "inference": {
            "h5_path": project_path("datasets", "antenna_dataset_20260606_161522.h5"),
            "checkpoint": project_path("checkpoints", "best_model.pt"),
            "index": 11106,
            "device": "auto",
            "save_dir": project_path("outputs", "infer"),
            "save_numpy": True,
            "warmup_iters": 20,
            "benchmark_iters": 100,
        },
    }
    return cfg


def update_config_from_args(cfg, args):

    if hasattr(args, "h5_path") and args.h5_path is not None:
        cfg["paths"]["h5_path"] = resolve_project_relative_path(args.h5_path)

    if hasattr(args, "output_dir") and args.output_dir:
        cfg["paths"]["output_dir"] = args.output_dir
        cfg["paths"]["log_dir"] = os.path.join(cfg["paths"]["output_dir"], "logs")

    if hasattr(args, "checkpoint_dir") and args.checkpoint_dir is not None:
        cfg["paths"]["checkpoint_dir"] = args.checkpoint_dir

    if hasattr(args, "epochs") and args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs

    if hasattr(args, "batch_size") and args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size

    if hasattr(args, "num_workers") and args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers

    if hasattr(args, "train_ratio") and args.train_ratio is not None:
        cfg["split"]["train_ratio"] = args.train_ratio

    if hasattr(args, "val_ratio") and args.val_ratio is not None:
        cfg["split"]["val_ratio"] = args.val_ratio

    if hasattr(args, "test_ratio") and args.test_ratio is not None:
        cfg["split"]["test_ratio"] = args.test_ratio

    if hasattr(args, "lr") and args.lr is not None:
        cfg["optim"]["lr"] = args.lr

    if hasattr(args, "weight_decay") and args.weight_decay is not None:
        cfg["optim"]["weight_decay"] = args.weight_decay

    if hasattr(args, "seed") and args.seed is not None:
        cfg["train"]["seed"] = args.seed
        cfg["split"]["seed"] = args.seed

    if hasattr(args, "device") and args.device is not None:
        cfg["train"]["device"] = args.device

    if hasattr(args, "early_stop_patience") and args.early_stop_patience is not None:
        cfg["train"]["early_stop_patience"] = args.early_stop_patience

    if hasattr(args, "checkpoint") and args.checkpoint is not None:
        cfg["test"]["checkpoint"] = resolve_project_relative_path(args.checkpoint)

    if hasattr(args, "resume") and args.resume is not None:
        cfg["train"]["resume"] = resolve_project_relative_path(args.resume)

    return cfg
