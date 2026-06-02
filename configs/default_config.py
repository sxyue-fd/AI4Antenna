# -*- coding: utf-8 -*-
import os


def get_default_config():
    cfg = {
        "paths": {
            #"h5_path": "./datasets/antenna_dataset_forward_proxy_20260415_232242.h5",
            "h5_path": "./datasets/antenna_dataset_forward_proxy_20260428_214708.h5",
            "split_mat": "./datasets/dataset_split_indices.mat",
            "output_dir": "./outputs",
            "checkpoint_dir": "./checkpoints",
            "log_dir": "./outputs/logs",
        },

        "train": {
            "seed": 42,
            "device": "auto",   # auto / cuda / cpu
            "epochs": 200,
            "resume": None,
            "early_stop_patience": 20,
            "amp": True,
            "cudnn_benchmark": True,
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
            "h5_path": "./datasets/antenna_dataset_forward_proxy_20260415_232242.h5",
            "checkpoint": "./checkpoints/best_model.pt",
            "index": 11106,
            "device": "auto",
            "save_dir": "outputs/infer",
            "save_numpy": True,
            "warmup_iters": 20,
            "benchmark_iters": 100,
        },
    }
    return cfg


def update_config_from_args(cfg, args):

    if hasattr(args, "h5_path") and args.h5_path is not None:
        cfg["paths"]["h5_path"] = args.h5_path

    if hasattr(args, "split_mat") and args.split_mat is not None:
        cfg["paths"]["split_mat"] = args.split_mat

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

    if hasattr(args, "lr") and args.lr is not None:
        cfg["optim"]["lr"] = args.lr

    if hasattr(args, "weight_decay") and args.weight_decay is not None:
        cfg["optim"]["weight_decay"] = args.weight_decay

    if hasattr(args, "seed") and args.seed is not None:
        cfg["train"]["seed"] = args.seed

    if hasattr(args, "device") and args.device is not None:
        cfg["train"]["device"] = args.device

    if hasattr(args, "early_stop_patience") and args.early_stop_patience is not None:
        cfg["train"]["early_stop_patience"] = args.early_stop_patience

    if hasattr(args, "checkpoint") and args.checkpoint is not None:
        cfg["test"]["checkpoint"] = args.checkpoint

    if hasattr(args, "resume") and args.resume is not None:
        cfg["train"]["resume"] = args.resume

    return cfg