# -*- coding: utf-8 -*-
import os


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_DIR = os.path.abspath(os.path.join(PROJECT_DIR, ".."))
DATASETS_DIR = os.path.join(WORKSPACE_DIR, "datasets")
FORWARD_SURROGATE_DIR = os.path.join(WORKSPACE_DIR, "Current forward surrogate")


def project_path(*parts):
    return os.path.join(PROJECT_DIR, *parts)


def workspace_path(*parts):
    return os.path.join(WORKSPACE_DIR, *parts)


def resolve_project_relative_path(path):
    if path is None or os.path.isabs(path) or os.path.exists(path):
        return path

    candidate = os.path.join(PROJECT_DIR, path)
    if os.path.exists(candidate):
        return candidate

    candidate = os.path.join(WORKSPACE_DIR, path)
    if os.path.exists(candidate):
        return candidate

    return path


def get_default_config():
    default_h5_path = os.path.join(
        DATASETS_DIR,
        "antenna_dataset_20260614_203654.preprocessed.standardized.h5",
    )
    default_surrogate_checkpoint = os.path.join(
        FORWARD_SURROGATE_DIR,
        "outputs",
        "train",
        "train_20260626_161012",
        "checkpoints",
        "best_model.pt",
    )

    cfg = {
        "paths": {
            "h5_path": default_h5_path,
            "output_dir": project_path("outputs"),
            "checkpoint_dir": project_path("checkpoints"),
            "log_dir": project_path("outputs", "logs"),
        },

        "surrogate": {
            "project_dir": FORWARD_SURROGATE_DIR,
            "checkpoint": default_surrogate_checkpoint,
        },

        "train": {
            "seed": 106,
            "device": "auto",
            "train_num_steps": 100000,
            "resume": None,
            "amp": True,
            "cudnn_benchmark": True,
            "grad_clip": None,
            "log_every": 100,
            "sample_every": 1000,
            "save_every": 10000,
        },

        "split": {
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "seed": 106,
        },

        "data": {
            "input_key": "current",
            "batch_size": 256,
            "eval_batch_size": 256,
            "num_workers": 4,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 2,
        },

        "model": {
            "base_dim": 64,
            "dim_mults": (1, 2, 4, 8),
            "condition_dim": 512,
            "s11_embed_dim": 256,
            "pattern_embed_dim": 256,
            "pattern_seq_len": 15,
            "condition_dropout": 0.1,
            "attention_resolutions": (16, 8, 4),
        },

        "diffusion": {
            "timesteps": 1000,
            "objective": "pred_noise",
            "beta_schedule": "cosine", # options: linear, cosine, sigmoid
            # Generated current lives in the standardized current space:
            # signed-log -> clip(T=4.0) -> z-score. The z-scored clip bound is 12.
            "x_start_clip": 12.0,
        },

        "optim": {
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "adam_betas": (0.9, 0.99),
        },

        "scheduler": {
            "type": "cosine_annealing",
            "t_max": None,
            "eta_min": 1e-6,
        },

        "sample": {
            "num_samples": 256,
            "save_tensors": False,
            "cfg_scale": None,
            "s11_cfg_scale": 5.0,
            "pattern_cfg_scale": 5.0,
        },

        "test": {
            "checkpoint": None,
            "num_samples": 256,
            "seed": 42,
        },
    }
    return cfg


def update_config_from_args(cfg, args):
    if hasattr(args, "h5_path") and args.h5_path is not None:
        cfg["paths"]["h5_path"] = resolve_project_relative_path(args.h5_path)

    if hasattr(args, "output_dir") and args.output_dir:
        cfg["paths"]["output_dir"] = resolve_project_relative_path(args.output_dir)
        cfg["paths"]["log_dir"] = os.path.join(cfg["paths"]["output_dir"], "logs")

    if hasattr(args, "checkpoint_dir") and args.checkpoint_dir is not None:
        cfg["paths"]["checkpoint_dir"] = resolve_project_relative_path(args.checkpoint_dir)

    if hasattr(args, "surrogate_checkpoint") and args.surrogate_checkpoint is not None:
        cfg["surrogate"]["checkpoint"] = resolve_project_relative_path(args.surrogate_checkpoint)

    if hasattr(args, "train_num_steps") and args.train_num_steps is not None:
        cfg["train"]["train_num_steps"] = args.train_num_steps

    if hasattr(args, "batch_size") and args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size

    if hasattr(args, "eval_batch_size") and args.eval_batch_size is not None:
        cfg["data"]["eval_batch_size"] = args.eval_batch_size

    if hasattr(args, "num_workers") and args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers

    if hasattr(args, "lr") and args.lr is not None:
        cfg["optim"]["lr"] = args.lr

    if hasattr(args, "weight_decay") and args.weight_decay is not None:
        cfg["optim"]["weight_decay"] = args.weight_decay

    if hasattr(args, "timesteps") and args.timesteps is not None:
        cfg["diffusion"]["timesteps"] = args.timesteps

    if hasattr(args, "sample_every") and args.sample_every is not None:
        cfg["train"]["sample_every"] = args.sample_every

    if hasattr(args, "save_every") and args.save_every is not None:
        cfg["train"]["save_every"] = args.save_every

    if hasattr(args, "log_every") and args.log_every is not None:
        cfg["train"]["log_every"] = args.log_every

    if hasattr(args, "num_samples") and args.num_samples is not None:
        cfg["sample"]["num_samples"] = args.num_samples
        cfg["test"]["num_samples"] = args.num_samples

    if hasattr(args, "cfg_scale") and args.cfg_scale is not None:
        cfg["sample"]["cfg_scale"] = args.cfg_scale
        cfg["sample"]["s11_cfg_scale"] = None
        cfg["sample"]["pattern_cfg_scale"] = None

    if hasattr(args, "s11_cfg_scale") and args.s11_cfg_scale is not None:
        cfg["sample"]["s11_cfg_scale"] = args.s11_cfg_scale
        cfg["sample"]["cfg_scale"] = None

    if hasattr(args, "pattern_cfg_scale") and args.pattern_cfg_scale is not None:
        cfg["sample"]["pattern_cfg_scale"] = args.pattern_cfg_scale
        cfg["sample"]["cfg_scale"] = None

    if hasattr(args, "seed") and args.seed is not None:
        cfg["train"]["seed"] = args.seed
        cfg["split"]["seed"] = args.seed

    if hasattr(args, "device") and args.device is not None:
        cfg["train"]["device"] = args.device

    if hasattr(args, "resume") and args.resume is not None:
        cfg["train"]["resume"] = resolve_project_relative_path(args.resume)

    if hasattr(args, "checkpoint") and args.checkpoint is not None:
        cfg["test"]["checkpoint"] = resolve_project_relative_path(args.checkpoint)

    return cfg
