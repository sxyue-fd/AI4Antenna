# -*- coding: utf-8 -*-
"""
Evaluate a trained current diffusion checkpoint through the forward surrogate.
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
from engine.evaluator import sample_and_evaluate
from engine.forward_surrogate import load_forward_surrogate
from models.current_diffusion import build_current_diffusion
from utils.io import ensure_dir, save_json
from utils.logger import create_logger
from utils.model_compat import (
    validate_diffusion_checkpoint_dataset_shapes,
    validate_forward_surrogate_shapes,
)
from utils.run_dirs import checkpoint_run_info, find_latest_train_checkpoint, make_run_id
from utils.seed import set_seed
from utils.visualize import save_diffusion_visualization_sample


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate conditional current diffusion model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Diffusion checkpoint path")
    parser.add_argument("--h5_path", type=str, default=None, help="HDF5 dataset path")
    parser.add_argument("--output_dir", type=str, default=None, help="Evaluation output directory")
    parser.add_argument("--surrogate_checkpoint", type=str, default=None, help="Forward surrogate checkpoint path")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of test samples")
    parser.add_argument("--eval_batch_size", type=int, default=None, help="Evaluation batch size")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument("--timesteps", type=int, default=None, help="Override diffusion timesteps")
    parser.add_argument("--cfg_scale", type=float, default=None, help="Legacy shared classifier-free guidance scale")
    parser.add_argument("--s11_cfg_scale", type=float, default=None, help="S11 classifier-free guidance scale")
    parser.add_argument("--pattern_cfg_scale", type=float, default=None, help="Pattern classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="auto / cuda / cpu")
    parser.add_argument("--num_visualize", type=int, default=5, help="Number of random test samples to visualize")
    parser.add_argument("--no_visualize", action="store_true", help="Disable random test sample visualization")
    return parser.parse_args()


def _device_from_cfg(cfg):
    device_str = cfg["train"]["device"]
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _resolve_checkpoint(cfg, args):
    checkpoint = cfg.get("test", {}).get("checkpoint")
    if checkpoint:
        return checkpoint
    latest = find_latest_train_checkpoint(cfg["paths"]["output_dir"], checkpoint_name="best_model.pt")
    if latest is not None:
        return latest
    latest = find_latest_train_checkpoint(cfg["paths"]["output_dir"], checkpoint_name="last_model.pt")
    if latest is not None:
        return latest
    raise FileNotFoundError("No diffusion checkpoint found. Pass --checkpoint explicitly.")


def _configure_test_dirs(cfg, checkpoint, checkpoint_path, explicit_output_dir=None):
    if explicit_output_dir:
        output_dir = explicit_output_dir
    else:
        output_root = cfg["paths"].get("output_root") or cfg["paths"].get("output_dir")
        run_id, _ = checkpoint_run_info(checkpoint, checkpoint_path)
        run_id = run_id or make_run_id("test")
        output_dir = os.path.join(output_root, "test", run_id)

    cfg["paths"]["output_dir"] = output_dir
    cfg["paths"]["log_dir"] = os.path.join(output_dir, "logs")
    cfg["paths"]["checkpoint_dir"] = os.path.join(output_dir, "checkpoints")
    return cfg


def _global_index(dataset, local_index):
    indices = getattr(dataset, "indices", None)
    if indices is None:
        return int(local_index)
    return int(indices[int(local_index)])


def _mae_mse(pred, target):
    diff = pred - target
    return float(torch.mean(torch.abs(diff)).item()), float(torch.mean(diff ** 2).item())


def _per_sample_metrics(
    y_pred_raw,
    y_true_raw,
    p_pred_raw,
    p_true_raw,
):
    y_mae, y_mse = _mae_mse(y_pred_raw, y_true_raw)
    p_mae, p_mse = _mae_mse(p_pred_raw, p_true_raw)

    return {
        "y_mae": y_mae,
        "y_mse": y_mse,
        "pattern_mae": p_mae,
        "pattern_mse": p_mse,
    }


def _apply_sample_cfg_scales(model_kwargs, cfg):
    sample_cfg = cfg.get("sample", {})
    s11_cfg_scale = sample_cfg.get("s11_cfg_scale")
    pattern_cfg_scale = sample_cfg.get("pattern_cfg_scale")
    cfg_scale = sample_cfg.get("cfg_scale")

    if s11_cfg_scale is not None or pattern_cfg_scale is not None:
        if s11_cfg_scale is not None:
            model_kwargs["s11_cfg_scale"] = float(s11_cfg_scale)
        if pattern_cfg_scale is not None:
            model_kwargs["pattern_cfg_scale"] = float(pattern_cfg_scale)
    elif cfg_scale is not None:
        model_kwargs["cfg_scale"] = float(cfg_scale)


def _uniform_denoising_timesteps(num_timesteps, num_steps=5):
    if num_steps <= 1:
        return [int(num_timesteps)]
    timesteps = [1 + (int(num_timesteps) - 1) * idx // (num_steps - 1) for idx in range(num_steps)]
    return list(reversed(timesteps))


@torch.no_grad()
def visualize_random_test_samples(
    diffusion_model,
    surrogate_model,
    test_dataset,
    standardizer,
    dataset_info,
    device,
    cfg,
    output_dir,
    num_visualize=5,
    logger=None,
):
    if num_visualize <= 0:
        return []

    num_visualize = min(int(num_visualize), len(test_dataset))
    generator = torch.Generator()
    generator.manual_seed(int(cfg.get("train", {}).get("seed", 106)))
    local_indices = torch.randperm(len(test_dataset), generator=generator)[:num_visualize].tolist()

    currents = []
    ys = []
    patterns = []
    y_raws = []
    p_raws = []
    records = []

    for local_index in local_indices:
        current, y, pattern, meta = test_dataset[int(local_index)]
        currents.append(current)
        ys.append(y)
        patterns.append(pattern)
        y_raws.append(meta["y_raw"])
        p_raws.append(meta["p_raw"])
        records.append(
            {
                "sample_name": f"sample_{len(records):02d}",
                "test_local_index": int(local_index),
                "dataset_index": _global_index(test_dataset, local_index),
            }
        )

    current_true = torch.stack(currents, dim=0)
    y = torch.stack(ys, dim=0)
    pattern = torch.stack(patterns, dim=0)
    y_true_raw = torch.stack(y_raws, dim=0)
    p_true_raw = torch.stack(p_raws, dim=0)

    model_kwargs = {
        "s11": y.to(device, non_blocking=True),
        "pattern": pattern.to(device, non_blocking=True),
    }
    _apply_sample_cfg_scales(model_kwargs, cfg)

    diffusion_model.eval()
    surrogate_model.eval()

    denoising_timesteps = _uniform_denoising_timesteps(diffusion_model.num_timesteps, num_steps=5)
    generated_current, denoising_snapshots, denoising_timesteps = diffusion_model.sample_with_snapshots(
        batch_size=num_visualize,
        snapshot_timesteps=denoising_timesteps,
        model_kwargs=model_kwargs,
        progress=False,
    )
    y_pred_norm, p_pred_norm = surrogate_model(generated_current)
    y_pred_raw = standardizer.denormalize_y(y_pred_norm).cpu()
    p_pred_raw = standardizer.denormalize_p(p_pred_norm).cpu()

    figure_dir = os.path.join(output_dir, "figures")
    ensure_dir(figure_dir)

    for idx, record in enumerate(records):
        save_diffusion_visualization_sample(
            output_dir=figure_dir,
            sample_name=record["sample_name"],
            generated_current=generated_current[idx].cpu(),
            target_current=current_true[idx],
            y_pred_raw=y_pred_raw[idx],
            y_true_raw=y_true_raw[idx],
            p_pred_raw=p_pred_raw[idx],
            p_true_raw=p_true_raw[idx],
            standardizer=standardizer,
            dataset_info=dataset_info,
            denoising_currents=denoising_snapshots[idx].cpu(),
            denoising_timesteps=denoising_timesteps,
        )
        record["denoising_timesteps"] = [int(timestep) for timestep in denoising_timesteps]
        record["metrics"] = _per_sample_metrics(
            y_pred_raw=y_pred_raw[idx],
            y_true_raw=y_true_raw[idx],
            p_pred_raw=p_pred_raw[idx],
            p_true_raw=p_true_raw[idx],
        )

    save_json(os.path.join(figure_dir, "visualization_indices.json"), records)

    if logger:
        logger.info("Saved %d visualization samples to: %s", num_visualize, figure_dir)

    return records


def main():
    args = parse_args()

    cfg = get_default_config()
    cfg = update_config_from_args(cfg, args)
    checkpoint_path = _resolve_checkpoint(cfg, args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "config" in checkpoint:
        saved_cfg = checkpoint["config"]
        saved_cfg["test"] = saved_cfg.get("test", {})
        saved_cfg["test"]["checkpoint"] = checkpoint_path
        cfg = update_config_from_args(saved_cfg, args)

    cfg = _configure_test_dirs(cfg, checkpoint, checkpoint_path, explicit_output_dir=args.output_dir)

    ensure_dir(cfg["paths"]["output_dir"])
    ensure_dir(cfg["paths"]["log_dir"])

    logger = create_logger(
        name="current_diffusion_test",
        log_dir=cfg["paths"]["log_dir"],
        log_filename="test.log",
    )

    set_seed(cfg["train"]["seed"])
    device = _device_from_cfg(cfg)
    logger.info("Using device: %s", device)
    logger.info("Checkpoint: %s", checkpoint_path)
    logger.info("Configuration:")
    logger.info("\n%s", pprint.pformat(cfg, sort_dicts=False))

    split_cfg = cfg.get("split", {})
    cfg["paths"]["h5_path"] = ensure_standardized_dataset(
        h5_path=cfg["paths"]["h5_path"],
        input_keys=["X", "current"],
        feed_sigma=2.0,
        compression="lzf",
        train_ratio=split_cfg.get("train_ratio", 0.8),
        val_ratio=split_cfg.get("val_ratio", 0.1),
        test_ratio=split_cfg.get("test_ratio", 0.1),
        seed=split_cfg.get("seed", cfg.get("train", {}).get("seed", 106)),
    )

    dataloaders, dataset_info, standardizer = build_diffusion_dataloaders(cfg)
    validate_diffusion_checkpoint_dataset_shapes(checkpoint, dataset_info, context="Diffusion checkpoint")

    diffusion_model = build_current_diffusion(cfg=cfg, dataset_info=dataset_info)
    diffusion_model.load_state_dict(checkpoint["model_state_dict"])
    diffusion_model.to(device)
    diffusion_model.eval()

    surrogate_model, surrogate_checkpoint = load_forward_surrogate(
        project_dir=cfg["surrogate"]["project_dir"],
        checkpoint_path=cfg["surrogate"]["checkpoint"],
        device=device,
    )
    validate_forward_surrogate_shapes(surrogate_checkpoint, dataset_info)

    sample_path = os.path.join(cfg["paths"]["output_dir"], "samples", "test_samples.pt")
    metrics = sample_and_evaluate(
        diffusion_model=diffusion_model,
        surrogate_model=surrogate_model,
        loader=dataloaders["test"],
        standardizer=standardizer,
        device=device,
        cfg=cfg,
        num_samples=cfg.get("test", {}).get("num_samples", 64),
        save_path=sample_path,
        logger=logger,
        split_name="test",
    )

    metrics_path = os.path.join(cfg["paths"]["output_dir"], "test_metrics.json")
    save_json(metrics_path, metrics)
    logger.info("Saved test metrics to: %s", metrics_path)

    if not args.no_visualize:
        visualize_random_test_samples(
            diffusion_model=diffusion_model,
            surrogate_model=surrogate_model,
            test_dataset=dataloaders["test"].dataset,
            standardizer=standardizer,
            dataset_info=dataset_info,
            device=device,
            cfg=cfg,
            output_dir=cfg["paths"]["output_dir"],
            num_visualize=args.num_visualize,
            logger=logger,
        )


if __name__ == "__main__":
    main()
