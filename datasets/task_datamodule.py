# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from datasets.datamodule import load_standardizer_for_dataset, resolve_standardizer_path
from datasets.split_loader import build_split_indices
from datasets.task_dataset import AntennaTaskDataset, RESPONSE_TASKS, STRUCTURE_TASKS


def load_preprocess_split(h5_path, cfg):
    """Load the split saved by datasets/preprocess.py, if available."""
    standardizer_path = resolve_standardizer_path(h5_path, cfg)
    payload = torch.load(standardizer_path, map_location="cpu", weights_only=True)
    split = payload.get("extra", {}).get("split")
    if not split or not all(name in split for name in ("train", "val", "test")):
        return None, standardizer_path

    return {
        name: np.asarray(split[name], dtype=np.int64)
        for name in ("train", "val", "test")
    }, standardizer_path


def build_or_load_split(h5_path, cfg, num_samples):
    split, standardizer_path = load_preprocess_split(h5_path, cfg)
    if split is not None:
        merged = np.concatenate([split[name] for name in ("train", "val", "test")])
        if len(merged) != num_samples:
            raise ValueError(
                f"Saved split contains {len(merged)} indices, but dataset has {num_samples} samples. "
                "Re-run datasets/preprocess.py for this dataset."
            )
        if merged.size and (int(merged.min()) < 0 or int(merged.max()) >= num_samples):
            raise ValueError("Saved dataset split contains out-of-range indices.")
        if len(np.unique(merged)) != num_samples:
            raise ValueError("Saved dataset split has duplicate or missing sample indices.")
        return split, standardizer_path, "standardizer_extra"

    split_cfg = cfg.get("split", {})
    split = build_split_indices(
        num_samples=num_samples,
        train_ratio=split_cfg.get("train_ratio", 0.8),
        val_ratio=split_cfg.get("val_ratio", 0.1),
        test_ratio=split_cfg.get("test_ratio", 0.1),
        seed=split_cfg.get("seed", cfg.get("train", {}).get("seed", 106)),
    )
    return split, standardizer_path, "fallback_config"


def _loader_kwargs(cfg):
    data_cfg = cfg.get("data", {})
    kwargs = {
        "batch_size": data_cfg["batch_size"],
        "num_workers": data_cfg["num_workers"],
        "pin_memory": data_cfg.get("pin_memory", False),
    }
    if data_cfg["num_workers"] > 0:
        kwargs["persistent_workers"] = data_cfg.get("persistent_workers", False)
        kwargs["prefetch_factor"] = data_cfg.get("prefetch_factor", 2)
    return kwargs


def build_task_dataloaders(cfg, standardizer=None):
    """Build DataLoaders for response or structure tasks.

    This is the unified task-aware entry point. It keeps the original
    build_dataloaders untouched for backward compatibility, and reuses the
    split saved during preprocessing whenever possible.
    """
    h5_path = cfg["paths"]["h5_path"]
    data_cfg = cfg.get("data", {})
    task_cfg = cfg.get("task", {})
    task_name = task_cfg.get("name", "current_to_response")

    if task_name == "x_to_response":
        default_input_key = "X"
    else:
        default_input_key = "current"

    input_key = task_cfg.get("input_key", data_cfg.get("input_key", default_input_key))
    x_key = task_cfg.get("x_key", data_cfg.get("x_key", "X"))

    dataset = AntennaTaskDataset(
        h5_path=h5_path,
        task_name=task_name,
        input_key=input_key,
        x_key=x_key,
        return_raw=(task_name in RESPONSE_TASKS),
    )

    if task_name in RESPONSE_TASKS:
        if standardizer is None:
            standardizer, standardizer_path = load_standardizer_for_dataset(h5_path, cfg)
        else:
            standardizer_path = data_cfg.get("standardizer_path")
        standardizer.validate_shapes(dataset.y_shape, dataset.pattern_shape, x_shape=None)
    else:
        standardizer = None

    split, split_standardizer_path, split_source = build_or_load_split(
        h5_path=h5_path,
        cfg=cfg,
        num_samples=len(dataset),
    )
    standardizer_path = data_cfg.get("standardizer_path") or split_standardizer_path

    kwargs = _loader_kwargs(cfg)
    loaders = {
        "train": DataLoader(Subset(dataset, split["train"]), shuffle=True, **kwargs),
        "val": DataLoader(Subset(dataset, split["val"]), shuffle=False, **kwargs),
        "test": DataLoader(Subset(dataset, split["test"]), shuffle=False, **kwargs),
    }

    dataset_info = {
        "task_name": task_name,
        "input_key": input_key,
        "standardized": True,
        "standardizer_path": standardizer_path,
        "split_source": split_source,
        "num_samples": len(dataset),
        "input_shape": dataset.input_shape,
        "y_shape": dataset.y_shape,
        "pattern_shape": dataset.pattern_shape,
        "pattern_metadata": dataset.pattern_metadata,
        "split_sizes": {name: int(len(indices)) for name, indices in split.items()},
    }
    if task_name in STRUCTURE_TASKS:
        dataset_info.update({
            "x_key": x_key,
            "x_shape": dataset.x_shape,
            "target_shape": dataset.structure_shape,
        })

    if task_name in RESPONSE_TASKS:
        return loaders, dataset_info, standardizer
    return loaders, dataset_info
