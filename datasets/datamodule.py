# -*- coding: utf-8 -*-

import os
from torch.utils.data import DataLoader, Subset

from datasets.h5_dataset import H5AntennaDataset
from datasets.split_loader import build_split_indices
from datasets.preprocess import compute_stats, TargetStandardizer


def get_standardizer_cache_path(h5_path, input_key="X"):
    base, _ = os.path.splitext(h5_path)
    suffix = "standardizer" if input_key == "X" else f"{input_key}.standardizer"
    return f"{base}.{suffix}.pt"


def load_or_compute_standardizer(
    h5_path,
    train_indices,
    x_shape,
    y_shape,
    pattern_shape,
    force_recompute=False,
    input_key="X",
    normalize_input=False,
    input_std_min=0.02,
    input_clip=20.0,
):
    stats_path = get_standardizer_cache_path(h5_path, input_key=input_key)

    if (not force_recompute) and os.path.isfile(stats_path):
        print(f"[standardizer] load cached stats from: {stats_path}")
        standardizer = TargetStandardizer.load(stats_path)
        if standardizer.is_compatible(
            y_shape,
            pattern_shape,
            x_shape=x_shape if normalize_input else None,
        ):
            return standardizer
        print("[standardizer] cached stats shape mismatch, recomputing...")

    print("[standardizer] computing stats from training split...")
    base_dataset = H5AntennaDataset(
        h5_path,
        standardizer=None,
        return_raw=False,
        input_key=input_key,
    )
    standardizer = compute_stats(
        base_dataset,
        train_indices,
        normalize_input=normalize_input,
        input_std_min=input_std_min,
        input_clip=input_clip,
    )
    standardizer.validate_shapes(
        y_shape,
        pattern_shape,
        x_shape=x_shape if normalize_input else None,
    )

    standardizer.save(
        stats_path,
        extra={
            "h5_path": h5_path,
            "input_key": input_key,
            "normalize_input": normalize_input,
            "input_std_min": input_std_min,
            "input_clip": input_clip,
            "num_train_samples": len(train_indices),
        }
    )
    print(f"[standardizer] saved stats to: {stats_path}")

    return standardizer


def build_dataloaders(cfg, standardizer=None):
    h5_path = cfg["paths"]["h5_path"]
    data_cfg = cfg.get("data", {})
    input_key = data_cfg.get("input_key", "X")
    normalize_input = data_cfg.get("normalize_input", False)
    input_std_min = data_cfg.get("input_std_min", 0.02)
    input_clip = data_cfg.get("input_clip", 20.0)

    base_dataset = H5AntennaDataset(
        h5_path,
        standardizer=None,
        return_raw=False,
        input_key=input_key,
    )
    split_cfg = cfg.get("split", {})
    split = build_split_indices(
        num_samples=len(base_dataset),
        train_ratio=split_cfg.get("train_ratio", 0.8),
        val_ratio=split_cfg.get("val_ratio", 0.1),
        test_ratio=split_cfg.get("test_ratio", 0.1),
        seed=split_cfg.get("seed", cfg.get("train", {}).get("seed", 42)),
    )

    if standardizer is None:
        force_recompute = cfg.get("preprocess", {}).get("force_recompute_stats", False)
        standardizer = load_or_compute_standardizer(
            h5_path=h5_path,
            train_indices=split["train"],
            x_shape=base_dataset.x_shape,
            y_shape=base_dataset.y_shape,
            pattern_shape=base_dataset.pattern_shape,
            force_recompute=force_recompute,
            input_key=input_key,
            normalize_input=normalize_input,
            input_std_min=input_std_min,
            input_clip=input_clip,
        )
    else:
        standardizer.validate_shapes(
            base_dataset.y_shape,
            base_dataset.pattern_shape,
            x_shape=base_dataset.x_shape if normalize_input else None,
        )

    train_full_dataset = H5AntennaDataset(
        h5_path,
        standardizer=standardizer,
        return_raw=False,
        input_key=input_key,
    )

    eval_full_dataset = H5AntennaDataset(
        h5_path,
        standardizer=standardizer,
        return_raw=True,
        input_key=input_key,
    )

    train_ds = Subset(train_full_dataset, split["train"])
    val_ds = Subset(eval_full_dataset, split["val"])
    test_ds = Subset(eval_full_dataset, split["test"])

    batch_size = cfg["data"]["batch_size"]
    num_workers = cfg["data"]["num_workers"]
    pin_memory = cfg["data"].get("pin_memory", False)
    persistent_workers = cfg["data"].get("persistent_workers", False)
    prefetch_factor = cfg["data"].get("prefetch_factor", 2)

    common_loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    if num_workers > 0:
        common_loader_kwargs["persistent_workers"] = persistent_workers
        common_loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        **common_loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **common_loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        shuffle=False,
        **common_loader_kwargs,
    )

    dataset_info = {
        "layout": train_full_dataset.layout,
        "input_key": input_key,
        "normalize_input": normalize_input,
        "x_shape": train_full_dataset.x_shape,
        "y_shape": train_full_dataset.y_shape,
        "pattern_shape": train_full_dataset.pattern_shape,
        "pattern_metadata": train_full_dataset.pattern_metadata,
        "num_samples": len(base_dataset),
        "split_sizes": {
            "train": len(split["train"]),
            "val": len(split["val"]),
            "test": len(split["test"]),
        },
    }

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }, dataset_info, standardizer
