# -*- coding: utf-8 -*-

import os

import h5py
from torch.utils.data import DataLoader, Subset

from datasets.h5_dataset import H5AntennaDataset
from datasets.preprocess import TargetStandardizer, get_unified_standardizer_path
from datasets.split_loader import build_split_indices


def _decode_attr(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def resolve_standardizer_path(h5_path, cfg):
    data_cfg = cfg.get("data", {})
    explicit_path = data_cfg.get("standardizer_path")
    if explicit_path:
        return explicit_path

    with h5py.File(h5_path, "r") as f:
        attr_path = _decode_attr(f.attrs.get("standardizer_path"))
        if attr_path and os.path.isfile(attr_path):
            return attr_path

    fallback_path = get_unified_standardizer_path(h5_path)
    if os.path.isfile(fallback_path):
        return fallback_path

    if ".preprocessed.standardized" in h5_path:
        raw_base = h5_path.replace(".preprocessed.standardized.h5", ".standardizer.pt")
        if os.path.isfile(raw_base):
            return raw_base

    raise FileNotFoundError(
        "Unified standardizer not found. Run datasets/preprocess.py first, "
        "or set cfg['data']['standardizer_path']."
    )


def load_standardizer_for_dataset(h5_path, cfg):
    standardizer_path = resolve_standardizer_path(h5_path, cfg)
    print(f"[standardizer] load unified stats from: {standardizer_path}")
    return TargetStandardizer.load(standardizer_path), standardizer_path


def build_dataloaders(cfg, standardizer=None):
    h5_path = cfg["paths"]["h5_path"]
    data_cfg = cfg.get("data", {})
    input_key = data_cfg.get("input_key", "X")

    base_dataset = H5AntennaDataset(
        h5_path,
        standardizer=None,
        return_raw=False,
        input_key=input_key,
    )
    if base_dataset.layout != "standardized":
        raise ValueError(
            f"Expected a .preprocessed.standardized.h5 dataset, got layout={base_dataset.layout}. "
            "Run: python datasets/preprocess.py <raw_dataset.h5>"
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
        standardizer, standardizer_path = load_standardizer_for_dataset(h5_path, cfg)
    else:
        standardizer_path = data_cfg.get("standardizer_path")

    standardizer.validate_shapes(
        base_dataset.y_shape,
        base_dataset.pattern_shape,
        x_shape=None,
    )

    train_full_dataset = H5AntennaDataset(
        h5_path,
        standardizer=None,
        return_raw=False,
        input_key=input_key,
    )

    eval_full_dataset = H5AntennaDataset(
        h5_path,
        standardizer=None,
        return_raw=True,
        input_key=input_key,
    )

    train_ds = Subset(train_full_dataset, split["train"])
    val_ds = Subset(eval_full_dataset, split["val"])
    test_ds = Subset(eval_full_dataset, split["test"])

    batch_size = data_cfg["batch_size"]
    num_workers = data_cfg["num_workers"]
    pin_memory = data_cfg.get("pin_memory", False)
    persistent_workers = data_cfg.get("persistent_workers", False)
    prefetch_factor = data_cfg.get("prefetch_factor", 2)

    common_loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    if num_workers > 0:
        common_loader_kwargs["persistent_workers"] = persistent_workers
        common_loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(train_ds, shuffle=True, **common_loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **common_loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **common_loader_kwargs)

    dataset_info = {
        "layout": train_full_dataset.layout,
        "input_key": input_key,
        "standardized": True,
        "standardizer_path": standardizer_path,
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
