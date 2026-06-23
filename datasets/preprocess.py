# -*- coding: utf-8 -*-

import argparse
import os
import sys

import h5py
import numpy as np
import torch

_WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, _WORKSPACE_DIR)

from datasets.build_preprocessed_h5 import ensure_preprocessed_h5, get_preprocessed_h5_path
from datasets.h5_dataset import H5AntennaDataset, read_h5_dataset_info
from datasets.split_loader import build_split_indices


DEFAULT_PREPROCESS_CONFIG = {
    #"h5_path": os.path.join(_WORKSPACE_DIR, "datasets", "antenna_dataset_20260614_203654.h5"),
    "h5_path": os.path.join(_WORKSPACE_DIR, "datasets", "antenna_dataset_20260614_203654.h5"),
    "input_keys": ["X", "current"],
    "feed_sigma": 2.0,
    "compression": "lzf",
    "force": False,
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "seed": 106,
}


class TargetStandardizer:
    def __init__(
        self,
        y_mean,
        y_std,
        p_mean,
        p_std,
        x_mean=None,
        x_std=None,
        input_stats=None,
    ):
        self.y_mean = y_mean
        self.y_std = y_std
        self.p_mean = p_mean
        self.p_std = p_std
        self.x_mean = x_mean
        self.x_std = x_std
        self.input_stats = input_stats or {}

    def normalize_x(self, x):
        if self.x_mean is None or self.x_std is None:
            return x
        x_mean = self.x_mean.to(x.device)
        x_std = self.x_std.to(x.device)
        return (x - x_mean) / x_std

    def normalize_input(self, name, x):
        stats = self.input_stats.get(name)
        if stats is None:
            return x
        if stats.get("transform") == "affine_01_to_m11":
            return x * 2.0 - 1.0
        if stats.get("transform") == "signed_log_clip_zscore":
            alpha = stats["alpha"].to(x.device)
            mean = stats["mean"].to(x.device)
            std = stats["std"].to(x.device)
            clip = float(stats["clip"])
            if x.ndim == 3:  # CNN view: (F*C, H, W)
                alpha, mean, std = (v.reshape(-1, 1, 1) for v in (alpha, mean, std))
            elif x.ndim == 4:  # one stored sample: (F, C, H, W)
                alpha, mean, std = (v.reshape(*v.shape, 1, 1) for v in (alpha, mean, std))
            else:  # batch stored view: (N, F, C, H, W)
                alpha, mean, std = (v.reshape(1, *v.shape, 1, 1) for v in (alpha, mean, std))
            transformed = torch.sign(x) * torch.log1p(torch.abs(x) / alpha)
            return (torch.clamp(transformed, -clip, clip) - mean) / std
        mean = stats["mean"].to(x.device)
        std = stats["std"].to(x.device)
        return (x - mean) / std

    def normalize_y(self, y):
        y_mean = self.y_mean.to(y.device)
        y_std = self.y_std.to(y.device)
        return (y - y_mean) / y_std

    def normalize_p(self, p):
        p_mean = self.p_mean.to(p.device)
        p_std = self.p_std.to(p.device)
        return (p - p_mean) / p_std

    def denormalize_y(self, y):
        y_mean = self.y_mean.to(y.device)
        y_std = self.y_std.to(y.device)
        return y * y_std + y_mean

    def denormalize_p(self, p):
        p_mean = self.p_mean.to(p.device)
        p_std = self.p_std.to(p.device)
        return p * p_std + p_mean

    def is_compatible(self, y_shape, p_shape, x_shape=None):
        y_stats_shape = tuple(self.y_mean.shape)
        p_stats_shape = tuple(self.p_mean.shape)
        target_ok = (
            y_stats_shape in {tuple(y_shape), (1,)}
            and tuple(self.y_std.shape) == y_stats_shape
            and p_stats_shape in {tuple(p_shape), tuple(p_shape[:-1]) + (1,)}
            and tuple(self.p_std.shape) == p_stats_shape
        )
        if not target_ok:
            return False
        if x_shape is None:
            return True
        if self.x_mean is None or self.x_std is None:
            return False
        return (
            tuple(self.x_mean.shape) == tuple(x_shape)
            and tuple(self.x_std.shape) == tuple(x_shape)
        )

    def validate_shapes(self, y_shape, p_shape, x_shape=None):
        if not self.is_compatible(y_shape, p_shape, x_shape=x_shape):
            raise ValueError(
                "Standardizer shape mismatch: "
                f"expected x={None if x_shape is None else tuple(x_shape)}, "
                f"y={tuple(y_shape)}, pattern={tuple(p_shape)}, "
                f"got x_mean={None if self.x_mean is None else tuple(self.x_mean.shape)}, "
                f"y_mean={tuple(self.y_mean.shape)}, p_mean={tuple(self.p_mean.shape)}. "
                "Recompute stats or use a checkpoint trained on this dataset."
            )

    def state_dict(self):
        state = {
            "y_mean": self.y_mean.detach().cpu(),
            "y_std": self.y_std.detach().cpu(),
            "p_mean": self.p_mean.detach().cpu(),
            "p_std": self.p_std.detach().cpu(),
        }
        if self.input_stats:
            state["input_stats"] = {
                name: {
                    key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                    for key, value in stats.items()
                }
                for name, stats in self.input_stats.items()
            }
        if self.x_mean is not None and self.x_std is not None:
            state["x_mean"] = self.x_mean.detach().cpu()
            state["x_std"] = self.x_std.detach().cpu()
        return state

    @classmethod
    def from_state_dict(cls, state_dict):
        return cls(
            y_mean=state_dict["y_mean"].float(),
            y_std=state_dict["y_std"].float(),
            p_mean=state_dict["p_mean"].float(),
            p_std=state_dict["p_std"].float(),
            x_mean=state_dict.get("x_mean", None).float() if state_dict.get("x_mean", None) is not None else None,
            x_std=state_dict.get("x_std", None).float() if state_dict.get("x_std", None) is not None else None,
            input_stats={
                name: {
                    key: value.float() if isinstance(value, torch.Tensor) else value
                    for key, value in stats.items()
                }
                for name, stats in state_dict.get("input_stats", {}).items()
            },
        )

    def save(self, path, extra=None):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "standardizer_stats": self.state_dict(),
        }
        if extra is not None:
            payload["extra"] = extra
        torch.save(payload, path)

    @classmethod
    def load(cls, path):
        payload = torch.load(path, map_location="cpu")
        if "standardizer_stats" in payload:
            state_dict = payload["standardizer_stats"]
        else:
            state_dict = payload
        return cls.from_state_dict(state_dict)


def compute_stats(
    dataset,
    indices,
    normalize_input=False,
):
    x_sum = 0
    x_sq = 0
    y_sum = 0
    y_sq = 0
    p_sum = 0
    p_sq = 0
    count = 0

    for idx in indices:
        x, y, p, _ = dataset[idx]
        x = x.double()
        y = y.double()
        p = p.double()

        if count == 0:
            if normalize_input:
                x_sum = torch.zeros_like(x)
                x_sq = torch.zeros_like(x)
            y_sum = torch.zeros_like(y)
            y_sq = torch.zeros_like(y)
            p_sum = torch.zeros_like(p)
            p_sq = torch.zeros_like(p)

        if normalize_input:
            x_sum += x
            x_sq += x * x
        y_sum += y
        y_sq += y * y
        p_sum += p
        p_sq += p * p
        count += 1

    y_mean = y_sum / count
    p_mean = p_sum / count

    y_std = torch.sqrt(torch.clamp(y_sq / count - y_mean**2, min=1e-8))
    p_std = torch.sqrt(torch.clamp(p_sq / count - p_mean**2, min=1e-8))

    x_mean = None
    x_std = None
    if normalize_input:
        x_mean = x_sum / count
        x_std = torch.sqrt(torch.clamp(x_sq / count - x_mean**2, min=1e-8))

    return TargetStandardizer(
        y_mean.float(), y_std.float(),
        p_mean.float(), p_std.float(),
        x_mean.float() if x_mean is not None else None,
        x_std.float() if x_std is not None else None,
    )


def get_standardized_h5_path(preprocessed_h5_path):
    base, ext = os.path.splitext(preprocessed_h5_path)
    if base.endswith(".preprocessed"):
        base = base[: -len(".preprocessed")] + ".preprocessed.standardized"
    else:
        base = base + ".standardized"
    return f"{base}{ext}"


def get_unified_standardizer_path(preprocessed_h5_path):
    base, _ = os.path.splitext(preprocessed_h5_path)
    if base.endswith(".preprocessed"):
        base = base[: -len(".preprocessed")]
    return f"{base}.standardizer.pt"


def get_raw_h5_path_from_standardized_path(standardized_h5_path):
    suffix = ".preprocessed.standardized.h5"
    if standardized_h5_path.endswith(suffix):
        return standardized_h5_path[: -len(suffix)] + ".h5"
    base, ext = os.path.splitext(standardized_h5_path)
    if base.endswith(".preprocessed.standardized"):
        return base[: -len(".preprocessed.standardized")] + ext
    return standardized_h5_path


def get_preprocessed_h5_path_from_standardized_path(standardized_h5_path):
    suffix = ".preprocessed.standardized.h5"
    if standardized_h5_path.endswith(suffix):
        return standardized_h5_path[: -len(suffix)] + ".preprocessed.h5"
    base, ext = os.path.splitext(standardized_h5_path)
    if base.endswith(".preprocessed.standardized"):
        return base[: -len(".standardized")] + ext
    return standardized_h5_path


def _is_current_standardized_h5(path):
    try:
        with h5py.File(path, "r") as f:
            return f.attrs.get("preprocessing_schema") == "antenna-v2"
    except OSError:
        return False


def _as_float_array(x):
    return np.asarray(x, dtype=np.float32)


def _training_mask(num_samples, indices):
    mask = np.zeros(num_samples, dtype=bool)
    mask[np.asarray(indices, dtype=np.int64)] = True
    return mask


def _compute_mean_std(preprocessed_h5_path, indices, input_keys, chunk_size=256):
    """Compute all statistics from training samples only, in their requested domains."""
    with h5py.File(preprocessed_h5_path, "r") as f:
        n = f["Y"].shape[0]
        train_mask = _training_mask(n, indices)
        y_sum = y_sq = 0.0
        p_sum = p_sq = None
        y_count = p_count = 0
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            selected = train_mask[start:end]
            if not selected.any():
                continue
            y = np.asarray(f["Y"][start:end][selected], dtype=np.float64)
            p = np.asarray(f["pattern"][start:end][selected], dtype=np.float64)
            # S11: one scalar mean/std shared by all 41 frequencies.
            y_sum += y.sum()
            y_sq += np.square(y).sum()
            y_count += y.size
            # Pattern: reduce sample and theta axes, retain (F, C).
            if p_sum is None:
                p_sum = np.zeros(p.shape[1:3], dtype=np.float64)
                p_sq = np.zeros_like(p_sum)
            p_sum += p.sum(axis=(0, 3))
            p_sq += np.square(p).sum(axis=(0, 3))
            p_count += p.shape[0] * p.shape[3]

        y_mean = np.asarray([y_sum / y_count], dtype=np.float32)
        y_std = np.asarray([np.sqrt(max(y_sq / y_count - y_mean[0] ** 2, 1e-8))], dtype=np.float32)
        p_mean = (p_sum / p_count)[..., None].astype(np.float32)
        p_std = np.sqrt(np.maximum(p_sq / p_count - (p_sum / p_count) ** 2, 1e-8))[..., None].astype(np.float32)

        input_stats = {}
        if "X" in input_keys:
            input_stats["X"] = {"transform": "affine_01_to_m11"}

        if "current" in input_keys:
            current = f["current"]
            if current.ndim != 5:
                raise ValueError(f"/current must be (N,F,C,H,W), got {current.shape}")
            _, nf, nc, _, _ = current.shape
            alpha = np.empty((nf, nc), dtype=np.float32)
            # p95 is exact, calculated independently per (frequency, component).
            for fi in range(nf):
                for ci in range(nc):
                    values = []
                    for start in range(0, n, chunk_size):
                        end = min(start + chunk_size, n)
                        selected = train_mask[start:end]
                        if selected.any():
                            values.append(np.abs(current[start:end, fi, ci][selected]).reshape(-1))
                    alpha[fi, ci] = max(float(np.percentile(np.concatenate(values), 95)), 1e-8)

            cur_sum = np.zeros((nf, nc), dtype=np.float64)
            cur_sq = np.zeros_like(cur_sum)
            cur_count = 0
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                selected = train_mask[start:end]
                if not selected.any():
                    continue
                j = np.asarray(current[start:end][selected], dtype=np.float64)
                transformed = np.sign(j) * np.log1p(np.abs(j) / alpha[None, :, :, None, None])
                transformed = np.clip(transformed, -4.0, 4.0)
                cur_sum += transformed.sum(axis=(0, 3, 4))
                cur_sq += np.square(transformed).sum(axis=(0, 3, 4))
                cur_count += transformed.shape[0] * transformed.shape[3] * transformed.shape[4]
            cur_mean = (cur_sum / cur_count).astype(np.float32)
            cur_std = np.sqrt(np.maximum(cur_sq / cur_count - (cur_sum / cur_count) ** 2, 1e-8)).astype(np.float32)
            input_stats["current"] = {
                "transform": "signed_log_clip_zscore", "alpha": torch.from_numpy(alpha),
                "clip": 4.0, "mean": torch.from_numpy(cur_mean), "std": torch.from_numpy(cur_std),
            }

    return TargetStandardizer(torch.from_numpy(y_mean), torch.from_numpy(y_std),
                              torch.from_numpy(p_mean), torch.from_numpy(p_std), input_stats=input_stats)


def _copy_metadata(src, dst):
    for name in ("freq_hz", "pattern_freq_hz", "pattern_theta_deg", "current_freq_hz"):
        if name in src and name not in dst:
            src.copy(name, dst)
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def _create_like(dst, name, source, compression, chunks=None):
    if name in dst:
        del dst[name]
    return dst.create_dataset(
        name,
        shape=source.shape,
        dtype=np.float32,
        compression=compression,
        chunks=chunks or source.chunks,
    )


def build_standardized_h5(
    preprocessed_h5_path,
    standardized_h5_path=None,
    standardizer_path=None,
    input_keys=None,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=106,
    compression="lzf",
    force=False,
):
    if not os.path.isfile(preprocessed_h5_path):
        raise FileNotFoundError(preprocessed_h5_path)

    input_keys = list(input_keys or ["X", "current"])
    standardized_h5_path = standardized_h5_path or get_standardized_h5_path(preprocessed_h5_path)
    standardizer_path = standardizer_path or get_unified_standardizer_path(preprocessed_h5_path)

    if os.path.isfile(standardized_h5_path) and os.path.isfile(standardizer_path) and not force:
        try:
            with h5py.File(standardized_h5_path, "r") as existing:
                compatible = existing.attrs.get("preprocessing_schema") == "antenna-v2"
            if compatible:
                print(f"[preprocess] reuse standardized h5: {standardized_h5_path}")
                print(f"[preprocess] reuse standardizer: {standardizer_path}")
                return standardized_h5_path, standardizer_path
            print("[preprocess] standardized cache uses an older preprocessing schema; rebuilding...")
        except OSError:
            pass

    base_dataset = H5AntennaDataset(preprocessed_h5_path, return_raw=False, input_key=input_keys[0])
    split = build_split_indices(
        num_samples=len(base_dataset),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    standardizer = _compute_mean_std(preprocessed_h5_path, split["train"], input_keys=input_keys)
    standardizer.save(
        standardizer_path,
        extra={
            "preprocessed_h5_path": preprocessed_h5_path,
            "standardized_h5_path": standardized_h5_path,
            "input_keys": input_keys,
            "split": {k: v.tolist() for k, v in split.items()},
        },
    )

    os.makedirs(os.path.dirname(standardized_h5_path) or ".", exist_ok=True)
    with h5py.File(preprocessed_h5_path, "r") as src, h5py.File(standardized_h5_path, "w") as dst:
        _copy_metadata(src, dst)
        dst.attrs["preprocessed"] = True
        dst.attrs["standardized"] = True
        dst.attrs["preprocessing_schema"] = "antenna-v2"
        dst.attrs["x_transform"] = "2*x-1"
        dst.attrs["current_transform"] = "signed_log(alpha=p95_train_per_f_c), clip=4, zscore_per_f_c"
        dst.attrs["y_stats_scope"] = "global_train_samples_and_frequencies"
        dst.attrs["pattern_stats_scope"] = "train_samples_and_theta_per_frequency_component"
        dst.attrs["standardizer_path"] = os.path.abspath(standardizer_path)
        dst.attrs["source_preprocessed_h5"] = os.path.abspath(preprocessed_h5_path)
        dst.attrs["preprocessed_inputs"] = ",".join(input_keys)

        y_src = src["Y"]
        p_src = src["pattern"]
        y_dst = _create_like(dst, "Y", y_src, compression)
        p_dst = _create_like(dst, "pattern", p_src, compression)
        y_raw_dst = _create_like(dst, "Y_raw", y_src, compression)
        p_raw_dst = _create_like(dst, "pattern_raw", p_src, compression)

        input_dsts = {}
        for key in input_keys:
            if key not in src:
                raise KeyError(f"HDF5 must contain /{key}")
            input_dsts[key] = _create_like(dst, key, src[key], compression)

        n = y_src.shape[0]
        for start in range(0, n, 256):
            end = min(start + 256, n)
            y = _as_float_array(y_src[start:end])
            p = _as_float_array(p_src[start:end])
            y_raw_dst[start:end] = y
            p_raw_dst[start:end] = p
            y_dst[start:end] = standardizer.normalize_y(torch.from_numpy(y)).numpy().astype(np.float32)
            p_dst[start:end] = standardizer.normalize_p(torch.from_numpy(p)).numpy().astype(np.float32)

            for key, x_dst in input_dsts.items():
                x = _as_float_array(src[key][start:end])
                x_dst[start:end] = standardizer.normalize_input(key, torch.from_numpy(x)).numpy().astype(np.float32)

            if end == n or end % 1024 == 0:
                print(f"[preprocess] standardized [{end}/{n}]")

    print(f"[preprocess] saved standardized h5: {standardized_h5_path}")
    print(f"[preprocess] saved unified standardizer: {standardizer_path}")
    return standardized_h5_path, standardizer_path


def preprocess_dataset(
    src_h5_path,
    input_keys=None,
    feed_sigma=2.0,
    compression="lzf",
    force=False,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=106,
):
    input_keys = input_keys or ["X", "current"]
    preprocessed_h5_path = ensure_preprocessed_h5(
        src_h5_path=src_h5_path,
        feed_sigma=feed_sigma,
        compression=compression,
        force=force,
        input_key=input_keys[0],
        input_keys=input_keys,
    )
    return build_standardized_h5(
        preprocessed_h5_path=preprocessed_h5_path,
        input_keys=input_keys,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        compression=compression,
        force=force,
    )


def ensure_standardized_dataset(
    h5_path,
    input_keys=None,
    feed_sigma=2.0,
    compression="lzf",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=106,
):
    """Return a standardized HDF5 path, creating missing preprocessing outputs."""
    input_keys = input_keys or ["X", "current"]

    if os.path.isfile(h5_path):
        if h5_path.endswith(".preprocessed.standardized.h5"):
            if _is_current_standardized_h5(h5_path):
                return h5_path
            raw_h5_path = get_raw_h5_path_from_standardized_path(h5_path)
            if not os.path.isfile(raw_h5_path):
                raise RuntimeError(
                    f"{h5_path} uses an obsolete preprocessing schema and source HDF5 is unavailable. "
                    "Restore the raw dataset and rerun preprocessing."
                )
            standardized_h5_path, _ = preprocess_dataset(
                src_h5_path=raw_h5_path, input_keys=input_keys, feed_sigma=feed_sigma,
                compression=compression, force=False, train_ratio=train_ratio,
                val_ratio=val_ratio, test_ratio=test_ratio, seed=seed,
            )
            return standardized_h5_path

        if h5_path.endswith(".preprocessed.h5"):
            standardized_h5_path = get_standardized_h5_path(h5_path)
            if os.path.isfile(standardized_h5_path) and _is_current_standardized_h5(standardized_h5_path):
                return standardized_h5_path
            standardized_h5_path, _ = build_standardized_h5(
                preprocessed_h5_path=h5_path,
                standardized_h5_path=standardized_h5_path,
                input_keys=input_keys,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed,
                compression=compression,
                force=False,
            )
            return standardized_h5_path

        preprocessed_h5_path = get_preprocessed_h5_path(h5_path)
        standardized_h5_path = get_standardized_h5_path(preprocessed_h5_path)
        if os.path.isfile(standardized_h5_path) and _is_current_standardized_h5(standardized_h5_path):
            return standardized_h5_path
        standardized_h5_path, _ = preprocess_dataset(
            src_h5_path=h5_path,
            input_keys=input_keys,
            feed_sigma=feed_sigma,
            compression=compression,
            force=False,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        return standardized_h5_path

    preprocessed_h5_path = get_preprocessed_h5_path_from_standardized_path(h5_path)
    if os.path.isfile(preprocessed_h5_path):
        standardized_h5_path, _ = build_standardized_h5(
            preprocessed_h5_path=preprocessed_h5_path,
            standardized_h5_path=h5_path,
            input_keys=input_keys,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            compression=compression,
            force=False,
        )
        return standardized_h5_path

    raw_h5_path = get_raw_h5_path_from_standardized_path(h5_path)
    if not os.path.isfile(raw_h5_path):
        raise FileNotFoundError(
            "Neither standardized, preprocessed, nor raw HDF5 dataset exists: "
            f"standardized={h5_path}, preprocessed={preprocessed_h5_path}, raw={raw_h5_path}"
        )

    standardized_h5_path, _ = preprocess_dataset(
        src_h5_path=raw_h5_path,
        input_keys=input_keys,
        feed_sigma=feed_sigma,
        compression=compression,
        force=False,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    return standardized_h5_path


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess antenna HDF5 dataset")
    parser.add_argument(
        "h5_path",
        nargs="?",
        default=DEFAULT_PREPROCESS_CONFIG["h5_path"],
        help="Raw MATLAB-layout .h5 path",
    )
    parser.add_argument(
        "--input_keys",
        default=",".join(DEFAULT_PREPROCESS_CONFIG["input_keys"]),
        help="Comma-separated inputs to preprocess",
    )
    parser.add_argument("--feed_sigma", type=float, default=DEFAULT_PREPROCESS_CONFIG["feed_sigma"])
    parser.add_argument("--compression", default=DEFAULT_PREPROCESS_CONFIG["compression"])
    parser.add_argument("--force", action="store_true", default=DEFAULT_PREPROCESS_CONFIG["force"])
    parser.add_argument("--train_ratio", type=float, default=DEFAULT_PREPROCESS_CONFIG["train_ratio"])
    parser.add_argument("--val_ratio", type=float, default=DEFAULT_PREPROCESS_CONFIG["val_ratio"])
    parser.add_argument("--test_ratio", type=float, default=DEFAULT_PREPROCESS_CONFIG["test_ratio"])
    parser.add_argument("--seed", type=int, default=DEFAULT_PREPROCESS_CONFIG["seed"])
    return parser.parse_args()


def main():
    args = parse_args()
    input_keys = [part.strip() for part in args.input_keys.split(",") if part.strip()]
    preprocess_dataset(
        src_h5_path=args.h5_path,
        input_keys=input_keys,
        feed_sigma=args.feed_sigma,
        compression=args.compression,
        force=args.force,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
