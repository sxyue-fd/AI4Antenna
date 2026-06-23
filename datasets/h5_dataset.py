# -*- coding: utf-8 -*-

import os
import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def build_feed_gaussian_map(height, width, feed_y, feed_x, sigma=1.5):
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )

    dist2 = (yy - float(feed_y)) ** 2 + (xx - float(feed_x)) ** 2
    heatmap = np.exp(-dist2 / (2.0 * sigma * sigma)).astype(np.float32)

    max_val = heatmap.max()
    if max_val > 0:
        heatmap /= max_val

    return heatmap


def convert_x_onehot_feed_to_gaussian(x, sigma=1.5):
    x = np.asarray(x, dtype=np.float32)

    if x.ndim == 2:
        encoded = x
        structure = (encoded > 0.5).astype(np.float32)
        feed_map = (encoded > 1.5).astype(np.float32)
    elif x.ndim == 3 and x.shape[2] == 1:
        encoded = x[:, :, 0]
        structure = (encoded > 0.5).astype(np.float32)
        feed_map = (encoded > 1.5).astype(np.float32)
    elif x.ndim == 3 and x.shape[2] == 2:
        structure = x[:, :, 0].astype(np.float32)
        feed_map = x[:, :, 1].astype(np.float32)
    else:
        raise ValueError(f"Expected x.shape=(H,W), (H,W,1), or (H,W,2), got {x.shape}")

    h, w = structure.shape

    feed_indices = np.argwhere(feed_map > 0.5)

    if len(feed_indices) == 0:
        raise ValueError("Feed point not found in input X")

    if len(feed_indices) > 1:
        raise ValueError(f"Expected one feed point, got {len(feed_indices)}")

    feed_y, feed_x = feed_indices[0]

    feed_gaussian = build_feed_gaussian_map(
        height=h,
        width=w,
        feed_y=int(feed_y),
        feed_x=int(feed_x),
        sigma=sigma,
    )

    x_new = np.stack([structure, feed_gaussian], axis=-1).astype(np.float32)
    return x_new


def normalize_x(x, sigma=1.5):
    x = convert_x_onehot_feed_to_gaussian(x, sigma=sigma)
    return np.transpose(x, (2, 1, 0)).astype(np.float32)


def normalize_current(current):
    current = np.asarray(current, dtype=np.float32)
    if current.ndim != 4:
        raise ValueError(f"Expected current sample shape=(H,W,C,F), got {current.shape}")
    h, w, c, f = current.shape
    # Keep frequency and component axes explicit in the preprocessed file:
    # (H, W, C, F) -> (F, C, W, H).
    return np.transpose(current, (3, 2, 1, 0)).astype(np.float32)


def normalize_y(y):
    y = np.asarray(y).squeeze()
    if y.ndim != 1:
        y = y.reshape(-1)
    return y.astype(np.float32)


def normalize_pattern(p):
    p = np.asarray(p, dtype=np.float32)
    if p.ndim != 3:
        raise ValueError(f"Expected pattern sample shape=(T,P,Fp), got {p.shape}")
    return np.transpose(p, (2, 1, 0)).astype(np.float32)


def _attr_is_true(value):
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def _read_meta_json(f):
    raw = f.attrs.get("meta_json")
    if raw is None:
        return {}
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    if isinstance(raw, np.ndarray):
        raw = raw.item()
    try:
        return json.loads(raw)
    except Exception:
        return {}


def infer_h5_layout(f, input_key="X"):
    """Infer dataset layout without assuming a fixed number of pattern frequencies."""
    input_name = input_key.strip("/")
    if input_name not in f:
        raise KeyError(f"HDF5 must contain /{input_name}")

    x_ds = f[f"/{input_name}"]
    y_ds = f["/Y"]
    p_ds = f["/pattern"]

    if y_ds.ndim != 2:
        raise ValueError(f"/Y expected 2 dimensions, got {y_ds.shape}")
    if p_ds.ndim != 4:
        raise ValueError(f"/pattern expected 4 dimensions, got {p_ds.shape}")

    is_marked_preprocessed = _attr_is_true(f.attrs.get("preprocessed", False))
    looks_preprocessed = (
        x_ds.ndim == 4
        and x_ds.shape[0] == y_ds.shape[0] == p_ds.shape[0]
        and x_ds.shape[1] >= 1
    )
    looks_matlab_raw = x_ds.shape[-1] == y_ds.shape[-1] == p_ds.shape[-1]

    if is_marked_preprocessed or looks_preprocessed:
        return {
            "kind": "standardized" if _attr_is_true(f.attrs.get("standardized", False)) else "preprocessed",
            "num_samples": int(x_ds.shape[0]),
            "x_shape": tuple(int(v) for v in x_ds.shape[1:]),
            "y_shape": tuple(int(v) for v in y_ds.shape[1:]),
            "pattern_shape": tuple(int(v) for v in p_ds.shape[1:]),
            "source_shapes": {
                input_name: tuple(int(v) for v in x_ds.shape),
                "Y": tuple(int(v) for v in y_ds.shape),
                "pattern": tuple(int(v) for v in p_ds.shape),
            },
        }

    if looks_matlab_raw:
        if input_name == "current":
            if x_ds.ndim != 5:
                raise ValueError(f"/current expected 5 dimensions, got {x_ds.shape}")
            h, w, c, f, n = x_ds.shape
            x_shape = (int(f), int(c), int(w), int(h))
        else:
            if x_ds.ndim != 4:
                raise ValueError(f"/{input_name} expected 4 dimensions, got {x_ds.shape}")
            h, w, _c, n = x_ds.shape
            x_shape = (2, int(w), int(h))
        t, p, fp, _ = p_ds.shape
        return {
            "kind": "matlab_raw",
            "num_samples": int(n),
            "x_shape": x_shape,
            "y_shape": (int(y_ds.shape[0]),),
            "pattern_shape": (int(fp), int(p), int(t)),
            "source_shapes": {
                input_name: tuple(int(v) for v in x_ds.shape),
                "Y": tuple(int(v) for v in y_ds.shape),
                "pattern": tuple(int(v) for v in p_ds.shape),
            },
        }

    raise ValueError(
        "Could not infer HDF5 layout. Expected either preprocessed "
        "(N,C,H,W)/(N,F)/(N,Fp,P,T) or MATLAB raw "
        "(H,W,C,N)/(F,N)/(T,P,Fp,N); got "
        f"{input_name}={x_ds.shape}, Y={y_ds.shape}, pattern={p_ds.shape}"
    )


def read_h5_dataset_info(h5_path, input_key="X"):
    with h5py.File(h5_path, "r") as f:
        for key in (f"/{input_key.strip('/')}", "/Y", "/pattern"):
            if key not in f:
                raise KeyError(f"HDF5 must contain {key}")

        layout = infer_h5_layout(f, input_key=input_key)
        meta = _read_meta_json(f)
        pattern_meta = meta.get("pattern", {}) if isinstance(meta, dict) else {}

        return {
            "layout": layout["kind"],
            "num_samples": layout["num_samples"],
            "x_shape": layout["x_shape"],
            "y_shape": layout["y_shape"],
            "pattern_shape": layout["pattern_shape"],
            "source_shapes": layout["source_shapes"],
            "pattern_metadata": pattern_meta,
        }


class H5AntennaDataset(Dataset):
    def __init__(self, h5_path, standardizer=None, feed_sigma=1.5, return_raw=True, input_key="X", flatten_current=True):
        if not os.path.isfile(h5_path):
            raise FileNotFoundError(h5_path)

        self.h5_path = h5_path
        self.standardizer = standardizer
        self.feed_sigma = feed_sigma
        self.return_raw = return_raw
        self.input_key = input_key.strip("/")
        self.flatten_current = flatten_current
        self._h5 = None

        with h5py.File(h5_path, "r") as f:
            if f"/{self.input_key}" not in f or "/Y" not in f or "/pattern" not in f:
                raise KeyError(f"HDF5 must contain /{self.input_key}, /Y, and /pattern")

            layout = infer_h5_layout(f, input_key=self.input_key)
            meta = _read_meta_json(f)
            pattern_meta = meta.get("pattern", {}) if isinstance(meta, dict) else {}

            self.layout = layout["kind"]
            self.is_standardized = self.layout == "standardized"
            self.n = layout["num_samples"]
            self.x_shape = layout["x_shape"]
            self.storage_x_shape = layout["x_shape"]
            # The HDF5 representation retains (F, C, H, W), while the
            # existing surrogate backbone consumes Conv2d tensors (F*C,H,W).
            if self.flatten_current and self.input_key == "current" and len(self.x_shape) == 4:
                f, c, h, w = self.x_shape
                self.x_shape = (f * c, h, w)
            self.y_shape = layout["y_shape"]
            self.pattern_shape = layout["pattern_shape"]
            self.source_shapes = layout["source_shapes"]
            self.pattern_metadata = pattern_meta

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        self._open()

        if self.layout in ("preprocessed", "standardized"):
            x = self._h5[f"/{self.input_key}"][idx]
            y = self._h5["/Y"][idx]
            p = self._h5["/pattern"][idx]
        elif self.layout == "matlab_raw":
            if self.input_key == "current":
                x = normalize_current(self._h5["/current"][:, :, :, :, idx])
            else:
                x = normalize_x(self._h5[f"/{self.input_key}"][:, :, :, idx], sigma=self.feed_sigma)
            y = normalize_y(self._h5["/Y"][:, idx])
            p = normalize_pattern(self._h5["/pattern"][:, :, :, idx])
        else:
            raise RuntimeError(f"Unsupported HDF5 layout: {self.layout}")

        x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        if self.flatten_current and self.input_key == "current" and x.ndim == 4:
            x = x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
        y = torch.from_numpy(np.asarray(y, dtype=np.float32))
        p = torch.from_numpy(np.asarray(p, dtype=np.float32))

        meta = {}

        if self.return_raw:
            if self.layout == "standardized" and "Y_raw" in self._h5 and "pattern_raw" in self._h5:
                y_raw = self._h5["/Y_raw"][idx]
                p_raw = self._h5["/pattern_raw"][idx]
                meta["y_raw"] = torch.from_numpy(np.asarray(y_raw, dtype=np.float32))
                meta["p_raw"] = torch.from_numpy(np.asarray(p_raw, dtype=np.float32))
            else:
                meta["y_raw"] = y.clone()
                meta["p_raw"] = p.clone()

        if self.standardizer is not None:
            if self.layout == "standardized":
                raise ValueError(
                    "Standardized HDF5 already contains normalized tensors; "
                    "do not pass a standardizer to H5AntennaDataset."
                )
            x = self.standardizer.normalize_x(x)
            y = self.standardizer.normalize_y(y)
            p = self.standardizer.normalize_p(p)

        return x, y, p, meta

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._h5 = None

    def __del__(self):
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass
