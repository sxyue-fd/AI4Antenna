# -*- coding: utf-8 -*-

import os
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
    if x.ndim != 3 or x.shape[2] != 2:
        raise ValueError(f"期望 x.shape=(H,W,2)，实际得到 {x.shape}")

    structure = x[:, :, 0].astype(np.float32)
    feed_map = x[:, :, 1].astype(np.float32)

    h, w = structure.shape

    feed_indices = np.argwhere(feed_map > 0.5)

    if len(feed_indices) == 0:
        raise ValueError("第二通道未找到馈电点（one-hot 中没有值为1的位置）")

    if len(feed_indices) > 1:
        raise ValueError(f"第二通道检测到多个馈电点，数量={len(feed_indices)}，不符合 one-hot 预期")

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
    return np.transpose(x, (2, 0, 1)).astype(np.float32)


def normalize_y(y):
    y = np.asarray(y).squeeze()
    if y.ndim != 1:
        y = y.reshape(-1)
    return y.astype(np.float32)


def normalize_pattern(p):
    return np.transpose(p, (2, 1, 0)).astype(np.float32)


class H5AntennaDataset(Dataset):
    def __init__(self, h5_path, standardizer=None, feed_sigma=1.5, return_raw=True):
        if not os.path.isfile(h5_path):
            raise FileNotFoundError(h5_path)

        self.h5_path = h5_path
        self.standardizer = standardizer
        self.feed_sigma = feed_sigma
        self.return_raw = return_raw
        self._h5 = None

        with h5py.File(h5_path, "r") as f:
            if "/X" not in f or "/Y" not in f or "/pattern" not in f:
                raise KeyError("HDF5 必须包含 /X, /Y, /pattern")

            x_ds = f["/X"]
            y_ds = f["/Y"]
            p_ds = f["/pattern"]

            if x_ds.ndim != 4:
                raise ValueError(f"/X 期望 shape=(N,C,H,W)，实际 {x_ds.shape}")
            if y_ds.ndim != 2:
                raise ValueError(f"/Y 期望 shape=(N,Dy)，实际 {y_ds.shape}")
            if p_ds.ndim != 4:
                raise ValueError(f"/pattern 期望 shape=(N,11,4,120)，实际 {p_ds.shape}")

            self.n = x_ds.shape[0]
            self.x_shape = tuple(x_ds.shape[1:])
            self.y_shape = tuple(y_ds.shape[1:])
            self.pattern_shape = tuple(p_ds.shape[1:])

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        self._open()

        x = self._h5["/X"][idx]
        y = self._h5["/Y"][idx]
        p = self._h5["/pattern"][idx]

        x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        y = torch.from_numpy(np.asarray(y, dtype=np.float32))
        p = torch.from_numpy(np.asarray(p, dtype=np.float32))

        meta = {}

        if self.return_raw:
            meta["y_raw"] = y.clone()
            meta["p_raw"] = p.clone()

        if self.standardizer is not None:
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