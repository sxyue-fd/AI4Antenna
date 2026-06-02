# -*- coding: utf-8 -*-
# 从原始 HDF5 构建预处理后的 HDF5
import os

import h5py
import numpy as np

from data.h5_dataset import normalize_x, normalize_y, normalize_pattern


def get_preprocessed_h5_path(src_h5_path):
    base, ext = os.path.splitext(src_h5_path)
    return f"{base}.preprocessed{ext}"


def build_preprocessed_h5(src_h5_path, dst_h5_path, feed_sigma=1.5, compression="lzf"):
    if not os.path.isfile(src_h5_path):
        raise FileNotFoundError(src_h5_path)

    os.makedirs(os.path.dirname(dst_h5_path) or ".", exist_ok=True)

    with h5py.File(src_h5_path, "r") as f_src:
        if "/X" not in f_src or "/Y" not in f_src or "/pattern" not in f_src:
            raise KeyError("源 HDF5 必须包含 /X, /Y, /pattern")

        x_src = f_src["/X"]           # (H, W, 2, N)
        y_src = f_src["/Y"]           # (41, N)
        p_src = f_src["/pattern"]     # (120, 4, 11, N)

        n = x_src.shape[-1]

        x0 = normalize_x(x_src[:, :, :, 0], sigma=feed_sigma)   # (2,16,16)
        y0 = normalize_y(y_src[:, 0])                           # (41,)
        p0 = normalize_pattern(p_src[:, :, :, 0])               # (11,4,120)

        x_chunk = min(128, n)
        y_chunk = min(1024, n)
        p_chunk = min(64, n)

        with h5py.File(dst_h5_path, "w") as f_dst:
            x_dst = f_dst.create_dataset(
                "X",
                shape=(n, *x0.shape),
                dtype=np.float32,
                compression=compression,
                chunks=(x_chunk, *x0.shape),
            )
            y_dst = f_dst.create_dataset(
                "Y",
                shape=(n, *y0.shape),
                dtype=np.float32,
                compression=compression,
                chunks=(y_chunk, *y0.shape),
            )
            p_dst = f_dst.create_dataset(
                "pattern",
                shape=(n, *p0.shape),
                dtype=np.float32,
                compression=compression,
                chunks=(p_chunk, *p0.shape),
            )

            f_dst.attrs["preprocessed"] = True
            f_dst.attrs["feed_sigma"] = float(feed_sigma)
            f_dst.attrs["x_layout"] = "NCHW"
            f_dst.attrs["y_layout"] = "N,D"
            f_dst.attrs["pattern_layout"] = "N,11,4,120"
            f_dst.attrs["x_chunk"] = x_chunk
            f_dst.attrs["y_chunk"] = y_chunk
            f_dst.attrs["p_chunk"] = p_chunk

            for i in range(n):
                x = normalize_x(x_src[:, :, :, i], sigma=feed_sigma)
                y = normalize_y(y_src[:, i])
                p = normalize_pattern(p_src[:, :, :, i])

                x_dst[i] = x
                y_dst[i] = y
                p_dst[i] = p

                if (i + 1) % 1000 == 0 or (i + 1) == n:
                    print(f"[preprocess] [{i + 1}/{n}] done")

    print(f"[preprocess] Saved to: {dst_h5_path}")


def ensure_preprocessed_h5(src_h5_path, feed_sigma=1.5, compression="lzf", force=False):
    dst_h5_path = get_preprocessed_h5_path(src_h5_path)

    need_build = force or (not os.path.isfile(dst_h5_path))

    if (not need_build) and os.path.isfile(dst_h5_path):
        src_mtime = os.path.getmtime(src_h5_path)
        dst_mtime = os.path.getmtime(dst_h5_path)
        if src_mtime > dst_mtime:
            print("[preprocess] source h5 is newer, rebuilding preprocessed h5...")
            need_build = True

    if need_build:
        print("[preprocess] building preprocessed h5...")
        build_preprocessed_h5(
            src_h5_path=src_h5_path,
            dst_h5_path=dst_h5_path,
            feed_sigma=feed_sigma,
            compression=compression,
        )
    else:
        print(f"[preprocess] reuse existing file: {dst_h5_path}")

    return dst_h5_path