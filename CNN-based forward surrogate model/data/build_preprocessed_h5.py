# -*- coding: utf-8 -*-
# 从原始 HDF5 构建预处理后的 HDF5
import os

import h5py
import numpy as np

from data.h5_dataset import H5AntennaDataset, read_h5_dataset_info


def get_preprocessed_h5_path(src_h5_path):
    base, ext = os.path.splitext(src_h5_path)
    return f"{base}.preprocessed{ext}"


def _shape_to_attr(shape):
    return np.asarray(shape, dtype=np.int64)


def _expected_preprocessed_shapes(src_h5_path):
    info = read_h5_dataset_info(src_h5_path)
    return {
        "X": (info["num_samples"], *info["x_shape"]),
        "Y": (info["num_samples"], *info["y_shape"]),
        "pattern": (info["num_samples"], *info["pattern_shape"]),
    }


def _preprocessed_h5_matches_source(src_h5_path, dst_h5_path):
    if not os.path.isfile(dst_h5_path):
        return False

    expected = _expected_preprocessed_shapes(src_h5_path)
    try:
        with h5py.File(dst_h5_path, "r") as f_dst:
            if not bool(f_dst.attrs.get("preprocessed", False)):
                return False
            for name, shape in expected.items():
                if name not in f_dst or tuple(f_dst[name].shape) != tuple(shape):
                    return False
    except OSError:
        return False

    return True


def _copy_optional_metadata(f_src, f_dst):
    for name in ("freq_hz", "pattern_freq_hz", "pattern_theta_deg"):
        if name in f_src and name not in f_dst:
            f_src.copy(name, f_dst)

    for key in ("meta_json", "num_samples"):
        if key in f_src.attrs:
            f_dst.attrs[key] = f_src.attrs[key]


def build_preprocessed_h5(src_h5_path, dst_h5_path, feed_sigma=1.5, compression="lzf"):
    if not os.path.isfile(src_h5_path):
        raise FileNotFoundError(src_h5_path)

    os.makedirs(os.path.dirname(dst_h5_path) or ".", exist_ok=True)

    src_info = read_h5_dataset_info(src_h5_path)
    if src_info["layout"] == "preprocessed":
        raise ValueError(f"Source HDF5 is already preprocessed: {src_h5_path}")

    dataset = H5AntennaDataset(src_h5_path, standardizer=None, feed_sigma=feed_sigma, return_raw=False)
    n = len(dataset)
    x0, y0, p0, _ = dataset[0]
    x0 = x0.numpy().astype(np.float32)
    y0 = y0.numpy().astype(np.float32)
    p0 = p0.numpy().astype(np.float32)

    x_chunk = min(128, n)
    y_chunk = min(1024, n)
    p_chunk = min(64, n)

    with h5py.File(src_h5_path, "r") as f_src, h5py.File(dst_h5_path, "w") as f_dst:
        _copy_optional_metadata(f_src, f_dst)

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
        f_dst.attrs["y_layout"] = "N,F"
        f_dst.attrs["pattern_layout"] = "N,Fp,P,T"
        f_dst.attrs["source_layout"] = src_info["layout"]
        f_dst.attrs["source_x_shape"] = _shape_to_attr(src_info["source_shapes"]["X"])
        f_dst.attrs["source_y_shape"] = _shape_to_attr(src_info["source_shapes"]["Y"])
        f_dst.attrs["source_pattern_shape"] = _shape_to_attr(src_info["source_shapes"]["pattern"])
        f_dst.attrs["x_chunk"] = x_chunk
        f_dst.attrs["y_chunk"] = y_chunk
        f_dst.attrs["p_chunk"] = p_chunk

        for i in range(n):
            x, y, p, _ = dataset[i]
            x_dst[i] = x.numpy().astype(np.float32)
            y_dst[i] = y.numpy().astype(np.float32)
            p_dst[i] = p.numpy().astype(np.float32)

            if (i + 1) % 1000 == 0 or (i + 1) == n:
                print(f"[preprocess] [{i + 1}/{n}] done")

    print(f"[preprocess] Saved to: {dst_h5_path}")


def ensure_preprocessed_h5(src_h5_path, feed_sigma=1.5, compression="lzf", force=False):
    src_info = read_h5_dataset_info(src_h5_path)
    if src_info["layout"] == "preprocessed":
        print(f"[preprocess] source already preprocessed: {src_h5_path}")
        return src_h5_path

    dst_h5_path = get_preprocessed_h5_path(src_h5_path)

    need_build = force or (not os.path.isfile(dst_h5_path))

    if (not need_build) and os.path.isfile(dst_h5_path):
        src_mtime = os.path.getmtime(src_h5_path)
        dst_mtime = os.path.getmtime(dst_h5_path)
        if src_mtime > dst_mtime:
            print("[preprocess] source h5 is newer, rebuilding preprocessed h5...")
            need_build = True

    if not need_build and not _preprocessed_h5_matches_source(src_h5_path, dst_h5_path):
        print("[preprocess] cached preprocessed h5 shape does not match source, rebuilding...")
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
