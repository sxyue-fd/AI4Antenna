# -*- coding: utf-8 -*-
# 从原始 HDF5 构建预处理后的 HDF5
import os

import h5py
import numpy as np

from datasets.h5_dataset import H5AntennaDataset, read_h5_dataset_info


def get_preprocessed_h5_path(src_h5_path, input_key="X"):
    base, ext = os.path.splitext(src_h5_path)
    return f"{base}.preprocessed{ext}"


def get_legacy_preprocessed_h5_path(src_h5_path, input_key="X"):
    if input_key == "X":
        return get_preprocessed_h5_path(src_h5_path, input_key=input_key)
    base, ext = os.path.splitext(src_h5_path)
    return f"{base}.{input_key}.preprocessed{ext}"


def _shape_to_attr(shape):
    return np.asarray(shape, dtype=np.int64)


def _expected_preprocessed_shapes(src_h5_path, input_key="X"):
    info = read_h5_dataset_info(src_h5_path, input_key=input_key)
    return {
        input_key: (info["num_samples"], *info["x_shape"]),
        "Y": (info["num_samples"], *info["y_shape"]),
        "pattern": (info["num_samples"], *info["pattern_shape"]),
    }


def get_input_chunk_size(input_key, num_samples):
    if input_key == "current":
        return min(8, num_samples)
    return min(128, num_samples)


def _expected_input_chunks(src_h5_path, input_key="X"):
    info = read_h5_dataset_info(src_h5_path, input_key=input_key)
    return (get_input_chunk_size(input_key, info["num_samples"]), *info["x_shape"])


def _preprocessed_h5_matches_source(src_h5_path, dst_h5_path, input_key="X"):
    if not os.path.isfile(dst_h5_path):
        return False

    expected = _expected_preprocessed_shapes(src_h5_path, input_key=input_key)
    try:
        with h5py.File(dst_h5_path, "r") as f_dst:
            if not bool(f_dst.attrs.get("preprocessed", False)):
                return False
            for name, shape in expected.items():
                if name not in f_dst or tuple(f_dst[name].shape) != tuple(shape):
                    return False
            if tuple(f_dst[input_key].chunks or ()) != tuple(_expected_input_chunks(src_h5_path, input_key=input_key)):
                return False
    except OSError:
        return False

    return True


def _delete_if_exists(f, name):
    if name in f:
        del f[name]


def _dataset_shape_matches(f, name, shape):
    return name in f and tuple(f[name].shape) == tuple(shape)


def _read_preprocessed_inputs(f):
    inputs = set()
    if "preprocessed_inputs" in f.attrs:
        raw_inputs = f.attrs["preprocessed_inputs"]
        if isinstance(raw_inputs, bytes):
            raw_inputs = raw_inputs.decode("utf-8", errors="ignore")
        inputs.update(str(raw_inputs).split(","))
    for name in ("X", "current"):
        if name in f:
            inputs.add(name)
    return inputs


def _write_preprocessed_inputs(f, inputs):
    f.attrs["preprocessed_inputs"] = ",".join(sorted(v for v in inputs if v))


def _copy_optional_metadata(f_src, f_dst):
    for name in ("freq_hz", "pattern_freq_hz", "pattern_theta_deg", "current_freq_hz"):
        if name in f_src and name not in f_dst:
            f_src.copy(name, f_dst)

    for key in ("meta_json", "num_samples"):
        if key in f_src.attrs:
            f_dst.attrs[key] = f_src.attrs[key]


def build_preprocessed_h5(
    src_h5_path,
    dst_h5_path,
    feed_sigma=1.5,
    compression="lzf",
    input_key="X",
    refresh_targets=False,
    force_input=False,
):
    if not os.path.isfile(src_h5_path):
        raise FileNotFoundError(src_h5_path)

    os.makedirs(os.path.dirname(dst_h5_path) or ".", exist_ok=True)

    src_info = read_h5_dataset_info(src_h5_path, input_key=input_key)
    if src_info["layout"] == "preprocessed":
        raise ValueError(f"Source HDF5 is already preprocessed: {src_h5_path}")

    dataset = H5AntennaDataset(
        src_h5_path,
        standardizer=None,
        feed_sigma=feed_sigma,
        return_raw=False,
        input_key=input_key,
        # Store current as (F, C, H, W); flattening is only a model-loader view.
        flatten_current=False,
    )
    n = len(dataset)
    x0, y0, p0, _ = dataset[0]
    x0 = x0.numpy().astype(np.float32)
    y0 = y0.numpy().astype(np.float32)
    p0 = p0.numpy().astype(np.float32)

    x_chunk = get_input_chunk_size(input_key, n)
    y_chunk = min(1024, n)
    p_chunk = min(64, n)

    expected_shapes = {
        input_key: (n, *x0.shape),
        "Y": (n, *y0.shape),
        "pattern": (n, *p0.shape),
    }

    with h5py.File(src_h5_path, "r") as f_src, h5py.File(dst_h5_path, "a") as f_dst:
        _copy_optional_metadata(f_src, f_dst)

        expected_x_chunks = (x_chunk, *x0.shape)
        rewrite_x = (
            force_input
            or not _dataset_shape_matches(f_dst, input_key, expected_shapes[input_key])
            or tuple(f_dst[input_key].chunks or ()) != expected_x_chunks
        )
        rewrite_y = refresh_targets or not _dataset_shape_matches(f_dst, "Y", expected_shapes["Y"])
        rewrite_p = refresh_targets or not _dataset_shape_matches(f_dst, "pattern", expected_shapes["pattern"])

        x_dst = None
        y_dst = None
        p_dst = None

        if rewrite_x:
            _delete_if_exists(f_dst, input_key)
            x_dst = f_dst.create_dataset(
                input_key,
                shape=expected_shapes[input_key],
                dtype=np.float32,
                compression=compression,
                chunks=expected_x_chunks,
            )

        if rewrite_y:
            _delete_if_exists(f_dst, "Y")
            y_dst = f_dst.create_dataset(
                "Y",
                shape=expected_shapes["Y"],
                dtype=np.float32,
                compression=compression,
                chunks=(y_chunk, *y0.shape),
            )

        if rewrite_p:
            _delete_if_exists(f_dst, "pattern")
            p_dst = f_dst.create_dataset(
                "pattern",
                shape=expected_shapes["pattern"],
                dtype=np.float32,
                compression=compression,
                chunks=(p_chunk, *p0.shape),
            )

        f_dst.attrs["preprocessed"] = True
        preprocessed_inputs = _read_preprocessed_inputs(f_dst)
        preprocessed_inputs.add(input_key)
        _write_preprocessed_inputs(f_dst, preprocessed_inputs)
        f_dst.attrs["feed_sigma"] = float(feed_sigma)
        f_dst.attrs["x_layout"] = "NCHW"
        f_dst.attrs["y_layout"] = "N,F"
        f_dst.attrs["pattern_layout"] = "N,Fp,P,T"
        f_dst.attrs["source_layout"] = src_info["layout"]
        f_dst.attrs[f"source_{input_key}_shape"] = _shape_to_attr(src_info["source_shapes"][input_key])
        f_dst.attrs["source_y_shape"] = _shape_to_attr(src_info["source_shapes"]["Y"])
        f_dst.attrs["source_pattern_shape"] = _shape_to_attr(src_info["source_shapes"]["pattern"])
        f_dst.attrs[f"{input_key}_chunk"] = x_chunk
        f_dst.attrs["y_chunk"] = y_chunk
        f_dst.attrs["p_chunk"] = p_chunk

        for i in range(n):
            x, y, p, _ = dataset[i]
            if x_dst is not None:
                x_dst[i] = x.numpy().astype(np.float32)
            if y_dst is not None:
                y_dst[i] = y.numpy().astype(np.float32)
            if p_dst is not None:
                p_dst[i] = p.numpy().astype(np.float32)

            if (i + 1) % 1000 == 0 or (i + 1) == n:
                print(f"[preprocess] [{i + 1}/{n}] done")

    print(f"[preprocess] Saved {input_key} to: {dst_h5_path}")


def _copy_legacy_input_if_available(src_h5_path, dst_h5_path, input_key="X"):
    legacy_path = get_legacy_preprocessed_h5_path(src_h5_path, input_key=input_key)
    if legacy_path == dst_h5_path or not os.path.isfile(legacy_path):
        return False

    if not _preprocessed_h5_matches_source(src_h5_path, legacy_path, input_key=input_key):
        return False

    with h5py.File(legacy_path, "r") as f_legacy, h5py.File(dst_h5_path, "a") as f_dst:
        if input_key in f_dst:
            return False
        f_legacy.copy(input_key, f_dst)
        for name in ("current_freq_hz", "freq_hz", "pattern_freq_hz", "pattern_theta_deg"):
            if name in f_legacy and name not in f_dst:
                f_legacy.copy(name, f_dst)
        f_dst.attrs["preprocessed"] = True
        preprocessed_inputs = _read_preprocessed_inputs(f_dst)
        preprocessed_inputs.add(input_key)
        _write_preprocessed_inputs(f_dst, preprocessed_inputs)

    print(f"[preprocess] merged legacy {input_key} cache into: {dst_h5_path}")
    return True


def _normalize_input_keys(input_key="X", input_keys=None):
    if input_keys is None:
        keys = [input_key]
    elif isinstance(input_keys, str):
        if input_keys.lower() == "all":
            keys = ["X", "current"]
        else:
            keys = [part.strip() for part in input_keys.split(",")]
    else:
        keys = list(input_keys)

    if input_key not in keys:
        keys.append(input_key)

    seen = set()
    unique_keys = []
    for key in keys:
        key = str(key).strip("/")
        if key and key not in seen:
            seen.add(key)
            unique_keys.append(key)
    return unique_keys


def ensure_preprocessed_h5(
    src_h5_path,
    feed_sigma=1.5,
    compression="lzf",
    force=False,
    input_key="X",
    input_keys=None,
):
    requested_keys = _normalize_input_keys(input_key=input_key, input_keys=input_keys)

    src_info = read_h5_dataset_info(src_h5_path, input_key=input_key)
    if src_info["layout"] == "preprocessed":
        print(f"[preprocess] source already preprocessed: {src_h5_path}")
        return src_h5_path

    dst_h5_path = get_preprocessed_h5_path(src_h5_path, input_key=input_key)

    for key in requested_keys:
        read_h5_dataset_info(src_h5_path, input_key=key)

        need_build = force or (not os.path.isfile(dst_h5_path))
        refresh_targets = force

        if (not need_build) and os.path.isfile(dst_h5_path):
            src_mtime = os.path.getmtime(src_h5_path)
            dst_mtime = os.path.getmtime(dst_h5_path)
            if src_mtime > dst_mtime:
                print("[preprocess] source h5 is newer, rebuilding preprocessed h5...")
                need_build = True
                refresh_targets = True

        if not need_build:
            _copy_legacy_input_if_available(src_h5_path, dst_h5_path, input_key=key)

        if not need_build and not _preprocessed_h5_matches_source(src_h5_path, dst_h5_path, input_key=key):
            print(f"[preprocess] cached preprocessed h5 missing/mismatched {key}, rebuilding...")
            need_build = True

        if need_build:
            print(f"[preprocess] building preprocessed h5 input: {key}")
            build_preprocessed_h5(
                src_h5_path=src_h5_path,
                dst_h5_path=dst_h5_path,
                feed_sigma=feed_sigma,
                compression=compression,
                input_key=key,
                refresh_targets=refresh_targets,
                force_input=force,
            )
        else:
            print(f"[preprocess] reuse existing {key} in: {dst_h5_path}")

    return dst_h5_path
