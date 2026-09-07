"""Inference, CNN-surrogate ranking, and MATLAB orchestration for AntGen."""

from __future__ import annotations

import gc
import json
import random
import subprocess
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from scipy.io import savemat

from datasets.datamodule import load_standardizer_for_dataset
from datasets.h5_dataset import H5AntennaDataset
from datasets.preprocess import TargetStandardizer
from datasets.split_loader import build_split_indices

from .contracts import (
    PATTERN_CHANNEL_ORDER,
    PATTERN_PLANES_PHI_DEG,
    PATTERN_VALUE_TYPE,
    SCHEMA_VERSION,
    SPATIAL_COORDINATE_CONVENTION,
)
from .layout import run_layout
from .metrics import rank_surrogate_candidates
from .model_loader import (
    latest_checkpoint,
    load_cnn_surrogate_model,
    load_diffusion_model,
    load_structure_model,
)
from .visualization import visualize_surrogate_stage


DEFAULT_GEOMETRY = {
    "patch_L_mm": 14.0,
    "patch_W_mm": 14.0,
    "sub_thick_mm": 2.5,
    "substrate_name": "Air",
    "board_L_mm": 30.0,
    "board_W_mm": 30.0,
    "feed_diam_mm": 0.0875,
    "overlap_m": 5e-6,
}

def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_dir(output_root: Path) -> Path:
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{run_id}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def _read_json_attr(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def read_simulation_metadata(h5_path: Path, y_size: int, pattern_shape: tuple[int, ...]):
    fp, channels, theta_count = pattern_shape
    if channels != 4:
        raise ValueError(f"MATLAB full-wave adapter expects 4 pattern channels, got {channels}")
    with h5py.File(h5_path, "r") as handle:
        def axis(name: str, fallback: np.ndarray) -> np.ndarray:
            return (
                np.asarray(handle[name], dtype=np.float64).reshape(-1)
                if name in handle
                else fallback
            )

        freq_hz = axis("freq_hz", np.linspace(8e9, 12e9, y_size))
        pattern_freq_hz = axis("pattern_freq_hz", np.linspace(8e9, 12e9, fp))
        theta_deg = axis("pattern_theta_deg", np.arange(1.0, 361.0, 3.0))
        metadata = _read_json_attr(handle.attrs.get("meta_json"))
        if "feed_sigma" in handle.attrs:
            feed_sigma = float(handle.attrs["feed_sigma"])
        else:
            feed_sigma = 2.0
            warnings.warn("HDF5 has no feed_sigma attribute; using training fallback sigma=2.0")

    if freq_hz.size != y_size:
        raise ValueError(f"freq_hz has {freq_hz.size} points, expected {y_size}")
    if pattern_freq_hz.size != fp:
        raise ValueError(f"pattern_freq_hz has {pattern_freq_hz.size} points, expected {fp}")
    if theta_deg.size != theta_count:
        raise ValueError(f"pattern_theta_deg has {theta_deg.size} points, expected {theta_count}")

    geometry = dict(DEFAULT_GEOMETRY)
    params = metadata.get("params", {}) if isinstance(metadata, dict) else {}
    stored_geometry = params.get("geom", {}) if isinstance(params, dict) else {}
    if isinstance(stored_geometry, dict):
        for key in geometry:
            if key in stored_geometry:
                geometry[key] = stored_geometry[key]
    if isinstance(params, dict) and "overlap_m" in params:
        geometry["overlap_m"] = params["overlap_m"]
    return freq_hz, pattern_freq_hz, theta_deg, geometry, feed_sigma


def _select_test_indices(
    num_samples: int,
    split_cfg: dict[str, Any],
    num_conditions: int,
    test_start: int,
    selection: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    split = build_split_indices(
        num_samples=num_samples,
        train_ratio=split_cfg.get("train_ratio", 0.8),
        val_ratio=split_cfg.get("val_ratio", 0.1),
        test_ratio=split_cfg.get("test_ratio", 0.1),
        seed=split_cfg.get("seed", seed),
    )
    test_indices = np.asarray(split["test"], dtype=np.int64)
    if num_conditions > test_indices.size:
        raise ValueError(
            f"Requested {num_conditions} conditions, but the test split has only {test_indices.size}"
        )
    if selection == "sequential":
        if test_start < 0 or test_start + num_conditions > test_indices.size:
            raise ValueError(
                f"test_start={test_start} with m={num_conditions} is outside a test split "
                f"of size {test_indices.size}"
            )
        offsets = np.arange(test_start, test_start + num_conditions, dtype=np.int64)
    elif selection == "random":
        offsets = np.random.default_rng(seed).choice(
            test_indices.size, size=num_conditions, replace=False
        ).astype(np.int64)
    else:
        raise ValueError(f"Unknown test selection: {selection}")
    return offsets, test_indices[offsets]


def _collect_conditions(dataset: H5AntennaDataset, dataset_indices: np.ndarray):
    y_norm, pattern_norm, y_raw, pattern_raw = [], [], [], []
    for index in dataset_indices:
        _, y, pattern, meta = dataset[int(index)]
        if "y_raw" not in meta or "p_raw" not in meta:
            raise ValueError("The evaluation dataset must expose raw Y and pattern targets")
        y_norm.append(y)
        pattern_norm.append(pattern)
        y_raw.append(meta["y_raw"])
        pattern_raw.append(meta["p_raw"])
    return tuple(torch.stack(items, dim=0) for items in (y_norm, pattern_norm, y_raw, pattern_raw))


def keep_feed_connected_component(metal: np.ndarray, feed_row: int, feed_col: int) -> np.ndarray:
    """Keep only the four-connected metal component containing the feed."""
    metal = np.asarray(metal, dtype=bool).copy()
    height, width = metal.shape
    if not (0 <= feed_row < height and 0 <= feed_col < width):
        raise ValueError(f"Feed {(feed_row, feed_col)} is outside topology shape {metal.shape}")
    metal[feed_row, feed_col] = True
    keep = np.zeros_like(metal)
    keep[feed_row, feed_col] = True
    stack = [(feed_row, feed_col)]
    while stack:
        row, col = stack.pop()
        for next_row, next_col in (
            (row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)
        ):
            if (
                0 <= next_row < height
                and 0 <= next_col < width
                and metal[next_row, next_col]
                and not keep[next_row, next_col]
            ):
                keep[next_row, next_col] = True
                stack.append((next_row, next_col))
    return keep


def _decode_structures(
    metal_prob: np.ndarray,
    feed_logits: np.ndarray,
    threshold: float,
    topology_postprocess: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Threshold metal, decode feed, force feed metal, then apply connectivity filtering."""
    metal = metal_prob >= threshold
    flat_feed = np.argmax(feed_logits.reshape(feed_logits.shape[0], -1), axis=1)
    height, width = metal.shape[-2:]
    feed_python = np.stack((flat_feed // width, flat_feed % width), axis=1).astype(np.int64)
    for index, (row, col) in enumerate(feed_python):
        metal[index, row, col] = True
        if topology_postprocess == "feed-component":
            metal[index] = keep_feed_connected_component(metal[index], int(row), int(col))
        elif topology_postprocess != "none":
            raise ValueError(f"Unknown topology postprocess: {topology_postprocess}")
    return metal, feed_python


def convert_topology_to_matlab(
    metal_python: np.ndarray, feed_python: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Preserve physical row/col axes and convert the feed from zero- to one-based."""
    metal_python = np.asarray(metal_python)
    feed_python = np.asarray(feed_python)
    if metal_python.shape[:-2] != feed_python.shape[:-1] or feed_python.shape[-1] != 2:
        raise ValueError("Topology and feed leading dimensions do not match")
    # datasets.normalize_x() has already restored MATLAB's physical (row, col)
    # spatial semantics.  A second transpose here rotates the antenna across
    # x=y and swaps the XOZ/YOZ radiation-pattern planes.
    metal_matlab = metal_python.astype(np.uint8)
    feed_matlab = (feed_python + 1).astype(np.int32)
    return metal_matlab, feed_matlab


def _validate_current_models(diffusion_checkpoint: dict[str, Any], structure_checkpoint: dict[str, Any]):
    diffusion_info = diffusion_checkpoint["dataset_info"]
    structure_info = structure_checkpoint["dataset_info"]
    current_shape = tuple(int(v) for v in diffusion_info["current_shape"])
    structure_input_shape = tuple(int(v) for v in structure_info["input_shape"])
    if current_shape != structure_input_shape:
        raise ValueError(
            f"Model mismatch: diffusion current={current_shape}, structure input={structure_input_shape}"
        )
    for key in ("y_shape", "pattern_shape"):
        diffusion_shape = tuple(int(v) for v in diffusion_info[key])
        structure_shape = tuple(int(v) for v in structure_info[key])
        if diffusion_shape != structure_shape:
            raise ValueError(
                f"Model mismatch: diffusion {key}={diffusion_shape}, "
                f"structure {key}={structure_shape}"
            )


def _validate_surrogate(
    checkpoint: dict[str, Any],
    topology_shape: tuple[int, int],
    y_shape: tuple[int, ...],
    pattern_shape: tuple[int, ...],
    pattern_freq_hz: np.ndarray,
    theta_deg: np.ndarray,
) -> None:
    info = checkpoint["dataset_info"]
    expected = {
        "x_shape": (2, *topology_shape),
        "y_shape": y_shape,
        "pattern_shape": pattern_shape,
    }
    for key, shape in expected.items():
        actual = tuple(int(v) for v in info[key])
        if actual != tuple(shape):
            raise ValueError(f"CNN surrogate {key}={actual}, expected {tuple(shape)}")
    pattern_meta = info.get("pattern_metadata", {})
    if pattern_meta.get("freq_hz") is not None and not np.allclose(
        np.asarray(pattern_meta["freq_hz"]), pattern_freq_hz
    ):
        raise ValueError("CNN surrogate pattern frequency axis differs from the selected dataset")
    if pattern_meta.get("theta_deg") is not None and not np.allclose(
        np.asarray(pattern_meta["theta_deg"]), theta_deg
    ):
        raise ValueError("CNN surrogate theta axis differs from the selected dataset")


def _matlab_geometry(geometry: dict[str, Any]) -> dict[str, Any]:
    return {
        "patch_L_mm": float(geometry["patch_L_mm"]),
        "patch_W_mm": float(geometry["patch_W_mm"]),
        "sub_thick_mm": float(geometry["sub_thick_mm"]),
        "substrate_name": str(geometry["substrate_name"]),
        "board_L_mm": float(geometry["board_L_mm"]),
        "board_W_mm": float(geometry["board_W_mm"]),
        "feed_diam_mm": float(geometry["feed_diam_mm"]),
        "overlap_m": float(geometry["overlap_m"]),
    }


def denormalize_generated_current(standardizer, current: np.ndarray, stored_shape: tuple[int, ...]) -> np.ndarray:
    """Invert the dataset's signed-log current transform for saved artifacts."""
    stats = standardizer.input_stats.get("current")
    if not stats or stats.get("transform") != "signed_log_clip_zscore":
        return np.asarray(current, dtype=np.float32)
    if len(stored_shape) != 4:
        raise ValueError(f"Expected stored current shape (F,C,H,W), got {stored_shape}")
    frequency_count, component_count, height, width = (int(v) for v in stored_shape)
    current = np.asarray(current, dtype=np.float32)
    if current.shape[-3:] != (frequency_count * component_count, height, width):
        raise ValueError(
            f"Generated current shape {current.shape} is incompatible with stored shape {stored_shape}"
        )
    explicit = current.reshape(*current.shape[:-3], frequency_count, component_count, height, width)
    prefix_dims = explicit.ndim - 4
    stats_shape = (1,) * prefix_dims + (frequency_count, component_count, 1, 1)
    alpha = np.asarray(stats["alpha"], dtype=np.float32).reshape(stats_shape)
    mean = np.asarray(stats["mean"], dtype=np.float32).reshape(stats_shape)
    std = np.asarray(stats["std"], dtype=np.float32).reshape(stats_shape)
    transformed = np.clip(explicit * std + mean, -float(stats["clip"]), float(stats["clip"]))
    return (np.sign(transformed) * alpha * np.expm1(np.abs(transformed))).astype(np.float32)


def build_cnn_surrogate_inputs(
    metal_python: np.ndarray,
    feed_python: np.ndarray,
    feed_sigma: float,
    standardizer: TargetStandardizer,
) -> torch.Tensor:
    """Encode binary topology + Gaussian feed exactly as the CNN training dataset."""
    metal = torch.as_tensor(np.asarray(metal_python), dtype=torch.float32)
    feed = torch.as_tensor(np.asarray(feed_python), dtype=torch.float32)
    if metal.ndim != 3 or feed.shape != (metal.shape[0], 2):
        raise ValueError(f"Expected metal=(B,H,W), feed=(B,2); got {metal.shape}, {feed.shape}")
    if feed_sigma <= 0:
        raise ValueError("feed_sigma must be positive")
    height, width = metal.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    distance2 = (
        (yy[None] - feed[:, 0, None, None]) ** 2
        + (xx[None] - feed[:, 1, None, None]) ** 2
    )
    gaussian = torch.exp(-distance2 / (2.0 * float(feed_sigma) ** 2))
    model_input = torch.stack((metal, gaussian), dim=1)
    return standardizer.normalize_input("X", model_input)


@torch.inference_mode()
def _predict_surrogate(
    model,
    standardizer: TargetStandardizer,
    model_input: torch.Tensor,
    device: torch.device,
    batch_size: int,
    m: int,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    y_batches, pattern_batches = [], []
    for start in range(0, model_input.shape[0], batch_size):
        end = min(start + batch_size, model_input.shape[0])
        y_norm, pattern_norm = model(model_input[start:end].to(device, non_blocking=True))
        y_batches.append(standardizer.denormalize_y(y_norm).cpu())
        pattern_batches.append(standardizer.denormalize_p(pattern_norm).cpu())
    y = torch.cat(y_batches).numpy().astype(np.float32)
    pattern = torch.cat(pattern_batches).numpy().astype(np.float32)
    return y.reshape(m, n, *y.shape[1:]), pattern.reshape(m, n, *pattern.shape[1:])


def _release_models(*objects) -> None:
    del objects
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_candidates(args, workspace: Path, run_dir: Path) -> dict[str, Any]:
    """Generate m*n designs, rank by CNN proxy, and prepare only m MATLAB jobs."""
    layout = run_layout(run_dir, create=True)
    device = resolve_device(args.device)
    set_seed(args.seed)

    diffusion_checkpoint_path = (
        Path(args.diffusion_checkpoint).resolve()
        if args.diffusion_checkpoint
        else latest_checkpoint(workspace / "Current diffusion model" / "outputs", "diffusion")
    )
    structure_checkpoint_path = (
        Path(args.structure_checkpoint).resolve()
        if args.structure_checkpoint
        else latest_checkpoint(workspace / "Current to structure UNet" / "outputs", "structure")
    )
    surrogate_checkpoint_path = (
        Path(args.cnn_surrogate_checkpoint).resolve()
        if args.cnn_surrogate_checkpoint
        else latest_checkpoint(
            workspace / "CNN-based forward surrogate model" / "outputs", "cnn_surrogate"
        )
    )
    print(f"[AntGen] device: {device}")
    print(f"[AntGen] diffusion checkpoint: {diffusion_checkpoint_path}")
    print(f"[AntGen] structure checkpoint: {structure_checkpoint_path}")

    diffusion_model, diffusion_checkpoint = load_diffusion_model(
        workspace, diffusion_checkpoint_path, device
    )
    structure_model, structure_checkpoint = load_structure_model(
        workspace, structure_checkpoint_path, device
    )
    _validate_current_models(diffusion_checkpoint, structure_checkpoint)
    diffusion_cfg = diffusion_checkpoint["config"]
    diffusion_info = diffusion_checkpoint["dataset_info"]
    structure_info = structure_checkpoint["dataset_info"]

    h5_path = Path(args.h5_path or diffusion_cfg["paths"]["h5_path"]).resolve()
    if not h5_path.is_file():
        raise FileNotFoundError(h5_path)
    dataset = H5AntennaDataset(
        str(h5_path), standardizer=None, return_raw=True, input_key="current", flatten_current=True
    )
    if dataset.layout != "standardized":
        raise ValueError(f"AntGen requires standardized HDF5; got layout={dataset.layout}")
    for key, actual in (
        ("current_shape", dataset.x_shape),
        ("y_shape", dataset.y_shape),
        ("pattern_shape", dataset.pattern_shape),
    ):
        expected = tuple(int(v) for v in diffusion_info[key])
        if tuple(actual) != expected:
            raise ValueError(
                f"Diffusion checkpoint {key}={expected}, selected dataset has {tuple(actual)}"
            )
    standardizer, standardizer_path = load_standardizer_for_dataset(str(h5_path), diffusion_cfg)
    standardizer.validate_shapes(dataset.y_shape, dataset.pattern_shape)
    test_offsets, dataset_indices = _select_test_indices(
        len(dataset), diffusion_cfg.get("split", {}), args.num_conditions,
        args.test_start, args.selection, args.seed,
    )
    y_norm, pattern_norm, y_raw, pattern_raw = _collect_conditions(dataset, dataset_indices)
    freq_hz, pattern_freq_hz, theta_deg, geometry, feed_sigma = read_simulation_metadata(
        h5_path, dataset.y_shape[0], tuple(dataset.pattern_shape)
    )

    m, n = args.num_conditions, args.num_candidates
    total_candidates = m * n
    repeated_y = y_norm.repeat_interleave(n, dim=0)
    repeated_pattern = pattern_norm.repeat_interleave(n, dim=0)
    generated_batches, probability_batches, feed_logit_batches = [], [], []
    sample_cfg = diffusion_cfg.get("sample", {})
    s11_cfg_scale = args.s11_cfg_scale
    if s11_cfg_scale is None:
        s11_cfg_scale = sample_cfg.get("s11_cfg_scale")
    pattern_cfg_scale = args.pattern_cfg_scale
    if pattern_cfg_scale is None:
        pattern_cfg_scale = sample_cfg.get("pattern_cfg_scale")

    with torch.inference_mode():
        for start in range(0, total_candidates, args.generation_batch_size):
            end = min(start + args.generation_batch_size, total_candidates)
            print(f"[AntGen] generating candidates {start + 1}-{end}/{total_candidates}")
            kwargs: dict[str, Any] = {
                "s11": repeated_y[start:end].to(device),
                "pattern": repeated_pattern[start:end].to(device),
            }
            if s11_cfg_scale is not None:
                kwargs["s11_cfg_scale"] = float(s11_cfg_scale)
            if pattern_cfg_scale is not None:
                kwargs["pattern_cfg_scale"] = float(pattern_cfg_scale)
            generated = diffusion_model.sample(
                batch_size=end - start, model_kwargs=kwargs, progress=args.diffusion_progress
            )
            outputs = structure_model(generated)
            generated_batches.append(generated.cpu())
            probability_batches.append(torch.sigmoid(outputs["metal_logits"]).cpu())
            feed_logit_batches.append(outputs["feed_logits"].cpu())

    generated_current_flat = torch.cat(generated_batches).numpy().astype(np.float32)
    metal_probability_flat = torch.cat(probability_batches).numpy()[:, 0].astype(np.float32)
    feed_logits_flat = torch.cat(feed_logit_batches).numpy()[:, 0].astype(np.float32)
    metal_flat, feed_flat = _decode_structures(
        metal_probability_flat, feed_logits_flat, args.metal_threshold, args.topology_postprocess
    )
    current_shape = generated_current_flat.shape[1:]
    topology_shape = tuple(int(v) for v in metal_flat.shape[1:])
    expected_topology_shape = tuple(int(v) for v in structure_info["target_shape"][-2:])
    if topology_shape != expected_topology_shape:
        raise ValueError(
            f"Structure model produced topology {topology_shape}, expected {expected_topology_shape}"
        )
    generated_current = generated_current_flat.reshape(m, n, *current_shape)
    generated_current_physical = denormalize_generated_current(
        standardizer, generated_current, tuple(int(v) for v in dataset.storage_x_shape)
    )
    metal_probability = metal_probability_flat.reshape(m, n, *topology_shape)
    metal_python = metal_flat.reshape(m, n, *topology_shape)
    feed_python = feed_flat.reshape(m, n, 2)
    metal_matlab, feed_matlab = convert_topology_to_matlab(metal_python, feed_python)

    np.savez_compressed(
        layout.candidates,
        generated_current_standardized=generated_current,
        generated_current_physical_python=generated_current_physical,
        metal_probability_python=metal_probability,
        metal_binary_python=metal_python.astype(np.uint8),
        feed_rc_python_0based=feed_python,
        metal_binary_matlab=metal_matlab,
        feed_rc_matlab_1based=feed_matlab,
        target_s11=y_raw.numpy().astype(np.float32),
        target_pattern=pattern_raw.numpy().astype(np.float32),
        test_offsets=test_offsets,
        dataset_indices=dataset_indices,
        freq_hz=freq_hz,
        pattern_freq_hz=pattern_freq_hz,
        pattern_theta_deg=theta_deg,
        feed_sigma=np.asarray(feed_sigma, dtype=np.float32),
    )

    # Free the two generative models before loading the large CNN surrogate.
    del diffusion_model, structure_model, diffusion_checkpoint, structure_checkpoint
    _release_models()
    print(f"[AntGen] CNN surrogate checkpoint: {surrogate_checkpoint_path}")
    surrogate_model, surrogate_checkpoint = load_cnn_surrogate_model(
        workspace, surrogate_checkpoint_path, device
    )
    _validate_surrogate(
        surrogate_checkpoint, topology_shape, tuple(dataset.y_shape), tuple(dataset.pattern_shape),
        pattern_freq_hz, theta_deg,
    )
    surrogate_standardizer = TargetStandardizer.from_state_dict(
        surrogate_checkpoint["standardizer_stats"]
    )
    surrogate_input = build_cnn_surrogate_inputs(
        metal_flat, feed_flat, feed_sigma, surrogate_standardizer
    )
    surrogate_s11, surrogate_pattern = _predict_surrogate(
        surrogate_model, surrogate_standardizer, surrogate_input, device,
        args.surrogate_batch_size, m, n,
    )
    np.savez_compressed(
        layout.surrogate_predictions,
        surrogate_s11=surrogate_s11,
        surrogate_pattern=surrogate_pattern,
    )
    surrogate_summary, surrogate_metrics = rank_surrogate_candidates(
        run_dir,
        surrogate_s11,
        surrogate_pattern,
        s11_weight=args.s11_metric_weight,
        pattern_weight=args.pattern_metric_weight,
    )
    selected_indices = np.asarray(
        surrogate_summary["selected_candidate_indices"], dtype=np.int64
    )
    condition_axis = np.arange(m)
    selected_metal_python = metal_python[condition_axis, selected_indices].astype(np.uint8)
    selected_feed_python = feed_python[condition_axis, selected_indices]
    selected_metal_matlab = metal_matlab[condition_axis, selected_indices]
    selected_feed_matlab = feed_matlab[condition_axis, selected_indices]
    selected_current_standardized = generated_current[condition_axis, selected_indices]
    selected_current_physical = generated_current_physical[condition_axis, selected_indices]

    np.savez_compressed(
        layout.selected_candidates,
        selected_candidate_indices=selected_indices,
        generated_current_standardized=selected_current_standardized,
        generated_current_physical_python=selected_current_physical,
        metal_binary_python=selected_metal_python,
        feed_rc_python_0based=selected_feed_python,
        metal_binary_matlab=selected_metal_matlab,
        feed_rc_matlab_1based=selected_feed_matlab,
        target_s11=y_raw.numpy().astype(np.float32),
        target_pattern=pattern_raw.numpy().astype(np.float32),
        test_offsets=test_offsets,
        dataset_indices=dataset_indices,
        freq_hz=freq_hz,
        pattern_freq_hz=pattern_freq_hz,
        pattern_theta_deg=theta_deg,
    )
    savemat(
        layout.matlab_input,
        {
            "structures": selected_metal_matlab[:, None, :, :],
            "feed_rc": selected_feed_matlab[:, None, :],
            "selected_candidate_indices": selected_indices.reshape(-1, 1),
            "target_s11": y_raw.numpy().astype(np.float32),
            "target_pattern": pattern_raw.numpy().astype(np.float32),
            "test_offsets": test_offsets.reshape(-1, 1),
            "dataset_indices": dataset_indices.reshape(-1, 1),
            "freq_hz": freq_hz.reshape(1, -1),
            "pattern_freq_hz": pattern_freq_hz.reshape(1, -1),
            "pattern_theta_deg": theta_deg.reshape(1, -1),
            "geometry": _matlab_geometry(geometry),
            "spatial_coordinate_convention": SPATIAL_COORDINATE_CONVENTION,
        },
        do_compression=True,
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "workflow": "generate m*n -> CNN surrogate rank -> full-wave m selected",
        "run_dir": str(layout.root),
        "workspace": str(workspace.resolve()),
        "h5_path": str(h5_path),
        "standardizer_path": str(Path(standardizer_path).resolve()),
        "checkpoints": {
            "diffusion": str(diffusion_checkpoint_path.resolve()),
            "current_to_structure": str(structure_checkpoint_path.resolve()),
            "cnn_forward_surrogate": str(surrogate_checkpoint_path.resolve()),
        },
        "device": str(device),
        "seed": int(args.seed),
        "num_conditions_m": int(m),
        "num_candidates_n": int(n),
        "num_fullwave_jobs": int(m),
        "selection": args.selection,
        "test_offsets": test_offsets.tolist(),
        "dataset_indices": dataset_indices.tolist(),
        "selected_candidate_indices": selected_indices.tolist(),
        "generation_batch_size": int(args.generation_batch_size),
        "surrogate_batch_size": int(args.surrogate_batch_size),
        "s11_cfg_scale": None if s11_cfg_scale is None else float(s11_cfg_scale),
        "pattern_cfg_scale": None if pattern_cfg_scale is None else float(pattern_cfg_scale),
        "metal_threshold": float(args.metal_threshold),
        "topology_postprocess": args.topology_postprocess,
        "postprocess_definition": (
            "threshold metal; global argmax feed; force feed metal; optionally retain only "
            "the feed-containing four-connected component"
        ),
        "feed_sigma": float(feed_sigma),
        "geometry": _matlab_geometry(geometry),
        "ranking": surrogate_summary["ranking"],
        "coordinate_conversion": (
            "Python and MATLAB both use physical (row,col); metal is unchanged; "
            "MATLAB feed_rc=Python feed_rc+1"
        ),
        "spatial_coordinate_convention": SPATIAL_COORDINATE_CONVENTION,
        "pattern_contract": {
            "channel_order": list(PATTERN_CHANNEL_ORDER),
            "value_type": PATTERN_VALUE_TYPE,
            "planes_phi_deg": list(PATTERN_PLANES_PHI_DEG),
        },
        "files": {
            "candidates": str(layout.candidates.relative_to(layout.root).as_posix()),
            "surrogate_predictions": str(layout.surrogate_predictions.relative_to(layout.root).as_posix()),
            "surrogate_ranking": str(layout.surrogate_ranking.relative_to(layout.root).as_posix()),
            "selected_candidates": str(layout.selected_candidates.relative_to(layout.root).as_posix()),
            "matlab_input": str(layout.matlab_input.relative_to(layout.root).as_posix()),
            "matlab_output": str(layout.matlab_output.relative_to(layout.root).as_posix()),
            "final_summary": str(layout.final_summary.relative_to(layout.root).as_posix()),
        },
    }
    with layout.manifest.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)

    if args.visualization_enabled:
        visualize_surrogate_stage(
            run_dir,
            surrogate_metrics,
            selected_indices,
            max_conditions=args.max_visualized_conditions,
            candidate_page_size=args.candidate_page_size,
            pattern_frequency_index=args.pattern_frequency_index,
            dpi=args.figure_dpi,
        )
    print(
        f"[AntGen] proxy ranking selected {m} structures from {m * n}; "
        f"MATLAB input: {layout.matlab_input}"
    )
    return manifest


def _matlab_quote(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def run_matlab_fullwave(
    antgen_root: Path,
    run_dir: Path,
    matlab_command: str = "matlab",
    matlab_workers: int = 0,
) -> Path:
    layout = run_layout(run_dir)
    if not layout.matlab_input.is_file():
        raise FileNotFoundError(layout.matlab_input)
    expression = (
        f"addpath('{_matlab_quote(antgen_root / 'matlab')}'); "
        f"run_fullwave_batch('{_matlab_quote(layout.matlab_input)}',"
        f"'{_matlab_quote(layout.matlab_output)}',{int(matlab_workers)});"
    )
    print(f"[AntGen] launching exactly m selected full-wave jobs with workers={matlab_workers}")
    completed = subprocess.run([matlab_command, "-batch", expression], check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"MATLAB exited with code {completed.returncode}")
    if not layout.matlab_output.is_file():
        raise FileNotFoundError(f"MATLAB completed without creating {layout.matlab_output}")
    print(f"[AntGen] MATLAB results: {layout.matlab_output}")
    return layout.matlab_output
