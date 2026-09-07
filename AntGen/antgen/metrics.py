"""Surrogate ranking and selected-candidate full-wave validation metrics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from .contracts import (
    PATTERN_CHANNEL_ORDER,
    PATTERN_CHANNEL_ORDER_CSV,
    PATTERN_PLANES_PHI_DEG,
    PATTERN_VALUE_TYPE,
    SCHEMA_VERSION,
    SPATIAL_COORDINATE_CONVENTION,
)
from .layout import run_layout


METRIC_NAMES = (
    "s11_mae",
    "s11_mse",
    "pattern_mae",
    "pattern_mse",
    "combined_mae",
    "combined_mse",
)


def response_metrics(
    s11_pred: np.ndarray,
    pattern_pred: np.ndarray,
    target_s11: np.ndarray,
    target_pattern: np.ndarray,
    s11_weight: float = 1.0,
    pattern_weight: float = 1.0,
) -> dict[str, np.ndarray]:
    """Compute per-candidate raw-space errors for arrays shaped ``(m,n,...)``."""
    if s11_weight < 0 or pattern_weight < 0 or s11_weight + pattern_weight <= 0:
        raise ValueError("Metric weights must be non-negative and not both zero")
    target_s11 = np.asarray(target_s11)[:, None, :]
    target_pattern = np.asarray(target_pattern)[:, None, ...]
    s11_diff = np.asarray(s11_pred) - target_s11
    pattern_diff = np.asarray(pattern_pred) - target_pattern
    s11_mae = np.mean(np.abs(s11_diff), axis=-1)
    s11_mse = np.mean(np.square(s11_diff), axis=-1)
    pattern_axes = tuple(range(2, pattern_diff.ndim))
    pattern_mae = np.mean(np.abs(pattern_diff), axis=pattern_axes)
    pattern_mse = np.mean(np.square(pattern_diff), axis=pattern_axes)
    weight_sum = s11_weight + pattern_weight
    return {
        "s11_mae": s11_mae,
        "s11_mse": s11_mse,
        "pattern_mae": pattern_mae,
        "pattern_mse": pattern_mse,
        "combined_mae": (s11_weight * s11_mae + pattern_weight * pattern_mae) / weight_sum,
        "combined_mse": (s11_weight * s11_mse + pattern_weight * pattern_mse) / weight_sum,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def _cell_text(value: Any) -> str:
    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.item()
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"U", "S"}:
            return "".join(value.reshape(-1).astype(str).tolist()).strip()
        return str(value.tolist())
    return "" if value is None else str(value).strip()


def _number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def rank_surrogate_candidates(
    run_dir: Path,
    s11_pred: np.ndarray,
    pattern_pred: np.ndarray,
    s11_weight: float = 1.0,
    pattern_weight: float = 1.0,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Rank all ``m*n`` candidates with the CNN surrogate and write reports."""
    layout = run_layout(run_dir, create=True)
    with np.load(layout.candidates, allow_pickle=False) as candidates:
        target_s11 = np.asarray(candidates["target_s11"], dtype=np.float64)
        target_pattern = np.asarray(candidates["target_pattern"], dtype=np.float64)
        dataset_indices = np.asarray(candidates["dataset_indices"], dtype=np.int64)
        test_offsets = np.asarray(candidates["test_offsets"], dtype=np.int64)
        feed_python = np.asarray(candidates["feed_rc_python_0based"], dtype=np.int64)
        feed_matlab = np.asarray(candidates["feed_rc_matlab_1based"], dtype=np.int64)
        metal = np.asarray(candidates["metal_binary_python"], dtype=np.uint8)

    s11_pred = np.asarray(s11_pred, dtype=np.float64)
    pattern_pred = np.asarray(pattern_pred, dtype=np.float64)
    m, n = s11_pred.shape[:2]
    if s11_pred.shape != (m, n, *target_s11.shape[1:]):
        raise ValueError("CNN surrogate S11 prediction shape does not match targets")
    if pattern_pred.shape != (m, n, *target_pattern.shape[1:]):
        raise ValueError("CNN surrogate pattern prediction shape does not match targets")

    metrics = response_metrics(
        s11_pred,
        pattern_pred,
        target_s11,
        target_pattern,
        s11_weight=s11_weight,
        pattern_weight=pattern_weight,
    )
    finite = np.ones((m, n), dtype=bool)
    for values in metrics.values():
        finite &= np.isfinite(values)

    rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    best_indices = np.empty(m, dtype=np.int64)
    for condition_index in range(m):
        valid = np.flatnonzero(finite[condition_index])
        if not valid.size:
            raise RuntimeError(f"CNN surrogate returned no finite candidate for condition {condition_index}")
        order = valid[
            np.argsort(metrics["combined_mae"][condition_index, valid], kind="stable")
        ]
        best_indices[condition_index] = int(order[0])
        ranks = {int(candidate): rank + 1 for rank, candidate in enumerate(order)}
        condition_rows = []
        for candidate_index in range(n):
            row: dict[str, Any] = {
                "condition_index": condition_index,
                "test_offset": int(test_offsets[condition_index]),
                "dataset_index": int(dataset_indices[condition_index]),
                "candidate_index": candidate_index,
                "surrogate_rank": ranks.get(candidate_index, ""),
                "selected_for_fullwave": candidate_index == best_indices[condition_index],
                "metal_pixels": int(metal[condition_index, candidate_index].sum()),
                "feed_row_python_0based": int(feed_python[condition_index, candidate_index, 0]),
                "feed_col_python_0based": int(feed_python[condition_index, candidate_index, 1]),
                "feed_row_matlab_1based": int(feed_matlab[condition_index, candidate_index, 0]),
                "feed_col_matlab_1based": int(feed_matlab[condition_index, candidate_index, 1]),
            }
            for name in METRIC_NAMES:
                row[f"surrogate_{name}"] = _number(metrics[name][condition_index, candidate_index])
            condition_rows.append(row)
        rows.extend(condition_rows)
        best_rows.append(next(row for row in condition_rows if row["selected_for_fullwave"]))

    fields = [
        "condition_index", "test_offset", "dataset_index", "candidate_index",
        "surrogate_rank", "selected_for_fullwave", "metal_pixels",
        "feed_row_python_0based", "feed_col_python_0based",
        "feed_row_matlab_1based", "feed_col_matlab_1based",
        *(f"surrogate_{name}" for name in METRIC_NAMES),
    ]
    rows.sort(key=lambda row: (row["condition_index"], row["surrogate_rank"] or n + 1))
    _write_csv(layout.surrogate_ranking, rows, list(fields))
    _write_csv(layout.surrogate_best, best_rows, list(fields))

    mean_of_selected = {
        name: float(np.mean(metrics[name][np.arange(m), best_indices])) for name in METRIC_NAMES
    }
    mean_of_all = {name: float(np.mean(metrics[name][finite])) for name in METRIC_NAMES}
    summary = {
        "schema_version": SCHEMA_VERSION,
        "ranking_model": "CNN-based forward surrogate model",
        "num_conditions_m": int(m),
        "num_candidates_per_condition_n": int(n),
        "num_candidates_ranked": int(m * n),
        "num_selected_for_fullwave": int(m),
        "selected_candidate_indices": best_indices.tolist(),
        "ranking": {
            "metric": "surrogate_combined_mae",
            "formula": "(s11_weight*s11_mae + pattern_weight*pattern_mae) / weight_sum",
            "s11_weight": float(s11_weight),
            "pattern_weight": float(pattern_weight),
        },
        "mean_surrogate_metrics_of_selected": mean_of_selected,
        "mean_surrogate_metrics_over_all_candidates": mean_of_all,
    }
    _write_json(layout.surrogate_summary, summary)
    print(f"[AntGen] CNN surrogate ranked {m * n} candidates: {layout.surrogate_ranking}")
    return summary, metrics


def analyze_fullwave_results(
    run_dir: Path,
    s11_weight: float = 1.0,
    pattern_weight: float = 1.0,
) -> dict[str, Any]:
    """Evaluate the ``m`` surrogate-selected candidates with MATLAB results."""
    layout = run_layout(run_dir)
    if not layout.selected_candidates.is_file():
        raise FileNotFoundError(layout.selected_candidates)
    if not layout.matlab_output.is_file():
        raise FileNotFoundError(layout.matlab_output)

    with layout.surrogate_summary.open("r", encoding="utf-8") as handle:
        surrogate_summary = json.load(handle)
    saved_ranking = surrogate_summary["ranking"]
    if not (
        np.isclose(float(saved_ranking["s11_weight"]), float(s11_weight))
        and np.isclose(float(saved_ranking["pattern_weight"]), float(pattern_weight))
    ):
        raise ValueError(
            "Metric weights differ from the weights used to select candidates. "
            "Rerun the infer stage and full-wave simulation after changing ranking weights."
        )

    with np.load(layout.selected_candidates, allow_pickle=False) as selected:
        selected_indices = np.asarray(selected["selected_candidate_indices"], dtype=np.int64)
        target_s11 = np.asarray(selected["target_s11"], dtype=np.float64)
        target_pattern = np.asarray(selected["target_pattern"], dtype=np.float64)
        dataset_indices = np.asarray(selected["dataset_indices"], dtype=np.int64)
        test_offsets = np.asarray(selected["test_offsets"], dtype=np.int64)
        freq_hz = np.asarray(selected["freq_hz"], dtype=np.float64).reshape(-1)
        pattern_freq_hz = np.asarray(selected["pattern_freq_hz"], dtype=np.float64).reshape(-1)
        pattern_theta_deg = np.asarray(selected["pattern_theta_deg"], dtype=np.float64).reshape(-1)

    matlab = loadmat(layout.matlab_output, squeeze_me=False, struct_as_record=False)
    coordinate_convention = _cell_text(matlab.get("spatial_coordinate_convention"))
    if coordinate_convention != SPATIAL_COORDINATE_CONVENTION:
        raise ValueError(
            "MATLAB output is missing the corrected physical row/col coordinate contract. "
            "Rerun infer and simulate with the current AntGen code."
        )
    pattern_channel_order = _cell_text(matlab.get("pattern_channel_order"))
    if pattern_channel_order != PATTERN_CHANNEL_ORDER_CSV:
        raise ValueError(
            f"MATLAB pattern channel order is {pattern_channel_order!r}; "
            f"expected {PATTERN_CHANNEL_ORDER_CSV!r}"
        )
    pattern_value_type = _cell_text(matlab.get("pattern_value_type"))
    if pattern_value_type != PATTERN_VALUE_TYPE:
        raise ValueError(
            f"MATLAB pattern value type is {pattern_value_type!r}; "
            f"expected {PATTERN_VALUE_TYPE!r}"
        )
    s11_sim = np.asarray(matlab["s11_sim"], dtype=np.float64)
    pattern_sim = np.asarray(matlab["pattern_sim"], dtype=np.float64)
    success = np.asarray(matlab["success"], dtype=bool)
    errors = matlab.get("error_messages")
    m = target_s11.shape[0]
    if success.shape != (m, 1):
        raise ValueError(f"Expected exactly m={m} MATLAB jobs with success shape {(m, 1)}, got {success.shape}")
    if s11_sim.shape != (m, 1, *target_s11.shape[1:]):
        raise ValueError("MATLAB S11 output shape does not match the selected targets")
    if pattern_sim.shape != (m, 1, *target_pattern.shape[1:]):
        raise ValueError("MATLAB pattern output shape does not match the selected targets")
    matlab_freq_hz = np.asarray(matlab["freq_hz"], dtype=np.float64).reshape(-1)
    matlab_pattern_freq_hz = np.asarray(matlab["pattern_freq_hz"], dtype=np.float64).reshape(-1)
    matlab_pattern_theta_deg = np.asarray(
        matlab["pattern_theta_deg"], dtype=np.float64
    ).reshape(-1)
    if not np.allclose(matlab_freq_hz, freq_hz):
        raise ValueError("MATLAB S11 frequency axis differs from the selected targets")
    if not np.allclose(matlab_pattern_freq_hz, pattern_freq_hz):
        raise ValueError("MATLAB pattern frequency axis differs from the selected targets")
    if not np.allclose(matlab_pattern_theta_deg, pattern_theta_deg):
        raise ValueError("MATLAB pattern theta axis differs from the selected targets")
    if "selected_candidate_indices" in matlab:
        matlab_selected = np.asarray(matlab["selected_candidate_indices"], dtype=np.int64).reshape(-1)
        if not np.array_equal(matlab_selected, selected_indices):
            raise ValueError("MATLAB output selected candidate indices do not match selection artifact")

    metrics = response_metrics(
        s11_sim,
        pattern_sim,
        target_s11,
        target_pattern,
        s11_weight=s11_weight,
        pattern_weight=pattern_weight,
    )
    valid = success[:, 0].copy()
    for values in metrics.values():
        valid &= np.isfinite(values[:, 0])

    rows: list[dict[str, Any]] = []
    for condition_index in range(m):
        row: dict[str, Any] = {
            "condition_index": condition_index,
            "test_offset": int(test_offsets[condition_index]),
            "dataset_index": int(dataset_indices[condition_index]),
            "selected_candidate_index": int(selected_indices[condition_index]),
            "fullwave_success": bool(valid[condition_index]),
        }
        for name in METRIC_NAMES:
            row[f"fullwave_{name}"] = _number(metrics[name][condition_index, 0])
        row["error"] = (
            _cell_text(errors[condition_index, 0])
            if errors is not None and errors.shape[:2] == (m, 1)
            else ""
        )
        rows.append(row)

    fields = [
        "condition_index", "test_offset", "dataset_index", "selected_candidate_index",
        "fullwave_success", *(f"fullwave_{name}" for name in METRIC_NAMES), "error",
    ]
    _write_csv(layout.fullwave_results, rows, list(fields))
    mean_metrics = {
        name: (float(np.mean(metrics[name][:, 0][valid])) if np.any(valid) else None)
        for name in METRIC_NAMES
    }
    fullwave_summary = {
        "schema_version": SCHEMA_VERSION,
        "num_conditions_m": int(m),
        "num_candidates_per_condition_simulated": 1,
        "num_fullwave_simulations_requested": int(m),
        "num_fullwave_successes": int(valid.sum()),
        "num_fullwave_failures": int(m - valid.sum()),
        "condition_coverage": float(valid.mean()) if m else 0.0,
        "selected_candidate_indices": selected_indices.tolist(),
        "spatial_coordinate_convention": SPATIAL_COORDINATE_CONVENTION,
        "pattern_contract": {
            "channel_order": list(PATTERN_CHANNEL_ORDER),
            "value_type": PATTERN_VALUE_TYPE,
            "planes_phi_deg": list(PATTERN_PLANES_PHI_DEG),
        },
        "matlab_version": _cell_text(matlab.get("matlab_version")),
        "matlab_release": _cell_text(matlab.get("matlab_release")),
        "mean_fullwave_metrics_of_selected": mean_metrics,
    }
    _write_json(layout.fullwave_summary, fullwave_summary)

    final_summary = {
        "schema_version": SCHEMA_VERSION,
        "workflow": "CNN surrogate ranks m*n candidates; MATLAB validates only m selected candidates",
        "surrogate_ranking": surrogate_summary,
        "fullwave_validation": fullwave_summary,
    }
    _write_json(layout.final_summary, final_summary)
    print(f"[AntGen] full-wave metrics for {m} selected candidates: {layout.fullwave_results}")
    print(f"[AntGen] final report: {layout.final_summary}")
    return final_summary
