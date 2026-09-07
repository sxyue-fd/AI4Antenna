"""Headless visual reports for surrogate ranking and full-wave validation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from .contracts import PATTERN_CHANNEL_ORDER
from .layout import run_layout
from .metrics import response_metrics


PATTERN_LABELS = tuple(label.replace("_", " ") for label in PATTERN_CHANNEL_ORDER)


def _condition_dir(layout, condition_index: int) -> Path:
    directory = layout.figures / f"condition_{condition_index:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _candidate_pages(
    output_dir: Path,
    metal: np.ndarray,
    feed: np.ndarray,
    combined_mae: np.ndarray,
    selected_index: int,
    page_size: int,
    dpi: int,
) -> None:
    order = np.argsort(combined_mae, kind="stable")
    page_size = max(1, int(page_size))
    for page_number, start in enumerate(range(0, order.size, page_size), start=1):
        page = order[start : start + page_size]
        columns = min(4, len(page))
        rows = math.ceil(len(page) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(3.1 * columns, 3.2 * rows), squeeze=False)
        for axis, candidate_index in zip(axes.flat, page):
            axis.imshow(metal[candidate_index], cmap="Blues", vmin=0, vmax=1, origin="upper")
            row, col = feed[candidate_index]
            axis.scatter(col, row, marker="o", s=75, facecolors="none", edgecolors="red", linewidths=1.8)
            rank = int(np.flatnonzero(order == candidate_index)[0]) + 1
            selected = int(candidate_index) == int(selected_index)
            axis.set_title(
                f"candidate {candidate_index} | rank {rank}\nCNN combined MAE={combined_mae[candidate_index]:.4g}",
                color="darkgreen" if selected else "black",
                fontsize=9,
            )
            if selected:
                for spine in axis.spines.values():
                    spine.set_edgecolor("limegreen")
                    spine.set_linewidth(3)
            axis.set_xticks([])
            axis.set_yticks([])
        for axis in axes.flat[len(page) :]:
            axis.axis("off")
        fig.suptitle("Postprocessed candidates (red circle = feed; green = full-wave selection)")
        fig.tight_layout()
        suffix = "" if order.size <= page_size else f"_page_{page_number:02d}"
        fig.savefig(output_dir / f"candidate_topologies{suffix}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def _selected_current(
    output_path: Path,
    current: np.ndarray,
    pattern_freq_hz: np.ndarray,
    dpi: int,
) -> None:
    # Channels: Jx_real, Jx_imag, Jy_real, Jy_imag.
    amplitude = np.sqrt(np.sum(np.square(current), axis=1))
    frequency_indices = sorted(set((0, amplitude.shape[0] // 2, amplitude.shape[0] - 1)))
    fig, axes = plt.subplots(1, len(frequency_indices), figsize=(4 * len(frequency_indices), 3.6), squeeze=False)
    vmax = float(np.max(np.log1p(amplitude))) or 1.0
    image = None
    for axis, frequency_index in zip(axes.flat, frequency_indices):
        image = axis.imshow(
            np.log1p(amplitude[frequency_index]),
            cmap="magma",
            origin="upper",
            vmin=0,
            vmax=vmax,
        )
        frequency = pattern_freq_hz[frequency_index] / 1e9
        axis.set_title(f"{frequency:g} GHz")
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("Selected generated current: log(1 + |J|)")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.78)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _response_figure(
    output_path: Path,
    topology: np.ndarray,
    feed: np.ndarray,
    freq_hz: np.ndarray,
    theta_deg: np.ndarray,
    pattern_freq_hz: np.ndarray,
    target_s11: np.ndarray,
    target_pattern: np.ndarray,
    surrogate_s11: np.ndarray,
    surrogate_pattern: np.ndarray,
    pattern_frequency_index: int,
    dpi: int,
    fullwave_s11: np.ndarray | None = None,
    fullwave_pattern: np.ndarray | None = None,
    fullwave_success: bool | None = None,
) -> None:
    fp = target_pattern.shape[0]
    frequency_index = int(np.clip(pattern_frequency_index, 0, fp - 1))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    topology_axis = axes[0, 0]
    topology_axis.imshow(topology, cmap="Blues", vmin=0, vmax=1, origin="upper")
    topology_axis.scatter(feed[1], feed[0], marker="o", s=90, facecolors="none", edgecolors="red", linewidths=2)
    topology_axis.set_title("Selected topology / feed")
    topology_axis.set_xticks([])
    topology_axis.set_yticks([])

    s11_axis = axes[0, 1]
    s11_axis.plot(freq_hz / 1e9, target_s11, label="target", linewidth=2)
    s11_axis.plot(freq_hz / 1e9, surrogate_s11, label="CNN surrogate", linestyle="--")
    if fullwave_s11 is not None and fullwave_success:
        s11_axis.plot(freq_hz / 1e9, fullwave_s11, label="full-wave", linestyle="-.")
    s11_axis.set_xlabel("Frequency (GHz)")
    s11_axis.set_ylabel("|S11| (linear)")
    s11_axis.set_title("S11 comparison")
    s11_axis.grid(alpha=0.25)
    s11_axis.legend(fontsize=8)

    info_axis = axes[0, 2]
    info_axis.axis("off")
    info_axis.text(
        0.02,
        0.95,
        f"Pattern slice: {pattern_freq_hz[frequency_index] / 1e9:g} GHz\n"
        + (
            "Full-wave: not run yet"
            if fullwave_success is None
            else ("Full-wave: success" if fullwave_success else "Full-wave: FAILED")
        ),
        va="top",
        fontsize=11,
    )

    pattern_axes = (axes[1, 0], axes[1, 1], axes[1, 2], info_axis.inset_axes([0.05, 0.05, 0.9, 0.62]))
    for channel, axis in enumerate(pattern_axes):
        axis.plot(theta_deg, target_pattern[frequency_index, channel], label="target", linewidth=1.8)
        axis.plot(theta_deg, surrogate_pattern[frequency_index, channel], label="CNN", linestyle="--")
        if fullwave_pattern is not None and fullwave_success:
            axis.plot(theta_deg, fullwave_pattern[frequency_index, channel], label="full-wave", linestyle="-.")
        axis.set_title(PATTERN_LABELS[channel], fontsize=9)
        axis.set_xlabel("Theta (deg)")
        axis.set_ylabel("Gain (linear)")
        axis.grid(alpha=0.2)
        axis.tick_params(labelsize=7)
    pattern_axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def visualize_surrogate_stage(
    run_dir: Path,
    metrics: dict[str, np.ndarray],
    selected_indices: np.ndarray,
    max_conditions: int = 20,
    candidate_page_size: int = 16,
    pattern_frequency_index: int = 2,
    dpi: int = 150,
) -> None:
    layout = run_layout(run_dir, create=True)
    with np.load(layout.candidates, allow_pickle=False) as candidates, np.load(
        layout.surrogate_predictions, allow_pickle=False
    ) as predictions:
        metal = candidates["metal_binary_python"]
        feed = candidates["feed_rc_python_0based"]
        current = candidates["generated_current_physical_python"]
        target_s11 = candidates["target_s11"]
        target_pattern = candidates["target_pattern"]
        freq_hz = candidates["freq_hz"]
        pattern_freq_hz = candidates["pattern_freq_hz"]
        theta_deg = candidates["pattern_theta_deg"]
        surrogate_s11 = predictions["surrogate_s11"]
        surrogate_pattern = predictions["surrogate_pattern"]

        count = min(int(max_conditions), metal.shape[0])
        for condition_index in range(count):
            output_dir = _condition_dir(layout, condition_index)
            selected = int(selected_indices[condition_index])
            _candidate_pages(
                output_dir,
                metal[condition_index],
                feed[condition_index],
                metrics["combined_mae"][condition_index],
                selected,
                candidate_page_size,
                dpi,
            )
            _selected_current(
                output_dir / "selected_current.png",
                current[condition_index, selected],
                pattern_freq_hz,
                dpi,
            )
            _response_figure(
                output_dir / "selected_surrogate_response.png",
                metal[condition_index, selected],
                feed[condition_index, selected],
                freq_hz,
                theta_deg,
                pattern_freq_hz,
                target_s11[condition_index],
                target_pattern[condition_index],
                surrogate_s11[condition_index, selected],
                surrogate_pattern[condition_index, selected],
                pattern_frequency_index,
                dpi,
                fullwave_success=None,
            )
    print(f"[AntGen] surrogate visualizations: {layout.figures}")


def visualize_fullwave_stage(
    run_dir: Path,
    s11_weight: float,
    pattern_weight: float,
    max_conditions: int = 20,
    pattern_frequency_index: int = 2,
    dpi: int = 150,
) -> None:
    layout = run_layout(run_dir)
    with np.load(layout.selected_candidates, allow_pickle=False) as selected, np.load(
        layout.surrogate_predictions, allow_pickle=False
    ) as predictions:
        selected_indices = selected["selected_candidate_indices"].astype(np.int64)
        metal = selected["metal_binary_python"]
        feed = selected["feed_rc_python_0based"]
        target_s11 = selected["target_s11"]
        target_pattern = selected["target_pattern"]
        freq_hz = selected["freq_hz"]
        pattern_freq_hz = selected["pattern_freq_hz"]
        theta_deg = selected["pattern_theta_deg"]
        condition_axis = np.arange(selected_indices.size)
        surrogate_s11 = predictions["surrogate_s11"][condition_axis, selected_indices]
        surrogate_pattern = predictions["surrogate_pattern"][condition_axis, selected_indices]

    matlab = loadmat(layout.matlab_output, squeeze_me=False, struct_as_record=False)
    fullwave_s11 = np.asarray(matlab["s11_sim"])[:, 0]
    fullwave_pattern = np.asarray(matlab["pattern_sim"])[:, 0]
    success = np.asarray(matlab["success"], dtype=bool)[:, 0]
    count = min(int(max_conditions), selected_indices.size)
    for condition_index in range(count):
        output_dir = _condition_dir(layout, condition_index)
        _response_figure(
            output_dir / "selected_fullwave_comparison.png",
            metal[condition_index],
            feed[condition_index],
            freq_hz,
            theta_deg,
            pattern_freq_hz,
            target_s11[condition_index],
            target_pattern[condition_index],
            surrogate_s11[condition_index],
            surrogate_pattern[condition_index],
            pattern_frequency_index,
            dpi,
            fullwave_s11=fullwave_s11[condition_index],
            fullwave_pattern=fullwave_pattern[condition_index],
            fullwave_success=bool(success[condition_index]),
        )

    surrogate_metric = response_metrics(
        surrogate_s11[:, None], surrogate_pattern[:, None], target_s11, target_pattern,
        s11_weight, pattern_weight,
    )["combined_mae"][:, 0]
    fullwave_metric = response_metrics(
        fullwave_s11[:, None], fullwave_pattern[:, None], target_s11, target_pattern,
        s11_weight, pattern_weight,
    )["combined_mae"][:, 0]
    fullwave_metric[~success] = np.nan
    x = np.arange(selected_indices.size)
    width = 0.38
    fig, axis = plt.subplots(figsize=(max(8, selected_indices.size * 0.55), 4.8))
    axis.bar(x - width / 2, surrogate_metric, width, label="CNN surrogate")
    axis.bar(x + width / 2, fullwave_metric, width, label="full-wave")
    axis.set_xlabel("Condition index")
    axis.set_ylabel("Combined MAE")
    axis.set_title("Selected-candidate surrogate vs full-wave error")
    axis.set_xticks(x)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(layout.figures / "summary_metrics.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[AntGen] full-wave visualizations: {layout.figures}")
