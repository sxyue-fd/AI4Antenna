# -*- coding: utf-8 -*-
from __future__ import annotations

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def _robust_abs_limit(values, percentile=99.0, fallback=1.0):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return fallback
    limit = float(np.percentile(np.abs(values), percentile))
    if limit <= 0:
        limit = float(np.max(np.abs(values)))
    return limit if limit > 0 else fallback


def _current_to_signed_log(current, standardizer, stored_current_shape):
    """Use the same visualization-domain conversion as Current diffusion model."""
    current = torch.as_tensor(current).detach().cpu().float()
    if current.ndim == 4 and current.shape[0] == 1:
        current = current[0]
    if current.ndim == 3:
        current = current.reshape(*stored_current_shape)
    if current.ndim != 4:
        raise ValueError(f"Expected current shape (C,H,W) or (F,C,H,W), got {tuple(current.shape)}")

    stats = getattr(standardizer, "input_stats", {}).get("current") if standardizer is not None else None
    if stats is None or stats.get("transform") != "signed_log_clip_zscore":
        return current.numpy(), "standardized current"

    mean = stats["mean"].detach().cpu().float().reshape(current.shape[0], current.shape[1], 1, 1)
    std = stats["std"].detach().cpu().float().reshape(current.shape[0], current.shape[1], 1, 1)
    return (current * std + mean).numpy(), "signed-log current"


def save_structure_prediction(
    path,
    metal_true,
    metal_prob,
    feed_true_index,
    feed_logits,
    threshold=0.5,
    input_current=None,
    standardizer=None,
    stored_current_shape=None,
    pattern_metadata=None,
):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    metal_true = metal_true.squeeze().detach().cpu()
    metal_prob = metal_prob.squeeze().detach().cpu()
    feed_logits = feed_logits.squeeze().detach().cpu()
    feed_pred_index = int(torch.argmax(feed_logits.flatten()).item())

    h, w = metal_true.shape
    feed_true_row = int(feed_true_index) // w
    feed_true_col = int(feed_true_index) % w
    feed_pred_row = feed_pred_index // w
    feed_pred_col = feed_pred_index % w

    true_metal = (metal_true >= 0.5).numpy().astype(np.int8)
    pred_metal = (metal_prob >= float(threshold)).numpy().astype(np.int8)
    true_metal[feed_true_row, feed_true_col] = 1
    pred_metal[feed_pred_row, feed_pred_col] = 1
    diff = pred_metal - true_metal

    current_fc = None
    current_domain = None
    if input_current is not None:
        if stored_current_shape is None:
            raise ValueError("stored_current_shape is required when input_current is provided")
        current_fc, current_domain = _current_to_signed_log(input_current, standardizer, stored_current_shape)

    current_rows = 0 if current_fc is None else int(current_fc.shape[0])
    fig, axes = plt.subplots(
        current_rows + 1,
        4,
        figsize=(15, 3.25 * current_rows + 4.0),
        squeeze=False,
        constrained_layout=True,
    )

    component_labels = ["Jx real", "Jx imag", "Jy real", "Jy imag"]
    pattern_metadata = pattern_metadata or {}
    freq_hz = pattern_metadata.get("freq_hz")
    if current_fc is not None:
        for freq_idx in range(current_fc.shape[0]):
            if freq_hz is not None and freq_idx < len(freq_hz):
                freq_label = f"{float(freq_hz[freq_idx]) / 1e9:g} GHz"
            else:
                freq_label = f"frequency {freq_idx + 1}"
            for comp_idx in range(current_fc.shape[1]):
                image = current_fc[freq_idx, comp_idx]
                vlim = _robust_abs_limit(image)
                axes[freq_idx, comp_idx].imshow(
                    image,
                    cmap="coolwarm",
                    vmin=-vlim,
                    vmax=vlim,
                    origin="upper",
                )
                if freq_idx == 0:
                    label = component_labels[comp_idx] if comp_idx < len(component_labels) else f"comp {comp_idx}"
                    axes[freq_idx, comp_idx].set_title(label)
                if comp_idx == 0:
                    axes[freq_idx, comp_idx].set_ylabel(freq_label, fontsize=10)

    structure_axes = axes[current_rows]
    structure_axes[0].imshow(true_metal, cmap="gray", vmin=0, vmax=1, origin="upper")
    structure_axes[0].scatter([feed_true_col], [feed_true_row], s=55, facecolors="none", edgecolors="red", linewidths=1.5)
    structure_axes[0].set_title("Target")

    structure_axes[1].imshow(pred_metal, cmap="gray", vmin=0, vmax=1, origin="upper")
    structure_axes[1].scatter([feed_pred_col], [feed_pred_row], s=55, facecolors="none", edgecolors="red", linewidths=1.5)
    structure_axes[1].set_title(f"Prediction (thr={threshold:g})")

    structure_axes[2].imshow(feed_logits, cmap="magma", origin="upper")
    structure_axes[2].scatter([feed_pred_col], [feed_pred_row], c="cyan", marker="x")
    structure_axes[2].set_title("Feed Logits")

    diff_im = structure_axes[3].imshow(diff, cmap="bwr", vmin=-1, vmax=1, origin="upper")
    structure_axes[3].scatter(
        [feed_true_col],
        [feed_true_row],
        s=45,
        marker="o",
        facecolors="none",
        edgecolors="lime",
        linewidths=1.4,
    )
    structure_axes[3].scatter([feed_pred_col], [feed_pred_row], s=45, marker="x", c="yellow", linewidths=1.4)
    structure_axes[3].set_title("Prediction - Target")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    cbar = fig.colorbar(diff_im, ax=structure_axes[3], fraction=0.046, pad=0.04)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(["missing", "same", "extra"])

    if current_domain is not None:
        fig.suptitle(f"Input current at all frequencies ({current_domain}) and structure prediction", fontsize=15)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
