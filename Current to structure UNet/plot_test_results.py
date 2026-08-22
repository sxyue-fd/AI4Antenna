# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import pickle

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot current-to-structure test predictions")
    parser.add_argument("--input", required=True, help="Pickled prediction rows")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--start", type=int, default=0)
    return parser.parse_args()


def draw_structure(ax, metal, feed_rc, title):
    ax.imshow(metal, cmap="gray", vmin=0, vmax=1, origin="upper")
    ax.scatter(
        [feed_rc[1]],
        [feed_rc[0]],
        s=55,
        facecolors="none",
        edgecolors="red",
        linewidths=1.5,
    )
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_difference(ax, pred_metal, true_metal, true_feed_rc, pred_feed_rc, title):
    diff = pred_metal.astype(np.int8) - true_metal.astype(np.int8)
    im = ax.imshow(diff, cmap="bwr", vmin=-1, vmax=1, origin="upper")
    ax.scatter(
        [true_feed_rc[1]],
        [true_feed_rc[0]],
        s=45,
        marker="o",
        facecolors="none",
        edgecolors="lime",
        linewidths=1.4,
        label="true feed",
    )
    ax.scatter(
        [pred_feed_rc[1]],
        [pred_feed_rc[0]],
        s=45,
        marker="x",
        c="yellow",
        linewidths=1.4,
        label="pred feed",
    )
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def save_test_grid(rows, output_path, start=0):
    if not rows:
        raise ValueError("No test samples were collected.")

    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(9, 2.6 * n), constrained_layout=True)
    if n == 1:
        axes = axes[None, :]

    diff_im = None
    for row_idx, item in enumerate(rows):
        draw_structure(
            axes[row_idx, 0],
            item["true_metal"],
            item["true_feed_rc"],
            f"test {item.get('test_offset', start + row_idx)} target",
        )
        draw_structure(
            axes[row_idx, 1],
            item["pred_metal"],
            item["pred_feed_rc"],
            f"prediction | IoU={item['metal_iou']:.3f}",
        )
        diff_im = draw_difference(
            axes[row_idx, 2],
            item["pred_metal"],
            item["true_metal"],
            item["true_feed_rc"],
            item["pred_feed_rc"],
            "prediction - target",
        )

    if diff_im is not None:
        cbar = fig.colorbar(diff_im, ax=axes[:, 2], fraction=0.018, pad=0.01)
        cbar.set_ticks([-1, 0, 1])
        cbar.set_ticklabels(["missing", "same", "extra"])

    fig.suptitle(
        "Current to Structure Test Samples | target feed: red circle / lime on diff, "
        "pred feed: red circle / yellow x on diff",
        fontsize=12,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    with open(args.input, "rb") as f:
        rows = pickle.load(f)
    save_test_grid(rows, args.output, start=args.start)


if __name__ == "__main__":
    main()
