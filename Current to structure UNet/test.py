# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_PARENT_PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PARENT_PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_PROJECT_DIR)

from configs.default_config import get_default_config, update_config_from_args
from datasets.task_datamodule import build_task_dataloaders
from models import CurrentToStructureUNet


def parse_args():
    parser = argparse.ArgumentParser(description="Test current-to-structure model and visualize predictions.")
    parser.add_argument("--h5_path", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default=None, help="auto / cuda / cpu")
    parser.add_argument("--start", type=int, default=1000, help="Zero-based start offset within the test split")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def find_latest_best_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None

    candidates = []
    for name in os.listdir(output_dir):
        run_dir = os.path.join(output_dir, name)
        ckpt_path = os.path.join(run_dir, "checkpoints", "best_model.pt")
        if name.startswith("train_") and os.path.isfile(ckpt_path):
            candidates.append(ckpt_path)

    if not candidates:
        return None

    candidates.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return candidates[0]


def feed_index_to_rc(feed_index, width):
    feed_index = int(feed_index)
    return feed_index // width, feed_index % width


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


@torch.no_grad()
def collect_predictions(model, loader, device, start, num_samples, threshold):
    rows = []
    seen = 0
    model.eval()

    for batch in loader:
        current = batch["current"].to(device, non_blocking=True)
        outputs = model(current)

        metal_prob = torch.sigmoid(outputs["metal_logits"]).cpu()
        pred_metal = metal_prob >= threshold
        pred_feed = torch.argmax(outputs["feed_logits"].flatten(1).cpu(), dim=1)

        true_metal = batch["metal"].cpu() >= 0.5
        true_feed = batch["feed_index"].cpu()

        batch_size = current.shape[0]
        height, width = pred_metal.shape[-2:]
        for i in range(batch_size):
            if seen < start:
                seen += 1
                continue

            true_feed_rc = feed_index_to_rc(true_feed[i], width)
            pred_feed_rc = feed_index_to_rc(pred_feed[i], width)
            pred_arr = pred_metal[i, 0].numpy()
            true_arr = true_metal[i, 0].numpy()

            pred_arr = pred_arr.copy()
            true_arr = true_arr.copy()
            pred_arr[pred_feed_rc] = True
            true_arr[true_feed_rc] = True

            rows.append(
                {
                    "true_metal": true_arr,
                    "pred_metal": pred_arr,
                    "true_feed_rc": true_feed_rc,
                    "pred_feed_rc": pred_feed_rc,
                    "feed_correct": true_feed_rc == pred_feed_rc,
                    "metal_iou": compute_iou(pred_arr, true_arr),
                    "test_offset": seen,
                }
            )
            seen += 1
            if len(rows) >= num_samples:
                return rows

    return rows


def compute_iou(pred, true):
    intersection = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


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
    cfg_defaults = update_config_from_args(get_default_config(), args)
    h5_path = args.h5_path or cfg_defaults["paths"]["h5_path"]
    output_dir = args.output_dir or os.path.join(cfg_defaults["paths"]["output_dir"], "test")
    device_name = args.device or cfg_defaults["train"]["device"]

    checkpoint_path = args.checkpoint or find_latest_best_checkpoint(cfg_defaults["paths"]["output_dir"])
    if checkpoint_path is None:
        raise FileNotFoundError(
            "No checkpoint was provided and no outputs/train_*/checkpoints/best_model.pt was found."
        )
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"HDF5 dataset not found: {h5_path}")
    if args.start < 0:
        raise ValueError("--start must be >= 0")
    if args.num_samples <= 0:
        raise ValueError("--num_samples must be > 0")

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    dataset_info = checkpoint["dataset_info"]
    cfg = checkpoint["config"]
    cfg["paths"]["h5_path"] = h5_path
    cfg["data"]["batch_size"] = max(1, min(args.num_samples, 64))
    cfg["data"]["num_workers"] = 0
    cfg["data"]["pin_memory"] = device.type == "cuda"
    cfg["data"]["persistent_workers"] = False

    loaders, _ = build_task_dataloaders(cfg)

    model = CurrentToStructureUNet(
        in_channels=int(dataset_info["input_shape"][0]),
        base_channels=cfg["model"]["base_channels"],
        dropout=0.0,
        target_size=cfg["model"]["target_size"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    rows = collect_predictions(
        model=model,
        loader=loaders["test"],
        device=device,
        start=args.start,
        num_samples=args.num_samples,
        threshold=args.threshold,
    )

    os.makedirs(output_dir, exist_ok=True)
    end = args.start + max(len(rows) - 1, 0)
    output_path = os.path.join(output_dir, f"test_{args.start}_to_{end}_{len(rows)}samples_prediction_grid.png")
    save_test_grid(rows, output_path, start=args.start)

    feed_acc = sum(1 for row in rows if row["feed_correct"]) / max(len(rows), 1)
    mean_iou = sum(row["metal_iou"] for row in rows) / max(len(rows), 1)
    print("checkpoint:", checkpoint_path)
    print("h5_path:", h5_path)
    print("test start offset:", args.start)
    print("samples visualized:", len(rows))
    print("feed accuracy in shown samples:", feed_acc)
    print("mean IoU in shown samples:", mean_iou)
    print("saved:", output_path)


if __name__ == "__main__":
    main()
