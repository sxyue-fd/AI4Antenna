# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOSS_COLUMNS = ("loss", "metal_bce", "metal_dice", "feed_ce", "feed_on_air")


def _read_training_log(csv_path):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)

    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {}
            for key, value in raw.items():
                if value is None or value == "":
                    continue
                try:
                    row[key] = float(value)
                except ValueError:
                    row[key] = value
            rows.append(row)
    return rows


def _values(rows, column):
    xs = []
    ys = []
    for row in rows:
        if "epoch" in row and column in row:
            xs.append(row["epoch"])
            ys.append(row[column])
    return xs, ys


def _values_by_x(rows, x_column, y_column):
    xs = []
    ys = []
    for row in rows:
        if x_column in row and y_column in row:
            xs.append(row[x_column])
            ys.append(row[y_column])
    return xs, ys


def _plot_column(ax, rows, column, label):
    xs, ys = _values(rows, column)
    if xs:
        ax.plot(xs, ys, marker="o", markersize=2.5, linewidth=1.4, label=label)


def _finish_axis(ax, title, ylabel):
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)


def plot_training_loss(csv_path, save_path):
    rows = _read_training_log(csv_path)
    if not rows:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    _plot_column(axes[0], rows, "train_loss", "train loss")
    _plot_column(axes[0], rows, "val_loss", "val loss")
    _finish_axis(axes[0], "Total Loss", "Loss")

    for name in LOSS_COLUMNS[1:]:
        _plot_column(axes[1], rows, f"train_{name}", f"train {name}")
        _plot_column(axes[1], rows, f"val_{name}", f"val {name}")
    _finish_axis(axes[1], "Component Losses", "Loss")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_iteration_val_accuracy(csv_path, save_path):
    rows = _read_training_log(csv_path)
    if not rows:
        return

    metal_x, metal_y = _values_by_x(rows, "global_step", "val_metal_acc")
    feed_x, feed_y = _values_by_x(rows, "global_step", "val_feed_acc")
    if not metal_x and not feed_x:
        return

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.8), constrained_layout=True)
    if metal_x:
        ax.plot(metal_x, metal_y, marker="o", markersize=3, linewidth=1.5, label="val metal acc")
    if feed_x:
        ax.plot(feed_x, feed_y, marker="o", markersize=3, linewidth=1.5, label="val feed acc")

    ax.set_title("Validation Accuracy by Iteration")
    ax.set_xlabel("Iteration / Global Step")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
