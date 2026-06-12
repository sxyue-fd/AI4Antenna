# -*- coding: utf-8 -*-

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def _plot_if_present(ax, df, column, label):
    if column in df.columns:
        run_ids = (df["epoch"].diff().fillna(1) <= 0).cumsum()
        for run_id, run_df in df.groupby(run_ids):
            run_label = label if run_id == 0 else f"{label} (run {run_id + 1})"
            ax.plot(run_df["epoch"], run_df[column], label=run_label)


def _plot_first_present(ax, df, columns, label):
    for column in columns:
        if column in df.columns:
            _plot_if_present(ax, df, column, label)
            return


def plot_training_metrics(csv_path, save_path):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Training log not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "epoch" not in df.columns:
        raise ValueError("CSV must contain an epoch column")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    ax = axes[0]
    _plot_if_present(ax, df, "train_loss", "Train Loss")
    _plot_if_present(ax, df, "val_loss", "Val Loss")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    _plot_if_present(ax, df, "train_y_loss", "Train Y Loss")
    _plot_first_present(ax, df, ["train_p_loss", "train_pattern_loss"], "Train P Loss")
    _plot_if_present(ax, df, "val_y_loss", "Val Y Loss")
    _plot_first_present(ax, df, ["val_p_loss", "val_pattern_loss"], "Val P Loss")
    ax.set_title("Sub Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[2]
    _plot_if_present(ax, df, "val_mae", "Val MAE")
    _plot_if_present(ax, df, "val_y_mae", "Val Y MAE")
    _plot_if_present(ax, df, "val_pattern_mae", "Val Pattern MAE")
    ax.set_title("Validation MAE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[3]
    _plot_if_present(ax, df, "val_mse", "Val MSE")
    _plot_if_present(ax, df, "val_y_mse", "Val Y MSE")
    _plot_if_present(ax, df, "val_pattern_mse", "Val Pattern MSE")
    ax.set_title("Validation MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main(csv_path="./outputs/logs/train_log.csv", save_path="./outputs/figures/training_metrics.png"):
    plot_training_metrics(csv_path=csv_path, save_path=save_path)


if __name__ == "__main__":
    main()
