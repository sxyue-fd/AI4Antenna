# -*- coding: utf-8 -*-

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def _plot_if_present(ax, df, x_col, column, label):
    if column in df.columns:
        values = pd.to_numeric(df[column], errors="coerce")
        mask = values.notna()
        if mask.any():
            ax.plot(df.loc[mask, x_col], values.loc[mask], label=label)


def plot_training_metrics(csv_path, save_path):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Training log not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "step" not in df.columns:
        raise ValueError("CSV must contain a step column")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    ax = axes[0]
    _plot_if_present(ax, df, "step", "train_loss", "Train Loss")
    ax.set_title("Diffusion Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    _plot_if_present(ax, df, "step", "lr", "Learning Rate")
    ax.set_title("Learning Rate")
    ax.set_xlabel("Step")
    ax.set_ylabel("LR")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[2]
    _plot_if_present(ax, df, "step", "sample_mae", "Sample MAE")
    _plot_if_present(ax, df, "step", "sample_y_mae", "S11 MAE")
    _plot_if_present(ax, df, "step", "sample_pattern_mae", "Pattern MAE")
    ax.set_title("Forward Surrogate MAE")
    ax.set_xlabel("Step")
    ax.set_ylabel("MAE")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[3]
    _plot_if_present(ax, df, "step", "sample_mse", "Sample MSE")
    _plot_if_present(ax, df, "step", "sample_y_mse", "S11 MSE")
    _plot_if_present(ax, df, "step", "sample_pattern_mse", "Pattern MSE")
    ax.set_title("Forward Surrogate MSE")
    ax.set_xlabel("Step")
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
