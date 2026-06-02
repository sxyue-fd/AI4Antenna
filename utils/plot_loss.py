# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def plot_main_loss(df, save_path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    
    if save_path is None:
        plt.show()

    plt.close()


def plot_sub_losses(df, save_path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_y_loss"], label="Train Y Loss")
    plt.plot(df["epoch"], df["val_y_loss"], label="Val Y Loss")
    plt.plot(df["epoch"], df["train_p_loss"], label="Train Pattern Loss")
    plt.plot(df["epoch"], df["val_p_loss"], label="Val Pattern Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Y Loss / Pattern Loss")
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    
    if save_path is None:
        plt.show()

    plt.close()


def main():
    csv_path = "./outputs/logs/train_log.csv"

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"未找到训练日志文件: {csv_path}")

    df = pd.read_csv(csv_path)

    if "epoch" not in df.columns:
        raise ValueError("CSV 中缺少 epoch 列。")

    plot_main_loss(
        df,
        save_path="./outputs/figures/loss_curve.png",
    )

    required_sub_cols = {
        "train_y_loss", "val_y_loss", "train_p_loss", "val_p_loss"
    }
    if required_sub_cols.issubset(set(df.columns)):
        plot_sub_losses(
            df,
            save_path="./outputs/figures/sub_loss_curve.png",
        )
    else:
        print("CSV 中未找到完整的子损失列，跳过子损失绘图。")


if __name__ == "__main__":
    main()