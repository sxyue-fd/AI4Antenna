# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import os
import numpy as np


def plot_s11(y_true, y_pred, save_path=None, title="S11 Prediction"):
    plt.figure()
    plt.plot(y_true, label="True")
    plt.plot(y_pred, label="Pred", linestyle="--")
    plt.xlabel("Frequency Index")
    plt.ylabel("|Gamma|")
    plt.title(title)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_pattern_slice(
        p_true,
        p_pred,
        freq_step=5,
        save_dir=None
):
    """
    每隔5个频点画一次4个极化的方向图

    参数:
        p_true: [Fp, P, T]
        p_pred: [Fp, P, T]

        Fp: 频点数
        P : 极化数（这里默认4个）
        T : 角度采样点

        freq_step: 频点间隔
        save_dir : 保存目录
    """

    Fp, P, T = p_true.shape

    # 角度
    angles_deg = np.arange(T) * 3
    angles_rad = np.deg2rad(angles_deg)

    # 创建保存目录
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # 每隔5个频点画一次
    for freq_idx in range(0, Fp, freq_step):

        # 2x2 子图，对应4个极化
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(10, 10),
            subplot_kw={'projection': 'polar'}
        )

        axes = axes.flatten()

        for pol_idx in range(min(P, 4)):

            ax = axes[pol_idx]

            true_curve = p_true[freq_idx, pol_idx]
            pred_curve = p_pred[freq_idx, pol_idx]

            # 绘图
            ax.plot(angles_rad, true_curve, label="True")
            ax.plot(angles_rad, pred_curve,
                    linestyle="--",
                    label="Pred")

            # 标题
            ax.set_title(f"Pol {pol_idx}")

            # 极坐标设置
            ax.set_theta_zero_location("E")
            ax.set_theta_direction(1)
            ax.set_thetagrids(np.arange(0, 360, 30))

        # 总标题
        fig.suptitle(f"Radiation Pattern @ Freq Index {freq_idx}")

        # 图例（只放一次）
        axes[0].legend(loc="upper right")

        plt.tight_layout()

        # 保存
        if save_dir is not None:
            save_path = os.path.join(
                save_dir,
                f"pattern_freq_{freq_idx}.png"
            )
            plt.savefig(save_path,
                        dpi=150,
                        bbox_inches="tight")

        plt.close(fig)


def save_prediction_example(output_dir, sample_idx, y_true, y_pred, p_true, p_pred):
    os.makedirs(output_dir, exist_ok=True)

    # S11
    plot_s11(
        y_true,
        y_pred,
        save_path=os.path.join(output_dir, f"s11_{sample_idx}.png"),
    )

    # Pattern slice
    plot_pattern_slice(
        p_true,
        p_pred,
        freq_step=5,
        save_dir=os.path.join(output_dir, f"pattern_{sample_idx}"),
    )