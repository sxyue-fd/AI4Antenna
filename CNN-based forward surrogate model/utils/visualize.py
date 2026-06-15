import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def default_pattern_freq_indices(fp):
    if fp <= 0:
        return []
    if fp == 1:
        return [0]
    if fp == 2:
        return [0, 1]
    return sorted({0, int(round((fp - 1) / 2)), fp - 1})


def plot_pattern_slice(
        p_true,
        p_pred,
        freq_indices=None,
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

        freq_indices: optional frequency indices. None means 8/10/12 GHz
            for uniformly sampled 8-12 GHz pattern data.
        save_dir : 保存目录
    """

    Fp, P, T = p_true.shape

    # 角度
    angles_deg = np.arange(T) * 3
    angles_rad = np.deg2rad(angles_deg)

    # 创建保存目录
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    if freq_indices is None:
        freq_indices = default_pattern_freq_indices(Fp)
    else:
        freq_indices = sorted({int(i) for i in freq_indices if 0 <= int(i) < Fp})

    for freq_idx in freq_indices:

        # 2x2 子图，对应4个极化
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(10, 10),
            subplot_kw={'projection': 'polar'}
        )

        axes = axes.flatten()
        plotted_pols = min(P, 4)
        radial_values = []
        for pol_idx in range(plotted_pols):
            radial_values.extend([
                np.asarray(p_true[freq_idx, pol_idx]).ravel(),
                np.asarray(p_pred[freq_idx, pol_idx]).ravel(),
            ])

        if radial_values:
            radial_values = np.concatenate(radial_values)
            radial_values = radial_values[np.isfinite(radial_values)]
            radial_max = float(np.max(radial_values)) if radial_values.size else 1.0
            if radial_max <= 0:
                radial_max = 1.0
        else:
            radial_max = 1.0

        for pol_idx in range(plotted_pols):

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
            ax.set_ylim(0, radial_max)

        for ax in axes[plotted_pols:]:
            ax.set_ylim(0, radial_max)

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
        save_dir=os.path.join(output_dir, f"pattern_{sample_idx}"),
    )
