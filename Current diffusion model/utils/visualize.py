# -*- coding: utf-8 -*-

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


def default_freq_indices(num_freqs):
    if num_freqs <= 0:
        return []
    if num_freqs == 1:
        return [0]
    if num_freqs == 2:
        return [0, 1]
    return sorted({0, int(round((num_freqs - 1) / 2)), num_freqs - 1})


def _format_freq_label(freq_hz, index):
    if freq_hz is None or index >= len(freq_hz):
        return f"f{index}"
    freq_ghz = float(freq_hz[index]) / 1e9
    return f"{freq_ghz:g} GHz"


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


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
    current = torch.as_tensor(current).detach().cpu().float()
    if current.ndim == 3:
        current = current.reshape(*stored_current_shape)
    if current.ndim != 4:
        raise ValueError(f"Expected current shape (C,H,W) or (F,C,H,W), got {tuple(current.shape)}")

    stats = getattr(standardizer, "input_stats", {}).get("current")
    if stats is None or stats.get("transform") != "signed_log_clip_zscore":
        return current.numpy(), "standardized current"

    mean = stats["mean"].detach().cpu().float().reshape(current.shape[0], current.shape[1], 1, 1)
    std = stats["std"].detach().cpu().float().reshape(current.shape[0], current.shape[1], 1, 1)
    signed_log = current * std + mean
    return signed_log.numpy(), "signed-log current"


def plot_current_frequency_slices(
    generated_current,
    target_current,
    standardizer,
    dataset_info,
    save_dir,
    freq_indices=None,
):
    stored_shape = tuple(dataset_info.get("stored_current_shape") or (5, 4, 32, 32))
    generated, domain_name = _current_to_signed_log(generated_current, standardizer, stored_shape)
    target, _ = _current_to_signed_log(target_current, standardizer, stored_shape)

    num_freqs, num_components, _, _ = target.shape
    freq_indices = freq_indices or default_freq_indices(num_freqs)
    freq_indices = [idx for idx in freq_indices if 0 <= idx < num_freqs]

    pattern_meta = dataset_info.get("pattern_metadata", {}) or {}
    freq_hz = pattern_meta.get("freq_hz")
    component_labels = ["Jx real", "Jx imag", "Jy real", "Jy imag"]

    os.makedirs(save_dir, exist_ok=True)

    for freq_idx in freq_indices:
        freq_label = _format_freq_label(freq_hz, freq_idx)
        fig, axes = plt.subplots(num_components, 3, figsize=(10, max(2.2 * num_components, 6.0)), squeeze=False)

        for comp_idx in range(num_components):
            true_img = target[freq_idx, comp_idx]
            pred_img = generated[freq_idx, comp_idx]
            diff_img = np.abs(pred_img - true_img)

            vlim = _robust_abs_limit(np.concatenate([true_img.ravel(), pred_img.ravel()]))
            diff_vmax = _robust_abs_limit(diff_img, fallback=1.0)
            comp_label = component_labels[comp_idx] if comp_idx < len(component_labels) else f"comp {comp_idx}"

            axes[comp_idx, 0].imshow(true_img, cmap="coolwarm", vmin=-vlim, vmax=vlim, origin="lower")
            axes[comp_idx, 1].imshow(pred_img, cmap="coolwarm", vmin=-vlim, vmax=vlim, origin="lower")
            axes[comp_idx, 2].imshow(diff_img, cmap="magma", vmin=0.0, vmax=diff_vmax, origin="lower")

            axes[comp_idx, 0].set_ylabel(comp_label, fontsize=9)

        for ax, title in zip(axes[0], ["True", "Generated", "Abs diff"]):
            ax.set_title(title)

        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(f"Current comparison @ {freq_label} ({domain_name})", y=0.995)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"current_freq_{freq_idx}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


def _current_magnitude_image(current, standardizer, stored_current_shape, freq_idx=None):
    current_np, domain_name = _current_to_signed_log(current, standardizer, stored_current_shape)
    if current_np.ndim != 4:
        raise ValueError(f"Expected current snapshot shape (F,C,H,W), got {current_np.shape}")

    num_freqs, num_components, _, _ = current_np.shape
    if freq_idx is None:
        freq_idx = num_freqs // 2
    freq_idx = int(np.clip(freq_idx, 0, num_freqs - 1))

    component_count = min(num_components, 4)
    image = np.sqrt(np.sum(current_np[freq_idx, :component_count] ** 2, axis=0))
    return image, domain_name, freq_idx


def plot_denoising_timeline(
    denoising_currents,
    denoising_timesteps,
    standardizer,
    dataset_info,
    save_path,
    freq_idx=None,
):
    denoising_currents = _to_numpy(denoising_currents)
    if denoising_currents.ndim not in (4, 5):
        raise ValueError(
            f"Expected denoising current snapshots with shape (S,C,H,W) or (S,F,C,H,W), got {denoising_currents.shape}"
        )

    stored_shape = tuple(dataset_info.get("stored_current_shape") or (5, 4, 32, 32))
    images = []
    domain_name = "current"
    resolved_freq_idx = freq_idx
    for snapshot in denoising_currents:
        image, domain_name, resolved_freq_idx = _current_magnitude_image(
            snapshot,
            standardizer=standardizer,
            stored_current_shape=stored_shape,
            freq_idx=resolved_freq_idx,
        )
        images.append(image)

    vmax = _robust_abs_limit(np.concatenate([image.ravel() for image in images]), percentile=99.0, fallback=1.0)
    pattern_meta = dataset_info.get("pattern_metadata", {}) or {}
    freq_label = _format_freq_label(pattern_meta.get("freq_hz"), resolved_freq_idx)

    num_steps = len(images)
    fig, axes = plt.subplots(1, num_steps, figsize=(max(2.2 * num_steps, 8.0), 2.8), squeeze=False)
    axes = axes[0]
    im = None
    for ax, image, timestep in zip(axes, images, denoising_timesteps):
        im = ax.imshow(image, cmap="magma", vmin=0.0, vmax=vmax, origin="lower")
        ax.set_title(f"t={int(timestep)}")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Denoising trajectory @ {freq_label} ({domain_name} magnitude)")
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_s11_comparison(y_pred, y_true, save_path):
    y_pred = _to_numpy(y_pred).reshape(-1)
    y_true = _to_numpy(y_true).reshape(-1)
    freq_idx = np.arange(y_true.shape[0])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(freq_idx, y_true, label="Target", linewidth=2.0)
    ax.plot(freq_idx, y_pred, label="Forward surrogate", linestyle="--", linewidth=2.0)
    ax.set_xlabel("Frequency index")
    ax.set_ylabel("|S11|")
    ax.set_title("S11 comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pattern_frequency_slices(
    p_pred,
    p_true,
    dataset_info,
    save_dir,
    freq_indices=None,
):
    p_pred = _to_numpy(p_pred)
    p_true = _to_numpy(p_true)
    if p_true.ndim != 3:
        raise ValueError(f"Expected pattern shape (Fp,P,T), got {p_true.shape}")

    num_freqs, num_components, num_angles = p_true.shape
    freq_indices = freq_indices or default_freq_indices(num_freqs)
    freq_indices = [idx for idx in freq_indices if 0 <= idx < num_freqs]

    pattern_meta = dataset_info.get("pattern_metadata", {}) or {}
    freq_hz = pattern_meta.get("freq_hz")
    theta_deg = pattern_meta.get("theta_deg")
    if theta_deg is None or len(theta_deg) != num_angles:
        theta_deg = np.arange(num_angles) * 3
    else:
        theta_deg = np.asarray(theta_deg, dtype=np.float32)
    theta_rad = np.deg2rad(theta_deg)

    channel_labels = pattern_meta.get("channels") or []
    os.makedirs(save_dir, exist_ok=True)

    for freq_idx in freq_indices:
        freq_label = _format_freq_label(freq_hz, freq_idx)
        plotted_components = min(num_components, 4)
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(10, 10),
            subplot_kw={"projection": "polar"},
        )
        axes = axes.flatten()

        radial_values = []
        for comp_idx in range(plotted_components):
            radial_values.extend([
                np.asarray(p_true[freq_idx, comp_idx]).ravel(),
                np.asarray(p_pred[freq_idx, comp_idx]).ravel(),
            ])
        radial_values = np.concatenate(radial_values)
        radial_values = radial_values[np.isfinite(radial_values)]
        radial_max = float(np.max(radial_values)) if radial_values.size else 1.0
        if radial_max <= 0:
            radial_max = 1.0

        for comp_idx in range(plotted_components):
            ax = axes[comp_idx]
            ax.plot(theta_rad, p_true[freq_idx, comp_idx], label="Target", linewidth=1.8)
            ax.plot(theta_rad, p_pred[freq_idx, comp_idx], label="Forward surrogate", linestyle="--", linewidth=1.8)
            label = channel_labels[comp_idx] if comp_idx < len(channel_labels) else f"pol {comp_idx}"
            ax.set_title(label)
            ax.set_theta_zero_location("E")
            ax.set_theta_direction(1)
            ax.set_thetagrids(np.arange(0, 360, 30))
            ax.set_ylim(0, radial_max)

        for ax in axes[plotted_components:]:
            ax.set_ylim(0, radial_max)

        axes[0].legend(loc="upper right")
        fig.suptitle(f"Radiation pattern comparison @ {freq_label}")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"pattern_freq_{freq_idx}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


def save_diffusion_visualization_sample(
    output_dir,
    sample_name,
    generated_current,
    target_current,
    y_pred_raw,
    y_true_raw,
    p_pred_raw,
    p_true_raw,
    standardizer,
    dataset_info,
    denoising_currents=None,
    denoising_timesteps=None,
):
    sample_dir = os.path.join(output_dir, sample_name)
    os.makedirs(sample_dir, exist_ok=True)

    pattern_shape = tuple(dataset_info["pattern_shape"])
    current_shape = tuple(dataset_info["stored_current_shape"])
    freq_indices = default_freq_indices(pattern_shape[0])
    current_freq_indices = default_freq_indices(current_shape[0])

    plot_current_frequency_slices(
        generated_current=generated_current,
        target_current=target_current,
        standardizer=standardizer,
        dataset_info=dataset_info,
        freq_indices=current_freq_indices,
        save_dir=os.path.join(sample_dir, "current"),
    )
    if denoising_currents is not None and denoising_timesteps is not None:
        plot_denoising_timeline(
            denoising_currents=denoising_currents,
            denoising_timesteps=denoising_timesteps,
            standardizer=standardizer,
            dataset_info=dataset_info,
            save_path=os.path.join(sample_dir, "denoising_timeline.png"),
        )
    plot_s11_comparison(
        y_pred=y_pred_raw,
        y_true=y_true_raw,
        save_path=os.path.join(sample_dir, "s11_comparison.png"),
    )
    plot_pattern_frequency_slices(
        p_pred=p_pred_raw,
        p_true=p_true_raw,
        dataset_info=dataset_info,
        freq_indices=freq_indices,
        save_dir=os.path.join(sample_dir, "pattern"),
    )
