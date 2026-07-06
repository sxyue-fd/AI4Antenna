# -*- coding: utf-8 -*-

import os

import torch


def _slice_meta(meta, end):
    sliced = {}
    for key, value in meta.items():
        if torch.is_tensor(value):
            sliced[key] = value[:end]
        else:
            sliced[key] = value
    return sliced


def _move_model_kwargs(y, p, device, cfg):
    kwargs = {
        "s11": y.to(device, non_blocking=True),
        "pattern": p.to(device, non_blocking=True),
    }
    sample_cfg = cfg.get("sample", {})
    s11_cfg_scale = sample_cfg.get("s11_cfg_scale")
    pattern_cfg_scale = sample_cfg.get("pattern_cfg_scale")
    cfg_scale = sample_cfg.get("cfg_scale")

    if s11_cfg_scale is not None or pattern_cfg_scale is not None:
        if s11_cfg_scale is not None:
            kwargs["s11_cfg_scale"] = float(s11_cfg_scale)
        if pattern_cfg_scale is not None:
            kwargs["pattern_cfg_scale"] = float(pattern_cfg_scale)
    elif cfg_scale is not None:
        kwargs["cfg_scale"] = float(cfg_scale)
    return kwargs


@torch.no_grad()
def sample_and_evaluate(
    diffusion_model,
    surrogate_model,
    loader,
    standardizer,
    device,
    cfg,
    num_samples=None,
    save_path=None,
    logger=None,
    split_name="val",
):
    diffusion_model.eval()
    surrogate_model.eval()

    if num_samples is None:
        num_samples = cfg.get("sample", {}).get("num_samples", 16)

    totals = {
        "y_abs": 0.0,
        "p_abs": 0.0,
        "y_sq": 0.0,
        "p_sq": 0.0,
        "y_numel": 0,
        "p_numel": 0,
    }
    saved_payload = None
    seen = 0

    for current_true, y, p, meta in loader:
        remaining = num_samples - seen
        if remaining <= 0:
            break

        if y.size(0) > remaining:
            current_true = current_true[:remaining]
            y = y[:remaining]
            p = p[:remaining]
            meta = _slice_meta(meta, remaining)

        model_kwargs = _move_model_kwargs(y, p, device, cfg)
        generated_current = diffusion_model.sample(
            batch_size=y.size(0),
            model_kwargs=model_kwargs,
            progress=False,
        )

        y_pred_norm, p_pred_norm = surrogate_model(generated_current)
        y_pred_raw = standardizer.denormalize_y(y_pred_norm).cpu()
        p_pred_raw = standardizer.denormalize_p(p_pred_norm).cpu()

        y_true_raw = meta["y_raw"]
        p_true_raw = meta["p_raw"]

        totals["y_abs"] += torch.sum(torch.abs(y_pred_raw - y_true_raw)).item()
        totals["p_abs"] += torch.sum(torch.abs(p_pred_raw - p_true_raw)).item()
        totals["y_sq"] += torch.sum((y_pred_raw - y_true_raw) ** 2).item()
        totals["p_sq"] += torch.sum((p_pred_raw - p_true_raw) ** 2).item()
        totals["y_numel"] += y_pred_raw.numel()
        totals["p_numel"] += p_pred_raw.numel()

        if saved_payload is None and save_path is not None and cfg.get("sample", {}).get("save_tensors", True):
            saved_payload = {
                "generated_current": generated_current.detach().cpu(),
                "target_current": current_true.detach().cpu(),
                "s11_condition": y.detach().cpu(),
                "pattern_condition": p.detach().cpu(),
                "s11_pred_raw": y_pred_raw,
                "pattern_pred_raw": p_pred_raw,
                "s11_true_raw": y_true_raw,
                "pattern_true_raw": p_true_raw,
            }

        seen += y.size(0)

    if totals["y_numel"] == 0 or totals["p_numel"] == 0:
        raise RuntimeError(f"No samples were evaluated for split {split_name!r}")

    metrics = {
        "y_mae": totals["y_abs"] / totals["y_numel"],
        "pattern_mae": totals["p_abs"] / totals["p_numel"],
        "mae": (totals["y_abs"] + totals["p_abs"]) / (totals["y_numel"] + totals["p_numel"]),
        "y_mse": totals["y_sq"] / totals["y_numel"],
        "pattern_mse": totals["p_sq"] / totals["p_numel"],
        "mse": (totals["y_sq"] + totals["p_sq"]) / (totals["y_numel"] + totals["p_numel"]),
        "num_samples": seen,
    }

    if save_path is not None and saved_payload is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        saved_payload["metrics"] = metrics
        torch.save(saved_payload, save_path)

    if logger:
        logger.info("[%s sample] %s", split_name, metrics)

    return metrics
