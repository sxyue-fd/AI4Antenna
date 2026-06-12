# -*- coding: utf-8 -*-

import torch
from engine.metrics import compute_metrics


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    criterion,
    standardizer,
    device,
    cfg,
    logger=None,
    split_name="val",
    use_amp=False,
):
    model.eval()

    total_loss = 0.0
    total_y_loss = 0.0
    total_p_loss = 0.0

    total_metrics = {
        "mae": 0.0,
        "y_mae": 0.0,
        "pattern_mae": 0.0,
        "mse": 0.0,
        "y_mse": 0.0,
        "pattern_mse": 0.0,
    }

    count = 0
    amp_device = "cuda" if device.type == "cuda" else "cpu"

    for x, y, p, meta in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        p = p.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
            y_pred, p_pred = model(x)
            loss = criterion(y_pred, y, p_pred, p)

            y_loss = torch.mean((y_pred - y) ** 2)
            p_loss = torch.mean((p_pred - p) ** 2)
            #y_loss = torch.mean(torch.abs(y_pred - y))
            #p_loss = torch.mean(torch.abs(p_pred - p))

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_y_loss += y_loss.item() * bs
        total_p_loss += p_loss.item() * bs
        count += bs

        y_pred_raw = standardizer.denormalize_y(y_pred).cpu()
        p_pred_raw = standardizer.denormalize_p(p_pred).cpu()

        y_true_raw = meta["y_raw"]
        p_true_raw = meta["p_raw"]

        metrics = compute_metrics(
            y_pred_raw,
            y_true_raw,
            p_pred_raw,
            p_true_raw,
        )

        for k in total_metrics:
            total_metrics[k] += metrics[k] * bs

    results = {
        "loss": total_loss / max(count, 1),
        "y_loss": total_y_loss / max(count, 1),
        "p_loss": total_p_loss / max(count, 1),
    }

    for k in total_metrics:
        results[k] = total_metrics[k] / max(count, 1)

    if logger:
        logger.info(f"[{split_name}] " + str(results))

    return results
