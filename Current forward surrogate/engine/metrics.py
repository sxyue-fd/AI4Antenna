# -*- coding: utf-8 -*-

import torch


def mae(pred, target):
    return torch.mean(torch.abs(pred - target))


def mse(pred, target):
    return torch.mean((pred - target) ** 2)


def compute_metrics(y_pred, y_true, p_pred, p_true):
    """Compute metrics in the original, denormalized target scale."""
    metrics = {
        "y_mae": mae(y_pred, y_true).item(),
        "pattern_mae": mae(p_pred, p_true).item(),
        "y_mse": mse(y_pred, y_true).item(),
        "pattern_mse": mse(p_pred, p_true).item(),
    }

    y_abs_sum = torch.sum(torch.abs(y_pred - y_true))
    p_abs_sum = torch.sum(torch.abs(p_pred - p_true))
    y_sq_sum = torch.sum((y_pred - y_true) ** 2)
    p_sq_sum = torch.sum((p_pred - p_true) ** 2)
    total_elements = y_pred.numel() + p_pred.numel()

    metrics["mae"] = ((y_abs_sum + p_abs_sum) / total_elements).item()
    metrics["mse"] = ((y_sq_sum + p_sq_sum) / total_elements).item()

    return metrics
