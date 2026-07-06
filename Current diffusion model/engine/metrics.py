# -*- coding: utf-8 -*-

import torch


def mae(pred, target):
    return torch.mean(torch.abs(pred - target))


def mse(pred, target):
    return torch.mean((pred - target) ** 2)


def compute_surrogate_metrics(y_pred, y_true, p_pred, p_true):
    y_abs_sum = torch.sum(torch.abs(y_pred - y_true))
    p_abs_sum = torch.sum(torch.abs(p_pred - p_true))
    y_sq_sum = torch.sum((y_pred - y_true) ** 2)
    p_sq_sum = torch.sum((p_pred - p_true) ** 2)

    y_numel = y_pred.numel()
    p_numel = p_pred.numel()
    total_numel = y_numel + p_numel

    return {
        "y_mae": (y_abs_sum / max(y_numel, 1)).item(),
        "pattern_mae": (p_abs_sum / max(p_numel, 1)).item(),
        "mae": ((y_abs_sum + p_abs_sum) / max(total_numel, 1)).item(),
        "y_mse": (y_sq_sum / max(y_numel, 1)).item(),
        "pattern_mse": (p_sq_sum / max(p_numel, 1)).item(),
        "mse": ((y_sq_sum + p_sq_sum) / max(total_numel, 1)).item(),
    }

