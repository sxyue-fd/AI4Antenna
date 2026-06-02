# -*- coding: utf-8 -*-

import torch


def mae(pred, target):
    return torch.mean(torch.abs(pred - target))


def compute_metrics(y_pred, y_true, p_pred, p_true):
    """
    输入为原始尺度（已反标准化）的张量
    """
    metrics = {}

    metrics["y_mae"] = mae(y_pred, y_true).item()
    metrics["pattern_mae"] = mae(p_pred, p_true).item()

    return metrics
