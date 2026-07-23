# -*- coding: utf-8 -*-
from __future__ import annotations

import torch


@torch.no_grad()
def compute_metrics(outputs, batch, threshold=0.5):
    metal_prob = torch.sigmoid(outputs["metal_logits"])
    metal_pred = metal_prob >= threshold
    metal_true = batch["metal"] >= 0.5

    intersection = torch.logical_and(metal_pred, metal_true).sum(dim=(1, 2, 3)).float()
    union = torch.logical_or(metal_pred, metal_true).sum(dim=(1, 2, 3)).float()
    metal_iou = torch.where(union > 0, intersection / union, torch.ones_like(union)).mean()
    metal_acc = (metal_pred == metal_true).float().mean()

    feed_pred = torch.argmax(outputs["feed_logits"].flatten(1), dim=1)
    feed_true = batch["feed_index"]
    feed_acc = (feed_pred == feed_true).float().mean()

    h, w = outputs["feed_logits"].shape[-2:]
    rows = feed_pred // w
    cols = feed_pred % w
    batch_idx = torch.arange(metal_pred.shape[0], device=metal_pred.device)
    feed_on_pred_metal = metal_pred[batch_idx, 0, rows, cols]
    feed_on_metal_rate = feed_on_pred_metal.float().mean()

    return {
        "metal_iou": metal_iou.item(),
        "metal_acc": metal_acc.item(),
        "feed_acc": feed_acc.item(),
        "feed_on_metal_rate": feed_on_metal_rate.item(),
    }
