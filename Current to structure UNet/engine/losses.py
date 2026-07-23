# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss_with_logits(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    dims = tuple(range(1, prob.ndim))
    intersection = torch.sum(prob * target, dim=dims)
    denom = torch.sum(prob + target, dim=dims)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


class StructureLoss(nn.Module):
    def __init__(
        self,
        metal_bce_weight=1.0,
        metal_dice_weight=1.0,
        feed_ce_weight=1.0,
        feed_label_smoothing=0.0,
        #feed_on_air_weight=0.1,
    ):
        super().__init__()
        self.metal_bce_weight = metal_bce_weight
        self.metal_dice_weight = metal_dice_weight
        self.feed_ce_weight = feed_ce_weight
        self.feed_label_smoothing = feed_label_smoothing
        #self.feed_on_air_weight = feed_on_air_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss(label_smoothing=feed_label_smoothing)

    def forward(self, outputs, batch):
        metal_logits = outputs["metal_logits"]
        feed_logits = outputs["feed_logits"]
        metal_target = batch["metal"]
        feed_index = batch["feed_index"]

        metal_bce = self.bce(metal_logits, metal_target)
        metal_dice = dice_loss_with_logits(metal_logits, metal_target)
        feed_ce = self.ce(feed_logits.flatten(1), feed_index)

        feed_prob = F.softmax(feed_logits.flatten(1), dim=1).view_as(feed_logits)
        metal_prob = torch.sigmoid(metal_logits)
        #feed_on_air = torch.mean(feed_prob * (1.0 - metal_prob))

        total = (
            self.metal_bce_weight * metal_bce
            + self.metal_dice_weight * metal_dice
            + self.feed_ce_weight * feed_ce
            #+ self.feed_on_air_weight * feed_on_air
        )

        return {
            "loss": total,
            "metal_bce": metal_bce.detach(),
            "metal_dice": metal_dice.detach(),
            "feed_ce": feed_ce.detach(),
            #"feed_on_air": feed_on_air.detach(),
        }


def build_loss(cfg):
    return StructureLoss(**cfg["loss"])