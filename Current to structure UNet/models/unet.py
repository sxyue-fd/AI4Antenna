# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        if in_ch == out_ch and stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + identity)
        return self.dropout(out)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.net = ResidualConvBlock(in_ch, out_ch, dropout, stride=2)

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ResidualConvBlock(out_ch + skip_ch, out_ch, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class CurrentToStructureUNet(nn.Module):
    """Residual U-Net that maps current maps to 16x16 topology logits."""

    def __init__(self, in_channels, base_channels=64, dropout=0.1, target_size=16):
        super().__init__()
        self.target_size = int(target_size)

        b = int(base_channels)
        self.enc32 = ResidualConvBlock(in_channels, b, dropout=dropout)
        self.enc16 = DownBlock(b, b * 2, dropout=dropout)
        self.enc8 = DownBlock(b * 2, b * 4, dropout=dropout)
        self.enc4 = DownBlock(b * 4, b * 8, dropout=dropout)
        #self.bridge = ResidualConvBlock(b * 8, b * 8, dropout=dropout)

        self.bridge = ResidualConvBlock(b * 4, b * 4, dropout=dropout)
        self.dec8 = UpBlock(b * 8, b * 4, b * 4, dropout=dropout)
        self.dec16 = UpBlock(b * 4, b * 2, b * 2, dropout=dropout)

        self.shared = ResidualConvBlock(b * 2, b, dropout=dropout)
        self.metal_head = nn.Conv2d(b, 1, kernel_size=1)
        self.feed_head = nn.Conv2d(b, 1, kernel_size=1)

    def forward(self, x):
        e32 = self.enc32(x)
        e16 = self.enc16(e32)
        e8 = self.enc8(e16)
        #e4 = self.enc4(e8)
        #z = self.bridge(e4)
        z = self.bridge(e8)
        #d8 = self.dec8(z, e8)
        #d16 = self.dec16(d8, e16)
        d16 = self.dec16(z, e16)
        feat = self.shared(d16)
        if feat.shape[-2:] != (self.target_size, self.target_size):
            feat = F.interpolate(feat, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False)

        return {
            "metal_logits": self.metal_head(feat),
            "feed_logits": self.feed_head(feat),
        }
