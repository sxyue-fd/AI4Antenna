# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class S11Encoder(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, s11):
        return self.net(s11)


class PatternEncoder(nn.Module):
    """Encode (Fp, P, T) radiation patterns into angle-aware tokens."""

    def __init__(
        self,
        fp: int,
        p: int,
        t: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        seq_len: int = 15,
    ):
        super().__init__()
        self.fp = fp
        self.p = p
        self.t = t
        self.seq_len = int(seq_len)
        in_channels = fp * p

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, stride=2, padding=2, padding_mode="circular"),
            nn.SiLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2, padding_mode="circular"),
            nn.SiLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=5, stride=2, padding=2, padding_mode="circular"),
            nn.SiLU(),
            nn.BatchNorm1d(128),
        )
        self.pool = nn.AdaptiveAvgPool1d(self.seq_len)
        self.proj = nn.Sequential(
            nn.Conv1d(128, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Conv1d(hidden_dim, embed_dim, kernel_size=1),
        )
        self.token_norm = nn.LayerNorm(embed_dim)

    def forward(self, pattern):
        if pattern.ndim != 4:
            raise ValueError(f"Expected pattern shape (B,Fp,P,T), got {tuple(pattern.shape)}")
        b, fp, p, t = pattern.shape
        if (fp, p, t) != (self.fp, self.p, self.t):
            raise ValueError(f"Expected pattern shape (*,{self.fp},{self.p},{self.t}), got {tuple(pattern.shape)}")
        x = pattern.reshape(b, fp * p, t)
        x = self.conv(x)
        x = self.pool(x)
        x = self.proj(x)
        x = x.transpose(1, 2).contiguous()
        return self.token_norm(x)


class CurrentConditionEncoder(nn.Module):
    def __init__(
        self,
        s11_dim: int,
        pattern_shape,
        s11_embed_dim: int = 256,
        pattern_embed_dim: int = 256,
        condition_dim: int = 512,
        pattern_seq_len: int = 15,
    ):
        super().__init__()
        fp, p, t = pattern_shape
        self.s11_encoder = S11Encoder(s11_dim, s11_embed_dim)
        self.pattern_encoder = PatternEncoder(
            fp,
            p,
            t,
            embed_dim=pattern_embed_dim,
            seq_len=pattern_seq_len,
        )
        self.s11_embed_dim = int(s11_embed_dim)
        self.pattern_embed_dim = int(pattern_embed_dim)
        self.condition_dim = int(condition_dim)

    def encode_s11(self, s11):
        return self.s11_encoder(s11)

    def encode_pattern(self, pattern):
        return self.pattern_encoder(pattern)

    def forward(self, s11, pattern):
        return self.encode_s11(s11), self.encode_pattern(pattern)
