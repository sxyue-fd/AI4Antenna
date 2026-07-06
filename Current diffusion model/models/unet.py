# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), kernel_size=2, stride=2)


def _group_count(dim, max_groups=32):
    for groups in range(min(max_groups, dim), 0, -1):
        if dim % groups == 0:
            return groups
    return 1


def _attention_heads(dim, max_heads=4):
    for heads in range(min(max_heads, dim), 0, -1):
        if dim % heads == 0:
            return heads
    return 1


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None].float() * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Block(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(_group_count(dim_out), dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, context_dim, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(context_dim, dim_out * 2))
        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context):
        context = self.mlp(context)
        context = context[:, :, None, None]
        scale_shift = context.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.dropout(h)
        h = self.block2(h)
        return h + self.res_conv(x)


class CrossAttention2d(nn.Module):
    def __init__(self, dim, context_dim, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(_group_count(dim), dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=_attention_heads(dim, heads),
            kdim=context_dim,
            vdim=context_dim,
            batch_first=True,
            bias=False,
        )

    def forward(self, x, context):
        if context is None:
            return x
        if context.ndim != 3:
            raise ValueError(f"Expected pattern context shape (B,seq,dim), got {tuple(context.shape)}")

        b, c, h, w = x.shape
        if context.shape[0] != b:
            raise ValueError(f"Pattern context batch mismatch: x={b}, context={context.shape[0]}")

        context = context.to(dtype=x.dtype)
        tokens = self.norm(x).flatten(2).transpose(1, 2)
        attended, _ = self.attn(tokens, context, context, need_weights=False)
        attended = attended.transpose(1, 2).reshape(b, c, h, w)
        return x + attended


class OptionalCrossAttention2d(nn.Module):
    def __init__(self, dim, context_dim, enabled):
        super().__init__()
        self.attn = CrossAttention2d(dim, context_dim) if enabled else None

    def forward(self, x, context):
        if self.attn is None:
            return x
        return self.attn(x, context)


class SelfAttention2d(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(_group_count(dim), dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=_attention_heads(dim, heads),
            batch_first=True,
            bias=False,
        )

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = self.norm(x).flatten(2).transpose(1, 2)
        attended, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        attended = attended.transpose(1, 2).reshape(b, c, h, w)
        return x + attended


class FiLMUnet(nn.Module):
    def __init__(
        self,
        dim,
        s11_condition_dim,
        pattern_context_dim,
        dim_mults=(1, 2, 4, 8),
        channels=20,
        condition_dropout=0.1,
        input_resolution=32,
        attention_resolutions=(16, 8, 4),
    ):
        super().__init__()
        self.init_conv = nn.Conv2d(channels, dim, 3, padding=1)
        self.channels = channels
        self.s11_condition_dim = s11_condition_dim
        self.pattern_context_dim = pattern_context_dim
        self.condition_dropout = condition_dropout
        self.attention_resolutions = set(int(v) for v in attention_resolutions)

        dims = [dim] + [dim * m for m in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))
        in_out_ups = [(b, a) for a, b in reversed(in_out)]

        context_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, context_dim),
        )
        self.s11_condition_mlp = nn.Sequential(
            nn.Linear(s11_condition_dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, context_dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        resolution = int(input_resolution)
        for index, (dim_in, dim_out) in enumerate(in_out):
            is_last = index == len(in_out) - 1
            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_in, context_dim=context_dim),
                        ResnetBlock(dim_in, dim_in, context_dim=context_dim),
                        OptionalCrossAttention2d(
                            dim_in,
                            context_dim=pattern_context_dim,
                            enabled=resolution in self.attention_resolutions,
                        ),
                        nn.Conv2d(dim_in, dim_out, 3, padding=1)
                        if is_last
                        else Downsample(dim_in, dim_out),
                    ]
                )
            )
            if not is_last:
                resolution //= 2

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, context_dim=context_dim)
        self.mid_self_attn = SelfAttention2d(mid_dim)
        self.mid_cross_attn = OptionalCrossAttention2d(
            mid_dim,
            context_dim=pattern_context_dim,
            enabled=resolution in self.attention_resolutions,
        )
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, context_dim=context_dim)

        up_resolution = resolution
        for index, (dim_in, dim_out) in enumerate(in_out_ups):
            is_first = index == 0
            current_resolution = up_resolution if is_first else up_resolution * 2
            self.ups.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(dim_in, dim_out, 3, padding=1)
                        if is_first
                        else Upsample(dim_in, dim_out),
                        ResnetBlock(dim_out * 2, dim_out, context_dim=context_dim),
                        ResnetBlock(dim_out * 2, dim_out, context_dim=context_dim),
                        OptionalCrossAttention2d(
                            dim_out,
                            context_dim=pattern_context_dim,
                            enabled=current_resolution in self.attention_resolutions,
                        ),
                    ]
                )
            )
            up_resolution = current_resolution

        self.final_conv = nn.Conv2d(dim, channels, 1)

    def _contexts(self, time, s11_condition, pattern_context):
        context = self.time_mlp(time)

        if s11_condition is not None:
            s11_context = self.s11_condition_mlp(s11_condition)
            if self.training and self.condition_dropout > 0:
                keep = (torch.rand(s11_context.shape[0], device=s11_context.device) > self.condition_dropout).to(
                    dtype=s11_context.dtype
                )
                s11_context = s11_context * keep[:, None]
            context = context + s11_context

        if pattern_context is not None and self.training and self.condition_dropout > 0:
            keep = (torch.rand(pattern_context.shape[0], device=pattern_context.device) > self.condition_dropout).to(
                dtype=pattern_context.dtype
            )
            pattern_context = pattern_context * keep[:, None, None]

        return context, pattern_context

    def cfg_forward(
        self,
        x,
        time,
        s11_condition=None,
        pattern_context=None,
        cfg_scale=None,
        s11_cfg_scale=None,
        pattern_cfg_scale=None,
    ):
        if cfg_scale is not None:
            if s11_cfg_scale is None:
                s11_cfg_scale = cfg_scale
            if pattern_cfg_scale is None:
                pattern_cfg_scale = cfg_scale

        s11_cfg_scale = 0.0 if s11_cfg_scale is None else float(s11_cfg_scale)
        pattern_cfg_scale = 0.0 if pattern_cfg_scale is None else float(pattern_cfg_scale)

        base_out = self.forward(
            x,
            time,
            s11_condition=s11_condition,
            pattern_context=pattern_context,
            cfg_scale=None,
            s11_cfg_scale=None,
            pattern_cfg_scale=None,
        )
        guided = base_out

        if s11_condition is not None and s11_cfg_scale != 0.0:
            without_s11 = self.forward(
                x,
                time,
                s11_condition=None,
                pattern_context=pattern_context,
                cfg_scale=None,
                s11_cfg_scale=None,
                pattern_cfg_scale=None,
            )
            guided = guided + s11_cfg_scale * (base_out - without_s11)

        if pattern_context is not None and pattern_cfg_scale != 0.0:
            without_pattern = self.forward(
                x,
                time,
                s11_condition=s11_condition,
                pattern_context=None,
                cfg_scale=None,
                s11_cfg_scale=None,
                pattern_cfg_scale=None,
            )
            guided = guided + pattern_cfg_scale * (base_out - without_pattern)

        return guided

    def forward(
        self,
        x,
        time,
        s11_condition=None,
        pattern_context=None,
        cfg_scale=None,
        s11_cfg_scale=None,
        pattern_cfg_scale=None,
    ):
        if cfg_scale is not None or s11_cfg_scale is not None or pattern_cfg_scale is not None:
            return self.cfg_forward(
                x,
                time,
                s11_condition=s11_condition,
                pattern_context=pattern_context,
                cfg_scale=cfg_scale,
                s11_cfg_scale=s11_cfg_scale,
                pattern_cfg_scale=pattern_cfg_scale,
            )

        context, pattern_context = self._contexts(time, s11_condition, pattern_context)
        x = self.init_conv(x)

        skips = []
        for resnet1, resnet2, cross_attn, downsample in self.downs:
            x = resnet1(x, context)
            skips.append(x)
            x = resnet2(x, context)
            x = cross_attn(x, pattern_context)
            skips.append(x)
            x = downsample(x)

        x = self.mid_block1(x, context)
        x = self.mid_self_attn(x)
        x = self.mid_cross_attn(x, pattern_context)
        x = self.mid_block2(x, context)

        for upsample, resnet1, resnet2, cross_attn in self.ups:
            x = upsample(x)
            skip2 = skips.pop()
            if x.shape[-2:] != skip2.shape[-2:]:
                x = F.interpolate(x, size=skip2.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat((x, skip2), dim=1)
            x = resnet1(x, context)

            skip1 = skips.pop()
            if x.shape[-2:] != skip1.shape[-2:]:
                x = F.interpolate(x, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat((x, skip1), dim=1)
            x = resnet2(x, context)
            x = cross_attn(x, pattern_context)

        return self.final_conv(x)
