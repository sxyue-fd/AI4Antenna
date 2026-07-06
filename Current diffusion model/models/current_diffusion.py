# -*- coding: utf-8 -*-

import torch.nn as nn

from .condition_encoder import CurrentConditionEncoder
from .gaussian_diffusion import GaussianDiffusion
from .unet import FiLMUnet


class CurrentDenoiser(nn.Module):
    def __init__(
        self,
        current_shape,
        s11_dim,
        pattern_shape,
        base_dim=64,
        dim_mults=(1, 2, 4, 8),
        condition_dim=512,
        s11_embed_dim=256,
        pattern_embed_dim=256,
        pattern_seq_len=15,
        condition_dropout=0.1,
        attention_resolutions=(16, 8, 4),
    ):
        super().__init__()
        channels = current_shape[0]
        self.condition_encoder = CurrentConditionEncoder(
            s11_dim=s11_dim,
            pattern_shape=pattern_shape,
            s11_embed_dim=s11_embed_dim,
            pattern_embed_dim=pattern_embed_dim,
            condition_dim=condition_dim,
            pattern_seq_len=pattern_seq_len,
        )
        self.unet = FiLMUnet(
            dim=base_dim,
            s11_condition_dim=s11_embed_dim,
            pattern_context_dim=pattern_embed_dim,
            dim_mults=dim_mults,
            channels=channels,
            condition_dropout=condition_dropout,
            input_resolution=current_shape[-1],
            attention_resolutions=attention_resolutions,
        )

    def forward(self, x, time, model_kwargs=None):
        model_kwargs = model_kwargs or {}
        s11_condition = model_kwargs.get("s11_condition")
        pattern_context = model_kwargs.get("pattern_context")
        cfg_scale = model_kwargs.get("cfg_scale")
        s11_cfg_scale = model_kwargs.get("s11_cfg_scale")
        pattern_cfg_scale = model_kwargs.get("pattern_cfg_scale")

        s11 = model_kwargs.get("s11")
        pattern = model_kwargs.get("pattern")
        if s11_condition is None and s11 is not None:
            s11_condition = self.condition_encoder.encode_s11(s11)
        if pattern_context is None and pattern is not None:
            pattern_context = self.condition_encoder.encode_pattern(pattern)

        return self.unet(
            x,
            time,
            s11_condition=s11_condition,
            pattern_context=pattern_context,
            cfg_scale=cfg_scale,
            s11_cfg_scale=s11_cfg_scale,
            pattern_cfg_scale=pattern_cfg_scale,
        )


def build_current_diffusion(cfg, dataset_info):
    current_shape = tuple(dataset_info["current_shape"])
    y_shape = tuple(dataset_info["y_shape"])
    pattern_shape = tuple(dataset_info["pattern_shape"])

    if len(current_shape) != 3:
        raise ValueError(f"Expected current_shape=(C,H,W), got {current_shape}")
    if len(y_shape) != 1:
        raise ValueError(f"Expected y_shape=(F,), got {y_shape}")
    if len(pattern_shape) != 3:
        raise ValueError(f"Expected pattern_shape=(Fp,P,T), got {pattern_shape}")

    model_cfg = cfg.get("model", {})
    diffusion_cfg = cfg.get("diffusion", {})

    denoiser = CurrentDenoiser(
        current_shape=current_shape,
        s11_dim=y_shape[0],
        pattern_shape=pattern_shape,
        base_dim=model_cfg.get("base_dim", 64),
        dim_mults=tuple(model_cfg.get("dim_mults", (1, 2, 4, 8))),
        condition_dim=model_cfg.get("condition_dim", 512),
        s11_embed_dim=model_cfg.get("s11_embed_dim", 256),
        pattern_embed_dim=model_cfg.get("pattern_embed_dim", 256),
        pattern_seq_len=model_cfg.get("pattern_seq_len", 15),
        condition_dropout=model_cfg.get("condition_dropout", 0.1),
        attention_resolutions=tuple(model_cfg.get("attention_resolutions", (16, 8, 4))),
    )

    return GaussianDiffusion(
        model=denoiser,
        sample_shape=current_shape,
        timesteps=diffusion_cfg.get("timesteps", 1000),
        objective=diffusion_cfg.get("objective", "pred_noise"),
        beta_schedule=diffusion_cfg.get("beta_schedule", "sigmoid"),
        x_start_clip=diffusion_cfg.get("x_start_clip", 12.0),
    )
