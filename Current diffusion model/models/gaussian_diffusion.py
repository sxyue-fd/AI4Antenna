# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn
from tqdm.auto import tqdm


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        sample_shape,
        timesteps=1000,
        objective="pred_noise",
        beta_schedule="sigmoid",
        x_start_clip=12.0,
    ):
        super().__init__()
        self.model = model
        self.sample_shape = tuple(int(v) for v in sample_shape)
        self.objective = objective
        self.x_start_clip = x_start_clip
        if objective not in {"pred_noise", "pred_x_start"}:
            raise ValueError("objective must be either pred_noise or pred_x_start")

        register_buffer = lambda name, val: self.register_buffer(name, val.float())

        betas = get_beta_schedule(beta_schedule, timesteps)
        self.num_timesteps = int(betas.shape[0])
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        register_buffer("betas", betas)
        register_buffer("alphas", alphas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        alphas_cumprod_prev = nn.functional.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )
        posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        register_buffer("posterior_std", torch.sqrt(posterior_var.clamp(min=1e-20)))

        snr = alphas_cumprod / (1 - alphas_cumprod)
        loss_weight = torch.ones_like(snr) if objective == "pred_noise" else snr
        register_buffer("loss_weight", loss_weight)

    def predict_start_from_noise(self, x_t, t, noise):
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t

    def predict_noise_from_start(self, x_t, t, x_start):
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - x_start * sqrt_alphas_cumprod_t) / sqrt_one_minus_alphas_cumprod_t

    def q_posterior(self, x_start, x_t, t):
        c1 = extract(self.posterior_mean_coef1, t, x_t.shape)
        c2 = extract(self.posterior_mean_coef2, t, x_t.shape)
        posterior_mean = c1 * x_start + c2 * x_t
        posterior_std = extract(self.posterior_std, t, x_t.shape)
        return posterior_mean, posterior_std

    @torch.no_grad()
    def p_sample(self, x_t, t: int, model_kwargs=None):
        model_kwargs = model_kwargs or {}
        t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)

        model_out = self.model(x_t, t_tensor, model_kwargs=model_kwargs)
        if self.objective == "pred_noise":
            x_start = self.predict_start_from_noise(x_t, t_tensor, model_out)
        else:
            x_start = model_out

        if self.x_start_clip is not None:
            clip = float(self.x_start_clip)
            x_start = x_start.clamp(-clip, clip)

        posterior_mean, posterior_std = self.q_posterior(x_start=x_start, x_t=x_t, t=t_tensor)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t_tensor != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        return posterior_mean + nonzero_mask * posterior_std * noise

    @torch.no_grad()
    def sample(self, batch_size=16, return_all_timesteps=False, model_kwargs=None, progress=True):
        model_kwargs = model_kwargs or {}
        shape = (batch_size, *self.sample_shape)
        img = torch.randn(shape, device=self.betas.device)
        imgs = [img]

        iterator = reversed(range(0, self.num_timesteps))
        if progress:
            iterator = tqdm(iterator, desc="sampling loop time step", total=self.num_timesteps)

        for t in iterator:
            img = self.p_sample(img, t, model_kwargs=model_kwargs)
            if return_all_timesteps:
                imgs.append(img)

        return img if not return_all_timesteps else torch.stack(imgs, dim=1)

    @torch.no_grad()
    def sample_with_snapshots(self, batch_size=16, snapshot_timesteps=None, model_kwargs=None, progress=True):
        model_kwargs = model_kwargs or {}
        if snapshot_timesteps is None:
            snapshot_timesteps = [self.num_timesteps, self.num_timesteps * 3 // 4, self.num_timesteps // 2, self.num_timesteps // 4, 1]

        ordered_timesteps = []
        seen = set()
        for timestep in snapshot_timesteps:
            timestep = int(timestep)
            if timestep < 1 or timestep > self.num_timesteps:
                raise ValueError(f"Snapshot timestep must be in [1,{self.num_timesteps}], got {timestep}")
            if timestep not in seen:
                ordered_timesteps.append(timestep)
                seen.add(timestep)

        shape = (batch_size, *self.sample_shape)
        img = torch.randn(shape, device=self.betas.device)
        snapshots = {}

        if self.num_timesteps in seen:
            snapshots[self.num_timesteps] = img.detach().clone()

        iterator = reversed(range(0, self.num_timesteps))
        if progress:
            iterator = tqdm(iterator, desc="sampling loop time step", total=self.num_timesteps)

        for t in iterator:
            img = self.p_sample(img, t, model_kwargs=model_kwargs)
            timestep_label = t + 1
            if timestep_label in seen and timestep_label not in snapshots:
                snapshots[timestep_label] = img.detach().clone()

        snapshot_tensor = torch.stack([snapshots[timestep] for timestep in ordered_timesteps], dim=1)
        return img, snapshot_tensor, ordered_timesteps

    def q_sample(self, x_start, t, noise):
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, model_kwargs=None, noise=None):
        model_kwargs = model_kwargs or {}
        b = x_start.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=x_start.device).long()
        noise = torch.randn_like(x_start) if noise is None else noise

        target = noise if self.objective == "pred_noise" else x_start
        loss_weight = extract(self.loss_weight, t, target.shape)
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_t, t, model_kwargs=model_kwargs)

        mse = nn.functional.mse_loss(model_out, target, reduction="none")
        return (loss_weight * mse).mean()


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (
        v_end - v_start
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, clamp_min, 0.999)


def get_beta_schedule(beta_schedule, timesteps):
    if beta_schedule == "linear":
        beta_schedule_fn = linear_beta_schedule
    elif beta_schedule == "cosine":
        beta_schedule_fn = cosine_beta_schedule
    elif beta_schedule == "sigmoid":
        beta_schedule_fn = sigmoid_beta_schedule
    else:
        raise ValueError(f"unknown beta schedule {beta_schedule}")

    return beta_schedule_fn(timesteps)
