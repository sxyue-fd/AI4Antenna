# -*- coding: utf-8 -*-

import os
import torch


class TargetStandardizer:
    def __init__(self, y_mean, y_std, p_mean, p_std):
        self.y_mean = y_mean
        self.y_std = y_std
        self.p_mean = p_mean
        self.p_std = p_std

    def normalize_y(self, y):
        y_mean = self.y_mean.to(y.device)
        y_std = self.y_std.to(y.device)
        return (y - y_mean) / y_std

    def normalize_p(self, p):
        p_mean = self.p_mean.to(p.device)
        p_std = self.p_std.to(p.device)
        return (p - p_mean) / p_std

    def denormalize_y(self, y):
        y_mean = self.y_mean.to(y.device)
        y_std = self.y_std.to(y.device)
        return y * y_std + y_mean

    def denormalize_p(self, p):
        p_mean = self.p_mean.to(p.device)
        p_std = self.p_std.to(p.device)
        return p * p_std + p_mean

    def is_compatible(self, y_shape, p_shape):
        return (
            tuple(self.y_mean.shape) == tuple(y_shape)
            and tuple(self.y_std.shape) == tuple(y_shape)
            and tuple(self.p_mean.shape) == tuple(p_shape)
            and tuple(self.p_std.shape) == tuple(p_shape)
        )

    def validate_shapes(self, y_shape, p_shape):
        if not self.is_compatible(y_shape, p_shape):
            raise ValueError(
                "Standardizer target shape mismatch: "
                f"expected y={tuple(y_shape)}, pattern={tuple(p_shape)}, "
                f"got y_mean={tuple(self.y_mean.shape)}, p_mean={tuple(self.p_mean.shape)}. "
                "Recompute stats or use a checkpoint trained on this dataset."
            )

    def state_dict(self):
        return {
            "y_mean": self.y_mean.detach().cpu(),
            "y_std": self.y_std.detach().cpu(),
            "p_mean": self.p_mean.detach().cpu(),
            "p_std": self.p_std.detach().cpu(),
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        return cls(
            y_mean=state_dict["y_mean"].float(),
            y_std=state_dict["y_std"].float(),
            p_mean=state_dict["p_mean"].float(),
            p_std=state_dict["p_std"].float(),
        )

    def save(self, path, extra=None):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "standardizer_stats": self.state_dict(),
        }
        if extra is not None:
            payload["extra"] = extra
        torch.save(payload, path)

    @classmethod
    def load(cls, path):
        payload = torch.load(path, map_location="cpu")
        if "standardizer_stats" in payload:
            state_dict = payload["standardizer_stats"]
        else:
            state_dict = payload
        return cls.from_state_dict(state_dict)


def compute_stats(dataset, indices):
    y_sum = 0
    y_sq = 0
    p_sum = 0
    p_sq = 0
    count = 0

    for idx in indices:
        _, y, p, _ = dataset[idx]
        y = y.double()
        p = p.double()

        if count == 0:
            y_sum = torch.zeros_like(y)
            y_sq = torch.zeros_like(y)
            p_sum = torch.zeros_like(p)
            p_sq = torch.zeros_like(p)

        y_sum += y
        y_sq += y * y
        p_sum += p
        p_sq += p * p
        count += 1

    y_mean = y_sum / count
    p_mean = p_sum / count

    y_std = torch.sqrt(torch.clamp(y_sq / count - y_mean**2, min=1e-8))
    p_std = torch.sqrt(torch.clamp(p_sq / count - p_mean**2, min=1e-8))

    return TargetStandardizer(
        y_mean.float(), y_std.float(),
        p_mean.float(), p_std.float()
    )
