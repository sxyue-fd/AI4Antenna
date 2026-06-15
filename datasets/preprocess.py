# -*- coding: utf-8 -*-

import os
import torch


class TargetStandardizer:
    def __init__(
        self,
        y_mean,
        y_std,
        p_mean,
        p_std,
        x_mean=None,
        x_std=None,
        x_std_min=0.02,
        x_clip=20.0,
    ):
        self.y_mean = y_mean
        self.y_std = y_std
        self.p_mean = p_mean
        self.p_std = p_std
        self.x_mean = x_mean
        self.x_std = x_std
        self.x_std_min = x_std_min
        self.x_clip = x_clip

    def normalize_x(self, x):
        if self.x_mean is None or self.x_std is None:
            return x
        x_mean = self.x_mean.to(x.device)
        x_std = self.x_std.to(x.device)
        if self.x_std_min is not None:
            x_std = torch.clamp(x_std, min=float(self.x_std_min))
        x = (x - x_mean) / x_std
        if self.x_clip is not None:
            x = torch.clamp(x, min=-float(self.x_clip), max=float(self.x_clip))
        return x

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

    def is_compatible(self, y_shape, p_shape, x_shape=None):
        target_ok = (
            tuple(self.y_mean.shape) == tuple(y_shape)
            and tuple(self.y_std.shape) == tuple(y_shape)
            and tuple(self.p_mean.shape) == tuple(p_shape)
            and tuple(self.p_std.shape) == tuple(p_shape)
        )
        if not target_ok:
            return False
        if x_shape is None:
            return True
        if self.x_mean is None or self.x_std is None:
            return False
        return (
            tuple(self.x_mean.shape) == tuple(x_shape)
            and tuple(self.x_std.shape) == tuple(x_shape)
        )

    def validate_shapes(self, y_shape, p_shape, x_shape=None):
        if not self.is_compatible(y_shape, p_shape, x_shape=x_shape):
            raise ValueError(
                "Standardizer shape mismatch: "
                f"expected x={None if x_shape is None else tuple(x_shape)}, "
                f"y={tuple(y_shape)}, pattern={tuple(p_shape)}, "
                f"got x_mean={None if self.x_mean is None else tuple(self.x_mean.shape)}, "
                f"y_mean={tuple(self.y_mean.shape)}, p_mean={tuple(self.p_mean.shape)}. "
                "Recompute stats or use a checkpoint trained on this dataset."
            )

    def state_dict(self):
        state = {
            "y_mean": self.y_mean.detach().cpu(),
            "y_std": self.y_std.detach().cpu(),
            "p_mean": self.p_mean.detach().cpu(),
            "p_std": self.p_std.detach().cpu(),
        }
        if self.x_mean is not None and self.x_std is not None:
            state["x_mean"] = self.x_mean.detach().cpu()
            state["x_std"] = self.x_std.detach().cpu()
            state["x_std_min"] = self.x_std_min
            state["x_clip"] = self.x_clip
        return state

    @classmethod
    def from_state_dict(cls, state_dict):
        return cls(
            y_mean=state_dict["y_mean"].float(),
            y_std=state_dict["y_std"].float(),
            p_mean=state_dict["p_mean"].float(),
            p_std=state_dict["p_std"].float(),
            x_mean=state_dict.get("x_mean", None).float() if state_dict.get("x_mean", None) is not None else None,
            x_std=state_dict.get("x_std", None).float() if state_dict.get("x_std", None) is not None else None,
            x_std_min=state_dict.get("x_std_min", 0.02),
            x_clip=state_dict.get("x_clip", 20.0),
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


def compute_stats(
    dataset,
    indices,
    normalize_input=False,
    input_std_min=0.02,
    input_clip=20.0,
):
    x_sum = 0
    x_sq = 0
    y_sum = 0
    y_sq = 0
    p_sum = 0
    p_sq = 0
    count = 0

    for idx in indices:
        x, y, p, _ = dataset[idx]
        x = x.double()
        y = y.double()
        p = p.double()

        if count == 0:
            if normalize_input:
                x_sum = torch.zeros_like(x)
                x_sq = torch.zeros_like(x)
            y_sum = torch.zeros_like(y)
            y_sq = torch.zeros_like(y)
            p_sum = torch.zeros_like(p)
            p_sq = torch.zeros_like(p)

        if normalize_input:
            x_sum += x
            x_sq += x * x
        y_sum += y
        y_sq += y * y
        p_sum += p
        p_sq += p * p
        count += 1

    y_mean = y_sum / count
    p_mean = p_sum / count

    y_std = torch.sqrt(torch.clamp(y_sq / count - y_mean**2, min=1e-8))
    p_std = torch.sqrt(torch.clamp(p_sq / count - p_mean**2, min=1e-8))

    x_mean = None
    x_std = None
    if normalize_input:
        x_mean = x_sum / count
        x_std = torch.sqrt(torch.clamp(x_sq / count - x_mean**2, min=1e-8))

    return TargetStandardizer(
        y_mean.float(), y_std.float(),
        p_mean.float(), p_std.float(),
        x_mean.float() if x_mean is not None else None,
        x_std.float() if x_std is not None else None,
        x_std_min=input_std_min,
        x_clip=input_clip,
    )
