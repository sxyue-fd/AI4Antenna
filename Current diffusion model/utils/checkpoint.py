# -*- coding: utf-8 -*-

import torch


def save_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    step=None,
    best_sample_mae=None,
    extra=None,
):
    ckpt = {
        "model_state_dict": model.state_dict(),
    }

    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()

    if step is not None:
        ckpt["step"] = step

    if best_sample_mae is not None:
        ckpt["best_sample_mae"] = best_sample_mae

    if extra is not None:
        ckpt.update(extra)

    torch.save(ckpt, path)


def load_checkpoint(path, model, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt

