# -*- coding: utf-8 -*-

import torch


def save_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    epoch=None,
    best_val_loss=None,
    extra=None,
):
    ckpt = {
        "model_state_dict": model.state_dict(),
    }

    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()

    if epoch is not None:
        ckpt["epoch"] = epoch

    if best_val_loss is not None:
        ckpt["best_val_loss"] = best_val_loss

    if extra is not None:
        ckpt.update(extra)

    torch.save(ckpt, path)


def load_checkpoint(path, model, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt