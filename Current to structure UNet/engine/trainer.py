# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import os

import torch

from .metrics import compute_metrics

from tqdm import tqdm


def move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def append_csv(path, row):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def aggregate(total, values, batch_size):
    for key, value in values.items():
        if isinstance(value, torch.Tensor):
            value = value.item()
        total[key] = total.get(key, 0.0) + float(value) * batch_size


def average(total, count):
    return {key: value / max(count, 1) for key, value in total.items()}


def _to_float_dict(values):
    out = {}
    for key, value in values.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().item()
        out[key] = float(value)
    return out


def _make_progress_iter(loader, label, log_interval):
    if label and log_interval > 0:
        total = len(loader) if hasattr(loader, "__len__") else None
        return tqdm(
            loader,
            desc=label,
            total=total,
            miniters=max(1, int(log_interval)),
            dynamic_ncols=True,
            leave=True,
        )
    return loader


def _update_tqdm_postfix(progress_iter, totals, count):
    if isinstance(progress_iter, tqdm):
        avg_loss = totals.get("loss", 0.0) / max(count, 1)
        progress_iter.set_postfix(loss=f"{avg_loss:.6g}")


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    use_amp,
    grad_clip_norm,
    progress_label=None,
    log_interval=20,
    epoch=0,
    global_step_start=0,
    iter_log_path=None,
    iter_log_interval=0,
    val_loader=None,
    iter_val_interval=0,
    iter_val_batches=5,
):
    model.train()
    totals = {}
    count = 0
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    progress_iter = _make_progress_iter(loader, progress_label, log_interval)
    global_step = int(global_step_start)

    for batch_idx, batch in enumerate(progress_iter, start=1):
        global_step += 1
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
            outputs = model(batch["current"])
            losses = criterion(outputs, batch)

        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(optimizer)
        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        bs = batch["current"].size(0)
        aggregate(totals, losses, bs)
        count += bs
        _update_tqdm_postfix(progress_iter, totals, count)

        should_log = iter_log_path and iter_log_interval > 0 and global_step % int(iter_log_interval) == 0
        should_val = (
            iter_log_path
            and val_loader is not None
            and iter_val_interval > 0
            and global_step % int(iter_val_interval) == 0
        )
        if should_log or should_val:
            train_values = _to_float_dict(losses)
            val_metrics = {}
            if should_val:
                val_metrics = evaluate_limited(
                    model,
                    val_loader,
                    criterion,
                    device,
                    use_amp=use_amp,
                    max_batches=iter_val_batches,
                )
                model.train()

            row = {
                "epoch": epoch,
                "iter": batch_idx,
                "global_step": global_step,
                "lr": optimizer.param_groups[0]["lr"],
                "batch_size": bs,
                "train_loss": train_values.get("loss", ""),
                "train_metal_bce": train_values.get("metal_bce", ""),
                "train_metal_dice": train_values.get("metal_dice", ""),
                "train_feed_ce": train_values.get("feed_ce", ""),
                "val_loss": val_metrics.get("loss", ""),
                "val_metal_bce": val_metrics.get("metal_bce", ""),
                "val_metal_dice": val_metrics.get("metal_dice", ""),
                "val_feed_ce": val_metrics.get("feed_ce", ""),
                "val_metal_iou": val_metrics.get("metal_iou", ""),
                "val_metal_acc": val_metrics.get("metal_acc", ""),
                "val_feed_acc": val_metrics.get("feed_acc", ""),
                "val_feed_on_metal_rate": val_metrics.get("feed_on_metal_rate", ""),
            }
            append_csv(iter_log_path, row)
            if isinstance(progress_iter, tqdm):
                postfix = {"loss": f"{train_values.get('loss', 0.0):.6g}"}
                if val_metrics:
                    postfix.update({
                        "val_metal_acc": f"{val_metrics.get('metal_acc', 0.0):.4f}",
                        "val_feed_acc": f"{val_metrics.get('feed_acc', 0.0):.4f}",
                    })
                progress_iter.set_postfix(**postfix)

    return average(totals, count), global_step


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=False, progress_label=None, log_interval=20):
    model.eval()
    totals = {}
    count = 0
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    progress_iter = _make_progress_iter(loader, progress_label, log_interval)

    for batch in progress_iter:
        batch = move_batch(batch, device)
        with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
            outputs = model(batch["current"])
            losses = criterion(outputs, batch)
        metrics = compute_metrics(outputs, batch)

        bs = batch["current"].size(0)
        aggregate(totals, losses, bs)
        aggregate(totals, metrics, bs)
        count += bs
        _update_tqdm_postfix(progress_iter, totals, count)

    return average(totals, count)


@torch.no_grad()
def evaluate_limited(model, loader, criterion, device, use_amp=False, max_batches=5):
    model.eval()
    totals = {}
    count = 0
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    max_batches = max(1, int(max_batches))

    for batch_idx, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
            outputs = model(batch["current"])
            losses = criterion(outputs, batch)
        metrics = compute_metrics(outputs, batch)

        bs = batch["current"].size(0)
        aggregate(totals, losses, bs)
        aggregate(totals, metrics, bs)
        count += bs
        if batch_idx >= max_batches:
            break

    return average(totals, count)


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_loss, cfg, dataset_info):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "config": cfg,
            "dataset_info": dataset_info,
        },
        path,
    )
