# -*- coding: utf-8 -*-

import csv
import os
from datetime import datetime

import torch

from engine.evaluator import sample_and_evaluate
from utils.checkpoint import save_checkpoint


def cycle(dl):
    while True:
        for data in dl:
            yield data


def append_csv(log_path, row):
    write_header = not os.path.exists(log_path)
    fieldnames = list(row.keys())

    if not write_header:
        with open(log_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            existing_header = next(reader, [])
        if existing_header != fieldnames:
            backup_path = log_path + ".bak"
            os.replace(log_path, backup_path)
            write_header = True

    with open(log_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def prepare_train_log(log_path, start_step):
    if start_step != 0 or not os.path.exists(log_path):
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{log_path}.{timestamp}.bak"
    os.replace(log_path, backup_path)


def _empty_log_row(step, lr, train_loss):
    return {
        "step": step,
        "lr": lr,
        "train_loss": train_loss,
        "sample_mae": None,
        "sample_y_mae": None,
        "sample_pattern_mae": None,
        "sample_mse": None,
        "sample_y_mse": None,
        "sample_pattern_mse": None,
    }


def _row_with_sample_metrics(row, metrics):
    row = dict(row)
    row.update(
        {
            "sample_mae": metrics["mae"],
            "sample_y_mae": metrics["y_mae"],
            "sample_pattern_mae": metrics["pattern_mae"],
            "sample_mse": metrics["mse"],
            "sample_y_mse": metrics["y_mse"],
            "sample_pattern_mse": metrics["pattern_mse"],
        }
    )
    return row


def _checkpoint_extra(cfg, dataset_info, standardizer, surrogate_checkpoint, scaler):
    return {
        "config": cfg,
        "dataset_info": dataset_info,
        "standardizer_stats": standardizer.state_dict(),
        "surrogate_dataset_info": surrogate_checkpoint.get("dataset_info"),
        "surrogate_checkpoint": cfg.get("surrogate", {}).get("checkpoint"),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "run_id": cfg["paths"].get("run_id"),
        "run_dir": cfg["paths"].get("run_dir"),
        "output_root": cfg["paths"].get("output_root"),
    }


def train_model(
    diffusion_model,
    surrogate_model,
    optimizer,
    scheduler,
    dataloaders,
    standardizer,
    device,
    cfg,
    scaler,
    logger=None,
    start_step=0,
    best_sample_mae=float("inf"),
    dataset_info=None,
    surrogate_checkpoint=None,
):
    train_num_steps = cfg["train"]["train_num_steps"]
    use_amp = cfg["train"].get("amp", True) and (device.type == "cuda")
    grad_clip = cfg["train"].get("grad_clip", 1.0)
    log_every = cfg["train"].get("log_every", 100)
    sample_every = cfg["train"].get("sample_every", 1000)
    save_every = cfg["train"].get("save_every", 10000)

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    train_iter = cycle(train_loader)

    ckpt_dir = cfg["paths"]["checkpoint_dir"]
    log_dir = cfg["paths"]["log_dir"]
    sample_dir = os.path.join(cfg["paths"]["output_dir"], "samples")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    log_csv_path = os.path.join(log_dir, "train_log.csv")
    prepare_train_log(log_csv_path, start_step)

    amp_device = "cuda" if device.type == "cuda" else "cpu"
    step = start_step
    last_loss = None

    while step < train_num_steps:
        current, y, p, _ = next(train_iter)
        current = current.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        p = p.to(device, non_blocking=True)
        model_kwargs = {"s11": y, "pattern": p}

        diffusion_model.train()
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
            loss = diffusion_model.p_losses(current, model_kwargs=model_kwargs)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        step += 1
        last_loss = float(loss.detach().item())
        current_lr = optimizer.param_groups[0]["lr"]

        should_log = (step % log_every == 0) or (step == 1) or (step >= train_num_steps)
        should_sample = (sample_every is not None and sample_every > 0 and step % sample_every == 0)

        row = _empty_log_row(step=step, lr=current_lr, train_loss=last_loss)

        if should_sample:
            sample_path = os.path.join(sample_dir, f"sample_step_{step}.pt")
            metrics = sample_and_evaluate(
                diffusion_model=diffusion_model,
                surrogate_model=surrogate_model,
                loader=val_loader,
                standardizer=standardizer,
                device=device,
                cfg=cfg,
                num_samples=cfg.get("sample", {}).get("num_samples", 16),
                save_path=sample_path,
                logger=logger,
                split_name="val",
            )
            row = _row_with_sample_metrics(row, metrics)

            if metrics["mae"] < best_sample_mae:
                best_sample_mae = metrics["mae"]
                save_checkpoint(
                    path=os.path.join(ckpt_dir, "best_model.pt"),
                    model=diffusion_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    best_sample_mae=best_sample_mae,
                    extra=_checkpoint_extra(cfg, dataset_info, standardizer, surrogate_checkpoint, scaler),
                )
                if logger:
                    logger.info("Saved best diffusion model at step %d", step)

        if should_log or should_sample:
            append_csv(log_csv_path, row)
            if logger:
                logger.info("[train] step=%d loss=%.6f lr=%.6g", step, last_loss, current_lr)

        if save_every is not None and save_every > 0 and step % save_every == 0:
            save_checkpoint(
                path=os.path.join(ckpt_dir, f"model_step_{step}.pt"),
                model=diffusion_model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                best_sample_mae=best_sample_mae,
                extra=_checkpoint_extra(cfg, dataset_info, standardizer, surrogate_checkpoint, scaler),
            )
            if logger:
                logger.info("Saved periodic checkpoint at step %d", step)

    return {
        "last_step": step,
        "last_loss": last_loss,
        "best_sample_mae": best_sample_mae,
    }
