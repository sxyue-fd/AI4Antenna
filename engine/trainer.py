# -*- coding: utf-8 -*-

import os
import csv
import torch

from engine.evaluator import evaluate_model
from utils.checkpoint import save_checkpoint


def append_csv(log_path, row):
    write_header = not os.path.exists(log_path)

    with open(log_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler,
    use_amp=False,
):
    model.train()

    total_loss = 0.0
    total_y_loss = 0.0
    total_p_loss = 0.0
    count = 0

    amp_device = "cuda" if device.type == "cuda" else "cpu"

    for x, y, p, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        p = p.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
            y_pred, p_pred = model(x)
            loss, y_loss, p_loss = criterion(
                y_pred,
                y,
                p_pred,
                p,
                return_components=True,
            )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_y_loss += y_loss.item() * bs
        total_p_loss += p_loss.item() * bs
        count += bs

    return {
        "loss": total_loss / max(count, 1),
        "y_loss": total_y_loss / max(count, 1),
        "p_loss": total_p_loss / max(count, 1),
    }


def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    dataloaders,
    standardizer,
    device,
    cfg,
    scaler,
    logger=None,
    start_epoch=0,
    best_val_loss=float("inf"),
    dataset_info=None,
):
    epochs = cfg["train"]["epochs"]
    use_amp = cfg["train"].get("amp", True) and (device.type == "cuda")

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    ckpt_dir = cfg["paths"]["checkpoint_dir"]
    log_dir = cfg["paths"]["log_dir"]

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_csv_path = os.path.join(log_dir, "train_log.csv")

    patience = cfg["train"].get("early_stop_patience", 10)
    no_improve = 0
    last_epoch = start_epoch - 1

    for epoch in range(start_epoch, epochs):
        logger.info(f"Epoch [{epoch + 1}/{epochs}]")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
        )
        logger.info(f"[train] {train_metrics}")

        val_metrics = evaluate_model(
            model=model,
            loader=val_loader,
            criterion=criterion,
            standardizer=standardizer,
            device=device,
            cfg=cfg,
            logger=logger,
            split_name="val",
            use_amp=use_amp,
        )

        scheduler.step(val_metrics["loss"])

        current_lr = optimizer.param_groups[0]["lr"]
        append_csv(
            log_csv_path,
            {
                "epoch": epoch + 1,
                "lr": current_lr,
                "train_loss": train_metrics["loss"],
                "train_y_loss": train_metrics["y_loss"],
                "train_p_loss": train_metrics["p_loss"],
                "val_loss": val_metrics["loss"],
                "val_y_loss": val_metrics["y_loss"],
                "val_p_loss": val_metrics["p_loss"],
                "val_y_mae": val_metrics.get("y_mae", None),
                "val_pattern_mae": val_metrics.get("pattern_mae", None),
            },
        )

        min_delta = cfg["train"].get("early_stop_min_delta", 1e-5)

        if val_metrics["loss"] < best_val_loss - min_delta:
            best_val_loss = val_metrics["loss"]
            no_improve = 0

            save_checkpoint(
                path=os.path.join(ckpt_dir, "best_model.pt"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_loss=best_val_loss,
                extra={
                    "config": cfg,
                    "dataset_info": dataset_info,
                    "scaler_state_dict": scaler.state_dict(),
                    "standardizer_stats": {
                        "y_mean": standardizer.y_mean.detach().cpu(),
                        "y_std": standardizer.y_std.detach().cpu(),
                        "p_mean": standardizer.p_mean.detach().cpu(),
                        "p_std": standardizer.p_std.detach().cpu(),
                    },
                },
            )
            logger.info("Saved best model")
        else:
            no_improve += 1

        last_epoch = epoch

        if no_improve >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    return {
        "best_val_loss": best_val_loss,
        "last_epoch": last_epoch,
    }