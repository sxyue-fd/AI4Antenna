# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import pprint
import subprocess
import sys
import time

import torch

_PARENT_PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PARENT_PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_PROJECT_DIR)

from configs.default_config import get_default_config, update_config_from_args
from datasets.task_datamodule import build_task_dataloaders
from engine.losses import build_loss
from engine.trainer import evaluate, save_checkpoint, train_one_epoch, append_csv
from models import CurrentToStructureUNet
from utils.common import ensure_dir, make_run_dir, save_json, set_seed


_PLOT_SCRIPT = os.path.join(os.path.dirname(__file__), "plot_logs.py")


def run_plot_process(kind, csv_path, output_path):
    result = subprocess.run(
        [
            sys.executable,
            _PLOT_SCRIPT,
            "--kind",
            kind,
            "--csv",
            csv_path,
            "--output",
            output_path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(message or f"Plot process exited with {result.returncode}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train current-to-structure U-Net")
    parser.add_argument("--h5_path", default=None, help="Preprocessed standardized HDF5 path")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="auto / cuda / cpu")
    parser.add_argument("--base_channels", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=None, help="Progress bar refresh interval in batches; default 1, 0 disables it")
    parser.add_argument("--iter_log_interval", type=int, default=None, help="Write train batch losses every N iterations; 0 disables iteration logging")
    parser.add_argument("--iter_val_interval", type=int, default=None, help="Run limited validation every N iterations; 0 disables iteration validation")
    parser.add_argument("--iter_val_batches", type=int, default=None, help="Number of validation batches used for each iteration validation")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = update_config_from_args(get_default_config(), args)

    run_dir = make_run_dir(cfg["paths"]["output_dir"], prefix="train")
    cfg["paths"]["run_dir"] = run_dir
    cfg["paths"]["checkpoint_dir"] = os.path.join(run_dir, "checkpoints")
    cfg["paths"]["log_dir"] = os.path.join(run_dir, "logs")
    ensure_dir(cfg["paths"]["checkpoint_dir"])
    ensure_dir(cfg["paths"]["log_dir"])

    set_seed(cfg["train"]["seed"])
    device_name = cfg["train"]["device"]
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print("Using device:", device)
    print("Configuration:")
    print(pprint.pformat(cfg, sort_dicts=False))
    save_json(os.path.join(run_dir, "train_config.json"), cfg)

    loaders, dataset_info = build_task_dataloaders(cfg)
    print("Dataset info:")
    print(pprint.pformat(dataset_info, sort_dicts=False))

    in_channels = int(dataset_info["input_shape"][0])
    model = CurrentToStructureUNet(
        in_channels=in_channels,
        base_channels=cfg["model"]["base_channels"],
        dropout=cfg["model"]["dropout"],
        target_size=cfg["model"]["target_size"],
    ).to(device)

    criterion = build_loss(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["train"]["epochs"],
        eta_min=cfg["scheduler"]["eta_min"],
    )

    use_amp = cfg["train"]["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_val_loss = float("inf")
    no_improve = 0

    log_csv = os.path.join(cfg["paths"]["log_dir"], "train_log.csv")
    iter_log_csv = os.path.join(cfg["paths"]["log_dir"], "iter_log.csv")
    loss_curve_path = os.path.join(cfg["paths"]["log_dir"], "loss_curve.png")
    iter_val_acc_curve_path = os.path.join(cfg["paths"]["log_dir"], "iter_val_accuracy_curve.png")
    best_checkpoint_path = os.path.join(cfg["paths"]["checkpoint_dir"], "best_model.pt")
    loss_curve_announced = False
    train_start_time = time.perf_counter()
    global_step = 0

    for epoch in range(cfg["train"]["epochs"]):
        epoch_start_time = time.perf_counter()
        train_metrics, global_step = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            scaler,
            device,
            use_amp,
            cfg["train"]["grad_clip_norm"],
            progress_label=f"Epoch {epoch + 1}/{cfg['train']['epochs']} train",
            log_interval=cfg["train"].get("progress_log_interval", 20),
            epoch=epoch + 1,
            global_step_start=global_step,
            iter_log_path=iter_log_csv,
            iter_log_interval=cfg["train"].get("iter_log_interval", 0),
            val_loader=loaders["val"],
            iter_val_interval=cfg["train"].get("iter_val_interval", 0),
            iter_val_batches=cfg["train"].get("iter_val_batches", 5),
        )
        val_metrics = evaluate(
            model,
            loaders["val"],
            criterion,
            device,
            use_amp=use_amp,
            progress_label=f"Epoch {epoch + 1}/{cfg['train']['epochs']} val  ",
            log_interval=cfg["train"].get("progress_log_interval", 20),
        )
        scheduler.step()
        epoch_time_sec = time.perf_counter() - epoch_start_time
        total_time_sec = time.perf_counter() - train_start_time

        row = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": epoch_time_sec,
            "total_time_sec": total_time_sec,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        append_csv(log_csv, row)
        try:
            run_plot_process("epoch", log_csv, loss_curve_path)
            if not loss_curve_announced:
                print("Loss curve will be updated at:", loss_curve_path)
                loss_curve_announced = True
        except Exception as exc:
            print(f"[warning] Failed to update loss curve: {exc}")
        print(row)
        print(
            f"Epoch {epoch + 1}/{cfg['train']['epochs']} time: "
            f"{epoch_time_sec:.2f}s, total training time: {total_time_sec:.2f}s"
        )

        val_loss = val_metrics["loss"]
        if val_loss < best_val_loss - cfg["train"]["early_stop_min_delta"]:
            best_val_loss = val_loss
            no_improve = 0
            save_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_loss,
                cfg,
                dataset_info,
            )
            print("Saved best model.")
        else:
            no_improve += 1

        if no_improve >= cfg["train"]["early_stop_patience"]:
            print(f"Early stopping at epoch {epoch + 1}.")
            break

    try:
        run_plot_process("iteration", iter_log_csv, iter_val_acc_curve_path)
        if os.path.isfile(iter_val_acc_curve_path):
            print("Iteration validation accuracy curve saved at:", iter_val_acc_curve_path)
    except Exception as exc:
        print(f"[warning] Failed to plot iteration validation accuracy curve: {exc}")

    if os.path.isfile(best_checkpoint_path):
        best_checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        print(f"Loaded best model for test evaluation: {best_checkpoint_path}")

    test_metrics = evaluate(
        model,
        loaders["test"],
        criterion,
        device,
        use_amp=False,
        progress_label="Test",
        log_interval=cfg["train"].get("progress_log_interval", 20),
    )
    print("Test metrics:", test_metrics)
    save_json(os.path.join(run_dir, "test_metrics.json"), test_metrics)
    save_checkpoint(
        os.path.join(cfg["paths"]["checkpoint_dir"], "last_model.pt"),
        model,
        optimizer,
        scheduler,
        epoch,
        best_val_loss,
        cfg,
        dataset_info,
    )
    print("Finished. Run directory:", run_dir)


if __name__ == "__main__":
    main()
