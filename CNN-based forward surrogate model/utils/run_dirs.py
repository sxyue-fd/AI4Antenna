# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from datetime import datetime


def make_run_id(prefix: str = "run") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def infer_run_dir_from_checkpoint_path(checkpoint_path: str) -> str | None:
    if not checkpoint_path:
        return None

    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    run_dir = os.path.dirname(ckpt_dir)
    run_id = os.path.basename(run_dir)
    if os.path.basename(ckpt_dir) == "checkpoints" and run_id.startswith("train_"):
        return run_dir
    return None


def checkpoint_run_info(checkpoint: dict, checkpoint_path: str | None = None) -> tuple[str | None, str | None]:
    run_id = checkpoint.get("run_id")
    run_dir = checkpoint.get("run_dir")

    if run_dir is None and "config" in checkpoint:
        paths = checkpoint["config"].get("paths", {})
        run_dir = paths.get("run_dir") or paths.get("output_dir")
        run_id = run_id or paths.get("run_id")

    if run_dir is None and checkpoint_path is not None:
        run_dir = infer_run_dir_from_checkpoint_path(checkpoint_path)

    if run_id is None and run_dir is not None:
        run_id = os.path.basename(os.path.normpath(run_dir))

    return run_id, run_dir


def configure_train_run_dirs(cfg: dict, resume_checkpoint: dict | None = None, resume_path: str | None = None) -> dict:
    output_root = cfg["paths"]["output_dir"]

    if resume_checkpoint is not None:
        run_id, run_dir = checkpoint_run_info(resume_checkpoint, resume_path)
        if run_dir is None:
            run_id = make_run_id("train")
            run_dir = os.path.join(output_root, "train", run_id)
    else:
        run_id = make_run_id("train")
        run_dir = os.path.join(output_root, "train", run_id)

    cfg["paths"]["output_root"] = output_root
    cfg["paths"]["run_id"] = run_id
    cfg["paths"]["run_dir"] = run_dir
    cfg["paths"]["output_dir"] = run_dir
    cfg["paths"]["checkpoint_dir"] = os.path.join(run_dir, "checkpoints")
    cfg["paths"]["log_dir"] = os.path.join(run_dir, "logs")
    return cfg


def configure_eval_output_dir(cfg: dict, kind: str, checkpoint: dict, checkpoint_path: str, explicit_output_dir: str | None = None) -> str:
    if explicit_output_dir:
        return explicit_output_dir

    output_root = cfg["paths"]["output_dir"]
    run_id, _ = checkpoint_run_info(checkpoint, checkpoint_path)
    if run_id is None:
        run_id = make_run_id(kind)

    return os.path.join(output_root, kind, run_id)


def find_latest_train_checkpoint(output_root: str, checkpoint_name: str = "best_model.pt") -> str | None:
    train_root = os.path.join(output_root, "train")
    if not os.path.isdir(train_root):
        return None

    candidates = []
    for name in os.listdir(train_root):
        path = os.path.join(train_root, name, "checkpoints", checkpoint_name)
        if os.path.isfile(path):
            candidates.append(path)

    if not candidates:
        return None

    candidates.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return candidates[0]
