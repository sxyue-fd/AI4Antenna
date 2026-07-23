# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import sys

import torch

_PARENT_PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PARENT_PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_PROJECT_DIR)

from configs.default_config import get_default_config, update_config_from_args
from datasets.datamodule import load_standardizer_for_dataset
from datasets.task_dataset import AntennaTaskDataset
from models import CurrentToStructureUNet
from utils.visualize import save_structure_prediction


def parse_args():
    parser = argparse.ArgumentParser(description="Infer one current-to-structure sample")
    parser.add_argument("--h5_path", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--index", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.2,
        help="Gaussian noise std added to the model input current tensor; 0 disables noise.",
    )
    parser.add_argument(
        "--noise_seed",
        type=int,
        default=None,
        help="Random seed for Gaussian input noise.",
    )
    parser.add_argument(
        "--no_noise",
        action="store_true",
        help="Disable Gaussian input noise and infer from the original current tensor.",
    )
    return parser.parse_args()


def find_latest_best_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None

    candidates = []
    for name in os.listdir(output_dir):
        run_dir = os.path.join(output_dir, name)
        ckpt_path = os.path.join(run_dir, "checkpoints", "best_model.pt")
        if name.startswith("train_") and os.path.isfile(ckpt_path):
            candidates.append(ckpt_path)

    if not candidates:
        return None

    candidates.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return candidates[0]


@torch.no_grad()
def main():
    args = parse_args()
    cfg_defaults = update_config_from_args(get_default_config(), args)
    h5_path = args.h5_path or cfg_defaults["paths"]["h5_path"]
    save_dir = args.save_dir or os.path.join(cfg_defaults["paths"]["output_dir"], "infer")
    device_name = args.device or cfg_defaults["train"]["device"]

    checkpoint_path = args.checkpoint or find_latest_best_checkpoint(cfg_defaults["paths"]["output_dir"])
    if checkpoint_path is None:
        raise FileNotFoundError(
            "No checkpoint was provided and no outputs/train_*/checkpoints/best_model.pt was found."
        )
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"HDF5 dataset not found: {h5_path}")
    if args.noise_std < 0:
        raise ValueError("--noise_std must be >= 0")
    effective_noise_std = 0.0 if args.no_noise else float(args.noise_std)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    dataset_info = checkpoint["dataset_info"]
    cfg = checkpoint["config"]

    task_cfg = cfg.get("task", {})
    ds = AntennaTaskDataset(
        h5_path=h5_path,
        task_name=task_cfg.get("name", "current_to_structure"),
        input_key=task_cfg.get("input_key", cfg["data"].get("input_key", "current")),
        x_key=task_cfg.get("x_key", cfg["data"].get("x_key", "X")),
        return_raw=False,
    )
    standardizer, standardizer_path = load_standardizer_for_dataset(h5_path, cfg)
    if args.index < 0 or args.index >= len(ds):
        raise IndexError(f"index={args.index} is out of range [0, {len(ds) - 1}]")

    sample = ds[args.index]
    current = sample["current"].unsqueeze(0).to(device)
    if effective_noise_std > 0:
        generator = None
        if args.noise_seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(args.noise_seed)
        noise = torch.randn(
            current.shape,
            dtype=current.dtype,
            device=current.device,
            generator=generator,
        ) * effective_noise_std
        current = current + noise

    model = CurrentToStructureUNet(
        in_channels=int(dataset_info["input_shape"][0]),
        base_channels=cfg["model"]["base_channels"],
        dropout=0.0,
        target_size=cfg["model"]["target_size"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    outputs = model(current)
    metal_prob = torch.sigmoid(outputs["metal_logits"])[0]
    feed_logits = outputs["feed_logits"][0]
    feed_pred = int(torch.argmax(feed_logits.flatten()).item())

    h, w = metal_prob.shape[-2:]
    print("checkpoint:", checkpoint_path)
    print("h5_path:", h5_path)
    print("standardizer:", standardizer_path)
    print("sample index:", args.index)
    print("no-noise mode:", args.no_noise)
    print("input noise std:", effective_noise_std)
    if effective_noise_std > 0 and args.noise_seed is not None:
        print("input noise seed:", args.noise_seed)
    print("pred feed row/col:", feed_pred // w, feed_pred % w)
    print("true feed row/col:", int(sample["feed_index"]) // w, int(sample["feed_index"]) % w)

    os.makedirs(save_dir, exist_ok=True)
    suffix = "_no_noise" if args.no_noise else ""
    if effective_noise_std > 0:
        noise_tag = f"{effective_noise_std:g}".replace(".", "p").replace("-", "m")
        suffix = f"_noise_std_{noise_tag}"
        if args.noise_seed is not None:
            suffix += f"_seed_{args.noise_seed}"
    fig_path = os.path.join(save_dir, f"sample_{args.index}{suffix}_structure.png")
    save_structure_prediction(
        fig_path,
        sample["metal"],
        metal_prob.cpu(),
        int(sample["feed_index"]),
        feed_logits.cpu(),
        threshold=args.threshold,
        input_current=current.cpu(),
        standardizer=standardizer,
        stored_current_shape=ds.input_dataset.storage_x_shape,
        pattern_metadata=ds.pattern_metadata,
    )
    print("saved:", fig_path)


if __name__ == "__main__":
    main()
