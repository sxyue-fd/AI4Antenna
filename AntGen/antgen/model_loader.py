"""Load the two sibling model projects without colliding on ``models``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import torch


def _load_package(alias: str, package_dir: Path) -> ModuleType:
    """Import a package under a private alias so relative imports still work."""
    init_path = package_dir / "__init__.py"
    if not init_path.is_file():
        raise FileNotFoundError(init_path)

    existing = sys.modules.get(alias)
    if existing is not None:
        return existing

    spec = importlib.util.spec_from_file_location(
        alias,
        init_path,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build an import spec for {package_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def latest_checkpoint(output_dir: Path, layout: str) -> Path:
    """Return the newest training ``best_model.pt`` for a known project layout."""
    if layout in {"diffusion", "cnn_surrogate"}:
        candidates = list(output_dir.glob("train/train_*/checkpoints/best_model.pt"))
    elif layout == "structure":
        candidates = list(output_dir.glob("train_*/checkpoints/best_model.pt"))
    else:
        raise ValueError(f"Unknown checkpoint layout: {layout}")
    if not candidates:
        raise FileNotFoundError(f"No best_model.pt found under {output_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_checkpoint(path: Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # Compatibility with older PyTorch releases.
        return torch.load(path, map_location=device)


def load_diffusion_model(workspace: Path, checkpoint_path: Path, device: torch.device):
    package = _load_package(
        "_antgen_current_diffusion_models",
        workspace / "Current diffusion model" / "models",
    )
    checkpoint = load_checkpoint(checkpoint_path, device="cpu")
    for key in ("config", "dataset_info", "model_state_dict"):
        if key not in checkpoint:
            raise KeyError(f"Diffusion checkpoint is missing {key!r}: {checkpoint_path}")
    model = package.build_current_diffusion(checkpoint["config"], checkpoint["dataset_info"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint


def load_structure_model(workspace: Path, checkpoint_path: Path, device: torch.device):
    package = _load_package(
        "_antgen_current_to_structure_models",
        workspace / "Current to structure UNet" / "models",
    )
    checkpoint = load_checkpoint(checkpoint_path, device="cpu")
    for key in ("config", "dataset_info", "model_state_dict"):
        if key not in checkpoint:
            raise KeyError(f"Structure checkpoint is missing {key!r}: {checkpoint_path}")

    cfg = checkpoint["config"]
    dataset_info = checkpoint["dataset_info"]
    model = package.CurrentToStructureUNet(
        in_channels=int(dataset_info["input_shape"][0]),
        base_channels=int(cfg["model"]["base_channels"]),
        dropout=0.0,
        target_size=int(cfg["model"]["target_size"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint


def load_cnn_surrogate_model(workspace: Path, checkpoint_path: Path, device: torch.device):
    package = _load_package(
        "_antgen_cnn_forward_surrogate_models",
        workspace / "CNN-based forward surrogate model" / "models",
    )
    checkpoint = load_checkpoint(checkpoint_path, device="cpu")
    for key in ("config", "dataset_info", "model_state_dict", "standardizer_stats"):
        if key not in checkpoint:
            raise KeyError(f"CNN surrogate checkpoint is missing {key!r}: {checkpoint_path}")
    model = package.build_forward_surrogate(checkpoint["config"], checkpoint["dataset_info"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint
