# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib
import importlib.util
import os
import sys

import torch


_PACKAGE_ALIAS = "_current_forward_surrogate_models"


def _load_package_alias(package_name: str, package_dir: str):
    if package_name in sys.modules:
        return sys.modules[package_name]

    init_path = os.path.join(package_dir, "__init__.py")
    if not os.path.isfile(init_path):
        raise FileNotFoundError(init_path)

    spec = importlib.util.spec_from_file_location(
        package_name,
        init_path,
        submodule_search_locations=[package_dir],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return module


def load_forward_surrogate(project_dir: str, checkpoint_path: str, device):
    if not os.path.isdir(project_dir):
        raise FileNotFoundError(f"Forward surrogate project not found: {project_dir}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Forward surrogate checkpoint not found: {checkpoint_path}")

    models_dir = os.path.join(project_dir, "models")
    _load_package_alias(_PACKAGE_ALIAS, models_dir)
    module = importlib.import_module(f"{_PACKAGE_ALIAS}.forward_surrogate_net")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint.get("config", {})
    dataset_info = checkpoint.get("dataset_info")
    if dataset_info is None:
        raise ValueError("Forward surrogate checkpoint is missing dataset_info")

    model = module.build_forward_surrogate(cfg=cfg, dataset_info=dataset_info)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad_(False)

    return model, checkpoint

