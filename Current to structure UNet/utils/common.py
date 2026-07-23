# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import random
from datetime import datetime

import numpy as np
import torch


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(path, obj):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def make_run_dir(output_root, prefix="train"):
    run_id = datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S")
    return os.path.join(output_root, run_id)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
