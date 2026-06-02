# -*- coding: utf-8 -*-

import numpy as np
import os
from scipy.io import loadmat
import h5py


def _flatten(x):
    return np.asarray(x).reshape(-1)


def load_split_indices(split_mat_path):
    if not os.path.isfile(split_mat_path):
        raise FileNotFoundError(split_mat_path)

    try:
        data = loadmat(split_mat_path)
        train_idx = _flatten(data["train_idx"])
        val_idx = _flatten(data["val_idx"])
        test_idx = _flatten(data["test_idx"])
    except NotImplementedError:
        # MATLAB v7.3
        with h5py.File(split_mat_path, "r") as f:
            train_idx = np.array(f["train_idx"]).reshape(-1)
            val_idx = np.array(f["val_idx"]).reshape(-1)
            test_idx = np.array(f["test_idx"]).reshape(-1)

    def to_zero_based(idx):
        idx = idx.astype(np.int64)
        if idx.min() >= 1:
            idx = idx - 1
        return idx

    return {
        "train": to_zero_based(train_idx),
        "val": to_zero_based(val_idx),
        "test": to_zero_based(test_idx),
    }