# -*- coding: utf-8 -*-

import numpy as np


def build_split_indices(num_samples, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")

    ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    if np.any(ratios < 0):
        raise ValueError("split ratios must be non-negative")

    ratio_sum = ratios.sum()
    if ratio_sum <= 0:
        raise ValueError("at least one split ratio must be positive")

    ratios = ratios / ratio_sum

    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples).astype(np.int64)

    if num_samples == 1:
        return {
            "train": indices,
            "val": indices[:0],
            "test": indices[:0],
        }

    if num_samples == 2:
        return {
            "train": indices[:1],
            "val": indices[:0],
            "test": indices[1:],
        }

    n_train = int(num_samples * ratios[0])
    n_val = int(num_samples * ratios[1])

    if num_samples >= 3:
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        n_test = num_samples - n_train - n_val

        if n_test < 1:
            shortage = 1 - n_test
            if n_train >= n_val and n_train > 1:
                n_train -= shortage
            elif n_val > 1:
                n_val -= shortage
            else:
                n_train -= shortage
            n_test = num_samples - n_train - n_val
    else:
        n_test = num_samples - n_train - n_val

    train_end = n_train
    val_end = n_train + n_val

    return {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:],
    }
