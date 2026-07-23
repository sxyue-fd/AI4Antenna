# -*- coding: utf-8 -*-
from __future__ import annotations

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.h5_dataset import H5AntennaDataset


RESPONSE_TASKS = {"x_to_response", "current_to_response"}
STRUCTURE_TASKS = {"current_to_structure"}
SUPPORTED_TASKS = RESPONSE_TASKS | STRUCTURE_TASKS


def structure_targets_from_x(x, feed_rc=None):
    """Convert encoded /X to metal topology and feed labels.

    Supported forms:
    - raw single-channel X: 0=air, 1=metal, 2=feed
    - standardized single-channel X: -1=air, 1=metal, 3=feed
    - two-channel X: channel 0 metal, channel 1 feed heatmap
    """
    x = np.asarray(x, dtype=np.float32)

    if x.ndim == 3 and x.shape[0] == 1:
        encoded = x[0]
        metal = encoded > 0.5
        feed = encoded > 1.5
    elif x.ndim == 3 and x.shape[0] == 2:
        metal = x[0] > 0.0
        feed = np.zeros_like(x[1], dtype=bool)
        feed.flat[int(np.argmax(x[1]))] = True
    elif x.ndim == 2:
        encoded = x
        metal = encoded > 0.5
        feed = encoded > 1.5
    else:
        raise ValueError(f"Unsupported /X sample shape: {x.shape}")

    if feed_rc is not None:
        row, col = np.asarray(feed_rc, dtype=np.int64).reshape(2)
        if row < 0 or col < 0 or row >= feed.shape[0] or col >= feed.shape[1]:
            raise ValueError(
                f"feed_rc={(int(row), int(col))} is outside target shape {feed.shape}"
            )
        feed = np.zeros_like(feed, dtype=bool)
        feed[int(row), int(col)] = True

    if not np.any(feed):
        raise ValueError("Feed point not found in /X sample.")

    if np.count_nonzero(feed) > 1:
        first = np.argwhere(feed)[0]
        feed = np.zeros_like(feed, dtype=bool)
        feed[tuple(first)] = True

    metal = np.logical_or(metal, feed)
    row, col = np.argwhere(feed)[0]
    height, width = feed.shape
    feed_index = int(row * width + col)

    return metal.astype(np.float32)[None], feed.astype(np.float32)[None], feed_index


class AntennaTaskDataset(Dataset):
    """Task adapter over H5AntennaDataset.

    This class reuses the existing HDF5/preprocessing-compatible dataset and
    only changes the output contract according to task_name.
    """

    def __init__(
        self,
        h5_path,
        task_name,
        input_key=None,
        x_key="X",
        return_raw=True,
    ):
        if task_name not in SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task_name={task_name!r}; expected one of {sorted(SUPPORTED_TASKS)}")

        self.h5_path = h5_path
        self.task_name = task_name
        self.x_key = x_key

        if input_key is None:
            input_key = "X" if task_name == "x_to_response" else "current"
        self.input_key = input_key

        self.input_dataset = H5AntennaDataset(
            h5_path,
            standardizer=None,
            return_raw=return_raw,
            input_key=self.input_key,
        )

        self.x_dataset = None
        if task_name in STRUCTURE_TASKS:
            self.x_dataset = H5AntennaDataset(
                h5_path,
                standardizer=None,
                return_raw=False,
                input_key=x_key,
                flatten_current=False,
            )
            if len(self.x_dataset) != len(self.input_dataset):
                raise ValueError("Input dataset and X dataset sample counts do not match.")

        self.input_shape = self.input_dataset.x_shape
        self.y_shape = self.input_dataset.y_shape
        self.pattern_shape = self.input_dataset.pattern_shape
        self.pattern_metadata = self.input_dataset.pattern_metadata
        self.feed_rc = None

        if self.x_dataset is not None:
            with h5py.File(h5_path, "r") as f:
                if "feed_rc" in f:
                    coord_space = f["feed_rc"].attrs.get("coordinate_space", "")
                    if isinstance(coord_space, bytes):
                        coord_space = coord_space.decode("utf-8", errors="ignore")
                    if coord_space == "python_transposed_x":
                        feed_rc = np.asarray(f["feed_rc"], dtype=np.int64)
                        if feed_rc.shape != (len(self.input_dataset), 2):
                            raise ValueError(
                                f"/feed_rc shape mismatch: expected {(len(self.input_dataset), 2)}, "
                                f"got {feed_rc.shape}"
                            )
                        self.feed_rc = feed_rc
            x0, _, _, _ = self.x_dataset[0]
            feed_rc0 = None if self.feed_rc is None else self.feed_rc[0]
            metal0, _, _ = structure_targets_from_x(x0.numpy(), feed_rc=feed_rc0)
            self.structure_shape = tuple(int(v) for v in metal0.shape)
            self.x_shape = self.x_dataset.x_shape
        else:
            self.structure_shape = None
            self.x_shape = None

    def __len__(self):
        return len(self.input_dataset)

    def __getitem__(self, idx):
        x, y, pattern, meta = self.input_dataset[idx]

        if self.task_name in RESPONSE_TASKS:
            return x, y, pattern, meta

        x_struct, _, _, _ = self.x_dataset[idx]
        feed_rc = None if self.feed_rc is None else self.feed_rc[idx]
        metal, feed_map, feed_index = structure_targets_from_x(x_struct.numpy(), feed_rc=feed_rc)
        return {
            "input": x,
            "current": x,
            "metal": torch.from_numpy(metal),
            "feed_map": torch.from_numpy(feed_map),
            "feed_index": torch.tensor(feed_index, dtype=torch.long),
        }
