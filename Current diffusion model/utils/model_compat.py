# -*- coding: utf-8 -*-


def _shape_tuple(value):
    if value is None:
        return None
    return tuple(int(v) for v in value)


def validate_diffusion_checkpoint_dataset_shapes(checkpoint, dataset_info, context="checkpoint"):
    saved_info = checkpoint.get("dataset_info")
    if not saved_info:
        return

    keys = ("current_shape", "y_shape", "pattern_shape")
    mismatches = []
    for key in keys:
        if key not in saved_info or key not in dataset_info:
            continue
        saved_shape = _shape_tuple(saved_info[key])
        current_shape = _shape_tuple(dataset_info[key])
        if saved_shape != current_shape:
            mismatches.append(f"{key}: checkpoint={saved_shape}, current={current_shape}")

    if mismatches:
        raise ValueError(
            f"{context} was trained for a different dataset shape. "
            + "; ".join(mismatches)
            + ". Train a new diffusion model or use a matching dataset."
        )


def validate_forward_surrogate_shapes(surrogate_checkpoint, dataset_info, context="forward surrogate"):
    saved_info = surrogate_checkpoint.get("dataset_info")
    if not saved_info:
        return

    checks = {
        "x_shape": dataset_info.get("current_shape"),
        "y_shape": dataset_info.get("y_shape"),
        "pattern_shape": dataset_info.get("pattern_shape"),
    }
    mismatches = []
    for key, expected in checks.items():
        if key not in saved_info or expected is None:
            continue
        saved_shape = _shape_tuple(saved_info[key])
        expected_shape = _shape_tuple(expected)
        if saved_shape != expected_shape:
            mismatches.append(f"{key}: surrogate={saved_shape}, diffusion_dataset={expected_shape}")

    if mismatches:
        raise ValueError(
            f"{context} checkpoint is incompatible with the diffusion dataset. "
            + "; ".join(mismatches)
        )

