def validate_checkpoint_dataset_shapes(checkpoint, dataset_info, context="checkpoint"):
    saved_info = checkpoint.get("dataset_info")
    if not saved_info:
        return

    keys = ("x_shape", "y_shape", "pattern_shape")
    mismatches = []
    for key in keys:
        if key not in saved_info or key not in dataset_info:
            continue
        saved_shape = tuple(saved_info[key])
        current_shape = tuple(dataset_info[key])
        if saved_shape != current_shape:
            mismatches.append(f"{key}: checkpoint={saved_shape}, current={current_shape}")

    if mismatches:
        raise ValueError(
            f"{context} was trained for a different dataset shape. "
            + "; ".join(mismatches)
            + ". Train a new model or use a dataset with matching frequency dimensions."
        )
