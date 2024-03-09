def get_resize_shape(original_shape: tuple[int, int], max_size: str) -> tuple[int, int]:
    if max(original_shape) <= max_size:
        return original_shape

    if original_shape[0] > original_shape[1]:
        new_shape = (
            max_size,
            round(original_shape[1] * (max_size / original_shape[0])),
        )
    else:
        new_shape = (
            round(original_shape[0] * (max_size / original_shape[1])),
            max_size,
        )

    return new_shape
