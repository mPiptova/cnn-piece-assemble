import numpy as np


def longest_continuous_subsequence(sequence: np.array, max_diff=2) -> np.array:
    mask = (sequence[1:] - sequence[:-1] < max_diff).astype(int)
    mask = np.pad(mask, (1, 1), "constant", constant_values=(0, 0))
    diff = mask[1:] - mask[:-1]
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    if len(starts) == 0:
        return np.array([])
    idx_max = (ends - starts).argmax()
    return sequence[starts[idx_max] : ends[idx_max]]
