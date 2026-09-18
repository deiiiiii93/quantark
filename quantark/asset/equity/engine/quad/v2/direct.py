"""Bounded direct application of the common block kernel."""
import numpy as np
from quantark.util.exceptions import NumericalError


def apply_direct(kernel, values, max_work_bytes):
    channels, cells, order = values.shape
    band = (len(kernel) - 1) // 2
    row_bytes = (2 * band + 1) * order * 8
    if row_bytes > max_work_bytes:
        raise NumericalError("QUAD V2 direct kernel exceeds workspace budget")
    block = max(1, min(cells, max_work_bytes // max(2 * row_bytes, 1)))
    out = np.empty_like(values)
    for c in range(channels):
        # Edge extension is identical for both backends.
        padded = np.pad(values[c], ((band, band), (0, 0)), mode="edge")
        for start in range(0, cells, block):
            stop = min(start + block, cells)
            view = np.lib.stride_tricks.sliding_window_view(
                padded[start : stop + 2 * band], 2 * band + 1, axis=0
            )
            matrix = np.ascontiguousarray(view.transpose(0, 2, 1)).reshape(
                stop - start, -1
            )
            out[c, start:stop] = matrix @ kernel.reshape(-1, order)
    return out
