"""Linear FFT application of the identical direct block kernel."""
import numpy as np
from scipy.fft import rfft, irfft, next_fast_len
from quantark.util.exceptions import NumericalError


def apply_fft(kernel, values, max_work_bytes, cache, key):
    channels, cells, order = values.shape
    band = (len(kernel) - 1) // 2
    length = next_fast_len(cells + 4 * band)
    required = (length // 2 + 1) * (order * order + 2 * channels * order) * 16
    if required > max_work_bytes:
        raise NumericalError(f"QUAD V2 FFT workspace requires {required} bytes")
    transform_key = ("fft", key, length)
    transformed = cache.get(transform_key)
    if transformed is None:
        transformed = rfft(kernel[::-1], n=length, axis=0)
        cache.put(transform_key, transformed)
    padded = np.pad(values, ((0, 0), (band, band), (0, 0)), mode="edge")
    spectrum = rfft(padded, n=length, axis=1)
    result = irfft(
        np.einsum("cfq,fqr->cfr", spectrum, transformed, optimize=True),
        n=length,
        axis=1,
    )
    return result[:, 2 * band : 2 * band + cells]
