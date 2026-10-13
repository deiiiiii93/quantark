"""Standard-error helpers for plain MC runs, including antithetic runs.

Antithetic layout (``apply_variance_reduction_to_normals`` and the engines'
own ``[z, -z]`` draws): within one batch the samples are ``[z-half; -z-half]``
truncated to ``n``.  With ``m = ceil(n / 2)`` the antithetic partner of sample
``i`` is sample ``m + i`` for ``i < n - m``.  For odd ``n`` the unpaired
``z[m - 1]`` is left out of the standard error only (the price keeps every
sample).

Pairs are the independent units of an antithetic run, so the standard error is
``std(pair means, ddof=1) / sqrt(#pairs)``.  The iid formula on the raw samples
ignores the (negative) correlation inside each pair and misstates the error.

Pairing is per batch.  A caller with several batches pairs each batch with
:func:`antithetic_pair_means`, concatenates the pair means, and calls
:func:`mean_and_std_error` once on the pooled unit set.

Caveat: the iid formula is NOT a valid error bar for deterministic
(non-randomized) QMC such as ``MonteCarloMethod.QUASI``: Sobol points are not
independent samples.  Use ``MonteCarloMethod.RANDOMIZED_QUASI`` (RQMC), whose
replications give a valid error bar.  The QMC path keeps the iid formula here
unchanged.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from quantark.util.exceptions import ValidationError

__all__ = ["antithetic_pair_means", "mean_and_std_error", "sample_std_error"]


def antithetic_pair_means(batch_samples: np.ndarray, n_paths: int) -> np.ndarray:
    """Pair means ``0.5 * (x[i] + x[m + i])`` of one antithetic batch.

    Parameters
    ----------
    batch_samples : np.ndarray
        Per-path samples of ONE batch laid out as ``[z-half; -z-half][:n]``.
    n_paths : int
        Number of paths ``n`` of the batch (must equal ``len(batch_samples)``).
    """
    x = np.asarray(batch_samples, dtype=float)
    n = int(n_paths)
    if x.ndim != 1 or x.shape[0] != n:
        raise ValidationError(
            f"antithetic batch must be 1-D with n_paths={n} samples, got shape {x.shape}"
        )
    m = (n + 1) // 2
    return 0.5 * (x[: n - m] + x[m:])


def mean_and_std_error(units: np.ndarray) -> Tuple[float, float]:
    """Mean and ``std(ddof=1) / sqrt(len)`` of independent units."""
    u = np.asarray(units, dtype=float)
    return float(u.mean()), float(u.std(ddof=1)) / math.sqrt(len(u))


def sample_std_error(samples: np.ndarray, *, antithetic: bool) -> float:
    """Standard error of the mean of per-path ``samples``.

    ``antithetic=False`` is the iid formula (bitwise the formula the engines
    inlined before).  ``antithetic=True`` assumes ONE batch in the
    ``[z-half; -z-half][:n]`` layout and uses the pair-mean formula; the price
    (the mean of all samples) is not touched.

    The iid formula is not a valid error bar for deterministic QMC; use
    ``RANDOMIZED_QUASI`` for a valid one.
    """
    x = np.asarray(samples, dtype=float)
    if not antithetic:
        return float(x.std(ddof=1)) / math.sqrt(len(x))
    return mean_and_std_error(antithetic_pair_means(x, len(x)))[1]
