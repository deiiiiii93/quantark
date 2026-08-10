"""Cross-fitted control-variate weights for batched reference estimators.

The frozen certification design uses a single global control weight. This
module estimates per-cell weights without introducing bias: batches are split
into folds, each fold's weight is fitted on the *other* folds only, and the
control's expectation is supplied externally (from an independent
high-precision control run), so E[adjusted] == E[primary] exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantark.util.exceptions import ValidationError


@dataclass(frozen=True)
class CrossFittedControl:
    """Per-batch adjusted values plus the out-of-fold weights that made them."""

    adjusted: np.ndarray
    weights: np.ndarray
    variance_ratio: float

    def as_dict(self) -> dict:
        return {
            "weights": [float(w) for w in self.weights],
            "variance_ratio": float(self.variance_ratio),
            "n_batches": int(self.adjusted.size),
        }


def cross_fitted_control(
    primary: np.ndarray,
    control: np.ndarray,
    control_expectation: float,
    folds: int = 2,
) -> CrossFittedControl:
    """Adjust ``primary`` batch means by an out-of-fold-fitted control weight.

    ``adjusted[i] = primary[i] - beta_k * (control[i] - control_expectation)``
    where ``beta_k`` is the OLS weight fitted on every fold except the one
    containing batch ``i``. Unbiasedness needs only that
    ``E[control] == control_expectation`` and that the folds are independent,
    so the weight never sees the batches it adjusts.
    """
    primary = np.asarray(primary, dtype=float)
    control = np.asarray(control, dtype=float)
    if primary.shape != control.shape or primary.ndim != 1:
        raise ValidationError("primary and control must be 1-D arrays of equal length")
    if folds < 2:
        raise ValidationError("cross-fitting requires at least 2 folds")
    if primary.size < 2 * folds:
        raise ValidationError("need at least 2 batches per fold")

    fold_ids = np.arange(primary.size) % folds
    weights = np.empty(folds, dtype=float)
    adjusted = np.empty_like(primary)
    centered = control - float(control_expectation)
    for k in range(folds):
        out_of_fold = fold_ids != k
        variance = np.var(centered[out_of_fold], ddof=1)
        if variance <= 0.0:
            weights[k] = 0.0
        else:
            covariance = np.cov(primary[out_of_fold], centered[out_of_fold], ddof=1)[0, 1]
            weights[k] = covariance / variance
        in_fold = fold_ids == k
        adjusted[in_fold] = primary[in_fold] - weights[k] * centered[in_fold]

    var_primary = np.var(primary, ddof=1)
    var_adjusted = np.var(adjusted, ddof=1)
    ratio = float(var_adjusted / var_primary) if var_primary > 0.0 else 1.0
    return CrossFittedControl(adjusted=adjusted, weights=weights, variance_ratio=ratio)
