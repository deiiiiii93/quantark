"""PFE / EE profile assembly from pathwise netted exposure, plus a Kupiec
unconditional-coverage backtest. Non-regulatory outputs only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantark.util.exceptions import ValidationError
from quantark.sacva.exposure._contract_provisional import _NEG_TOL

CHI2_95_DF1 = 3.841458820694124   # χ²(0.95, df=1); avoids a scipy dependency
# "linear" = Hyndman-Fan type 7 (interpolated, default); "inverted_cdf" = HF type 1
# (conservative upper order statistic); "higher" is also conservative.
_QUANTILE_METHODS = ("linear", "inverted_cdf", "higher", "lower", "nearest", "midpoint")


@dataclass
class PFEProfileAssembler:
    confidences_bps: tuple = (9500, 9900)
    quantile_method: str = "linear"
    m_tail_min: int = 10

    def __post_init__(self):
        if self.quantile_method not in _QUANTILE_METHODS:
            raise ValidationError(f"invalid quantile_method {self.quantile_method}")
        for bps in self.confidences_bps:
            if not isinstance(bps, int) or not (0 <= bps <= 10000):
                raise ValidationError("confidence must be integer bps in [0,10000]")

    def assemble(self, exposure, times) -> dict:
        E = np.asarray(exposure, float)
        T = np.asarray(times, float)
        if E.ndim != 2:
            raise ValidationError("exposure must be (n_paths, n_times)")
        if T.ndim != 1 or T.shape[0] != E.shape[1]:
            raise ValidationError("times length must match exposure time dimension")
        if not np.all(np.isfinite(E)) or not np.all(np.isfinite(T)):
            raise ValidationError("non-finite exposure/times")
        if np.any(np.diff(T) <= 0):
            raise ValidationError("times must be strictly increasing")
        if np.any(E < -_NEG_TOL):
            raise ValidationError("exposure must be >= 0")
        n = E.shape[0]
        ee = E.mean(axis=0)
        pfe = {}
        for bps in self.confidences_bps:
            conf = bps / 10000.0
            # integer tail-adequacy check (avoids float rounding falsely rejecting
            # exact thresholds, e.g. bps=9999, n=100000, m_tail_min=10)
            if bps < 10000 and n * (10000 - bps) < self.m_tail_min * 10000:
                raise ValidationError(
                    f"insufficient tail paths for {bps}bps: "
                    f"{n * (10000 - bps) / 10000:.1f} < {self.m_tail_min}")
            pfe[bps] = np.quantile(E, conf, axis=0, method=self.quantile_method)
        trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # 2.x removed trapz
        return {"ee_undiscounted": ee, "pfe": pfe, "epe": float(trapz(ee, T))}


def kupiec_pof(num_exceptions, num_obs, confidence, test_level_crit=CHI2_95_DF1):
    """Kupiec proportion-of-failures unconditional-coverage LR test.
    Returns (LR, reject). H0: exception rate == (1 - confidence)."""
    n = int(num_obs)
    x = int(num_exceptions)
    if n <= 0 or not (0.0 < confidence < 1.0):
        raise ValidationError("invalid Kupiec inputs")
    if x < 0 or x > n:
        raise ValidationError("num_exceptions must be in [0, num_obs]")
    p = 1.0 - confidence
    pi = x / n
    if x == 0:
        lr = -2.0 * (n * np.log(1 - p))
    elif x == n:
        lr = -2.0 * (n * np.log(p))
    else:
        lr = -2.0 * ((n - x) * np.log(1 - p) + x * np.log(p)
                     - (n - x) * np.log(1 - pi) - x * np.log(pi))
    return float(lr), bool(lr > test_level_crit)
