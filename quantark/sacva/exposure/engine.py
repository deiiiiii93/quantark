"""Exposure engine interface + EE profile (spec §3.2).

The exposure method can vary like VaR's MC-vs-historical split, but the two
backends target different measures and consumers: only a RISK_NEUTRAL,
regulatory-eligible profile may feed SA-CVA capital (MAR50.34(1)). The historical
backend (real-world, non-eligible) ships from a separate worktree.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np

from quantark.util.exceptions import ValidationError


class Measure(Enum):
    RISK_NEUTRAL = "risk_neutral"
    REAL_WORLD = "real_world"


@dataclass(frozen=True)
class ExposureProfile:
    times: np.ndarray
    epe_discounted: np.ndarray
    measure: Measure
    regulatory_eligible: bool

    def __post_init__(self) -> None:
        if not isinstance(self.measure, Measure):
            raise ValidationError("measure must be a Measure enum")
        if not isinstance(self.regulatory_eligible, bool):
            raise ValidationError("regulatory_eligible must be a bool")
        t = np.asarray(self.times, dtype=float)
        ee = np.asarray(self.epe_discounted, dtype=float)
        if t.shape != ee.shape:
            raise ValidationError("times and epe_discounted must have equal shape")
        if t.ndim != 1 or not np.all(np.isfinite(t)):
            raise ValidationError("times must be a finite 1-D array")
        if t.size > 1 and np.any(np.diff(t) <= 0):
            raise ValidationError("times must be strictly increasing")
        if t.size == 0 or t[0] != 0.0:
            raise ValidationError(
                "times must start at valuation (t0=0); else the [0, t0] default "
                "interval is dropped from the CVA integral")
        if not np.all(np.isfinite(ee)):
            raise ValidationError("non-finite EPE")
        if np.any(ee < -1e-9):
            raise ValidationError("EPE must be >= 0")
        if self.regulatory_eligible and self.measure is not Measure.RISK_NEUTRAL:
            raise ValidationError(
                "regulatory_eligible profile must be RISK_NEUTRAL measure")
        # store immutable validated copies so the invariants cannot be mutated away
        t = t.copy(); t.flags.writeable = False
        ee = ee.copy(); ee.flags.writeable = False
        object.__setattr__(self, "times", t)
        object.__setattr__(self, "epe_discounted", ee)


def aggregate_epe(trade_value_arrays, enforceable, df):
    """Discounted EPE per date.

    enforceable ⇒ df·mean_paths(max(Σ_trades V, 0));
    else         ⇒ df·mean_paths(Σ_trades max(V, 0))  (MAR50.35).
    Positive-part is taken PATHWISE before averaging.
    """
    if not isinstance(enforceable, bool):  # truthiness would silently mis-net (MAR50.35)
        raise ValidationError("enforceable must be a bool")
    if not trade_value_arrays:
        raise ValidationError("no trade value arrays to aggregate")
    arrays = [np.asarray(a, dtype=float) for a in trade_value_arrays]
    shape0 = arrays[0].shape
    for a in arrays:
        if a.ndim != 2:
            raise ValidationError("each trade value array must be 2-D (n_paths, n_dates)")
        if a.shape != shape0:
            raise ValidationError("trade value arrays must all share the same shape")
        if not np.all(np.isfinite(a)):
            raise ValidationError("trade value arrays must be finite")
    if shape0[0] < 1:
        raise ValidationError("trade value arrays must have >= 1 path")
    if shape0[1] < 1:
        raise ValidationError("trade value arrays must have >= 1 exposure date")
    n_dates = shape0[1]
    df = np.asarray(df, dtype=float)
    if df.ndim != 1 or df.shape != (n_dates,):
        raise ValidationError("df must be 1-D of length n_dates")
    if np.any(df < 0) or not np.all(np.isfinite(df)):
        raise ValidationError("df must be finite and non-negative")
    stacked = np.stack(arrays, axis=0)
    if enforceable:
        netted = np.maximum(stacked.sum(axis=0), 0.0)
    else:
        netted = np.maximum(stacked, 0.0).sum(axis=0)
    epe = df * netted.mean(axis=0)
    if not np.all(np.isfinite(epe)):
        raise ValidationError("aggregated EPE is non-finite (overflow?)")
    return epe


class ExposureEngine(ABC):
    @abstractmethod
    def compute(self, counterparty) -> ExposureProfile:
        ...
