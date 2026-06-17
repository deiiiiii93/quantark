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
        t = np.asarray(self.times, dtype=float)
        ee = np.asarray(self.epe_discounted, dtype=float)
        if t.shape != ee.shape:
            raise ValidationError("times and epe_discounted must have equal shape")
        if t.ndim != 1 or (t.size > 1 and np.any(np.diff(t) <= 0)):
            raise ValidationError("times must be 1-D strictly increasing")
        if not np.all(np.isfinite(ee)):
            raise ValidationError("non-finite EPE")
        if np.any(ee < -1e-9):
            raise ValidationError("EPE must be >= 0")
        if self.regulatory_eligible and self.measure is not Measure.RISK_NEUTRAL:
            raise ValidationError(
                "regulatory_eligible profile must be RISK_NEUTRAL measure")


def aggregate_epe(trade_value_arrays, enforceable, df):
    """Discounted EPE per date.

    enforceable ⇒ df·mean_paths(max(Σ_trades V, 0));
    else         ⇒ df·mean_paths(Σ_trades max(V, 0))  (MAR50.35).
    Positive-part is taken PATHWISE before averaging.
    """
    if not trade_value_arrays:
        raise ValidationError("no trade value arrays to aggregate")
    stacked = np.stack([np.asarray(a, dtype=float) for a in trade_value_arrays], axis=0)
    df = np.asarray(df, dtype=float)
    if stacked.shape[2] != df.shape[0]:
        raise ValidationError("df length must match number of exposure dates")
    if np.any(df < 0) or not np.all(np.isfinite(df)):
        raise ValidationError("df must be finite and non-negative")
    if enforceable:
        netted = np.maximum(stacked.sum(axis=0), 0.0)
    else:
        netted = np.maximum(stacked, 0.0).sum(axis=0)
    return df * netted.mean(axis=0)


class ExposureEngine(ABC):
    @abstractmethod
    def compute(self, counterparty) -> ExposureProfile:
        ...
