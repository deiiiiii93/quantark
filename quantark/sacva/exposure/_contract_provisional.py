"""PROVISIONAL — MC-owned exposure contract; reconcile at merge.

Additive superset of the shapes defined in
``docs/superpowers/plans/2026-06-17-sacva-portfolio-integration.md``. The four
MC positional fields of :class:`ExposureProfile` (``times, epe_discounted,
measure, regulatory_eligible``) keep their order so the Monte-Carlo session's
construction still works; historical-only fields are appended with defaults.

Deleted once the canonical ``quantark.sacva.exposure`` contract lands; see the
merge-gate test in ``test/test_sacva_historical_exposure.py``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from quantark.util.exceptions import ValidationError
from quantark.util.numerical import Tolerance

CONTRACT_VERSION = "provisional-1"  # bump on any field/semantic change
_NEG_TOL = Tolerance.ZERO


class Measure(Enum):
    RISK_NEUTRAL = "risk_neutral"
    REAL_WORLD = "real_world"


@dataclass(frozen=True)
class ExposureNode:
    """One exposure-grid node. ``event_side``: -1 pre-cashflow, +1 post, 0 plain."""

    time: float
    event_side: int = 0
    event_order: int = 0
    is_report_node: bool = True


def _readonly(a) -> np.ndarray:
    arr = np.array(a, dtype=float, copy=True)
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True)
class ExposureProfile:
    # --- MC positional fields (KEEP ORDER) ---
    times: np.ndarray
    epe_discounted: Optional[np.ndarray]   # MC regulatory field; None on historical
    measure: Measure
    regulatory_eligible: bool
    # --- historical additive fields ---
    ee_undiscounted: Optional[np.ndarray] = None
    discounted_ee_nonreg: Optional[np.ndarray] = None
    pfe: Optional[dict] = None             # {confidence_bps: np.ndarray}
    epe: Optional[float] = None            # spec field name (scalar time-weighted EPE)
    metadata: Optional[dict] = None

    @property
    def epe_scalar(self):                  # back-compat read-only alias
        return self.epe

    def __post_init__(self):
        times = _readonly(self.times)
        if times.ndim != 1 or not np.all(np.isfinite(times)):
            raise ValidationError("times must be finite 1D")
        if np.any(np.diff(times) <= 0):
            raise ValidationError("times must be strictly increasing")
        object.__setattr__(self, "times", times)
        n = len(times)

        for name in ("epe_discounted", "ee_undiscounted", "discounted_ee_nonreg"):
            v = getattr(self, name)
            if v is not None:
                arr = _readonly(v)
                if arr.shape != (n,):
                    raise ValidationError(f"{name} must have shape ({n},)")
                if not np.all(np.isfinite(arr)):
                    raise ValidationError(f"non-finite {name}")
                if np.any(arr < -_NEG_TOL):
                    raise ValidationError(f"{name} must be >= 0")
                object.__setattr__(self, name, arr)

        if self.pfe is not None:
            frozen = {}
            for bps, arr in self.pfe.items():
                if not isinstance(bps, int) or not (0 <= bps <= 10000):
                    raise ValidationError(f"pfe key must be integer bps in [0,10000], got {bps}")
                a = _readonly(arr)
                if a.shape != (n,):
                    raise ValidationError(f"pfe[{bps}] must have shape ({n},)")
                if not np.all(np.isfinite(a)) or np.any(a < -_NEG_TOL):
                    raise ValidationError(f"pfe[{bps}] must be finite and >= 0")
                frozen[int(bps)] = a
            object.__setattr__(self, "pfe", MappingProxyType(frozen))
        if self.epe is not None and (not np.isfinite(self.epe) or self.epe < -_NEG_TOL):
            raise ValidationError("epe must be finite and >= 0")
        if self.metadata is not None:
            object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

        # eligibility invariant (two-directional)
        if self.regulatory_eligible:
            if self.measure is not Measure.RISK_NEUTRAL:
                raise ValidationError("regulatory_eligible requires RISK_NEUTRAL")
            if self.epe_discounted is None:
                raise ValidationError("regulatory_eligible requires epe_discounted")
        if self.measure is Measure.REAL_WORLD and self.epe_discounted is not None:
            raise ValidationError("REAL_WORLD must not populate epe_discounted")
        if self.epe_discounted is None and self.ee_undiscounted is None:
            raise ValidationError("profile needs epe_discounted or ee_undiscounted")


class ExposureEngine(ABC):
    @abstractmethod
    def compute(self, counterparty) -> ExposureProfile: ...


# ---------------------------------------------------------------------------
# PROVISIONAL repricing scaffold (throwaway-at-merge): a single-factor analytic
# value surface, minimal portfolio containers, and the netting aggregation, so
# the historical engine is testable before the MC repricer exists.
# ---------------------------------------------------------------------------
@runtime_checkable
class ValueSurface(Protocol):
    def value_at(self, states: np.ndarray, t: float, discrete_state) -> np.ndarray: ...


@dataclass
class AnalyticValueSurface:
    """Wraps a vectorized closed-form kernel ``kernel(S_array, t, discrete_state)``."""

    kernel: object
    state_labels: tuple = (None,)

    def value_at(self, states, t, discrete_state):
        v = np.asarray(self.kernel(np.asarray(states, float), float(t), discrete_state), float)
        if not np.all(np.isfinite(v)):
            raise ValidationError("analytic value_at produced non-finite values")
        return v


@dataclass
class BoundedAnalyticValueSurface(AnalyticValueSurface):
    low: float = -np.inf
    high: float = np.inf
    extrapolate: bool = False

    def value_at(self, states, t, discrete_state):
        s = np.asarray(states, float)
        if not self.extrapolate and (np.any(s < self.low) or np.any(s > self.high)):
            raise ValidationError(
                "state out of value-surface bounds (no silent clip; size the surface "
                "to the realized path range or enable extrapolate)")
        return super().value_at(s, t, discrete_state)


@dataclass
class CVATrade:
    trade_id: str
    surface: object
    factor_key: str
    quantity: float = 1.0
    # capability flags (spec §3.2 scope guard)
    requires_continuous_barrier: bool = False
    requires_fx_conversion: bool = False
    foreign_underlying: bool = False
    n_state_factors: int = 1


@dataclass
class NettingSet:
    set_id: str
    trades: list
    netting_enforceable: bool = True


@dataclass
class Counterparty:
    name: str
    netting_sets: list


def aggregate_epe(trade_value_arrays, enforceable, df):
    stacked = np.stack(trade_value_arrays, axis=0)     # (n_trades, n_paths, n_dates)
    if enforceable:
        netted = np.maximum(stacked.sum(axis=0), 0.0)
    else:
        netted = np.maximum(stacked, 0.0).sum(axis=0)
    return np.asarray(df, float) * netted.mean(axis=0)
