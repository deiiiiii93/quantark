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
