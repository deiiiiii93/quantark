"""Exposure engine interface + EE profile (spec §3.2).

The exposure method can vary like VaR's MC-vs-historical split, but the two
backends target different measures and consumers: only a RISK_NEUTRAL,
regulatory-eligible profile may feed SA-CVA capital (MAR50.34(1)). The historical
backend (real-world, non-eligible) ships from a separate worktree.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Optional

import numpy as np

from quantark.util.exceptions import ValidationError


class Measure(Enum):
    RISK_NEUTRAL = "risk_neutral"
    REAL_WORLD = "real_world"


@dataclass(frozen=True)
class ExposureProfile:
    """Exposure profile shared by the MC (regulatory) and historical (non-regulatory)
    backends.

    Risk-neutral MC populates ``epe_discounted`` (the only field SA-CVA capital
    reads). The real-world historical backend leaves ``epe_discounted = None`` and
    populates the additive, explicitly non-regulatory fields ``ee_undiscounted`` /
    ``discounted_ee_nonreg`` / ``pfe`` / ``epe`` (PFE/limits/backtesting). The
    eligibility invariant is enforced both ways so a real-world profile can never
    look regulatory (MAR50.34(1)).
    """

    times: np.ndarray
    epe_discounted: Optional[np.ndarray]   # regulatory field; None on historical profiles
    measure: Measure
    regulatory_eligible: bool
    # --- additive (historical / non-regulatory) ---
    ee_undiscounted: Optional[np.ndarray] = None
    discounted_ee_nonreg: Optional[np.ndarray] = None
    pfe: Optional[dict] = None             # {confidence_bps: np.ndarray}
    epe: Optional[float] = None            # scalar time-weighted EPE
    metadata: Optional[dict] = None

    def __post_init__(self) -> None:
        if not isinstance(self.measure, Measure):
            raise ValidationError("measure must be a Measure enum")
        if not isinstance(self.regulatory_eligible, bool):
            raise ValidationError("regulatory_eligible must be a bool")
        t = np.asarray(self.times, dtype=float)
        if t.ndim != 1 or not np.all(np.isfinite(t)):
            raise ValidationError("times must be a finite 1-D array")
        if t.size > 1 and np.any(np.diff(t) <= 0):
            raise ValidationError("times must be strictly increasing")
        if t.size == 0 or t[0] != 0.0:
            raise ValidationError(
                "times must start at valuation (t0=0); else the [0, t0] default "
                "interval is dropped from the CVA integral")
        t = t.copy(); t.flags.writeable = False
        object.__setattr__(self, "times", t)
        n = t.size

        # validate + freeze each optional non-negative array against the time axis
        for name in ("epe_discounted", "ee_undiscounted", "discounted_ee_nonreg"):
            v = getattr(self, name)
            if v is not None:
                arr = np.asarray(v, dtype=float)
                if arr.shape != (n,):
                    raise ValidationError(f"{name} must have shape ({n},)")
                if not np.all(np.isfinite(arr)):
                    raise ValidationError(f"non-finite {name}")
                if np.any(arr < -1e-9):
                    raise ValidationError(f"{name} must be >= 0")
                arr = arr.copy(); arr.flags.writeable = False
                object.__setattr__(self, name, arr)

        if self.pfe is not None:
            frozen = {}
            for bps, arr in self.pfe.items():
                if not isinstance(bps, int) or isinstance(bps, bool) or not (0 <= bps <= 10000):
                    raise ValidationError(f"pfe key must be integer bps in [0,10000], got {bps}")
                a = np.asarray(arr, dtype=float)
                if a.shape != (n,):
                    raise ValidationError(f"pfe[{bps}] must have shape ({n},)")
                if not np.all(np.isfinite(a)) or np.any(a < -1e-9):
                    raise ValidationError(f"pfe[{bps}] must be finite and >= 0")
                a = a.copy(); a.flags.writeable = False
                frozen[int(bps)] = a
            object.__setattr__(self, "pfe", MappingProxyType(frozen))
        if self.epe is not None and (not np.isfinite(self.epe) or self.epe < -1e-9):
            raise ValidationError("epe must be finite and >= 0")
        if self.metadata is not None:
            object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

        # eligibility invariant (two-directional): only a RISK_NEUTRAL profile that
        # actually carries the regulatory discounted-EE field may be eligible.
        if self.regulatory_eligible:
            if self.measure is not Measure.RISK_NEUTRAL:
                raise ValidationError(
                    "regulatory_eligible profile must be RISK_NEUTRAL measure")
            if self.epe_discounted is None:
                raise ValidationError("regulatory_eligible profile requires epe_discounted")
        # Note: a REAL_WORLD profile is always regulatory_eligible=False (the check
        # above), and RegulatoryCVAEngine consumes only regulatory_eligible profiles,
        # so the capital boundary holds via the eligibility flag. A REAL_WORLD profile
        # MAY still carry epe_discounted (used by MC tests as a rejected fixture); the
        # historical engine leaves it None by construction.
        if self.epe_discounted is None and self.ee_undiscounted is None:
            raise ValidationError("profile needs epe_discounted or ee_undiscounted")


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
