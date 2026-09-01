"""Strike-quoted listed-option normalizer: parity, OTM filter, IV inversion."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from quantark.util.exceptions import ValidationError
from quantark.util.numerical import safe_log
from quantark.volcalibration.snapshot import QuoteSnapshot

# Parity quality gates, promoted from
# example/mo_volmodels/10_calibration_diagnostics.py.
MAX_ABS_PARITY_IMPLIED_RATE = 0.10
MAX_PARITY_RMSE_FORWARD_RATIO = 0.01

MIN_PARITY_PAIRS = 3


@dataclass
class ExpirySlice:
    """One expiry's quotes, indexed by strike for calls and puts separately."""

    expiry_label: str
    expiry_date: Optional[str]
    T: float
    calls: Dict[float, float]
    puts: Dict[float, float]
    volume: Dict[Tuple[float, str], int] = field(default_factory=dict)


@dataclass(frozen=True)
class ParityResult:
    """Carry recovered from one expiry via put-call parity."""

    r: float
    forward: float
    discount_factor: float
    q: float
    n_pairs: int
    rmse_over_forward: float


def iter_expiries(snapshot: QuoteSnapshot) -> List[ExpirySlice]:
    """Reshape each expiry's flat quote list into strike-indexed maps."""
    out: List[ExpirySlice] = []
    for exp in snapshot.expiries:
        calls: Dict[float, float] = {}
        puts: Dict[float, float] = {}
        volume: Dict[Tuple[float, str], int] = {}
        for quote in exp["quotes"]:
            strike = float(quote["strike"])
            kind = quote["type"]
            price = snapshot.quote_price(quote)
            (calls if kind == "C" else puts)[strike] = price
            volume[(strike, kind)] = int(quote.get("volume", 0) or 0)
        label = str(exp.get("expiry_date") or exp.get("contract_month"))
        out.append(
            ExpirySlice(
                expiry_label=label,
                expiry_date=exp.get("expiry_date"),
                T=float(exp["T_years"]),
                calls=calls,
                puts=puts,
                volume=volume,
            )
        )
    return out


def imply_forward_and_rate(sl: ExpirySlice, s0: float) -> ParityResult:
    """Recover ``(r, F, DF, q)`` for one expiry from put-call parity.

    Parity is model-free: ``C(K) - P(K) = DF * (F - K)`` is a straight line in
    ``K`` with slope ``-DF`` and intercept ``DF * F``, so one OLS fit yields the
    market discount factor and the forward together.  A slice that violates the
    quality gates is rejected, never repaired.
    """
    pairs = sorted(set(sl.calls) & set(sl.puts))
    if len(pairs) < MIN_PARITY_PAIRS:
        raise ValidationError(
            f"expiry {sl.expiry_label}: only {len(pairs)} paired strikes "
            f"(< {MIN_PARITY_PAIRS})"
        )
    K = np.asarray(pairs, dtype=float)
    y = np.asarray([sl.calls[k] - sl.puts[k] for k in pairs], dtype=float)
    slope, intercept = np.polyfit(K, y, 1)
    df = float(-slope)
    if df <= 0.0:
        raise ValidationError(
            f"expiry {sl.expiry_label}: non-positive discount factor {df:.4g} "
            "(arbitrage-violating quotes) — excluded, not fabricated"
        )
    forward = float(intercept / df)
    if not math.isfinite(forward) or forward <= 0.0:
        raise ValidationError(
            f"expiry {sl.expiry_label}: non-positive implied forward {forward:.4g}"
        )
    residual = y - (slope * K + intercept)
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    rmse_over_forward = rmse / forward
    if rmse_over_forward > MAX_PARITY_RMSE_FORWARD_RATIO:
        raise ValidationError(
            f"expiry {sl.expiry_label}: parity RMSE/forward "
            f"{rmse_over_forward:.4g} exceeds {MAX_PARITY_RMSE_FORWARD_RATIO}"
        )
    r = float(-safe_log(df) / sl.T)
    if abs(r) > MAX_ABS_PARITY_IMPLIED_RATE:
        raise ValidationError(
            f"expiry {sl.expiry_label}: implied rate {r:.4g} exceeds "
            f"+/-{MAX_ABS_PARITY_IMPLIED_RATE}"
        )
    q = float(r - safe_log(forward / s0) / sl.T)
    if not math.isfinite(q):
        raise ValidationError(f"expiry {sl.expiry_label}: non-finite carry q={q}")
    return ParityResult(
        r=r,
        forward=forward,
        discount_factor=df,
        q=q,
        n_pairs=len(pairs),
        rmse_over_forward=rmse_over_forward,
    )
