"""Shared helpers for the MO vol-model suite: snapshot IO, put-call parity, OTM
filtering, Black-IV inversion, and plotting. Pure quantark (no akshare)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from quantark.util.numerical import safe_log

_REQUIRED = {"fetched_at", "underlying", "expiries"}


def load_snapshot(path) -> dict:
    """Load and lightly validate an MO snapshot JSON."""
    snap = json.loads(Path(path).read_text())
    missing = _REQUIRED - set(snap)
    if missing:
        raise ValueError(f"snapshot missing keys: {sorted(missing)}")
    if "spot" not in snap.get("underlying", {}):
        raise ValueError("snapshot underlying missing 'spot'")
    return snap


@dataclass
class ExpirySlice:
    """One expiry's quotes, indexed by strike for calls and puts separately."""

    expiry_date: str
    T: float
    calls: Dict[float, float]  # strike -> last price
    puts: Dict[float, float]
    volume: Dict[Tuple[float, str], int] = field(default_factory=dict)


def iter_expiries(snapshot: dict) -> List[ExpirySlice]:
    """Reshape the flat quote list of each expiry into strike-indexed maps."""
    out: List[ExpirySlice] = []
    for exp in snapshot["expiries"]:
        calls: Dict[float, float] = {}
        puts: Dict[float, float] = {}
        vol: Dict[Tuple[float, str], int] = {}
        for q in exp["quotes"]:
            (calls if q["type"] == "C" else puts)[float(q["strike"])] = float(q["last"])
            vol[(float(q["strike"]), q["type"])] = int(q.get("volume", 0))
        out.append(ExpirySlice(exp["expiry_date"], float(exp["T_years"]), calls, puts, vol))
    return out


@dataclass
class ParityResult:
    """Carry recovered from one expiry via put-call parity."""

    r: float
    forward: float
    discount_factor: float
    q: float
    n_pairs: int


def imply_forward_and_rate(sl: "ExpirySlice", s0: float) -> "ParityResult":
    """Recover (r, forward, DF, q) for one expiry from put-call parity.

    Model: for each paired strike K,  C(K) - P(K) = DF * (F - K),
    which is linear in K with slope = -DF and intercept = DF*F.
    """
    pairs = sorted(set(sl.calls) & set(sl.puts))
    if len(pairs) < 3:
        raise ValueError(f"expiry {sl.expiry_date}: only {len(pairs)} paired strikes (<3)")
    K = np.array(pairs)
    y = np.array([sl.calls[k] - sl.puts[k] for k in pairs])
    # Put-call parity is model-free: C - P = DF*(F - K) is a straight line in K.
    # Slope = -DF, intercept = DF*F. A single OLS fit yields both the market
    # discount factor and the forward. (Near-ATM weighting is possible via the
    # `w=` arg of polyfit; OLS is used here — clean on arbitrage-consistent quotes.)
    a, b = np.polyfit(K, y, 1)
    df = -a
    if df <= 0.0:
        raise ValueError(
            f"expiry {sl.expiry_date}: non-positive discount factor {df:.4g} "
            "(arbitrage-violating quotes) — excluded, not fabricated"
        )
    forward = float(b / df)
    r = float(-safe_log(df) / sl.T)
    q = float(r - safe_log(forward / s0) / sl.T)
    return ParityResult(r=r, forward=forward, discount_factor=float(df), q=q, n_pairs=len(pairs))
