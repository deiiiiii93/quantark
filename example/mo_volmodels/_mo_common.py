"""Shared helpers for the MO vol-model suite: snapshot IO, put-call parity, OTM
filtering, Black-IV inversion, and plotting. Pure quantark (no akshare)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

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
