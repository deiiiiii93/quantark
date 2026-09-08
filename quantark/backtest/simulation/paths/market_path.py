"""``MarketPath``: a batch of simulated joint market paths (spec 4.1).

Every day of every path is a full market snapshot: spot, the scalar ATM vol
the pricer receives, a flat continuously-compounded rate, and the
constant-maturity carry curve ``B(T_k) = ln(F(T_k)/S)`` on a fixed tenor
grid measured from that day.  ``B(0) = 0`` is implicit.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

from quantark.util.exceptions import ValidationError

DEFAULT_TENOR_GRID = np.array([1 / 12, 2 / 12, 3 / 12, 6 / 12, 9 / 12, 1.0, 1.5])


def trading_calendar(start: date, n_days: int, *, holidays: Sequence[date] = ()) -> pd.DatetimeIndex:
    """``n_days`` Monday-Friday days from ``start`` (inclusive when it trades), skipping ``holidays``."""
    if n_days < 1:
        raise ValidationError("n_days must be at least 1")
    closed = {pd.Timestamp(h).date() for h in holidays}
    out = []
    cur = pd.Timestamp(start).date()
    while len(out) < n_days:
        if cur.weekday() < 5 and cur not in closed:
            out.append(pd.Timestamp(cur))
        cur += timedelta(days=1)
    return pd.DatetimeIndex(out)


def _dates_as_ns(dates: pd.DatetimeIndex) -> np.ndarray:
    """Dates as int64 nanoseconds.

    ``DatetimeIndex.asi8`` counts in the index's OWN resolution, which pandas
    infers (seconds for whole days, nanoseconds elsewhere), so a fingerprint
    or an npz built on it would depend on how the index happened to be
    constructed.  Pin the unit here instead.
    """
    return np.asarray(dates.values, dtype="datetime64[ns]").astype(np.int64)


def _validate_tenor_grid(tenor_grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(tenor_grid, dtype=float)
    if grid.ndim != 1 or grid.size == 0:
        raise ValidationError("tenor_grid must be a non-empty 1-D array")
    if not np.all(np.isfinite(grid)) or grid[0] <= 0.0 or np.any(np.diff(grid) <= 0.0):
        raise ValidationError("tenor_grid must be positive, finite and strictly increasing")
    return grid


@dataclass(frozen=True)
class StartState:
    """The inception snapshot a generator integrates from."""

    spot: float
    atm_vol: float
    rate: float
    carry: np.ndarray  # (n_tenors,)

    def __post_init__(self) -> None:
        carry = np.asarray(self.carry, dtype=float)
        if carry.ndim != 1 or not np.all(np.isfinite(carry)):
            raise ValidationError("StartState.carry must be a finite 1-D array")
        object.__setattr__(self, "carry", carry)
        if not (self.spot > 0.0 and self.atm_vol > 0.0):
            raise ValidationError("StartState needs positive spot and atm_vol")
        if not np.isfinite(self.rate):
            raise ValidationError("StartState.rate must be finite")


@dataclass(frozen=True)
class MarketPath:
    """A batch of paths on one trading calendar (spec 4.1)."""

    dates: pd.DatetimeIndex
    spot: np.ndarray       # (n_paths, n_days)
    atm_vol: np.ndarray    # (n_paths, n_days)
    rate: np.ndarray       # (n_paths, n_days)
    carry: np.ndarray      # (n_paths, n_days, n_tenors)
    tenor_grid: np.ndarray  # (n_tenors,)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        dates = pd.DatetimeIndex(self.dates)
        if len(dates) == 0 or not dates.is_monotonic_increasing or not dates.is_unique:
            raise ValidationError("dates must be a non-empty, strictly increasing DatetimeIndex")
        grid = _validate_tenor_grid(self.tenor_grid)
        spot = np.asarray(self.spot, dtype=float)
        vol = np.asarray(self.atm_vol, dtype=float)
        rate = np.asarray(self.rate, dtype=float)
        carry = np.asarray(self.carry, dtype=float)
        n_days = len(dates)
        if spot.ndim != 2 or spot.shape[1] != n_days:
            raise ValidationError(f"spot must have shape (n_paths, {n_days}), got {spot.shape}")
        n_paths = spot.shape[0]
        for name, arr in (("atm_vol", vol), ("rate", rate)):
            if arr.shape != (n_paths, n_days):
                raise ValidationError(f"{name} must have shape {(n_paths, n_days)}, got {arr.shape}")
        if carry.shape != (n_paths, n_days, grid.size):
            raise ValidationError(f"carry must have shape {(n_paths, n_days, grid.size)}, got {carry.shape}")
        for name, arr in (("spot", spot), ("atm_vol", vol), ("rate", rate), ("carry", carry)):
            if not np.all(np.isfinite(arr)):
                raise ValidationError(f"{name} contains non-finite values")
        if np.any(spot <= 0.0) or np.any(vol <= 0.0):
            raise ValidationError("spot and atm_vol must be positive")
        object.__setattr__(self, "dates", dates)
        object.__setattr__(self, "spot", spot)
        object.__setattr__(self, "atm_vol", vol)
        object.__setattr__(self, "rate", rate)
        object.__setattr__(self, "carry", carry)
        object.__setattr__(self, "tenor_grid", grid)
        object.__setattr__(self, "meta", dict(self.meta))

    @property
    def n_paths(self) -> int:
        return int(self.spot.shape[0])

    @property
    def n_days(self) -> int:
        return int(self.spot.shape[1])

    @property
    def n_tenors(self) -> int:
        return int(self.tenor_grid.size)

    def path(self, i: int) -> "MarketPath":
        """The single-path batch for path ``i`` (meta carries ``path_index``)."""
        if not 0 <= i < self.n_paths:
            raise ValidationError(f"path index {i} out of range for {self.n_paths} paths")
        return MarketPath(
            dates=self.dates, spot=self.spot[i: i + 1], atm_vol=self.atm_vol[i: i + 1],
            rate=self.rate[i: i + 1], carry=self.carry[i: i + 1], tenor_grid=self.tenor_grid,
            meta={**self.meta, "path_index": int(i)},
        )

    def fingerprint(self) -> str:
        """sha256 of the arrays, dates and tenor grid; ``meta`` is excluded."""
        h = hashlib.sha256()
        h.update(_dates_as_ns(self.dates).tobytes())
        for arr in (self.tenor_grid, self.spot, self.atm_vol, self.rate, self.carry):
            h.update(str(arr.shape).encode())
            h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
        return h.hexdigest()

    def to_npz(self, path) -> None:
        """Write the batch to a compressed npz (``meta`` as a JSON string)."""
        np.savez_compressed(
            Path(path), dates=_dates_as_ns(self.dates), spot=self.spot, atm_vol=self.atm_vol, rate=self.rate,
            carry=self.carry, tenor_grid=self.tenor_grid, meta=np.array(json.dumps(self.meta, default=str)),
        )

    @classmethod
    def from_npz(cls, path) -> "MarketPath":
        """Read a batch written by :meth:`to_npz`; the fingerprint round trips."""
        with np.load(Path(path), allow_pickle=False) as z:
            return cls(
                dates=pd.DatetimeIndex(z["dates"].astype("datetime64[ns]")), spot=z["spot"], atm_vol=z["atm_vol"],
                rate=z["rate"], carry=z["carry"], tenor_grid=z["tenor_grid"], meta=json.loads(str(z["meta"])),
            )
