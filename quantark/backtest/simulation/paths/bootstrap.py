"""Politis-Romano stationary block bootstrap of the joint daily state (spec 5.1)."""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from quantark.util.exceptions import ValidationError

from .history import PathHistory
from .market_path import MarketPath, StartState

TRADING_DAYS_PER_YEAR = 252.0


def stationary_block_indices(
    rng: np.random.Generator, n_source: int, n_paths: int, n_days: int, mean_block_days: int
) -> np.ndarray:
    """Row indices into the source: geometric blocks (mean ``mean_block_days``), circular."""
    if n_source < 1 or n_paths < 1 or n_days < 1 or mean_block_days < 1:
        raise ValidationError("n_source, n_paths, n_days and mean_block_days must be positive")
    p_new = 1.0 / float(mean_block_days)
    starts = rng.integers(0, n_source, size=(n_paths, n_days))
    new_block = rng.random((n_paths, n_days)) < p_new
    idx = np.empty((n_paths, n_days), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for d in range(1, n_days):
        idx[:, d] = np.where(new_block[:, d], starts[:, d], (idx[:, d - 1] + 1) % n_source)
    return idx


class StationaryBlockBootstrap:
    """Resample the realised joint daily changes into forward-looking paths.

    Day 0 of every path is the declared ``start`` state verbatim; days 1
    onward integrate resampled ``PathHistory.changes()`` rows.  ``vol_floor``
    is a floor on the SIMULATED vol, so it never rewrites the start state --
    ``meta['vol_floor_hits']`` counts the generated path-days it bound on.
    """

    def __init__(
        self,
        history: PathHistory,
        *,
        mean_block_days: int,
        demean_returns: bool,
        annual_drift: float,
        vol_floor: float,
        carry_mode: Literal["changes", "levels"],
        start: StartState,
        calendar: pd.DatetimeIndex,
    ) -> None:
        if int(mean_block_days) < 1:
            raise ValidationError("mean_block_days must be at least 1")
        if carry_mode not in ("changes", "levels"):
            raise ValidationError("carry_mode must be 'changes' or 'levels'")
        if not np.isfinite(annual_drift) or not np.isfinite(vol_floor) or vol_floor <= 0.0:
            raise ValidationError("annual_drift must be finite and vol_floor positive")
        if start.carry.shape != history.tenor_grid.shape:
            raise ValidationError("start.carry must be on the history's tenor grid")
        self.history = history
        self.mean_block_days = int(mean_block_days)
        self.demean_returns = bool(demean_returns)
        self.annual_drift = float(annual_drift)
        self.vol_floor = float(vol_floor)
        self.carry_mode = carry_mode
        self.start = start
        self.calendar = pd.DatetimeIndex(calendar)

    def generate(self, n_paths: int, n_days: int, *, seed: int) -> MarketPath:
        if n_paths < 1 or n_days < 1:
            raise ValidationError("n_paths and n_days must be positive")
        if n_days > len(self.calendar):
            raise ValidationError(f"calendar has {len(self.calendar)} days, {n_days} requested")
        rng = np.random.default_rng(int(seed))
        changes = self.history.changes()                     # (m, 3 + K)
        returns = changes[:, 0]
        if self.demean_returns:
            returns = returns - returns.mean()
        returns = returns + self.annual_drift / TRADING_DAYS_PER_YEAR
        n_tenors = self.history.tenor_grid.size
        idx = stationary_block_indices(rng, changes.shape[0], n_paths, n_days - 1, self.mean_block_days) \
            if n_days > 1 else np.empty((n_paths, 0), dtype=np.int64)

        log_spot = np.empty((n_paths, n_days))
        log_spot[:, 0] = np.log(self.start.spot)
        vol = np.empty((n_paths, n_days))
        vol[:, 0] = self.start.atm_vol
        rate = np.empty((n_paths, n_days))
        rate[:, 0] = self.start.rate
        carry = np.empty((n_paths, n_days, n_tenors))
        carry[:, 0, :] = self.start.carry
        floor_hits = 0
        for d in range(1, n_days):
            rows = idx[:, d - 1]
            log_spot[:, d] = log_spot[:, d - 1] + returns[rows]
            raw_vol = vol[:, d - 1] + changes[rows, 1]
            floor_hits += int(np.count_nonzero(raw_vol < self.vol_floor))
            vol[:, d] = np.maximum(raw_vol, self.vol_floor)
            rate[:, d] = rate[:, d - 1] + changes[rows, 2]
            if self.carry_mode == "changes":
                carry[:, d, :] = carry[:, d - 1, :] + changes[rows, 3:]
            else:
                carry[:, d, :] = self.history.carry[rows + 1]   # the level of the resampled day
        meta = {
            "generator": "stationary_block_bootstrap", "seed": int(seed),
            "mean_block_days": self.mean_block_days, "demean_returns": self.demean_returns,
            "annual_drift": self.annual_drift, "vol_floor": self.vol_floor, "carry_mode": self.carry_mode,
            "history_fingerprint": self.history.source_fingerprint, "vol_floor_hits": floor_hits,
            "start": {"spot": self.start.spot, "atm_vol": self.start.atm_vol, "rate": self.start.rate},
        }
        return MarketPath(dates=self.calendar[:n_days], spot=np.exp(log_spot), atm_vol=vol, rate=rate,
                          carry=carry, tenor_grid=self.history.tenor_grid.copy(), meta=meta)
