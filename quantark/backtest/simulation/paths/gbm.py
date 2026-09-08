"""GBM spot with a real-world drift and a vol rule (spec 5.2)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd

from quantark.util.exceptions import ValidationError

from .market_path import MarketPath, StartState, _validate_tenor_grid

TRADING_DAYS_PER_YEAR = 252.0


@dataclass(frozen=True)
class ConstantVol:
    """The pricer sees one ATM vol on every day of every path."""

    level: float

    def __post_init__(self) -> None:
        if not self.level > 0.0:
            raise ValidationError("ConstantVol.level must be positive")

    def apply(self, log_returns: np.ndarray, start_vol: float) -> np.ndarray:
        n_paths, n_ret = log_returns.shape
        return np.full((n_paths, n_ret + 1), self.level)


@dataclass(frozen=True)
class StickyRealisedVol:
    """``atm_vol_d = a + b * realised_vol(window)`` once ``window`` returns exist, ``start_vol`` before."""

    a: float
    b: float
    window: int

    def __post_init__(self) -> None:
        if int(self.window) < 1:
            raise ValidationError("StickyRealisedVol.window must be at least 1")
        if not (np.isfinite(self.a) and np.isfinite(self.b)):
            raise ValidationError("StickyRealisedVol coefficients must be finite")

    def apply(self, log_returns: np.ndarray, start_vol: float) -> np.ndarray:
        n_paths, n_ret = log_returns.shape
        out = np.full((n_paths, n_ret + 1), float(start_vol))
        w = int(self.window)
        for d in range(w, n_ret + 1):
            realised = log_returns[:, d - w: d].std(axis=1, ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)
            out[:, d] = self.a + self.b * realised
        if np.any(out <= 0.0):
            raise ValidationError("StickyRealisedVol produced a non-positive vol; raise a or b")
        return out


class GBMPaths:
    """Geometric Brownian spot on a fixed calendar with a held or scheduled carry curve."""

    def __init__(
        self,
        *,
        start: StartState,
        calendar: pd.DatetimeIndex,
        mu: float,
        sigma: float,
        vol_rule: Union[ConstantVol, StickyRealisedVol],
        carry_schedule: Optional[np.ndarray],
        rate: float,
        tenor_grid: np.ndarray,
    ) -> None:
        if not np.isfinite(mu) or not sigma > 0.0 or not np.isfinite(rate):
            raise ValidationError("mu and rate must be finite and sigma positive")
        grid = _validate_tenor_grid(tenor_grid)
        if grid.size != start.carry.size:
            raise ValidationError("start.carry must be on tenor_grid")
        self.tenor_grid = grid
        if carry_schedule is not None:
            carry_schedule = np.asarray(carry_schedule, dtype=float)
            if carry_schedule.ndim != 2 or carry_schedule.shape[1] != start.carry.size:
                raise ValidationError("carry_schedule must have shape (n_days, n_tenors)")
        self.start, self.calendar = start, pd.DatetimeIndex(calendar)
        self.mu, self.sigma, self.vol_rule, self.rate = float(mu), float(sigma), vol_rule, float(rate)
        self.carry_schedule = carry_schedule

    def generate(self, n_paths: int, n_days: int, *, seed: int) -> MarketPath:
        if n_paths < 1 or n_days < 1:
            raise ValidationError("n_paths and n_days must be positive")
        if n_days > len(self.calendar):
            raise ValidationError(f"calendar has {len(self.calendar)} days, {n_days} requested")
        if self.carry_schedule is not None and self.carry_schedule.shape[0] < n_days:
            raise ValidationError("carry_schedule is shorter than n_days")
        rng = np.random.default_rng(int(seed))
        dt = 1.0 / TRADING_DAYS_PER_YEAR
        z = rng.standard_normal((n_paths, n_days - 1))
        log_returns = (self.mu - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * z
        log_spot = np.log(self.start.spot) + np.concatenate(
            [np.zeros((n_paths, 1)), np.cumsum(log_returns, axis=1)], axis=1
        )
        vol = self.vol_rule.apply(log_returns, self.start.atm_vol)
        n_tenors = self.start.carry.size
        if self.carry_schedule is None:
            carry = np.broadcast_to(self.start.carry, (n_paths, n_days, n_tenors)).copy()
        else:
            carry = np.broadcast_to(self.carry_schedule[:n_days], (n_paths, n_days, n_tenors)).copy()
        meta = {"generator": "gbm", "seed": int(seed), "mu": self.mu, "sigma": self.sigma,
                "vol_rule": {"type": type(self.vol_rule).__name__, **self.vol_rule.__dict__},
                "carry_schedule": self.carry_schedule is not None, "rate": self.rate,
                "start": {"spot": self.start.spot, "atm_vol": self.start.atm_vol, "rate": self.start.rate}}
        return MarketPath(dates=self.calendar[:n_days], spot=np.exp(log_spot), atm_vol=vol,
                          rate=np.full((n_paths, n_days), self.rate), carry=carry,
                          tenor_grid=self.tenor_grid.copy(), meta=meta)
