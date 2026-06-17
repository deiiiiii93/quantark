"""Historical market data + real-world calibration for the historical exposure
engine (non-regulatory). Returns are adjusted **log returns**; vols are EWMA
(RiskMetrics-style); correlation is a validation *diagnostic*, never a generation
recoloring step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from quantark.util.exceptions import ValidationError


@dataclass
class HistoricalMarketDataSet:
    """Per-factor historical level series, aligned on a common business-day index."""

    levels: dict                  # {factor_key: pd.Series indexed by date}
    min_raw_obs: int = 250

    def __post_init__(self):
        if not self.levels:
            raise ValidationError("empty HistoricalMarketDataSet")
        for k, s in self.levels.items():
            if not isinstance(s, pd.Series) or s.isna().any():
                raise ValidationError(f"factor {k}: levels must be NaN-free Series")
            if not isinstance(s.index, pd.DatetimeIndex):
                raise ValidationError(f"factor {k}: index must be a DatetimeIndex")
            if not s.index.is_monotonic_increasing or s.index.has_duplicates:
                raise ValidationError(f"factor {k}: index must be strictly increasing dates")
            if (s <= 0).any():
                raise ValidationError(f"factor {k}: non-positive level (log return undefined)")

    def _aligned_levels(self) -> pd.DataFrame:
        df = pd.DataFrame(self.levels).dropna(how="any")   # intersection of dates
        if len(df) < self.min_raw_obs:
            raise ValidationError(
                f"insufficient aligned history: {len(df)} < min_raw_obs={self.min_raw_obs}")
        return df

    def valuation_date(self):
        return self._aligned_levels().index[-1]

    def log_returns(self) -> pd.DataFrame:
        return np.log(self._aligned_levels()).diff().dropna(how="any")

    def today_level(self, key: str) -> float:
        return float(self._aligned_levels()[key].iloc[-1])

    def reconcile_today(self, key: str, env_spot: float):
        from quantark.util.numerical import is_close
        if not is_close(self.today_level(key), env_spot):
            raise ValidationError(
                f"factor {key}: today level {self.today_level(key)} != env spot {env_spot}")
