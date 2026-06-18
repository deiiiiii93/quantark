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
            if not np.isfinite(s.to_numpy(dtype=float)).all():
                raise ValidationError(f"factor {k}: non-finite level")
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


class DriftMode(Enum):
    EMPIRICAL_MEAN = "empirical_mean"
    ZERO_LOG_MEAN = "zero_log_mean"
    USER_SUPPLIED = "user_supplied"


@dataclass
class HistoricalCalibration:
    """Real-world calibration: adjusted log-return drift, EWMA conditional vol, and
    a correlation *diagnostic* (never used to recolour generated paths)."""

    data: HistoricalMarketDataSet
    user_drift: dict = field(default_factory=dict)   # {key: daily log drift} for USER_SUPPLIED
    vol_floor: float = 1e-8

    def __post_init__(self):
        self._r = self.data.log_returns()
        if len(self._r) < 2:
            raise ValidationError("need >= 2 returns")

    def mu_hat(self, key) -> float:
        return float(self._r[key].mean())

    def _target_mu(self, key, mode) -> float:
        if mode is DriftMode.EMPIRICAL_MEAN:
            return self.mu_hat(key)
        if mode is DriftMode.ZERO_LOG_MEAN:
            return 0.0
        if mode is DriftMode.USER_SUPPLIED:
            if key not in self.user_drift:
                raise ValidationError(f"USER_SUPPLIED drift missing for {key}")
            return float(self.user_drift[key])
        raise ValidationError(f"unknown drift mode {mode}")

    def adjusted_log_returns(self, keys, modes):
        out = self._r[list(keys)].copy()
        for key in keys:
            if key not in modes:
                raise ValidationError(f"no drift mode for factor {key}")
            out[key] = out[key] - self.mu_hat(key) + self._target_mu(key, modes[key])
        return out

    def ewma_sigma(self, key, lam=0.94) -> np.ndarray:
        if not (0.0 < lam < 1.0):
            raise ValidationError("EWMA lambda must be in (0,1)")
        r = self._r[key].to_numpy()
        mu = self.mu_hat(key)
        var = np.empty(len(r))
        var[0] = np.var(r, ddof=1)                   # sample variance seed
        for t in range(1, len(r)):
            var[t] = lam * var[t - 1] + (1 - lam) * (r[t - 1] - mu) ** 2
        return np.maximum(np.sqrt(var), self.vol_floor)

    def ewma_sigma_today(self, key, lam=0.94) -> float:
        r = self._r[key].to_numpy()
        mu = self.mu_hat(key)
        sig = self.ewma_sigma(key, lam)
        var_today = lam * sig[-1] ** 2 + (1 - lam) * (r[-1] - mu) ** 2
        return float(max(np.sqrt(var_today), self.vol_floor))

    def standardized_residuals(self, key, lam=0.94) -> np.ndarray:
        r = self._r[key].to_numpy()
        mu = self.mu_hat(key)
        return (r - mu) / self.ewma_sigma(key, lam)

    def correlation_diagnostic(self) -> np.ndarray:
        # DIAGNOSTIC ONLY — never a recolouring step; co-movement comes from the
        # multivariate common-time-index resampling in path_generator.
        from quantark.util.numerical import Tolerance
        Z = np.column_stack([self.standardized_residuals(k) for k in self._r.columns])
        with np.errstate(invalid="ignore", divide="ignore"):   # degenerate factor -> NaN, caught below
            C = np.atleast_2d(np.corrcoef(Z, rowvar=False))
        if not np.all(np.isfinite(C)):
            raise ValidationError("non-finite correlation diagnostic (degenerate factor)")
        if np.min(np.linalg.eigvalsh(C)) < -Tolerance.ZERO:
            raise ValidationError("correlation diagnostic is not PSD")
        return C
