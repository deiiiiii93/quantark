"""Historical forward path generation: replay (raw / drift-adjusted) and
bootstrap (IID / block-FHS / stationary). Output is a real-world state tensor
``states[n_paths, n_grid, n_factors]`` consumed unchanged by the (provisional)
repricer.

Provisional-grid note: the grid is **year-fraction** based; each interval is
mapped to a deterministic business-day count and levels are built by compounding
that many daily log-returns (NO sqrt-t scaling). A full calendar/event grid is
the MC session's ``ExposureGrid`` (merge follow-up).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_zero
from quantark.sacva.exposure.historical.calibration import HistoricalCalibration
from quantark.sacva.exposure.historical.resampling import Resampler, ResamplingScheme

from enum import Enum


class PathMode(Enum):
    REPLAY_RAW = "replay_raw"
    REPLAY_DRIFT_ADJUSTED = "replay_drift_adjusted"
    BOOTSTRAP = "bootstrap"


@dataclass
class HistoricalPathGenerator:
    calibration: HistoricalCalibration
    factor_keys: tuple
    today_levels: dict
    lam: float = 0.94
    min_replay_windows: int = 20
    business_days_per_year: float = 252.0

    def __post_init__(self):
        for k in self.factor_keys:
            if k not in self.today_levels:
                raise ValidationError(f"missing today level for {k}")

    def _step_days(self, grid):
        if grid.ndim != 1 or len(grid) < 2 or not np.all(np.isfinite(grid)):
            raise ValidationError("grid_times must be a finite 1D array with >= 2 points")
        if not is_zero(grid[0]):
            raise ValidationError("grid_times must start at 0")
        if np.any(np.diff(grid) <= 0):
            raise ValidationError("grid_times must be strictly increasing")
        days = np.rint(np.diff(grid) * self.business_days_per_year).astype(int)
        if np.any(days < 1):
            raise ValidationError("grid interval shorter than one business day")
        return days

    def _today_vec(self):
        return np.array([self.today_levels[k] for k in self.factor_keys], float)

    def generate(self, path_mode, grid, *, scheme=None, block_length=None,
                 expected_block_length=None, n_paths=None, seed=0, drift_modes=None):
        grid = np.asarray(grid, float)
        days = self._step_days(grid)
        n_steps = int(days.sum())
        keys = list(self.factor_keys)
        if path_mode in (PathMode.REPLAY_RAW, PathMode.REPLAY_DRIFT_ADJUSTED):
            step_r = self._replay(path_mode, n_steps, keys, drift_modes)
        elif path_mode is PathMode.BOOTSTRAP:
            if scheme is None or n_paths is None:
                raise ValidationError("BOOTSTRAP requires scheme and n_paths")
            step_r = self._bootstrap(scheme, block_length, expected_block_length,
                                     n_paths, n_steps, seed, keys, drift_modes)
        else:
            raise ValidationError(f"unknown path_mode {path_mode}")
        return self._levels_on_grid(self._today_vec(), step_r, days)

    def _replay(self, path_mode, n_steps, keys, drift_modes):
        if path_mode is PathMode.REPLAY_RAW:
            R = self.calibration._r[keys].to_numpy()   # drift modes irrelevant (pure replay)
        else:
            if drift_modes is None:
                raise ValidationError("REPLAY_DRIFT_ADJUSTED requires explicit drift_modes")
            R = self.calibration.adjusted_log_returns(keys, drift_modes).to_numpy()
        n_windows = R.shape[0] - n_steps + 1
        if n_windows < self.min_replay_windows:
            raise ValidationError(
                f"insufficient replay windows {n_windows} < {self.min_replay_windows} "
                f"(history too short; no silent bootstrap fallback)")
        out = np.empty((n_windows, n_steps, len(keys)))
        for w in range(n_windows):
            out[w] = R[w:w + n_steps]
        return out

    def _bootstrap(self, scheme, block_length, expected_block_length,
                   n_paths, n_steps, seed, keys, drift_modes):
        if drift_modes is None:
            raise ValidationError("BOOTSTRAP requires explicit drift_modes")
        modes = drift_modes
        # min_raw_obs counts aligned LEVELS; resampled vectors are RETURNS (one fewer),
        # so the equivalent return-vector minimum is min_raw_obs - 1.
        min_obs = max(self.calibration.data.min_raw_obs - 1, 0)
        if scheme is ResamplingScheme.BLOCK_FHS:
            Z = np.column_stack(
                [self.calibration.standardized_residuals(k, self.lam) for k in keys])
            res = Resampler(scheme, block_length=block_length, seed=seed)
            z = res.sample(Z, n_paths, n_steps, min_raw_obs=min_obs)
            mu = np.array([self.calibration._target_mu(k, modes[k]) for k in keys])
            sig0 = np.array([self.calibration.ewma_sigma_today(k, self.lam) for k in keys])
            return self._fhs_reinflate(z, mu, sig0)
        R = self.calibration.adjusted_log_returns(keys, modes).to_numpy()
        res = Resampler(scheme, block_length=block_length,
                        expected_block_length=expected_block_length, seed=seed)
        return res.sample(R, n_paths, n_steps, min_raw_obs=min_obs)

    def _fhs_reinflate(self, z, mu, sig0):
        # pathwise EWMA recursion: r_k = mu + sig_k z_k; sig^2_{k+1}=λσ²_k+(1-λ)(r_k-mu)²
        n_paths, n_steps, _n_fac = z.shape
        out = np.empty_like(z)
        var = np.tile(sig0 ** 2, (n_paths, 1))
        floor2 = self.calibration.vol_floor ** 2
        for k in range(n_steps):
            sig = np.sqrt(np.maximum(var, floor2))
            r = mu + sig * z[:, k, :]
            out[:, k, :] = r
            var = self.lam * var + (1 - self.lam) * (r - mu) ** 2
        return out

    def _levels_on_grid(self, S0, step_r, days):
        cum = np.cumsum(step_r, axis=1)
        levels_daily = S0 * np.exp(cum)
        boundaries = np.concatenate([[0], np.cumsum(days)])
        cols = [np.broadcast_to(S0, (step_r.shape[0], len(S0)))]
        for b in boundaries[1:]:
            cols.append(levels_daily[:, b - 1, :])
        return np.stack(cols, axis=1)
