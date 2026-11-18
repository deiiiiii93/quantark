"""
The two models of the real-data study, on a real date.

* **Test model** — Black on the index as one underlying (q = 0, the index is
  total return), vol = the index's own trailing historical vol; per-name
  delta = Δ_I · nᵢ (the whole-index order, split by current weights).
* **Reference** — the synthetic study's multi-name RQMC engine with per-name
  trailing historical vols and correlation, the real rebalance schedule
  (spec D10: ``elapsed = 0``, shares encoded through the rebalance anchor) and
  CRN central bumps per name; pathwise per-name deltas are kept for the exact
  Euler decomposition (spec D11).

Clock: one year = 244 sessions; every time is (sessions from the as-of date) /
244 on the exchange calendar, extended by weekdays past its horizon (spec D6).
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from quantark.util.exceptions import ValidationError
from quantark.util.numerical import Tolerance, safe_exp, safe_log, safe_sqrt

HERE = Path(__file__).resolve().parent


def load_module(alias: str, path: Path):
    """Load a sibling file under a unique module name (other studies also have ``_common`` etc.)."""
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


cm = load_module("csi_basket_delta_common", HERE.parent / "csi_basket_delta" / "_common.py")
_index = load_module("csi_real_data_index", HERE / "_index.py")

TRADING_DAYS = _index.TRADING_DAYS


# ---------------------------------------------------------------------------
# Calendar and schedule
# ---------------------------------------------------------------------------


def _position(cal: pd.DatetimeIndex, day) -> int:
    day = pd.Timestamp(day)
    if day not in cal:
        raise ValidationError(f"{day.date()} is not a session of the calendar")
    return int(cal.get_loc(day))


def extended_calendar(calendar: pd.DatetimeIndex, until) -> pd.DatetimeIndex:
    """The exchange calendar, extended by weekdays past its last session up to ``until``."""
    cal = pd.DatetimeIndex(calendar).sort_values()
    until = pd.Timestamp(until)
    if until <= cal[-1]:
        return cal
    extra = pd.bdate_range(cal[-1] + pd.Timedelta(days=1), until)
    return cal.append(extra)


def expiry_date(calendar: pd.DatetimeIndex, start, sessions: int = TRADING_DAYS) -> Tuple[pd.Timestamp, bool]:
    """The session ``sessions`` after ``start`` (spec D2) and whether the weekday fallback was used."""
    cal = pd.DatetimeIndex(calendar).sort_values()
    start = pd.Timestamp(start)
    pos = _position(cal, start)
    if pos + sessions < len(cal):
        return cal[pos + sessions], False
    ext = extended_calendar(cal, start + pd.Timedelta(days=int(sessions * 1.6) + 30))
    return ext[pos + sessions], True


def sessions_between(calendar: pd.DatetimeIndex, a, b) -> int:
    cal = extended_calendar(calendar, b)
    return _position(cal, b) - _position(cal, a)


def remaining_schedule(calendar: pd.DatetimeIndex, rule: str, asof, expiry) -> np.ndarray:
    """Engine schedule from ``asof``: rule rebalance dates strictly inside (asof, expiry), then expiry.

    Times are sessions from ``asof`` / 244 on the (extended) exchange calendar.
    """
    asof, expiry = pd.Timestamp(asof), pd.Timestamp(expiry)
    if expiry <= asof:
        raise ValidationError(f"expiry {expiry.date()} must be after the as-of date {asof.date()}")
    cal = extended_calendar(calendar, expiry)
    rebs = _index.rebalance_dates(cal, rule, asof, expiry)
    rebs = rebs[rebs < expiry]
    p0 = _position(cal, asof)
    counts = [_position(cal, d) - p0 for d in rebs] + [_position(cal, expiry) - p0]
    return np.asarray(counts, dtype=float) / TRADING_DAYS


# ---------------------------------------------------------------------------
# Calibration (spec D13: demeaned daily log returns, ddof = 1, x244)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Calibration:
    """Trailing-window historical parameters on one as-of date."""

    vols: np.ndarray
    corr: np.ndarray
    index_vol: float
    n_obs: int
    psd_repaired: bool


def _psd_corr(corr: np.ndarray) -> Tuple[np.ndarray, bool]:
    vals, vecs = np.linalg.eigh(corr)
    if vals.min() >= -Tolerance.EIGENVALUE_MIN:          # rounding-level negatives are accepted by the engine
        return corr, False
    fixed = (vecs * np.clip(vals, 0.0, None)) @ vecs.T
    d = np.sqrt(np.diag(fixed))
    return fixed / np.outer(d, d), True


def calibrate(prices: pd.DataFrame, index_level: pd.Series, asof, window: int = TRADING_DAYS) -> Calibration:
    """Per-name vols/correlation and the index's own vol from the ``window`` returns ending at ``asof``."""
    asof = pd.Timestamp(asof)
    px = prices.loc[:asof]
    lv = index_level.loc[:asof]
    if len(px) < window + 1 or len(lv) < window + 1:
        raise ValidationError(f"need {window + 1} closes up to {asof.date()}, have {min(len(px), len(lv))}")
    if px.index[-1] != asof or not px.index[-(window + 1):].equals(lv.index[-(window + 1):]):
        raise ValidationError(f"{asof.date()} is not a price session, or prices and index levels misalign")
    r = np.diff(np.log(px.iloc[-(window + 1):].to_numpy(dtype=float)), axis=0)
    cov = np.cov(r, rowvar=False, ddof=1) * TRADING_DAYS
    vols = np.sqrt(np.diag(cov))
    corr = cov / np.outer(vols, vols)
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    corr, repaired = _psd_corr(corr)
    ri = np.diff(np.log(lv.iloc[-(window + 1):].to_numpy(dtype=float)))
    index_vol = float(np.std(ri, ddof=1) * np.sqrt(TRADING_DAYS))
    return Calibration(vols=vols, corr=corr, index_vol=index_vol, n_obs=int(window), psd_repaired=repaired)


# ---------------------------------------------------------------------------
# Test model: Black on the index
# ---------------------------------------------------------------------------


def black_call(index_level: float, strike: float, T: float, vol: float, rate: float) -> Tuple[float, float, float]:
    """Black–Scholes call on the index with q = 0, per index unit: (pv, Δ_I, Γ_I)."""
    if T <= 0.0 or vol <= 0.0:
        raise ValidationError("black_call needs T > 0 and vol > 0")
    v = vol * safe_sqrt(T)
    d1 = (safe_log(index_level / strike) + (rate + 0.5 * vol * vol) * T) / v
    d2 = d1 - v
    pv = index_level * norm.cdf(d1) - strike * safe_exp(-rate * T) * norm.cdf(d2)
    return float(pv), float(norm.cdf(d1)), float(norm.pdf(d1) / (index_level * v))


# ---------------------------------------------------------------------------
# Reference: per-name RQMC on the synthetic study's engine
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RefResult:
    """Reference values per index unit (callers scale by their own units)."""

    pv: float
    pv_se: float
    delta_index: float
    delta_index_se: float
    deltas_bump: np.ndarray
    deltas_bump_se: np.ndarray
    deltas_pathwise: np.ndarray
    deltas_pathwise_se: np.ndarray
    noise_cov: Optional[np.ndarray] = None   # covariance of the means of [bump deltas..., Δ_I] across batches


def engine_basket(cal: Calibration, weights_target: np.ndarray, rule: str, rate: float) -> "cm.CSIBasket":
    n = len(weights_target)
    return cm.CSIBasket(
        names=tuple(f"N{i}" for i in range(n)),
        target_weights=np.asarray(weights_target, dtype=float),
        vols=np.asarray(cal.vols, dtype=float),
        div_yields=np.zeros(n),
        corr=np.asarray(cal.corr, dtype=float),
        rate=float(rate),
        decrement=0.0,
        rebalance=rule,
    )


def engine_state(spots: np.ndarray, shares: np.ndarray, weights_target: np.ndarray) -> "cm.CSIState":
    """Spec D10: elapsed = 0, current shares encoded through the rebalance anchor."""
    spots = np.asarray(spots, dtype=float)
    shares = np.asarray(shares, dtype=float)
    level = float(shares @ spots)
    anchor = np.asarray(weights_target, dtype=float) * level / shares
    return cm.CSIState(level, anchor, spots / anchor, 0.0)


def reference_call(spots, shares, weights_target, cal: Calibration, rule: str, strike: float,
                   schedule: np.ndarray, rate: float, n_paths: int, n_batches: int, seed: int) -> RefResult:
    """Reference call per index unit with a fixed strike and the real schedule (spec D9–D11)."""
    basket = engine_basket(cal, weights_target, rule, rate)
    state = engine_state(spots, shares, weights_target)
    schedule = np.asarray(schedule, dtype=float)
    option = cm.CSIOption(1.0, float(schedule[-1]), True)
    res = cm.mc_price_and_deltas(basket, state, option, n_paths=n_paths, n_batches=n_batches, seed=seed,
                                 strike=float(strike), schedule=schedule)
    stack = np.column_stack([np.asarray(res.batches["deltas_fd"], dtype=float),
                             np.asarray(res.batches["delta_index"], dtype=float).reshape(-1, 1)])
    noise_cov = np.cov(stack, rowvar=False, ddof=1) / stack.shape[0]
    return RefResult(
        pv=float(res.price), pv_se=float(res.price_se),
        delta_index=float(res.delta_index), delta_index_se=float(res.delta_index_se),
        deltas_bump=np.asarray(res.deltas_fd, dtype=float), deltas_bump_se=np.asarray(res.deltas_fd_se, dtype=float),
        deltas_pathwise=np.asarray(res.deltas_pathwise, dtype=float),
        deltas_pathwise_se=np.asarray(res.deltas_pathwise_se, dtype=float),
        noise_cov=noise_cov,
    )


# ---------------------------------------------------------------------------
# Gap decomposition and the first-order split (spec D11, D12)
# ---------------------------------------------------------------------------


def gap_decomposition(ref_pathwise_cash: np.ndarray, test_dindex: float, ref_dindex: float,
                      weights: np.ndarray, index_level: float, units: float) -> Tuple[np.ndarray, np.ndarray]:
    """Per-name (total-delta part, split part) of test − reference(pathwise), CNY per 1% move.

    total part = wᵢ·I·(Δ_I^test − Δ_I^ref)·u·1%; split part = (wᵢ·I·Δ_I^ref − Sᵢ·Δᵢ^pw)·u·1%.
    """
    w = np.asarray(weights, dtype=float)
    scale = float(units) * 0.01
    total = w * index_level * (test_dindex - ref_dindex) * scale
    split = (w * index_level * ref_dindex - np.asarray(ref_pathwise_cash, dtype=float)) * scale
    return total, split


def first_order_split(cal: Calibration, weights: np.ndarray, index_level: float, strike: float, tau: float,
                      T: float, rate: float, units: float, rule: str) -> Tuple[np.ndarray, bool]:
    """Closed-form split part (M1 − REF) to first order in the stub τ, CNY per 1% (q = 0).

    −τ·wᵢ·I·(βᵢ − 1)·σ_I²·I·Γ·u·1% with β, σ_I² = wᵀΣw from the per-name calibration and Γ from
    Black at σ_I (spec D12).  Not meaningful for rule ``none`` (τ = T): flagged ``valid = False``.
    """
    w = np.asarray(weights, dtype=float)
    sigma = cal.vols[:, None] * cal.corr * cal.vols[None, :]
    var_index = float(w @ sigma @ w)
    beta = sigma @ w / var_index
    _, _, gamma = black_call(index_level, strike, T, float(np.sqrt(var_index)), rate)
    pred = -tau * w * index_level * (beta - 1.0) * var_index * index_level * gamma * units * 0.01
    return pred, rule != "none"
