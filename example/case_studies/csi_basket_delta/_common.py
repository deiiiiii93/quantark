"""
Shared pricer for the CSI basket-delta case study.

A customized strategy index (CSI) is a weighted basket of names (shares,
ETFs, indices).  The desk sells a European option on the index level and must
decide how to split its delta across the names.  This module prices that
option two ways and returns per-name deltas:

* a moment-matched (MM) lognormal on the index, the desk-style fast model,
  with an analytic per-name chain rule;
* an exact Monte Carlo reference (added in the MC section below) with
  pathwise per-name deltas.

Index rule
----------
Between rebalances the index holds fixed share counts ``n_i``; at each
rebalance date the counts reset to ``w_i * I / S_i``.  A continuous decrement
``d`` p.a. accrues on the level::

    I(t) = exp(-d (t - t_k)) * sum_i n_i(k) S_i(t)          t in [t_k, t_k+1)

Rebalance ``none`` is the fixed-share basket; ``daily`` is (close to) a
constant mix.  The current state is part-way through a period: the
decrement accrued since the last rebalance is folded into ``n_i`` so that
``I(now) = sum_i n_i S_i(now)`` holds exactly (spec D5).

Hedge-delta invariant (spec D14)
--------------------------------
Every delta holds the strike ``K`` and the unit count at the *unbumped*
``I(now)``.  A bump moves one name's price through ``CSIState.moves`` and the
strike is passed explicitly.

All times are year fractions on a 244-trading-day year (spec D4).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.stats import norm

# `import quantark.montecarlo` as the first quantark import hits a library
# import cycle (montecarlo.conditional_snowball -> asset -> snowball MC engine
# -> montecarlo.conditional_snowball).  Importing quantark.asset first breaks
# it; the cycle itself is out of scope for this study.
import quantark.asset  # noqa: F401  (import order, see above)
from quantark.montecarlo import SobolNormalGenerator, apply_brownian_bridge_multi_asset
from quantark.util.exceptions import NumericalError, ValidationError
from quantark.util.numerical import (
    Tolerance,
    is_close,
    is_zero,
    safe_exp,
    safe_log,
    safe_sqrt,
)

TRADING_DAYS = 244
SOBOL_MAX_DIM = 21201  # scipy's Sobol direction-number table

PERIODS: Dict[str, Optional[float]] = {
    "none": None,
    "quarterly": 0.25,
    "monthly": 1.0 / 12.0,
    "weekly": 5.0 / TRADING_DAYS,
    "daily": 1.0 / TRADING_DAYS,
}


def period_length(rebalance: str) -> Optional[float]:
    """Rebalance period in years, or ``None`` for a fixed-share basket."""
    if rebalance not in PERIODS:
        raise ValidationError(
            f"unknown rebalance '{rebalance}', expected one of {sorted(PERIODS)}"
        )
    return PERIODS[rebalance]


def _frozen_array(values, name: str, ndim: int) -> np.ndarray:
    arr = np.array(values, dtype=float, copy=True)
    if arr.ndim != ndim:
        raise ValidationError(f"{name} must be {ndim}-dimensional, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValidationError(f"{name} must be finite")
    arr.setflags(write=False)
    return arr


def corr_factor(corr: np.ndarray) -> np.ndarray:
    """Symmetric eigen square root ``F`` with ``F @ F.T == corr``.

    Works for singular PSD matrices (e.g. uniform rho = 1), where Cholesky
    fails.  Eigenvalues are clipped at zero; callers validate PSD first, so
    the clip only removes rounding noise.
    """
    vals, vecs = np.linalg.eigh(np.asarray(corr, dtype=float))
    vals = np.clip(vals, 0.0, None)
    return vecs @ np.diag(np.sqrt(vals)) @ vecs.T


# ----------------------------------------------------------------------------
# Contract objects
# ----------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class CSIBasket:
    """Index definition and per-name GBM dynamics.

    Attributes:
        names: Constituent labels.
        target_weights: Weights restored at each rebalance; sum to 1.
        vols: Lognormal vols, > 0.
        div_yields: Continuous dividend yields.
        corr: Correlation matrix (symmetric, unit diagonal, PSD; may be
            singular).
        rate: Flat continuously-compounded rate.
        decrement: Index decrement, p.a. (continuous).
        rebalance: One of ``none, quarterly, monthly, weekly, daily``.
    """

    names: Tuple[str, ...]
    target_weights: np.ndarray
    vols: np.ndarray
    div_yields: np.ndarray
    corr: np.ndarray
    rate: float
    decrement: float = 0.0
    rebalance: str = "none"

    def __post_init__(self) -> None:
        names = tuple(self.names)
        n = len(names)
        if n == 0:
            raise ValidationError("basket must have at least one name")
        object.__setattr__(self, "names", names)
        for attr in ("target_weights", "vols", "div_yields"):
            arr = _frozen_array(getattr(self, attr), attr, 1)
            if arr.shape != (n,):
                raise ValidationError(f"{attr} must have length {n}, got {arr.shape}")
            object.__setattr__(self, attr, arr)
        corr = _frozen_array(self.corr, "corr", 2)
        if corr.shape != (n, n):
            raise ValidationError(f"corr must be {n}x{n}, got {corr.shape}")
        object.__setattr__(self, "corr", corr)

        if not is_close(float(np.sum(self.target_weights)), 1.0, rel_tol=Tolerance.RELATIVE):
            raise ValidationError(
                f"target_weights must sum to 1, got {np.sum(self.target_weights)}"
            )
        if np.any(self.target_weights <= 0.0):
            raise ValidationError("target_weights must be positive")
        if np.any(self.vols <= 0.0):
            raise ValidationError("vols must be positive")
        if not np.allclose(corr, corr.T, rtol=0.0, atol=Tolerance.ZERO):
            raise ValidationError("corr must be symmetric")
        if not np.allclose(np.diag(corr), 1.0, rtol=0.0, atol=Tolerance.ZERO):
            raise ValidationError("corr must have a unit diagonal")
        if np.any(np.abs(corr) > 1.0 + Tolerance.ZERO):
            raise ValidationError("corr entries must lie in [-1, 1]")
        min_eig = float(np.min(np.linalg.eigvalsh(corr)))
        if min_eig < -Tolerance.EIGENVALUE_MIN:
            raise ValidationError(
                f"corr must be positive semi-definite, min eigenvalue {min_eig:.3e}"
            )
        if self.decrement < 0.0:
            raise ValidationError("decrement must be non-negative")
        period_length(self.rebalance)

    @property
    def n_names(self) -> int:
        return len(self.names)

    def covariance_daily(self) -> np.ndarray:
        """Daily log-return covariance diag(sigma) rho diag(sigma) / 244."""
        s = np.diag(self.vols)
        return s @ self.corr @ s / TRADING_DAYS


@dataclass(frozen=True, eq=False)
class CSIState:
    """Where we are inside the current rebalance period.

    Attributes:
        index_at_rebalance: Index level at the last rebalance, I(t0-).
        spots_at_rebalance: Name prices at the last rebalance.
        moves: S_i(now) / S_i(t0-); ones by default.
        elapsed: Years since the last rebalance (for ``none``: since launch,
            used only to fold the decrement).
    """

    index_at_rebalance: float
    spots_at_rebalance: np.ndarray
    moves: Optional[np.ndarray] = None
    elapsed: float = 0.0

    def __post_init__(self) -> None:
        spots = _frozen_array(self.spots_at_rebalance, "spots_at_rebalance", 1)
        object.__setattr__(self, "spots_at_rebalance", spots)
        moves = np.ones_like(spots) if self.moves is None else self.moves
        moves = _frozen_array(moves, "moves", 1)
        if moves.shape != spots.shape:
            raise ValidationError("moves must match spots_at_rebalance")
        object.__setattr__(self, "moves", moves)
        if self.index_at_rebalance <= 0.0 or np.any(spots <= 0.0) or np.any(moves <= 0.0):
            raise ValidationError("index level, spots and moves must be positive")
        if self.elapsed < 0.0:
            raise ValidationError("elapsed must be non-negative")


@dataclass(frozen=True)
class CSIOption:
    """European option on the index, struck today (spec D3).

    Attributes:
        moneyness: K / I(now).
        maturity: Years to expiry.
        is_call: Call if True, put otherwise.
        notional_cny: Index notional today; units = notional / I(now).
    """

    moneyness: float
    maturity: float
    is_call: bool = True
    notional_cny: float = 1e8

    def __post_init__(self) -> None:
        if self.moneyness <= 0.0:
            raise ValidationError("moneyness must be positive")
        if self.maturity <= 0.0 or is_zero(self.maturity):
            raise ValidationError("maturity must be positive")
        if self.notional_cny <= 0.0:
            raise ValidationError("notional_cny must be positive")

    def strike(self, index_now: float) -> float:
        return self.moneyness * index_now

    def units(self, index_now: float) -> float:
        return self.notional_cny / index_now


@dataclass(frozen=True)
class StateQuantities:
    """Derived current-state quantities."""

    shares: np.ndarray
    spots: np.ndarray
    index_level: float
    current_weights: np.ndarray


def state_quantities(basket: CSIBasket, state: CSIState) -> StateQuantities:
    """Current share counts (decrement folded in), prices, level and weights."""
    if state.spots_at_rebalance.shape != (basket.n_names,):
        raise ValidationError("state has the wrong number of names for the basket")
    period = period_length(basket.rebalance)
    if period is not None and not (0.0 <= state.elapsed < period - Tolerance.ZERO):
        raise ValidationError(
            f"elapsed {state.elapsed} outside [0, {period}) for rebalance '{basket.rebalance}'"
        )
    shares = (
        safe_exp(-basket.decrement * state.elapsed)
        * basket.target_weights
        * state.index_at_rebalance
        / state.spots_at_rebalance
    )
    spots = state.spots_at_rebalance * state.moves
    level = float(np.dot(shares, spots))
    return StateQuantities(
        shares=shares,
        spots=spots,
        index_level=level,
        current_weights=shares * spots / level,
    )


def bump_state(state: CSIState, i: int, rel_bump: float) -> CSIState:
    """Same state with name ``i``'s price scaled by ``1 + rel_bump``."""
    moves = np.array(state.moves, dtype=float)
    moves[i] *= 1.0 + rel_bump
    return replace(state, moves=moves)


def scale_state(state: CSIState, factor: float) -> CSIState:
    """Same state with every name's price scaled by ``factor``."""
    return replace(state, moves=np.array(state.moves, dtype=float) * factor)


def schedule_times(basket: CSIBasket, state: CSIState, maturity: float) -> np.ndarray:
    """Simulation dates from now: ``[t1, rebalance dates..., T]``.

    ``t1`` is the end of the current (stub) period, or ``T`` if the stub
    reaches maturity.  Dates are ``t1 + k P``; any date within tolerance of
    ``T`` is dropped so there is no sliver period (spec D19).
    """
    period = period_length(basket.rebalance)
    if period is None:
        return np.array([maturity])
    t1 = period - state.elapsed
    if t1 > maturity or is_close(t1, maturity, rel_tol=Tolerance.RELATIVE):
        return np.array([maturity])
    n_max = int(np.floor((maturity - t1) / period)) + 1
    dates = t1 + period * np.arange(n_max)
    keep = [
        t for t in dates if t < maturity and not is_close(t, maturity, rel_tol=Tolerance.RELATIVE)
    ]
    return np.array(keep + [maturity])


# ----------------------------------------------------------------------------
# Moment-matched lognormal (desk-style fast model)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class MMResult:
    """Moment-matched price and deltas (per one index-unit option).

    Attributes:
        price: Option value per index unit.
        strike: Strike used.
        forward: E[I_T] (= m1).
        sigma_eff: Matched lognormal vol, sqrt(ln(m2/m1^2)/T).
        m1, m2: First two moments of I_T.
        delta_index: dV/dI along a proportional bump of all names.
        deltas: dV/dS_i, per name, strike held fixed.
    """

    price: float
    strike: float
    forward: float
    sigma_eff: float
    m1: float
    m2: float
    delta_index: float
    deltas: np.ndarray


def _moments_and_gradients(
    basket: CSIBasket, q: StateQuantities, times: np.ndarray
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """m1, m2 of I_T and their gradients in S_i(now) (shares held fixed)."""
    r, d = basket.rate, basket.decrement
    carry = r - basket.div_yields
    cov = np.outer(basket.vols, basket.vols) * basket.corr

    # stub [0, t1] with current shares
    tau = float(times[0])
    a = q.shares * q.spots * safe_exp(carry * tau)
    e = np.exp(cov * tau)
    ea = e @ a
    m1 = safe_exp(-d * tau) * float(np.sum(a))
    m2 = safe_exp(-2.0 * d * tau) * float(a @ ea)
    dm1 = safe_exp(-d * tau) * a / q.spots
    dm2 = safe_exp(-2.0 * d * tau) * 2.0 * (a / q.spots) * ea

    # full periods with target weights: independent factors
    f1_total, f2_total = 1.0, 1.0
    for dt in np.diff(times):
        aw = basket.target_weights * safe_exp(carry * dt)
        ew = np.exp(cov * dt)
        f1_total *= safe_exp(-d * dt) * float(np.sum(aw))
        f2_total *= safe_exp(-2.0 * d * dt) * float(aw @ ew @ aw)
    return float(m1 * f1_total), float(m2 * f2_total), dm1 * f1_total, dm2 * f2_total


def mm_price_and_deltas(
    basket: CSIBasket,
    state: CSIState,
    option: CSIOption,
    strike: Optional[float] = None,
) -> MMResult:
    """Moment-matched lognormal price and analytic per-name deltas.

    Args:
        basket: Index definition.
        state: Current state.
        option: The option; strike = moneyness * I(now) unless given.
        strike: Explicit strike (used by bumped revaluations, spec D14).
    """
    q = state_quantities(basket, state)
    k = option.strike(q.index_level) if strike is None else float(strike)
    t = option.maturity
    times = schedule_times(basket, state, t)
    m1, m2, dm1, dm2 = _moments_and_gradients(basket, q, times)

    v = float(safe_log(m2 / (m1 * m1)))
    if v <= 0.0 or is_zero(v):
        # TODO: zero matched variance needs its own intrinsic-forward branch;
        # no study scenario reaches it (all vols > 0).
        raise NumericalError(f"matched total variance {v:.3e} is not positive")
    sv = safe_sqrt(v)
    df = safe_exp(-basket.rate * t)
    d1 = (safe_log(m1 / k) + 0.5 * v) / sv
    d2 = d1 - sv
    if option.is_call:
        price = df * (m1 * norm.cdf(d1) - k * norm.cdf(d2))
        v_f = df * norm.cdf(d1)
    else:
        price = df * (k * norm.cdf(-d2) - m1 * norm.cdf(-d1))
        v_f = -df * norm.cdf(-d1)
    v_v = df * m1 * norm.pdf(d1) / (2.0 * sv)

    dv = dm2 / m2 - 2.0 * dm1 / m1
    deltas = v_f * dm1 + v_v * dv
    delta_index = v_f * m1 / q.index_level
    return MMResult(
        price=float(price),
        strike=k,
        forward=m1,
        sigma_eff=float(safe_sqrt(v / t)),
        m1=m1,
        m2=m2,
        delta_index=float(delta_index),
        deltas=deltas,
    )


def mm_fd_deltas(
    basket: CSIBasket, state: CSIState, option: CSIOption, bump: float = 1e-5
) -> np.ndarray:
    """Central FD of the MM price in each S_i, strike held fixed (AC8)."""
    q = state_quantities(basket, state)
    k = option.strike(q.index_level)
    out = np.empty(basket.n_names)
    for i in range(basket.n_names):
        up = mm_price_and_deltas(basket, bump_state(state, i, bump), option, strike=k).price
        dn = mm_price_and_deltas(basket, bump_state(state, i, -bump), option, strike=k).price
        out[i] = (up - dn) / (2.0 * bump * q.spots[i])
    return out


def mm_third_derivatives(
    basket: CSIBasket, state: CSIState, option: CSIOption, bump: float = 1e-3
) -> np.ndarray:
    """d^3 V^MM / dS_i^3 by FD of the analytic delta (for the D18 allowance)."""
    q = state_quantities(basket, state)
    k = option.strike(q.index_level)
    out = np.empty(basket.n_names)
    for i in range(basket.n_names):
        h = bump * q.spots[i]
        up = mm_price_and_deltas(basket, bump_state(state, i, bump), option, strike=k).deltas[i]
        mid = mm_price_and_deltas(basket, state, option, strike=k).deltas[i]
        dn = mm_price_and_deltas(basket, bump_state(state, i, -bump), option, strike=k).deltas[i]
        out[i] = (up - 2.0 * mid + dn) / (h * h)
    return out


# ----------------------------------------------------------------------------
# Scenario factories (design section 5)
# ----------------------------------------------------------------------------

BASE_NAMES = ("LC1", "LC2", "TECH", "ETF", "GOLD")
BASE_WEIGHTS = np.array([0.25, 0.20, 0.15, 0.30, 0.10])
BASE_VOLS = np.array([0.25, 0.25, 0.45, 0.18, 0.10])
BASE_DIVS = np.array([0.025, 0.030, 0.003, 0.015, 0.0])
BASE_SPOTS = np.array([20.0, 35.0, 80.0, 4.0, 6.0])
BASE_RATE = 0.02


def base_corr() -> np.ndarray:
    """Equities/ETF at 0.6, ETF vs large caps 0.85, GOLD at 0.05."""
    c = np.full((5, 5), 0.6)
    c[3, 0] = c[0, 3] = 0.85
    c[3, 1] = c[1, 3] = 0.85
    c[4, :] = c[:, 4] = 0.05
    np.fill_diagonal(c, 1.0)
    return c


def uniform_corr(n: int, rho: float) -> np.ndarray:
    c = np.full((n, n), float(rho))
    np.fill_diagonal(c, 1.0)
    return c


def base_basket(rebalance: str = "none", decrement: float = 0.0, **over) -> CSIBasket:
    """The stylised 5-name CSI."""
    kw: Dict[str, Any] = dict(
        names=BASE_NAMES,
        target_weights=BASE_WEIGHTS,
        vols=BASE_VOLS,
        div_yields=BASE_DIVS,
        corr=base_corr(),
        rate=BASE_RATE,
        decrement=decrement,
        rebalance=rebalance,
    )
    kw.update(over)
    return CSIBasket(**kw)


def control_basket(rebalance: str = "none") -> CSIBasket:
    """Exchangeable control: equal weights, vols, yields and correlation."""
    return CSIBasket(
        names=BASE_NAMES,
        target_weights=np.full(5, 0.2),
        vols=np.full(5, 0.25),
        div_yields=np.full(5, 0.015),
        corr=uniform_corr(5, 0.5),
        rate=BASE_RATE,
        rebalance=rebalance,
    )


def concentration_basket(rebalance: str = "none") -> CSIBasket:
    """Control dynamics with the base (unequal) weights: NOT a control."""
    return CSIBasket(
        names=BASE_NAMES,
        target_weights=BASE_WEIGHTS,
        vols=np.full(5, 0.25),
        div_yields=np.full(5, 0.015),
        corr=uniform_corr(5, 0.5),
        rate=BASE_RATE,
        rebalance=rebalance,
    )


def two_name_basket(rho: float, rebalance: str = "none") -> CSIBasket:
    """EQ (30%, w 0.6) + HEDGE (15%, w 0.4): the only place rho = -1 is feasible."""
    return CSIBasket(
        names=("EQ", "HEDGE"),
        target_weights=np.array([0.6, 0.4]),
        vols=np.array([0.30, 0.15]),
        div_yields=np.array([0.02, 0.0]),
        corr=uniform_corr(2, rho),
        rate=BASE_RATE,
        rebalance=rebalance,
    )


def default_state(basket: CSIBasket, elapsed: float = 0.0, moves=None) -> CSIState:
    """Index 1000 at the last rebalance, stylised spots (base basket prices)."""
    n = basket.n_names
    spots = BASE_SPOTS[:n] if n <= len(BASE_SPOTS) else np.full(n, 10.0)
    if n == 2:
        spots = np.array([30.0, 5.0])
    return CSIState(
        index_at_rebalance=1000.0,
        spots_at_rebalance=spots,
        moves=moves,
        elapsed=elapsed,
    )


# ----------------------------------------------------------------------------
# Monte Carlo reference (exact sampling at the schedule dates)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class MCResult:
    """RQMC estimates per one index-unit option, each with its SE across batches.

    ``batches`` holds the per-batch arrays (keys ``price``, ``delta_index``,
    ``deltas_pathwise``, ``deltas_fd``, ``m1``, ``m2``) so paired differences
    can be formed on the same paths (spec D7).
    """

    price: float
    price_se: float
    delta_index: float
    delta_index_se: float
    deltas_pathwise: np.ndarray
    deltas_pathwise_se: np.ndarray
    deltas_fd: np.ndarray
    deltas_fd_se: np.ndarray
    sample_m1: float
    sample_m1_se: float
    sample_m2: float
    sample_m2_se: float
    euler_max_rel_err: float
    strike: float
    fd_bump: float
    n_paths: int
    n_batches: int
    batches: Dict[str, np.ndarray] = field(repr=False, default_factory=dict)


def _mean_se(x: np.ndarray) -> Tuple[Any, Any]:
    """Mean and SE over axis 0 (batches)."""
    x = np.asarray(x, dtype=float)
    mean = x.mean(axis=0)
    if x.shape[0] < 2:
        return mean, np.full_like(mean, np.nan)
    return mean, x.std(axis=0, ddof=1) / np.sqrt(x.shape[0])


def _simulate_batch(
    basket: CSIBasket,
    q: StateQuantities,
    times: np.ndarray,
    factor: np.ndarray,
    gen: Any,
    n_paths: int,
    batch_id: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One RQMC batch: returns (I_T (P,), stub growth G1 (N, P), later product (P,)).

    One batch is exactly one ``SobolNormalGenerator.normal`` call (spec D17).
    Sobol dimension ``k * N + i`` drives bridge level ``k`` of name ``i``.
    """
    n, steps = basket.n_names, len(times)
    z = gen.normal(n_paths, n * steps, batch_id=batch_id)
    z = z.reshape(n_paths, steps, n).transpose(2, 0, 1)  # (N, P, S)
    dw = apply_brownian_bridge_multi_asset(z, times)
    del z
    dw = np.einsum("ij,jps->ips", factor, dw)  # correlate names per step
    dt = np.diff(np.concatenate(([0.0], times)))
    carry = basket.rate - basket.div_yields
    drift = (carry - 0.5 * basket.vols**2)[:, None] * dt[None, :]  # (N, S)
    g = np.exp(drift[:, None, :] + basket.vols[:, None, None] * dw)
    del dw

    d = basket.decrement
    g1 = g[:, :, 0]
    i1 = safe_exp(-d * times[0]) * np.einsum("i,ip->p", q.shares * q.spots, g1)
    if steps > 1:
        per = np.einsum("i,ips->ps", basket.target_weights, g[:, :, 1:])
        per *= np.exp(-d * dt[1:])[None, :]
        later = np.prod(per, axis=1)
    else:
        later = np.ones(n_paths)
    return i1 * later, g1, later


def mc_price_and_deltas(
    basket: CSIBasket,
    state: CSIState,
    option: CSIOption,
    n_paths: int = 2**14,
    n_batches: int = 32,
    seed: int = 20261117,
    fd_bump: float = Tolerance.BUMP_SPOT,
    strike: Optional[float] = None,
) -> MCResult:
    """Exact-sampling RQMC price, pathwise per-name deltas, Delta_I and CRN FD.

    The pathwise derivative only touches the stub (later periods restart from
    target weights)::

        dI_T/dS_i = exp(-d tau) n_i G_i I_T / I(t1)

    so ``sum_i S_i dI_T/dS_i = I_T`` holds on every path (Euler identity).
    The FD deltas (method M3) bump name ``i``'s price by ``fd_bump`` on the
    same paths, strike held fixed (spec D8, D14).

    Args:
        n_paths: Paths per RQMC batch; must be a power of two.
        n_batches: Independent scrambles; SEs are taken across batches.
    """
    if n_paths < 2 or n_batches < 1:
        raise ValidationError("need n_paths >= 2 and n_batches >= 1")
    q = state_quantities(basket, state)
    k = option.strike(q.index_level) if strike is None else float(strike)
    t = option.maturity
    times = schedule_times(basket, state, t)
    if basket.n_names * len(times) > SOBOL_MAX_DIM:
        raise ValidationError(f"Sobol dimension cap ({SOBOL_MAX_DIM}) exceeded")
    factor = corr_factor(basket.corr)
    gen = SobolNormalGenerator()
    gen.base_seed = int(seed)
    gen.strict_power_of_two = True
    df = safe_exp(-basket.rate * t)
    stub_dec = safe_exp(-basket.decrement * times[0])
    sign = 1.0 if option.is_call else -1.0

    keys = ("price", "delta_index", "deltas_pathwise", "deltas_fd", "m1", "m2")
    out: Dict[str, list] = {key: [] for key in keys}
    euler = 0.0
    for b in range(n_batches):
        i_t, g1, later = _simulate_batch(basket, q, times, factor, gen, n_paths, b)
        payoff = np.maximum(sign * (i_t - k), 0.0)
        itm = (sign * (i_t - k)) > 0.0
        d_it = stub_dec * (q.shares[:, None] * g1) * later[None, :]  # (N, P)
        euler = max(euler, float(np.max(np.abs(q.spots @ d_it - i_t) / i_t)))
        out["price"].append(df * payoff.mean())
        out["delta_index"].append(df * sign * np.mean(itm * i_t) / q.index_level)
        out["deltas_pathwise"].append(df * sign * np.mean(itm[None, :] * d_it, axis=1))
        fd = np.empty(basket.n_names)
        for i in range(basket.n_names):
            shift = stub_dec * q.shares[i] * fd_bump * q.spots[i] * g1[i] * later
            up = np.maximum(sign * (i_t + shift - k), 0.0)
            dn = np.maximum(sign * (i_t - shift - k), 0.0)
            fd[i] = df * np.mean(up - dn) / (2.0 * fd_bump * q.spots[i])
        out["deltas_fd"].append(fd)
        out["m1"].append(i_t.mean())
        out["m2"].append(np.mean(i_t * i_t))

    batches = {key: np.asarray(val, dtype=float) for key, val in out.items()}
    est = {key: _mean_se(val) for key, val in batches.items()}
    return MCResult(
        price=float(est["price"][0]),
        price_se=float(est["price"][1]),
        delta_index=float(est["delta_index"][0]),
        delta_index_se=float(est["delta_index"][1]),
        deltas_pathwise=est["deltas_pathwise"][0],
        deltas_pathwise_se=est["deltas_pathwise"][1],
        deltas_fd=est["deltas_fd"][0],
        deltas_fd_se=est["deltas_fd"][1],
        sample_m1=float(est["m1"][0]),
        sample_m1_se=float(est["m1"][1]),
        sample_m2=float(est["m2"][0]),
        sample_m2_se=float(est["m2"][1]),
        euler_max_rel_err=euler,
        strike=k,
        fd_bump=fd_bump,
        n_paths=n_paths,
        n_batches=n_batches,
        batches=batches,
    )


# ----------------------------------------------------------------------------
# Delta methods (design section 4) and metrics
# ----------------------------------------------------------------------------

METHOD_LABELS = {
    "REF": "MC pathwise reference",
    "M1": "Delta_I x current shares (whole-index order)",
    "M2": "Delta_I x target weights (shortcut)",
    "M3": "per-name FD, MC with common random numbers",
    "M4": "Delta_I x current shares, moment-matched",
    "M5": "per-name derivative, moment-matched",
}


@dataclass(frozen=True)
class MethodDeltas:
    """Per-name deltas of one method and its paired gap to the reference.

    Attributes:
        key: ``REF`` or ``M1``..``M5``.
        deltas: dV/dS_i per index-unit option (mean over batches for MC).
        diff_batches: Per-batch (method - REF), shape (B, N).
        diff_mean: Mean gap to REF.
        diff_se: Paired SE of that gap (spec D7).
    """

    key: str
    deltas: np.ndarray
    diff_batches: np.ndarray
    diff_mean: np.ndarray
    diff_se: np.ndarray


def delta_methods(
    basket: CSIBasket,
    state: CSIState,
    option: CSIOption,
    mc: MCResult,
    mm: MMResult,
) -> Dict[str, MethodDeltas]:
    """REF and M1..M5 per design section 4, with paired gaps to REF."""
    q = state_quantities(basket, state)
    ref_b = mc.batches["deltas_pathwise"]
    di_b = mc.batches["delta_index"][:, None]
    target_shares = basket.target_weights * q.index_level / q.spots
    per_batch = {
        "REF": ref_b,
        "M1": di_b * q.shares[None, :],
        "M2": di_b * target_shares[None, :],
        "M3": mc.batches["deltas_fd"],
    }
    fixed = {
        "M4": mm.delta_index * q.shares,
        "M5": np.asarray(mm.deltas, dtype=float),
    }
    result: Dict[str, MethodDeltas] = {}
    for key, arr in per_batch.items():
        diff = arr - ref_b
        dm, ds = _mean_se(diff)
        result[key] = MethodDeltas(key, arr.mean(axis=0), diff, dm, ds)
    for key, vec in fixed.items():
        diff = vec[None, :] - ref_b
        dm, ds = _mean_se(diff)
        result[key] = MethodDeltas(key, vec, diff, dm, ds)
    return result


def delta_cash_error_per_pct(
    method: MethodDeltas, spots: np.ndarray, units: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Gap to REF in CNY per 1% move of each name: (mean, SE)."""
    scale = spots * units * 0.01
    return method.diff_mean * scale, method.diff_se * scale


def cash_error_batches(method: MethodDeltas, spots: np.ndarray, units: float) -> np.ndarray:
    """Per-batch delta-cash errors (B, N), CNY per unit return of each name."""
    return method.diff_batches * (spots * units)[None, :]


def first_order_m1_gap(
    basket: CSIBasket, state: CSIState, option: CSIOption, rel_bump: float = 1e-3
) -> np.ndarray:
    """Closed-form M1 - REF to first order in the stub, CNY per 1% move.

    After the next rebalance date t1 the option depends on the names only
    through I(t1), so the whole error comes from the stub tau = t1 - t:

        REF_i - M1_i = tau w_i I [ (beta_i - 1) sigma_I^2 I Gamma + (qbar - q_i) Delta ]

    (cash delta per index unit), with current weights w, beta_i =
    (Sigma w)_i / sigma_I^2, sigma_I^2 = w' Sigma w, qbar = w'q, and Delta,
    Gamma the MM index delta and gamma (strike fixed).  The Ito drift of the
    weights cancels the Delta that a naive expansion puts in the vol term.
    """
    q = state_quantities(basket, state)
    w, level = q.current_weights, q.index_level
    tau = float(schedule_times(basket, state, option.maturity)[0])
    sig = np.diag(basket.vols) @ basket.corr @ np.diag(basket.vols)
    var_index = float(w @ sig @ w)
    beta = sig @ w / var_index
    q_bar = float(w @ basket.div_yields)
    mm = mm_price_and_deltas(basket, state, option)
    up = mm_price_and_deltas(basket, scale_state(state, 1 + rel_bump), option, strike=mm.strike).price
    dn = mm_price_and_deltas(basket, scale_state(state, 1 - rel_bump), option, strike=mm.strike).price
    gamma = (up - 2.0 * mm.price + dn) / (level * rel_bump) ** 2
    ref_minus_m1 = tau * w * level * (
        (beta - 1.0) * var_index * level * gamma + (q_bar - basket.div_yields) * mm.delta_index
    )
    return -ref_minus_m1 * option.units(level) * 0.01


def fd_bias_allowance(
    basket: CSIBasket, state: CSIState, option: CSIOption, fd_bump: float
) -> np.ndarray:
    """D18: |(h S_i)^2 / 6 * d^3V^MM/dS_i^3| per name (central-FD bias)."""
    q = state_quantities(basket, state)
    third = mm_third_derivatives(basket, state, option)
    return np.abs((fd_bump * q.spots) ** 2 / 6.0 * third)


@dataclass(frozen=True)
class ResidualSigma:
    """Daily P&L sigma of a delta mismatch (spec D16).

    Attributes:
        sigma: sqrt(max(q, 0)) in CNY.
        q: Bias-corrected e'Ce - tr(C Cov(e_bar)).
        q_se: Delta-method SE of q.
        resolved: q > 3 q_se.
    """

    sigma: float
    q: float
    q_se: float
    resolved: bool


def residual_daily_pnl_sigma(
    err_cash_batches: np.ndarray, cov_daily: np.ndarray
) -> ResidualSigma:
    """Bias-corrected sqrt(e' C e) from per-batch cash-delta errors (B, N)."""
    e = np.asarray(err_cash_batches, dtype=float)
    c = np.asarray(cov_daily, dtype=float)
    b = e.shape[0]
    e_bar = e.mean(axis=0)
    if b > 1:
        cov_mean = np.atleast_2d(np.cov(e, rowvar=False, ddof=1)) / b
    else:
        cov_mean = np.zeros_like(c)
    q_val = float(e_bar @ c @ e_bar) - float(np.trace(c @ cov_mean))
    var_q = 4.0 * float(e_bar @ c @ cov_mean @ c @ e_bar)
    q_se = float(np.sqrt(max(var_q, 0.0)))
    return ResidualSigma(
        sigma=float(np.sqrt(max(q_val, 0.0))),
        q=q_val,
        q_se=q_se,
        resolved=bool(q_val > 3.0 * q_se),
    )


# ----------------------------------------------------------------------------
# Scenario evaluation -> JSON-ready dict (used by scripts 01-04)
# ----------------------------------------------------------------------------

STUDY_PATHS = 2**14
STUDY_BATCHES = 32  # spec D17: >= 30 for study runs
STUDY_SEED = 20261117
SE_MULTIPLE = 3.0


def _floats(x) -> list:
    return [float(v) for v in np.atleast_1d(x)]


def evaluate_scenario(
    basket: CSIBasket,
    state: CSIState,
    option: CSIOption,
    label: str = "",
    n_paths: int = STUDY_PATHS,
    n_batches: int = STUDY_BATCHES,
    seed: int = STUDY_SEED,
) -> Dict[str, Any]:
    """Price one scenario both ways and score M1..M5 against REF.

    Returns a JSON-ready dict.  Cash figures are CNY per 1% move of a name on
    ``option.notional_cny`` of index notional.  A gap is ``resolved`` when
    |gap| > 3 paired SE (+ the D18 FD bias allowance for M3), spec D7/AC12.
    """
    q = state_quantities(basket, state)
    units = option.units(q.index_level)
    mc = mc_price_and_deltas(basket, state, option, n_paths=n_paths, n_batches=n_batches, seed=seed)
    mm = mm_price_and_deltas(basket, state, option)
    methods = delta_methods(basket, state, option, mc, mm)
    bias = fd_bias_allowance(basket, state, option, mc.fd_bump)
    cash_scale = q.spots * units * 0.01
    cov = basket.covariance_daily()

    method_out: Dict[str, Any] = {}
    for key in ("M1", "M2", "M3", "M4", "M5"):
        gap, se = delta_cash_error_per_pct(methods[key], q.spots, units)
        allow = bias * cash_scale if key == "M3" else np.zeros_like(gap)
        resolved = np.abs(gap) > SE_MULTIPLE * se + allow
        rs = residual_daily_pnl_sigma(cash_error_batches(methods[key], q.spots, units), cov)
        method_out[key] = {
            "label": METHOD_LABELS[key],
            "gap_cash_per_pct": _floats(gap),
            "gap_se": _floats(se),
            "gap_rel_to_ref": _floats(methods[key].diff_mean / mc.deltas_pathwise),
            "resolved": [bool(v) for v in resolved],
            "residual_sigma_cny": rs.sigma,
            "residual_q": rs.q,
            "residual_q_se": rs.q_se,
            "residual_resolved": rs.resolved,
        }
    return {
        "label": label,
        "names": list(basket.names),
        "rebalance": basket.rebalance,
        "decrement": basket.decrement,
        "vols": _floats(basket.vols),
        "div_yields": _floats(basket.div_yields),
        "target_weights": _floats(basket.target_weights),
        "current_weights": _floats(q.current_weights),
        "corr": [_floats(row) for row in basket.corr],
        "elapsed": float(state.elapsed),
        "moves": _floats(state.moves),
        "moneyness": option.moneyness,
        "maturity": option.maturity,
        "is_call": option.is_call,
        "index_level": q.index_level,
        "units": units,
        "strike": mc.strike,
        "n_paths": n_paths,
        "n_batches": n_batches,
        "price": {"mc": mc.price, "mc_se": mc.price_se, "mm": mm.price,
                  "mm_minus_mc": mm.price - mc.price},
        "delta_index": {"mc": mc.delta_index, "mc_se": mc.delta_index_se, "mm": mm.delta_index},
        "total_delta_cash_per_pct": {
            "mc": mc.delta_index * q.index_level * units * 0.01,
            "mc_se": mc.delta_index_se * q.index_level * units * 0.01,
            "mm": mm.delta_index * q.index_level * units * 0.01,
        },
        "sigma_eff_mm": mm.sigma_eff,
        "ref_cash_per_pct": _floats(mc.deltas_pathwise * cash_scale),
        "ref_se": _floats(mc.deltas_pathwise_se * cash_scale),
        "fd_bias_cash_per_pct": _floats(bias * cash_scale),
        "euler_max_rel_err": mc.euler_max_rel_err,
        "methods": method_out,
    }


def ref_corr_band(
    basket: CSIBasket,
    state: CSIState,
    option: CSIOption,
    corr_up: np.ndarray,
    corr_dn: np.ndarray,
    n_paths: int = STUDY_PATHS,
    n_batches: int = STUDY_BATCHES,
    seed: int = STUDY_SEED,
) -> list:
    """D13: |REF(rho+0.1) - REF(rho-0.1)| / 2, CNY per 1%, same seed."""
    q = state_quantities(basket, state)
    units = option.units(q.index_level)
    k = option.strike(q.index_level)
    up = mc_price_and_deltas(replace(basket, corr=corr_up), state, option, n_paths, n_batches, seed, strike=k)
    dn = mc_price_and_deltas(replace(basket, corr=corr_dn), state, option, n_paths, n_batches, seed, strike=k)
    return _floats(np.abs(up.deltas_pathwise - dn.deltas_pathwise) / 2.0 * q.spots * units * 0.01)
