"""Andreasen–Huge (2011) local volatility: an opt-in construction that reprices its input implied vols.

``build_dupire_local_vol`` evaluates Dupire pointwise at the quoted maturities and interpolates in
between, which does not preserve each segment's forward variance. This module instead calibrates,
expiry by expiry, a local vol theta_i(K) such that ONE fully implicit step of the forward Dupire
equation from the previous expiry's price slice reprices every quoted option exactly (Andreasen &
Huge, "Volatility interpolation", Risk 2011; QuantLib ``AndreasenHugeVolatilityInterpl``).

Variables, with deterministic carry and calibration forward F_t (log-linear between expiries, exact at
every expiry):

    y = ln(K / S0)                         quoted strikes are grid nodes
    c(t, y) = E[(S_t - K)^+] / F_t         (call model; the put model is analogous)
    dc/dt = 1/2 sigma^2 (c_yy - c_y) - mu(t) c_y

theta_i is piecewise linear in y through the quoted strikes (flat outside; Le Floc'h: piecewise
constant explodes on fine grids). The call and the put model are calibrated separately, and the local
vol is read on the out-of-the-money side (puts below the forward, calls above), where the Dupire
numerator does not cancel an intrinsic value.

The repricing local vol is Dupire on the AH price surface: within segment i, with tau = t - T_{i-1},
the partial step [I - tau A] c(t) = c(T_{i-1}) (+ boundary terms) defines c(t), and

    sigma^2(t, y) = 2 (c_t + mu_i c_y) / (c_yy - c_y)

with c_t from differentiating the partial step. It equals theta_i^2 only at tau -> 0+ and drifts inside
the segment; theta is the vol of the discrete one-step model, not the continuous local vol.

``AndreasenHugeLocalVol`` is the lazy reference object (local vol at any t); engines take the
``LocalVolSurface`` from ``to_local_vol_surface`` / ``build_andreasen_huge_local_vol``. The engine-facing
value is keyed through forward moneyness, sigma(t, S) = sigma_AH(t, S * F_t / F_env(t)), so an engine
drifting with the curves' own forward sees exactly the calibrated law of S_t / F_t.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import solve_banded
from scipy.optimize import least_squares
from scipy.special import ndtr

from quantark.util.exceptions import NumericalError, ValidationError
from quantark.volmodels.localvol.surface import LocalVolSurface

TAIL_GROWTH = 1.04          # geometric growth of the tail cells beyond the outer quoted strikes
TAIL_CELL_CAP = 0.05        # largest tail cell in y before the M-matrix cap
_SQRT_2PI = math.sqrt(2.0 * math.pi)
_SIDES = ("call", "put")


class LocalVolConstruction(Enum):
    """How a local-vol surface is built from a quoted implied-vol grid."""

    DUPIRE = "dupire"
    ANDREASEN_HUGE = "andreasen_huge"


@dataclass(frozen=True)
class AndreasenHugeConfig:
    """Numerical and calibration-set settings of the Andreasen–Huge construction.

    Attributes:
        cells_per_gap: equal grid cells between consecutive quoted strikes.
        rows_per_segment: sampled ``LocalVolSurface`` rows per expiry segment.
        domain_log_width: the grid extends this far in ln K beyond the outer quoted strikes.
        theta_bounds: admissible range of the calibrated one-step vols.
        exact_tol: largest admissible |model - market| implied-vol error per fitted quote.
        max_abs_d: opt-in quote rule; fit only quotes with |ln(K/F_T)| / (iv sqrt(T)) <= max_abs_d.
            ``None`` (default) fits every quote.
        density_floor: nodes whose AH density is below this fraction of its maximum take their local
            vol from the nearest readable nodes (declared convention; counted in ``unreadable``).
        jump_width: time offset of the sampled row holding the right limit at each expiry.
    """

    cells_per_gap: int = 8
    rows_per_segment: int = 64
    domain_log_width: float = math.log(8.0)
    theta_bounds: tuple = (0.05, 5.0)
    exact_tol: float = 1e-10
    max_abs_d: Optional[float] = None
    density_floor: float = 1e-10
    jump_width: float = 1e-6

    def __post_init__(self) -> None:
        if not isinstance(self.cells_per_gap, int) or self.cells_per_gap < 1:
            raise ValidationError("cells_per_gap must be a positive integer")
        if not isinstance(self.rows_per_segment, int) or self.rows_per_segment < 1:
            raise ValidationError("rows_per_segment must be a positive integer")
        lo, hi = self.theta_bounds
        if not (math.isfinite(lo) and math.isfinite(hi)) or lo <= 0 or hi <= lo:
            raise ValidationError("theta_bounds must satisfy 0 < lower < upper < inf")
        for name in ("domain_log_width", "exact_tol", "jump_width"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValidationError(f"{name} must be finite and positive")
        if not (0 < self.density_floor < 1):
            raise ValidationError("density_floor must lie in (0, 1)")
        if self.max_abs_d is not None and (not math.isfinite(self.max_abs_d) or self.max_abs_d <= 0):
            raise ValidationError("max_abs_d must be finite and positive when given")


@dataclass(frozen=True)
class AHCalibrationRecord:
    """Calibration outcome of one expiry on one side (call or put model)."""

    expiry: float
    side: str
    worst_vol_error: float
    quotes: int
    excluded: int
    nfev: int


# ---------------------------------------------------------------------------- grid and operators

def _build_grid(y_knots: np.ndarray, per_gap: int, width: float, h_max: float) -> tuple[np.ndarray, np.ndarray]:
    pieces = [np.linspace(a, b, per_gap + 1)[:-1] for a, b in zip(y_knots[:-1], y_knots[1:])]
    core = np.concatenate(pieces + [y_knots[-1:]])

    def tail(start: float, h: float, stop: float, sign: float) -> list:
        out, y = [], start
        while sign * (stop - y) > 0:
            h = min(h * TAIL_GROWTH, h_max)
            y += sign * h
            out.append(y)
        return out

    h_lo = core[1] - core[0] if core.size > 1 else h_max
    h_hi = core[-1] - core[-2] if core.size > 1 else h_max
    lo = tail(core[0], min(h_lo, h_max), core[0] - width, -1.0)[::-1]
    hi = tail(core[-1], min(h_hi, h_max), core[-1] + width, 1.0)
    y = np.concatenate((lo, core, hi))
    knot_idx = len(lo) + np.arange(y_knots.size) * per_gap
    return y, knot_idx


@dataclass
class _Stencils:
    """Three-point non-uniform D_y and G = D_yy - D_y on interior nodes (lower, diagonal, upper)."""

    dy: tuple
    g: tuple

    @classmethod
    def on(cls, y: np.ndarray) -> "_Stencils":
        hm = y[1:-1] - y[:-2]
        hp = y[2:] - y[1:-1]
        s = hm + hp
        dy = (-hp / (hm * s), (hp - hm) / (hm * hp), hm / (hp * s))
        dyy = (2.0 / (hm * s), -2.0 / (hm * hp), 2.0 / (hp * s))
        return cls(dy=dy, g=tuple(a - b for a, b in zip(dyy, dy)))

    @staticmethod
    def apply(coef: tuple, v: np.ndarray) -> np.ndarray:
        lo, di, up = coef
        return lo * v[:-2] + di * v[1:-1] + up * v[2:]


def _interp_matrix(y: np.ndarray, knots: np.ndarray) -> np.ndarray:
    w = np.zeros((y.size, knots.size))
    for j in range(knots.size):
        e = np.zeros(knots.size)
        e[j] = 1.0
        w[:, j] = np.interp(y, knots, e)      # linear in y, flat outside the knots
    return w


def _banded(gen: tuple, tau: float) -> np.ndarray:
    lo, di, up = gen
    ab = np.zeros((3, di.size))
    ab[0, 1:] = -tau * up[:-1]
    ab[1, :] = 1.0 - tau * di
    ab[2, :-1] = -tau * lo[1:]
    return ab


# ---------------------------------------------------------------------------- one side (call or put)

class _Side:
    """One instrument's AH model: calibrated theta per segment and the price slice at each expiry."""

    def __init__(self, side: str, owner: "AndreasenHugeLocalVol", iv_grid: np.ndarray):
        self.side = side
        self.o = owner
        self.thetas: list = []          # node vectors, one per segment
        self.knots: list = []           # (active strikes, theta at them) per segment
        self.starts: list = []          # price slice at each segment's start
        self.records: list = []
        self._calibrate(iv_grid)

    # market and boundaries
    def market(self, t: float, iv: np.ndarray) -> tuple:
        m = self.o.strikes / self.o.forward(t)
        sd = iv * math.sqrt(t)
        d1 = (-np.log(m) + 0.5 * sd * sd) / sd
        price = ndtr(d1) - m * ndtr(d1 - sd) if self.side == "call" else m * ndtr(sd - d1) - ndtr(-d1)
        vega = math.sqrt(t) * np.exp(-0.5 * d1 * d1) / _SQRT_2PI
        return price, vega

    def payoff(self) -> np.ndarray:
        k = self.o._k_over_s0
        return np.maximum(1.0 - k, 0.0) if self.side == "call" else np.maximum(k - 1.0, 0.0)

    def boundary(self, t: float) -> tuple:
        f = self.o.forward(t) / self.o.spot
        k = self.o._k_over_s0
        if self.side == "call":
            return max(1.0 - k[0] / f, 0.0), 0.0
        return 0.0, max(k[-1] / f - 1.0, 0.0)

    def boundary_rate(self, t: float, mu: float) -> tuple:
        f = self.o.forward(t) / self.o.spot
        k = self.o._k_over_s0
        if self.side == "call":
            return mu * k[0] / f, 0.0
        return 0.0, -mu * k[-1] / f

    # operators
    def generator(self, theta_nodes: np.ndarray, mu: float) -> tuple:
        st = self.o._stencils
        th2 = theta_nodes[1:-1] ** 2
        gen = tuple(0.5 * th2 * g - mu * d for g, d in zip(st.g, st.dy))
        if np.any(gen[0] <= 0) or np.any(gen[2] <= 0):
            raise NumericalError("Andreasen-Huge generator is not an M-matrix (cell check bypassed)")
        return gen

    def step(self, theta_nodes: np.ndarray, mu: float, c_start: np.ndarray, t: float, tau: float):
        gen = self.generator(theta_nodes, mu)
        b_lo, b_hi = self.boundary(t)
        rhs = c_start[1:-1].copy()
        rhs[0] += tau * gen[0][0] * b_lo
        rhs[-1] += tau * gen[2][-1] * b_hi
        ab = _banded(gen, tau)
        c = np.concatenate(([b_lo], solve_banded((1, 1), ab, rhs), [b_hi]))
        return c, gen, ab

    # calibration
    def _calibrate(self, iv_grid: np.ndarray) -> None:
        o = self.o
        cfg = o.config
        c = self.payoff()
        t_prev = 0.0
        prev_nodes = None
        for i, t in enumerate(o.expiries):
            t = float(t)
            tau = t - t_prev
            mu = o._mu[i]
            target, vega = self.market(t, iv_grid[i])
            active = np.ones(o.strikes.size, dtype=bool)
            if cfg.max_abs_d is not None:
                active = np.abs(np.log(o.strikes / o.forward(t))) / (iv_grid[i] * math.sqrt(t)) <= cfg.max_abs_d
                if int(active.sum()) < 2:
                    raise NumericalError(f"expiry {t:.6f}: fewer than 2 quotes within |d| <= {cfg.max_abs_d}")
            y_act = o._y_knots[active]
            idx = o._knot_idx[active]
            weights = _interp_matrix(o._y, y_act)
            tgt, vg = target[active], vega[active]
            guess = iv_grid[i][active] if prev_nodes is None else np.interp(y_act, o._y, prev_nodes)
            lo, hi = cfg.theta_bounds
            start = c

            def resid(p, start=start, weights=weights, tau=tau, mu=mu, t=t, idx=idx, tgt=tgt, vg=vg):
                return (self.step(weights @ p, mu, start, t, tau)[0][idx] - tgt) / vg

            def jac(p, start=start, weights=weights, tau=tau, mu=mu, t=t, idx=idx, vg=vg):
                th = weights @ p
                new, _, ab = self.step(th, mu, start, t, tau)
                gc = _Stencils.apply(o._stencils.g, new)
                rhs = tau * (th[1:-1] * gc)[:, None] * weights[1:-1]
                full = np.zeros((o._y.size, p.size))
                full[1:-1] = solve_banded((1, 1), ab, rhs)
                return full[idx] / vg[:, None]

            sol = least_squares(resid, np.clip(guess, lo, hi), jac=jac, bounds=(lo, hi), method="trf",
                                xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=200)
            err = np.abs(sol.fun)
            worst = float(err.max())
            self.records.append(AHCalibrationRecord(expiry=t, side=self.side, worst_vol_error=worst,
                                                    quotes=int(active.sum()), excluded=int((~active).sum()),
                                                    nfev=int(sol.nfev)))
            if worst > cfg.exact_tol:
                j = int(np.argmax(err))
                raise NumericalError(
                    f"Andreasen-Huge {self.side} calibration not exact at expiry {t:.6f}: strike "
                    f"{o.strikes[active][j]:.6g} implied-vol error {worst:.3e} > {cfg.exact_tol:.1e}; "
                    "check the quotes (calendar/butterfly arbitrage) or opt into the max_abs_d quote rule")
            nodes = weights @ sol.x
            self.thetas.append(nodes)
            self.knots.append((o.strikes[active].copy(), sol.x.copy()))
            self.starts.append(c)
            c = self.step(nodes, mu, c, t, tau)[0]
            prev_nodes, t_prev = nodes, t

    # repricing local vol on the grid (calibration coordinates)
    def sigma_nodes(self, i: int, t: float, tau: float) -> np.ndarray:
        o = self.o
        th = self.thetas[i]
        if tau <= 0.0:
            return th.copy()
        mu = o._mu[i]
        c, gen, ab = self.step(th, mu, self.starts[i], t, tau)
        lc = _Stencils.apply(gen, c)
        b_lo, b_hi = self.boundary(t)
        db_lo, db_hi = self.boundary_rate(t, mu)
        rhs = lc.copy()
        rhs[0] += gen[0][0] * b_lo + tau * gen[0][0] * db_lo
        rhs[-1] += gen[2][-1] * b_hi + tau * gen[2][-1] * db_hi
        ct = solve_banded((1, 1), ab, rhs)
        gc = _Stencils.apply(o._stencils.g, c)
        cy = _Stencils.apply(o._stencils.dy, c)
        num = 2.0 * (ct + mu * cy)
        ok = (gc > o.config.density_floor * np.max(gc)) & (num > 0.0)
        valid = np.flatnonzero(ok)
        if valid.size == 0:
            raise NumericalError(f"Andreasen-Huge {self.side} local vol unreadable at t={t:.6f}")
        inner = np.sqrt(num[valid] / gc[valid])
        filled = np.interp(np.arange(gc.size), valid, inner)     # linear across holes, flat beyond
        o.unreadable[self.side] = max(o.unreadable[self.side], int(gc.size - valid.size))
        return np.concatenate((filled[:1], filled, filled[-1:]))


# ---------------------------------------------------------------------------- public objects

class AndreasenHugeLocalVol:
    """Lazy Andreasen–Huge repricing local vol (reference object; engines take ``to_local_vol_surface``).

    Attributes:
        spot: spot S0 the grid is anchored on.
        expiries: quoted maturities (years).
        strikes: quoted strikes.
        config: the ``AndreasenHugeConfig`` used.
        report: one ``AHCalibrationRecord`` per expiry and side.
        unreadable: largest number of nodes per side whose local vol came from the fill rule.
    """

    def __init__(self, iv_surface, spot: float, rate_curve, div_yield: Callable[[float], float],
                 config: Optional[AndreasenHugeConfig] = None):
        self.config = config or AndreasenHugeConfig()
        strikes = np.asarray(iv_surface.strikes, dtype=float)
        expiries = np.asarray(iv_surface.maturities, dtype=float)
        iv = np.asarray(iv_surface.iv_grid, dtype=float)
        _validate_inputs(strikes, expiries, iv, spot)
        self.spot = float(spot)
        self.strikes, self.expiries = strikes, expiries
        self._rate_curve, self._div_yield = rate_curve, div_yield
        fwd = np.array([self._env_forward(float(t)) for t in expiries])
        if not np.all(np.isfinite(fwd)) or np.any(fwd <= 0):
            raise ValidationError("rate_curve / div_yield give a non-finite forward")
        self._ln_fwd = np.log(fwd)
        times = np.concatenate(([0.0], expiries))
        self._mu = np.diff(np.concatenate(([math.log(self.spot)], self._ln_fwd))) / np.diff(times)
        self._y_knots = np.log(strikes / self.spot)
        theta_min = self.config.theta_bounds[0]
        mu_max = float(np.max(np.abs(self._mu)))
        h_cap = math.inf if mu_max == 0.0 else theta_min ** 2 / (2.0 * mu_max)
        gaps = np.diff(self._y_knots) / self.config.cells_per_gap
        if np.any(gaps > h_cap):
            j = int(np.argmax(gaps > h_cap))
            raise ValidationError(
                f"strike gap {strikes[j]:.6g}-{strikes[j + 1]:.6g}: cells of {gaps[j]:.3g} in ln K exceed "
                f"{h_cap:.3g} = theta_min^2 / (2 max|mu|); raise cells_per_gap or the lower theta bound")
        self._y, self._knot_idx = _build_grid(self._y_knots, self.config.cells_per_gap,
                                              self.config.domain_log_width, min(TAIL_CELL_CAP, h_cap))
        self._k_over_s0 = np.exp(self._y)
        self._stencils = _Stencils.on(self._y)
        self.unreadable = {"call": 0, "put": 0}
        self._cache: dict = {}
        self._sides = {s: _Side(s, self, iv) for s in _SIDES}
        self.report = tuple(r for i in range(expiries.size) for s in _SIDES for r in [self._sides[s].records[i]])

    # forwards
    def _env_forward(self, t: float) -> float:
        if t <= 0.0:
            return self.spot
        return self.spot * math.exp((float(self._rate_curve.get_rate(t)) - float(self._div_yield(t))) * t)

    def forward(self, t: float) -> float:
        """Calibration forward F_t: the curve forward at every expiry, log-linear in between."""
        t = float(t)
        if t <= 0.0:
            return self.spot
        i, t_start, _ = self._segment(t)
        ln_start = math.log(self.spot) if i == 0 else float(self._ln_fwd[i - 1])
        return math.exp(ln_start + self._mu[i] * (t - t_start))

    def _segment(self, t: float) -> tuple:
        last = t > self.expiries[-1]
        i = min(int(np.searchsorted(self.expiries, t, side="left")), self.expiries.size - 1)
        t_start = 0.0 if i == 0 else float(self.expiries[i - 1])
        if last:
            return i, t_start, float(self.expiries[-1]) - t_start
        return i, t_start, max(t - t_start, 0.0)

    # calibration-coordinate local vol
    def _sigma_nodes(self, t: float) -> np.ndarray:
        t = min(max(float(t), 0.0), float(self.expiries[-1]))
        hit = self._cache.get(t)
        if hit is None:
            i, _, tau = self._segment(t)
            below = self._y < math.log(self.forward(t) / self.spot)
            hit = np.where(below, self._sides["put"].sigma_nodes(i, t, tau),
                           self._sides["call"].sigma_nodes(i, t, tau))
            if len(self._cache) > 50000:
                self._cache.clear()
            self._cache[t] = hit
        return hit

    def _calibration_local_vol(self, spot: ArrayLike, t: float) -> np.ndarray:
        """Local vol of S_t / F_t at the calibration forward (no forward-moneyness map)."""
        s = np.asarray(spot, dtype=float)
        return np.interp(np.log(s / self.spot), self._y, self._sigma_nodes(t))

    def local_vol(self, spot: ArrayLike, t: ArrayLike) -> "float | np.ndarray":
        """Engine-facing local vol sigma(t, S) = sigma_AH(t, S * F_t / F_env(t)) (vectorised in spot)."""
        s = np.asarray(spot, dtype=float)
        tt = np.asarray(t, dtype=float)
        if not (np.all(np.isfinite(s)) and np.all(np.isfinite(tt))) or np.any(s <= 0):
            raise ValidationError("spot must be finite and positive and t finite")
        if tt.ndim == 0:
            out = self._mapped(s, float(tt))
            return out if out.shape else float(out)
        s_b, t_b = np.broadcast_arrays(s, tt)
        out = np.empty(s_b.shape, dtype=float)
        for tv in np.unique(t_b):
            mask = t_b == tv
            out[mask] = self._mapped(s_b[mask], float(tv))
        return out

    def _mapped(self, s: np.ndarray, t: float) -> np.ndarray:
        tc = min(max(t, 0.0), float(self.expiries[-1]))
        ratio = self.forward(tc) / self._env_forward(tc)
        return self._calibration_local_vol(s * ratio, tc)

    def theta_knots(self, segment: int, side: str) -> tuple:
        """(fitted strikes, theta at them) of one segment and side (diagnostics)."""
        if side not in _SIDES:
            raise ValidationError("side must be 'call' or 'put'")
        strikes, theta = self._sides[side].knots[segment]
        return strikes.copy(), theta.copy()

    def discrete_repricing_errors(self) -> np.ndarray:
        """Worst |implied-vol error| of the one-step model at each expiry: shape (n_expiries, 2 sides)."""
        out = np.zeros((self.expiries.size, 2))
        for k, s in enumerate(_SIDES):
            for i, rec in enumerate(self._sides[s].records):
                out[i, k] = rec.worst_vol_error
        return out

    def to_local_vol_surface(self, rows_per_segment: Optional[int] = None) -> LocalVolSurface:
        """Sample the engine-facing local vol onto a ``LocalVolSurface`` (rows at 0, each expiry's right
        limit, and ``rows_per_segment`` uniform rows per segment; strikes = the AH grid nodes)."""
        rows = self.config.rows_per_segment if rows_per_segment is None else rows_per_segment
        if not isinstance(rows, int) or rows < 1:
            raise ValidationError("rows_per_segment must be a positive integer")
        times = [0.0]
        t_prev = 0.0
        for t in self.expiries:
            t = float(t)
            if t_prev > 0.0:
                times.append(t_prev + self.config.jump_width)
            times.extend(float(x) for x in np.linspace(t_prev, t, rows + 1)[1:])
            t_prev = t
        times = np.array(sorted(set(times)))
        strikes = self.spot * self._k_over_s0
        lv = np.vstack([self.local_vol(strikes, float(t)) for t in times])
        return LocalVolSurface(strike_grid=strikes, time_grid=times, lv_grid=lv)


def _validate_inputs(strikes: np.ndarray, expiries: np.ndarray, iv: np.ndarray, spot: float) -> None:
    if strikes.ndim != 1 or strikes.size < 2:
        raise ValidationError("Andreasen-Huge needs at least 2 strikes")
    if expiries.ndim != 1 or expiries.size < 1:
        raise ValidationError("Andreasen-Huge needs at least 1 expiry")
    if iv.shape != (expiries.size, strikes.size):
        raise ValidationError(f"iv_grid shape {iv.shape} must be (n_expiries, n_strikes)")
    if not np.all(np.isfinite(strikes)) or np.any(strikes <= 0) or np.any(np.diff(strikes) <= 0):
        raise ValidationError("strikes must be finite, positive and strictly increasing")
    if not np.all(np.isfinite(expiries)) or np.any(expiries <= 0) or np.any(np.diff(expiries) <= 0):
        raise ValidationError("expiries must be finite, positive and strictly increasing")
    if not np.all(np.isfinite(iv)) or np.any(iv <= 0):
        raise ValidationError("implied vols must be finite and positive")
    if not math.isfinite(spot) or spot <= 0:
        raise ValidationError(f"spot must be finite and positive, got {spot}")


def calibrate_andreasen_huge(iv_surface, spot: float, rate_curve, div_yield: Callable[[float], float],
                             config: Optional[AndreasenHugeConfig] = None) -> AndreasenHugeLocalVol:
    """Calibrate the Andreasen–Huge model to a quoted implied-vol grid (lazy reference object).

    Args:
        iv_surface: grid with ``strikes``, ``maturities`` and ``iv_grid`` (n_maturities, n_strikes).
        spot: current spot.
        rate_curve: ``RateCurve``; zero rates enter only through the forward.
        div_yield: callable T -> continuously-compounded dividend (or foreign) zero yield.
        config: ``AndreasenHugeConfig`` (defaults if None).

    Raises:
        ValidationError: invalid inputs, or a strike gap too coarse for the carry (cell check).
        NumericalError: a fitted quote is not repriced within ``config.exact_tol``.
    """
    return AndreasenHugeLocalVol(iv_surface, spot, rate_curve, div_yield, config)


def build_andreasen_huge_local_vol(iv_surface, spot: float, rate_curve, div_yield: Callable[[float], float],
                                   config: Optional[AndreasenHugeConfig] = None) -> LocalVolSurface:
    """Andreasen–Huge local vol sampled onto a ``LocalVolSurface`` (same arguments as
    ``build_dupire_local_vol``; opt-in alternative that reprices its inputs)."""
    return calibrate_andreasen_huge(iv_surface, spot, rate_curve, div_yield, config).to_local_vol_surface()
