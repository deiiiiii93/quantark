"""Independent Gaussian-transition solver for a discretely monitored snowball on the intraday clock.

NOT a production engine, and it imports nothing from the QUAD / PDE / MC engine
packages: it reads only the resolved intraday context (numerical twin, remaining
events, clock-wrapped surface, curves) and the product's own payoff functions.
It is the deterministic reference of the daily-KI intraday study (study revision
2026-09-18): the RQMC arm cannot resolve that study's budgets, and qualifies
this solver case by case instead.

Representation. Each state's value at an event instant is a right-continuous
piecewise-linear function of y = ln S with explicit jumps on knots:

    F(y) = f_0 + sum_k dF_k * clamp((y - x_k) / h_k, 0, 1) + sum_k J_k * H(y - x_k),

dF_k = F(x_{k+1}-) - F(x_k), and flat extension beyond the end knots. Barrier
levels are knots, so a knock-out or knock-in replacement is an EXACT jump. Its
expectation under Y ~ N(mu, v) is closed form, with R(a) = E[(Y - a)+]:

    E[clamp((Y - x_k)/h_k, 0, 1)] = (R(x_k) - R(x_{k+1})) / h_k,   E[H(Y - x_k)] = Phi(d_k),

and the first and second mu-derivatives follow from dR/dmu = Phi(d),
dPhi(d)/dmu = phi(d)/sqrt(v), dphi(d)/dmu = -d phi(d)/sqrt(v). The only error is
the piecewise-linear representation of the SMOOTH continuation between jumps,
O(h^2); :func:`ladder_estimate` turns a nested refinement ladder into a value
and a declared radius.

At the first remaining event the continuation is re-evaluated exactly on a local
grid scaled to the remaining standard deviation around ln S, so the final
(possibly one-second) propagation never sees the global grid's kinks; that last
step is a single exact pointwise expectation with exact derivatives. A zero-
variance interval is an exact shift.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, inf, isfinite, log, log2, sqrt
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.special import ndtr

from quantark.intraday.events import EventKind
from quantark.intraday.timestamp import calendar_year_fraction

_SAT = 9.0             # standard deviations beyond which a ramp / Heaviside is saturated (Phi(-9) ~ 1e-19)
_LOCAL_STD = 15.0      # half-width of the local grid at the first event, in remaining standard deviations
_CHUNK = 256
MIN_POINTS = 101       # the coarsest admissible level: a 26-knot local grid
_INV_SQRT_2PI = 1.0 / sqrt(2.0 * np.pi)

@dataclass(frozen=True)
class LadderPolicy:
    """How a nested refinement ladder becomes a value and an uncertainty radius.

    One object executes (:func:`ladder_estimate`) and serializes (:meth:`describe`, which goes into the
    certificate's contract), so the recorded policy is the policy that ran and a changed parameter is a
    changed contract. Bump ``version`` with any change of logic that the parameters do not express.

    A ladder radius is a CALIBRATED NUMERICAL ESTIMATE, not an analytical bound: finitely many
    contracting extrapolants do not prove the next ones contract, and a spacing ladder cannot see an
    error its levels share (the builder accounts for those separately). It is accepted as the gate's
    allowance only with its calibration evidence.
    """

    version: int = 2
    discretization_order: int = 2        # O(h^2): Richardson weight (2^p v_n - v_{n-1}) / (2^p - 1)
    refinement_ratio: int = 2            # every level halves every spacing in play
    min_levels: int = 5                  # four for the estimate, one coarser to calibrate the rule one level down
    order_window: Tuple[float, float] = (1.5, 2.5)
    contraction: float = 0.5             # successive extrapolant spreads must shrink at least this much
    max_contraction: float = 16.0        # O(h^4) at best: a smaller spread is luck and is not credited
    unconfirmed_safety: float = 3.0      # Roache's grid-convergence-index factor for an unconfirmed order
    stagnation_floor: float = 1e-13      # relative: differences below it are floating point
    roundoff_factor: float = 16.0        # multiples of eps x value span x instants (an estimate, builder-side)

    def describe(self) -> dict:
        lo, hi = self.order_window
        weight = self.refinement_ratio ** self.discretization_order
        return {
            "version": self.version,
            "radius_kind": "calibrated_numerical_estimate",
            "parameters": {
                "discretization_order": self.discretization_order, "refinement_ratio": self.refinement_ratio,
                "min_levels": self.min_levels, "order_window": [lo, hi], "contraction": self.contraction,
                "max_contraction": self.max_contraction, "unconfirmed_safety": self.unconfirmed_safety,
                "stagnation_floor": self.stagnation_floor, "roundoff_factor": self.roundoff_factor,
            },
            "branches": {
                "exact": ("the last two differences are below the stagnation floor AND the solve names an exactness "
                          "basis; value: finest level; radius: the larger difference"),
                "stagnant": ("the last two differences are below the stagnation floor and no exactness basis is named: "
                             "equal values do not establish exactness; radius: infinite"),
                "geometric": (f"the last two differences share a sign with observed order in [{lo}, {hi}] and successive "
                              f"extrapolants contract to {self.contraction} or less; value: Richardson extrapolant "
                              f"({weight} v_n - v_(n-1)) / {weight - 1}; radius: the last extrapolant spread, never below "
                              f"1/{self.max_contraction} of the previous spread"),
                "correction": ("order confirmed as for geometric, extrapolants do not contract; value: Richardson "
                               "extrapolant; radius: the whole correction |E_n - v_n|"),
                "unextrapolated": (f"order not confirmed (outside the window, or alternating differences) and the last "
                                   f"difference did not grow; value: finest level; radius: {self.unconfirmed_safety} x the "
                                   "previous difference"),
                "unbounded": "the last difference grew; value: finest level; radius: infinite",
                "uncalibrated": ("the same rule applied one level down (without the finest level) did not cover this "
                                 "value with its own radius; radius: infinite"),
            },
            "calibration": ("in-ladder: |value - value one level down| <= radius one level down, recorded per quantity; "
                            "an infinite radius one level down is no evidence and does not calibrate"),
        }


LADDER_POLICY = LadderPolicy()


@dataclass(frozen=True)
class LadderEstimate:
    """A value and its uncertainty radius from one nested refinement ladder (see :func:`ladder_estimate`)."""

    value: float
    radius: float
    rule: str                                  # a key of LadderPolicy.describe()["branches"]
    observed_order: Optional[float]
    extrapolants: Tuple[float, ...]
    basis: Optional[str] = None                # the named exactness basis, for rule "exact" only
    calibration: Optional[dict] = None         # the rule one level down against this value; None for "exact"


def _rule(v: Sequence[float], policy: LadderPolicy, exact_basis: Optional[str]) -> LadderEstimate:
    """The policy on the four finest levels of ``v``, without calibration."""
    floor = policy.stagnation_floor * max(1.0, abs(v[-1]))
    d_prev, d_last = v[-2] - v[-3], v[-1] - v[-2]
    weight = float(policy.refinement_ratio ** policy.discretization_order)
    extrapolants = tuple((weight * v[k] - v[k - 1]) / (weight - 1.0) for k in range(len(v) - 3, len(v)))
    if abs(d_last) <= floor and abs(d_prev) <= floor:
        if exact_basis is None:
            return LadderEstimate(v[-1], inf, "stagnant", None, extrapolants)
        return LadderEstimate(v[-1], max(abs(d_last), abs(d_prev)), "exact", None, extrapolants, basis=exact_basis)
    if exact_basis is not None:
        raise ValueError(f"exactness basis {exact_basis!r} claimed for a ladder that does not stagnate: {list(v)}")
    order = log2(abs(d_prev) / abs(d_last)) if abs(d_last) > 0.0 and abs(d_prev) > 0.0 else inf
    if d_prev * d_last > 0.0 and policy.order_window[0] <= order <= policy.order_window[1]:
        s1, s2 = abs(extrapolants[1] - extrapolants[0]), abs(extrapolants[2] - extrapolants[1])
        if s2 <= policy.contraction * s1:
            return LadderEstimate(extrapolants[2], max(s2, s1 / policy.max_contraction), "geometric", order, extrapolants)
        return LadderEstimate(extrapolants[2], abs(extrapolants[2] - v[-1]), "correction", order, extrapolants)
    if abs(d_last) <= abs(d_prev):
        return LadderEstimate(v[-1], policy.unconfirmed_safety * abs(d_prev), "unextrapolated", order, extrapolants)
    return LadderEstimate(v[-1], inf, "unbounded", order, extrapolants)


def ladder_estimate(values: Sequence[float], policy: LadderPolicy = LADDER_POLICY,
                    exact_basis: Optional[str] = None) -> LadderEstimate:
    """Turn a nested ladder of an O(h^2) scheme into a value and a calibrated uncertainty radius.

    Every branch is declared in ``policy.describe()``. The estimate comes from the four finest levels.
    It is then CALIBRATED: the same rule on the ladder without its finest level must have covered this
    value with its own radius, or the radius is infinite (``uncalibrated``) -- a rule that was too
    optimistic one level down is not trusted at this one. ``exact_basis`` names why a stagnating ladder
    is exact (a terminated claim, a claim decided at the valuation instant, an analytical single
    integral); without one, stagnation is unresolved, because equal values do not establish exactness.
    """
    v = [float(x) for x in values]
    if len(v) < policy.min_levels:
        raise ValueError(f"a ladder needs at least {policy.min_levels} levels, got {len(v)}")
    fine = _rule(v, policy, exact_basis)
    if fine.rule == "exact" or not isfinite(fine.radius):
        return fine
    coarse = _rule(v[:-1], policy, None)
    move = abs(fine.value - coarse.value)
    calibration = {"coarse_value": coarse.value, "coarse_radius": coarse.radius, "coarse_rule": coarse.rule, "move": move,
                   "covered": isfinite(coarse.radius) and move <= coarse.radius}
    if not calibration["covered"]:
        return LadderEstimate(fine.value, inf, "uncalibrated", fine.observed_order, fine.extrapolants, calibration=calibration)
    return LadderEstimate(fine.value, fine.radius, fine.rule, fine.observed_order, fine.extrapolants, calibration=calibration)


# ---------------------------------------------------------------------------
# piecewise-linear-with-jumps functions
@dataclass
class PLJ:
    """Right-continuous piecewise-linear function with jumps on its knots (``j[0] == 0``)."""

    x: np.ndarray
    f: np.ndarray
    j: np.ndarray

    @staticmethod
    def sample(x, values) -> "PLJ":
        x = np.asarray(x, dtype=float)
        return PLJ(x, np.asarray(values, dtype=float), np.zeros_like(x))

    def left(self) -> np.ndarray:
        return self.f - self.j

    def __call__(self, y) -> np.ndarray:
        y = np.atleast_1d(np.asarray(y, dtype=float))
        x, f, left = self.x, self.f, self.left()
        k = np.searchsorted(x, y, side="right") - 1
        out = np.empty_like(y)
        below, above = k < 0, k >= len(x) - 1
        inside = ~(below | above)
        out[below] = f[0]
        out[above] = f[-1]
        ki = k[inside]
        frac = (y[inside] - x[ki]) / (x[ki + 1] - x[ki])
        out[inside] = f[ki] + (left[ki + 1] - f[ki]) * frac
        return out


def with_knots(fn: PLJ, knots) -> PLJ:
    """The same function on a knot vector that also contains ``knots`` (inserted points are continuous)."""
    new = np.union1d(fn.x, np.asarray(knots, dtype=float))
    if len(new) == len(fn.x):
        return fn
    f = fn(new)
    j = np.zeros_like(new)
    idx = np.searchsorted(new, fn.x)
    j[idx] = fn.j
    return PLJ(new, f, j)


def splice(below: PLJ, above: PLJ, b: int) -> PLJ:
    """``below`` for y < x_b, ``above`` for y >= x_b; both on the same knots."""
    f = np.concatenate([below.f[:b], above.f[b:]])
    j = np.concatenate([below.j[:b], above.j[b:]])
    j[b] = above.f[b] - below.left()[b] if b > 0 else 0.0
    return PLJ(below.x, f, j)


def constant_like(fn: PLJ, value: float) -> PLJ:
    return PLJ(fn.x, np.full_like(fn.x, float(value)), np.zeros_like(fn.x))


def expect(fn: PLJ, mu, v: float, disc: float, *, derivatives: bool = False):
    """disc * E[F(mu + sqrt(v) Z)] for each mu, optionally with its first two mu-derivatives."""
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    if v == 0.0:
        return disc * fn(mu), None, None
    s = sqrt(v)
    x, f, jumps = fn.x, fn.f, fn.j
    left = fn.left()
    n = len(x)
    value = np.empty_like(mu)
    d1 = np.empty_like(mu) if derivatives else None
    d2 = np.empty_like(mu) if derivatives else None
    order = np.argsort(mu)
    for start in range(0, len(mu), _CHUNK):
        sel = order[start:start + _CHUNK]
        m = mu[sel]
        a = max(int(np.searchsorted(x, m.min() - _SAT * s, side="right")) - 1, 0)
        b = min(int(np.searchsorted(x, m.max() + _SAT * s, side="left")) + 1, n - 1)
        xs = x[a:b + 1]
        dF = left[a + 1:b + 1] - f[a:b]
        h = np.diff(xs)
        dist = m[:, None] - xs[None, :]
        z = dist / s
        cdf = ndtr(z)
        pdf = np.exp(-0.5 * z * z) * _INV_SQRT_2PI
        r = dist * cdf + s * pdf
        ramp = (r[:, :-1] - r[:, 1:]) / h
        ramp = np.where(z[:, 1:] > _SAT, 1.0, np.where(z[:, :-1] < -_SAT, 0.0, ramp))
        base = f[a]                           # every knot <= a is saturated: telescopes to F(x_a)
        value[sel] = base + ramp @ dF + cdf[:, 1:] @ jumps[a + 1:b + 1]
        if derivatives:
            d1[sel] = ((cdf[:, :-1] - cdf[:, 1:]) / h) @ dF + (pdf[:, 1:] / s) @ jumps[a + 1:b + 1]
            d2[sel] = ((pdf[:, :-1] - pdf[:, 1:]) / (h * s)) @ dF + ((-z[:, 1:] * pdf[:, 1:]) / v) @ jumps[a + 1:b + 1]
    if derivatives:
        return disc * value, disc * d1, disc * d2
    return disc * value, None, None


# ---------------------------------------------------------------------------
# market moments on the context's clock
def moments(ctx, t0: float, t1: float, strike: float) -> Tuple[float, float, float]:
    """(log-mean m, variance v, discount factor) of ln S from tau t0 to t1."""
    env, spot = ctx.pricing_env, float(ctx.pricing_env.spot)
    w1 = float(env.vol_surface.total_variance(strike, t1, spot)) if t1 > 0.0 else 0.0
    w0 = float(env.vol_surface.total_variance(strike, t0, spot)) if t0 > 0.0 else 0.0
    df1 = float(env.get_discount_factor(t1)) if t1 > 0.0 else 1.0
    df0 = float(env.get_discount_factor(t0)) if t0 > 0.0 else 1.0
    q1 = float(env.get_div_yield(t1)) * t1 if t1 > 0.0 else 0.0
    q0 = float(env.get_div_yield(t0)) * t0 if t0 > 0.0 else 0.0
    v = w1 - w0
    if v < 0.0:
        raise ValueError("negative forward variance in the reference")
    carry = -log(df1 / df0) - (q1 - q0)
    return carry - 0.5 * v, v, df1 / df0


# ---------------------------------------------------------------------------
# snowball
def _ko_hits_above(prod) -> bool:
    return not bool(prod.is_reverse)          # standard: KO when S >= B; reverse: S <= B


def _apply_instant(ctx, prod, events, v0: PLJ, v1: PLJ, t: float) -> Tuple[PLJ, PLJ]:
    """Backward application of the events of one instant: reverse priority (terminal, coupon, KO, KI)."""
    disable = bool(prod.barrier_config.disable_ko_after_ki)
    for e in [e for e in events if e.kind is EventKind.KO]:
        pay = calendar_year_fraction(ctx.valuation_timestamp, e.payment_timestamp)
        delay = float(ctx.pricing_env.get_discount_factor(pay)) / (float(ctx.pricing_env.get_discount_factor(t)) if t > 0.0 else 1.0)
        cash = float(e.cash) * delay
        lb = log(float(e.barrier))
        v0, v1 = with_knots(v0, [lb]), with_knots(v1, [lb])
        b = int(np.searchsorted(v0.x, lb))
        if _ko_hits_above(prod):
            v0 = splice(v0, constant_like(v0, cash), b)
            v1 = v1 if disable else splice(v1, constant_like(v1, cash), b)
        else:
            v0 = splice(constant_like(v0, cash), v0, b)
            v1 = v1 if disable else splice(constant_like(v1, cash), v1, b)
    for e in [e for e in events if e.kind is EventKind.KI]:
        lb = log(float(e.barrier))
        v0, v1 = with_knots(v0, [lb]), with_knots(v1, [lb])
        b = int(np.searchsorted(v0.x, lb))
        if prod.is_reverse:                   # KI when S >= B
            v0 = splice(v0, v1, b)
        else:                                 # KI when S <= B
            v0 = splice(v1, v0, b)
    return v0, v1


def _terminal(ctx, prod, x: np.ndarray, t_mat: float) -> Tuple[PLJ, PLJ]:
    term = ctx.timeline.terminal()
    pay = calendar_year_fraction(ctx.valuation_timestamp, term.payment_timestamp)
    delay = float(ctx.pricing_env.get_discount_factor(pay)) / float(ctx.pricing_env.get_discount_factor(t_mat)) if t_mat > 0.0 else float(
        ctx.pricing_env.get_discount_factor(pay)) if pay > 0.0 else 1.0
    s = np.exp(x)
    v0 = np.array([prod.get_maturity_payoff_v0(float(si), ctx.pricing_env) for si in s]) * delay
    v1 = np.array([prod.get_maturity_payoff_v1(float(si), ctx.pricing_env) for si in s]) * delay
    return PLJ.sample(x, v0), PLJ.sample(x, v1)


def _instants(ctx):
    groups: Dict[float, list] = {}
    for e in ctx.numerical.remaining_events:
        groups.setdefault(ctx.numerical.event_taus[e.event_id], []).append(e)
    return sorted(groups.items())


def _barrier_logs(ctx, prod):
    levels = {log(float(e.barrier)) for e in ctx.numerical.remaining_events if e.barrier is not None}
    levels.add(log(float(prod.strike)))
    return levels


_SWEEP_CACHE: Dict[tuple, Tuple[PLJ, PLJ]] = {}


def _time_homogeneous(env) -> bool:
    """Whether the market means the same absolute schedule from any valuation instant."""
    from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, NoDividend
    return (type(env.rate_curve) is FlatRateCurve and type(env.vol_surface) is FlatVolSurface
            and (env.div_yield is None or type(env.div_yield) in (NoDividend, ContinuousDividendYield)))


def _unclocked(env):
    """The market inside the intraday clock: the clock's own identity (profile, calendar) is keyed separately, and its
    repr carries the valuation anchor, which would key an anchor-free sweep on every horizon."""
    from types import SimpleNamespace
    from quantark.param.vol import TradingClockVolSurface
    surface = env.vol_surface
    return SimpleNamespace(rate_curve=env.rate_curve, div_yield=env.div_yield,
                           vol_surface=surface.inner if type(surface) is TradingClockVolSurface else surface)


def _domain(ctx, prod, instants, width_std: float) -> Tuple[float, float, list]:
    """(lo, hi, barrier and strike levels) of the global log-spot grid; the value is extended flat beyond it."""
    t_second, t_mat = instants[1][0], instants[-1][0]
    _, w_rest, _ = moments(ctx, t_second, t_mat, float(prod.strike))
    levels = sorted(_barrier_logs(ctx, prod) | {log(float(prod.initial_price))})
    half = width_std * sqrt(max(w_rest, 1e-12)) + 0.5
    return round(levels[0] - half, 6), round(levels[-1] + half, 6), levels


def local_points(points: int) -> int:
    """Knots of the uniform local grid at the first event: nested with the global ladder, a quarter as many cells.

    ``points // 4`` with a floor of 201 (the Gate C control) repeated one local grid across a whole ladder of
    small levels, and a repeated discretization reads as convergence (review 2026-09-18, R1). The count is now
    exact and the level is refused when it cannot be.
    """
    if points < MIN_POINTS or (points - 1) % 4 != 0:
        raise ValueError(f"a grid level needs at least {MIN_POINTS} points with (points - 1) divisible by 4, so that the "
                         f"local grid is nested too; got {points}")
    return (points - 1) // 4 + 1


def _global_sweep(ctx, prod, instants, points: int, width_std: float) -> Tuple[PLJ, PLJ]:
    """Backward sweep from maturity down to the SECOND remaining instant (events applied). Cached.

    The cached functions of ln S are anchor-free (moments between absolute
    instants). Event TIMES are not: they are measured from the valuation
    instant, so the caller must use its own. A market that is not
    time-homogeneous (a term curve re-read from another instant is another
    schedule) keys the cache on the valuation instant as well.
    """
    from quantark.intraday.timestamp import to_utc
    strike = float(prod.strike)
    t_mat = instants[-1][0]
    lo, hi, levels = _domain(ctx, prod, instants, width_std)
    from quantark.execution import greeks as summaries
    from quantark.intraday.context import market_snapshot_id, value_tree
    env = ctx.request.pricing_env
    # a greek bump replaces only ctx.pricing_env: the request's summaries would serve the unbumped sweep, so a bumped
    # context keys on the CONTENT of its own market (a bump label is not an identity), read inside the clock
    num_env = _unclocked(ctx.pricing_env)
    anchor = (None if _time_homogeneous(env) and _time_homogeneous(num_env)
              else to_utc(ctx.valuation_timestamp).isoformat())
    bumped = None if ctx.market_snapshot_id == market_snapshot_id(env) else repr(value_tree((
        summaries._vol_summary(num_env), summaries._rate_summary(num_env), summaries._div_summary(num_env))))
    key = (points, width_std, lo, hi, anchor, bumped, tuple(e.event_id for _, g in instants[1:] for e in g),
           ctx.request.variance_profile.identity(), ctx.request.session_calendar.identity(),
           repr(value_tree((summaries._vol_summary(env), summaries._rate_summary(env), summaries._div_summary(env)))),
           repr(value_tree(ctx.request.product)))
    hit = _SWEEP_CACHE.get(key)
    if hit is not None:
        return hit
    x = np.union1d(np.linspace(lo, hi, points), np.array(levels))
    v0, v1 = _terminal(ctx, prod, x, t_mat)
    v0, v1 = _apply_instant(ctx, prod, instants[-1][1], v0, v1, t_mat)
    t_next = t_mat
    for t, events in reversed(instants[1:-1]):
        m, v, disc = moments(ctx, t, t_next, strike)
        c0 = PLJ.sample(v0.x, expect(v0, v0.x + m, v, disc)[0])
        c1 = PLJ.sample(v1.x, expect(v1, v1.x + m, v, disc)[0])
        v0, v1 = _apply_instant(ctx, prod, events, c0, c1, t)
        t_next = t
    _SWEEP_CACHE[key] = (v0, v1)
    return v0, v1


def solve_snowball(ctx, points: int, width_std: float, diagnostics: Optional[dict] = None) -> Tuple[float, float, float]:
    """(value, point delta, point gamma) of the remaining claim, pending receivables excluded.

    ``diagnostics``, when given, is filled with what the solve actually did: the number of remaining
    instants, the spacing of each discretization in play (``None`` for one that is not used), the global
    domain, and ``basis`` -- the reason the result is exact when it structurally is (``"terminated"``: no
    contingent claim remains; ``"decided_at_valuation"``: the only remaining instant is the valuation
    instant and its events are decided at the known spot), else ``None``.
    """
    prod = ctx.numerical.product
    note = {} if diagnostics is None else diagnostics
    note.update(points=points, instants=0, global_spacing=None, local_spacing=None, domain=None, basis=None)
    if ctx.numerical.terminated:
        note["basis"] = "terminated"
        return 0.0, 0.0, 0.0
    if ctx.timeline.continuous_ki_barrier is not None:
        raise NotImplementedError("reference: continuous KI is not in the reference inventory")
    if hasattr(prod, "coupon_config"):
        raise NotImplementedError("reference: Phoenix coupons are not in the reference inventory")
    strike, spot = float(prod.strike), float(ctx.pricing_env.spot)
    ln_s = log(spot)
    instants = _instants(ctx)
    t1, events1 = instants[0]
    m1, v1_var, disc1 = moments(ctx, 0.0, t1, strike)
    sd1 = sqrt(v1_var)
    n_local = local_points(points)
    local_half = max(_LOCAL_STD * sd1, 1e-9)
    local = np.union1d(np.linspace(ln_s + m1 - local_half, ln_s + m1 + local_half, n_local),
                       np.array([lv for lv in _barrier_logs(ctx, prod) if abs(lv - ln_s - m1) <= local_half]))
    note["instants"] = len(instants)
    if sd1 > 0.0:                               # a zero-variance first interval is an exact shift: no local discretization
        note["local_spacing"] = 2.0 * local_half / (n_local - 1)
    if len(instants) == 1:
        c0, c1 = _terminal(ctx, prod, local, t1)
    else:
        g0, g1 = _global_sweep(ctx, prod, instants, points, width_std)
        lo, hi, _ = _domain(ctx, prod, instants, width_std)
        note["global_spacing"], note["domain"] = (hi - lo) / (points - 1), [lo, hi]
        t2 = instants[1][0]                     # this context's own time to the second instant
        m12, v12, disc12 = moments(ctx, t1, t2, strike)
        c0 = PLJ.sample(local, expect(g0, local + m12, v12, disc12)[0])
        c1 = PLJ.sample(local, expect(g1, local + m12, v12, disc12)[0])
    c0, c1 = _apply_instant(ctx, prod, events1, c0, c1, t1)
    branch = c1 if ctx.numerical.knocked_in else c0
    if t1 == 0.0:                               # an event exactly at valuation under BEFORE: pointwise, no derivative
        if len(instants) == 1:
            note["basis"] = "decided_at_valuation"
        return _decided_at_spot(prod, events1, branch, ln_s), float("nan"), float("nan")
    value, d1, d2 = expect(branch, np.array([ln_s + m1]), v1_var, disc1, derivatives=True)
    c, cx, cxx = float(value[0]), float(d1[0]), float(d2[0])
    return c, cx / spot, (cxx - cx) / (spot * spot)


def _decided_at_spot(prod, events, branch: PLJ, ln_s: float) -> float:
    """The value of events decided at the known spot, ON a barrier level included.

    A PLJ is right-continuous, which is the contract's side only for a barrier hit from below (standard KO: S >= B).
    Every autocallable barrier is inclusive (``AutocallableLifecycleTracker._barrier_hit``), so a spot exactly on a
    KI level of a standard contract, or on a reverse contract's KO level, takes the left limit.
    """
    k = int(np.searchsorted(branch.x, ln_s))
    if k >= len(branch.x) or branch.x[k] != ln_s or branch.j[k] == 0.0:
        return float(branch(ln_s)[0])
    hit_from_below = {log(float(e.barrier)): (_ko_hits_above(prod) if e.kind is EventKind.KO else bool(prod.is_reverse))
                      for e in events if e.barrier is not None and e.kind in (EventKind.KO, EventKind.KI)}
    if ln_s not in hit_from_below:
        raise ValueError(f"reference: a jump at the spot {exp(ln_s)!r} that no event of this instant owns")
    return float(branch.f[k] if hit_from_below[ln_s] else branch.left()[k])


# ---------------------------------------------------------------------------
# the identified analytical case: one remaining instant is one Gaussian integral
_QUAD_HALF_WIDTH = 12.0       # standard deviations integrated; the tail beyond is bounded, not ignored


def single_instant_quadrature(ctx) -> Optional[dict]:
    """Value, point delta and point gamma of a claim with ONE remaining instant, by adaptive quadrature.

    With a single instant ahead the value is ``disc * E[f(Y)]``, ``Y ~ N(ln S + m, v)``, where ``f`` is the
    instant's pointwise outcome: knock-out cash, the knocked-in payoff at or beyond the KI level, else the alive
    payoff. The integral is split at every level where ``f`` jumps or kinks, so Gauss-Kronrod sees smooth
    pieces; derivatives differentiate the Gaussian weight. This does not use the piecewise-linear grid at all,
    which is what makes it an exactness BASIS for a stagnating ladder rather than another level of it.

    Returns ``None`` unless exactly one instant remains, strictly ahead, on a live claim. Otherwise a dict of
    ``value``, ``delta``, ``gamma`` and their absolute ``errors`` (QUADPACK's estimate plus the bounded tail).
    """
    from scipy.integrate import quad

    prod = ctx.numerical.product
    if ctx.numerical.terminated:
        return None
    instants = _instants(ctx)
    t1, events = instants[0]
    if len(instants) != 1 or t1 <= 0.0:
        return None
    strike, spot = float(prod.strike), float(ctx.pricing_env.spot)
    m1, v, disc = moments(ctx, 0.0, t1, strike)
    if v <= 0.0:
        return None
    sd, mu = sqrt(v), log(spot) + m1
    term = ctx.timeline.terminal()
    pay = calendar_year_fraction(ctx.valuation_timestamp, term.payment_timestamp)
    delay = float(ctx.pricing_env.get_discount_factor(pay)) / float(ctx.pricing_env.get_discount_factor(t1))
    disable = bool(prod.barrier_config.disable_ko_after_ki)
    knock_outs, knock_ins = [], []
    for e in events:
        if e.kind is EventKind.KO:
            ko_pay = calendar_year_fraction(ctx.valuation_timestamp, e.payment_timestamp)
            cash = float(e.cash) * float(ctx.pricing_env.get_discount_factor(ko_pay)) / float(ctx.pricing_env.get_discount_factor(t1))
            knock_outs.append((log(float(e.barrier)), cash))
        elif e.kind is EventKind.KI:
            knock_ins.append(log(float(e.barrier)))
    above, reverse, alive = _ko_hits_above(prod), bool(prod.is_reverse), not ctx.numerical.knocked_in

    def outcome(y: float) -> float:
        for level, cash in knock_outs:
            if ((y >= level) if above else (y <= level)) and (alive or not disable):
                return cash
        hit = any((y >= level) if reverse else (y <= level) for level in knock_ins)
        state_in = (not alive) or hit
        payoff = prod.get_maturity_payoff_v1 if state_in else prod.get_maturity_payoff_v0
        return float(payoff(exp(y), ctx.pricing_env)) * delay

    lo, hi = mu - _QUAD_HALF_WIDTH * sd, mu + _QUAD_HALF_WIDTH * sd
    breaks = sorted({lv for lv in [lv for lv, _ in knock_outs] + knock_ins + [log(strike)] if lo < lv < hi})
    edges = [lo, *breaks, hi]
    import warnings
    from scipy.integrate import IntegrationWarning

    span = max(abs(outcome(lo)), abs(outcome(hi)), *(abs(c) for _, c in knock_outs), 1e-300)
    weights = (lambda z: 1.0, lambda z: z / sd, lambda z: (z * z - 1.0) / v)          # value, d/dmu, d2/dmu2
    scales = (span, span / sd, span / v)                                              # the size each integral can reach
    totals, errors = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    with warnings.catch_warnings():
        warnings.simplefilter("error", IntegrationWarning)       # an unconverged piece is no exactness basis
        try:
            for a, b in zip(edges, edges[1:]):
                # the right end of a piece is its left limit: a right-continuous jump at b belongs to the next piece
                inside = b - 1e-15 * max(1.0, abs(b))
                for k, weight in enumerate(weights):
                    part, err = quad(lambda y: outcome(min(y, inside)) * weight((y - mu) / sd)
                                     * exp(-0.5 * ((y - mu) / sd) ** 2) * _INV_SQRT_2PI / sd, a, b,
                                     epsabs=1e-12 * scales[k], epsrel=1e-12, limit=400)
                    totals[k] += part
                    errors[k] += max(err, 1e-13 * scales[k])     # QUADPACK's estimate, never credited below roundoff
        except IntegrationWarning:
            return None
    tail = 2.0 * float(ndtr(-_QUAD_HALF_WIDTH)) * span * (1.0 + _QUAD_HALF_WIDTH / sd + (_QUAD_HALF_WIDTH ** 2 + 1.0) / v)
    c, cx, cxx = (disc * t for t in totals)
    e0, e1, e2 = (disc * e + tail for e in errors)
    return {"value": c, "delta": cx / spot, "gamma": (cxx - cx) / (spot * spot),
            "errors": {"value": e0, "delta": e1 / spot, "gamma": (e2 + e1) / (spot * spot)}, "pieces": len(edges) - 1}


# ---------------------------------------------------------------------------
# what a spacing ladder cannot see: errors every level shares
def residual_components(ctx, width_std: float, policy: LadderPolicy = LADDER_POLICY) -> dict:
    """Sup-norm allowance for the errors common to every level of a ladder, with each component named.

    * ``truncation`` (analytical bound): the global grid is finite and extended flat. A path reaches beyond it
      with probability at most ``2 Phi(-a)`` per side by reflection, ``a`` the distance from the spot (bumped 1%
      either way) to the end in total standard deviations after allowing the whole drift; the local grid adds
      ``2 Phi(-15)``. The mis-valued amount is at most the value span.
    * ``saturation`` (analytical bound): ramps and jumps further than 9 standard deviations are set to 0 or 1,
      at most ``Phi(-9)`` of the total variation per expectation, per state, per instant.
    * ``roundoff`` (estimate): ``roundoff_factor x eps x span x instants``.

    ``span`` bounds the value function: terminal payoffs at the domain ends and every knock-out cash.
    """
    prod = ctx.numerical.product
    if ctx.numerical.terminated:
        return {"span": 0.0, "truncation": 0.0, "saturation": 0.0, "roundoff": 0.0, "total": 0.0, "first_sd": 0.0}
    instants = _instants(ctx)
    strike, spot = float(prod.strike), float(ctx.pricing_env.spot)
    t1, t_mat = instants[0][0], instants[-1][0]
    _, v1, _ = moments(ctx, 0.0, t1, strike)
    m_all, w_all, _ = moments(ctx, 0.0, t_mat, strike) if t_mat > 0.0 else (0.0, 0.0, 1.0)
    if len(instants) > 1:
        lo, hi, _ = _domain(ctx, prod, instants, width_std)
    else:
        lo, hi = log(spot) - _LOCAL_STD * sqrt(v1) - 1.0, log(spot) + _LOCAL_STD * sqrt(v1) + 1.0
    cashes = [abs(float(e.cash)) for e in ctx.numerical.remaining_events if e.kind is EventKind.KO]
    ends = [abs(float(f(exp(y), ctx.pricing_env))) for y in (lo, hi)
            for f in (prod.get_maturity_payoff_v0, prod.get_maturity_payoff_v1)]
    span = max(cashes + ends + [0.0])
    truncation = 2.0 * float(ndtr(-_LOCAL_STD)) * span
    if len(instants) > 1 and w_all > 0.0:
        sd_all, reach = sqrt(w_all), abs(m_all) + log(1.01)
        a_lo, a_hi = (log(spot) - reach - lo) / sd_all, (hi - log(spot) - reach) / sd_all
        truncation += 2.0 * span * (float(ndtr(-a_lo)) + float(ndtr(-a_hi)))
    saturation = 4.0 * float(ndtr(-_SAT)) * span * len(instants)
    roundoff = policy.roundoff_factor * float(np.finfo(float).eps) * span * len(instants)
    return {"span": span, "truncation": truncation, "saturation": saturation, "roundoff": roundoff,
            "total": truncation + saturation + roundoff, "first_sd": sqrt(v1)}
