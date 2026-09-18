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
from math import exp, inf, log, log2, sqrt
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.special import ndtr

from quantark.intraday.events import EventKind
from quantark.intraday.timestamp import calendar_year_fraction

_SAT = 9.0             # standard deviations beyond which a ramp / Heaviside is saturated (Phi(-9) ~ 1e-19)
_LOCAL_STD = 15.0      # half-width of the local grid at the first event, in remaining standard deviations
_CHUNK = 256
_INV_SQRT_2PI = 1.0 / sqrt(2.0 * np.pi)

#: The observed order over the three finest levels must support the O(h^2) model before Richardson is used.
ORDER_WINDOW: Tuple[float, float] = (1.5, 2.5)
#: Richardson turns O(h^2) into O(h^4) at best: successive extrapolants contract at most 16x per doubling.
#: A smaller spread is luck, and is not credited.
MAX_CONTRACTION = 16.0
#: Safety factor on the largest recent movement of a ladder whose order is not confirmed (Roache's grid
#: convergence index uses 3 for exactly that case).
UNCONFIRMED_SAFETY = 3.0
#: Differences below this relative floor are floating point: the solve is exact at this resolution.
EXACT_FLOOR = 1e-13
MIN_LADDER_LEVELS = 4


@dataclass(frozen=True)
class LadderEstimate:
    """A value and its declared error radius from one nested refinement ladder (see :func:`ladder_estimate`)."""

    value: float
    radius: float
    rule: str                                  # exact | geometric | correction | unextrapolated | unbounded
    observed_order: Optional[float]
    extrapolants: Tuple[float, ...]


def ladder_estimate(values: Sequence[float]) -> LadderEstimate:
    """Turn a nested ladder (each level halves h) of an O(h^2) scheme into a value and an error radius.

    A refinement difference is an estimate, not a bound, so both the value and the radius are chosen by
    what the ladder shows, and every branch is declared:

    * ``exact``: the last difference is floating point. Value ``v_n``, radius that difference.
    * The ladder CONFIRMS second order -- the last two differences share a sign and their observed order
      lies in :data:`ORDER_WINDOW`. The value is the Richardson extrapolant of the two finest levels,
      ``E_n = (4 v_n - v_{n-1}) / 3``, and the radius is
        - ``geometric``: successive extrapolants contract by 2x or more. If that continues the remaining
          movement is a geometric series bounded by the last spread ``|E_n - E_{n-1}|``; the spread is
          never credited below ``|E_{n-1} - E_{n-2}| / 16`` (no better than O(h^4)).
        - ``correction``: the extrapolants do not contract (their spread sits on a non-asymptotic floor).
          The claim is then only that extrapolating did not make the finest level worse: the radius is
          the whole correction ``|E_n - v_n|``, which bounds the extrapolant's error for any order >= 1.3.
    * ``unextrapolated``: the order is NOT confirmed (it is outside the window, or the differences
      alternate) but the ladder still converges. Richardson is unjustified: the value is the finest level
      and the radius is :data:`UNCONFIRMED_SAFETY` times the largest movement over the last two doublings.
    * ``unbounded``: the last difference grew. Nothing is claimed; the radius is infinite.
    """
    v = [float(x) for x in values]
    if len(v) < MIN_LADDER_LEVELS:
        raise ValueError(f"a ladder needs at least {MIN_LADDER_LEVELS} levels, got {len(v)}")
    floor = EXACT_FLOOR * max(1.0, abs(v[-1]))
    d_prev, d_last = v[-2] - v[-3], v[-1] - v[-2]
    extrapolants = tuple((4.0 * v[k] - v[k - 1]) / 3.0 for k in range(len(v) - 3, len(v)))
    if abs(d_last) <= floor:
        return LadderEstimate(v[-1], abs(d_last), "exact", None, extrapolants)
    order = log2(abs(d_prev) / abs(d_last)) if abs(d_prev) > floor else inf
    if d_prev * d_last > 0.0 and ORDER_WINDOW[0] <= order <= ORDER_WINDOW[1]:
        s1, s2 = abs(extrapolants[1] - extrapolants[0]), abs(extrapolants[2] - extrapolants[1])
        if s2 <= 0.5 * s1:
            return LadderEstimate(extrapolants[2], max(s2, s1 / MAX_CONTRACTION), "geometric", order, extrapolants)
        return LadderEstimate(extrapolants[2], abs(extrapolants[2] - v[-1]), "correction", order, extrapolants)
    if abs(d_last) <= abs(d_prev):
        return LadderEstimate(v[-1], UNCONFIRMED_SAFETY * abs(d_prev), "unextrapolated", order, extrapolants)
    return LadderEstimate(v[-1], inf, "unbounded", order, extrapolants)


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
    t_second, t_mat = instants[1][0], instants[-1][0]
    _, w_rest, _ = moments(ctx, t_second, t_mat, strike)
    levels = sorted(_barrier_logs(ctx, prod) | {log(float(prod.initial_price))})
    half = width_std * sqrt(max(w_rest, 1e-12)) + 0.5
    lo, hi = round(levels[0] - half, 6), round(levels[-1] + half, 6)
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


def solve_snowball(ctx, points: int, width_std: float) -> Tuple[float, float, float]:
    prod = ctx.numerical.product
    if ctx.numerical.terminated:
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
    local_half = max(_LOCAL_STD * sd1, 1e-9)
    local = np.union1d(np.linspace(ln_s + m1 - local_half, ln_s + m1 + local_half, max(points // 4, 201)),
                       np.array([lv for lv in _barrier_logs(ctx, prod) if abs(lv - ln_s - m1) <= local_half]))
    if len(instants) == 1:
        c0, c1 = _terminal(ctx, prod, local, t1)
    else:
        g0, g1 = _global_sweep(ctx, prod, instants, points, width_std)
        t2 = instants[1][0]                     # this context's own time to the second instant
        m12, v12, disc12 = moments(ctx, t1, t2, strike)
        c0 = PLJ.sample(local, expect(g0, local + m12, v12, disc12)[0])
        c1 = PLJ.sample(local, expect(g1, local + m12, v12, disc12)[0])
    c0, c1 = _apply_instant(ctx, prod, events1, c0, c1, t1)
    branch = c1 if ctx.numerical.knocked_in else c0
    if t1 == 0.0:                               # an event exactly at valuation under BEFORE: pointwise, no derivative
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
