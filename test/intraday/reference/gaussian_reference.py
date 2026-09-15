"""Independent Gaussian-transition reference for the intraday Gate C.

NOT a production engine, and it imports nothing from the QUAD / PDE / MC
engine packages: it reads only the resolved intraday context (numerical twin,
remaining events, clock-wrapped surface, curves) and the product's own payoff
functions.

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
O(h^2), estimated by Richardson over three grid levels.

At the first remaining event the continuation is re-evaluated exactly on a local
grid scaled to the remaining standard deviation around ln S, so the final
(possibly one-second) propagation never sees the global grid's kinks; that last
step is a single exact pointwise expectation with exact derivatives. A zero-
variance interval is an exact shift.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt
from typing import Dict, Tuple

import numpy as np
from scipy.special import ndtr
from scipy.stats import norm

from quantark.intraday.events import EventKind
from quantark.intraday.timestamp import calendar_year_fraction

_SAT = 9.0             # standard deviations beyond which a ramp / Heaviside is saturated (Phi(-9) ~ 1e-19)
_LOCAL_STD = 15.0      # half-width of the local grid at the first event, in remaining standard deviations
_CHUNK = 256
_INV_SQRT_2PI = 1.0 / sqrt(2.0 * np.pi)


@dataclass(frozen=True)
class ReferenceResult:
    price: float
    delta: float
    gamma: float
    grid_points: int
    uncertainty_price: float
    uncertainty_delta: float
    uncertainty_gamma: float
    method: str = "dense_gaussian_backward"


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


def _global_sweep(ctx, prod, instants, points: int, width_std: float) -> Tuple[PLJ, PLJ, float]:
    """Backward sweep from maturity down to the SECOND remaining instant (events applied). Cached."""
    strike = float(prod.strike)
    t_second, t_mat = instants[1][0], instants[-1][0]
    _, w_rest, _ = moments(ctx, t_second, t_mat, strike)
    levels = sorted(_barrier_logs(ctx, prod) | {log(float(prod.initial_price))})
    half = width_std * sqrt(max(w_rest, 1e-12)) + 0.5
    lo, hi = round(levels[0] - half, 6), round(levels[-1] + half, 6)
    from quantark.execution import greeks as summaries
    from quantark.intraday.context import value_tree
    key = (points, width_std, lo, hi, tuple(e.event_id for _, g in instants[1:] for e in g),
           ctx.request.variance_profile.identity(), ctx.request.session_calendar.identity(),
           repr(value_tree((summaries._vol_summary(ctx.request.pricing_env), summaries._rate_summary(ctx.request.pricing_env),
                            summaries._div_summary(ctx.request.pricing_env)))),
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
    _SWEEP_CACHE[key] = (v0, v1, t_next)
    return v0, v1, t_next


def _solve_snowball(ctx, points: int, width_std: float) -> Tuple[float, float, float]:
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
        g0, g1, t2 = _global_sweep(ctx, prod, instants, points, width_std)
        m12, v12, disc12 = moments(ctx, t1, t2, strike)
        c0 = PLJ.sample(local, expect(g0, local + m12, v12, disc12)[0])
        c1 = PLJ.sample(local, expect(g1, local + m12, v12, disc12)[0])
    c0, c1 = _apply_instant(ctx, prod, events1, c0, c1, t1)
    branch = c1 if ctx.numerical.knocked_in else c0
    if t1 == 0.0:                               # an event exactly at valuation under BEFORE: pointwise, no derivative
        return float(branch(ln_s)[0]), float("nan"), float("nan")
    value, d1, d2 = expect(branch, np.array([ln_s + m1]), v1_var, disc1, derivatives=True)
    c, cx, cxx = float(value[0]), float(d1[0]), float(d2[0])
    return c, cx / spot, (cxx - cx) / (spot * spot)


def reference_snowball(ctx, *, points=(2001, 4001, 8001), width_std=8.0) -> ReferenceResult:
    """Price, delta, gamma of the remaining claim (excluding pending receivables) with Richardson uncertainty."""
    vals = [_solve_snowball(ctx, p, width_std) for p in points]
    (_, _, _), (p1, dl1, g1), (p2, dl2, g2) = vals

    def unc(a, b):
        return abs(b - a) / 3.0 if np.isfinite(a) and np.isfinite(b) else float("nan")

    return ReferenceResult(price=p2, delta=dl2, gamma=g2, grid_points=points[-1],
                           uncertainty_price=unc(p1, p2), uncertainty_delta=unc(dl1, dl2), uncertainty_gamma=unc(g1, g2))


# ---------------------------------------------------------------------------
# closed forms
def reference_digital(ctx) -> ReferenceResult:
    """Cash-or-nothing digital: N(d2) with W from the clock and R - Q from the curves; exact (no uncertainty)."""
    prod, env = ctx.numerical.product, ctx.pricing_env
    S, K, T = float(env.spot), float(prod.strike), float(ctx.numerical.maturity_tau)
    W = float(env.vol_surface.total_variance(K, T, S)) if T > 0.0 else 0.0
    R = -log(float(env.get_discount_factor(T))) if T > 0.0 else 0.0
    Q = float(env.get_div_yield(T)) * T if T > 0.0 else 0.0
    pay = calendar_year_fraction(ctx.valuation_timestamp, ctx.timeline.terminal().payment_timestamp)
    df_pay = float(env.get_discount_factor(pay)) if pay > 0.0 else 1.0
    payout = float(prod.payout) * float(getattr(prod, "contract_multiplier", 1.0))
    call = prod.option_type.name == "CALL"
    if W == 0.0:
        F = S * exp(R - Q)
        pays = F > K if call else F < K
        return ReferenceResult(payout * df_pay * (1.0 if pays else 0.0), float("nan"), float("nan"), 0, 0.0, 0.0, 0.0,
                               "closed_form_deterministic")
    sw = sqrt(W)
    d2 = (log(S / K) + (R - Q) - 0.5 * W) / sw
    sign = 1.0 if call else -1.0
    price = payout * df_pay * norm.cdf(sign * d2)
    delta = sign * payout * df_pay * norm.pdf(d2) / (S * sw)
    gamma = -delta * (1.0 + d2 / sw) / S
    return ReferenceResult(price, delta, gamma, 0, 0.0, 0.0, 0.0, "closed_form_bs_effective_variance")


def _band(s0: float, lo, hi, sw: float) -> Tuple[float, float]:
    """(P, E[S_T 1{.}]) for lo <= S_T < hi, ln S_T ~ N(ln s0 - sw^2/2, sw^2); None = unbounded."""
    def tail(level):
        if level is None:
            return 1.0, s0
        d = (log(s0 / level) - 0.5 * sw * sw) / sw
        return float(ndtr(d)), s0 * float(ndtr(d + sw))
    p_lo, e_lo = tail(lo)
    if hi is None:
        return p_lo, e_lo
    p_hi, e_hi = tail(hi)
    return p_lo - p_hi, e_lo - e_hi


def _payoff_band(s0: float, K: float, is_call: bool, lo, hi, sw: float) -> float:
    """E[vanilla payoff(S_T) 1{lo <= S_T < hi}] under zero carry."""
    if is_call:
        lo = K if lo is None else max(lo, K)
    else:
        hi = K if hi is None else min(hi, K)
    if lo is not None and hi is not None and hi <= lo:
        return 0.0
    p, e = _band(s0, lo, hi, sw)
    return e - K * p if is_call else K * p - e


def barrier_zero_carry(S: float, K: float, H: float, u: float, sigma: float, *, is_call: bool, is_up: bool,
                       is_knock_out: bool, rebate: float = 0.0) -> float:
    """Continuously monitored single barrier under r = q = 0, total variance sigma^2 u.

    Zero carry makes S a martingale with log drift -sigma^2/2 per unit variance
    time, and the reflection principle gives, for a payoff f supported beyond the
    barrier, E[f(S_T) 1{hit}] = (S/H) E^{S'}[f(S_T)] with S' = H^2/S (lambda = 1/2
    in Reiner-Rubinstein). Paths ending on the far side have hit for certain.
    A knock-out's rebate is paid on hit, a knock-in's at expiry if never hit;
    at r = 0 its timing is irrelevant.
    """
    if u <= 0.0:
        raise ValueError("variance time must be positive")
    sw = sigma * sqrt(u)
    vanilla = _payoff_band(S, K, is_call, None, None, sw)
    if (S >= H) if is_up else (S <= H):
        return rebate if is_knock_out else vanilla
    reflected = H * H / S
    if is_up:
        knock_in = _payoff_band(S, K, is_call, H, None, sw) + (S / H) * _payoff_band(reflected, K, is_call, None, H, sw)
        p_hit = _prob_hit_up(S, H, sw)
    else:
        knock_in = _payoff_band(S, K, is_call, None, H, sw) + (S / H) * _payoff_band(reflected, K, is_call, H, None, sw)
        p_hit = _prob_hit_down(S, H, sw)
    if is_knock_out:
        return vanilla - knock_in + rebate * p_hit
    return knock_in + rebate * (1.0 - p_hit)


def _prob_hit_up(S, H, sw):
    """P(max_t S_t >= H) for S < H under zero carry (log drift -sw^2/2 per unit variance time)."""
    return norm.cdf((log(S / H) - 0.5 * sw * sw) / sw) + (S / H) * norm.cdf((log(S / H) + 0.5 * sw * sw) / sw)


def _prob_hit_down(S, H, sw):
    """P(min_t S_t <= H) for S > H under zero carry."""
    return norm.cdf((log(H / S) + 0.5 * sw * sw) / sw) + (S / H) * norm.cdf((log(H / S) - 0.5 * sw * sw) / sw)
