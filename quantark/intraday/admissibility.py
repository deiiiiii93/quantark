"""When a closed-form continuous barrier / touch formula is exact on the intraday clock.

Reiner-Rubinstein and the one-touch formula assume ln S has a constant drift
per unit of variance. On the intraday clock that holds in exactly two cases:

* ``uniform_calendar_rate``: the instantaneous variance rate dW/dtau is the same
  on every segment up to expiry (no zero-weight segment, no weekend or holiday
  folded into an overnight), AND the forward rate and the dividend carry are
  flat up to the payment. The clock-wrapped surface then returns one constant
  sigma_eff = sqrt(W/T) and the engine's inputs are exact on the calendar axis.
* ``zero_carry_time_change``: r = q = 0 exactly. Then X = ln S - ln S0 satisfies
  X = -U/2 + B(U) in variance time U, a Brownian motion with unit drift ratio
  whatever the profile (plateaus included): price with maturity u(T) and
  sigma' = sqrt(W/u) under zero rates; pay-at-hit timing is irrelevant at r = 0.

Anything else is inadmissible, with the first violated condition as reason.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Optional

from quantark.intraday.timestamp import calendar_year_fraction
from quantark.util.numerical import is_close

_REL = 1e-12
_ABS = 1e-14


@dataclass(frozen=True)
class Admissibility:
    admissible: bool
    mode: str                                   # "uniform_calendar_rate" | "zero_carry_time_change" | "inadmissible"
    reason: str
    sigma_effective: Optional[float] = None     # uniform mode: constant calendar-axis sigma
    variance_time_maturity: Optional[float] = None   # time-change mode: u(T) on the trading clock
    total_variance: Optional[float] = None


def _knots(ctx, t_end: float):
    taus = {0.0, t_end}
    for s in ctx.time_map.segments:
        for t in (s.tau_start, s.tau_end):
            if 0.0 < t < t_end:
                taus.add(t)
    return sorted(taus)


def _rates(ctx, knots):
    """Discount factors, cumulative carry, and zero rate / dividend yield at every positive knot.

    A zero rate equal at every knot is a flat forward between them; comparing
    zero rates avoids finite-differencing discount factors over sub-day steps
    (relative round-off ~1e-11 there).
    """
    env = ctx.pricing_env
    positive = [t for t in knots if t > 0.0]
    dfs = [float(env.get_discount_factor(t)) for t in positive]
    carry = [float(env.get_div_yield(t)) * t for t in positive]
    zero = [float(env.get_rate(t)) for t in positive]
    div = [float(env.get_div_yield(t)) for t in positive]
    return dfs, carry, zero, div


def analytical_barrier_admissibility(ctx) -> Admissibility:
    env, num = ctx.pricing_env, ctx.numerical
    T = float(num.maturity_tau)
    if T <= 0.0:
        return Admissibility(False, "inadmissible", "expiry is at the valuation instant")
    if getattr(env.vol_surface, "is_smile", False):
        return Admissibility(False, "inadmissible", "the volatility surface has a smile: no single barrier volatility")
    strike = float(getattr(num.product, "strike", None) or getattr(num.product, "barrier"))
    spot = float(env.spot)
    W = float(env.vol_surface.total_variance(strike, T, spot))
    t_pay = calendar_year_fraction(ctx.valuation_timestamp, ctx.timeline.terminal().payment_timestamp)
    pay_knots = _knots(ctx, max(t_pay, T))
    dfs, carry, fwd, div = _rates(ctx, pay_knots)

    var_knots = _knots(ctx, T)
    w = [0.0 if t == 0.0 else float(env.vol_surface.total_variance(strike, t, spot)) for t in var_knots]
    var_rates = [(w[i + 1] - w[i]) / (var_knots[i + 1] - var_knots[i]) for i in range(len(var_knots) - 1)]
    uniform_variance = W > 0.0 and all(is_close(v, var_rates[0], rel_tol=_REL, abs_tol=_ABS) for v in var_rates)
    flat_rate = all(is_close(f, fwd[0], rel_tol=_REL, abs_tol=_ABS) for f in fwd)
    flat_div = all(is_close(d, div[0], rel_tol=_REL, abs_tol=_ABS) for d in div)
    if uniform_variance and flat_rate and flat_div:
        return Admissibility(True, "uniform_calendar_rate", "", sigma_effective=sqrt(W / T), total_variance=W)

    zero_carry = all(df == 1.0 for df in dfs) and all(c == 0.0 for c in carry)
    if zero_carry:
        if W <= 0.0:
            return Admissibility(False, "inadmissible", "no variance accrues before expiry: the payoff is deterministic")
        return Admissibility(True, "zero_carry_time_change", "", variance_time_maturity=float(ctx.time_map.to_trading(T)),
                             total_variance=W)
    if not uniform_variance:
        reason = ("drift per unit variance is not constant: the variance clock is not a constant calendar rate up to expiry "
                  "(sessions, breaks, overnights or weekends accrue differently) while rates/dividends are not zero")
    elif not flat_rate:
        reason = "drift per unit variance is not constant: the forward rate is not flat up to the payment"
    else:
        reason = "drift per unit variance is not constant: the dividend carry is not flat up to the payment"
    return Admissibility(False, "inadmissible", reason, total_variance=W)
