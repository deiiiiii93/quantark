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

Both conditions are statements about whole intervals, proven from the laws
``quantark.intraday.coefficients`` declares rather than from spot samples. The
variance rate is compared across every declared variance break -- clock segment
boundaries and volatility pillars -- between which total variance is affine. Each
carry coefficient is flat BY ITS LAW when it is one affine piece; otherwise it is
sampled at its own pillars and, for a quadratic cumulative law (a linearly
interpolated zero rate or yield), at the midpoint of every piece -- the first
piece from the valuation instant included (review 2026-09-16 R2). A curve family
that declares no law is inadmissible outright: matching samples would prove nothing.

Anything else is inadmissible, with the first violated condition as reason.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Optional

from quantark.intraday.coefficients import coefficient_breaks
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


def _variance_knots(breaks, t_end: float):
    """0, every declared variance break below ``t_end``, and ``t_end``."""
    return sorted({0.0, float(t_end)} | {t for t in breaks.variance if 0.0 < t < t_end})


def _carry_samples(pillars, degree: int, t_end: float):
    """Where one carry coefficient must be sampled to prove it flat up to ``t_end``; ``()`` when its law proves it.

    A cumulative carry C(t) that is a polynomial of degree d on a piece equals c * t there iff it matches at d + 1
    points of the piece. The left end of the first piece is the valuation instant, where C(0) = 0 = c * 0 for every
    family, so it matches for free and is never queried. One affine piece is therefore flat BY ITS LAW, with nothing
    to sample -- which matters: a frozen-market roll rebuilds its zero rates from discount-factor ratios, whose
    round-off (~3e-11 relative over an hour) no sample comparison at 1e-12 survives. A pillar or a quadratic law
    needs samples: every pillar and ``t_end``, plus every piece's midpoint when the law is quadratic.
    """
    if degree <= 1 and not pillars:
        return ()
    knots = sorted({0.0, float(t_end)} | {float(t) for t in pillars if 0.0 < float(t) < t_end})
    if degree >= 2:
        knots = sorted(set(knots) | {0.5 * (lo + hi) for lo, hi in zip(knots, knots[1:])})
    return tuple(t for t in knots if t > 0.0)


def _flat(values) -> bool:
    return all(is_close(v, values[0], rel_tol=_REL, abs_tol=_ABS) for v in values)


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
    # Sampling proves constancy only across a family's OWN declared pieces: a term
    # structure whose pillar sits mid-session has a different variance rate on each
    # side of it, and no pair of session-boundary samples can see that.
    breaks = coefficient_breaks(ctx, max(t_pay, T))
    if not breaks.qualified:
        return Admissibility(False, "inadmissible",
                             f"drift per unit variance cannot be qualified here: {breaks.reason}", total_variance=W)
    t_carry = max(t_pay, T)
    rate_samples = _carry_samples(breaks.rate, breaks.rate_degree, t_carry)
    div_samples = _carry_samples(breaks.div, breaks.div_degree, t_carry)

    var_knots = _variance_knots(coefficient_breaks(ctx, T), T)
    w = [0.0 if t == 0.0 else float(env.vol_surface.total_variance(strike, t, spot)) for t in var_knots]
    var_rates = [(w[i + 1] - w[i]) / (var_knots[i + 1] - var_knots[i]) for i in range(len(var_knots) - 1)]
    uniform_variance = W > 0.0 and all(is_close(v, var_rates[0], rel_tol=_REL, abs_tol=_ABS) for v in var_rates)
    # Zero rates / yields equal at every sample of a piecewise-polynomial law are a flat forward / carry.
    flat_rate = _flat([float(env.get_rate(t)) for t in rate_samples]) if rate_samples else True
    flat_div = _flat([float(env.get_div_yield(t)) for t in div_samples]) if div_samples else True
    if uniform_variance and flat_rate and flat_div:
        return Admissibility(True, "uniform_calendar_rate", "", sigma_effective=sqrt(W / T), total_variance=W)

    # Zero carry is exact equality on the same samples; a flat law is read once, at the payment horizon.
    zero_carry = (all(float(env.get_discount_factor(t)) == 1.0 for t in (rate_samples or (t_carry,)))
                  and all(float(env.get_div_yield(t)) * t == 0.0 for t in (div_samples or (t_carry,))))
    if zero_carry:
        if W <= 0.0:
            return Admissibility(False, "inadmissible", "no variance accrues before expiry: the payoff is deterministic")
        return Admissibility(True, "zero_carry_time_change", "", variance_time_maturity=float(ctx.time_map.to_trading(T)),
                             total_variance=W)
    if not uniform_variance:
        culprit = next((i for i, v in enumerate(var_rates)
                        if not is_close(v, var_rates[0], rel_tol=_REL, abs_tol=_ABS)), None)
        where = "" if culprit is None else (
            f" (dW/dtau is {var_rates[0]:.6g} on [{var_knots[0]:.6g}, {var_knots[1]:.6g}] but "
            f"{var_rates[culprit]:.6g} on [{var_knots[culprit]:.6g}, {var_knots[culprit + 1]:.6g}])")
        reason = ("drift per unit variance is not constant: the variance rate differs across the coefficient intervals "
                  "up to expiry — sessions, breaks, overnights, weekends or volatility pillars accrue differently — "
                  f"while rates/dividends are not zero{where}")
    elif not flat_rate:
        reason = "drift per unit variance is not constant: the forward rate is not flat up to the payment"
    else:
        reason = "drift per unit variance is not constant: the dividend carry is not flat up to the payment"
    return Admissibility(False, "inadmissible", reason, total_variance=W)
