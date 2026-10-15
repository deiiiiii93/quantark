"""
The value of a discretely monitored barrier contract through its first future observation.

The analytic barrier engines price a discrete schedule with the Broadie-Glasserman-Kou shift: the continuous
formula on a barrier moved away from the spot. That formula has no meaning for a spot beyond the CONTRACTUAL
barrier when no observation is due now: the contract is still alive and the next observation decides it. The
value is then taken through that observation exactly - the Gaussian step from the valuation instant to the first
future observation ``t1``, the barrier test at ``t1`` - and the engine's own shifted formula is used only from
``t1`` on, where it applies (the alive side of the barrier)::

    V(S0) = exp(-r t1) * E[ hit(S_t1) ? H(S_t1) : C(S_t1) ]

``H`` is the value at ``t1`` of a breach at ``t1``, ``C`` the engine's continuation from ``t1``. Both are values
in ``t1`` money under the engine's flat ``(r, q, sigma)``. The expectation is a composite Gauss-Legendre rule in
the standard-normal variable with the barrier(s) as panel edges, so each panel's integrand is smooth, truncated at
``TRUNCATION_SD`` standard deviations (normal tail below 1e-22). Each panel set is integrated at two orders and the
difference is the error check: beyond ``Tolerance.RELATIVE`` (relative to the value, floor 1) it raises
``NumericalError`` rather than return an unresolved number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np
from scipy.special import ndtr

from quantark.asset.equity.product.option.observation_schedule import ObservationSchedule
from quantark.util.exceptions import NumericalError, PricingError, ValidationError
from quantark.util.enum import ObservationAggregation
from quantark.util.enum.engine_enums import DiscreteBarrierApproximation
from quantark.util.numerical import Tolerance, is_zero, safe_exp, safe_log, safe_sqrt

TRUNCATION_SD = 10.0
LOW_ORDER = 8
HIGH_ORDER = 12


@dataclass(frozen=True)
class FirstObservation:
    """The first resolved observation strictly after the valuation instant and what follows it.

    Attributes:
        time: its observation time ``t1`` (> 0).
        lower: the level breached at or below (None: no lower edge).
        upper: the level breached at or above (None: no upper edge).
        settlement_time: the payment time of a cash amount triggered by it.
        payoff: its resolved cash amount.
        remaining: how many observations follow it.
    """

    time: float
    lower: Optional[float]
    upper: Optional[float]
    settlement_time: float
    payoff: float
    remaining: int


def resolve_current_records(schedule, pricing_env, **resolve_kwargs) -> List:
    """The schedule's records at or after the valuation instant, resolved (past records are not resolvable:
    the settlement resolver refuses a negative determination time)."""
    current = [
        rec
        for rec in schedule.records
        if (rec.observation_time is not None and rec.observation_time >= 0.0)
        or (rec.observation_time is None and rec.observation_date is not None
            and rec.observation_date >= pricing_env.valuation_date)
    ]
    if not current:
        return []
    return ObservationSchedule(
        records=current,
        aggregation_mode=schedule.aggregation_mode,
        frequency=schedule.frequency,
    ).resolve(pricing_env, **resolve_kwargs)


def validate_discrete_barrier_approximation(value) -> DiscreteBarrierApproximation:
    """``value`` when it is a ``DiscreteBarrierApproximation`` member; anything else raises ``ValidationError``."""
    if not isinstance(value, DiscreteBarrierApproximation):
        raise ValidationError(
            "discrete_barrier_approximation must be a DiscreteBarrierApproximation member "
            f"({', '.join(m.name for m in DiscreteBarrierApproximation)}), got {value!r}"
        )
    return value


def projects_through_first_observation(
    mode: DiscreteBarrierApproximation, beyond: bool, resolved_records: Callable[[], Sequence]
) -> bool:
    """Whether a DISCRETE schedule with no record deciding now is valued through its first future observation.

    ``BGK`` never projects (the shifted closed form at today's spot). ``FIRST_OBSERVATION_BGK`` projects beyond the
    level (as before the switch) and, inside it, whenever an observation remains after the valuation instant
    (``resolved_records()``, read only then; none left: the closed form, as before).
    """
    if mode is DiscreteBarrierApproximation.BGK:
        return False
    return beyond or any(
        rec.observation_time > 0.0 and not is_zero(rec.observation_time) for rec in resolved_records()
    )


def observed_at_valuation(resolved_records: Sequence) -> List:
    """The resolved records observed at the valuation instant."""
    return [rec for rec in resolved_records if is_zero(rec.observation_time)]


def first_future_observation(resolved_records: Sequence, schedule, *, single: bool, is_up: bool = True) -> FirstObservation:
    """The first record strictly after the valuation instant (``single``: one barrier on the side ``is_up``)."""
    if schedule.aggregation_mode != ObservationAggregation.STOP_FIRST_HIT:
        raise PricingError(
            "A spot beyond a discretely monitored barrier is priced through the first observation only for "
            f"STOP_FIRST_HIT schedules, got {schedule.aggregation_mode.value}"
        )
    future = [rec for rec in resolved_records if rec.observation_time > 0.0 and not is_zero(rec.observation_time)]
    if not future:
        raise PricingError("No observation remains after the valuation instant.")
    rec = future[0]
    if single:
        lower, upper = (None, rec.barrier) if is_up else (rec.barrier, None)
    else:
        lower, upper = rec.lower_barrier, rec.upper_barrier
    return FirstObservation(
        time=float(rec.observation_time),
        lower=lower,
        upper=upper,
        settlement_time=float(rec.settlement_time if rec.settlement_time is not None else rec.observation_time),
        payoff=float(rec.payoff),
        remaining=len(future) - 1,
    )


def through_first_observation(
    *,
    spot: float,
    first: FirstObservation,
    rate: float,
    div: float,
    vol: float,
    hit_value: Callable[[float], float],
    continuation: Callable[[float], float],
    remaining_time: float,
    hit_is_constant: bool = False,
) -> float:
    """``exp(-r t1) E[hit ? hit_value(S_t1) : continuation(S_t1)]`` under GBM with flat ``rate``, ``div``, ``vol``.

    ``remaining_time`` (maturity minus ``t1``) sets the panel width: the continuation varies on the scale of its own
    remaining standard deviation, so panels are ``min(1, sqrt(remaining_time / t1))`` wide in the normal variable.
    ``hit_is_constant``: the breach value does not depend on the spot (cash), so its region is that value times the
    breach probability in closed form.
    """
    t1 = first.time
    sd = vol * float(safe_sqrt(t1))
    mean = float(safe_log(spot)) + (rate - div - 0.5 * vol * vol) * t1

    def z_of(level: float) -> float:
        return (float(safe_log(level)) - mean) / sd

    lo, hi = -TRUNCATION_SD, TRUNCATION_SD
    regions = []  # (a, b, function): breached below, alive, breached above
    z_low = z_of(first.lower) if first.lower is not None else None
    z_up = z_of(first.upper) if first.upper is not None else None
    if z_low is not None and z_low > lo:
        regions.append((lo, min(z_low, hi), hit_value))
    a = max(lo, z_low) if z_low is not None else lo
    b = min(hi, z_up) if z_up is not None else hi
    if a < b:
        regions.append((a, b, continuation))
    if z_up is not None and z_up < hi:
        regions.append((max(z_up, lo), hi, hit_value))

    width = min(1.0, math.sqrt(max(remaining_time, 0.0) / t1)) if remaining_time > 0.0 else 1.0
    width = max(width, 0.05)

    exact_part = 0.0
    if hit_is_constant:
        cash = hit_value(spot)
        breach_mass = (float(ndtr(z_low)) if z_low is not None else 0.0) + (
            float(ndtr(-z_up)) if z_up is not None else 0.0
        )
        exact_part = cash * breach_mass
        regions = [region for region in regions if region[2] is not hit_value]

    values = []
    for order in (LOW_ORDER, HIGH_ORDER):
        gx, gw = np.polynomial.legendre.leggauss(order)
        total = exact_part
        for a, b, fn in regions:
            if b <= a:
                continue
            panels = max(1, int(math.ceil((b - a) / width)))
            cuts = np.linspace(a, b, panels + 1)
            for left, right in zip(cuts[:-1], cuts[1:]):
                half, mid = 0.5 * (right - left), 0.5 * (right + left)
                for x, w in zip(mid + half * gx, half * gw):
                    density = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
                    total += w * density * fn(float(safe_exp(mean + sd * x)))
        values.append(total)

    low, high = values
    if abs(high - low) > Tolerance.RELATIVE * max(1.0, abs(high)):
        raise NumericalError(
            "The expectation through the first observation did not resolve: "
            f"orders {LOW_ORDER} and {HIGH_ORDER} differ by {abs(high - low):.3e}"
        )
    return float(safe_exp(-rate * t1)) * high
