"""Contract rules on a barrier's discrete observation schedule shared by every pricing route (MC, PDE, QUAD,
analytical).

Neutral module: no engine of any kind is imported here, so the PDE core, the MC engines, the QUAD adapters and the
analytical engines read one rule and raise one message: the knock-in record payoff refusal, the record deciding at
the valuation instant (``valuation_barrier_terms`` / ``valuation_double_barrier_terms``, W18b / W18c) and the level
a closed form monitors forward (``forward_barrier_level`` / ``forward_double_barrier_levels``, W18c).
"""
from __future__ import annotations

from typing import Optional, Tuple

from quantark.util.enum import ObservationType
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_close, is_greater_than_or_close


def refuse_knock_in_record_payoff(product) -> None:
    """A knock-in has one rebate (spec 3C §5.2 (a)): it is paid when NO observation hits, so no record's payoff can
    apply. A schedule record paying anything other than ``product.rebate`` is refused; a record without a payoff, or
    one paying the rebate, is the same contract. Knock-outs (a record's payoff is the hit cash) are not concerned.

    First the 1D PDE core's rule (``_KnockInAdapter.validate``); every route reads it since W18b (user ruling
    2026-10-10, option (a)).
    """
    if not getattr(product, "is_knock_in", False):
        return
    schedule = getattr(product, "observation_schedule", None)
    if schedule is None:
        return
    for rec in schedule.records:
        if rec.payoff is not None and not is_close(rec.payoff, product.rebate):
            raise ValidationError(
                f"a knock-in has one rebate ({product.rebate}); a schedule record pays {rec.payoff} "
                "(spec 3C §5.2 (a))")


# ======================================================================================================================
# The deciding record at the valuation instant (W18b, user approval 2026-10-10) and the forward level (W18c)
# ======================================================================================================================

def _beyond(product, spot: float, level: float) -> bool:
    """``product.is_barrier_hit`` at ``level``: up >= level, down <= level."""
    return spot >= level if product.is_up_barrier else spot <= level


def _record_time(record, pricing_env) -> Optional[float]:
    """A record's time from the valuation instant, or ``None`` when it lies behind it. A dated record on the
    valuation instant is at 0.0 (the settlement resolver's rule); one before it is past and never resolved (the year
    fraction refuses a date behind the valuation date)."""
    if record.observation_date is not None and record.observation_time is None:
        if record.observation_date < pricing_env.valuation_date:
            return None
        if record.observation_date == pricing_env.valuation_date:
            return 0.0
    t = float(record.resolve_time(pricing_env))
    return t if is_greater_than_or_close(t, 0.0) else None


def _knock_out_cash(product) -> float:
    """The product's knock-out cash, the default of a record without a payoff: a sharkfin's ``knock_out_rebate``,
    every other barrier product's ``rebate``."""
    if hasattr(product, "knock_out_rebate"):
        return float(product.knock_out_rebate)
    return float(product.rebate)


def _deciding_record(product, pricing_env):
    """The first record at or after the valuation instant of a DISCRETE schedule, or ``None`` (CONTINUOUS / EXPIRY
    monitoring, no schedule, or no record left)."""
    schedule = getattr(product, "observation_schedule", None)
    if product.observation_type != ObservationType.DISCRETE or schedule is None or not schedule.records:
        return None
    first = None
    for record in schedule.records:
        t = _record_time(record, pricing_env)
        if t is not None and (first is None or t < first[0]):
            first = (t, record)
    return None if first is None else first[1]


def valuation_barrier_terms(product, pricing_env) -> Tuple[float, float]:
    """(level, knock-out cash) deciding a DISCRETE barrier at the valuation instant (W18b, user approval 2026-10-10).

    The first record at or after the valuation instant decides -- the record on the valuation date when there is
    one, the next observation otherwise (the barrier PDE's ``_next_observation`` rule): its own level (a record
    without one takes ``product.barrier``, the product's normalisation) and its payoff (unset: the product's
    knock-out cash, ``rebate`` / a sharkfin's ``knock_out_rebate``), as the barrier MC's discrete path reads each
    resolved record (``_build_discrete_grid``, ``payoffs[first_idx]``). CONTINUOUS / EXPIRY monitoring, or no record
    left: ``product.barrier`` and the product's knock-out cash. Shared by the barrier MC and the analytical engines
    (W18c).
    """
    record = _deciding_record(product, pricing_env)
    if record is None:
        return float(product.barrier), _knock_out_cash(product)
    level = record.barrier if record.barrier is not None else product.barrier
    cash = record.payoff if record.payoff is not None else _knock_out_cash(product)
    return float(level), float(cash)


def valuation_double_barrier_terms(product, pricing_env) -> Tuple[float, float, float]:
    """(upper, lower, knock-out cash) deciding a DISCRETE two-barrier schedule at the valuation instant (W18c, A8):
    ``valuation_barrier_terms`` on the record's ``upper_barrier`` / ``lower_barrier`` (unset: the product's)."""
    record = _deciding_record(product, pricing_env)
    if record is None:
        return float(product.upper_barrier), float(product.lower_barrier), _knock_out_cash(product)
    upper = record.upper_barrier if record.upper_barrier is not None else product.upper_barrier
    lower = record.lower_barrier if record.lower_barrier is not None else product.lower_barrier
    cash = record.payoff if record.payoff is not None else _knock_out_cash(product)
    return float(upper), float(lower), float(cash)


def _uniform_remaining_levels(product, pricing_env):
    """The one ``(barrier, upper, lower)`` every remaining record (t > 0) of a DISCRETE schedule shares; ``()`` when
    the product is not discretely monitored or no record remains; ``None`` when the levels differ."""
    schedule = getattr(product, "observation_schedule", None)
    if product.observation_type != ObservationType.DISCRETE or schedule is None or not schedule.records:
        return ()
    levels = schedule.remaining_levels(pricing_env)
    if not levels:
        return ()
    return levels[0] if len(levels) == 1 else None


def forward_barrier_level(product, pricing_env) -> Optional[float]:
    """The level a single-barrier closed form monitors forward (W18c, spec 2026-10-13 I4).

    DISCRETE: the level every remaining record (t > 0) shares -- it may differ from ``product.barrier``; ``None``
    when they differ (no closed form: ``ObservationSchedule.assert_analytical_ready`` refuses every forward route).
    CONTINUOUS / EXPIRY monitoring, or no record ahead (nothing to monitor forward): ``product.barrier``.
    """
    levels = _uniform_remaining_levels(product, pricing_env)
    if levels is None:
        return None
    if not levels or levels[0] is None:
        return float(product.barrier)
    return float(levels[0])


def forward_double_barrier_levels(product, pricing_env) -> Optional[Tuple[float, float]]:
    """``forward_barrier_level`` for a two-barrier product: the (upper, lower) every remaining record shares (A8:
    both levels identical across the remaining records); ``None`` when they differ; the product's levels outside
    DISCRETE monitoring or with no record ahead."""
    levels = _uniform_remaining_levels(product, pricing_env)
    if levels is None:
        return None
    if not levels:
        return float(product.upper_barrier), float(product.lower_barrier)
    _, upper, lower = levels
    return (float(upper if upper is not None else product.upper_barrier),
            float(lower if lower is not None else product.lower_barrier))


def refuse_mixed_forward_levels(schedule, pricing_env, **ready_kwargs) -> None:
    """Raise the "no closed form" refusal of a schedule whose remaining records carry different levels (W18c, A8):
    ``ObservationSchedule.assert_analytical_ready`` raises it (``ValidationError``), after its frequency and payoff
    checks. Called by an analytical engine once ``forward_barrier_level`` / ``forward_double_barrier_levels``
    returned ``None``."""
    schedule.assert_analytical_ready(pricing_env=pricing_env, **ready_kwargs)
    raise ValidationError(  # unreachable: forward_*_level returned None, so the level check above has raised
        "Analytical barrier shift requires one barrier level across the remaining observation records.")
