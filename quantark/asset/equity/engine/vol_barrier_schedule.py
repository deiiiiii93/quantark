"""Discrete observation schedules on the vol-model barrier routes (local-vol / Heston / SLV MC, Heston / SLV PDE):
the observation times (W12), the schedule's barrier level and rebate, and the first-hit-only aggregation (W18).

Shared by the MC engines and the PDE solvers (no engine of either kind is imported here).
"""
from __future__ import annotations

from typing import List, Tuple

from quantark.util.enum.option_enums import ObservationAggregation
from quantark.util.exceptions import ValidationError


def refuse_non_first_hit(engine_name: str, product) -> None:
    """The vol-model barrier kernels monitor a discrete schedule first-hit (``STOP_FIRST_HIT``): the first
    breaching observation decides, once. Any other aggregation is refused, numeric and dated alike (W18, user ruling
    2026-10-10 (a)); CONTINUOUS / EXPIRY monitoring carries no schedule and is unaffected."""
    schedule = getattr(product, "observation_schedule", None)
    if schedule is None:
        return
    mode = schedule.aggregation_mode
    if mode != ObservationAggregation.STOP_FIRST_HIT:
        # TODO(vol-aggregation): ACCUMULATE / BEST / WORST need per-observation cash in the vol kernels.
        raise ValidationError(
            f"{engine_name}: the vol-model barrier kernels monitor a discrete schedule first-hit "
            f"({ObservationAggregation.STOP_FIRST_HIT.value}) only; aggregation {mode.value} is refused")


def schedule_barrier_terms(product) -> Tuple[float, float]:
    """(barrier level, rebate) the vol kernels price: the discrete schedule's own terms (W18, user ruling 2026-10-10).

    The routes accept a discrete schedule only when every record carries the same level and payoff (varying terms
    are refused before this is reached). The records' common level is the level monitored, exactly as the
    Black-Scholes barrier MC monitors each record at its own ``rec.barrier`` (``_build_discrete_grid``); a record
    without a level takes ``product.barrier``, the product's own normalisation rule. A knock-out pays the hit
    record's payoff (the Black-Scholes MC's ``payoffs[first_idx]``), a missing payoff ``product.rebate``. A
    knock-in's rebate is paid when no observation hits, so no record's payoff applies: ``product.rebate``, as the
    Black-Scholes MC pays it. Without a schedule (CONTINUOUS / EXPIRY): ``product.barrier`` and ``product.rebate``.
    """
    barrier, rebate = float(product.barrier), float(product.rebate)
    schedule = getattr(product, "observation_schedule", None)
    records = getattr(schedule, "records", None) if schedule is not None else None
    if not records:
        return barrier, rebate
    levels = {rec.barrier if rec.barrier is not None else product.barrier for rec in records}
    payoffs = {rec.payoff if rec.payoff is not None else product.rebate for rec in records}
    if len(levels) > 1 or len(payoffs) > 1:
        raise ValidationError("per-observation ObservationSchedule (varying barriers/payoffs) is out of scope for v1")
    level, = levels
    payoff, = payoffs
    return float(level), (float(payoff) if product.is_knock_out else rebate)


def discrete_observation_times(product, pricing_env, maturity: float) -> List[float]:
    """The DISCRETE schedule's observation times, in years from the valuation instant.

    A numeric schedule is read from ``product.observation_dates`` verbatim (``BarrierOption`` mirrors its times
    there), so a numeric contract's kernel inputs do not move. A DATED schedule has no times until it is resolved:
    it is resolved against ``pricing_env`` exactly as the Black-Scholes barrier MC resolves it (single barrier, the
    product's barrier and rebate as defaults, the product's settlement context), and the resolved times are what the
    kernels receive (W12). A record behind the valuation date is refused by the resolution, as on every other route.

    The limits are the numeric path's, unchanged and identical for both forms: no observation at all, or one after
    ``maturity``, is refused. What the resolved records carry beyond their time (per-record settlement, a pay-at-hit
    delay) is checked by the routes' settlement guard, which resolves the same schedule; per-observation varying
    barriers or payoffs are refused before this is reached.
    """
    schedule = getattr(product, "observation_schedule", None)
    if schedule is not None and schedule.records and schedule.uses_dates():
        resolved = schedule.resolve(
            pricing_env,
            default_barrier=product.barrier,
            default_payoff=product.rebate,
            require_single=True,
            product=product,
        )
        dates = [float(rec.observation_time) for rec in resolved]
    else:
        dates = product.observation_dates or []
    if not dates:
        raise ValidationError("discrete monitoring requires observation_dates")
    T = float(maturity)
    for d in dates:
        if d > T + 1e-12:
            raise ValidationError(f"observation date {d} exceeds maturity {T}")
    return list(dates)
