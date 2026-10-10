"""Discrete observation times for the vol-model barrier routes (local-vol / Heston / SLV MC, Heston / SLV PDE).

Shared by the MC engines and the PDE solvers (no engine of either kind is imported here).
"""
from __future__ import annotations

from typing import List

from quantark.util.exceptions import ValidationError


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
