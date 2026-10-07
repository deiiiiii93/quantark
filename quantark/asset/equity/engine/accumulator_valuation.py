"""An accumulator observation at the valuation instant (spec 2026-10-07 theta boundary, R1).

Theta keeps an observation that lands on the bumped date at time 0: it is observed now, on the known spot, so it
is settled here rather than priced as a zero-maturity leg (the analytic legs cannot have a zero maturity; the MC
grid would need a zero-length step, which would also move every later draw). A breach of the knock-out barrier
accrues nothing that day; under TERMINATION it ends the contract, paying the knock-out rebate.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

from quantark.util.enum import AccumulatorKnockOutType
from quantark.util.numerical import is_zero

__all__ = ["settle_valuation_instant"]


def settle_valuation_instant(product, spot: float, legs: Sequence[tuple]) -> Tuple[float, List[tuple], bool]:
    """Split the legs observed at the valuation instant off ``legs`` and settle them on ``spot``.

    Args:
        product: The ``AccumulatorOption``.
        spot: The spot at the valuation instant.
        legs: The engine's resolved legs, ``(determination_time, payment_df, ...)`` in time order.

    Returns:
        ``(value, remaining_legs, ended)``: the settled observations' present value (contract money, multiplier
        included), the legs still to be priced, and whether a TERMINATION knock-out ended the contract now
        (``remaining_legs`` is then empty and ``value`` includes the rebate).
    """
    value = 0.0
    index = 0
    while index < len(legs) and is_zero(float(legs[index][0])):
        payment_df = float(legs[index][1])
        if spot >= product.knock_out_barrier:
            if product.knock_out_type == AccumulatorKnockOutType.TERMINATION:
                rebate = product.get_knock_out_rebate_cash() * product.contract_multiplier
                return value + rebate * payment_df, [], True
        else:
            value += product.get_observation_payoff(spot) * payment_df
        index += 1
    return value, list(legs[index:]), False
