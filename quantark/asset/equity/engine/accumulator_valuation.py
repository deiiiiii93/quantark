"""An accumulator observation at the valuation instant (spec 2026-10-07 theta boundary, R1).

Theta keeps an observation that lands on the bumped date at time 0: it is observed now, on the known spot, so it
is settled here rather than priced as a zero-maturity leg (the analytic legs cannot have a zero maturity; the MC
grid would need a zero-length step, which would also move every later draw). A breach of the knock-out barrier
accrues nothing that day; under TERMINATION it ends the contract, paying the knock-out rebate.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

from quantark.util.enum import AccumulatorKnockOutType
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_zero

__all__ = ["realized_value", "settle_at_expiry", "settle_valuation_instant"]


def realized_value(product, pricing_env, terminal_timing=None) -> float:
    """The booked accrual of ``past_observations`` as the accumulator engines count it: at face when each fixing
    settled on its own date, discounted to the terminal payment when settlement is deferred to expiry."""
    realized = product.get_realized_accrual()
    if product.settlement_at_expiry:
        if terminal_timing is None:
            from quantark.asset.equity.engine.settlement_support import resolve_terminal_timing
            terminal_timing = resolve_terminal_timing(product, pricing_env)
        realized *= terminal_timing.payment_df
    return realized


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


def settle_at_expiry(product, spot: float, legs: Sequence[tuple]) -> Tuple[float, bool]:
    """Settle an accumulator at its maturity instant (maturity 0.0): every remaining observation is AT the instant.

    The observation on the last day (a dated contract valued on its exercise date, or a numeric one aged onto it
    keeping the landing record) is live and decided on ``spot``, exactly as ``settle_valuation_instant`` decides it
    at any other instant (user ruling 2026-10-09, option (a)).

    Returns:
        ``(value, ended)``: the settled observations' present value and whether a TERMINATION knock-out ended the
        contract now (the terminal extra-shares leg then pays nothing).

    Raises:
        ValidationError: an observation after the maturity instant, which no settled value can include.
    """
    value, remaining, ended = settle_valuation_instant(product, spot, legs)
    if remaining:
        raise ValidationError(
            f"Accumulator observations after the maturity instant (t = {[float(leg[0]) for leg in remaining]}) "
            "cannot be settled at expiry"
        )
    return value, ended
