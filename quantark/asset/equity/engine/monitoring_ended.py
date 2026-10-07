"""A discretely monitored contract whose monitoring has ended (spec 2026-10-07 theta boundary, 3.3).

``time_shift`` leaves such a product with an empty schedule marked ``monitoring_ended``: no barrier can be hit any
more, so what remains is a function of S_T alone, priced as a static replication on the analytic engines (the 3C
``ExpiryReplication``: European vanillas and cash digitals on the product's own expiry, settlement and multiplier).
A fixed cash amount R is R x (cash digital call + cash digital put at one strike), exact since P(S_T = strike) = 0.
Engines expose the legs through ``BaseEngine.static_replication``, so ``GreeksCalculator`` takes every greek as the
weighted sum of its own greeks of the legs.
"""
from __future__ import annotations

from quantark.util.enum import ObservationType, OptionType

from .pde.core.adapters.expiry_replication import ExpiryReplication, Leg

__all__ = ["monitoring_ended", "remaining_legs", "remaining_replication"]

_C, _P = OptionType.CALL, OptionType.PUT


def monitoring_ended(product) -> bool:
    """Whether ``product`` is discretely monitored and its schedule has no observation left."""
    schedule = getattr(product, "observation_schedule", None)
    return (getattr(product, "observation_type", None) == ObservationType.DISCRETE and schedule is not None
            and schedule.monitoring_ended and not schedule.records)


def _cash(amount: float, strike: float) -> list:
    return [] if amount == 0.0 else [Leg(amount, "digital", strike, _C), Leg(amount, "digital", strike, _P)]


def _spread(weight: float, strike: float, cap: float, kind: OptionType) -> list:
    """``weight`` x the payoff capped at ``cap``: C(K) - C(cap) for a call, P(K) - P(cap) for a put."""
    return [Leg(weight, "vanilla", strike, kind), Leg(-weight, "vanilla", cap, kind)]


def remaining_legs(product, knocked_in: bool = False) -> list:
    """The remaining contract once monitoring has ended (``knocked_in``: a knock-in already triggered)."""
    from quantark.asset.equity.product.option import (BarrierOption, DoubleBarrierOption, DoubleOneTouchOption,
                                                      DoubleSharkfinOption, OneTouchOption, SingleSharkfinOption)

    if isinstance(product, BarrierOption):
        if product.is_knock_out or knocked_in:
            return [Leg(product.participation_rate, "vanilla", product.strike, product.option_type)]
        return _cash(product.rebate, product.strike)
    if isinstance(product, DoubleBarrierOption):
        if product.is_knock_out or knocked_in:
            return [Leg(1.0, "vanilla", product.strike, product.option_type)]
        return _cash(product.rebate, product.strike)
    if isinstance(product, OneTouchOption):
        return [] if product.is_one_touch else _cash(product.rebate, product.barrier)
    if isinstance(product, DoubleOneTouchOption):
        return [] if product.is_double_one_touch else _cash(product.rebate, product.upper_barrier)
    if isinstance(product, SingleSharkfinOption):
        kind = _C if product.is_call() else _P
        return (_spread(product.participation_rate, product.strike, product.barrier, kind)
                + _cash(product.no_hit_rebate, product.strike))
    if isinstance(product, DoubleSharkfinOption):
        kind = _C if product.is_call() else _P
        cap = product.upper_barrier if product.is_call() else product.lower_barrier
        return (_spread(product.participation_rate, product.strike, cap, kind)
                + _cash(product.no_hit_rebate, product.strike))
    raise TypeError(f"no remaining contract defined for {type(product).__name__}")


def _multiplier(product) -> float:
    """The legs' contract multiplier: the touch engines carry none (their products have no multiplier in price)."""
    from quantark.asset.equity.product.option import DoubleOneTouchOption, OneTouchOption

    if isinstance(product, (OneTouchOption, DoubleOneTouchOption)):
        return 1.0
    return float(product.contract_multiplier)


def remaining_replication(product, knocked_in: bool = False) -> ExpiryReplication:
    """The remaining contract's legs as an ``ExpiryReplication`` on the product's expiry and settlement."""
    return ExpiryReplication(remaining_legs(product, knocked_in), product, _multiplier(product))
