"""
Unit tests for quad input adapter resolution.
"""

import pytest

from quantark.asset.equity.engine.quad.quad_adapters import (
    BarrierQuadInputAdapter,
    OneTouchQuadInputAdapter,
    resolve_quad_adapter,
)
from quantark.asset.equity.product.option import BarrierOption, EuropeanVanillaOption, OneTouchOption
from quantark.util.enum import (
    BarrierDirection,
    BarrierType,
    ObservationType,
    OptionType,
    TouchType,
)
from quantark.util.exceptions import PricingError


def test_resolve_adapter_barrier_option():
    option = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=110.0,
        barrier_type=BarrierType.UP_OUT,
        maturity=1.0,
        observation_type=ObservationType.EXPIRY,
    )

    adapter = resolve_quad_adapter(option)

    assert isinstance(adapter, BarrierQuadInputAdapter)


def test_resolve_adapter_one_touch_option():
    option = OneTouchOption(
        barrier=110.0,
        barrier_direction=BarrierDirection.UP,
        maturity=1.0,
        rebate=5.0,
        payment_at_hit=False,
        touch_type=TouchType.ONE_TOUCH,
        observation_type=ObservationType.EXPIRY,
    )

    adapter = resolve_quad_adapter(option)

    assert isinstance(adapter, OneTouchQuadInputAdapter)


def test_resolve_adapter_unsupported_product():
    option = EuropeanVanillaOption(
        strike=100.0,
        option_type=OptionType.CALL,
        maturity=1.0,
    )

    with pytest.raises(PricingError):
        resolve_quad_adapter(option)
