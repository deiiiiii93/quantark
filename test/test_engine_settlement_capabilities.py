"""Settlement capability registry and fail-closed guard tests."""

from datetime import datetime

import pytest

from quantark.asset.equity.engine.analytical import (
    DigitalOptionAnalyticalEngine,
)
from quantark.asset.equity.lifecycle import AutocallableLifecycleState
from quantark.asset.equity.engine.capabilities import (
    SettlementSupport,
    VolDynamicsType,
    get_engine_capability,
)
from quantark.asset.equity.product.option import (
    CashOrNothingDigitalOption,
)
from quantark.asset.equity.settlement import (
    SettlementConvention,
    SettlementLagUnit,
)
from quantark.execution.errors import CapabilityError
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.enum.engine_enums import EngineType


class _NonOptedDigitalEngine(DigitalOptionAnalyticalEngine):
    """Test sentinel for the shared fail-closed guard."""

    settlement_support = SettlementSupport.NONE
    supports_lifecycle_state = False


@pytest.mark.parametrize(
    ("dynamics", "engine_type", "supported"),
    [
        (VolDynamicsType.BSM, EngineType.MONTE_CARLO, True),
        (VolDynamicsType.BSM, EngineType.PDE, True),
        (VolDynamicsType.BSM, EngineType.QUADRATURE, True),
        (VolDynamicsType.LOCAL_VOL, EngineType.MONTE_CARLO, True),
        (VolDynamicsType.LOCAL_VOL, EngineType.PDE, True),
        (VolDynamicsType.LOCAL_VOL, EngineType.QUADRATURE, False),
        (VolDynamicsType.HESTON, EngineType.MONTE_CARLO, True),
        (VolDynamicsType.HESTON, EngineType.PDE, True),
        (VolDynamicsType.HESTON, EngineType.QUADRATURE, False),
        (VolDynamicsType.SLV, EngineType.MONTE_CARLO, True),
        (VolDynamicsType.SLV, EngineType.PDE, True),
        (VolDynamicsType.SLV, EngineType.QUADRATURE, False),
    ],
)
def test_existing_model_engine_matrix_keeps_support_and_declares_settlement(
    dynamics, engine_type, supported
):
    capability = get_engine_capability(dynamics, engine_type)

    assert capability.supported is supported
    assert isinstance(capability.settlement_support, SettlementSupport)


def test_non_opted_in_engine_rejects_delayed_settlement_before_pricing():
    product = CashOrNothingDigitalOption(
        strike=100.0,
        payout=10.0,
        option_type=OptionType.CALL,
        maturity=1.0,
        settlement_convention=SettlementConvention(
            lag=2 / 365.0,
            lag_unit=SettlementLagUnit.YEAR_FRACTION,
        ),
    )

    with pytest.raises(CapabilityError, match="settlement"):
        _NonOptedDigitalEngine().price(product, object())


def test_zero_lag_keeps_legacy_digital_price():
    env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.03),
        div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=datetime(2026, 7, 30),
    )
    legacy = CashOrNothingDigitalOption(
        strike=100.0,
        payout=10.0,
        option_type=OptionType.CALL,
        maturity=1.0,
    )
    zero_lag = CashOrNothingDigitalOption(
        strike=100.0,
        payout=10.0,
        option_type=OptionType.CALL,
        maturity=1.0,
        settlement_convention=SettlementConvention(),
    )
    engine = DigitalOptionAnalyticalEngine()

    assert engine.price(zero_lag, env) == engine.price(legacy, env)


def test_non_opted_in_engine_rejects_lifecycle_state_before_pricing():
    product = CashOrNothingDigitalOption(
        strike=100.0,
        payout=10.0,
        option_type=OptionType.CALL,
        maturity=1.0,
    )

    with pytest.raises(CapabilityError, match="lifecycle_state"):
        _NonOptedDigitalEngine().price(
            product,
            object(),
            lifecycle_state=AutocallableLifecycleState(),
        )
