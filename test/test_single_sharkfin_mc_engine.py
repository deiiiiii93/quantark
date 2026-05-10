from datetime import datetime

import pytest

from asset.equity.engine.analytical import SingleSharkfinOptionAnalyticalEngine
from asset.equity.engine.mc import SingleSharkfinOptionMCEngine
from asset.equity.param import MCParams
from asset.equity.product.option import (
    EuropeanVanillaOption,
    ObservationRecord,
    ObservationSchedule,
    SingleSharkfinOption,
)
from param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from priceenv import PricingEnvironment
from util.enum import (
    ObservationAggregation,
    ObservationFrequency,
    ObservationType,
    OptionType,
)
from util.enum.engine_enums import MonteCarloMethod
from util.exceptions import PricingError, ValidationError


def _pricing_env(
    spot: float = 100.0,
    rate: float = 0.03,
    div: float = 0.01,
    vol: float = 0.2,
) -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        rate_curve=FlatRateCurve(rate=rate),
        vol_surface=FlatVolSurface(volatility=vol),
        div_yield=ContinuousDividendYield(div_yield=div),
        valuation_date=datetime(2024, 1, 1),
    )


def test_expiry_mc_matches_analytical_within_sampling_error():
    env = _pricing_env(vol=0.18)
    product = SingleSharkfinOption(
        strike=95.0,
        option_type=OptionType.CALL,
        barrier=120.0,
        maturity=1.0,
        participation_rate=0.7,
        knock_out_rebate=2.0,
        no_hit_rebate=0.5,
        observation_type=ObservationType.EXPIRY,
    )
    mc_engine = SingleSharkfinOptionMCEngine(
        params=MCParams(num_paths=32768, time_steps=1, seed=7),
        method=MonteCarloMethod.QUASI,
    )
    analytical = SingleSharkfinOptionAnalyticalEngine()

    mc_price = mc_engine.price(product, env)
    analytical_price = analytical.price(product, env)

    assert mc_engine.get_last_std_error() is not None
    assert mc_price == pytest.approx(analytical_price, abs=0.08)


def test_discrete_daily_mc_prices_and_reports_std_error():
    env = _pricing_env(vol=0.2)
    product = SingleSharkfinOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=120.0,
        maturity=1.0,
        participation_rate=1.0,
        knock_out_rebate=1.0,
        no_hit_rebate=0.25,
        observation_type=ObservationType.DISCRETE,
        observation_frequency=ObservationFrequency.DAILY,
    )
    engine = SingleSharkfinOptionMCEngine(
        params=MCParams(num_paths=4096, time_steps=252, seed=11),
        method=MonteCarloMethod.QUASI,
    )

    price = engine.price(product, env)

    assert price > 0.0
    assert engine.get_last_std_error() is not None
    assert engine.get_last_std_error() >= 0.0


def test_contract_multiplier_scales_price_and_std_error():
    env = _pricing_env()
    params = MCParams(num_paths=4096, time_steps=32, seed=42)
    base = SingleSharkfinOption(
        strike=100.0,
        option_type=OptionType.PUT,
        barrier=80.0,
        maturity=1.0,
        participation_rate=0.8,
        knock_out_rebate=1.0,
        no_hit_rebate=0.25,
        observation_type=ObservationType.CONTINUOUS,
    )
    scaled = SingleSharkfinOption(
        strike=100.0,
        option_type=OptionType.PUT,
        barrier=80.0,
        maturity=1.0,
        participation_rate=0.8,
        knock_out_rebate=1.0,
        no_hit_rebate=0.25,
        observation_type=ObservationType.CONTINUOUS,
        contract_multiplier=100.0,
    )
    base_engine = SingleSharkfinOptionMCEngine(params=params)
    scaled_engine = SingleSharkfinOptionMCEngine(params=params)

    base_price = base_engine.price(base, env)
    scaled_price = scaled_engine.price(scaled, env)

    assert scaled_price == pytest.approx(100.0 * base_price, rel=1e-12, abs=1e-12)
    assert scaled_engine.get_last_std_error() == pytest.approx(
        100.0 * base_engine.get_last_std_error(), rel=1e-12, abs=1e-12
    )


def test_pay_at_hit_rebate_exceeds_pay_at_expiry_rebate_for_positive_rates():
    env = _pricing_env(rate=0.05)
    params = MCParams(num_paths=8192, time_steps=64, seed=5)
    at_expiry = SingleSharkfinOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=120.0,
        maturity=1.0,
        participation_rate=0.0,
        knock_out_rebate=10.0,
        no_hit_rebate=0.0,
        observation_type=ObservationType.CONTINUOUS,
        pay_at_hit=False,
    )
    at_hit = SingleSharkfinOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=120.0,
        maturity=1.0,
        participation_rate=0.0,
        knock_out_rebate=10.0,
        no_hit_rebate=0.0,
        observation_type=ObservationType.CONTINUOUS,
        pay_at_hit=True,
    )

    expiry_engine = SingleSharkfinOptionMCEngine(params=params)
    hit_engine = SingleSharkfinOptionMCEngine(params=params)

    assert hit_engine.price(at_hit, env) > expiry_engine.price(at_expiry, env)


def test_rejects_non_sharkfin_product():
    engine = SingleSharkfinOptionMCEngine()
    with pytest.raises(PricingError, match="only supports SingleSharkfinOption"):
        engine.price(
            EuropeanVanillaOption(
                strike=100.0,
                option_type=OptionType.CALL,
                maturity=1.0,
            ),
            _pricing_env(),
        )


def test_rejects_non_first_hit_discrete_schedule():
    product = SingleSharkfinOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=120.0,
        maturity=1.0,
        observation_type=ObservationType.DISCRETE,
        observation_schedule=ObservationSchedule(
            records=[
                ObservationRecord(observation_time=0.5),
                ObservationRecord(observation_time=1.0),
            ],
            aggregation_mode=ObservationAggregation.ACCUMULATE,
        ),
    )
    engine = SingleSharkfinOptionMCEngine(
        params=MCParams(num_paths=128, time_steps=2, seed=1)
    )

    with pytest.raises(ValidationError, match="STOP_FIRST_HIT"):
        engine.price(product, _pricing_env())
