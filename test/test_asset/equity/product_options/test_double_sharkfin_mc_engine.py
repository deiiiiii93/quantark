from datetime import datetime

import pytest

from quantark.asset.equity.engine.mc import DoubleSharkfinOptionMCEngine
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option import DoubleSharkfinOption
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationFrequency, ObservationType, OptionType
from quantark.util.enum.engine_enums import EngineType, MonteCarloMethod
from quantark.util.exceptions import PricingError


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


def test_expiry_mc_matches_reasonable_payoff_range():
    product = DoubleSharkfinOption(
        strike=100.0,
        option_type=OptionType.CALL,
        lower_barrier=70.0,
        upper_barrier=130.0,
        maturity=1.0,
        participation_rate=0.8,
        knock_out_rebate=2.0,
        no_hit_rebate=0.5,
        observation_type=ObservationType.EXPIRY,
    )
    engine = DoubleSharkfinOptionMCEngine(
        params=MCParams(num_paths=4096, time_steps=16, seed=123)
    )

    price = engine.price(product, _pricing_env())

    assert 0.0 <= price <= 30.0
    assert engine.get_last_std_error() is not None
    assert engine.get_last_std_error() >= 0.0


def test_payoff_scales_by_contract_multiplier():
    env = _pricing_env()
    params = MCParams(num_paths=4096, time_steps=16, seed=77)
    base = DoubleSharkfinOption(
        strike=100.0,
        option_type=OptionType.PUT,
        lower_barrier=70.0,
        upper_barrier=130.0,
        maturity=1.0,
        observation_type=ObservationType.EXPIRY,
    )
    scaled = DoubleSharkfinOption(
        strike=100.0,
        option_type=OptionType.PUT,
        lower_barrier=70.0,
        upper_barrier=130.0,
        maturity=1.0,
        observation_type=ObservationType.EXPIRY,
        contract_multiplier=100.0,
    )

    assert DoubleSharkfinOptionMCEngine(params=params).price(scaled, env) == pytest.approx(
        100.0 * DoubleSharkfinOptionMCEngine(params=params).price(base, env),
        rel=1e-12,
        abs=1e-12,
    )


def test_discrete_daily_mc_runs_with_generated_schedule():
    product = DoubleSharkfinOption(
        strike=100.0,
        option_type=OptionType.CALL,
        lower_barrier=80.0,
        upper_barrier=120.0,
        maturity=0.25,
        knock_out_rebate=1.0,
        observation_type=ObservationType.DISCRETE,
        observation_frequency=ObservationFrequency.DAILY,
    )
    engine = DoubleSharkfinOptionMCEngine(
        params=MCParams(num_paths=2048, time_steps=16, seed=42)
    )

    price = engine.price(product, _pricing_env())

    assert price >= 0.0
    assert product.get_observation_times()


def test_continuous_pay_at_hit_is_worth_more_than_pay_at_expiry():
    env = _pricing_env(rate=0.05)
    params = MCParams(num_paths=4096, time_steps=64, seed=99)
    at_expiry = DoubleSharkfinOption(
        strike=100.0,
        option_type=OptionType.CALL,
        lower_barrier=80.0,
        upper_barrier=120.0,
        maturity=1.0,
        participation_rate=0.0,
        knock_out_rebate=10.0,
        observation_type=ObservationType.CONTINUOUS,
        pay_at_hit=False,
    )
    at_hit = DoubleSharkfinOption(
        strike=100.0,
        option_type=OptionType.CALL,
        lower_barrier=80.0,
        upper_barrier=120.0,
        maturity=1.0,
        participation_rate=0.0,
        knock_out_rebate=10.0,
        observation_type=ObservationType.CONTINUOUS,
        pay_at_hit=True,
    )

    assert DoubleSharkfinOptionMCEngine(params=params).price(at_hit, env) > (
        DoubleSharkfinOptionMCEngine(params=params).price(at_expiry, env)
    )


def test_two_level_enum_method_selection():
    engine = DoubleSharkfinOptionMCEngine(
        method=EngineType.MONTE_CARLO(MonteCarloMethod.QUASI)
    )

    assert engine.method == MonteCarloMethod.QUASI


def test_rejects_non_double_sharkfin_product():
    from quantark.asset.equity.product.option import EuropeanVanillaOption

    engine = DoubleSharkfinOptionMCEngine(params=MCParams(num_paths=128))
    with pytest.raises(PricingError, match="only supports DoubleSharkfinOption"):
        engine.price(
            EuropeanVanillaOption(
                strike=100.0,
                option_type=OptionType.CALL,
                maturity=1.0,
            ),
            _pricing_env(),
        )
