from datetime import datetime

import pytest

from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.engine.quad import PhoenixQuadEngine, SnowballQuadEngine
from quantark.asset.equity.param import QuadParams
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.snowball_config import BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
    TermStructureDividendYield,
    TermStructureVolSurface,
)
from quantark.priceenv import PricingEnvironment
from quantark.rfq import (
    RFQEngineSpec,
    RFQInputMode,
    RFQObjectInput,
    RFQRequest,
    RFQTarget,
    RFQTargetLabel,
    RFQTermsheetInput,
    RFQUnknownSpec,
    quote_rfq,
)
from quantark.util.enum import CouponPayType, ObservationType, OptionType
from quantark.util.exceptions import PricingError, ValidationError


def create_vanilla_env(
    *,
    spot: float = 100.0,
    vol: float = 0.20,
    rate: float = 0.05,
    div_yield: float = 0.02,
) -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div_yield),
        valuation_date=datetime(2024, 1, 1),
    )


def create_snowball_inputs(ko_rate: float = 0.15) -> tuple[SnowballOption, PricingEnvironment]:
    barrier_config = BarrierConfig(
        ko_barrier=103.0,
        ko_rate=ko_rate,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
        ki_barrier=75.0,
        ki_observation_type=ObservationType.CONTINUOUS,
    )
    snowball = SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=barrier_config,
        payoff_config=PayoffConfig(rebate_rate=0.0, include_principal=True),
        maturity=1.0,
        contract_multiplier=1.0,
    )
    env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.03),
        div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=datetime(2024, 1, 1),
    )
    return snowball, env


def create_phoenix_inputs(coupon_rate: float = 0.02) -> tuple[PhoenixOption, PricingEnvironment]:
    barrier_config = BarrierConfig(
        ko_barrier=1.0e9,
        ko_rate=0.0,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[0.5, 1.0],
        ki_barrier=None,
    )
    coupon_config = CouponBarrierConfig(
        coupon_barrier=[85.0, 85.0],
        coupon_rate=coupon_rate,
        coupon_pay_type=CouponPayType.INSTANT,
        memory_coupon=False,
    )
    payoff_config = PayoffConfig(rebate_rate=0.0, include_principal=True)
    phoenix = PhoenixOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=barrier_config,
        coupon_config=coupon_config,
        payoff_config=payoff_config,
        maturity=1.0,
        contract_multiplier=1.0,
    )
    env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.03),
        div_yield=ContinuousDividendYield(div_yield=0.00),
        valuation_date=datetime(2024, 1, 1),
    )
    return phoenix, env


def test_object_rfq_solves_fair_q_for_vanilla():
    product = EuropeanVanillaOption(
        strike=100.0,
        option_type=OptionType.CALL,
        maturity=1.0,
    )
    env = create_vanilla_env(div_yield=0.02)
    engine = BlackScholesEngine()
    target_price = engine.price(product, env)

    request = RFQRequest(
        input_mode=RFQInputMode.OBJECT,
        object_input=RFQObjectInput(product=product, pricing_env=env, engine=engine),
        unknown=RFQUnknownSpec(
            field_path="pricing_env.div_yield.div_yield",
            lower_bound=0.0,
            upper_bound=0.10,
        ),
        target=RFQTarget(label=RFQTargetLabel.PRICE, value=target_price),
    )

    quote = quote_rfq(request)

    assert quote.status.value == "success"
    assert quote.solved_value == pytest.approx(0.02, abs=1e-7)
    assert quote.achieved_price == pytest.approx(target_price, abs=1e-8)
    assert quote.residual == pytest.approx(0.0, abs=1e-8)
    assert quote.request_summary["product_type"] == "EuropeanVanillaOption"


def test_object_rfq_solves_fair_flat_vol_for_vanilla():
    product = EuropeanVanillaOption(
        strike=100.0,
        option_type=OptionType.CALL,
        maturity=1.0,
    )
    env = create_vanilla_env(vol=0.24)
    engine = BlackScholesEngine()
    target_price = engine.price(product, env)

    request = RFQRequest(
        input_mode=RFQInputMode.OBJECT,
        object_input=RFQObjectInput(product=product, pricing_env=env, engine=engine),
        unknown=RFQUnknownSpec(
            field_path="pricing_env.vol_surface.volatility",
            lower_bound=0.05,
            upper_bound=0.60,
        ),
        target=RFQTarget(label=RFQTargetLabel.PREMIUM, value=target_price),
    )

    quote = quote_rfq(request)

    assert quote.solved_value == pytest.approx(0.24, abs=1e-7)
    assert quote.achieved_price == pytest.approx(target_price, abs=1e-8)


def test_termsheet_rfq_solves_snowball_fair_ko_rate():
    snowball, env = create_snowball_inputs(ko_rate=0.15)
    engine = SnowballQuadEngine(params=QuadParams(grid_points=201))
    target_price = engine.price(snowball, env)

    request = RFQRequest(
        input_mode=RFQInputMode.TERMSHEET,
        termsheet_input=RFQTermsheetInput(
            product_type="snowball_option",
            product_kwargs={
                "initial_price": 100.0,
                "strike": 100.0,
                "barrier_config": {
                    "ko_barrier": 103.0,
                    "ko_rate": 0.10,
                    "ko_observation_type": ObservationType.DISCRETE,
                    "ko_observation_dates": [0.25, 0.5, 0.75, 1.0],
                    "ki_barrier": 75.0,
                    "ki_observation_type": ObservationType.CONTINUOUS,
                },
                "payoff_config": {
                    "rebate_rate": 0.0,
                    "include_principal": True,
                },
                "maturity": 1.0,
                "contract_multiplier": 1.0,
            },
            market_kwargs={
                "valuation_date": datetime(2024, 1, 1),
                "spot": 100.0,
                "volatility": 0.20,
                "rate": 0.03,
                "q": 0.01,
            },
            engine_spec=RFQEngineSpec(
                engine_name="snowball_quad_engine",
                params_type="quad_params",
                params_kwargs={"grid_points": 201},
            ),
        ),
        unknown=RFQUnknownSpec(
            field_path="barrier_config.ko_rate",
            lower_bound=0.05,
            upper_bound=0.30,
        ),
        target=RFQTarget(label=RFQTargetLabel.REOFFER, value=target_price),
    )

    quote = quote_rfq(request)

    assert quote.solved_value == pytest.approx(0.15, abs=1e-6)
    assert quote.achieved_price == pytest.approx(target_price, abs=1e-6)


def test_termsheet_rfq_solves_phoenix_fair_coupon_rate():
    phoenix, env = create_phoenix_inputs(coupon_rate=0.02)
    engine = PhoenixQuadEngine(params=QuadParams(grid_points=201))
    target_price = engine.price(phoenix, env)

    request = RFQRequest(
        input_mode=RFQInputMode.TERMSHEET,
        termsheet_input=RFQTermsheetInput(
            product_type="phoenix_option",
            product_kwargs={
                "initial_price": 100.0,
                "strike": 100.0,
                "barrier_config": {
                    "ko_barrier": 1.0e9,
                    "ko_rate": 0.0,
                    "ko_observation_type": ObservationType.DISCRETE,
                    "ko_observation_dates": [0.5, 1.0],
                    "ki_barrier": None,
                },
                "coupon_config": {
                    "coupon_barrier": [85.0, 85.0],
                    "coupon_rate": 0.01,
                    "coupon_pay_type": CouponPayType.INSTANT,
                    "memory_coupon": False,
                },
                "payoff_config": {
                    "rebate_rate": 0.0,
                    "include_principal": True,
                },
                "maturity": 1.0,
                "contract_multiplier": 1.0,
            },
            market_kwargs={
                "valuation_date": datetime(2024, 1, 1),
                "spot": 100.0,
                "volatility": 0.20,
                "rate": 0.03,
                "q": 0.00,
            },
            engine_spec=RFQEngineSpec(
                engine_name="phoenix_quad_engine",
                params_type="quad_params",
                params_kwargs={"grid_points": 201},
            ),
        ),
        unknown=RFQUnknownSpec(
            field_path="coupon_config.coupon_rate",
            lower_bound=0.0,
            upper_bound=0.10,
        ),
        target=RFQTarget(label=RFQTargetLabel.PRICE, value=target_price),
    )

    quote = quote_rfq(request)

    assert quote.solved_value == pytest.approx(0.02, abs=1e-6)
    assert quote.achieved_price == pytest.approx(target_price, abs=1e-6)


def test_missing_explicit_target_raises_validation_error():
    product = EuropeanVanillaOption(
        strike=100.0,
        option_type=OptionType.CALL,
        maturity=1.0,
    )
    env = create_vanilla_env()
    engine = BlackScholesEngine()

    with pytest.raises(ValidationError):
        RFQRequest(
            input_mode=RFQInputMode.OBJECT,
            object_input=RFQObjectInput(
                product=product, pricing_env=env, engine=engine
            ),
            unknown=RFQUnknownSpec(field_path="q", lower_bound=0.0, upper_bound=0.1),
            target=None,  # type: ignore[arg-type]
        )


def test_unsupported_product_type_raises_validation_error():
    request = RFQRequest(
        input_mode=RFQInputMode.TERMSHEET,
        termsheet_input=RFQTermsheetInput(
            product_type="not_a_product",
            product_kwargs={},
            market_kwargs={
                "valuation_date": datetime(2024, 1, 1),
                "rate": 0.03,
            },
            engine_spec=RFQEngineSpec(engine_name="black_scholes_engine"),
        ),
        unknown=RFQUnknownSpec(field_path="q", lower_bound=0.0, upper_bound=0.1),
        target=RFQTarget(label=RFQTargetLabel.PRICE, value=1.0),
    )

    with pytest.raises(ValidationError, match="Unsupported RFQ product_type"):
        quote_rfq(request)


def test_unsupported_unknown_field_raises_validation_error():
    product = EuropeanVanillaOption(
        strike=100.0,
        option_type=OptionType.CALL,
        maturity=1.0,
    )
    env = create_vanilla_env()
    engine = BlackScholesEngine()

    request = RFQRequest(
        input_mode=RFQInputMode.OBJECT,
        object_input=RFQObjectInput(product=product, pricing_env=env, engine=engine),
        unknown=RFQUnknownSpec(
            field_path="not.supported",
            lower_bound=0.0,
            upper_bound=1.0,
        ),
        target=RFQTarget(label=RFQTargetLabel.PRICE, value=1.0),
    )

    with pytest.raises(ValidationError, match="Unsupported RFQ field_path"):
        quote_rfq(request)


def test_invalid_bounds_raise_validation_error():
    with pytest.raises(ValidationError, match="lower_bound"):
        RFQUnknownSpec(field_path="q", lower_bound=0.1, upper_bound=0.1)


def test_term_structure_dividend_unknown_rejected():
    product = EuropeanVanillaOption(
        strike=100.0,
        option_type=OptionType.CALL,
        maturity=1.0,
    )
    env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=TermStructureDividendYield(times=[0.5, 1.0], yields=[0.01, 0.02]),
        valuation_date=datetime(2024, 1, 1),
    )
    engine = BlackScholesEngine()
    request = RFQRequest(
        input_mode=RFQInputMode.OBJECT,
        object_input=RFQObjectInput(product=product, pricing_env=env, engine=engine),
        unknown=RFQUnknownSpec(field_path="q", lower_bound=0.0, upper_bound=0.1),
        target=RFQTarget(label=RFQTargetLabel.PRICE, value=5.0),
    )

    with pytest.raises(ValidationError, match="ContinuousDividendYield"):
        quote_rfq(request)


def test_term_structure_vol_unknown_rejected():
    product = EuropeanVanillaOption(
        strike=100.0,
        option_type=OptionType.CALL,
        maturity=1.0,
    )
    env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=TermStructureVolSurface(times=[0.5, 1.0], vols=[0.2, 0.25]),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.02),
        valuation_date=datetime(2024, 1, 1),
    )
    engine = BlackScholesEngine()
    request = RFQRequest(
        input_mode=RFQInputMode.OBJECT,
        object_input=RFQObjectInput(product=product, pricing_env=env, engine=engine),
        unknown=RFQUnknownSpec(field_path="vol", lower_bound=0.05, upper_bound=0.5),
        target=RFQTarget(label=RFQTargetLabel.PRICE, value=5.0),
    )

    with pytest.raises(ValidationError, match="FlatVolSurface"):
        quote_rfq(request)


def test_target_not_bracketed_raises_pricing_error():
    product = EuropeanVanillaOption(
        strike=100.0,
        option_type=OptionType.CALL,
        maturity=1.0,
    )
    env = create_vanilla_env(div_yield=0.02)
    engine = BlackScholesEngine()

    request = RFQRequest(
        input_mode=RFQInputMode.OBJECT,
        object_input=RFQObjectInput(product=product, pricing_env=env, engine=engine),
        unknown=RFQUnknownSpec(field_path="q", lower_bound=0.0, upper_bound=0.01),
        target=RFQTarget(label=RFQTargetLabel.PRICE, value=engine.price(product, env)),
    )

    with pytest.raises(PricingError, match="not bracketed"):
        quote_rfq(request)


def test_solver_non_convergence_raises_pricing_error():
    product = EuropeanVanillaOption(
        strike=100.0,
        option_type=OptionType.CALL,
        maturity=1.0,
    )
    env = create_vanilla_env(div_yield=0.02)
    engine = BlackScholesEngine()
    target_price = engine.price(product, env)

    request = RFQRequest(
        input_mode=RFQInputMode.OBJECT,
        object_input=RFQObjectInput(product=product, pricing_env=env, engine=engine),
        unknown=RFQUnknownSpec(field_path="q", lower_bound=0.0, upper_bound=0.1),
        target=RFQTarget(label=RFQTargetLabel.PRICE, value=target_price),
    )

    with pytest.raises(PricingError, match="did not converge"):
        quote_rfq(request, max_iterations=1, price_tolerance=1e-30, value_tolerance=0.0)


def test_quote_serialization_contains_expected_fields():
    product = EuropeanVanillaOption(
        strike=100.0,
        option_type=OptionType.CALL,
        maturity=1.0,
    )
    env = create_vanilla_env(div_yield=0.02)
    engine = BlackScholesEngine()
    target_price = engine.price(product, env)

    quote = quote_rfq(
        RFQRequest(
            input_mode=RFQInputMode.OBJECT,
            object_input=RFQObjectInput(
                product=product, pricing_env=env, engine=engine
            ),
            unknown=RFQUnknownSpec(field_path="q", lower_bound=0.0, upper_bound=0.1),
            target=RFQTarget(label=RFQTargetLabel.PRICE, value=target_price),
        )
    )
    payload = quote.to_dict()

    assert payload["quote_id"].startswith("rfq-")
    assert payload["status"] == "success"
    assert payload["solved_value"] == pytest.approx(0.02, abs=1e-7)
    assert payload["residual"] == pytest.approx(0.0, abs=1e-8)
    assert payload["engine_summary"]["engine_class"] == "BlackScholesEngine"
    assert payload["request_summary"]["product_type"] == "EuropeanVanillaOption"
