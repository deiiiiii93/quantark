"""Contract, lifecycle, refinement and API checks for the opt-in engine."""
from copy import deepcopy
from datetime import datetime
import math
import numpy as np
import pytest
from scipy.stats import norm

from quantark.asset.equity.engine.quad import (
    SnowballQuadEngineV2,
    PhoenixQuadEngineV2,
    KOResetSnowballQuadEngineV2,
)
from quantark.asset.equity.param import QuadV2Params
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.snowball_config import (
    BarrierConfig,
    PayoffConfig,
    AccrualConfig,
    AirbagConfig,
)
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option import create_ko_reset_snowball
from quantark.asset.equity.lifecycle import AutocallableLifecycleState
from quantark.param import (
    SpotQuote,
    FlatRateCurve,
    FlatVolSurface,
    ContinuousDividendYield,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import (
    ObservationType,
    ProtectionType,
    CouponPayType,
    PostKOScheduleMode,
)
from quantark.util.exceptions import NumericalError, ValidationError


def environment(spot=100.0, vol=0.2, r=0.03, q=0.01):
    surface = FlatVolSurface(max(vol, 1e-8))
    surface.volatility = vol
    return PricingEnvironment(
        spot_quote=SpotQuote(spot),
        vol_surface=surface,
        rate_curve=FlatRateCurve(r),
        div_yield=ContinuousDividendYield(q),
        valuation_date=datetime(2026, 9, 11),
    )


def snowball(
    times=(0.5, 1.0),
    ki_times=None,
    ko=103.0,
    ki=75.0,
    reverse=False,
    continuous=False,
    **kwargs,
):
    return SnowballOption(
        initial_price=100.0,
        strike=100.0,
        contract_multiplier=1.0,
        maturity=times[-1],
        is_reverse=reverse,
        barrier_config=BarrierConfig(
            ko_barrier=ko,
            ko_rate=0.12,
            ko_observation_dates=list(times),
            ki_barrier=ki,
            ki_continuous=continuous,
            ki_observation_type=ObservationType.CONTINUOUS
            if continuous
            else ObservationType.DISCRETE,
            ki_observation_dates=list(ki_times if ki_times is not None else times),
        ),
        payoff_config=kwargs.pop(
            "payoff_config", PayoffConfig(rebate_rate=0.12, include_principal=False)
        ),
        **kwargs,
    )


def test_single_observation_against_independent_truncated_lognormal():
    p = snowball(times=(1.0,))
    e = environment()
    got = SnowballQuadEngineV2().calculate_point_greeks(p, e)
    sd = 0.2
    d2 = (math.log(100 / 75) + (0.03 - 0.01 - 0.5 * sd**2)) / sd
    expected = math.exp(-0.03) * (12 - 112 * norm.cdf(-d2)) + 100 * math.exp(
        -0.01
    ) * norm.cdf(-d2 - sd)
    assert got["price"] == pytest.approx(expected, abs=1e-11)


@pytest.mark.parametrize("backend", ["direct", "fft"])
def test_monthly_reference_and_component_reconciliation(backend):
    p = snowball(times=tuple(i / 12 for i in range(1, 13)))
    engine = SnowballQuadEngineV2(QuadV2Params(backend=backend))
    got = engine.calculate_point_greeks(p, environment())
    # Independent frozen Gaussian panel/reference audit, with an 8-scramble
    # exact-GBM QMC interval in example/gaussian_quad_comparison.
    assert got["price"] == pytest.approx(1.9103640029206575, abs=1e-8)
    assert got["delta"] == pytest.approx(0.007568552357756336, abs=1e-8)
    components = engine.price_components(p, environment())
    assert abs(components["reconciliation_error"]) < 1e-11
    assert components["ko"] > 0 and components["terminal"] < 0


def test_drift_domain_failure_is_fixed():
    p = snowball(times=tuple(i / 12 for i in range(1, 13)))
    result = SnowballQuadEngineV2().price(p, environment(vol=0.005, q=0.40))
    # Forward is so far below strike/KI that the omitted probabilities are
    # negligible; the exact limiting discounted stock-minus-strike is a control.
    exact = 100 * math.exp(-0.40) - 100 * math.exp(-0.03)
    assert result == pytest.approx(exact, abs=2e-8)


def test_short_first_ko_does_not_underresolve_or_expand_entire_grid():
    p = snowball(times=(0.0001, 0.5, 1.0), ki_times=(1.0,))
    engine = SnowballQuadEngineV2()
    result = engine.calculate_point_greeks(p, environment(spot=103.001))
    assert result["price"] == pytest.approx(3.2277326782, abs=1e-8)
    assert engine._prepared.diagnostics["nodes"] < 4000
    context = engine._prepared
    s, b = 103.001, 1e-5
    prices = context.evaluate([s * (1 - b), s, s * (1 + b)])["price"]
    assert (prices[2] - prices[0]) / (2 * s * b) == pytest.approx(
        result["delta"], rel=5e-5
    )


def test_curve_and_scalar_use_same_frozen_evaluator():
    p = snowball()
    env = environment()
    engine = SnowballQuadEngineV2()
    spots = np.linspace(70, 110, 21)
    curve = engine.calculate_spot_greeks_curve(p, env, spots)
    context = engine._prepared
    for row in curve:
        env.spot_quote.spot = row["spot"]
        scalar = engine.calculate_point_greeks(p, env)
        assert engine._prepared is context
        for name in ("price", "delta", "gamma"):
            assert scalar[name] == pytest.approx(row[name], abs=2e-11)
    old = context.evaluate([100.0])["price"][0]
    env.rate_curve = FlatRateCurve(0.08)
    new = engine.price(p, env)
    assert engine._prepared is not context
    assert context.evaluate([100.0])["price"][0] == old
    with pytest.raises(ValidationError, match="outside prepared"):
        context.evaluate([140.0])


def test_point_greeks_close_matched_bumps_and_desk_bumps_remain_separate():
    p = snowball()
    env = environment(spot=76.0)
    engine = SnowballQuadEngineV2()
    point = engine.calculate_point_greeks(p, env)
    context = engine._prepared
    s, b = env.spot, 1e-4
    values = context.evaluate([s * (1 - b), s, s * (1 + b)])["price"]
    assert (values[2] - values[0]) / (2 * s * b) == pytest.approx(
        point["delta"], abs=1e-6
    )
    assert (values[2] - 2 * values[1] + values[0]) / (s * b) ** 2 == pytest.approx(
        point["gamma"], abs=2e-6
    )
    desk = engine.calculate_greeks(p, env)
    values = context.evaluate([s * 0.99, s, s * 1.01])["price"]
    assert desk["delta"] == pytest.approx(
        (values[2] - values[0]) / (s * 0.02), abs=1e-12
    )


def test_terminal_cap_and_airbag_multiple_breaks():
    p = snowball(
        times=(1.0,),
        ko=1e6,
        payoff_config=PayoffConfig(
            rebate_rate=0.12,
            include_principal=False,
            protection_type=ProtectionType.PARTIAL,
            protection_rate=0.2,
        ),
        airbag_config=AirbagConfig(
            airbag_barrier=82.0, airbag_strike=110.0, airbag_participation_rate=1.5
        ),
    )
    p._otc_lifecycle_knocked_in = True
    e = environment(spot=82.0)
    result = SnowballQuadEngineV2().price(p, e)
    from scipy.integrate import quad

    breaks = [0.0, 75.0, 82.0, 100.0, 110.0, np.inf]

    def f(s):
        density = math.exp(-0.5 * (math.log(s / 82) / 0.2) ** 2) / (
            s * 0.2 * math.sqrt(2 * math.pi)
        )
        return p.get_maturity_payoff_v1(s, e) * density * math.exp(-0.03)

    independent = sum(
        quad(f, a, b, epsabs=1e-10)[0] for a, b in zip(breaks[:-1], breaks[1:])
    )
    assert result == pytest.approx(independent, abs=2e-8)


@pytest.mark.parametrize("reverse", [False, True])
def test_deterministic_ko_and_ki_equality(reverse):
    p = snowball(
        times=(0.5, 1.0),
        ki=120.0 if reverse else 80.0,
        ko=97.0 if reverse else 103.0,
        reverse=reverse,
    )
    s = 97.0 if reverse else 103.0
    engine = SnowballQuadEngineV2()
    assert engine.price(p, environment(spot=s, vol=0.0, r=0.0, q=0.0)) == pytest.approx(
        6.0, abs=1e-12
    )
    p = snowball(
        times=(1.0,),
        ko=50.0 if reverse else 150.0,
        ki=120.0 if reverse else 80.0,
        reverse=reverse,
    )
    s = 120.0 if reverse else 80.0
    result = engine.calculate_point_greeks(
        p, environment(spot=s, vol=0.0, r=0.0, q=0.0)
    )
    assert result["price"] == pytest.approx(-20.0, abs=1e-10)
    assert np.isnan(result["delta"])


def test_near_coincident_dates_fail_before_large_allocation():
    p = snowball(times=(0.5, 1.0), ki_times=(0.5 + 1e-14, 1.0))
    with pytest.raises(NumericalError, match="resource limit"):
        SnowballQuadEngineV2(QuadV2Params(max_nodes=20_000)).price(p, environment())
    # User-declared clock equivalence is explicit and preserves both payloads.
    result = SnowballQuadEngineV2(QuadV2Params(event_time_tolerance=1e-12)).price(
        p, environment()
    )
    assert result == pytest.approx(
        SnowballQuadEngineV2().price(snowball(), environment()), abs=1e-10
    )


def test_lifecycle_knocked_in_and_unknown_subclass():
    p = snowball(times=(1.0,), ko=1e6)
    e = environment()
    state = AutocallableLifecycleState(knocked_in=True)
    v = 0.2
    d1 = (0.03 - 0.01 + 0.5 * v * v) / v
    expected = 100 * math.exp(-0.01) * norm.cdf(-d1) - 100 * math.exp(-0.03) * norm.cdf(
        -d1 + v
    )
    assert SnowballQuadEngineV2().price(p, e, lifecycle_state=state) == pytest.approx(
        expected, abs=1e-11
    )

    class Unknown(SnowballOption):
        pass

    p.__class__ = Unknown
    with pytest.raises(ValidationError):
        SnowballQuadEngineV2().price(p, e)


@pytest.mark.parametrize("memory", [False, True])
@pytest.mark.parametrize("pay_type", [CouponPayType.INSTANT, CouponPayType.EXPIRY])
def test_phoenix_deterministic_coupon_memory(memory, pay_type):
    p = PhoenixOption(
        initial_price=100.0,
        strike=100.0,
        contract_multiplier=1.0,
        maturity=1.0,
        barrier_config=BarrierConfig(
            ko_barrier=200.0,
            ko_rate=0.0,
            ko_observation_dates=[0.5, 1.0],
            ki_barrier=None,
        ),
        coupon_config=CouponBarrierConfig(
            coupon_barrier=[110.0, 90.0],
            coupon_rate=0.02,
            memory_coupon=memory,
            coupon_pay_type=pay_type,
        ),
        payoff_config=PayoffConfig(rebate_rate=0.0, include_principal=True),
    )
    result = PhoenixQuadEngineV2().price_components(
        p, environment(vol=0.0, r=0.0, q=0.0)
    )
    assert result["price"] == pytest.approx(102.0 if memory else 101.0, abs=1e-11)
    assert result["coupon"] == pytest.approx(2.0 if memory else 1.0, abs=1e-11)


def test_reset_contract_ends_on_pre_maturity_when_not_ki():
    p = create_ko_reset_snowball(
        initial_price=100.0,
        strike=100.0,
        maturity_pre=1.0,
        maturity_post=2.0,
        post_ko_mode=PostKOScheduleMode.ABSOLUTE,
        ki_continuous=False,
    )
    e = environment(vol=0.0, r=0.0, q=0.0)
    # Deterministic spot remains between the barriers, so only pre-KI tenor
    # and its maturity payoff are owned by this path.
    expected = p.get_maturity_payoff_v0(e.spot, pricing_env=e)
    assert KOResetSnowballQuadEngineV2().price(p, e) == pytest.approx(
        expected, abs=1e-10
    )


def test_event_probabilities_and_independent_terminal_cashflow():
    p = snowball(times=(1.0,))
    env = environment()
    stats = SnowballQuadEngineV2().calculate_event_stats(p, env)
    ko_prob = norm.cdf(-math.log(103 / 100) / 0.2)
    ki_prob = norm.cdf(math.log(75 / 100) / 0.2)
    assert stats.ko_probability[0] == pytest.approx(ko_prob, abs=1e-13)
    assert stats.ki_probability == pytest.approx(ki_prob, abs=1e-13)
    assert stats.ki_ever_probability == pytest.approx(ki_prob, abs=1e-13)
    assert stats.expected_discounted_ko_cashflow[0] == pytest.approx(
        12 * math.exp(-0.03) * ko_prob, abs=1e-12
    )
    assert abs(stats.reconciliation_error) < 1e-12
    assert stats.pv == pytest.approx(
        sum(stats.expected_discounted_cashflows), abs=1e-12
    )


def test_full_risk_facade_preserves_curve_shapes_and_aging():
    from quantark.asset.equity.riskmeasures.greeks_calculator import GreeksCalculator

    p = snowball()
    env = environment()
    engine = SnowballQuadEngineV2()
    result = GreeksCalculator(QuadV2Params()).calculate(
        p,
        env,
        engine,
        greeks=["delta", "gamma", "vega", "rho", "dividend_rho", "theta"],
    )
    assert set(result) == {"delta", "gamma", "vega", "rho", "dividend_rho", "theta"}
    assert np.isfinite(list(result.values())).all()
    shifted = deepcopy(env)
    bump = QuadV2Params().get_effective_bump_config().rate_bump
    shifted.rate_curve = env.rate_curve.parallel_shifted(bump)
    assert result["rho"] == pytest.approx(
        (engine.price(p, shifted) - engine.price(p, env)) * 0.01 / bump, abs=1e-10
    )
