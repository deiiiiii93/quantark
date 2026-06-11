"""
Unit tests for SnowballQuadEngine.

Focus on direct regime-switching quadrature behavior for discrete KO with
discrete/continuous KI monitoring.
"""

import sys
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.engine.quad.quad_math import QuadratureMath
from quantark.asset.equity.param import QuadParams
from quantark.asset.equity.product.option.snowball_config import (
    AccrualConfig,
    AirbagConfig,
    BarrierConfig,
    PayoffConfig,
)
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
    TermStructureVolSurface,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import KnockInMonitoringMode, ObservationType
from quantark.util.exceptions import NumericalError, ValidationError


def create_pricing_env(
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


def create_barrier_config(
    ko_barrier: float,
    ki_barrier: float,
    ki_continuous: bool,
    ko_observation_dates: list[float] = None,
    ki_observation_dates: list[float] = None,
    disable_ko_after_ki: bool = False,
) -> BarrierConfig:
    if ko_observation_dates is None:
        ko_observation_dates = [0.25, 0.5, 0.75, 1.0]
    return BarrierConfig(
        ko_barrier=ko_barrier,
        ko_rate=0.15,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=ko_observation_dates,
        ki_barrier=ki_barrier,
        ki_observation_type=(
            ObservationType.CONTINUOUS if ki_continuous else ObservationType.DISCRETE
        ),
        ki_observation_dates=ki_observation_dates,
        ki_continuous=ki_continuous,
        disable_ko_after_ki=disable_ko_after_ki,
    )


def create_standard_snowball(barrier_config: BarrierConfig) -> SnowballOption:
    return SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=barrier_config,
        contract_multiplier=10_000.0,
        maturity=1.0,
        is_reverse=False,
    )


def create_reverse_snowball(barrier_config: BarrierConfig) -> SnowballOption:
    return SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=barrier_config,
        contract_multiplier=10_000.0,
        maturity=1.0,
        is_reverse=True,
    )


def test_standard_snowball_quad_price_positive():
    env = create_pricing_env()
    barrier_config = create_barrier_config(
        ko_barrier=103.0, ki_barrier=75.0, ki_continuous=True
    )
    snowball = create_standard_snowball(barrier_config)
    engine = SnowballQuadEngine(params=QuadParams(grid_points=301))
    price = engine.price(snowball, env)
    assert np.isfinite(price)
    assert price > 0.0


def test_discrete_ki_vs_continuous_ki():
    env = create_pricing_env()
    barrier_cont = create_barrier_config(
        ko_barrier=103.0, ki_barrier=75.0, ki_continuous=True
    )
    barrier_disc = create_barrier_config(
        ko_barrier=103.0,
        ki_barrier=75.0,
        ki_continuous=False,
        ki_observation_dates=[0.5, 1.0],
    )
    snowball_cont = create_standard_snowball(barrier_cont)
    snowball_disc = create_standard_snowball(barrier_disc)
    engine = SnowballQuadEngine(params=QuadParams(grid_points=301))
    price_cont = engine.price(snowball_cont, env)
    price_disc = engine.price(snowball_disc, env)
    assert price_cont <= price_disc + 1e-6


def test_reverse_snowball_quad_price_positive():
    env = create_pricing_env()
    barrier_config = create_barrier_config(
        ko_barrier=97.0, ki_barrier=125.0, ki_continuous=True
    )
    snowball = create_reverse_snowball(barrier_config)
    engine = SnowballQuadEngine(params=QuadParams(grid_points=301))
    price = engine.price(snowball, env)
    assert np.isfinite(price)
    assert price > 0.0


def test_airbag_snowball_prices_higher_than_standard():
    env = create_pricing_env(spot=95.0, vol=0.30)
    barrier_config = create_barrier_config(
        ko_barrier=103.0, ki_barrier=85.0, ki_continuous=True
    )
    airbag_config = AirbagConfig(
        airbag_barrier=80.0,
        airbag_participation_rate=0.5,
        airbag_strike=90.0,
    )
    standard = SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=barrier_config,
        contract_multiplier=10_000.0,
        maturity=1.0,
        is_reverse=False,
    )
    airbag = SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=barrier_config,
        airbag_config=airbag_config,
        contract_multiplier=10_000.0,
        maturity=1.0,
        is_reverse=False,
    )
    engine = SnowballQuadEngine(params=QuadParams(grid_points=301))
    price_standard = engine.price(standard, env)
    price_airbag = engine.price(airbag, env)
    assert price_airbag > price_standard or price_airbag == pytest.approx(
        price_standard, abs=1e-6
    )


def test_call_rebate_v0_supported():
    env = create_pricing_env()
    barrier_config = create_barrier_config(
        ko_barrier=103.0, ki_barrier=75.0, ki_continuous=True
    )
    no_rebate = SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=barrier_config,
        payoff_config=PayoffConfig(rebate_rate=0.0, include_principal=True),
        contract_multiplier=10_000.0,
        maturity=1.0,
        is_reverse=False,
    )
    call_rebate = SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=barrier_config,
        payoff_config=PayoffConfig(
            rebate_rate=0.0,
            include_principal=True,
            call_rebate_enabled=True,
            call_strike=90.0,
            call_participation_rate=0.5,
        ),
        contract_multiplier=10_000.0,
        maturity=1.0,
        is_reverse=False,
    )
    engine = SnowballQuadEngine(params=QuadParams(grid_points=301))
    price_no_rebate = engine.price(no_rebate, env)
    price_call_rebate = engine.price(call_rebate, env)
    assert price_call_rebate > price_no_rebate or price_call_rebate == pytest.approx(
        price_no_rebate, abs=1e-6
    )


def test_disable_ko_after_ki_reduces_value():
    env = create_pricing_env(vol=0.30)
    barrier_config = create_barrier_config(
        ko_barrier=103.0,
        ki_barrier=85.0,
        ki_continuous=False,
        ki_observation_dates=[0.5, 1.0],
        disable_ko_after_ki=True,
    )
    barrier_config_enabled = create_barrier_config(
        ko_barrier=103.0,
        ki_barrier=85.0,
        ki_continuous=False,
        ki_observation_dates=[0.5, 1.0],
        disable_ko_after_ki=False,
    )
    snowball_disabled = create_standard_snowball(barrier_config)
    snowball_enabled = create_standard_snowball(barrier_config_enabled)
    engine = SnowballQuadEngine(params=QuadParams(grid_points=301))
    price_disabled = engine.price(snowball_disabled, env)
    price_enabled = engine.price(snowball_enabled, env)
    assert price_disabled < price_enabled or price_disabled == pytest.approx(
        price_enabled, abs=1e-6
    )


def test_quad_applies_immediate_ko_at_valuation_observation():
    env = create_pricing_env(spot=150.0)
    barrier_config = BarrierConfig(
        ko_barrier=103.0,
        ko_rate=0.10,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[0.0, 0.5, 1.0],
        ki_barrier=None,
    )
    product = SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=barrier_config,
        payoff_config=PayoffConfig(include_principal=True),
        accrual_config=AccrualConfig(is_annualized=False),
        contract_multiplier=1.0,
        maturity=1.0,
        is_reverse=False,
    )
    engine = SnowballQuadEngine(params=QuadParams(grid_points=301))

    price = engine.price(product, env)
    ko_record_0 = product.resolve_ko_observations(env)[0]

    assert price == pytest.approx(ko_record_0.payoff, abs=1e-10)

    stats = engine.calculate_event_stats(product, env)
    assert stats is not None
    assert stats.pv == pytest.approx(ko_record_0.payoff, abs=1e-10)
    assert stats.ko_times[0] == pytest.approx(0.0)
    assert stats.ko_probability[0] == pytest.approx(1.0)
    assert stats.expected_discounted_maturity_cashflow == pytest.approx(0.0)


def test_quad_applies_immediate_ki_at_valuation_observation():
    env = create_pricing_env(spot=70.0)
    barrier_config = BarrierConfig(
        ko_barrier=150.0,
        ko_rate=0.10,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[0.5, 1.0],
        ki_barrier=75.0,
        ki_observation_type=ObservationType.DISCRETE,
        ki_observation_dates=[0.0, 0.5, 1.0],
        ki_continuous=False,
    )
    product = SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=barrier_config,
        payoff_config=PayoffConfig(include_principal=True),
        accrual_config=AccrualConfig(is_annualized=False),
        contract_multiplier=1.0,
        maturity=1.0,
        is_reverse=False,
    )
    lifecycle_product = SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=barrier_config,
        payoff_config=PayoffConfig(include_principal=True),
        accrual_config=AccrualConfig(is_annualized=False),
        contract_multiplier=1.0,
        maturity=1.0,
        is_reverse=False,
    )
    setattr(lifecycle_product, "_otc_lifecycle_knocked_in", True)
    engine = SnowballQuadEngine(params=QuadParams(grid_points=301))

    price = engine.price(product, env)
    lifecycle_price = engine.price(lifecycle_product, env)

    assert price == pytest.approx(lifecycle_price, abs=1e-10)

    stats = engine.calculate_event_stats(product, env)
    assert stats is not None
    assert stats.ki_probability == pytest.approx(1.0)
    assert stats.ki_times[0] == pytest.approx(0.0)


def test_diffuse_fft_vectorized_matches_single_surface_path():
    params = QuadParams(grid_points=301)
    engine = SnowballQuadEngine(params=params)
    math_utils = QuadratureMath(
        grid_x=params.grid_points,
        spot=100.0,
        maturity=1.0,
        vol_max=0.2,
        num_std_devs=params.num_std_devs,
        fft_padding_factor=params.fft_padding_factor,
        fft_filter_alpha=params.fft_filter_alpha,
        fft_filter_power=params.fft_filter_power,
    )
    tau_step = 0.5 * 0.2 * 0.2 / 252.0
    alpha = -0.5
    beta = 0.25
    prefactor = math.exp(-beta * tau_step) / math.sqrt(math.pi * tau_step) / 2.0
    omega_array = np.exp(
        -(math_utils.z_grid**2) / (4.0 * tau_step) - alpha * math_utils.z_grid
    )
    values = np.vstack(
        [
            np.exp(-0.5 * math_utils.grid**2),
            np.maximum(100.0 * np.exp(math_utils.grid) - 100.0, 0.0),
            np.maximum(100.0 - 100.0 * np.exp(math_utils.grid), 0.0),
        ]
    )

    vectorized = engine._diffuse_fft(
        values,
        math_utils,
        omega_array,
        prefactor,
        0,
        params.grid_points - 1,
        (params.grid_points - 1) % 2,
        alpha,
        beta,
        tau_step,
    )
    rowwise = np.vstack(
        [
            engine._diffuse_fft(
                row,
                math_utils,
                omega_array,
                prefactor,
                0,
                params.grid_points - 1,
                (params.grid_points - 1) % 2,
                alpha,
                beta,
                tau_step,
            )
            for row in values
        ]
    )

    assert vectorized == pytest.approx(rowwise, rel=1e-12, abs=1e-12)


def test_dense_discrete_ki_does_not_delegate_to_continuous_bridge(monkeypatch):
    env = create_pricing_env()
    ki_dates = [(index + 1) / 130.0 for index in range(130)]
    barrier_config = create_barrier_config(
        ko_barrier=103.0,
        ki_barrier=75.0,
        ki_continuous=False,
        ki_observation_dates=ki_dates,
    )
    product = create_standard_snowball(barrier_config)
    engine = SnowballQuadEngine(params=QuadParams(grid_points=301))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("discrete KI must not use Brownian-bridge diffusion")

    monkeypatch.setattr(engine, "_diffuse_with_bridge", fail_if_called)

    assert np.isfinite(engine.price(product, env))


def test_dense_discrete_ki_refines_under_resolved_grid():
    engine = SnowballQuadEngine(params=QuadParams(grid_points=301))
    times = [(index + 1) / 244.0 for index in range(244)]

    resolved = engine._resolve_grid_points(maturity=1.0, vol=0.20, times=times)

    assert resolved > engine.params.grid_points
    assert resolved % 2 == 1


def test_adaptive_grid_refinement_can_be_disabled():
    engine = SnowballQuadEngine(
        params=QuadParams(grid_points=301, min_diffusion_stddev_cells=0.0)
    )
    times = [(index + 1) / 244.0 for index in range(244)]

    assert engine._resolve_grid_points(maturity=1.0, vol=0.20, times=times) == 301


def test_adaptive_grid_refinement_respects_safety_cap():
    engine = SnowballQuadEngine(
        params=QuadParams(grid_points=301, max_adaptive_grid_points=501)
    )
    times = [(index + 1) / 244.0 for index in range(244)]

    with pytest.raises(NumericalError, match="max_adaptive_grid_points=501"):
        engine._resolve_grid_points(maturity=1.0, vol=0.20, times=times)


def create_daily_ki_snowball(num_observations: int = 252) -> SnowballOption:
    ki_dates = [(index + 1) / num_observations for index in range(num_observations)]
    barrier_config = create_barrier_config(
        ko_barrier=103.0,
        ki_barrier=75.0,
        ki_continuous=False,
        ki_observation_dates=ki_dates,
    )
    return create_standard_snowball(barrier_config)


def test_ki_monitoring_mode_defaults_to_exact_discrete():
    assert QuadParams().ki_monitoring_mode is KnockInMonitoringMode.EXACT_DISCRETE


def test_ki_monitoring_mode_accepts_string_and_rejects_unknown():
    params = QuadParams(ki_monitoring_mode="BGK_APPROXIMATION")
    assert params.ki_monitoring_mode is KnockInMonitoringMode.BGK_APPROXIMATION

    with pytest.raises(ValidationError, match="ki_monitoring_mode"):
        QuadParams(ki_monitoring_mode="continuous_magic")


def test_bgk_mode_matches_exact_discrete_on_regular_daily_schedule():
    env = create_pricing_env()
    product = create_daily_ki_snowball()

    exact = SnowballQuadEngine(params=QuadParams()).price(product, env)
    bgk = SnowballQuadEngine(
        params=QuadParams(ki_monitoring_mode=KnockInMonitoringMode.BGK_APPROXIMATION)
    ).price(product, env)

    assert abs(bgk - exact) < 5e-4 * abs(exact)


def test_bgk_mode_routes_dense_discrete_ki_through_bridge(monkeypatch):
    env = create_pricing_env()
    product = create_daily_ki_snowball()
    engine = SnowballQuadEngine(
        params=QuadParams(ki_monitoring_mode=KnockInMonitoringMode.BGK_APPROXIMATION)
    )

    bridge_calls = []
    original = engine._diffuse_with_bridge

    def record_bridge(*args, **kwargs):
        bridge_calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, "_diffuse_with_bridge", record_bridge)

    assert np.isfinite(engine.price(product, env))
    assert bridge_calls


def test_bgk_mode_rejects_irregular_ki_schedule():
    env = create_pricing_env()
    ki_dates = [(month + 1) / 12.0 for month in range(6)] + [
        0.5 + (index + 1) / 252.0 for index in range(126)
    ]
    barrier_config = create_barrier_config(
        ko_barrier=103.0,
        ki_barrier=75.0,
        ki_continuous=False,
        ki_observation_dates=ki_dates,
    )
    product = create_standard_snowball(barrier_config)
    engine = SnowballQuadEngine(
        params=QuadParams(ki_monitoring_mode=KnockInMonitoringMode.BGK_APPROXIMATION)
    )

    with pytest.raises(ValidationError, match="regular"):
        engine.price(product, env)


def test_bgk_mode_rejects_late_starting_daily_window():
    """A daily schedule omitting its first eight days must not be BGK-converted.

    Continuous monitoring would cover the unmonitored opening window and
    materially overstate the knock-in probability near the barrier.
    """
    env = create_pricing_env()
    ki_dates = [(index + 8) / 365.0 for index in range(358)]
    barrier_config = create_barrier_config(
        ko_barrier=103.0,
        ki_barrier=75.0,
        ki_continuous=False,
        ki_observation_dates=ki_dates,
    )
    product = SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=barrier_config,
        contract_multiplier=10_000.0,
        maturity=ki_dates[-1],
        is_reverse=False,
    )
    engine = SnowballQuadEngine(
        params=QuadParams(ki_monitoring_mode=KnockInMonitoringMode.BGK_APPROXIMATION)
    )

    with pytest.raises(ValidationError, match="horizon"):
        engine.price(product, env)


def test_bgk_mode_rejects_partial_monitoring_window():
    env = create_pricing_env()
    ki_dates = [0.5 + (index + 1) / 252.0 for index in range(126)]
    barrier_config = create_barrier_config(
        ko_barrier=103.0,
        ki_barrier=75.0,
        ki_continuous=False,
        ki_observation_dates=ki_dates,
    )
    product = create_standard_snowball(barrier_config)
    engine = SnowballQuadEngine(
        params=QuadParams(ki_monitoring_mode=KnockInMonitoringMode.BGK_APPROXIMATION)
    )

    with pytest.raises(ValidationError, match="horizon"):
        engine.price(product, env)


def test_bgk_mode_rejects_varying_ki_barrier_schedule():
    env = create_pricing_env()
    num_observations = 252
    ki_dates = [(index + 1) / num_observations for index in range(num_observations)]
    step_down_barriers = [
        75.0 - 2.0 * index / (num_observations - 1) for index in range(num_observations)
    ]
    barrier_config = create_barrier_config(
        ko_barrier=103.0,
        ki_barrier=step_down_barriers,
        ki_continuous=False,
        ki_observation_dates=ki_dates,
    )
    product = create_standard_snowball(barrier_config)
    engine = SnowballQuadEngine(
        params=QuadParams(ki_monitoring_mode=KnockInMonitoringMode.BGK_APPROXIMATION)
    )

    with pytest.raises(ValidationError, match="constant"):
        engine.price(product, env)


def test_bgk_mode_prices_as_shifted_continuous_with_discrete_valuation_state():
    """BGK pricing must equal the equivalent shifted-continuous product.

    Covers two regressions: valuation-time KI state must follow the
    contractual discrete semantics (a spot between the shifted and the
    contractual barrier is NOT knocked in without a t=0 observation), and
    the spatial grid must align to the model's shifted barrier.
    """
    num_observations = 252
    vol, ki_barrier = 0.20, 75.0
    shifted = ki_barrier * math.exp(
        -0.5825971579390107 * vol * math.sqrt(1.0 / num_observations)
    )
    spot_between = (shifted + ki_barrier) / 2.0
    env = create_pricing_env(spot=spot_between, vol=vol)

    product = create_daily_ki_snowball(num_observations)
    bgk_engine = SnowballQuadEngine(
        params=QuadParams(ki_monitoring_mode=KnockInMonitoringMode.BGK_APPROXIMATION)
    )
    bgk_price = bgk_engine.price(product, env)

    continuous_config = create_barrier_config(
        ko_barrier=103.0, ki_barrier=shifted, ki_continuous=True
    )
    continuous_product = create_standard_snowball(continuous_config)
    continuous_price = SnowballQuadEngine(params=QuadParams()).price(
        continuous_product, env
    )

    assert bgk_price == pytest.approx(continuous_price, abs=1e-9)

    stats = bgk_engine.calculate_event_stats(product, env)
    assert stats.ki_probability < 1.0


def test_bgk_mode_rejects_alternating_spacing():
    ki_dates = []
    t = 0.0
    index = 0
    while t + (1.0 if index % 2 == 0 else 7.0) / 365.0 <= 1.0:
        t += (1.0 if index % 2 == 0 else 7.0) / 365.0
        ki_dates.append(t)
        index += 1
    barrier_config = create_barrier_config(
        ko_barrier=103.0,
        ki_barrier=75.0,
        ki_continuous=False,
        ki_observation_dates=ki_dates,
    )
    product = create_standard_snowball(barrier_config)
    engine = SnowballQuadEngine(
        params=QuadParams(
            ki_monitoring_mode=KnockInMonitoringMode.BGK_APPROXIMATION,
            bgk_min_ki_observations=2,
        )
    )

    with pytest.raises(ValidationError, match="regular"):
        engine.price(product, create_pricing_env())


def test_bgk_mode_rejects_sparse_schedule():
    barrier_config = create_barrier_config(
        ko_barrier=103.0,
        ki_barrier=75.0,
        ki_continuous=False,
        ki_observation_dates=[0.5, 1.0],
    )
    product = create_standard_snowball(barrier_config)
    engine = SnowballQuadEngine(
        params=QuadParams(ki_monitoring_mode=KnockInMonitoringMode.BGK_APPROXIMATION)
    )

    with pytest.raises(ValidationError, match="bgk_min_ki_observations"):
        engine.price(product, create_pricing_env())


def test_bgk_mode_accepts_business_day_weekend_gaps():
    ki_dates = []
    t = 0.0
    index = 0
    while True:
        gap = (3.0 if index % 5 == 4 else 1.0) / 365.0
        if t + gap > 1.0:
            break
        t += gap
        ki_dates.append(t)
        index += 1
    barrier_config = create_barrier_config(
        ko_barrier=103.0,
        ki_barrier=75.0,
        ki_continuous=False,
        ko_observation_dates=[(month + 1) * ki_dates[-1] / 12.0 for month in range(12)],
        ki_observation_dates=ki_dates,
    )
    product = SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=barrier_config,
        contract_multiplier=10_000.0,
        maturity=ki_dates[-1] + 1.0 / 365.0,
        is_reverse=False,
    )
    engine = SnowballQuadEngine(
        params=QuadParams(ki_monitoring_mode=KnockInMonitoringMode.BGK_APPROXIMATION)
    )

    assert np.isfinite(engine.price(product, create_pricing_env()))


def test_bgk_mode_rejects_unstable_volatility():
    env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=TermStructureVolSurface(times=[0.1, 1.0], vols=[0.40, 0.20]),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.02),
        valuation_date=datetime(2024, 1, 1),
    )
    product = create_daily_ki_snowball()
    engine = SnowballQuadEngine(
        params=QuadParams(ki_monitoring_mode=KnockInMonitoringMode.BGK_APPROXIMATION)
    )

    with pytest.raises(ValidationError, match="volatility"):
        engine.price(product, env)
