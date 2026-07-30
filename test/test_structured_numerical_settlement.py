"""Settlement-date reconciliation for structured PDE and QUAD engines."""

from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.capabilities import (
    SettlementSupport,
    VolDynamicsType,
    validate_engine_capability,
)
from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.pde import (
    GridConfig,
    KOResetSnowballPDESolver,
    PhoenixPDESolver,
    SnowballPDESolver,
)
from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import (
    KOResetSnowballQuadEngine,
)
from quantark.asset.equity.engine.quad.phoenix_quad_engine import (
    PhoenixQuadEngine,
)
from quantark.asset.equity.engine.quad.quad_adapters import (
    resolve_structured_payment_timings,
)
from quantark.asset.equity.engine.quad.snowball_quad_engine import (
    SnowballQuadEngine,
)
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option.phoenix_config import (
    CouponBarrierConfig,
)
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.observation_schedule import (
    ObservationRecord,
    ObservationSchedule,
)
from quantark.asset.equity.product.option import create_ko_reset_snowball
from quantark.asset.equity.product.option.snowball_config import (
    AccrualConfig,
    BarrierConfig,
    PayoffConfig,
)
from quantark.asset.equity.product.option.snowball_option import SnowballOption
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
from quantark.util.calendar.day_counter import DayCountConvention
from quantark.util.enum import (
    CouponPayType,
    ObservationType,
    PostKOScheduleMode,
)
from quantark.util.enum.engine_enums import EngineType


RATE = 0.05
LAG = 0.10


def _env() -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=RATE),
        div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=datetime(2026, 1, 1),
    )


def _convention():
    return SettlementConvention(
        lag=LAG,
        lag_unit=SettlementLagUnit.YEAR_FRACTION,
    )


def _barriers():
    return BarrierConfig(
        ko_barrier=105.0,
        ko_rate=0.10,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[0.5, 1.0],
        ki_barrier=70.0,
        ki_observation_type=ObservationType.CONTINUOUS,
        ki_continuous=True,
    )


def _snowball(
    convention=None,
    *,
    pay_type=CouponPayType.INSTANT,
    barrier_config=None,
):
    return SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=barrier_config or _barriers(),
        payoff_config=PayoffConfig(include_principal=True),
        accrual_config=AccrualConfig(
            coupon_pay_type=pay_type,
        ),
        maturity=1.0,
        settlement_convention=convention,
    )


def _phoenix(convention=None):
    return PhoenixOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=_barriers(),
        coupon_config=CouponBarrierConfig(
            coupon_barrier=[80.0, 80.0],
            coupon_rate=0.04,
            coupon_pay_type=CouponPayType.INSTANT,
            day_count_convention=DayCountConvention.ACT_365,
            memory_coupon=False,
        ),
        payoff_config=PayoffConfig(include_principal=True),
        maturity=1.0,
        settlement_convention=convention,
    )


def _pde_params():
    return PDEParams(
        grid=GridConfig(points=161, steps_per_day=1.0),
    )


@pytest.mark.parametrize(
    ("factory", "engine_factory"),
    [
        (_snowball, lambda: SnowballPDESolver(params=_pde_params())),
        (_snowball, lambda: SnowballQuadEngine(QuadParams(grid_points=401))),
        (_phoenix, lambda: PhoenixPDESolver(params=_pde_params())),
        (_phoenix, lambda: PhoenixQuadEngine(QuadParams(grid_points=401))),
    ],
)
def test_uniform_lag_scales_all_structured_cashflows(
    factory,
    engine_factory,
):
    env = _env()
    immediate = engine_factory().price(factory(), env)
    delayed = engine_factory().price(factory(_convention()), env)

    assert delayed == pytest.approx(
        immediate * np.exp(-RATE * LAG),
        rel=2.0e-9,
        abs=2.0e-8,
    )


@pytest.mark.parametrize(
    "engine",
    [
        SnowballPDESolver(params=_pde_params()),
        SnowballQuadEngine(QuadParams(grid_points=201)),
        PhoenixPDESolver(params=_pde_params()),
        PhoenixQuadEngine(QuadParams(grid_points=201)),
    ],
)
def test_structured_engines_declare_event_and_terminal_support(engine):
    assert engine.settlement_support is SettlementSupport.EVENT_AND_TERMINAL


def test_quad_recording_nodes_do_not_extend_transition_horizon():
    engine = SnowballQuadEngine(QuadParams(grid_points=201))
    records = _snowball(_convention()).resolve_ko_observations(_env())

    times = engine._insert_settlement_times(
        [0.5, 1.0],
        records,
        maturity=1.0,
    )

    assert times == pytest.approx([0.5, 0.6, 1.0])
    assert max(times) == pytest.approx(1.0)


def test_expiry_paid_ko_records_use_terminal_payment_time():
    records = _snowball(
        _convention(),
        pay_type=CouponPayType.EXPIRY,
    ).resolve_ko_observations(_env())

    assert [record.settlement_time for record in records] == pytest.approx(
        [1.0 + LAG, 1.0 + LAG]
    )


def test_structured_timing_bundle_preserves_explicit_event_times():
    schedule = ObservationSchedule(
        records=[
            ObservationRecord(
                observation_time=0.50,
                settlement_time=0.57,
                barrier=105.0,
            ),
            ObservationRecord(
                observation_time=1.00,
                settlement_time=1.14,
                barrier=105.0,
            ),
        ]
    )
    product = _snowball(
        _convention(),
        barrier_config=BarrierConfig(
            ko_barrier=105.0,
            ko_rate=0.10,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_schedule=schedule,
            ki_barrier=70.0,
            ki_observation_type=ObservationType.CONTINUOUS,
            ki_continuous=True,
        ),
    )
    records = product.resolve_ko_observations(_env())
    timings = resolve_structured_payment_timings(product, _env(), records)

    assert timings.event_payment_times == pytest.approx([0.57, 1.14])
    assert timings.event_delay_dfs == pytest.approx(
        np.exp(-RATE * np.asarray([0.07, 0.14]))
    )
    assert timings.terminal.payment_time == pytest.approx(1.0 + LAG)


def test_ko_reset_expiry_payments_use_terminal_payment_time():
    product = create_ko_reset_snowball(
        initial_price=100.0,
        strike=100.0,
        maturity_pre=1.0,
        maturity_post=2.0,
        post_ko_mode=PostKOScheduleMode.ABSOLUTE,
        ki_continuous=True,
        coupon_pay_type=CouponPayType.EXPIRY,
    )
    product.settlement_convention = _convention()
    env = _env()
    engines = [
        KOResetSnowballPDESolver(params=_pde_params()),
        KOResetSnowballQuadEngine(QuadParams(grid_points=201)),
    ]

    for engine in engines:
        for config in (
            product.barrier_config,
            product.post_barrier_config,
        ):
            records = engine._resolve_ko_records(product, env, config)
            assert [
                record.settlement_time for record in records
            ] == pytest.approx([2.0 + LAG] * len(records))


def test_payment_lag_does_not_change_quad_event_probabilities():
    env = _env()
    params = QuadParams(grid_points=301)
    immediate = SnowballQuadEngine(params).calculate_event_stats(
        _snowball(), env
    )
    delayed = SnowballQuadEngine(params).calculate_event_stats(
        _snowball(_convention()), env
    )

    assert delayed.ko_probability == pytest.approx(immediate.ko_probability)
    assert delayed.survival_probability == pytest.approx(
        immediate.survival_probability
    )
    assert delayed.ki_ever_probability == pytest.approx(
        immediate.ki_ever_probability
    )


def test_delayed_snowball_mc_pde_quad_reconcile():
    product = _snowball(_convention())
    env = _env()
    values = np.asarray(
        [
            SnowballMCEngine(
                MCParams(num_paths=32768, seed=19)
            ).price(product, env),
            SnowballPDESolver(_pde_params()).price(product, env),
            SnowballQuadEngine(
                QuadParams(grid_points=801)
            ).price(product, env),
        ]
    )

    assert np.max(values) - np.min(values) <= 0.003 * np.mean(values)


@pytest.mark.parametrize(
    "dynamics",
    [
        VolDynamicsType.LOCAL_VOL,
        VolDynamicsType.HESTON,
        VolDynamicsType.SLV,
    ],
)
def test_non_bsm_quad_paths_fail_closed(dynamics):
    with pytest.raises(CapabilityError, match="QUAD is not supported"):
        validate_engine_capability(dynamics, EngineType.QUADRATURE)
