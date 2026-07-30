"""Node-specific settlement timing for barrier/touch PDE and QUAD engines."""

from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.pde import (
    BarrierPDESolver,
    LocalVolBarrierPDESolver,
)
from quantark.asset.equity.engine.pde import barrier_vol_pde_solvers
from quantark.asset.equity.engine.quad import DiscreteQuadEngine
from quantark.asset.equity.engine.quad.quad_adapters import (
    BarrierQuadInputAdapter,
)
from quantark.asset.equity.param import PDEParams, QuadParams
from quantark.asset.equity.product.option import (
    BarrierOption,
    ObservationRecord,
    ObservationSchedule,
    OneTouchOption,
)
from quantark.asset.equity.settlement import (
    SettlementConvention,
    SettlementLagUnit,
)
from quantark.param import (
    ContinuousDividendYield,
    FlatVolSurface,
    SpotQuote,
)
from quantark.param.rrf.rate_curve import LinearRateCurve
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import (
    BarrierDirection,
    BarrierType,
    ObservationAggregation,
    ObservationType,
    OptionType,
    TouchType,
)


MATURITY = 1.0
LAG = 0.10


@pytest.fixture
def env():
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=LinearRateCurve(
            [(0.25, 0.01), (0.50, 0.025), (1.0, 0.06), (1.2, 0.07)]
        ),
        div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=datetime(2026, 1, 1),
    )


def _lagged():
    return SettlementConvention(
        lag=LAG,
        lag_unit=SettlementLagUnit.YEAR_FRACTION,
    )


def _discrete_barrier(convention):
    return BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=105.0,
        barrier_type=BarrierType.UP_OUT,
        maturity=MATURITY,
        rebate=7.0,
        pay_at_hit=True,
        observation_type=ObservationType.DISCRETE,
        observation_schedule=ObservationSchedule(
            records=[
                ObservationRecord(
                    observation_time=0.50,
                    barrier=105.0,
                    payoff=7.0,
                )
            ],
            aggregation_mode=ObservationAggregation.STOP_FIRST_HIT,
        ),
        settlement_convention=convention,
    )


def test_barrier_pde_event_node_uses_payment_delay_ratio(env):
    product = _discrete_barrier(_lagged())
    solver = BarrierPDESolver(PDEParams(grid_size=51, time_steps=4))
    t_vec = np.array([0.0, 0.5, 1.0])
    solver._setup_observation_indices(
        product,
        env,
        MATURITY,
        t_vec,
        resolve_kwargs={
            "default_barrier": product.barrier,
            "default_payoff": product.rebate,
            "require_single": True,
        },
    )
    grid = np.zeros((3, 3))
    s_vec = np.array([90.0, 100.0, 110.0])

    solver._apply_step_modifications(
        grid,
        np.log(s_vec),
        s_vec,
        t_idx=1,
        tau=0.5,
        product=product,
        pricing_env=env,
    )

    assert grid[-1, 1] == pytest.approx(
        7.0
        * env.get_discount_factor(0.50 + LAG)
        / env.get_discount_factor(0.50)
    )


def test_barrier_pde_expiry_paid_boundary_uses_terminal_delay(env):
    product = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=105.0,
        barrier_type=BarrierType.UP_OUT,
        maturity=MATURITY,
        rebate=7.0,
        pay_at_hit=False,
        observation_type=ObservationType.CONTINUOUS,
        settlement_convention=_lagged(),
    )
    solver = BarrierPDESolver(PDEParams(grid_size=51, time_steps=4))
    solver._total_tau = MATURITY
    grid = np.zeros((3, 3))
    s_vec = np.array([90.0, 100.0, 110.0])

    solver.set_boundary_conditions(
        grid,
        np.log(s_vec),
        s_vec,
        t_idx=1,
        tau=0.5,
        product=product,
        pricing_env=env,
    )

    assert grid[-1, 1] == pytest.approx(
        7.0
        * env.get_discount_factor(MATURITY + LAG)
        / env.get_discount_factor(0.50)
    )


def test_barrier_pde_expiry_paid_discrete_node_uses_terminal_delay(env):
    product = _discrete_barrier(_lagged())
    product.pay_at_hit = False
    solver = BarrierPDESolver(PDEParams(grid_size=51, time_steps=4))
    t_vec = np.array([0.0, 0.5, 1.0])
    solver._setup_observation_indices(
        product,
        env,
        MATURITY,
        t_vec,
        resolve_kwargs={
            "default_barrier": product.barrier,
            "default_payoff": product.rebate,
            "require_single": True,
        },
    )
    grid = np.zeros((3, 3))
    s_vec = np.array([90.0, 100.0, 110.0])

    solver._apply_step_modifications(
        grid,
        np.log(s_vec),
        s_vec,
        t_idx=1,
        tau=0.5,
        product=product,
        pricing_env=env,
    )

    assert grid[-1, 1] == pytest.approx(
        7.0
        * env.get_discount_factor(MATURITY + LAG)
        / env.get_discount_factor(0.50)
    )


def test_barrier_quad_adapter_uses_resolved_event_payment_df(env):
    product = _discrete_barrier(_lagged())
    adapter = BarrierQuadInputAdapter()
    engine = DiscreteQuadEngine(QuadParams(grid_points=201))
    context = adapter.build_pricing_context(product, env, engine)
    resolved = adapter.resolve_schedule(product, env, context)
    inputs = adapter.build_inputs(product, resolved, context)

    assert resolved[0].settlement_time == pytest.approx(0.50 + LAG)
    assert inputs.b_plus[0] == pytest.approx(
        7.0
        * env.get_discount_factor(0.50 + LAG)
        / env.get_discount_factor(0.50)
    )
    assert inputs.observation_times[-1] == pytest.approx(MATURITY)


def test_no_touch_quad_uses_terminal_payment_timing(env):
    def _product(convention):
        return OneTouchOption(
            barrier=120.0,
            barrier_direction=BarrierDirection.UP,
            maturity=MATURITY,
            rebate=10.0,
            payment_at_hit=False,
            touch_type=TouchType.NO_TOUCH,
            observation_type=ObservationType.DISCRETE,
            observation_schedule=ObservationSchedule(
                records=[
                    ObservationRecord(
                        observation_time=0.50,
                        barrier=120.0,
                        payoff=10.0,
                    )
                ]
            ),
            settlement_convention=convention,
        )

    engine = DiscreteQuadEngine(QuadParams(grid_points=401))
    immediate = engine.price(_product(None), env)
    delayed = engine.price(_product(_lagged()), env)

    assert delayed == pytest.approx(
        immediate
        * env.get_discount_factor(MATURITY + LAG)
        / env.get_discount_factor(MATURITY),
        rel=2.0e-10,
    )


def test_vol_model_barrier_pde_scales_terminal_kernel_value(
    env, monkeypatch
):
    monkeypatch.setattr(
        barrier_vol_pde_solvers,
        "price_barrier_lv_pde",
        lambda *_args, **_kwargs: 8.0,
    )
    engine = LocalVolBarrierPDESolver(
        params=PDEParams(grid_size=51, time_steps=4),
        local_vol_surface=object(),
    )

    def _product(convention):
        return BarrierOption(
            strike=100.0,
            option_type=OptionType.CALL,
            barrier=130.0,
            barrier_type=BarrierType.UP_OUT,
            maturity=MATURITY,
            rebate=2.0,
            pay_at_hit=False,
            observation_type=ObservationType.EXPIRY,
            settlement_convention=convention,
        )

    immediate = engine.price(_product(None), env)
    delayed = engine.price(_product(_lagged()), env)

    assert delayed == pytest.approx(
        immediate
        * env.get_discount_factor(MATURITY + LAG)
        / env.get_discount_factor(MATURITY)
    )
