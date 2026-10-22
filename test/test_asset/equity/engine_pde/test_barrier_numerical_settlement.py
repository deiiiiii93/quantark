"""Node-specific settlement timing for barrier/touch PDE and QUAD engines."""

from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.pde import (
    BarrierPDESolver,
    LocalVolBarrierPDESolver,
)
from quantark.asset.equity.engine.quad import DiscreteQuadEngine
from quantark.asset.equity.engine.quad.quad_adapters import (
    BarrierQuadInputAdapter,
)
from quantark.asset.equity.param import PDEParams, QuadParams
from quantark.asset.equity.engine.pde.grid.config import GridConfig
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
    # the breached node's cash, not the projection: the 0.7.0 nodal mask (0.8.0 projects by default, 3A §5.4)
    solver = BarrierPDESolver(PDEParams(grid=GridConfig(points=51, time_steps=4), event_projection="nodal"))
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
    solver = BarrierPDESolver(PDEParams(grid=GridConfig(points=51, time_steps=4)))
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
    # the breached node's cash, not the projection: the 0.7.0 nodal mask (0.8.0 projects by default, 3A §5.4)
    solver = BarrierPDESolver(PDEParams(grid=GridConfig(points=51, time_steps=4), event_projection="nodal"))
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


def test_vol_model_barrier_pde_scales_terminal_kernel_value(env):
    # spec 3C §5.3: the local-vol barrier marches on the core; a terminal lag scales the whole terminal value
    from pde_routes.cases import surface_const
    engine = LocalVolBarrierPDESolver(local_vol_surface=surface_const(0.20))

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
        / env.get_discount_factor(MATURITY),
        rel=1e-12,                                                   # measured 6.7e-16
    )


def test_local_vol_barrier_prices_a_lagged_pay_at_hit_rebate():
    """The kernel refused a lagged first-hit cashflow (CapabilityError); the core pays it at hit + lag. On a flat
    rate the lag scales every cashflow by exp(-r L): the reference is exp(-r L) x the exact recursion."""
    import math

    from pde_routes.barrier_reference import knock_out
    from pde_routes.cases import CONST, barrier, env_const, obs_dates, surface_const

    product = barrier("UO", True, 130.0, obs="monthly", rebate=5.0, at_hit=True)
    product.settlement_convention = _lagged()
    got = LocalVolBarrierPDESolver(local_vol_surface=surface_const()).price(product, env_const())
    ref = knock_out(payoff="call", strike=100.0, lower=None, upper=130.0, cash=5.0, at_hit=True,
                    obs=obs_dates("monthly"), spots=[100.0], vol=CONST["vol"], r=CONST["rate"],
                    q=CONST["div"])[0][100.0][0] * math.exp(-CONST["rate"] * LAG)
    assert abs(got - ref) <= 1e-3                                    # measured 2.5e-4; ignoring the lag: 1.3e-2
