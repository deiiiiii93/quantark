"""
Regression tests for the 2026-07 PDE/QUAD audit findings.

Each test is keyed to an audit finding and is written to FAIL on the
pre-fix code for the exact defect described, and pass once fixed:

- Rannacher off-by-one (base_pde_solver / backward_operator)
- QUAD pay_at_hit leaking into knock-in / no-touch parity decomposition
- QUAD boundary-point evaluation using the wrong barrier index
- Double-barrier / one-touch PDE applying continuous-monitoring boundary
  conditions and a phantom terminal observation to DISCRETE products
- Term-structure-inconsistent rebate discounting in sibling solvers
- Phoenix PDE paying accrued memory coupons at KO regardless of the
  coupon-barrier condition (MC semantics are the reference)
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantark.asset.equity.engine.mc.barrier_option_mc_engine import BarrierOptionMCEngine
from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
from quantark.asset.equity.engine.pde import (
    BarrierPDESolver,
    DoubleBarrierPDESolver,
    OneTouchPDESolver,
)
from quantark.asset.equity.engine.pde.backward_operator import BackwardOperator
from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.quad import BarrierQuadEngine, OneTouchQuadEngine
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option import (
    BarrierOption,
    DoubleBarrierOption,
    OneTouchOption,
    PhoenixOption,
)
from quantark.asset.equity.product.option.observation_schedule import (
    ObservationRecord,
    ObservationSchedule,
)
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.snowball_config import BarrierConfig, PayoffConfig
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.rrf.rate_curve import LinearRateCurve
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar.day_counter import DayCountConvention
from quantark.util.enum import (
    BarrierDirection,
    BarrierType,
    CouponPayType,
    DoubleBarrierType,
    ObservationAggregation,
    ObservationType,
    OptionType,
    ProtectionType,
    TouchType,
)
from quantark.util.enum.engine_enums import MonteCarloMethod


def create_pricing_env(
    spot: float = 100.0,
    vol: float = 0.20,
    rate: float = 0.03,
    div: float = 0.01,
    rate_curve=None,
) -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=rate_curve if rate_curve is not None else FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div),
        valuation_date=datetime(2024, 1, 1),
    )


def build_schedule(times, barrier, payoff=0.0) -> ObservationSchedule:
    return ObservationSchedule(
        records=[
            ObservationRecord(observation_time=float(t), barrier=barrier, payoff=payoff)
            for t in times
        ],
        aggregation_mode=ObservationAggregation.STOP_FIRST_HIT,
    )


# ============================================================================
# Audit #5 — Rannacher off-by-one: default rannacher_steps=1 must smooth
# exactly one backward-Euler step off the terminal payoff.
# ============================================================================


class TestRannacherTerminalSmoothing:
    def test_default_rannacher_smooths_first_backward_step(self):
        t_vec = np.linspace(0.0, 1.0, 11)
        params = PDEParams()  # use_rannacher=True, rannacher_steps=1
        theta = BackwardOperator.theta_by_step(
            t_vec, np.diff(t_vec), params, discontinuity_times=None
        )
        # rannacher_steps=1 => exactly ONE implicit-Euler step at the terminal.
        assert theta[-1] == 1.0
        assert np.allclose(theta[:-1], params.theta)

    def test_rannacher_steps_two_smooths_two_steps(self):
        t_vec = np.linspace(0.0, 1.0, 11)
        params = PDEParams(rannacher_steps=2)
        theta = BackwardOperator.theta_by_step(
            t_vec, np.diff(t_vec), params, discontinuity_times=None
        )
        assert theta[-1] == 1.0
        assert theta[-2] == 1.0
        assert np.allclose(theta[:-2], params.theta)


# ============================================================================
# Audit #4 — QUAD adapters must not let pay_at_hit / payment_at_hit leak into
# the knock-in / no-touch parity decomposition. For those products the rebate
# is paid at expiry by definition, so the flag cannot affect the price.
# ============================================================================


class TestQuadPayAtHitGating:
    DATES = (0.25, 0.5, 0.75, 1.0)

    def _no_touch(self, payment_at_hit: bool) -> OneTouchOption:
        return OneTouchOption(
            barrier=90.0,
            barrier_direction=BarrierDirection.DOWN,
            maturity=1.0,
            rebate=5.0,
            payment_at_hit=payment_at_hit,
            touch_type=TouchType.NO_TOUCH,
            observation_type=ObservationType.DISCRETE,
            observation_dates=list(self.DATES),
        )

    def test_no_touch_price_independent_of_payment_at_hit(self):
        env = create_pricing_env()
        engine = OneTouchQuadEngine(params=QuadParams(grid_points=801))
        price_hit = engine.price(self._no_touch(payment_at_hit=True), env)
        price_expiry = engine.price(self._no_touch(payment_at_hit=False), env)
        assert price_hit == pytest.approx(price_expiry, rel=1e-10, abs=1e-10)

    def _knock_in(self, pay_at_hit: bool) -> BarrierOption:
        return BarrierOption(
            strike=100.0,
            option_type=OptionType.CALL,
            barrier=80.0,
            barrier_type=BarrierType.DOWN_IN,
            maturity=1.0,
            rebate=5.0,
            pay_at_hit=pay_at_hit,
            observation_type=ObservationType.DISCRETE,
            observation_dates=list(self.DATES),
        )

    def test_knock_in_price_independent_of_pay_at_hit(self):
        env = create_pricing_env()
        engine = BarrierQuadEngine(params=QuadParams(grid_points=801))
        price_hit = engine.price(self._knock_in(pay_at_hit=True), env)
        price_expiry = engine.price(self._knock_in(pay_at_hit=False), env)
        assert price_hit == pytest.approx(price_expiry, rel=1e-10, abs=1e-10)


# ============================================================================
# Audit #3 — QUAD core: boundary-point evaluation of V_{M-1} must use the
# terminal-step barrier levels (index M), exactly as the grid-wide evaluation
# does. Triggered by a schedule whose barrier is NOT observed at maturity.
# ============================================================================


class TestQuadBoundaryIndexSchedule:
    def test_up_out_schedule_ending_before_maturity_matches_mc(self):
        env = create_pricing_env()
        dates = [0.25, 0.5, 0.75]  # no observation at maturity => K_M != K_{M-1}
        schedule = build_schedule(times=dates, barrier=110.0)
        option = BarrierOption(
            strike=100.0,
            option_type=OptionType.CALL,
            barrier=110.0,
            barrier_type=BarrierType.UP_OUT,
            maturity=1.0,
            rebate=0.0,
            observation_type=ObservationType.DISCRETE,
            observation_schedule=schedule,
        )

        quad_price = BarrierQuadEngine(params=QuadParams(grid_points=801)).price(
            option, env
        )
        mc_price = BarrierOptionMCEngine(
            params=MCParams(num_paths=131072, time_steps=252, seed=11, use_qmc=True),
            method=MonteCarloMethod.QUASI,
        ).price(option, env)

        # MC reference is stable to ~0.1% across seeds here; the boundary-index
        # defect shifts the QUAD price by ~3%, so rel=0.01 discriminates cleanly.
        assert quad_price == pytest.approx(mc_price, rel=0.01)


# ============================================================================
# Audit #1/#2 — Discrete double-barrier PDE: no continuous-monitoring boundary
# forcing between observation dates, and no phantom terminal observation.
# A double KO with an unreachable lower barrier must agree with the (already
# fixed) single-barrier solver on the same discrete schedule.
# ============================================================================


class TestDiscreteDoubleBarrierPDE:
    DATES = [0.25, 0.5, 0.75]  # deliberately no observation at maturity

    def test_double_ko_unreachable_lower_matches_single_barrier(self):
        env = create_pricing_env()
        double = DoubleBarrierOption(
            strike=100.0,
            option_type=OptionType.CALL,
            upper_barrier=115.0,
            lower_barrier=40.0,  # unreachable: ~4.6 sigma below spot
            barrier_type=DoubleBarrierType.KNOCK_OUT,
            maturity=1.0,
            rebate=0.0,
            observation_type=ObservationType.DISCRETE,
            observation_dates=list(self.DATES),
        )
        single = BarrierOption(
            strike=100.0,
            option_type=OptionType.CALL,
            barrier=115.0,
            barrier_type=BarrierType.UP_OUT,
            maturity=1.0,
            rebate=0.0,
            observation_type=ObservationType.DISCRETE,
            observation_dates=list(self.DATES),
        )

        params = PDEParams(grid_size=400, time_steps=200)
        double_price = DoubleBarrierPDESolver(params).price(double, env)
        single_price = BarrierPDESolver(params).price(single, env)

        # The two solvers use different spatial domains, so they carry ~2%
        # relative discretization gap at this resolution (shrinking to ~1% at
        # grid=1600/steps=800). The defect this test guards against produced a
        # 5x collapse (0.48 vs 2.54), so rel=0.03 discriminates cleanly.
        assert double_price == pytest.approx(single_price, rel=0.03)

    def test_double_ko_rebate_discounting_matches_single_barrier_on_steep_curve(self):
        # Audit #6: rebate value at interior steps must use the forward
        # discount factor DF(t, T), not DF(0, tau_remaining).
        curve = LinearRateCurve(pillars=[(0.25, 0.01), (0.5, 0.02), (1.0, 0.06)])
        env = create_pricing_env(vol=0.15, rate_curve=curve)
        double = DoubleBarrierOption(
            strike=1_000_000.0,  # negligible vanilla leg: price is the rebate leg
            option_type=OptionType.CALL,
            upper_barrier=105.0,
            lower_barrier=40.0,
            barrier_type=DoubleBarrierType.KNOCK_OUT,
            maturity=1.0,
            rebate=5.0,
            observation_type=ObservationType.DISCRETE,
            observation_dates=[0.5],
        )
        single = BarrierOption(
            strike=1_000_000.0,
            option_type=OptionType.CALL,
            barrier=105.0,
            barrier_type=BarrierType.UP_OUT,
            maturity=1.0,
            rebate=5.0,
            observation_type=ObservationType.DISCRETE,
            observation_dates=[0.5],
        )

        params = PDEParams(grid_size=400, time_steps=200)
        double_price = DoubleBarrierPDESolver(params).price(double, env)
        single_price = BarrierPDESolver(params).price(single, env)

        assert double_price == pytest.approx(single_price, rel=0.02, abs=0.02)


# ============================================================================
# Audit #1/#2 — Discrete one-touch PDE vs QUAD (independent implementation):
# no barrier forcing between the scheduled dates and no phantom terminal
# observation at maturity.
# ============================================================================


class TestDiscreteOneTouchPDE:
    def test_no_touch_discrete_without_terminal_observation_matches_quad(self):
        env = create_pricing_env()
        option = OneTouchOption(
            barrier=110.0,
            barrier_direction=BarrierDirection.UP,
            maturity=1.0,
            rebate=5.0,
            payment_at_hit=False,
            touch_type=TouchType.NO_TOUCH,
            observation_type=ObservationType.DISCRETE,
            observation_dates=[0.25, 0.5, 0.75],  # no maturity observation
        )

        pde_price = OneTouchPDESolver(PDEParams(grid_size=400, time_steps=200)).price(
            option, env
        )
        quad_price = OneTouchQuadEngine(params=QuadParams(grid_points=801)).price(
            option, env
        )

        assert pde_price == pytest.approx(quad_price, rel=0.02, abs=0.05)

    def test_one_touch_discrete_without_terminal_observation_matches_quad(self):
        env = create_pricing_env()
        option = OneTouchOption(
            barrier=110.0,
            barrier_direction=BarrierDirection.UP,
            maturity=1.0,
            rebate=5.0,
            payment_at_hit=False,
            touch_type=TouchType.ONE_TOUCH,
            observation_type=ObservationType.DISCRETE,
            observation_dates=[0.25, 0.5, 0.75],
        )

        pde_price = OneTouchPDESolver(PDEParams(grid_size=400, time_steps=200)).price(
            option, env
        )
        quad_price = OneTouchQuadEngine(params=QuadParams(grid_points=801)).price(
            option, env
        )

        assert pde_price == pytest.approx(quad_price, rel=0.02, abs=0.05)


# ============================================================================
# Audit #13 — the event-distribution sweep must run the SAME damping schedule
# as the pricing sweep. Pre-fix it ignored event_theta/event_rannacher_steps
# (using rannacher_steps/theta=1.0 instead), so changing those knobs left the
# KO probability bit-identical — proving the stats were computed on a
# different discretization than the npv they decompose.
# ============================================================================


class TestEventStatsThetaConsistency:
    def test_event_theta_knobs_reach_event_distribution_sweep(self):
        from quantark.asset.equity.engine.pde import SnowballPDESolver
        from quantark.asset.equity.product.option.snowball_config import (
            BarrierConfig as SnowballBarrierConfig,
        )
        from quantark.asset.equity.product.option import SnowballOption

        env = create_pricing_env(div=0.0)
        barrier_config = SnowballBarrierConfig(
            ko_barrier=103.0,
            ko_rate=0.12,
            ko_observation_dates=[i / 12 for i in range(1, 13)],
            ki_barrier=75.0,
            ki_continuous=True,
        )
        snowball = SnowballOption(
            initial_price=100.0,
            strike=100.0,
            barrier_config=barrier_config,
            contract_multiplier=1.0,
            maturity=1.0,
        )

        def ko_prob(event_theta: float, event_steps: int) -> np.ndarray:
            solver = SnowballPDESolver(
                PDEParams(
                    grid_size=200,
                    time_steps=100,
                    event_theta=event_theta,
                    event_rannacher_steps=event_steps,
                )
            )
            return np.asarray(
                solver.calculate_event_stats(snowball, env).ko_probability
            )

        prob_default = ko_prob(event_theta=1.0, event_steps=1)
        prob_damped = ko_prob(event_theta=0.5, event_steps=3)

        # The knobs change the discretization, so the (grid-converging)
        # probabilities must respond — bit-identical values mean the event
        # sweep ignored them.
        assert not np.array_equal(prob_default, prob_damped)


# ============================================================================
# Audit #8 — Phoenix memory coupon at KO: accrued (missed) coupons are paid
# only if the current observation's coupon condition is met (MC semantics).
# Constructed so the divergence is structural: coupon_barrier > ko_barrier
# means every KO event lands below the coupon barrier with accrued coupons.
# ============================================================================


class TestPhoenixMemoryCouponAtKO:
    def test_memory_phoenix_ko_below_coupon_barrier_pde_matches_mc(self):
        env = create_pricing_env(div=0.0)
        ko_dates = [i / 12 for i in range(1, 13)]
        # KI configured to never trigger (discrete, negligible barrier) so the
        # test isolates the memory-coupon-at-KO semantics.
        ki_schedule = ObservationSchedule(
            records=[ObservationRecord(observation_time=1.0, barrier=1.0e-6)]
        )
        barrier_config = BarrierConfig(
            ko_barrier=103.0,
            ko_rate=0.0,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=ko_dates,
            ki_barrier=1.0e-6,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_schedule=ki_schedule,
            ki_continuous=False,
        )
        coupon_config = CouponBarrierConfig(
            coupon_barrier=110.0,  # above the KO barrier: KO forfeits accrued coupons
            coupon_rate=0.02,
            coupon_pay_type=CouponPayType.INSTANT,
            day_count_convention=DayCountConvention.ACT_365,
            memory_coupon=True,
        )
        payoff_config = PayoffConfig(
            rebate_rate=0.0,
            include_principal=False,
            participation_rate=1.0,
            protection_type=ProtectionType.NONE,
        )
        phoenix = PhoenixOption(
            initial_price=100.0,
            strike=100.0,
            barrier_config=barrier_config,
            coupon_config=coupon_config,
            payoff_config=payoff_config,
            contract_multiplier=1.0,
            maturity=1.0,
        )

        pde_price = PhoenixPDESolver(
            params=PDEParams(grid_size=200, time_steps=100)
        ).price(phoenix, env)
        mc_price = PhoenixMCEngine(
            params=MCParams(num_paths=65536, seed=42),
            method=MonteCarloMethod.QUASI,
        ).price(phoenix, env)

        assert pde_price == pytest.approx(mc_price, rel=0.02, abs=0.10)
