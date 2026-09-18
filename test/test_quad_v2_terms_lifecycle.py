"""Deterministic term clocks, lifecycle ownership and market invalidation."""
from datetime import timedelta
import math

import numpy as np
import pytest
from scipy.special import ndtr

from test_quad_v2_engine import environment, snowball
from quantark.asset.equity.engine.quad import SnowballQuadEngineV2, PhoenixQuadEngineV2
from quantark.asset.equity.param import QuadV2Params
from quantark.asset.equity.lifecycle import AutocallableLifecycleState
from quantark.asset.equity.lifecycle.cashflows import ValuationPoint
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.snowball_config import (
    BarrierConfig,
    PayoffConfig,
    AccrualConfig,
)
from quantark.param.vol.vol_surface import (
    TermStructureVolSurface,
    ParallelShiftVolSurface,
)
from quantark.param.rrf.rate_curve import LinearRateCurve
from quantark.util.enum import ProtectionType
from quantark.util.exceptions import NumericalError


class PlateauVariance:
    quad_v2_deterministic_variance = True
    exposes_exact_total_variance = True

    def total_variance(self, strike, t, spot):
        return np.interp(
            t, [0.0, 0.2, 0.3, 0.45, 0.7, 1.0], [0.0, 0.0, 0.004, 0.004, 0.014, 0.02]
        )

    def get_vol(self, strike, t, spot):
        return float(np.sqrt(self.total_variance(strike, t, spot) / t)) if t else 0.0


def test_zero_first_and_interior_variance_preserve_vanilla_semigroup():
    p = snowball(times=(0.2, 0.3, 0.45, 0.7, 1.0), ko=1e8)
    p._otc_lifecycle_knocked_in = True
    e = environment()
    e.vol_surface = PlateauVariance()
    got = SnowballQuadEngineV2().calculate_point_greeks(p, e)
    sd = math.sqrt(0.02)
    d1 = (0.03 - 0.01 + 0.5 * 0.02) / sd
    expected = 100 * math.exp(-0.01) * ndtr(-d1) - 100 * math.exp(-0.03) * ndtr(
        -d1 + sd
    )
    assert got["price"] == pytest.approx(expected, abs=2e-8)
    assert got["delta"] == pytest.approx(math.exp(-0.01) * ndtr(-d1), abs=2e-7)


def test_continuous_variance_knots_have_independent_time_change_control():
    p = snowball(
        times=(1.0,),
        continuous=True,
        ko=1e8,
        payoff_config=PayoffConfig(
            rebate_rate=0.12,
            include_principal=False,
            protection_type=ProtectionType.FULL,
        ),
    )
    e = environment(r=0.0, q=0.0)
    e.vol_surface = TermStructureVolSurface(
        times=[0.3, 0.6, 1.0], vols=[0.2, 0.3, 0.25]
    )
    engine = SnowballQuadEngineV2()
    got = engine.price(p, e)
    v = 0.25**2
    m = -v / 2
    d = math.log(100 / 75)
    sd = math.sqrt(v)
    expected = 12 * (ndtr((d + m) / sd) - math.exp(-2 * m * d / v) * ndtr((m - d) / sd))
    assert got == pytest.approx(expected, abs=2e-6)
    assert engine._prepared.diagnostics["events"] == 3
    assert (
        engine._prepared.diagnostics["continuous_model"] == "piecewise_constant_exact"
    )


def test_general_continuous_curves_require_explicit_time_approximation():
    p = snowball(times=(0.25,), continuous=True)
    e = environment()
    e.rate_curve = LinearRateCurve(pillars=[(0.0, 0.01), (0.25, 0.05)])
    with pytest.raises(NotImplementedError, match="time-refinement"):
        SnowballQuadEngineV2().price(p, e)
    params = QuadV2Params(
        continuous_term_structure="piecewise_constant", continuous_steps_per_year=16
    )
    engine = SnowballQuadEngineV2(params)
    value = engine.price(p, e)
    assert np.isfinite(value)
    assert (
        engine._prepared.diagnostics["continuous_model"]
        == "piecewise_constant_approximation"
    )


def test_continuous_event_budget_precedes_subdivision_allocation():
    p = snowball(times=(1.0,), continuous=True)
    e = environment()
    e.rate_curve = LinearRateCurve(pillars=[(0.0, 0.01), (1.0, 0.05)])
    params = QuadV2Params(
        continuous_term_structure="piecewise_constant",
        continuous_steps_per_year=10**12,
        max_events=32,
    )
    with pytest.raises(NumericalError, match="subdivision"):
        SnowballQuadEngineV2(params).price(p, e)


def test_continuous_parallel_flat_vol_shift_is_exact_and_full_risk_runs():
    from quantark.asset.equity.riskmeasures.greeks_calculator import GreeksCalculator

    p = snowball(times=(0.5, 1.0), continuous=True)
    e = environment()
    e.vol_surface = ParallelShiftVolSurface(e.vol_surface, 0.01)
    engine = SnowballQuadEngineV2()
    assert engine.price(p, e) == pytest.approx(
        SnowballQuadEngineV2().price(p, environment(vol=0.21)), abs=1e-10
    )
    assert (
        engine._prepared.diagnostics["continuous_model"] == "piecewise_constant_exact"
    )
    risk = GreeksCalculator(QuadV2Params()).calculate(
        p, e, engine, greeks=["delta", "gamma", "vega", "rho", "dividend_rho", "theta"]
    )
    assert np.isfinite(list(risk.values())).all()


def test_terminated_lifecycle_owns_pending_receivable_and_zero_spot_risk():
    e = environment()
    p = snowball()
    state = AutocallableLifecycleState()
    state.mark_ko(
        e.valuation_date,
        cashflow=105.0,
        settlement_date=e.valuation_date + timedelta(days=10),
        valuation_point=ValuationPoint(date=e.valuation_date),
    )
    engine = SnowballQuadEngineV2()
    result = engine.calculate_point_greeks(p, e, lifecycle_state=state)
    assert result["price"] == pytest.approx(105 * math.exp(-0.03 * 10 / 365), abs=1e-10)
    assert result["delta"] == 0 and result["gamma"] == 0
    components = engine.price_components(p, e, lifecycle_state=state)
    assert components["pending"] == result["price"]
    assert components["terminal"] == 0 and components["ko"] == 0


def test_event_cashflow_ledger_includes_realized_pending_receivables():
    from quantark.asset.equity.lifecycle.cashflows import RealizedCashflow
    from quantark.asset.equity.lifecycle.events import LifecycleEventType

    e = environment()
    p = snowball()
    state = AutocallableLifecycleState(
        valuation_point=ValuationPoint(date=e.valuation_date)
    )
    state.ledger.register(
        RealizedCashflow(
            cashflow_id="fixed-coupon",
            event_type=LifecycleEventType.COUPON,
            amount=2.0,
            determination_date=e.valuation_date,
            payment_date=e.valuation_date + timedelta(days=10),
        )
    )
    engine = SnowballQuadEngineV2()
    stats = engine.calculate_event_stats(p, e, lifecycle_state=state)
    assert sum(stats.expected_discounted_cashflows) == pytest.approx(
        stats.pv, abs=1e-11
    )
    assert stats.pv - engine.price(p, e) == pytest.approx(
        2 * math.exp(-0.03 * 10 / 365), abs=1e-11
    )


def test_explicit_before_after_phase_at_valuation_ko():
    p = snowball(
        times=(0.0, 0.5), accrual_config=AccrualConfig(accrual_factors=[0.25, 0.5])
    )
    e = environment(spot=105.0)
    engine = SnowballQuadEngineV2()
    assert engine.price(p, e, event_phase="before") == pytest.approx(3.0, abs=1e-12)
    assert engine.price(p, e, event_phase="after") != pytest.approx(3.0, abs=1e-4)
    assert engine.calculate_greeks(p, e, event_phase="after")["price"] == pytest.approx(
        engine.price(p, e, event_phase="after"), abs=1e-12
    )
    stats = engine.calculate_event_stats(p, e, event_phase="after")
    assert np.all(stats.ko_times > 0)
    assert stats.pv == pytest.approx(engine.price(p, e, event_phase="after"), abs=1e-12)


def test_aged_fixed_phoenix_coupon_memory_is_paid_once():
    p = PhoenixOption(
        initial_price=100.0,
        strike=100.0,
        contract_multiplier=1.0,
        maturity=0.5,
        barrier_config=BarrierConfig(
            ko_barrier=200.0, ko_rate=0.0, ko_observation_dates=[0.5], ki_barrier=None
        ),
        coupon_config=CouponBarrierConfig(
            coupon_barrier=80.0,
            coupon_rate=0.12,
            memory_coupon=True,
            fixed_coupon_year_fraction=0.25,
        ),
        payoff_config=PayoffConfig(rebate_rate=0.0, include_principal=True),
    )
    state = AutocallableLifecycleState(coupon_memory_count=2)
    result = PhoenixQuadEngineV2().price_components(
        p, environment(vol=0.0, r=0.0, q=0.0), lifecycle_state=state
    )
    assert result["price"] == pytest.approx(109.0, abs=1e-12)
    assert result["coupon"] == pytest.approx(9.0, abs=1e-12)


def test_desk_greeks_rebuild_a_narrow_prepared_range():
    p = snowball()
    e = environment()
    engine = SnowballQuadEngineV2()
    engine.prepare(p, e, [100.0])
    assert np.isfinite(engine.calculate_greeks(p, e)["gamma"])


def test_framework_price_and_event_stats_preserve_lifecycle():
    from quantark.execution.contracts import (
        PricingRequest,
        PricingOperation,
        OutputKind,
    )
    from quantark.execution.context import default_context

    p = snowball(times=(1.0,), ko=1e8)
    env = environment()
    state = AutocallableLifecycleState(knocked_in=True)
    engine = SnowballQuadEngineV2()
    context = default_context(environ={})
    price = engine.execute(PricingRequest(p, env, lifecycle_state=state), context)
    assert price.value == pytest.approx(
        engine.price(p, env, lifecycle_state=state), abs=1e-12
    )
    stats = engine.execute(
        PricingRequest(
            p,
            env,
            operation=PricingOperation.EVENT_STATS,
            outputs=frozenset({OutputKind.PV, OutputKind.EVENT_STATS}),
            lifecycle_state=state,
        ),
        context,
    )
    assert stats.value.ki_probability == pytest.approx(1.0, abs=1e-12)
    assert stats.value.pv == pytest.approx(price.value, abs=1e-12)


def test_framework_event_phase_is_honored_and_unknown_options_rejected():
    from quantark.execution.contracts import PricingRequest
    from quantark.execution.context import default_context
    from quantark.execution.errors import CapabilityError

    p = snowball(
        times=(0.0, 0.5), accrual_config=AccrualConfig(accrual_factors=[0.25, 0.5])
    )
    env = environment(spot=105.0)
    engine = SnowballQuadEngineV2()
    context = default_context(environ={})
    result = engine.execute(
        PricingRequest(p, env, operation_options=(("event_phase", "after"),)), context
    )
    assert result.value == pytest.approx(
        engine.price(p, env, event_phase="after"), abs=1e-12
    )
    with pytest.raises(CapabilityError, match="event_phase operation option"):
        engine.execute(
            PricingRequest(p, env, operation_options=(("ignored", True),)), context
        )
