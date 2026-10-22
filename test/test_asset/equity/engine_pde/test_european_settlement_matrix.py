"""Cross-engine settlement timing contract for European vanilla options."""

from datetime import datetime, timedelta
from math import exp

import pytest

from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.engine.mc import EuropeanMCEngine
from quantark.asset.equity.engine.pde import EuropeanPDESolver
from quantark.asset.equity.engine.pde.grid import GridConfig
from quantark.asset.equity.engine.quad import EuropeanQuadEngine
from quantark.asset.equity.lifecycle import (
    BarrierLifecycleState,
    LifecycleCashflowLedger,
    LifecycleEventType,
    RealizedCashflow,
    ValuationPoint,
)
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.settlement import (
    SettlementConvention,
    SettlementLagUnit,
)
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.param.rrf import LinearRateCurve
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.enum.engine_enums import MonteCarloMethod
from quantark.util.exceptions import ValidationError


VALUATION_DATE = datetime(2026, 7, 30)
MATURITY = 0.25
PAYMENT_LAG = 2.0 / 365.0


def _engine(engine_name):
    if engine_name == "analytical":
        return BlackScholesEngine()
    if engine_name == "mc":
        return EuropeanMCEngine(
            params=MCParams(num_paths=4096, time_steps=8, seed=20260730),
            method=MonteCarloMethod.PSEUDO,
        )
    if engine_name == "pde":
        return EuropeanPDESolver(
            params=PDEParams(
                accuracy="fast",
                grid=GridConfig(points=121, steps_per_day=1.0),
            )
        )
    if engine_name == "quad":
        return EuropeanQuadEngine(params=QuadParams(grid_points=501))
    raise AssertionError(f"unknown test engine {engine_name!r}")


@pytest.fixture
def flat_env():
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.03),
        div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=VALUATION_DATE,
    )


@pytest.fixture
def interpolated_env():
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=LinearRateCurve(
            pillars=[
                (0.0, 0.01),
                (MATURITY, 0.025),
                (MATURITY + PAYMENT_LAG, 0.08),
                (1.0, 0.04),
            ]
        ),
        div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=VALUATION_DATE,
    )


def _option(
    option_type=OptionType.CALL,
    *,
    convention=None,
    exercise_date=None,
    settlement_date=None,
):
    terms = {
        "strike": 100.0,
        "option_type": option_type,
        "settlement_convention": convention,
    }
    if exercise_date is None:
        terms["maturity"] = MATURITY
    else:
        terms["exercise_date"] = exercise_date
        terms["settlement_date"] = settlement_date
    return EuropeanVanillaOption(**terms)


@pytest.mark.parametrize("engine_name", ["analytical", "mc", "pde", "quad"])
def test_numeric_lag_uses_curve_exact_payment_df_without_extending_horizon(
    engine_name, interpolated_env
):
    engine = _engine(engine_name)
    immediate = _option()
    delayed = _option(
        convention=SettlementConvention(
            lag=PAYMENT_LAG,
            lag_unit=SettlementLagUnit.YEAR_FRACTION,
        )
    )

    immediate_pv = engine.price(immediate, interpolated_env)
    delayed_pv = engine.price(delayed, interpolated_env)
    curve_ratio = (
        interpolated_env.get_discount_factor(MATURITY + PAYMENT_LAG)
        / interpolated_env.get_discount_factor(MATURITY)
    )

    assert delayed_pv == pytest.approx(
        immediate_pv * curve_ratio,
        rel=2.0e-12,
        abs=2.0e-12,
    )


@pytest.mark.parametrize("engine_name", ["analytical", "mc", "pde", "quad"])
def test_explicit_terminal_settlement_date_uses_date_payment_time(
    engine_name, flat_env
):
    exercise_date = VALUATION_DATE + timedelta(days=91)
    settlement_date = exercise_date + timedelta(days=2)
    immediate = _option(
        exercise_date=exercise_date,
        settlement_date=exercise_date,
    )
    delayed = _option(
        exercise_date=exercise_date,
        settlement_date=settlement_date,
    )
    engine = _engine(engine_name)

    immediate_pv = engine.price(immediate, flat_env)
    delayed_pv = engine.price(delayed, flat_env)
    td = 91.0 / 365.0
    tp = 93.0 / 365.0
    curve_ratio = (
        flat_env.get_discount_factor(tp)
        / flat_env.get_discount_factor(td)
    )

    assert curve_ratio == pytest.approx(exp(-0.03 * 2.0 / 365.0))
    assert delayed_pv == pytest.approx(
        immediate_pv * curve_ratio,
        rel=2.0e-12,
        abs=2.0e-12,
    )


@pytest.mark.parametrize("engine_name", ["analytical", "mc", "pde", "quad"])
def test_zero_lag_is_legacy_price_identity(engine_name, flat_env):
    engine = _engine(engine_name)
    legacy = _option()
    zero_lag = _option(convention=SettlementConvention())

    assert engine.price(zero_lag, flat_env) == pytest.approx(
        engine.price(legacy, flat_env),
        rel=1.0e-14,
        abs=1.0e-14,
    )


@pytest.mark.parametrize("engine_name", ["analytical", "mc", "pde", "quad"])
def test_delayed_put_call_parity_uses_determination_forward(
    engine_name, flat_env
):
    convention = SettlementConvention(
        lag=PAYMENT_LAG,
        lag_unit=SettlementLagUnit.YEAR_FRACTION,
    )
    engine = _engine(engine_name)
    call = _option(OptionType.CALL, convention=convention)
    put = _option(OptionType.PUT, convention=convention)

    lhs = engine.price(call, flat_env) - engine.price(put, flat_env)
    determination_df = flat_env.get_discount_factor(MATURITY)
    payment_df = flat_env.get_discount_factor(MATURITY + PAYMENT_LAG)
    if engine_name == "mc":
        immediate_parity = (
            engine.price(_option(OptionType.CALL), flat_env)
            - engine.price(_option(OptionType.PUT), flat_env)
        )
        rhs = immediate_parity * payment_df / determination_df
        tolerance = 2.0e-12
    else:
        determination_forward = (
            flat_env.spot
            * exp(-0.01 * MATURITY)
            / determination_df
        )
        rhs = payment_df * (determination_forward - call.strike)
        tolerance = 3.0e-2 if engine_name in {"pde", "quad"} else 2.0e-12
    assert lhs == pytest.approx(rhs, abs=tolerance)


def _pending_state(*, terminal):
    cashflow = RealizedCashflow(
        cashflow_id="european:expiry",
        event_type=LifecycleEventType.EXPIRY,
        amount=12.5,
        determination_date=VALUATION_DATE - timedelta(days=1),
        payment_date=VALUATION_DATE + timedelta(days=2),
    )
    return BarrierLifecycleState(
        alive=not terminal,
        expired=terminal,
        valuation_point=ValuationPoint(date=VALUATION_DATE),
        ledger=LifecycleCashflowLedger([cashflow]),
    )


@pytest.mark.parametrize("engine_name", ["analytical", "mc", "pde", "quad"])
def test_terminal_lifecycle_returns_only_authoritative_pending_cashflow(
    engine_name, flat_env
):
    state = _pending_state(terminal=True)
    expected = 12.5 * flat_env.get_discount_factor(2.0 / 365.0)

    assert _engine(engine_name).price(
        _option(),
        flat_env,
        lifecycle_state=state,
    ) == pytest.approx(expected)


@pytest.mark.parametrize("engine_name", ["analytical", "mc", "pde", "quad"])
def test_terminal_lifecycle_without_authoritative_cashflow_fails_closed(
    engine_name, flat_env
):
    state = BarrierLifecycleState(
        alive=False,
        expired=True,
        valuation_point=ValuationPoint(date=VALUATION_DATE),
    )

    with pytest.raises(ValidationError, match="realized cashflow"):
        _engine(engine_name).price(
            _option(),
            flat_env,
            lifecycle_state=state,
        )


def test_live_lifecycle_adds_earlier_pending_receivable(flat_env):
    engine = BlackScholesEngine()
    product = _option()
    contingent = engine.price(product, flat_env)

    with_pending = engine.price(
        product,
        flat_env,
        lifecycle_state=_pending_state(terminal=False),
    )

    assert with_pending == pytest.approx(
        contingent + 12.5 * flat_env.get_discount_factor(2.0 / 365.0)
    )


def test_terminal_lifecycle_greeks_are_fixed_receivable_greeks(flat_env):
    greeks = EuropeanPDESolver(
        params=PDEParams(
            accuracy="fast",
            grid=GridConfig(points=121, steps_per_day=1.0),
        )
    ).calculate_greeks(
        _option(),
        flat_env,
        lifecycle_state=_pending_state(terminal=True),
    )

    assert greeks == pytest.approx(
        {
            "price": 12.5 * flat_env.get_discount_factor(2.0 / 365.0),
            "delta": 0.0,
            "gamma": 0.0,
        }
    )
