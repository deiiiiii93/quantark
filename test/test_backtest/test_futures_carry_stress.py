"""Unspanned carry risk: tail, interpolation shape and joint finite shocks.

The defining property of these scenarios is that spot and every listed quote
are unchanged, so a futures hedge cannot respond at all.  A product that
still moves is carrying risk the nodal rhoq column says nothing about.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pytest

from bucket_hedge_fixtures import (
    AnalyticQuote,
    flat_forward_carry_forward,
    linear_book_pricer,
    squared_far_future_price,
)
from quantark.asset.equity.market import IndexFuturesQuote
from quantark.backtest.replay.carry_context import CarryCurveContext
from quantark.backtest.replay.carry_stress import (
    SCENARIO_FAMILIES,
    CarryScenario,
    ShapeStressedDividendYield,
    TailStressedDividendYield,
    run_scenario,
    shape_scenario,
    stressed_dividend,
    tail_scenario,
)
from quantark.param import FlatRateCurve
from quantark.util.exceptions import ValidationError

RATE = 0.0
SPOT = 100.0
TENORS = (0.25, 0.50)
PRICES = (100.0, 100.0)
CONTRACTS = ("IF2503", "IF2506")
LAST_TENOR = TENORS[-1]


def context(extrapolation: str = "flat_forward_carry"):
    return CarryCurveContext(
        quotes=tuple(
            IndexFuturesQuote(contract=c, maturity=t, price=p, multiplier=1.0)
            for c, t, p in zip(CONTRACTS, TENORS, PRICES)
        ),
        spot=SPOT,
        rate_curve=FlatRateCurve(rate=RATE),
        extrapolation=extrapolation,
        underlying="index",
        valuation_date=datetime(2025, 3, 3),
    )


def forward_of(dividend, time: float, spot: float = SPOT) -> float:
    return spot * math.exp((RATE - float(dividend.get_yield(time))) * time)


def forward_claim_pricer(maturity: float, notional: float = 1.0):
    """PV of ``notional * S_T``: exposure to the curve at exactly one tenor."""

    def price_at(spot, dividend):
        return (
            notional
            * math.exp(-RATE * maturity)
            * float(spot)
            * math.exp((RATE - float(dividend.get_yield(maturity))) * maturity)
        )

    return price_at


# ---------------------------------------------------------------------------
# The independent tail stress
# ---------------------------------------------------------------------------


def test_the_tail_stress_leaves_spot_and_every_listed_quote_alone():
    ctx = context()
    scenario = tail_scenario(0.01)
    moved, dividend = stressed_dividend(ctx, scenario)
    assert moved.spot == ctx.spot
    assert moved.prices == ctx.prices
    # ... and every listed tenor still reproduces its own quote.
    for quote in ctx.quotes:
        assert forward_of(dividend, quote.maturity) == pytest.approx(
            quote.price, rel=1e-12
        )


def test_the_tail_stress_moves_the_one_year_forward_by_the_design_factor():
    ctx = context()
    _, dividend = stressed_dividend(ctx, tail_scenario(-0.01))
    base = forward_of(ctx.dividend(), 1.0)
    stressed = forward_of(dividend, 1.0)
    # F -> F exp(-lambda (T - T_n)) with lambda = -0.01, T = 1, T_n = 0.5.
    assert stressed / base - 1.0 == pytest.approx(math.expm1(-(-0.01) * 0.5) - 0.0)
    assert stressed / base == pytest.approx(math.exp(0.005))


def test_a_one_year_claim_loses_49_8752_bp_to_a_one_point_tail_shift():
    ctx = context()
    price_at = forward_claim_pricer(1.0, notional=1_000_000.0)
    result = run_scenario(price_at, ctx, tail_scenario(0.01), {})
    base = price_at(ctx.spot, ctx.dividend())
    assert result.status == "ok"
    assert 10_000.0 * result.product_pnl / base == pytest.approx(-49.8752, abs=1e-3)
    # dV/dlambda = -(T - T_n) V to first order.
    assert result.product_pnl / base == pytest.approx(
        math.expm1(-0.01 * 0.5), rel=1e-12
    )


def test_no_futures_hedge_can_respond_to_the_tail_stress():
    ctx = context()
    scenario = tail_scenario(0.01)
    assert not scenario.moves_listed_quotes
    for holdings in ({}, {CONTRACTS[0]: -3.0, CONTRACTS[1]: 1.5}, {CONTRACTS[1]: 99.0}):
        result = run_scenario(
            forward_claim_pricer(1.0, notional=1_000_000.0), ctx, scenario, holdings
        )
        assert result.hedge_pnl == 0.0
        assert result.book_pnl == pytest.approx(result.product_pnl)


def test_the_tail_extra_yield_is_zero_at_the_origin_and_at_every_anchor():
    base = context().dividend()
    stressed = TailStressedDividendYield(base, last_tenor=LAST_TENOR, rate_shift=0.01)
    for t in (1e-9, 0.1, 0.25, 0.5):
        assert stressed.get_yield(t) == pytest.approx(base.get_yield(t), abs=1e-12)
    assert stressed.get_yield(1.0) - base.get_yield(1.0) == pytest.approx(
        0.01 * 0.5 / 1.0
    )


# ---------------------------------------------------------------------------
# The interpolation-shape stress
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("interval", [(0.0, 0.25), (0.25, 0.5)])
def test_the_shape_stress_vanishes_at_both_anchors_and_peaks_in_the_middle(interval):
    ctx = context()
    epsilon = 0.01
    moved, dividend = stressed_dividend(
        ctx, shape_scenario(epsilon, interval)
    )
    assert moved.prices == ctx.prices
    start, end = interval
    base = ctx.dividend()
    for anchor in (start, end):
        if anchor > 0.0:
            assert forward_of(dividend, anchor) == pytest.approx(
                forward_of(base, anchor), rel=1e-12
            )
    middle = 0.5 * (start + end)
    ratio = forward_of(dividend, middle) / forward_of(base, middle)
    assert math.log(ratio) == pytest.approx(epsilon, rel=1e-12)


def test_the_first_interval_shape_stress_has_a_finite_yield_at_zero():
    """``L(t)/t`` would divide by zero; its limit ``4 eps / B`` does not."""
    base = context().dividend()
    epsilon = 0.01
    stressed = ShapeStressedDividendYield(
        base, start=0.0, end=0.25, log_forward_shift=epsilon
    )
    # The array path is the one grid samplers use, and it must not produce a
    # warning, an inf or a NaN at the origin.
    with np.errstate(divide="raise", invalid="raise"):
        values = stressed.get_yield(np.array([0.0, 1e-12, 1e-8, 0.1]))
    assert np.all(np.isfinite(values))
    limit = -4.0 * epsilon / 0.25
    assert values[0] == pytest.approx(limit, rel=1e-12)
    for t in (1e-8, 1e-6, 1e-4):
        assert stressed.get_yield(t) - base.get_yield(t) == pytest.approx(
            limit, rel=1e-3
        )
    # A scalar call at the origin keeps the BASE curve's own contract; the
    # carry-implied view has always rejected T <= 0 there.
    with pytest.raises(ValidationError):
        stressed.get_yield(0.0)


def test_the_shape_stress_is_zero_outside_its_interval():
    base = context().dividend()
    stressed = ShapeStressedDividendYield(
        base, start=0.25, end=0.5, log_forward_shift=0.01
    )
    for t in (0.1, 0.25, 0.5, 0.9):
        assert stressed.get_yield(t) == pytest.approx(base.get_yield(t), abs=1e-14)


def test_a_shape_stress_a_resampling_builder_would_erase_still_moves_the_price():
    ctx = context()
    price_at = forward_claim_pricer(0.375, notional=1_000_000.0)
    result = run_scenario(price_at, ctx, shape_scenario(0.01, (0.25, 0.5)), {})
    assert result.status == "ok"
    assert result.hedge_pnl == 0.0
    # Every anchor is unchanged, so re-fitting the quotes would report zero.
    assert result.stressed_prices == result.base_prices
    assert result.product_pnl != pytest.approx(0.0, abs=1.0)


# ---------------------------------------------------------------------------
# Scalar and array evaluation agree
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stress",
    [
        lambda base: TailStressedDividendYield(base, LAST_TENOR, 0.01),
        lambda base: ShapeStressedDividendYield(base, 0.0, 0.25, 0.01),
        lambda base: ShapeStressedDividendYield(base, 0.25, 0.5, -0.01),
    ],
)
def test_scalar_and_vector_calls_agree_including_at_zero(stress):
    stressed = stress(context().dividend())
    times = np.array([0.0, 1e-6, 0.1, 0.25, 0.375, 0.5, 1.0, 2.0])
    vector = stressed.get_yield(times)
    assert np.all(np.isfinite(vector))
    for index, t in enumerate(times):
        if t == 0.0:
            continue
        assert vector[index] == pytest.approx(stressed.get_yield(float(t)), abs=1e-12)


def test_a_stressed_curve_still_supports_a_parallel_shift():
    stressed = TailStressedDividendYield(context().dividend(), LAST_TENOR, 0.01)
    shifted = stressed.parallel_shifted(1e-4)
    for t in (0.1, 0.5, 1.0):
        assert shifted.get_yield(t) - stressed.get_yield(t) == pytest.approx(1e-4)


# ---------------------------------------------------------------------------
# Joint finite shocks and the curvature label
# ---------------------------------------------------------------------------


def test_a_joint_shock_names_every_move_once():
    ctx = context()
    scenario = CarryScenario(
        scenario_id="joint",
        family="joint_spot_carry",
        spot_shift_rel=0.02,
        parallel_yield_shift=0.005,
    )
    moved, _ = stressed_dividend(ctx, scenario)
    assert moved.spot == pytest.approx(SPOT * 1.02)
    for quote in ctx.quotes:
        assert moved.prices[quote.contract] == pytest.approx(
            quote.price * math.exp(-quote.maturity * 0.005)
        )
    assert scenario.moves_listed_quotes


def test_a_joint_shock_is_explained_by_the_pinned_spot_and_listed_forwards():
    """Frozen-carry spot neutrality is NOT zero P&L when the quotes are pinned.

    The primary-policy book has ``D_book = 0`` and ``R_parallel = 0``, yet a
    joint spot-and-carry move still pays ``D_F * dS``.  The coordinate the
    P&L actually decomposes in is ``(S pinned, listed F)``, which is exactly
    why design section 9.3 attributes to those two terms and not to the
    frozen-carry delta.
    """
    ctx = context()
    coefficients = (1.0, -0.5)
    price_at = linear_book_pricer(
        tenors=TENORS, futures_coefficients=coefficients, rate=RATE
    )
    holdings = {CONTRACTS[0]: -3.0, CONTRACTS[1]: 1.5}
    scenario = CarryScenario(
        scenario_id="joint-small",
        family="joint_spot_carry",
        spot_shift_rel=1e-4,
        parallel_yield_shift=1e-5,
    )
    result = run_scenario(price_at, ctx, scenario, holdings)
    assert result.status == "ok"

    delta_f = 1.0  # the spot coefficient: every listed quote is pinned
    spot_change = result.stressed_spot - result.spot
    linear_spot_pinned = delta_f * spot_change
    linear_listed_forwards = sum(
        (coefficient + holdings[contract] * 1.0)
        * (result.stressed_prices[contract] - result.base_prices[contract])
        for coefficient, contract in zip(coefficients, CONTRACTS)
    )
    assert result.book_pnl == pytest.approx(
        linear_spot_pinned + linear_listed_forwards, rel=1e-12
    )
    # The carry leg cancels to FIRST order for this book. What is left is
    # the second order of exp(-T eps), not zero, and it is nine orders below
    # the pinned-spot leg that survives.
    second_order = -2.0 * PRICES[0] * math.expm1(-TENORS[0] * 1e-5) + 1.0 * PRICES[
        1
    ] * math.expm1(-TENORS[1] * 1e-5)
    assert linear_listed_forwards == pytest.approx(second_order, rel=1e-9)
    assert abs(linear_listed_forwards) < 1e-8
    assert result.book_pnl == pytest.approx(SPOT * 1e-4, rel=1e-6)
    assert result.hedge_pnl == pytest.approx(0.0, abs=1e-9)


def test_a_squared_future_residual_is_carry_curvature_not_spot_gamma():
    ctx = context()

    def price_at(spot, dividend):
        forward = forward_of(dividend, LAST_TENOR, spot=spot)
        return forward * forward

    scenario = CarryScenario(
        scenario_id="carry-curvature",
        family="parallel_carry",
        parallel_yield_shift=0.01,
    )
    # Spot never moves, so a spot-gamma term would be identically zero.
    assert scenario.spot_shift_rel == 0.0
    base_far = PRICES[-1]
    linear = -2.0 * base_far * base_far * LAST_TENOR * 0.01
    result = run_scenario(price_at, ctx, scenario, {}, linear_prediction=linear)
    assert result.status == "ok"
    assert result.stressed_spot == pytest.approx(SPOT)
    exact = (base_far * math.exp(-LAST_TENOR * 0.01)) ** 2 - base_far**2
    assert result.product_pnl == pytest.approx(exact, rel=1e-12)
    assert result.repricing_error == pytest.approx(exact - linear, rel=1e-9)
    assert result.repricing_error != pytest.approx(0.0, abs=1e-6)
    assert squared_far_future_price(
        (AnalyticQuote(CONTRACTS[1], LAST_TENOR, base_far),)
    ) == pytest.approx(base_far**2)


# ---------------------------------------------------------------------------
# Definitions, records and fail-closed behaviour
# ---------------------------------------------------------------------------


def test_scenario_records_carry_their_units_and_definition():
    scenario = tail_scenario(0.01, scenario_id="tail-up")
    metadata = scenario.metadata()
    assert metadata["scenario_id"] == "tail-up"
    assert metadata["family"] == "independent_tail"
    assert metadata["tail_rate_shift"] == 0.01
    assert metadata["yield_units"] == "absolute decimal annual yield"
    assert metadata["log_forward_units"] == "absolute log forward displacement"
    assert metadata["finite"] is True
    assert set(SCENARIO_FAMILIES) >= {
        "independent_tail",
        "interpolation_shape",
        "joint_spot_carry",
    }


def test_a_result_keeps_both_quote_maps_and_the_holdings_kind():
    ctx = context()
    result = run_scenario(
        forward_claim_pricer(1.0), ctx, tail_scenario(0.01), {}, holdings_kind="ideal"
    )
    assert result.holdings_kind == "ideal"
    assert result.base_prices == ctx.prices
    assert result.stressed_prices == ctx.prices
    assert result.spot == SPOT
    assert result.stressed_spot == SPOT


def test_an_unsupported_scenario_is_a_failed_record_not_zero_risk():
    ctx = context()

    def price_at(spot, dividend):
        raise ValidationError("engine cannot price this state")

    result = run_scenario(price_at, ctx, tail_scenario(0.01), {})
    assert result.status == "failed"
    assert "cannot price" in result.reason
    assert math.isnan(result.product_pnl)
    assert math.isnan(result.book_pnl)


def test_a_non_finite_scenario_price_is_a_failed_record():
    ctx = context()
    result = run_scenario(
        lambda spot, dividend: float("nan"), ctx, tail_scenario(0.01), {}
    )
    assert result.status == "failed"
    assert math.isnan(result.book_pnl)


def test_invalid_scenario_definitions_are_rejected():
    with pytest.raises(ValidationError):
        CarryScenario(scenario_id="", family="independent_tail")
    with pytest.raises(ValidationError):
        CarryScenario(scenario_id="x", family="vega")
    with pytest.raises(ValidationError):
        CarryScenario(scenario_id="x", family="spot", spot_shift_rel=float("nan"))
    with pytest.raises(ValidationError):
        CarryScenario(scenario_id="x", family="spot", spot_shift_rel=-1.0)
    with pytest.raises(ValidationError):
        CarryScenario(
            scenario_id="x",
            family="interpolation_shape",
            shape_interval=(0.5, 0.25),
        )
    with pytest.raises(ValidationError):
        ShapeStressedDividendYield(
            context().dividend(), start=0.5, end=0.5, log_forward_shift=0.01
        )
    with pytest.raises(ValidationError):
        TailStressedDividendYield(
            context().dividend(), last_tenor=0.0, rate_shift=0.01
        )
    with pytest.raises(ValidationError):
        run_scenario(
            forward_claim_pricer(1.0),
            context(),
            tail_scenario(0.01),
            {},
            holdings_kind="target",
        )


def test_the_tail_formula_matches_the_independent_closed_form():
    ctx = context()
    analytic = tuple(
        AnalyticQuote(c, t, p, 1.0) for c, t, p in zip(CONTRACTS, TENORS, PRICES)
    )
    _, dividend = stressed_dividend(ctx, tail_scenario(0.01))
    for t in (0.75, 1.0, 2.0):
        base = flat_forward_carry_forward(SPOT, analytic, t)
        assert forward_of(dividend, t) == pytest.approx(
            base * math.exp(-0.01 * max(t - LAST_TENOR, 0.0)), rel=1e-12
        )
