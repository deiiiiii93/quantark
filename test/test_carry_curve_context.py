"""Immutable carry scenarios: builder parity, invariances and fail-closed input.

The two invariances of revised design section 2.2 are what license every
policy formula, so they are tested here on the first interval, an interior
interval, a node, the tail and the one-node limit -- under BOTH supported
conventions.
"""

from __future__ import annotations

import math
from datetime import datetime

import pytest

from bucket_hedge_fixtures import (
    AnalyticQuote,
    flat_forward_carry_forward,
    flat_q_forward,
)
from quantark.asset.equity.market import IndexFuturesQuote
from quantark.backtest.replay.carry_context import (
    SUPPORTED_EXTRAPOLATIONS,
    CarryCurveContext,
)
from quantark.backtest.replay.dividend_source import term_dividend_yield
from quantark.param import FlatRateCurve
from quantark.util.exceptions import ValidationError

RATE = 0.02
SPOT = 4700.0
VALUATION = datetime(2025, 3, 3)

QUOTES = (
    IndexFuturesQuote(
        contract="IF2503",
        maturity=0.25,
        price=4680.0,
        multiplier=200.0,
        expiry_date=datetime(2025, 6, 1),
    ),
    IndexFuturesQuote(
        contract="IF2506",
        maturity=0.50,
        price=4655.0,
        multiplier=200.0,
        expiry_date=datetime(2025, 9, 1),
    ),
    IndexFuturesQuote(
        contract="IF2509",
        maturity=0.75,
        price=4610.0,
        multiplier=200.0,
        expiry_date=datetime(2025, 12, 1),
    ),
)

# One tenor before the first node, one strictly inside an interval, the nodes
# themselves and one tail time.
SAMPLE_TIMES = (0.05, 0.1, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0, 1.5)


def make_context(
    *, extrapolation: str = "flat_q", quotes=QUOTES, spot: float = SPOT
) -> CarryCurveContext:
    return CarryCurveContext(
        quotes=quotes,
        spot=spot,
        rate_curve=FlatRateCurve(rate=RATE),
        extrapolation=extrapolation,
        underlying="index",
        valuation_date=VALUATION,
    )


def analytic_quotes(quotes=QUOTES):
    return tuple(
        AnalyticQuote(q.contract, q.maturity, q.price, q.multiplier) for q in quotes
    )


def forward_of(context: CarryCurveContext, time: float) -> float:
    """``F(t) = S exp((r - q(t)) t)`` read back from the built dividend."""
    q = float(context.dividend().get_yield(time))
    return context.spot * math.exp((RATE - q) * time)


# ---------------------------------------------------------------------------
# The context builds exactly what the shared builder builds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("extrapolation", SUPPORTED_EXTRAPOLATIONS)
@pytest.mark.parametrize("time", SAMPLE_TIMES)
def test_base_dividend_matches_the_shared_builder(extrapolation, time):
    context = make_context(extrapolation=extrapolation)
    expected = term_dividend_yield(
        QUOTES,
        spot=SPOT,
        rate_curve=FlatRateCurve(rate=RATE),
        extrapolation=extrapolation,
        underlying="index",
    )
    assert context.dividend().get_yield(time) == pytest.approx(
        expected.get_yield(time), rel=0, abs=0
    )


@pytest.mark.parametrize("time", SAMPLE_TIMES)
def test_flat_q_forward_matches_the_independent_closed_form(time):
    context = make_context(extrapolation="flat_q")
    reference = flat_q_forward(SPOT, analytic_quotes(), time, RATE)
    assert forward_of(context, time) == pytest.approx(reference, rel=1e-13)


@pytest.mark.parametrize("time", SAMPLE_TIMES)
def test_flat_forward_carry_forward_matches_the_independent_closed_form(time):
    context = make_context(extrapolation="flat_forward_carry")
    reference = flat_forward_carry_forward(SPOT, analytic_quotes(), time)
    assert forward_of(context, time) == pytest.approx(reference, rel=1e-13)


def test_the_conventions_agree_on_the_first_interval_and_differ_in_the_tail():
    flat_q = make_context(extrapolation="flat_q")
    carry = make_context(extrapolation="flat_forward_carry")
    assert forward_of(flat_q, 0.1) == pytest.approx(forward_of(carry, 0.1), rel=1e-13)
    assert forward_of(flat_q, 1.5) != pytest.approx(forward_of(carry, 1.5), rel=1e-6)


@pytest.mark.parametrize("extrapolation", SUPPORTED_EXTRAPOLATIONS)
def test_quotes_reproduce_their_own_nodes(extrapolation):
    context = make_context(extrapolation=extrapolation)
    for quote in QUOTES:
        assert forward_of(context, quote.maturity) == pytest.approx(
            quote.price, rel=1e-12
        )


# ---------------------------------------------------------------------------
# Invariance 1: proportional scaling leaves the zero curve unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("extrapolation", SUPPORTED_EXTRAPOLATIONS)
@pytest.mark.parametrize("node_count", [1, 2, 3])
def test_proportional_spot_and_quote_scaling_preserves_the_zero_curve(
    extrapolation, node_count
):
    scale = 1.037
    base = make_context(extrapolation=extrapolation, quotes=QUOTES[:node_count])
    from dataclasses import replace as dc_replace

    scaled = CarryCurveContext(
        quotes=tuple(dc_replace(q, price=q.price * scale) for q in base.quotes),
        spot=base.spot * scale,
        rate_curve=base.rate_curve,
        extrapolation=extrapolation,
        underlying=base.underlying,
        valuation_date=base.valuation_date,
    )
    for time in SAMPLE_TIMES:
        assert scaled.dividend().get_yield(time) == pytest.approx(
            base.dividend().get_yield(time), abs=1e-13
        )


# ---------------------------------------------------------------------------
# Invariance 2: the exponential quote transformation shifts the whole curve
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("extrapolation", SUPPORTED_EXTRAPOLATIONS)
@pytest.mark.parametrize("node_count", [1, 2, 3])
@pytest.mark.parametrize("shift", [1e-4, -1e-4, 0.01])
def test_parallel_yield_shift_moves_every_pricing_time_by_the_shift(
    extrapolation, node_count, shift
):
    base = make_context(extrapolation=extrapolation, quotes=QUOTES[:node_count])
    shifted = base.parallel_yield_shift(shift)
    for time in SAMPLE_TIMES:
        assert shifted.dividend().get_yield(time) - base.dividend().get_yield(
            time
        ) == pytest.approx(shift, abs=1e-12)


@pytest.mark.parametrize("extrapolation", SUPPORTED_EXTRAPOLATIONS)
def test_parallel_shift_is_the_sum_of_the_node_shifts_on_the_quotes(extrapolation):
    base = make_context(extrapolation=extrapolation)
    shift = 3e-4
    parallel = base.parallel_yield_shift(shift)
    stacked = base
    for contract in base.contracts:
        stacked = stacked.bump_node_yield(contract, shift)
    assert parallel.prices == pytest.approx(stacked.prices)


# ---------------------------------------------------------------------------
# One-coordinate scenarios touch exactly one coordinate
# ---------------------------------------------------------------------------


def test_node_yield_bump_moves_exactly_one_quote_by_the_exponential_factor():
    base = make_context()
    shift = 2.5e-4
    bumped = base.bump_node_yield("IF2506", shift)
    target = base.quote("IF2506")
    assert bumped.prices["IF2506"] == pytest.approx(
        target.price * math.exp(-target.maturity * shift), rel=0, abs=0
    )
    assert bumped.prices["IF2503"] == base.prices["IF2503"]
    assert bumped.prices["IF2509"] == base.prices["IF2509"]
    assert bumped.spot == base.spot


def test_node_yield_bump_moves_that_node_by_the_shift():
    base = make_context()
    shift = 1e-4
    bumped = base.bump_node_yield("IF2506", shift)
    assert bumped.implied_yield("IF2506") - base.implied_yield(
        "IF2506"
    ) == pytest.approx(shift, abs=1e-12)
    assert bumped.implied_yield("IF2503") == pytest.approx(
        base.implied_yield("IF2503"), abs=1e-15
    )


def test_spot_bump_pins_every_listed_price():
    base = make_context()
    moved = base.with_spot(SPOT * 1.01)
    assert moved.prices == base.prices
    assert moved.spot == pytest.approx(SPOT * 1.01)
    # Pinning the quotes is exactly what makes the zero curve move.
    assert moved.dividend().get_yield(0.25) != pytest.approx(
        base.dividend().get_yield(0.25), abs=1e-9
    )


def test_futures_bump_adds_points_to_one_quote_at_fixed_spot():
    base = make_context()
    bumped = base.bump_future("IF2503", 1.0)
    assert bumped.prices["IF2503"] == pytest.approx(base.prices["IF2503"] + 1.0)
    assert bumped.prices["IF2506"] == base.prices["IF2506"]
    assert bumped.spot == base.spot


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


def test_scenarios_never_mutate_the_original_context():
    quotes = list(QUOTES)
    rate_curve = FlatRateCurve(rate=RATE)
    base = CarryCurveContext(
        quotes=quotes,
        spot=SPOT,
        rate_curve=rate_curve,
        extrapolation="flat_q",
        underlying="index",
        valuation_date=VALUATION,
    )
    before = base.coordinates()
    base.with_spot(SPOT * 1.05)
    base.bump_future("IF2503", 5.0)
    base.bump_node_yield("IF2503", 0.01)
    base.parallel_yield_shift(0.01)
    quotes.clear()
    assert base.coordinates() == before
    assert base.spot == SPOT
    assert base.rate_curve is rate_curve
    assert rate_curve.get_rate(0.25) == pytest.approx(RATE)


def test_scenarios_preserve_quote_metadata():
    bumped = make_context().bump_node_yield("IF2503", 1e-3)
    quote = bumped.quote("IF2503")
    assert quote.expiry_date == datetime(2025, 6, 1)
    assert quote.multiplier == 200.0
    assert quote.maturity == 0.25


def test_context_is_frozen():
    context = make_context()
    with pytest.raises(Exception):
        context.spot = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Fail-closed input
# ---------------------------------------------------------------------------


def test_unsupported_conventions_are_rejected():
    for extrapolation in ("surface_forward_carry", "surface_forwards", "flat", ""):
        with pytest.raises(ValidationError):
            make_context(extrapolation=extrapolation)


def test_duplicate_identifiers_and_repeated_tenors_are_rejected():
    from dataclasses import replace as dc_replace

    duplicate = (QUOTES[0], dc_replace(QUOTES[1], contract="IF2503"))
    with pytest.raises(ValidationError):
        make_context(quotes=duplicate)
    repeated = (QUOTES[0], dc_replace(QUOTES[1], maturity=0.25))
    with pytest.raises(ValidationError):
        make_context(quotes=repeated)
    with pytest.raises(ValidationError):
        make_context(quotes=(QUOTES[1], QUOTES[0]))


def test_nonfinite_spot_and_empty_universe_are_rejected():
    for spot in (float("nan"), float("inf"), 0.0, -1.0):
        with pytest.raises(ValidationError):
            make_context(spot=spot)
    with pytest.raises(ValidationError):
        make_context(quotes=())


def test_unknown_contracts_are_rejected():
    base = make_context()
    with pytest.raises(ValidationError):
        base.bump_future("IF2512", 1.0)
    with pytest.raises(ValidationError):
        base.bump_node_yield("IF2512", 1e-4)
    with pytest.raises(ValidationError):
        base.quote("IF2512")


def test_nonfinite_scenario_arguments_are_rejected():
    base = make_context()
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValidationError):
            base.bump_future("IF2503", bad)
        with pytest.raises(ValidationError):
            base.bump_node_yield("IF2503", bad)
        with pytest.raises(ValidationError):
            base.parallel_yield_shift(bad)
    with pytest.raises(ValidationError):
        base.bump_future("IF2503", 0.0)


def test_a_scenario_price_that_goes_non_positive_is_rejected():
    base = make_context()
    with pytest.raises(ValidationError):
        base.bump_future("IF2503", -base.prices["IF2503"])
    with pytest.raises(ValidationError):
        base.with_spot(-1.0)


def test_a_rate_curve_without_get_rate_is_rejected():
    with pytest.raises(ValidationError):
        CarryCurveContext(
            quotes=QUOTES,
            spot=SPOT,
            rate_curve=object(),
            extrapolation="flat_q",
            underlying="index",
            valuation_date=VALUATION,
        )


def test_coordinate_comparison_notices_every_axis():
    base = make_context()
    assert base.same_coordinates_as(make_context())
    assert not base.same_coordinates_as(base.with_spot(SPOT + 1.0))
    assert not base.same_coordinates_as(base.bump_future("IF2503", 1.0))
    assert not base.same_coordinates_as(
        make_context(extrapolation="flat_forward_carry")
    )
    assert not base.same_coordinates_as(make_context(quotes=QUOTES[:2]))
