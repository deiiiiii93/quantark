"""IndexFuturesCurve: implied carry from futures marks (spec tests 1, 2, 9)."""
import math

import pytest

from quantark.util.enum import EquityDividendInputMode, FuturesCarryRiskMode
from quantark.util.exceptions import ValidationError


def test_futures_carry_risk_mode_values():
    assert FuturesCarryRiskMode.MARKET_PRICE.value == "market_price"
    assert FuturesCarryRiskMode.THEORETICAL_CARRY.value == "theoretical_carry"
    assert FuturesCarryRiskMode.IMPLIED_FUTURES_CARRY.value == "implied_futures_carry"


def test_equity_dividend_input_mode_values():
    assert EquityDividendInputMode.FLAT_DIVIDEND.value == "flat_dividend"
    assert EquityDividendInputMode.TERM_DIVIDEND.value == "term_dividend"


from quantark.asset.equity.market import (
    IndexFuturesCurve,
    IndexFuturesQuote,
    bump_term_yield_node,
    hedge_hands,
)
from quantark.param import FlatRateCurve
from quantark.param.div.dividend_yield import TermStructureDividendYield


def _quotes_from_q(spot, r, times, qs, multiplier=200.0):
    """Generate marks from known q(T): F = S * exp((r - q) * T)."""
    return [
        IndexFuturesQuote(
            contract=f"IC{i:02d}",
            maturity=t,
            price=spot * math.exp((r - q) * t),
            multiplier=multiplier,
        )
        for i, (t, q) in enumerate(zip(times, qs))
    ]


def _curve(spot=100.0, r=0.03, times=(0.25, 0.5, 1.0), qs=(0.01, 0.02, 0.015)):
    return IndexFuturesCurve(
        underlying="IC", spot=spot, quotes=_quotes_from_q(spot, r, list(times), list(qs))
    )


# --- spec test 1: implied carry recovers known q(T) ---

def test_implied_yields_recover_known_carry():
    times, qs = [0.25, 0.5, 1.0], [0.01, 0.02, 0.015]
    curve = _curve(times=times, qs=qs)
    implied = curve.implied_yields(FlatRateCurve(0.03))
    assert implied == pytest.approx(qs, abs=1e-12)


def test_to_dividend_yield_curve_nodes():
    curve = _curve()
    term = curve.to_dividend_yield_curve(FlatRateCurve(0.03))
    assert isinstance(term, TermStructureDividendYield)
    assert term.times == [0.25, 0.5, 1.0]


def test_negative_implied_carry_contango_marks():
    # spec demo marks: contango => negative implied q, must construct fine
    quotes = [
        IndexFuturesQuote("IC00", maturity=0.03, price=5008.0, multiplier=200.0),
        IndexFuturesQuote("IC01", maturity=0.10, price=5020.0, multiplier=200.0),
        IndexFuturesQuote("IC02", maturity=0.18, price=5036.0, multiplier=200.0),
        IndexFuturesQuote("IC03", maturity=0.32, price=5064.0, multiplier=200.0),
    ]
    curve = IndexFuturesCurve(underlying="IC", spot=5000.0, quotes=quotes)
    term = curve.to_dividend_yield_curve(FlatRateCurve(0.03))
    assert all(y < 0.0 for y in term.yields)


# --- spec test 2: bumping one contract changes only that node ---

def test_bump_contract_changes_only_one_node():
    curve = _curve()
    rate_curve = FlatRateCurve(0.03)
    base = curve.to_dividend_yield_curve(rate_curve).yields
    bumped = curve.bump_contract("IC01", 1.0).to_dividend_yield_curve(rate_curve).yields
    q1, f1, t1 = base[1], curve.quotes[1].price, curve.quotes[1].maturity
    expected_q1 = 0.03 - math.log((f1 + 1.0) / 100.0) / t1
    assert bumped[1] == pytest.approx(expected_q1, abs=1e-14)
    assert bumped[0] == base[0] and bumped[2] == base[2]
    # dq ~= -dF / (T * F) for small bumps
    assert bumped[1] - q1 == pytest.approx(-1.0 / (t1 * f1), rel=1e-2)


def test_bump_contract_validation():
    curve = _curve()
    with pytest.raises(ValidationError):
        curve.bump_contract("XX99", 1.0)
    with pytest.raises(ValidationError):
        curve.bump_contract("IC00", -1e9)  # non-positive bumped price


# --- construction validation (incl. spec test 9 beta + >=2 quotes) ---

def test_quote_validation():
    with pytest.raises(ValidationError):
        IndexFuturesQuote("", maturity=0.25, price=100.0, multiplier=200.0)
    with pytest.raises(ValidationError):
        IndexFuturesQuote("IC00", maturity=-0.1, price=100.0, multiplier=200.0)
    with pytest.raises(ValidationError):
        IndexFuturesQuote("IC00", maturity=0.25, price=0.0, multiplier=200.0)
    with pytest.raises(ValidationError):
        IndexFuturesQuote("IC00", maturity=0.25, price=100.0, multiplier=0.0)
    with pytest.raises(ValidationError):
        IndexFuturesQuote("IC00", maturity=0.25, price=100.0, multiplier=200.0, beta=1.5)


def test_curve_validation():
    q = _quotes_from_q(100.0, 0.03, [0.25, 0.5], [0.01, 0.02])
    with pytest.raises(ValidationError):
        IndexFuturesCurve(underlying="", spot=100.0, quotes=q)
    with pytest.raises(ValidationError):
        IndexFuturesCurve(underlying="IC", spot=0.0, quotes=q)
    with pytest.raises(ValidationError):
        IndexFuturesCurve(underlying="IC", spot=100.0, quotes=q[:1])  # < 2 quotes
    dup = [q[0], IndexFuturesQuote("IC00", maturity=0.5, price=101.0, multiplier=200.0)]
    with pytest.raises(ValidationError):
        IndexFuturesCurve(underlying="IC", spot=100.0, quotes=dup)
    unsorted = [q[1], q[0]]
    with pytest.raises(ValidationError):
        IndexFuturesCurve(underlying="IC", spot=100.0, quotes=unsorted)
    with pytest.raises(ValidationError):
        IndexFuturesCurve(underlying="IC", spot=100.0, quotes=q, interpolation="cubic")


# --- spec test 4: hedge hands conversion ---

def test_hedge_hands_synthetic():
    assert [hedge_hands(d, 10.0) for d in (10.0, 20.0, 30.0, 40.0)] == [
        -1.0, -2.0, -3.0, -4.0,
    ]
    assert hedge_hands(600.0, 300.0) == -2.0


def test_delta_per_hand_is_multiplier():
    curve = _curve()
    assert curve.delta_per_hand("IC00") == 200.0


def test_curve_quotes_snapshot_immune_to_caller_mutation():
    quotes = _quotes_from_q(100.0, 0.03, [0.25, 0.5], [0.01, 0.02])
    curve = IndexFuturesCurve(underlying="IC", spot=100.0, quotes=quotes)
    base = curve.to_dividend_yield_curve(FlatRateCurve(0.03)).yields
    quotes.append(
        IndexFuturesQuote("IC99", maturity=2.0, price=110.0, multiplier=200.0)
    )
    quotes[0] = IndexFuturesQuote("IC00", maturity=0.25, price=99.0, multiplier=200.0)
    assert isinstance(curve.quotes, tuple)
    assert len(curve.quotes) == 2
    assert curve.to_dividend_yield_curve(FlatRateCurve(0.03)).yields == base


def test_bump_term_yield_node():
    term = TermStructureDividendYield(times=[0.25, 0.5], yields=[0.01, 0.02])
    bumped = bump_term_yield_node(term, 1, 0.0001)
    assert bumped.yields == pytest.approx([0.01, 0.0201])
    assert term.yields == pytest.approx([0.01, 0.02])  # original untouched
    with pytest.raises(ValidationError):
        bump_term_yield_node(term, 2, 0.0001)
