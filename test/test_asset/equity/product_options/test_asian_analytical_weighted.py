"""Weighted analytical Asian pricing (sub-project D).

Turnbull-Wakeman (arithmetic) and geometric-discrete are validated against the
weighted MC reference. Methods that cannot represent non-uniform weights
(Kemna-Vorst, Levy, Curran, discrete-HHM) and floating-strike must reject
non-uniform weights rather than silently mis-price.
"""
from datetime import datetime

import pytest

from quantark.asset.equity.product.option.asian_option import (
    AsianObservationRecord,
    AsianOption,
)
from quantark.asset.equity.engine.analytical import AsianOptionAnalyticalEngine
from quantark.asset.equity.engine.mc import AsianOptionMCEngine
from quantark.asset.equity.param import MCParams
from quantark.priceenv import PricingEnvironment
from quantark.param import (
    SpotQuote,
    FlatVolSurface,
    FlatRateCurve,
    ContinuousDividendYield,
)
from quantark.util.enum import AveragingType, OptionType, AsianStrikeType
from quantark.util.enum.engine_enums import AsianAnalyticalMethod, MonteCarloMethod
from quantark.util.exceptions import ValidationError


def _env(spot=100.0, rate=0.05, vol=0.20, div=0.0):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div),
        valuation_date=datetime(2024, 1, 1),
    )


def _weighted_option(weights, averaging=AveragingType.ARITHMETIC, strike=100.0,
                     option_type=OptionType.CALL,
                     strike_type=AsianStrikeType.FIXED):
    times = [0.25, 0.5, 0.75, 1.0]
    return AsianOption(
        strike=strike,
        option_type=option_type,
        asian_strike_type=strike_type,
        averaging_type=averaging,
        maturity=1.0,
        observation_records=[
            AsianObservationRecord(observation_time=t, weight=w)
            for t, w in zip(times, weights)
        ],
    )


def _mc():
    return AsianOptionMCEngine(
        params=MCParams(num_paths=400000, seed=20240622),
        method=MonteCarloMethod.QUASI,
    )


# --- weighted TW vs MC ------------------------------------------------------

def test_weighted_turnbull_wakeman_matches_mc():
    env = _env(vol=0.25)
    opt = _weighted_option([1.0, 2.0, 3.0, 4.0])
    tw = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.TURNBULL_WAKEMAN)
    mc = _mc()
    p_tw = tw.price(opt, env)
    p_mc = mc.price(opt, env)
    se = mc.get_last_result().std_error
    assert abs(p_tw - p_mc) < max(0.05, 4.0 * se)


def test_weighted_tw_differs_from_uniform():
    env = _env(vol=0.25)
    tw = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.TURNBULL_WAKEMAN)
    p_back = tw.price(_weighted_option([1.0, 1.0, 1.0, 5.0]), env)
    p_uniform = tw.price(_weighted_option([1.0, 1.0, 1.0, 1.0]), env)
    assert p_back > p_uniform + 0.1


def test_uniform_explicit_weights_equal_unweighted():
    env = _env(vol=0.22)
    tw = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.TURNBULL_WAKEMAN)
    times = [0.25, 0.5, 0.75, 1.0]
    weighted = _weighted_option([1.0, 1.0, 1.0, 1.0])
    unweighted = AsianOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0,
        observation_records=[AsianObservationRecord(observation_time=t) for t in times],
    )
    assert tw.price(weighted, env) == pytest.approx(tw.price(unweighted, env), rel=1e-9)


# --- weighted geometric vs MC -----------------------------------------------

def test_weighted_geometric_discrete_matches_mc():
    env = _env(vol=0.25)
    opt = _weighted_option([1.0, 2.0, 3.0, 4.0], averaging=AveragingType.GEOMETRIC)
    geo = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.GEOMETRIC_DISCRETE)
    mc = _mc()
    p_geo = geo.price(opt, env)
    p_mc = mc.price(opt, env)
    se = mc.get_last_result().std_error
    assert abs(p_geo - p_mc) < max(0.05, 4.0 * se)


# --- rejection of unsupported weighted methods ------------------------------

@pytest.mark.parametrize("method", [
    AsianAnalyticalMethod.LEVY,
    AsianAnalyticalMethod.CURRAN,
    AsianAnalyticalMethod.DISCRETE_HHM,
    AsianAnalyticalMethod.KEMNA_VORST,
])
def test_nonuniform_weights_rejected_by_unsupported_methods(method):
    env = _env()
    averaging = (
        AveragingType.GEOMETRIC
        if method == AsianAnalyticalMethod.KEMNA_VORST
        else AveragingType.ARITHMETIC
    )
    opt = _weighted_option([1.0, 2.0, 3.0, 4.0], averaging=averaging)
    engine = AsianOptionAnalyticalEngine(method=method)
    with pytest.raises(ValidationError, match="weight"):
        engine.price(opt, env)


def test_nonuniform_weights_rejected_for_floating_strike():
    env = _env()
    opt = _weighted_option([1.0, 2.0, 3.0, 4.0], strike_type=AsianStrikeType.FLOATING)
    tw = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.TURNBULL_WAKEMAN)
    with pytest.raises(ValidationError, match="weight"):
        tw.price(opt, env)


# --- review-driven edge cases ------------------------------------------------

def test_inprogress_floating_transform_renormalizes_all_weights():
    """In-progress (m>0) floating-strike: dropping the terminal fixing must
    renormalize past AND future weights so they still sum to 1 (P2). The
    floating symmetry itself is an approximation for in-progress averages, so
    we assert the weight invariant directly rather than MC agreement."""
    env = _env(spot=100.0, rate=0.05, vol=0.20)
    recs = [
        AsianObservationRecord(observation_time=-0.5, observed_price=98.0),
        AsianObservationRecord(observation_time=-0.25, observed_price=101.0),
        AsianObservationRecord(observation_time=0.5),
        AsianObservationRecord(observation_time=1.0),  # terminal fixing == T
    ]
    opt = AsianOption(
        strike=0.0, option_type=OptionType.CALL,
        asian_strike_type=AsianStrikeType.FLOATING,
        averaging_type=AveragingType.ARITHMETIC, maturity=1.0,
        observation_records=recs,
    )
    tw = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.TURNBULL_WAKEMAN)

    captured = {}
    original = tw._price_fixed_strike_internal

    def spy(params, method, is_call):
        captured["past"] = list(params["past_weights"])
        captured["future"] = list(params["future_weights"])
        return original(params, method, is_call)

    tw._price_fixed_strike_internal = spy
    tw.price(opt, env)

    total = sum(captured["past"]) + sum(captured["future"])
    assert total == pytest.approx(1.0)
    # one future fixing dropped: 3 remaining fixings, each 1/3 for a uniform schedule
    assert captured["past"] == pytest.approx([1 / 3, 1 / 3])
    assert captured["future"] == pytest.approx([1 / 3])


def test_all_past_geometric_weighted_uses_weights():
    """All-past geometric must honor non-uniform weights (P2)."""
    env = _env()
    recs = [
        AsianObservationRecord(observation_time=-0.5, observed_price=100.0, weight=1.0),
        AsianObservationRecord(observation_time=-0.25, observed_price=400.0, weight=3.0),
    ]
    opt = AsianOption(
        strike=200.0, option_type=OptionType.CALL,
        averaging_type=AveragingType.GEOMETRIC, maturity=1.0,
        observation_records=recs,
    )
    geo = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.GEOMETRIC_DISCRETE)
    # weighted geo avg = (100^1 * 400^3)^(1/4) ~= 282.84 ; equal-weight would be 200 (payoff 0)
    expected_avg = (100.0 ** 1 * 400.0 ** 3) ** 0.25
    import math
    expected = max(expected_avg - 200.0, 0.0) * math.exp(-0.05)
    assert geo.price(opt, env) == pytest.approx(expected, rel=1e-6)


def test_explicit_uniform_weights_not_rejected_by_levy():
    """Explicit but uniform weights are mathematically unweighted -> not rejected (P3)."""
    env = _env()
    opt = _weighted_option([2.0, 2.0, 2.0, 2.0])  # all equal -> uniform
    levy = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.LEVY)
    # should price, not raise
    assert levy.price(opt, env) > 0.0
