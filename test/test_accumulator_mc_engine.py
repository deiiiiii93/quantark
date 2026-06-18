"""
Tests for the AccumulatorMCEngine.

The Monte Carlo engine is the exact discrete-monitoring benchmark. Its strongest
checks cross-validate against the analytical engine:

* No-barrier degenerate: both engines must agree (each collapses to a vanilla
  call/geared-put strip).
* SINGLE_DAY with an active barrier: the analytical SINGLE_DAY price is *exact*
  (expiry-monitored call legs, vanilla put legs -- no barrier-shift), so MC must
  agree to within Monte Carlo standard error.
"""

from datetime import datetime

import pytest

from quantark.asset.equity.engine.analytical import AccumulatorAnalyticalEngine
from quantark.asset.equity.engine.mc import AccumulatorMCEngine
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option import AccumulatorOption, EuropeanVanillaOption
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import (
    AccumulatorKnockOutType,
    MonteCarloMethod,
    ObservationType,
    OptionType,
)
from quantark.util.exceptions import PricingError


def _pricing_env(spot=100.0, rate=0.03, div=0.01, vol=0.22) -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        rate_curve=FlatRateCurve(rate=rate),
        vol_surface=FlatVolSurface(volatility=vol),
        div_yield=ContinuousDividendYield(div_yield=div),
        valuation_date=datetime(2024, 1, 1),
    )


def _monthly_obs(n=12):
    return [round((i + 1) / 12.0, 10) for i in range(n)]


def test_rejects_wrong_product_type():
    engine = AccumulatorMCEngine()
    bad = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    with pytest.raises(PricingError, match="AccumulatorOption"):
        engine.price(bad, _pricing_env())


def test_mc_near_expiry_returns_realized_accrual():
    env = _pricing_env()
    acc = AccumulatorOption(
        strike=95.0,
        knock_out_barrier=108.0,
        option_type=OptionType.CALL,
        maturity=1e-12,
        daily_share_accumulation=1.0,
        observation_dates=[1e-12],
        knock_out_type=AccumulatorKnockOutType.SINGLE_DAY,
        past_observations=[(-0.2, 100.0)],  # locked-in +5
    )
    price = AccumulatorMCEngine().price(acc, env)
    assert price == pytest.approx(5.0, abs=1e-9)


def test_mc_matches_analytical_no_barrier_termination():
    env = _pricing_env()
    obs = _monthly_obs()
    acc = AccumulatorOption(
        strike=96.0,
        knock_out_barrier=1.0e6,  # no knock-out -> exact vanilla strip both ways
        option_type=OptionType.CALL,
        maturity=1.0,
        daily_share_accumulation=1.0,
        gearing=2.0,
        knock_out_type=AccumulatorKnockOutType.TERMINATION,
        observation_dates=obs,
    )
    analytic = AccumulatorAnalyticalEngine().price(acc, env)
    mc_engine = AccumulatorMCEngine(
        MCParams(num_paths=120_000, seed=7), method=MonteCarloMethod.QUASI
    )
    mc = mc_engine.price(acc, env)
    assert mc == pytest.approx(analytic, rel=2e-3)


def test_mc_matches_analytical_single_day_active_barrier():
    # SINGLE_DAY analytical price is exact (no barrier shift), so MC must agree
    # within a few standard errors.
    env = _pricing_env()
    obs = _monthly_obs()
    acc = AccumulatorOption(
        strike=96.0,
        knock_out_barrier=107.0,  # active barrier
        option_type=OptionType.CALL,
        maturity=1.0,
        daily_share_accumulation=1.0,
        gearing=2.0,
        knock_out_type=AccumulatorKnockOutType.SINGLE_DAY,
        observation_dates=obs,
    )
    analytic = AccumulatorAnalyticalEngine().price(acc, env)
    mc_engine = AccumulatorMCEngine(MCParams(num_paths=200_000, seed=11))
    mc = mc_engine.price(acc, env)
    std_err = mc_engine.get_last_std_error() or 0.0
    # Agreement within 4 standard errors (exact analytical reference).
    assert abs(mc - analytic) < 4.0 * std_err + 1e-6


def test_mc_matches_analytical_termination_active_barrier_with_rebate():
    # Full TERMINATION machinery: cumulative up-out legs + one-touch rebate.
    # Analytical uses the BGK discrete barrier shift, so agreement with the exact
    # MC benchmark is to a modest relative tolerance.
    env = _pricing_env()
    obs = _monthly_obs()
    acc = AccumulatorOption(
        strike=96.0,
        knock_out_barrier=107.0,
        option_type=OptionType.CALL,
        maturity=1.0,
        initial_price=100.0,
        notional=100.0,  # daily shares = 1.0; rebate cash = rate * notional
        gearing=2.0,
        knock_out_type=AccumulatorKnockOutType.TERMINATION,
        knock_out_rebate_rate=0.01,
        observation_dates=obs,
    )
    analytic = AccumulatorAnalyticalEngine().price(acc, env)
    mc = AccumulatorMCEngine(
        MCParams(num_paths=200_000, seed=5), method=MonteCarloMethod.QUASI
    ).price(acc, env)
    # ~2% gap is the BGK discrete-barrier-shift approximation in the analytical
    # TERMINATION price vs the exact MC benchmark (the no-barrier and SINGLE_DAY
    # cases, which carry no barrier shift, agree far more tightly).
    assert mc == pytest.approx(analytic, rel=3e-2)


def test_mc_zero_maturity_includes_extra_shares():
    # At expiry the extra-shares leg has a deterministic terminal value even
    # though there is no remaining accrual.
    env = _pricing_env(spot=90.0)
    acc = AccumulatorOption(
        strike=95.0,
        knock_out_barrier=108.0,
        option_type=OptionType.CALL,
        maturity=1e-12,
        daily_share_accumulation=1.0,
        observation_dates=[1e-12],
        knock_out_type=AccumulatorKnockOutType.SINGLE_DAY,
        extra_shares_at_expiry=1.0,
    )
    price = AccumulatorMCEngine().price(acc, env)
    # short up-and-out put: -(K - S)+ = -(95 - 90) = -5
    assert price == pytest.approx(-5.0, abs=1e-9)


def test_mc_empty_observations_still_prices_extra_shares():
    # An accumulator with no accrual fixings but a terminal extra-shares leg must
    # still value that leg; cross-check against the analytical engine.
    env = _pricing_env()
    acc = AccumulatorOption(
        strike=98.0,
        knock_out_barrier=108.0,
        option_type=OptionType.CALL,
        maturity=1.0,
        daily_share_accumulation=1.0,
        observation_type=ObservationType.EXPIRY,
        observation_dates=[],
        knock_out_type=AccumulatorKnockOutType.SINGLE_DAY,
        extra_shares_at_expiry=1.0,
    )
    analytic = AccumulatorAnalyticalEngine().price(acc, env)
    assert analytic != pytest.approx(0.0)  # the extra leg is non-trivial
    mc = AccumulatorMCEngine(
        MCParams(num_paths=200_000, seed=9), method=MonteCarloMethod.QUASI
    ).price(acc, env)
    assert mc == pytest.approx(analytic, rel=5e-3)


def test_mc_contract_multiplier_scales_price():
    env = _pricing_env()
    obs = _monthly_obs()
    common = dict(
        strike=96.0,
        knock_out_barrier=107.0,
        option_type=OptionType.CALL,
        maturity=1.0,
        daily_share_accumulation=1.0,
        observation_dates=obs,
        knock_out_type=AccumulatorKnockOutType.SINGLE_DAY,
    )
    base = AccumulatorMCEngine(MCParams(num_paths=40_000, seed=3)).price(
        AccumulatorOption(**common, contract_multiplier=1.0), env
    )
    scaled = AccumulatorMCEngine(MCParams(num_paths=40_000, seed=3)).price(
        AccumulatorOption(**common, contract_multiplier=10.0), env
    )
    assert scaled == pytest.approx(10.0 * base, rel=1e-10)
