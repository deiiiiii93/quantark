"""MC with sigma_step == 0 on holiday steps: drift-only advance, finite PV."""
import math
from datetime import datetime, timedelta

import numpy as np
import pytest

from quantark.asset.equity.engine.mc.euro_mc_engine import EuropeanMCEngine
from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import SpotQuote
from quantark.param.div.dividend_yield import ContinuousDividendYield
from quantark.param.rrf.rate_curve import FlatRateCurve
from quantark.param.vol.vol_surface import FlatVolSurface
from quantark.param.vol.trading_clock_surface import TradingClockVolSurface
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.calendar.trading_clock import BusinessTimeMap, TradingClock
from quantark.util.enum import ObservationType, OptionType

R, Q, SIGMA_TD, SPOT = 0.02, 0.01, 0.20, 100.0


def _cn_calendar():
    return create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))


def _env(anchor, horizon_days=500):
    cal = _cn_calendar()
    m = BusinessTimeMap(
        TradingClock(cal, 244), anchor, anchor + timedelta(days=horizon_days)
    )
    return (
        PricingEnvironment(
            rate_curve=FlatRateCurve(R),
            valuation_date=anchor,
            spot_quote=SpotQuote(SPOT),
            vol_surface=TradingClockVolSurface(FlatVolSurface(SIGMA_TD), m),
            div_yield=ContinuousDividendYield(Q),
        ),
        m,
        cal,
    )


def test_snowball_mc_finite_over_cny_with_trading_clock_surface():
    env, _m, _cal = _env(datetime(2026, 2, 9))
    product = SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=BarrierConfig(
            ko_barrier=1.03,
            ko_rate=0.15,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=0.75,
            ki_observation_type=ObservationType.CONTINUOUS,
        ),
        payoff_config=None,
        contract_multiplier=1.0,
        maturity=1.0,
        is_reverse=False,
    )
    price = SnowballMCEngine(
        params=MCParams(num_paths=20_000, time_steps=252, seed=7)
    ).price(product, env)
    assert np.isfinite(price)


def test_european_mc_zero_vol_matches_deterministic_forward():
    """All-holiday horizon control: anchor on the first day of a >=4-day
    closure block, maturity inside it — tau_td == 0 throughout, so the
    terminal spot is the deterministic forward and the call prices to
    DF * max(F - K, 0) with zero MC error."""
    cal = _cn_calendar()
    d = datetime(2026, 2, 1)
    while True:
        block = 0
        probe = d
        while not cal.is_business_day(probe):
            block += 1
            probe += timedelta(days=1)
        if block >= 4:
            break
        d += timedelta(days=1)
    env, m, _ = _env(d)
    tau = 2.0 / 365.0
    assert m.to_trading(tau) == 0.0          # setup sanity: pure plateau

    option = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=tau
    )
    price = EuropeanMCEngine(
        params=MCParams(num_paths=2_000, time_steps=8, seed=11)
    ).price(option, env)
    forward = SPOT * math.exp((R - Q) * tau)
    expected = math.exp(-R * tau) * max(forward - 100.0, 0.0)
    assert price == pytest.approx(expected, abs=1e-10)
