"""Spec §7.2 European parity + §7.5 node marginals on the calendar axis."""
import math
from datetime import datetime, timedelta

import numpy as np
import pytest
from scipy import stats

from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.param import SpotQuote
from quantark.param.div.dividend_yield import ContinuousDividendYield
from quantark.param.rrf.rate_curve import FlatRateCurve
from quantark.param.vol.vol_surface import FlatVolSurface
from quantark.param.vol.trading_clock_surface import TradingClockVolSurface
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.calendar.trading_clock import BusinessTimeMap, TradingClock
from quantark.util.enum import OptionType

ANCHOR = datetime(2026, 2, 9)
R, Q, SIGMA_TD, SPOT, K = 0.02, 0.01, 0.20, 100.0, 100.0


def _env_and_map():
    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
    m = BusinessTimeMap(TradingClock(cal, 244), ANCHOR, datetime(2027, 6, 9))
    env = PricingEnvironment(
        rate_curve=FlatRateCurve(R), valuation_date=ANCHOR,
        spot_quote=SpotQuote(SPOT),
        vol_surface=TradingClockVolSurface(FlatVolSurface(SIGMA_TD), m),
        div_yield=ContinuousDividendYield(Q),
    )
    return env, m, cal


def _bs_with_total_variance(S, K_, df, carry_df, w):
    """Closed-form BS in (DF, total-variance) form — the parity oracle."""
    f = S * carry_df / df
    if w <= 0.0:
        return df * max(f - K_, 0.0)
    sw = math.sqrt(w)
    d1 = (math.log(f / K_) + 0.5 * w) / sw
    return df * (f * stats.norm.cdf(d1) - K_ * stats.norm.cdf(d1 - sw))


def test_european_parity_across_cny():
    env, m, cal = _env_and_map()
    d = ANCHOR + timedelta(days=90)          # a trading date past CNY
    while not cal.is_business_day(d):
        d += timedelta(days=1)
    tau_cal = (d - ANCHOR).days / 365.0
    tau_td = m.to_trading(tau_cal)
    opt = EuropeanVanillaOption(strike=K, option_type=OptionType.CALL, maturity=tau_cal)
    price = BlackScholesEngine().price(opt, env)
    oracle = _bs_with_total_variance(
        SPOT, K, math.exp(-R * tau_cal), math.exp(-Q * tau_cal),
        SIGMA_TD**2 * tau_td,
    )
    assert price == pytest.approx(oracle, rel=1e-12)


def test_node_marginals_variance_is_trading_and_carry_is_calendar():
    """Spec §7.5 via TermCoefficients: accumulated step variance to each node
    equals w_td(node); accumulated (r - q)·dt equals the calendar carry."""
    from quantark.priceenv.term_sampling import TermCoefficients
    env, m, cal = _env_and_map()
    t = np.linspace(0.0, 60.0 / 365.0, 60 * 4 + 1)
    tc = TermCoefficients.from_env(env, t, ref_strike=K)
    w_acc = np.cumsum(tc.step_vols**2 * np.diff(t))
    w_expected = SIGMA_TD**2 * np.asarray(m.to_trading(t[1:]))
    np.testing.assert_allclose(w_acc, w_expected, rtol=0, atol=1e-14)
    carry_acc = np.cumsum((tc.fwd_rates - tc.fwd_carry) * np.diff(t))
    np.testing.assert_allclose(carry_acc, (R - Q) * t[1:], rtol=0, atol=1e-13)
