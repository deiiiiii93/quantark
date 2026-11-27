"""CalendarQuotedVolSurface: a calendar-quoted implied surface on the BUSINESS_DAYS trading axis (run 7 D2, AC5).

The wrapper owns a DATE-KNOT map: one knot per business date d in [anchor, horizon], u_d = the env time of d exactly
as ``date_axis.env_time`` computes it, c_d = (d - anchor).days / 365. It reproduces the calendar-clock total variance
sigma_cal(c_d)^2 * c_d at EVERY business date, Mondays and the first day after the Spring Festival included (unlike
``BusinessTimeMap.to_calendar``, whose knots sit at the end of each trading day).
"""
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.engine.quad.v2.engine import SnowballQuadEngineV2
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.asset.equity.riskmeasures.greeks.bump_envs import bump_unit_vol
from quantark.param import SpotQuote
from quantark.param.rrf.rate_curve import FlatRateCurve
from quantark.param.vol import (
    CalendarQuotedVolSurface,
    DateKnotTimeMap,
    FlatVolSurface,
    GridVolSurface,
    TermStructureVolSurface,
)
from quantark.priceenv import CurveQuotingBasis, PricingEnvironment
from quantark.priceenv.date_axis import env_time, env_times
from quantark.util.calendar import BusinessDayConvention, CalendarType, DayCountConvention, create_calendar
from quantark.util.calendar.trading_clock import BusinessTimeMap, TradingClock
from quantark.util.enum import ObservationType, OptionType
from quantark.util.exceptions import ValidationError

CAL = create_calendar(CalendarType.CHINA_SSE, year_range=(2025, 2029))
D = 244
ANCHOR = datetime(2026, 1, 16)          # a Friday
HORIZON = datetime(2027, 6, 30)
MONDAY = datetime(2026, 1, 19)
WEDNESDAY = datetime(2026, 1, 21)
NEXT_MONDAY = datetime(2026, 1, 26)
POST_FESTIVAL = datetime(2026, 2, 24)   # first SSE trading day after the 2026 Spring Festival block
SPOT = 100.0
STRIKES = (70.0, 95.0, 100.0, 104.0, 130.0)


def _inners():
    return {
        "flat": FlatVolSurface(0.21),
        "term": TermStructureVolSurface(times=[0.02, 0.1, 0.4, 1.0, 1.5], vols=[0.32, 0.27, 0.23, 0.21, 0.205]),
        "grid": GridVolSurface(
            strikes=[80.0, 100.0, 120.0],
            maturities=[0.05, 0.25, 1.0, 1.5],
            iv_grid=np.array([[0.35, 0.30, 0.28], [0.30, 0.26, 0.25], [0.27, 0.23, 0.22], [0.26, 0.22, 0.215]]),
        ),
    }


def _map(anchor=ANCHOR, calendar=CAL, days_per_year=D, horizon=HORIZON):
    return DateKnotTimeMap(TradingClock(calendar, days_per_year), anchor, horizon)


def _business_dates(start, end, calendar=CAL):
    out, d = [], start + timedelta(days=1)
    while d <= end:
        if calendar.is_business_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out


def _cal_t(d, anchor=ANCHOR):
    return (d - anchor).days / 365.0


def _bd_env(vol_surface, valuation_date=ANCHOR, rate=0.0, **kw):
    return PricingEnvironment(
        rate_curve=FlatRateCurve(rate), valuation_date=valuation_date, spot_quote=SpotQuote(SPOT),
        vol_surface=vol_surface, day_count_convention=DayCountConvention.BUSINESS_DAYS, bus_days_in_year=D,
        calendar=CAL, rate_curve_quoting=CurveQuotingBasis.CALENDAR_YEAR, trading_clock_horizon=HORIZON, **kw)


def _cal_env(vol_surface, rate=0.0):
    return PricingEnvironment(rate_curve=FlatRateCurve(rate), valuation_date=ANCHOR, spot_quote=SpotQuote(SPOT),
                              vol_surface=vol_surface)


def test_window_has_the_non_trading_blocks_it_claims():
    assert ANCHOR.weekday() == 4 and MONDAY.weekday() == 0 and NEXT_MONDAY.weekday() == 0
    assert WEDNESDAY.weekday() == 2
    assert CAL.is_business_day(POST_FESTIVAL)
    assert not any(CAL.is_business_day(POST_FESTIVAL - timedelta(days=k)) for k in range(1, 8))


def test_knots_are_the_env_times_of_every_business_date_bitwise():
    m = _map()
    env = _bd_env(FlatVolSurface(0.2))
    dates = _business_dates(ANCHOR, HORIZON)
    assert list(m.knot_dates) == [ANCHOR] + dates
    expected_u = np.r_[0.0, env_times(env, dates)]
    assert np.array_equal(m.u_knots, expected_u)
    assert np.array_equal(m.c_knots, np.r_[0.0, [_cal_t(d) for d in dates]])
    for d in (MONDAY, WEDNESDAY, POST_FESTIVAL, dates[-1]):
        assert m.to_calendar(env_time(env, d)) == _cal_t(d)


def test_is_a_business_time_map_with_the_clock_the_env_checks():
    m = _map()
    assert isinstance(m, BusinessTimeMap)       # the env's re-anchor path only moves BusinessTimeMaps
    assert m.clock.calendar is CAL and m.clock.days_per_year == D
    assert m.anchor_date == ANCHOR and m.horizon_date == HORIZON


@pytest.mark.parametrize("name", ["flat", "term", "grid"])
def test_total_variance_is_the_calendar_variance_at_every_business_date(name):
    inner = _inners()[name]
    s = CalendarQuotedVolSurface(inner, _map())
    env = _bd_env(FlatVolSurface(0.2))
    dates = _business_dates(ANCHOR, datetime(2027, 3, 31))
    us = env_times(env, dates)
    for d, u in zip(dates, us):
        c = _cal_t(d)
        for k in STRIKES:
            w_cal = inner.get_vol(k, c, SPOT) ** 2 * c
            assert s.get_vol(k, u, SPOT) ** 2 * u == pytest.approx(w_cal, rel=1e-14, abs=0.0)
            assert s.total_variance(k, u, SPOT) == pytest.approx(w_cal, rel=1e-14, abs=0.0)


@pytest.mark.parametrize("name", ["flat", "term", "grid"])
def test_monday_and_post_festival_carry_the_full_weekend_and_holiday_variance(name):
    inner = _inners()[name]
    s = CalendarQuotedVolSurface(inner, _map())
    env = _bd_env(FlatVolSurface(0.2))
    assert env_time(env, MONDAY) == 1.0 / D
    for d, days in ((MONDAY, 3), (POST_FESTIVAL, 39)):
        assert (d - ANCHOR).days == days
        u, c = env_time(env, d), days / 365.0
        for k in STRIKES:
            assert s.get_vol(k, u, SPOT) ** 2 * u == pytest.approx(inner.get_vol(k, c, SPOT) ** 2 * c, rel=1e-14)


def test_vectorized_total_variance_and_zero():
    s = CalendarQuotedVolSurface(_inners()["term"], _map())
    env = _bd_env(FlatVolSurface(0.2))
    us = np.r_[0.0, env_times(env, [MONDAY, WEDNESDAY, POST_FESTIVAL])]
    w = s.total_variance(100.0, us, SPOT)
    assert w.shape == us.shape and w[0] == 0.0
    for j in range(1, 4):
        assert w[j] == s.total_variance(100.0, float(us[j]), SPOT)


def test_is_smile_follows_the_inner_and_exact_variance_marker():
    m = _map()
    assert CalendarQuotedVolSurface(_inners()["flat"], m).is_smile is False
    assert CalendarQuotedVolSurface(_inners()["grid"], m).is_smile is True
    assert CalendarQuotedVolSurface(_inners()["flat"], m).exposes_exact_total_variance is True


def test_zero_time_limit_uses_the_initial_slope_of_the_date_knots():
    m = _map()
    slope = (3.0 / 365.0) / (1.0 / D)           # Friday -> Monday: three calendar days for one trading day
    assert m.initial_slope() == pytest.approx(slope, rel=1e-15)
    for name, inner in _inners().items():
        s = CalendarQuotedVolSurface(inner, m)
        for k in STRIKES:
            assert s.get_vol(k, 0.0, SPOT) == pytest.approx(
                inner.get_vol(k, 0.0, SPOT) * np.sqrt(m.initial_slope()), rel=1e-15)


def test_raises_for_negative_time_and_past_the_last_knot():
    m = _map()
    s = CalendarQuotedVolSurface(FlatVolSurface(0.2), m)
    last = float(m.u_knots[-1])
    with pytest.raises(ValidationError):
        s.get_vol(100.0, -1e-9, SPOT)
    with pytest.raises(ValidationError):
        s.get_vol(100.0, last + 1e-9, SPOT)
    with pytest.raises(ValidationError):
        s.total_variance(100.0, np.array([0.1, last + 1e-9]), SPOT)
    with pytest.raises(ValidationError):
        s.total_variance(100.0, -1e-9, SPOT)
    assert s.get_vol(100.0, last, SPOT) > 0.0


def test_holiday_anchor_preceding_counts_from_the_previous_business_day():
    saturday = datetime(2026, 1, 17)
    m = _map(anchor=saturday)
    env = _bd_env(FlatVolSurface(0.2), valuation_date=saturday)
    assert m.to_calendar(env_time(env, MONDAY)) == 2.0 / 365.0
    assert env_time(env, MONDAY) == 1.0 / D


def test_holiday_anchor_rolled_forward_refuses():
    """FOLLOWING from a Saturday puts Monday at u = 0 with two calendar days of variance: not representable."""
    with pytest.raises(ValidationError):
        DateKnotTimeMap(TradingClock(CAL, D), datetime(2026, 1, 17), HORIZON,
                        holiday_time_adjustment=BusinessDayConvention.FOLLOWING)


def test_map_needs_a_business_date_after_the_anchor():
    with pytest.raises(ValidationError):
        _map(horizon=ANCHOR)
    with pytest.raises(ValidationError):
        _map(anchor=datetime(2026, 2, 13), horizon=datetime(2026, 2, 20))   # festival block only


def test_env_keeps_a_wrapper_on_its_valuation_date_and_re_anchors_it():
    s = CalendarQuotedVolSurface(_inners()["term"], _map())
    env = _bd_env(s)
    assert env.vol_surface is s
    new_val = datetime(2026, 2, 11)              # a Wednesday before the festival
    env.valuation_date = new_val
    moved = env.vol_surface
    assert type(moved) is CalendarQuotedVolSurface and moved.inner is s.inner
    assert moved.time_map.anchor_date == new_val and moved.time_map.horizon_date == HORIZON
    assert list(moved.time_map.knot_dates)[:2] == [new_val, datetime(2026, 2, 12)]
    u = env_time(env, POST_FESTIVAL)
    c = (POST_FESTIVAL - new_val).days / 365.0
    assert moved.time_map.to_calendar(u) == c
    assert env.get_vol(100.0, u) ** 2 * u == pytest.approx(s.inner.get_vol(100.0, c, SPOT) ** 2 * c, rel=1e-14)


def test_env_re_anchors_a_wrapper_built_on_another_date_at_construction():
    s = CalendarQuotedVolSurface(_inners()["flat"], _map(anchor=datetime(2026, 1, 5)))
    env = _bd_env(s)
    assert env.vol_surface.time_map.anchor_date == ANCHOR
    assert np.array_equal(env.vol_surface.time_map.u_knots, _map().u_knots)


def test_env_refuses_a_wrapper_on_another_clock():
    with pytest.raises(ValidationError):
        _bd_env(CalendarQuotedVolSurface(FlatVolSurface(0.2), _map(days_per_year=252)))
    other = create_calendar(CalendarType.CHINA_SSE, year_range=(2025, 2029))
    assert other is not CAL
    with pytest.raises(ValidationError):
        _bd_env(CalendarQuotedVolSurface(FlatVolSurface(0.2), _map(calendar=other)))


def test_parallel_shifted_shifts_the_calendar_quoted_inner():
    m = _map()
    s = CalendarQuotedVolSurface(_inners()["term"], m)
    shifted = s.parallel_shifted(0.01)
    assert type(shifted) is CalendarQuotedVolSurface and shifted.time_map is m
    assert shifted.inner.vols == [v + 0.01 for v in s.inner.vols]
    env = _bd_env(FlatVolSurface(0.2))
    u, c = env_time(env, POST_FESTIVAL), _cal_t(POST_FESTIVAL)
    shifted_inner = TermStructureVolSurface(times=list(s.inner.times), vols=[v + 0.01 for v in s.inner.vols])
    assert shifted.get_vol(100.0, u, SPOT) ** 2 * u == pytest.approx(
        shifted_inner.get_vol(100.0, c, SPOT) ** 2 * c, rel=1e-14)
    flat = CalendarQuotedVolSurface(FlatVolSurface(0.2), m).parallel_shifted(0.01)
    assert flat.get_vol(100.0, u, SPOT) ** 2 * u == pytest.approx(0.21 ** 2 * c, rel=1e-14)


def test_with_time_map_and_to_inner_time():
    s = CalendarQuotedVolSurface(_inners()["flat"], _map())
    other = _map(anchor=datetime(2026, 1, 20))
    assert s.with_time_map(other).time_map is other and s.with_time_map(other).inner is s.inner
    env = _bd_env(FlatVolSurface(0.2))
    assert s.to_inner_time(env_time(env, MONDAY)) == 3.0 / 365.0


def test_bump_unit_vol_is_the_inner_calendar_vol():
    inner = _inners()["grid"]
    env = _bd_env(CalendarQuotedVolSurface(inner, _map()))
    for d in (MONDAY, POST_FESTIVAL):
        u = env_time(env, d)
        assert bump_unit_vol(env, 95.0, u) == inner.get_vol(95.0, _cal_t(d), SPOT)


def _numeric_snowball(env):
    ko_dates = [datetime(2026, 2, 24), datetime(2026, 3, 16), datetime(2026, 4, 16), datetime(2026, 5, 18)]
    ko_times = [env_time(env, d) for d in ko_dates]
    ki_times = [float(t) for t in env_times(env, _business_dates(ANCHOR, ko_dates[-1]))]
    return SnowballOption(
        initial_price=100.0, strike=100.0,
        barrier_config=BarrierConfig(
            ko_barrier=103.0, ko_rate=0.0, ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=ko_times, ki_barrier=75.0, ki_observation_type=ObservationType.DISCRETE,
            ki_observation_dates=ki_times, ki_continuous=False),
        contract_multiplier=1.0, maturity=ko_times[-1], is_reverse=False)


@pytest.mark.parametrize("name", ["flat", "term"])
def test_quad_v2_accepts_a_deterministic_inner(name):
    env = _bd_env(CalendarQuotedVolSurface(_inners()[name], _map()))
    price = float(SnowballQuadEngineV2().price(_numeric_snowball(env), env))
    assert np.isfinite(price) and price > 0.0


def test_quad_v2_refuses_a_smile_inner():
    env = _bd_env(CalendarQuotedVolSurface(_inners()["grid"], _map()))
    with pytest.raises(NotImplementedError):
        SnowballQuadEngineV2().price(_numeric_snowball(env), env)


def test_quad_v2_gate_direct():
    from quantark.asset.equity.engine.quad.v2.engine import AutocallableQuadEngineV2

    contract = SimpleNamespace(events=[SimpleNamespace(time=0.25)])
    product = SimpleNamespace(strike=100.0)
    for name in ("flat", "term"):
        env = _bd_env(CalendarQuotedVolSurface(_inners()[name], _map()))
        AutocallableQuadEngineV2._validate_market(product, env, contract)
    env = _bd_env(CalendarQuotedVolSurface(_inners()["grid"], _map()))
    with pytest.raises(NotImplementedError):
        AutocallableQuadEngineV2._validate_market(product, env, contract)


@pytest.mark.parametrize("name", ["flat", "term"])
@pytest.mark.parametrize("expiry", [WEDNESDAY, NEXT_MONDAY, POST_FESTIVAL])
@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_european_on_the_trading_clock_equals_the_calendar_clock_at_zero_rates(name, expiry, option_type):
    """r = q = 0, so the D2b cash-wrapper lag cannot enter: the business-day price is the calendar price."""
    inner = _inners()[name]
    bd_env = _bd_env(CalendarQuotedVolSurface(inner, _map()))
    cal_env = _cal_env(inner)
    engine = BlackScholesEngine()
    opt = EuropeanVanillaOption(strike=103.0, option_type=option_type, exercise_date=expiry)
    p_bd, p_cal = engine.price(opt, bd_env), engine.price(opt, cal_env)
    assert opt.get_maturity(bd_env) == env_time(bd_env, expiry)
    assert opt.get_maturity(cal_env) == _cal_t(expiry)
    assert p_bd == pytest.approx(p_cal, rel=1e-12, abs=0.0)


def _lagged_calendar_time(expiry):
    """Where the 0.15.0 CALENDAR_YEAR cash wrapper reads the calendar curve for ``expiry`` (D2b)."""
    env = _bd_env(FlatVolSurface(0.2))
    return float(env.rate_curve.time_map.to_calendar(env_time(env, expiry)))


@pytest.mark.parametrize("expiry", [WEDNESDAY, NEXT_MONDAY, POST_FESTIVAL])
def test_d2b_cash_wrapper_lag_measurement_at_three_percent(expiry):
    """r = 3%, q = 0: the vol is exact, so the business-day price is the calendar price with the rate read at the
    0.15.0 cash wrapper's lagged calendar time. Mid-week the lag is zero; the first trading day after a weekend or a
    holiday is short by the non-trading days (DF to Monday = DF_cal(Saturday)). Measured values in the run report."""
    r = 0.03
    inner = _inners()["flat"]
    bd_env = _bd_env(CalendarQuotedVolSurface(inner, _map()), rate=r)
    opt = EuropeanVanillaOption(strike=103.0, option_type=OptionType.CALL, exercise_date=expiry)
    engine = BlackScholesEngine()
    p_bd = engine.price(opt, bd_env)
    p_cal = engine.price(opt, _cal_env(inner, rate=r))
    c_true, c_lag = _cal_t(expiry), _lagged_calendar_time(expiry)
    # the same contract on the calendar clock with r*T = r*c_lag: the whole difference is the cash lag
    p_lag = engine.price(opt, _cal_env(inner, rate=r * c_lag / c_true))
    assert p_bd == pytest.approx(p_lag, rel=1e-12, abs=0.0)
    if expiry == WEDNESDAY:
        assert c_lag == c_true and p_bd == pytest.approx(p_cal, rel=1e-12, abs=0.0)
    else:
        assert c_lag < c_true
    print(f"D2b {expiry:%a %Y-%m-%d}: cal days {(expiry - ANCHOR).days}, lagged cal days {c_lag * 365:.6f}, "
          f"P_bd={p_bd:.12f} P_cal={p_cal:.12f} diff={p_bd - p_cal:.6e} rel={(p_bd - p_cal) / p_cal:.6e}")


def test_deepcopy_and_pickle_keep_the_clock_and_the_knots():
    import copy
    import pickle

    env = _bd_env(CalendarQuotedVolSurface(_inners()["term"], _map()))
    u = env_time(env, POST_FESTIVAL)
    for twin in (copy.deepcopy(env), pickle.loads(pickle.dumps(env))):
        assert twin.vol_surface.time_map.clock.calendar is twin.calendar
        assert np.array_equal(twin.vol_surface.time_map.u_knots, env.vol_surface.time_map.u_knots)
        assert twin.get_vol(100.0, u) == env.get_vol(100.0, u)
        twin.valuation_date = WEDNESDAY
        assert twin.vol_surface.time_map.anchor_date == WEDNESDAY
