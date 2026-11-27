"""Replay on a business-day clock (run 7, .ship/2026-11-26-replay-business-day-clock D3-D7; AC1a, AC1, AC2, AC4, AC6, AC9).

``AutocallableEngineConfig.trading_clock`` puts every env replay builds for a day on one BUSINESS_DAYS clock: the cash
curves are declared CALENDAR_YEAR (the env wraps them), the vol is wrapped in ``CalendarQuotedVolSurface`` (or used
native under ``vol_quoting=TRADING_YEAR``), and the env-time conversions replay does itself (pending receivable,
event dates, ``pricing_q``) read the env's clock. ``trading_clock=None`` keeps today's calendar-day envs bitwise.

Synthetic markets only. AC1a is the independent anchor: on a calendar where every day is a business day and
``bus_days_in_year=365`` the trading clock IS the ACT/365 clock, so the clock path must reproduce the default replay.
"""
from __future__ import annotations

import copy
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from quantark.asset.equity.engine.pde.grid.config import GridConfig
from quantark.asset.equity.param import MCParams, PDEParams
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.snowball_config import BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit
from quantark.backtest.replay import (
    AutocallableBacktestConfig,
    AutocallableBacktestEngine,
    AutocallableEngineConfig,
    AutocallableMarketDataSet,
    HedgeSpec,
    ImpliedBasisYield,
    ReplayBacktestConfig,
    ReplayBacktestEngine,
    ReplayProduct,
    SignedDividendYield,
    TradingClockConfig,
)
from quantark.backtest.replay.engine_factory import create_pricing_engine
from quantark.backtest.replay.market import derive_implied_dividend_yield
from quantark.backtest.transaction_costs import ZeroCostModel
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.basis import FlatBasisYield
from quantark.param.div.dividend_yield import ContinuousDividendYield, DividendYield
from quantark.param.vol import CalendarQuotedVolSurface, DateKnotTimeMap
from quantark.priceenv import CurveQuotingBasis, PricingEnvironment
from quantark.priceenv.date_axis import env_time
from quantark.util.calendar import BusinessDayConvention, Calendar, CalendarType, DayCountConvention, create_calendar
from quantark.util.calendar.trading_clock import TradingClock
from quantark.util.enum import ObservationType
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import ValidationError

CQ = CurveQuotingBasis.CALENDAR_YEAR
SSE = create_calendar(CalendarType.CHINA_SSE, year_range=(2025, 2028))
#: every day a business day: with D = 365 the trading clock is the ACT/365 clock (AC1a)
ALL_DAYS = Calendar(holidays=set(), weekend_days=set(), name="ALL_DAYS")
UNDERLYING = "IDX"
RATE = 0.02
FUT_EXPIRY = datetime(2026, 6, 19)
#: SSE window across the 2026 New Year holiday (Jan 1-4 closed): Mon 12-29 .. Wed 01-07
SSE_WINDOW = [d for d in (datetime(2025, 12, 29) + timedelta(days=k) for k in range(10)) if SSE.is_business_day(d)]
#: consecutive calendar days, weekend included (business days of ALL_DAYS only)
ALL_WINDOW = [datetime(2026, 1, 8) + timedelta(days=k) for k in range(5)]
SPOTS = [100.0, 98.5, 101.0, 99.0, 102.0, 100.5, 97.0, 99.5]


# --------------------------------------------------------------------------- fixtures


def _market(days, spots=SPOTS, vol=0.22, expiries=(FUT_EXPIRY,)):
    dates = pd.DatetimeIndex(days)
    spots = list(spots[: len(days)])
    futures = []
    for d, s in zip(dates, spots):
        for k, expiry in enumerate(expiries):
            futures.append({"date": d, "contract": f"IF{k}", "futures_price": s * (0.995 - 0.004 * k),
                            "expiry_date": pd.Timestamp(expiry), "multiplier": 300.0})
    return AutocallableMarketDataSet.from_dataframes(
        spot_data=pd.DataFrame({"date": dates, "spot": spots}),
        vol_data=pd.DataFrame({"date": dates, "volatility": [vol] * len(days)}),
        rate_data=pd.DataFrame({"date": dates, "rate": [RATE] * len(days)}),
        futures_data=pd.DataFrame(futures),
    )


def _monthly(cal, t0, months):
    out = []
    for k in range(1, months + 1):
        month = (t0.month - 1 + k) % 12 + 1
        year = t0.year + (t0.month - 1 + k) // 12
        d = datetime(year, month, min(t0.day, 28))
        while not cal.is_business_day(d):
            d += timedelta(days=1)
        out.append(d)
    return out


def dated_snowball(cal, t0, months=3, ko=104.0, ki=90.0, ko_rate=0.12, settlement=None):
    dates = _monthly(cal, t0, months)
    product = SnowballOption(
        initial_price=100.0, strike=100.0, contract_multiplier=1.0, initial_date=t0, exercise_date=dates[-1],
        barrier_config=BarrierConfig(
            ko_barrier=ko, ko_rate=ko_rate, ko_observation_type=ObservationType.DISCRETE,
            ko_observation_schedule=ObservationSchedule(
                records=[ObservationRecord(observation_date=d, barrier=ko) for d in dates]),
            ki_barrier=ki, ki_observation_type=ObservationType.DISCRETE,
            ki_observation_schedule=ObservationSchedule(
                records=[ObservationRecord(observation_date=d, barrier=ki) for d in dates])),
        payoff_config=PayoffConfig(rebate_rate=ko_rate, include_principal=False),
    )
    if settlement is not None:
        product.settlement_convention = settlement
        product._validate_settlement_convention()
    return product


def dated_ko_reset(cal, t0, pre_months=2, post_months=4, ki=90.0):
    pre, post = _monthly(cal, t0, pre_months), _monthly(cal, t0, post_months)
    rec = lambda d, b: ObservationRecord(observation_date=d, barrier=b)  # noqa: E731
    return KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, contract_multiplier=1.0, initial_date=t0, exercise_date=post[-1],
        barrier_config=BarrierConfig(
            ko_barrier=104.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
            ko_observation_schedule=ObservationSchedule(records=[rec(d, 104.0) for d in pre]),
            ki_barrier=ki, ki_observation_type=ObservationType.DISCRETE,
            ki_observation_schedule=ObservationSchedule(records=[rec(d, ki) for d in post])),
        post_barrier_config=BarrierConfig(
            ko_barrier=97.0, ko_rate=0.03, ko_observation_type=ObservationType.DISCRETE,
            ko_observation_schedule=ObservationSchedule(records=[rec(d, 97.0) for d in post])),
        payoff_config=PayoffConfig(rebate_rate=0.15, include_principal=False),
    )


ENGINES = {
    "quad": dict(pricing_engine_type=EngineType.QUADRATURE),
    # a total time-step budget: the default steps-per-business-day density scales with the env's
    # bus_days_in_year (252 on the calendar-day env, 365 on the AC1a clock), which would compare two grids
    "pde": dict(pricing_engine_type=EngineType.PDE, pde_params=PDEParams(grid=GridConfig(time_steps=120))),
    "mc": dict(pricing_engine_type=EngineType.MONTE_CARLO, mc_params=MCParams(num_paths=2000, seed=11),
               event_stats_engine_type=EngineType.QUADRATURE),
}


def _engine_config(kind, clock=None, **kw):
    return AutocallableEngineConfig(**ENGINES[kind], trading_clock=clock, **kw)


def _book(products, market, engine_config, **kw):
    kw.setdefault("calculate_event_probabilities", True)
    return ReplayBacktestConfig(
        products=[ReplayProduct(product=p, quantity=q, position_id=i + 1, has_lifecycle=True)
                  for i, (p, q) in enumerate(products)],
        market_data=market, hedge=HedgeSpec(kind="futures", multiplier=300.0), engine_config=engine_config,
        transaction_cost_model=ZeroCostModel(), underlying=UNDERLYING, terminate_on_lifecycle_end=False, **kw)


def _single(product, market, engine_config, **kw):
    kw.setdefault("calculate_event_probabilities", True)
    kw.setdefault("calculate_surfaces", False)
    return AutocallableBacktestConfig(
        product=product, market_data=market, engine_config=engine_config, transaction_cost_model=ZeroCostModel(),
        product_quantity=-1.0, underlying=UNDERLYING, terminate_on_lifecycle_end=False, **kw)


def _df(results, name):
    value = getattr(results, name)
    return value() if callable(value) else value


class _Spy:
    """Delegating pricing engine that records every (product, env, price) of the day loop."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = []

    def price(self, product, env):
        value = self.inner.price(product, env)
        self.calls.append((pd.Timestamp(env.valuation_date).normalize(), product, env, float(value)))
        return value

    def calculate_greeks(self, product, env):
        return self.inner.calculate_greeks(product, env)

    def __getattr__(self, name):
        return getattr(self.inner, name)


# --------------------------------------------------------------------------- D3: the clock-wrappable flat types


def test_clock_flat_types_are_library_curves_with_finiteness_only():
    from quantark.backtest.replay.market import ClockImpliedBasisYield, ClockSignedDividendYield
    from quantark.param.basis.basis_yield import BasisYield

    q = ClockSignedDividendYield(1.7)          # no library sanity bound: the implied carry near expiry is large
    assert isinstance(q, DividendYield) and q.get_yield(0.3) == 1.7
    assert not q.is_identically_zero() and ClockSignedDividendYield(0.0).is_identically_zero()
    assert ClockSignedDividendYield(-1e-300).is_identically_zero() is False
    assert q.parallel_shifted(0.1) == ClockSignedDividendYield(1.7 + 0.1)
    b = ClockImpliedBasisYield(-0.9)
    assert isinstance(b, BasisYield) and b.get_basis_yield(2.0) == -0.9
    assert ClockImpliedBasisYield(0.0).is_identically_zero() and not b.is_identically_zero()
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValidationError):
            ClockSignedDividendYield(bad)
        with pytest.raises(ValidationError):
            ClockImpliedBasisYield(bad)


# --------------------------------------------------------------------------- AC2: the default path keeps today's envs


def _replay(config):
    return ReplayBacktestEngine(config)._replays[0]


def test_default_build_env_is_todays_calendar_day_env():
    t0 = SSE_WINDOW[0]
    market = _market(SSE_WINDOW)
    replay = _replay(_book([(dated_snowball(SSE, t0), -1.0)], market, _engine_config("quad")))
    day = pd.Timestamp(SSE_WINDOW[1])
    selected = market.get_futures_slice(day).iloc[0]
    env, *_ = replay.build_env(day, market.get_market_row(day), selected)
    assert type(env.vol_surface) is FlatVolSurface and type(env.rate_curve) is FlatRateCurve
    assert type(env.div_yield) is SignedDividendYield and type(env.basis_yield) is ImpliedBasisYield
    assert env.day_count_convention is DayCountConvention.CALENDAR_DAYS
    assert env.calendar is None and env.bus_days_in_year == 252 and env.trading_clock_horizon is None
    assert env.holiday_time_adjustment is BusinessDayConvention.PRECEDING
    assert (env.rate_curve_quoting, env.div_yield_quoting, env.basis_yield_quoting) == (None, None, None)


# --------------------------------------------------------------------------- the clock env


def _clock(cal=SSE, days=244, **kw):
    return TradingClockConfig(calendar=cal, bus_days_in_year=days, **kw)


def _expected_horizon(market_days, last_product_date):
    return max(max(market_days), last_product_date) + timedelta(days=366)


@pytest.mark.parametrize("hta", [BusinessDayConvention.PRECEDING, BusinessDayConvention.FOLLOWING])
def test_clock_build_env_is_a_business_day_env_with_calendar_quoted_curves(hta):
    t0 = SSE_WINDOW[0]
    product = dated_snowball(SSE, t0)
    market = _market(SSE_WINDOW)
    clock = _clock(holiday_time_adjustment=hta)
    replay = _replay(_book([(product, -1.0)], market, _engine_config("quad", clock)))
    horizon = _expected_horizon(SSE_WINDOW, product.exercise_date)
    assert replay.trading_clock_horizon == horizon
    day = pd.Timestamp(SSE_WINDOW[2])
    selected = market.get_futures_slice(day).iloc[0]
    env, basis, implied_q, _ = replay.build_env(day, market.get_market_row(day), selected)
    assert env.day_count_convention is DayCountConvention.BUSINESS_DAYS
    assert env.calendar is SSE and env.bus_days_in_year == 244 and env.holiday_time_adjustment is hta
    assert (env.rate_curve_quoting, env.div_yield_quoting, env.basis_yield_quoting) == (CQ, CQ, CQ)
    assert env.trading_clock_horizon == horizon
    from quantark.backtest.replay.market import ClockImpliedBasisYield, ClockSignedDividendYield

    assert type(env.div_yield.inner) is ClockSignedDividendYield and env.div_yield.inner.div_yield == implied_q
    assert type(env.basis_yield.inner) is ClockImpliedBasisYield and env.basis_yield.inner.basis_yield == basis
    assert type(env.rate_curve.inner) is FlatRateCurve
    vol = env.vol_surface
    assert type(vol) is CalendarQuotedVolSurface and type(vol.inner) is FlatVolSurface
    assert isinstance(vol.time_map, DateKnotTimeMap)
    assert vol.time_map.clock.calendar is SSE and vol.time_map.clock.days_per_year == 244
    assert vol.time_map.holiday_time_adjustment is hta
    assert vol.time_map.anchor_date == day.to_pydatetime() and vol.time_map.horizon_date == horizon


def test_trading_year_vol_quoting_keeps_the_vol_native():
    market = _market(SSE_WINDOW)
    replay = _replay(_book([(dated_snowball(SSE, SSE_WINDOW[0]), -1.0)], market,
                           _engine_config("quad", _clock(vol_quoting=CurveQuotingBasis.TRADING_YEAR))))
    day = pd.Timestamp(SSE_WINDOW[1])
    env, *_ = replay.build_env(day, market.get_market_row(day), market.get_futures_slice(day).iloc[0])
    assert type(env.vol_surface) is FlatVolSurface
    assert env.day_count_convention is DayCountConvention.BUSINESS_DAYS


def test_the_probe_measures_the_remaining_maturity_on_the_env_clock():
    t0 = SSE_WINDOW[0]
    product = dated_snowball(SSE, t0)
    market = _market(SSE_WINDOW)
    replay = _replay(_book([(product, -1.0)], market, _engine_config("quad", _clock())))
    day = pd.Timestamp(SSE_WINDOW[3])
    env, *_ = replay.build_env(day, market.get_market_row(day), market.get_futures_slice(day).iloc[0])
    remaining = replay._remaining_maturity_years(day, market.get_market_row(day))
    assert remaining == env_time(env, product.exercise_date)
    assert remaining == SSE.count_business_days(day.to_pydatetime(), product.exercise_date, include_start=False) / 244


def test_the_book_horizon_covers_every_product():
    t0 = SSE_WINDOW[0]
    short, long_ = dated_snowball(SSE, t0, months=2), dated_snowball(SSE, t0, months=5)
    engine = ReplayBacktestEngine(_book([(short, -1.0), (long_, 1.0)], _market(SSE_WINDOW),
                                        _engine_config("quad", _clock())))
    horizon = _expected_horizon(SSE_WINDOW, long_.exercise_date)
    assert [r.trading_clock_horizon for r in engine._replays] == [horizon, horizon]


# --------------------------------------------------------------------------- AC6: refusals on the clock path


def test_a_numeric_schedule_product_is_refused_on_the_clock():
    from quantark.asset.equity.product.option import create_standard_snowball

    numeric = create_standard_snowball(
        initial_price=100.0, strike=100.0, maturity=0.25, contract_multiplier=1.0, ko_barrier=103.0,
        ki_barrier=90.0, ko_rate=0.1, num_observations=2, ko_observation_dates=[0.1, 0.25],
        ki_observation_type=ObservationType.DISCRETE, ki_continuous=False, ki_observation_dates=[0.1, 0.25],
        include_principal=False)
    with pytest.raises(ValidationError, match="numeric"):
        ReplayBacktestEngine(_book([(numeric, -1.0)], _market(SSE_WINDOW), _engine_config("quad", _clock())))
    ReplayBacktestEngine(_book([(numeric, -1.0)], _market(SSE_WINDOW), _engine_config("quad")))  # default: fine


def test_a_replay_day_off_the_clock_calendar_is_refused():
    days = [datetime(2026, 1, 9), datetime(2026, 1, 10), datetime(2026, 1, 12)]   # Fri, SAT, Mon
    engine = ReplayBacktestEngine(_book([(dated_snowball(SSE, days[0]), -1.0)], _market(days),
                                        _engine_config("quad", _clock()), calculate_event_probabilities=False))
    with pytest.raises(ValidationError, match="business day"):
        engine.run()


def test_the_mc_event_stats_fallback_lets_a_clock_refusal_through():
    from quantark.backtest.replay.product_replay import ProductReplay

    class Refusing:
        def calculate_event_stats(self, product, env):
            raise ValidationError("clock refusal")

    t0 = SSE_WINDOW[0]
    product = dated_snowball(SSE, t0)
    for clock, expect_raise in ((_clock(), True), (None, False)):
        replay = ProductReplay(
            product=product, product_quantity=-1.0, has_lifecycle=True, lifecycle=None, surface_engine=None,
            event_stats_engine=Refusing(), engine_config=_engine_config("quad", clock, event_stats_fallback="mc"),
            market_data=_market(SSE_WINDOW), start_date=pd.Timestamp(t0), underlying=UNDERLYING, actions_sink=[],
            event_prob_sink=[], daily_event_sink=[], surfaces_sink=[])
        day = pd.Timestamp(SSE_WINDOW[1])
        market = replay.market_data
        env, *_ = replay.build_env(day, market.get_market_row(day), market.get_futures_slice(day).iloc[0])
        if expect_raise:
            with pytest.raises(ValidationError, match="clock refusal"):
                replay._calculate_event_stats(product, env)
        else:
            _, source = replay._calculate_event_stats(product, env)
            assert source == "mc_fallback"


def test_the_dashboard_refuses_to_reprice_a_trading_clock_result():
    from quantark.backtest.replay.dashboard import AutocallableBacktestDashboard

    days = SSE_WINDOW[:2]
    results = AutocallableBacktestEngine(_single(dated_snowball(SSE, days[0]), _market(days),
                                                 _engine_config("quad", _clock()),
                                                 calculate_event_probabilities=False)).run()
    with pytest.raises(ValidationError, match="trading clock"):
        AutocallableBacktestDashboard(results)._build_risk_snapshots(_df(results, "states_df"))


def test_results_metadata_records_the_clock():
    days = SSE_WINDOW[:2]
    clock = _clock()
    product = dated_snowball(SSE, days[0])
    summary = AutocallableBacktestEngine(_single(product, _market(days), _engine_config("quad", clock),
                                                 calculate_event_probabilities=False)).run().get_summary()
    info = summary["trading_clock"]
    assert info["day_count_convention"] == "BUSINESS_DAYS"
    assert info["calendar"] == SSE.name and info["bus_days_in_year"] == 244
    assert info["holiday_time_adjustment"] == "PRECEDING" and info["vol_quoting"] == "CALENDAR_YEAR"
    assert info["trading_clock_horizon"] == _expected_horizon(days, product.exercise_date).date().isoformat()
    default = AutocallableBacktestEngine(_single(dated_snowball(SSE, days[0]), _market(days),
                                                 _engine_config("quad"),
                                                 calculate_event_probabilities=False)).run().get_summary()
    assert "trading_clock" not in default


# --------------------------------------------------------------------------- AC1a: the independent anchor


_GREEK_COLS = ("price", "delta", "gamma", "vega", "theta", "rho", "dividend_sensitivity", "basis_sensitivity")


def _assert_frames_close(a: pd.DataFrame, b: pd.DataFrame, rel=1e-12):
    assert list(a.columns) == list(b.columns) and len(a) == len(b)
    for col in a.columns:
        x, y = a[col], b[col]
        if pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x):
            xv, yv = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
            assert np.array_equal(np.isnan(xv), np.isnan(yv)), col
            ok = ~np.isnan(xv)
            scale = np.maximum(np.abs(xv[ok]), np.abs(yv[ok]))
            assert np.all(np.abs(xv[ok] - yv[ok]) <= rel * scale + 1e-300), (col, xv, yv)
        else:
            assert list(x) == list(y), col


def _frames(results):
    out = {}
    for name in ("states_df", "greeks_df", "daily_event_summary_df", "event_probability_df", "surfaces_df"):
        frame = _df(results, name)
        out[name] = frame.drop(columns=[c for c in ("pricing_seconds",) if c in frame.columns])
    return out


@pytest.mark.parametrize("kind", ["quad", "pde", "mc"])
@pytest.mark.parametrize("shape", ["single", "book"])
def test_ac1a_an_all_business_day_clock_at_365_is_the_default_replay(kind, shape):
    t0 = ALL_WINDOW[0]
    clock = _clock(cal=ALL_DAYS, days=365)

    def run(clock_or_none):
        market = _market(ALL_WINDOW)
        a = dated_snowball(ALL_DAYS, t0)
        if shape == "single":
            # the spot x q surface grid rides along on QUAD (it reprices on the clock-path flat q type)
            return AutocallableBacktestEngine(_single(a, market, _engine_config(kind, clock_or_none),
                                                      calculate_surfaces=kind == "quad")).run()
        b = dated_snowball(ALL_DAYS, t0, months=4, ko=103.0, ki=88.0)
        return ReplayBacktestEngine(_book([(a, -1.0), (b, -2.0)], market, _engine_config(kind, clock_or_none))).run()

    on_clock, default = _frames(run(clock)), _frames(run(None))
    for name in on_clock:
        _assert_frames_close(on_clock[name], default[name])
    assert len(on_clock["greeks_df"]) == len(ALL_WINDOW)


# --------------------------------------------------------------------------- AC1: replay day == hand-built env


def _hand_env(day, market, clock, horizon):
    row = market.get_market_row(day)
    fut = market.get_futures_slice(day).iloc[0]
    ttm = (pd.Timestamp(fut["expiry_date"]).normalize() - day).days / 365.0
    basis, q = derive_implied_dividend_yield(rate=row["rate"], spot=row["spot"],
                                             futures_price=float(fut["futures_price"]), time_to_maturity=ttm)
    tmap = DateKnotTimeMap(TradingClock(clock.calendar, clock.bus_days_in_year), day.to_pydatetime(), horizon,
                           clock.holiday_time_adjustment)
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=row["spot"], asset_name=UNDERLYING),
        vol_surface=CalendarQuotedVolSurface(FlatVolSurface(row["volatility"]), tmap),
        rate_curve=FlatRateCurve(row["rate"]), div_yield=ContinuousDividendYield(q),
        basis_yield=FlatBasisYield(basis), valuation_date=day.to_pydatetime(),
        day_count_convention=DayCountConvention.BUSINESS_DAYS, bus_days_in_year=clock.bus_days_in_year,
        calendar=clock.calendar, holiday_time_adjustment=clock.holiday_time_adjustment, rate_curve_quoting=CQ,
        div_yield_quoting=CQ, basis_yield_quoting=CQ, trading_clock_horizon=horizon)


@pytest.mark.parametrize("kind, make", [
    ("quad", dated_snowball), ("pde", dated_snowball), ("mc", dated_snowball), ("quad", dated_ko_reset)],
    ids=["snowball-quad", "snowball-pde", "snowball-mc", "ko_reset-quad"])
def test_ac1_each_replay_day_prices_as_a_hand_built_business_day_env(kind, make):
    assert not any(SSE.is_business_day(datetime(2026, 1, d)) for d in (1, 2, 3, 4))   # the holiday is inside
    t0 = SSE_WINDOW[0]
    product = make(SSE, t0)
    market = _market(SSE_WINDOW)
    clock = _clock()
    config = _engine_config(kind, clock)
    engine = ReplayBacktestEngine(_book([(product, -1.0)], market, config, calculate_event_probabilities=False))
    spy = _Spy(engine._pricing_engines[0])
    engine._pricing_engines[0] = spy
    engine.run()
    horizon = _expected_horizon(SSE_WINDOW, product.exercise_date)
    priced_days = sorted({d for d, *_ in spy.calls})
    assert priced_days == [pd.Timestamp(d) for d in SSE_WINDOW]
    for day, priced, env, value in spy.calls:
        hand = _hand_env(day, market, clock, horizon)
        expected = float(create_pricing_engine(priced, config).price(priced, hand))
        assert value == pytest.approx(expected, rel=1e-12, abs=0.0), day


# --------------------------------------------------------------------------- AC3: _env_with keeps the clock


def test_env_with_keeps_the_clock_fields_and_wraps_a_replacement_dividend():
    from quantark.backtest.replay.product_replay import _env_with

    market = _market(SSE_WINDOW)
    replay = _replay(_book([(dated_snowball(SSE, SSE_WINDOW[0]), -1.0)], market, _engine_config("quad", _clock())))
    day = pd.Timestamp(SSE_WINDOW[2])
    env, *_ = replay.build_env(day, market.get_market_row(day), market.get_futures_slice(day).iloc[0])
    native = ContinuousDividendYield(0.03)
    bumped = _env_with(env, spot=101.0, div_yield=native, underlying=UNDERLYING)
    for name in ("day_count_convention", "calendar", "bus_days_in_year", "holiday_time_adjustment",
                 "rate_curve_quoting", "div_yield_quoting", "basis_yield_quoting", "trading_clock_horizon"):
        assert getattr(bumped, name) is getattr(env, name), name
    assert bumped.vol_surface is env.vol_surface and bumped.rate_curve is env.rate_curve
    assert bumped.basis_yield is env.basis_yield
    assert bumped.div_yield.inner is native and bumped.div_yield.time_map.anchor_date == env.valuation_date
    assert float(bumped.spot) == 101.0
    same = _env_with(env, spot=99.0, underlying=UNDERLYING)
    assert same.div_yield is env.div_yield


# --------------------------------------------------------------------------- D5: env-time conversions


def test_event_dates_are_the_business_dates_of_the_engine_times():
    t0 = SSE_WINDOW[0]
    product = dated_snowball(SSE, t0)
    results = ReplayBacktestEngine(_book([(product, -1.0)], _market(SSE_WINDOW), _engine_config("quad", _clock()))
                                   ).run()
    events = _df(results, "event_probability_df")
    ko_dates = {pd.Timestamp(r.observation_date) for r in product.barrier_config.ko_observation_schedule.records}
    ko = events[events["event_type"] == "KO"]
    assert len(ko) and set(pd.to_datetime(ko["event_date"])) <= ko_dates


def test_a_time_on_no_business_date_is_refused():
    market = _market(SSE_WINDOW)
    replay = _replay(_book([(dated_snowball(SSE, SSE_WINDOW[0]), -1.0)], market, _engine_config("quad", _clock())))
    day = pd.Timestamp(SSE_WINDOW[1])
    env, *_ = replay.build_env(day, market.get_market_row(day), market.get_futures_slice(day).iloc[0])
    assert replay._event_date(day, 3.0 / 244.0, env) == pd.Timestamp(SSE.add_business_days(day.to_pydatetime(), 3))
    with pytest.raises(ValidationError, match="business date"):
        replay._event_date(day, 3.5 / 244.0, env)


def test_pending_receivable_discounts_on_the_env_clock():
    market = _market(SSE_WINDOW)
    replay = _replay(_book([(dated_snowball(SSE, SSE_WINDOW[0]), -1.0)], market, _engine_config("quad", _clock())))
    day = pd.Timestamp(SSE_WINDOW[1])
    env, *_ = replay.build_env(day, market.get_market_row(day), market.get_futures_slice(day).iloc[0])
    settle = datetime(2026, 1, 12)
    lifecycle = replay.lifecycle
    lifecycle.pending_settlement_cashflow = 5.0
    lifecycle.settlement_date = settle
    expected = 5.0 * env.get_discount_factor(env_time(env, settle))
    assert replay.pending_receivable_pv(env, day) == expected


# --------------------------------------------------------------------------- AC9: horizon + pricing_q


def test_ac9_a_window_ending_before_maturity_with_t_plus_n_settlement_runs():
    t0 = SSE_WINDOW[0]
    lag = SettlementConvention(lag=5, lag_unit=SettlementLagUnit.BUSINESS_DAYS, calendar=SSE)
    product = dated_snowball(SSE, t0, settlement=lag)
    engine = ReplayBacktestEngine(_book([(product, -1.0)], _market(SSE_WINDOW), _engine_config("quad", _clock())))
    horizon = engine._replays[0].trading_clock_horizon
    assert horizon >= SSE.add_business_days(product.exercise_date, 5) + timedelta(days=366)
    results = engine.run()
    assert len(_df(results, "states_df")) == len(SSE_WINDOW)


def test_ac9_pricing_q_on_the_clock_is_the_calendar_quote():
    t0 = SSE_WINDOW[0]
    expiries = (datetime(2026, 1, 16), datetime(2026, 2, 20), datetime(2026, 3, 20), FUT_EXPIRY)

    def states(clock):
        config = _engine_config("quad", clock, dividend_source="futures_curve")
        return ReplayBacktestEngine(_book([(dated_snowball(SSE, t0), -1.0)], _market(SSE_WINDOW, expiries=expiries),
                                          config, calculate_event_probabilities=False)).run()

    on_clock, default = _df(states(_clock()), "states_df"), _df(states(None), "states_df")
    assert np.array_equal(on_clock["pricing_q"].to_numpy(), default["pricing_q"].to_numpy())
    assert on_clock["pricing_q"].nunique() > 1


# --------------------------------------------------------------------------- AC4: holidays move time by the convention


def _with_extra_holiday(cal, day):
    return Calendar(holidays=set(cal.holidays) | {day}, weekend_days=set(cal.weekend_days), name=cal.name + "+1")


@pytest.mark.parametrize("hta", [BusinessDayConvention.PRECEDING, BusinessDayConvention.FOLLOWING])
def test_ac4_a_dcn_date_on_a_clock_holiday_sits_where_the_convention_puts_it(hta):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parents[1]))
    from dcn_fixtures import SSE as DCN_SSE  # noqa: E402
    from dcn_run_fixtures import BOOKED, booked_dcn  # noqa: E402

    from quantark.asset.equity.product.option.dcn_grid import build_dcn_grid_context

    dcn = booked_dcn()
    row = next(m.observation_date for m in dcn.schedule.monthly if m.observation_date > BOOKED + timedelta(days=5))
    cal = _with_extra_holiday(DCN_SSE, row)                    # the clock closes a DCN observation date
    days = [d for d in (BOOKED + timedelta(days=k) for k in range(5)) if cal.is_business_day(d)][:2]
    clock = TradingClockConfig(calendar=cal, bus_days_in_year=244, holiday_time_adjustment=hta)
    config = AutocallableEngineConfig(pricing_engine_type=EngineType.MONTE_CARLO,
                                      surface_engine_type=EngineType.MONTE_CARLO,
                                      event_stats_engine_type=EngineType.MONTE_CARLO,
                                      mc_params=MCParams(num_paths=256, seed=3, use_qmc=True), trading_clock=clock)
    engine = ReplayBacktestEngine(ReplayBacktestConfig(
        products=[ReplayProduct(product=dcn, quantity=1.0, position_id=1, has_lifecycle=True)],
        market_data=_market(days, spots=[5500.0] * len(days), expiries=(datetime(2023, 6, 16),)), hedge=HedgeSpec(kind="spot", multiplier=1.0),
        engine_config=config, transaction_cost_model=ZeroCostModel(), underlying=UNDERLYING,
        fixed_dividend_yield=0.10, calculate_event_probabilities=False, terminate_on_lifecycle_end=False))
    spy = _Spy(engine._pricing_engines[0])
    engine._pricing_engines[0] = spy
    engine.run()
    adjusted = cal.adjust_date(row, hta)
    assert adjusted != row
    for day, priced, env, _ in spy.calls:
        assert env.holiday_time_adjustment is hta and env.calendar is cal
        anchor = priced.schedule.valuation_date
        oracle = cal.count_business_days(anchor, adjusted, include_start=False) / 244.0
        assert env_time(env, row, anchor=anchor) == oracle
        grid = build_dcn_grid_context(priced, env)
        col = next(i for i, ds in enumerate(grid.column_dates) if row in ds)
        assert grid.times[col] == oracle


def test_ac4_a_snowball_date_on_a_clock_holiday_collapses_to_the_previous_business_day():
    t0 = SSE_WINDOW[0]
    product = dated_snowball(SSE, t0)
    first = product.barrier_config.ko_observation_schedule.records[0].observation_date
    cal = _with_extra_holiday(SSE, first)
    market = _market(SSE_WINDOW)
    for hta in (BusinessDayConvention.PRECEDING, BusinessDayConvention.FOLLOWING):
        replay = _replay(_book([(copy.deepcopy(product), -1.0)], market,
                               _engine_config("quad", TradingClockConfig(calendar=cal, bus_days_in_year=244,
                                                                         holiday_time_adjustment=hta))))
        day = pd.Timestamp(SSE_WINDOW[1])
        env, *_ = replay.build_env(day, market.get_market_row(day), market.get_futures_slice(day).iloc[0])
        t = product.barrier_config.ko_observation_schedule.records[0].resolve_time(env)
        previous = cal.adjust_date(first, BusinessDayConvention.PRECEDING)
        assert t == cal.count_business_days(day.to_pydatetime(), previous, include_start=False) / 244.0


# --------------------------------------------------------------------------- D6: flat_atm_remaining on the clock


def test_d6_flat_atm_remaining_samples_the_artifact_at_the_calendar_tenor(tmp_path):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parents[1]))
    from golden.replay_golden import fixtures  # noqa: E402

    from quantark.param.vol.surface_history import VolSurfaceHistory

    history = VolSurfaceHistory(fixtures.write_localvol_history(tmp_path))
    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2023, 2026))
    days = [datetime(2024, 1, 2), datetime(2024, 1, 3), datetime(2024, 1, 4)]
    t0 = days[0]

    def run(clock):
        product = dated_snowball(cal, t0, months=8)   # tenor between the artifact's 0.5 and 1.0 pillars
        market = _market(days)
        market.surface_history = history
        config = _engine_config("quad", clock, vol_source="surface", surface_vol_mode="flat_atm_remaining")
        engine = ReplayBacktestEngine(_book([(product, -1.0)], market, config, calculate_event_probabilities=False))
        spy = _Spy(engine._pricing_engines[0])
        engine._pricing_engines[0] = spy
        engine.run()
        return product, spy.calls

    product, on_clock = run(_clock(cal=cal))
    _, default = run(None)
    assert len(on_clock) == len(default) >= len(days)
    for (day, _, env, _), (dday, _, denv, _) in zip(on_clock, default):
        assert day == dday
        vol = env.vol_surface
        assert type(vol) is CalendarQuotedVolSurface and type(vol.inner) is FlatVolSurface
        # the calendar ATM sample: the default replay's flat vol, bitwise
        assert vol.inner.volatility == denv.vol_surface.volatility
        atm = history.surface_for(day).term_structure_vol_surface()
        tenor = (product.exercise_date - day.to_pydatetime()).days / 365.0
        assert vol.inner.volatility == float(atm.get_vol(0.0, tenor, 0.0))
        assert env.div_yield.inner.__class__ is denv.div_yield.__class__
