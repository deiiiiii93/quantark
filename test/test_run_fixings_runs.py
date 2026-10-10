"""Run fixings through every run path (spec docs/superpowers/specs/2026-10-09-run-fixings-design.md §7 X1-X3, X6)."""
import copy
import warnings
from datetime import timedelta

import pytest

import run_fixings_fixtures as F
from quantark.util.exceptions import ValidationError
from quantark.util.fixings import MissingFixingWarning
from quantark.util.numerical import is_close


def _rows(recorder, pid=None):
    return [(f.index, f.outcome, f.spot, f.gap) for f in recorder.recorded if pid is None or f.position_id == pid]


def _hand_rows(kind, spot_of=F.spot_on):
    """A DAYS-day run's recordings: every fixing dated on/before day DAYS - 2 (the last day's is still live), at its
    own day's spot, never a gap."""
    outs = F.hand_outcomes(kind, F.DAYS - 2, spot_of)
    return [(i, outs[i], spot_of(F.day_of(F.fixing_dates()[i])), False) for i in sorted(outs)]


# ================================================================ equity backtest
def _backtest(product, kind, adapter, tmp_path, frequency="D"):
    from quantark.backtest import BacktestConfig, BacktestEngine, ZeroCostModel
    from quantark.backtest.strategy import DeltaNeutralStrategy
    from quantark.portfolio import Position

    cfg = BacktestConfig(
        strategy=DeltaNeutralStrategy(delta_threshold=1e12), start_date=F.START,
        end_date=adapter.dates[-1].to_pydatetime(), underlying=F.UNDERLYING,
        initial_positions=[Position(product=product, quantity=1.0, entry_price=0.0, underlying=F.UNDERLYING,
                                    engine=F.ENGINES[kind](), entry_timestamp=F.START)],
        market_data_adapter=adapter, transaction_cost_model=ZeroCostModel(), handle_lifecycle_events=False,
        frequency=frequency, results_path=str(tmp_path),
    )
    engine = BacktestEngine(cfg)
    return engine, engine.run()


@pytest.mark.parametrize("kind", ["ra", "asian"])
def test_equity_backtest_x1_x2_x6(kind, tmp_path):
    booked = F.MAKERS[(kind, "dated")]()
    before = copy.deepcopy(booked)
    (eng_d, res_d), warns = F.count_missing(_backtest, booked, kind, F.DailyAdapter(F.DAYS), tmp_path)
    eng_n, res_n = _backtest(F.MAKERS[(kind, "numeric")](), kind, F.DailyAdapter(F.DAYS), tmp_path)
    values_d, values_n = res_d.get_value_series().tolist(), res_n.get_value_series().tolist()
    engine = F.ENGINES[kind]()
    for k in range(F.DAYS):                                                      # X1a: bitwise vs by hand
        ref = F.quiet_price(engine, F.by_hand(kind, k - 1), F.eq_env(k))
        assert values_d[k] == ref, (k, values_d[k], ref)
        assert is_close(values_n[k], values_d[k], rel_tol=1e-12)                 # X2 values
    assert warns == 0                                                            # X1c
    assert _rows(eng_d.run_fixings) == _rows(eng_n.run_fixings) == _hand_rows(kind)  # X2 sequences, X1b spot
    pid = list(eng_d.portfolio.positions)[0]                                     # X6
    assert eng_d.portfolio.positions[pid].product.observation_records == F.by_hand(kind, F.DAYS - 2).observation_records
    assert booked.observation_records == before.observation_records             # X3g


def test_equity_backtest_x1b_the_fixing_does_not_move_later(tmp_path):
    other = lambda k: F.spot_on(k) if k <= 5 else 80.0                          # noqa: E731  differs from day 6 on
    a, _ = _backtest(F.ra_dated(), "ra", F.DailyAdapter(F.DAYS), tmp_path)
    b, _ = _backtest(F.ra_dated(), "ra", F.DailyAdapter(F.DAYS, spot_of=other), tmp_path)
    assert _rows(a.run_fixings)[0] == _rows(b.run_fixings)[0] == (0, True, 105.0, False)


def test_equity_backtest_x3b_weekly_gap_takes_the_previous_instant(tmp_path):
    adapter = F.DailyAdapter(5, freq="W-MON")                                  # days 0, 7, 14, 21, 28
    (eng, res), warns = F.count_missing(_backtest, F.ra_dated(), "ra", adapter, tmp_path, "W")
    assert warns == 0
    assert _rows(eng.run_fixings) == [(0, True, 105.0, True), (1, False, 115.0, True)]
    assert res.get_value_series().tolist()[3] == F.quiet_price(F.ENGINES["ra"](), F.ra_dated({0: True, 1: False}),
                                                               F.eq_env(21))


def test_equity_backtest_x3e_hourly_refuses_a_participant(tmp_path):
    with pytest.raises(ValidationError, match="time of day"):
        _backtest(F.ra_dated(), "ra", F.DailyAdapter(3, freq="h"), tmp_path, "H")


def test_equity_backtest_numeric_participant_no_longer_warns_constant_maturity(tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _backtest(F.ra_numeric(), "ra", F.DailyAdapter(6), tmp_path)
    assert not [w for w in caught if "no roll rule" in str(w.message)]


def _short(kind, form, expiry_day=10):
    """A twin expiring on run day ``expiry_day`` with fixings on days 4 and 8."""
    from quantark.asset.equity.product.option import RangeAccrualObservationRecord, RangeAccrualOption
    from quantark.asset.equity.product.option.asian_option import AsianObservationRecord, AsianOption
    from quantark.util.enum import AsianStrikeType, AveragingType, OptionType

    days = (4, 8)
    if form == "dated":
        term = dict(exercise_date=F.START + timedelta(days=expiry_day))
        when = [dict(observation_date=F.START + timedelta(days=d)) for d in days]
    else:
        term = dict(maturity=expiry_day / 365.0)
        when = [dict(observation_time=d / 365.0) for d in days]
    if kind == "ra":
        extra = dict(initial_date=F.START) if form == "dated" else {}
        return RangeAccrualOption(initial_price=100.0, range_config=F._range_config(), **term, **extra,
                                  observation_records=[RangeAccrualObservationRecord(**w) for w in when])
    return AsianOption(strike=100.0, option_type=OptionType.CALL, asian_strike_type=AsianStrikeType.FIXED,
                       averaging_type=AveragingType.ARITHMETIC, **term,
                       observation_records=[AsianObservationRecord(**w) for w in when])


@pytest.mark.parametrize("form", ["dated", "numeric"])
@pytest.mark.parametrize("kind", ["ra", "asian"])
def test_equity_backtest_twins_refuse_a_run_past_the_maturity(kind, form, tmp_path):
    # A numeric participant now ages, so a run past its maturity is refused as the dated twin's always was (spec §8
    # row 2 + standing twin rule; settlement in runs is a follow-up, spec §10). Both run up to the day before it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _backtest(_short(kind, form), kind, F.DailyAdapter(10), tmp_path)        # days 0..9
        with pytest.raises(ValidationError):
            _backtest(_short(kind, form), kind, F.DailyAdapter(14), tmp_path)    # days 0..13


# ================================================================ equity dynamic scenario
def _eq_dynamic(products, spot_of=F.spot_on, n=F.DAYS, start=F.START, env=None):
    import contextlib
    import io

    from quantark.dynamicscenario import DynamicScenarioConfig, DynamicScenarioEngine
    from quantark.dynamicscenario.path.day_path import DayPath, DayStep, ParameterChange
    from quantark.stresstest.stress.stress_types import StressType

    pf = F.eq_book(*products, env=env)
    steps = [DayStep(day_index=k, changes=[ParameterChange("spot", StressType.VALUE, spot_of(k))]) for k in range(n)]
    engine = DynamicScenarioEngine(DynamicScenarioConfig(calculate_greeks=True, handle_lifecycle_events=False))
    with contextlib.redirect_stdout(io.StringIO()):
        res = engine.run(pf, DayPath(name="w6", steps=steps, start_date=start))
    return engine, res, pf


def _greeks(product, env):
    from quantark.asset.equity.riskmeasures import GreeksCalculator
    from quantark.dynamicscenario import DynamicScenarioConfig
    use_analytical = DynamicScenarioConfig().greeks_method == "analytical"      # the run's own route
    return F.eq_book(product, env=env).get_portfolio_greeks(GreeksCalculator(), use_analytical=use_analytical)


@pytest.mark.parametrize("kind", ["ra", "asian"])
def test_equity_dynamic_x1_x2_x3g_x6(kind):
    booked = F.MAKERS[(kind, "dated")]()
    before = copy.deepcopy(booked)
    (eng_d, res_d, pf), warns = F.count_missing(_eq_dynamic, [booked])
    eng_n, res_n, _ = _eq_dynamic([F.MAKERS[(kind, "numeric")]()])
    engine = F.ENGINES[kind]()
    for k in range(F.DAYS):
        ref = F.quiet_price(engine, F.by_hand(kind, k - 1), F.eq_env(k))
        assert res_d.day_results[k].portfolio_value == ref, k                    # X1a
        assert is_close(res_n.day_results[k].portfolio_value, ref, rel_tol=1e-12)  # X2
    assert warns == 0                                                            # X1c
    assert _rows(eng_d.run_fixings) == _rows(eng_n.run_fixings) == _hand_rows(kind)
    assert pf.positions[list(pf.positions)[0]].product is booked                 # X3g: input book untouched
    assert booked.observation_records == before.observation_records


def test_equity_dynamic_x3h_greeks_equal_the_hand_recorded_contract():
    _, res, _ = _eq_dynamic([F.ra_dated()])
    for k in (5, 10, 14):
        got, want = res.day_results[k].greeks, _greeks(F.by_hand("ra", k - 1), F.eq_env(k))
        assert {g: got[g] for g in ("delta", "gamma", "vega", "theta")} == {g: want[g] for g in ("delta", "gamma", "vega", "theta")}


def test_equity_dynamic_x3a_todays_fixing_is_live_then_history_at_its_day():
    spot_of = lambda k: 109.9 if k == F.FIX_DAY else F.spot_on(k)               # noqa: E731
    eng, res, _ = _eq_dynamic([F.ra_dated()], spot_of=spot_of, n=8)
    live = _greeks(F.ra_dated(), F.eq_env(F.FIX_DAY, spot=109.9))
    assert res.day_results[F.FIX_DAY].greeks["delta"] == live["delta"]
    from quantark.asset.equity.riskmeasures import GreeksCalculator
    g1 = GreeksCalculator().calculate(F.ra_dated(), F.eq_env(F.FIX_DAY, spot=109.9), F.ENGINES["ra"]())
    assert is_close(g1["delta"], -1.057904, rel_tol=0.0, abs_tol=5e-7)          # the spec's G1 number
    assert _rows(eng.run_fixings)[0] == (0, True, 109.9, False)


def test_equity_dynamic_x3b_a_numeric_fixing_between_days_takes_the_previous_day():
    times = [4.5 / 365.0, 18 / 365.0, 120 / 365.0, 240 / 365.0, 1.0]
    eng, _, _ = _eq_dynamic([F.ra_numeric(times=times)], n=7, spot_of=lambda k: 100.0 + k)
    assert _rows(eng.run_fixings) == [(0, True, 104.0, True)]


def test_equity_dynamic_x3c_history_warns_once_and_the_first_day_is_live():
    dates = [F.START - timedelta(days=3), F.START] + F.fixing_dates()[1:]
    prod = F.ra_dated(dates=dates, initial_date=F.START - timedelta(days=10))
    (eng, _, _), warns = F.count_missing(_eq_dynamic, [prod], n=3)
    assert warns == 1
    assert _rows(eng.run_fixings) == [(1, True, 105.0, False)]


def test_equity_dynamic_x3d_undated_path_records_nothing_and_values_as_frozen_at_start():
    eng, res, _ = _eq_dynamic([F.ra_dated()], start=None)
    assert eng.run_fixings.recorded == ()
    for k in range(F.DAYS):
        assert res.day_results[k].portfolio_value == F.quiet_price(F.ENGINES["ra"](), F.ra_dated(),
                                                                   F.eq_env(0, spot=F.spot_on(k)))


def test_equity_dynamic_x3e_time_stamped_dates_refuse_a_participant():
    with pytest.raises(ValidationError, match="time of day"):
        _eq_dynamic([F.ra_dated()], start=F.START + timedelta(hours=15), n=3)


# ================================================================ FX backtest + FX dynamic scenario
import pandas as pd  # noqa: E402

FX_KINDS = ["domestic", "foreign", "quanto"]


def _fx_backtest(products, index=None, env=None, spot_of=F.fx_spot):
    from quantark.backtest import FXBacktestConfig, FXBacktestEngine
    from quantark.backtest.strategy import FXDeltaNeutralStrategy
    from quantark.backtest.transaction_costs import ZeroCostModel

    index = index if index is not None else pd.date_range(F.START, periods=F.DAYS, freq="D")
    cfg = FXBacktestConfig(portfolio=F.fx_book(*products, env=env), market_path=F.fx_path(index, spot_of),
                           strategy=FXDeltaNeutralStrategy(delta_threshold=1e12, rebalance_frequency="on_threshold"),
                           transaction_cost_model=ZeroCostModel())
    engine = FXBacktestEngine(cfg)
    values = [r["portfolio_value"] for r in engine.run().rows]
    return engine, values, cfg.portfolio, cfg.market_path["EURUSD_spot"].tolist()


def _fx_dynamic(products, start=F.START, env=None):
    from quantark.dynamicscenario import FXDynamicScenarioConfig, FXDynamicScenarioEngine
    from quantark.dynamicscenario.path.day_path import DayPath, DayStep, ParameterChange
    from quantark.stresstest.stress.stress_types import StressType

    steps = [DayStep(day_index=k, changes=[ParameterChange("spot", StressType.VALUE, F.fx_spot(k))])
             for k in range(F.DAYS)]
    engine = FXDynamicScenarioEngine(FXDynamicScenarioConfig(calculate_greeks=True))
    pf = F.fx_book(*products, env=env)
    res = engine.run(pf, DayPath(name="w6fx", steps=steps, start_date=start))
    # the run's quoted spots (R1): the engine applies a VALUE spot change as a return (bump_env), so a day's spot can
    # sit an ULP off fx_spot(k); the hand references use the spot the run quoted
    spots = [d.market_state.spot["EURUSD"] for d in res.day_results]
    return engine, [d.portfolio_value for d in res.day_results], pf, spots


def _fx_hand_rows(kind, spot_of=F.fx_spot):
    outs = F.fx_hand_outcomes(kind, F.DAYS - 2, spot_of)
    return [(i, outs[i], spot_of(F.day_of(F.fixing_dates()[i])), False) for i in sorted(outs)]


def _spy_last_advanced(monkeypatch):
    """Record the portfolio of every ``RunFixingRecorder.advance`` (the run's working copy), calling the original."""
    from quantark.util.run_fixings import RunFixingRecorder
    seen = []
    original = RunFixingRecorder.advance

    def spy(self, portfolio):
        seen.append(portfolio)
        return original(self, portfolio)

    monkeypatch.setattr(RunFixingRecorder, "advance", spy)
    return seen


def _final_product(seen):
    pf = seen[-1]
    (position,) = pf.positions.values()
    return position.product


def _assert_x6(final, want, aged_by=0.0):
    """X6 record by record: outcomes and every field bitwise; a numeric record's time to its aged hand time."""
    import dataclasses
    got_recs, want_recs = final.get_observation_records(), want.get_observation_records()
    assert len(got_recs) == len(want_recs)
    for got, ref in zip(got_recs, want_recs):
        g, w = dataclasses.asdict(got), dataclasses.asdict(ref)
        if w["observation_time"] is not None:
            assert is_close(g.pop("observation_time"), w.pop("observation_time") - aged_by, rel_tol=0.0,
                            abs_tol=1e-15)
        assert g == w


@pytest.mark.parametrize("run", [_fx_backtest, _fx_dynamic], ids=["backtest", "dynamic"])
@pytest.mark.parametrize("kind", FX_KINDS)
def test_fx_runs_x1_x2_x3g_x6(run, kind, monkeypatch):
    q = kind == "quanto"
    booked = F.fx_ra(kind, "dated")
    booked_n = F.fx_ra(kind, "numeric")
    before = copy.deepcopy(booked)
    seen = _spy_last_advanced(monkeypatch)
    (eng_d, values_d, pf, spots), warns = F.count_missing(run, [booked], env=F.fx_env(0, quanto=q))
    final_d = _final_product(seen)
    eng_n, values_n, _, spots_n = run([booked_n], env=F.fx_env(0, quanto=q))
    final_n = _final_product(seen)
    assert spots_n == spots and all(is_close(s, F.fx_spot(k), rel_tol=1e-15) for k, s in enumerate(spots))
    spot_of = spots.__getitem__
    engine = F.FX_CLASSES[kind][1]()
    for k in range(F.DAYS):
        hand = F.fx_ra(kind, "dated", F.fx_hand_outcomes(kind, k - 1, spot_of))
        ref = F.quiet_price(engine, hand, F.fx_env(k, spot=spots[k], quanto=q))
        assert values_d[k] == ref, (k, values_d[k], ref)                        # X1a
        assert is_close(values_n[k], ref, rel_tol=1e-12)                         # X2
    assert warns == 0                                                            # X1c
    assert _rows(eng_d.run_fixings) == _rows(eng_n.run_fixings) == _fx_hand_rows(kind, spot_of)
    hand = F.fx_hand_outcomes(kind, F.DAYS - 2, spot_of)                         # X6, both forms
    _assert_x6(final_d, booked.with_fixings_recorded(hand))
    _assert_x6(final_n, booked_n.with_fixings_recorded(hand), aged_by=(F.DAYS - 1) / 365.0)
    assert is_close(final_n.maturity, 1.0 - (F.DAYS - 1) / 365.0, rel_tol=0.0, abs_tol=1e-15)
    assert pf.positions[list(pf.positions)[0]].product is booked
    assert booked.observation_records == before.observation_records             # X3g
    assert booked_n.observation_records == F.fx_ra(kind, "numeric").observation_records


def test_fx_backtest_x1b_the_fixing_does_not_move_later():
    other = lambda k: F.fx_spot(k) if k <= 5 else 1.0                           # noqa: E731
    a, *_ = _fx_backtest([F.fx_ra("domestic", "dated")])
    b, *_ = _fx_backtest([F.fx_ra("domestic", "dated")], spot_of=other)
    assert _rows(a.run_fixings)[0] == _rows(b.run_fixings)[0] == (0, True, 1.25, False)


def test_fx_backtest_x3b_a_saturday_fixing_takes_fridays_spot_with_gap():
    sat = F.START + timedelta(days=5)                                           # 2024-01-06
    prod = F.fx_ra("domestic", "dated", dates=[sat] + F.fixing_dates()[1:])
    index = pd.bdate_range(F.START, periods=10)
    (eng, *_), warns = F.count_missing(_fx_backtest, [prod], index=index, spot_of=lambda k: 1.20 + 0.01 * k)
    assert warns == 0
    assert _rows(eng.run_fixings)[0] == (0, True, 1.24, True)                  # Friday (day 4) spot, on Monday


def _business_env(day=0):
    from quantark.util.calendar import CalendarType, DayCountConvention, create_calendar
    cal = create_calendar(CalendarType.NONE, year_range=(2024, 2025))
    return F.fx_env(day, day_count_convention=DayCountConvention.BUSINESS_DAYS, bus_days_in_year=252, calendar=cal), cal


def _business_twin(dates):
    from quantark.util.calendar import DayCountConvention, calculate_year_fraction
    _, cal = _business_env()
    yf = lambda d: calculate_year_fraction(F.START, d, DayCountConvention.BUSINESS_DAYS, 252, calendar=cal)  # noqa: E731
    # the twin's maturity is the dated expiry on the same business-day clock (261/252, not 1.0)
    numeric = F.fx_ra("domestic", "numeric", times=[yf(d) for d in dates], maturity=yf(F.START + timedelta(days=365)))
    return numeric, F.fx_ra("domestic", "dated", dates=dates)


def test_fx_backtest_x2_on_a_business_day_clock():
    numeric, dated = _business_twin(F.fixing_dates())
    env, _ = _business_env()
    eng_d, values_d, *_ = _fx_backtest([dated], index=pd.bdate_range(F.START, periods=18), env=env)
    eng_n, values_n, *_ = _fx_backtest([numeric], index=pd.bdate_range(F.START, periods=18), env=env)
    assert _rows(eng_d.run_fixings) == _rows(eng_n.run_fixings) and _rows(eng_d.run_fixings)
    assert all(is_close(a, b, rel_tol=1e-12) for a, b in zip(values_d, values_n))


def test_business_day_clock_on_a_calendar_path_records_at_the_last_live_instant():
    """Review Focus 4 / Ruling 4: Friday's fixing is t = 0 through the weekend on a business-day clock."""
    numeric, dated = _business_twin(F.fixing_dates())
    env, _ = _business_env()
    spot_of = lambda k: 1.25 + 0.001 * k                                        # noqa: E731
    eng_d, *_ = _fx_backtest([dated], env=env, spot_of=spot_of, index=pd.date_range(F.START, periods=10))
    eng_n, *_ = _fx_backtest([numeric], env=env, spot_of=spot_of, index=pd.date_range(F.START, periods=10))
    assert _rows(eng_d.run_fixings)[0] == _rows(eng_n.run_fixings)[0] == (0, True, spot_of(4), False)   # Friday's


def test_fx_dynamic_x3d_undated_path_records_nothing():
    eng, values, _, spots = _fx_dynamic([F.fx_ra("domestic", "dated")], start=None)
    assert eng.run_fixings.recorded == ()
    assert values[10] == F.quiet_price(F.FX_CLASSES["domestic"][1](), F.fx_ra("domestic", "dated"),
                                       F.fx_env(0, spot=spots[10]))


def test_fx_backtest_ruling2_a_saturday_record_on_a_business_day_clock_is_live_on_friday():
    """Ruling 2: on BUSINESS_DAYS a Saturday record is t = 0 on Friday (no weekend rows on the path): recorded at
    Friday's spot with gap=False, dated and numeric twins identical."""
    sat = F.START + timedelta(days=5)                                           # 2024-01-06
    numeric, dated = _business_twin([sat] + F.fixing_dates()[1:])
    env, _ = _business_env()
    index = pd.bdate_range(F.START, periods=10)
    spot_of = lambda k: 1.20 + 0.01 * k                                         # noqa: E731
    (eng_d, values_d, *_), warns = F.count_missing(_fx_backtest, [dated], index=index, env=env, spot_of=spot_of)
    eng_n, values_n, *_ = _fx_backtest([numeric], index=index, env=env, spot_of=spot_of)
    assert warns == 0
    assert _rows(eng_d.run_fixings)[0] == _rows(eng_n.run_fixings)[0] == (0, True, spot_of(4), False)
    assert _rows(eng_d.run_fixings) == _rows(eng_n.run_fixings)
    assert all(is_close(a, b, rel_tol=1e-12) for a, b in zip(values_d, values_n))


def test_fx_backtest_x3e_a_15h_path_refuses_a_participant():
    index = pd.date_range(F.START + timedelta(hours=15), periods=3, freq="D")
    with pytest.raises(ValidationError, match="time of day"):
        _fx_backtest([F.fx_ra("domestic", "dated")], index=index)


def test_fx_dynamic_x3e_time_stamped_dates_refuse_a_participant():
    with pytest.raises(ValidationError, match="time of day"):
        _fx_dynamic([F.fx_ra("domestic", "dated")], start=F.START + timedelta(hours=15))


def test_fx_dynamic_x3f_a_path_before_the_env_date_refuses_a_participant():
    with pytest.raises(ValidationError, match="cannot go back"):
        _fx_dynamic([F.fx_ra("domestic", "dated")], start=F.START - timedelta(days=1))


# ================================================================ replay (the Asian: no range-accrual route, drift 1)
def _replay(product, n=F.DAYS, first=0, spot_of=F.spot_on, explain=None):
    from replay_golden import fixtures

    from quantark.backtest.replay.config import AutocallableEngineConfig
    from quantark.backtest.replay.engine import ReplayBacktestEngine
    from quantark.backtest.replay.market import AutocallableMarketDataSet
    from quantark.util.enum.engine_enums import EngineType

    dates = pd.date_range(F.START + timedelta(days=first), periods=n, freq="D")
    spots = [spot_of(first + k) for k in range(n)]
    fut = [{"date": d, "contract": "IF2403", "futures_price": s * 1.004, "expiry_date": pd.Timestamp("2024-03-15"),
            "multiplier": 300.0} for d, s in zip(dates, spots)]
    cfg = fixtures.make_book_config()
    cfg.market_data = AutocallableMarketDataSet.from_dataframes(
        spot_data=pd.DataFrame({"date": dates, "spot": spots}),
        vol_data=pd.DataFrame({"date": dates, "volatility": [F.VOL] * n}),
        rate_data=pd.DataFrame({"date": dates, "rate": [F.RATE] * n}), futures_data=pd.DataFrame(fut))
    cfg.products = [fixtures.BookProduct(product=product, quantity=1.0, position_id=1, has_lifecycle=False)]
    cfg.calculate_event_probabilities = False
    cfg.engine_config = AutocallableEngineConfig(pricing_engine_type=EngineType.ANALYTICAL)
    if explain is not None:
        cfg.pnl_explain = explain
    engine = ReplayBacktestEngine(cfg)
    res = engine.run()
    return engine, res, res.states_df()["product_mtm"].tolist()


def test_replay_x1_x2_x3g_x6():
    booked = F.asian_dated()
    before = copy.deepcopy(booked)
    (eng_d, _, mtm_d), warns = F.count_missing(_replay, booked)
    eng_n, _, mtm_n = _replay(F.asian_numeric())
    for k in (0, 3, 4, 5, 10, 14, 18, 20, 24):
        _, _, window = _replay(F.by_hand("asian", k - 1), n=2, first=k)          # X1a: a window starting at k
        assert is_close(mtm_d[k], window[0], rel_tol=1e-12), k
        assert is_close(mtm_n[k], mtm_d[k], rel_tol=1e-12), k                    # X2
    assert warns == 0                                                            # X1c
    fills_d, fills_n = eng_d._replays[0]._run_fills, eng_n._replays[0]._run_fills
    assert fills_d == fills_n == F.hand_outcomes("asian", F.DAYS - 2)            # X1b / X2 / X6
    assert booked.observation_records == before.observation_records             # X3g


def test_replay_x1b_the_fixing_does_not_move_later():
    other = lambda k: F.spot_on(k) if k <= 5 else 80.0                          # noqa: E731
    a, _, _ = _replay(F.asian_dated(), n=10)
    b, _, _ = _replay(F.asian_dated(), n=10, spot_of=other)
    assert a._replays[0]._run_fills[0] == b._replays[0]._run_fills[0] == 105.0


# ================================================================ X3i: the run explain recorders
def _refill_and_chain(explain_df, recon_df, values):
    refill = explain_df[explain_df["factor"] == "fixing_refill"]
    assert len(refill) > 0 and (refill["pnl"] == 0.0).all()
    assert recon_df["ok"].all()                                                  # the rows sum to each day's PnL
    pos = recon_df[(recon_df["level"] == "position") & (recon_df["method"] == "waterfall")]
    assert is_close(pos["expected"].sum(), values[-1] - values[0], rel_tol=0.0, abs_tol=1e-9)   # pv_t1(N) == pv_t0(N+1)


@pytest.mark.parametrize("form", ["dated", "numeric"])
@pytest.mark.parametrize("kind", ["ra", "asian"])
def test_x3i_equity_backtest_explain(kind, form, tmp_path):
    """Plan drift 7 / spec E5: a fixing passing mid-run no longer reads as a contract replacement."""
    from quantark.backtest import BacktestConfig, BacktestEngine, ZeroCostModel
    from quantark.backtest.strategy import DeltaNeutralStrategy
    from quantark.pnlexplain import ExplainMethod, PnLExplainConfig
    from quantark.portfolio import Position

    adapter = F.DailyAdapter(12)
    cfg = BacktestConfig(
        strategy=DeltaNeutralStrategy(delta_threshold=1e12), start_date=F.START,
        end_date=adapter.dates[-1].to_pydatetime(), underlying=F.UNDERLYING,
        initial_positions=[Position(product=F.MAKERS[(kind, form)](), quantity=1.0, entry_price=0.0,
                                    underlying=F.UNDERLYING, engine=F.ENGINES[kind](), entry_timestamp=F.START)],
        market_data_adapter=adapter, transaction_cost_model=ZeroCostModel(), handle_lifecycle_events=False,
        results_path=str(tmp_path), pnl_explain=PnLExplainConfig(methods=(ExplainMethod.WATERFALL,)),
    )
    res = BacktestEngine(cfg).run()
    _refill_and_chain(res.explain_df, res.explain_reconciliation_df, res.get_value_series().tolist())


def test_x3i_replay_explain():
    from quantark.pnlexplain import ExplainMethod, PnLExplainConfig

    _, res, mtm = _replay(F.asian_dated(), n=12, explain=PnLExplainConfig(methods=(ExplainMethod.WATERFALL,)))
    _refill_and_chain(res.explain_df(), res.explain_reconciliation_df(), mtm)   # book results: methods


# ================================================================ W6 x W8: a run reaching the exercise date
# A dated contract valued ON its exercise date prices its settled value (W8, user ruling 2026-10-09 option (a)), so
# a numeric participant aged to its maturity is no longer floored at 1e-8: it is exactly 0.0 and equals its twin.
_SHORT_SETTLED = {"ra": 100.0 * 0.10, "asian": (F.spot_on(4) + F.spot_on(8)) / 2.0 - 100.0}   # fixings 105, 109


def _backtest_explained(product, kind, n, tmp_path):
    from quantark.pnlexplain import ExplainMethod, PnLExplainConfig

    from quantark.backtest import BacktestConfig, BacktestEngine, ZeroCostModel
    from quantark.backtest.strategy import DeltaNeutralStrategy
    from quantark.portfolio import Position

    adapter = F.DailyAdapter(n)
    cfg = BacktestConfig(
        strategy=DeltaNeutralStrategy(delta_threshold=1e12), start_date=F.START,
        end_date=adapter.dates[-1].to_pydatetime(), underlying=F.UNDERLYING,
        initial_positions=[Position(product=product, quantity=1.0, entry_price=0.0, underlying=F.UNDERLYING,
                                    engine=F.ENGINES[kind](), entry_timestamp=F.START)],
        market_data_adapter=adapter, transaction_cost_model=ZeroCostModel(), handle_lifecycle_events=False,
        results_path=str(tmp_path), pnl_explain=PnLExplainConfig(methods=(ExplainMethod.WATERFALL,)),
    )
    return BacktestEngine(cfg).run()


@pytest.mark.parametrize("kind", ["ra", "asian"])
def test_equity_backtest_twins_settle_on_the_exercise_date_and_refuse_the_day_after(kind, tmp_path):
    """Days 0..10 (T = day 10): both twins value their settled value on T, equal to each other. A run to T + 1 is
    refused for both (the dated contract has expired; the numeric one is past its maturity)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, res_d = _backtest(_short(kind, "dated"), kind, F.DailyAdapter(11), tmp_path)
        _, res_n = _backtest(_short(kind, "numeric"), kind, F.DailyAdapter(11), tmp_path)
        values_d, values_n = res_d.get_value_series().tolist(), res_n.get_value_series().tolist()
        assert is_close(values_d[-1], _SHORT_SETTLED[kind], rel_tol=1e-12)
        assert is_close(values_n[-1], values_d[-1], rel_tol=1e-12)
        for form in ("dated", "numeric"):
            with pytest.raises(ValidationError):
                _backtest(_short(kind, form), kind, F.DailyAdapter(12), tmp_path)


@pytest.mark.parametrize("kind", ["ra", "asian"])
def test_equity_backtest_explain_a_numeric_participant_reaching_its_maturity(kind, tmp_path):
    """The explain step ending ON the maturity: the numeric participant aged to exactly 0.0 is a time step (it was
    refused against the trackers' 1e-8 floor) and explains as its dated twin does."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_d = _backtest_explained(_short(kind, "dated"), kind, 11, tmp_path)
        res_n = _backtest_explained(_short(kind, "numeric"), kind, 11, tmp_path)
    for res in (res_d, res_n):
        assert res.explain_reconciliation_df["ok"].all()
    key = ["date", "factor"]
    rows_d = res_d.explain_df.sort_values(key).reset_index(drop=True)
    rows_n = res_n.explain_df.sort_values(key).reset_index(drop=True)
    assert rows_d[key].equals(rows_n[key])
    for a, b in zip(rows_d["pnl"], rows_n["pnl"]):
        assert is_close(a, b, rel_tol=1e-9, abs_tol=1e-12), (a, b)


def test_replay_numeric_participant_on_its_maturity_is_zero_and_equals_the_dated_twin():
    """The replay's pricing copy on T: the numeric Asian aged to exactly 0.0 (no longer floored at 1e-8) prices the
    dated twin's settled value. (The replay's own maturity settlement of an Asian is out of scope -- run-fixings spec
    §10, cash for range accruals or Asians at expiry inside runs -- so the path stops the day before T.)"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        copies = {}
        for form in ("dated", "numeric"):
            eng, _, _ = _replay(_short("asian", form), n=10)                      # days 0..9, fixings 4 and 8 recorded
            copies[form] = eng._replays[0].product_for_date(pd.Timestamp(F.START + timedelta(days=10)), F.eq_env(10))
        assert copies["numeric"].maturity == 0.0
        engine = F.ENGINES["asian"]()
        dated = F.quiet_price(engine, copies["dated"], F.eq_env(10))
        assert is_close(dated, _SHORT_SETTLED["asian"], rel_tol=1e-12)
        assert is_close(F.quiet_price(engine, copies["numeric"], F.eq_env(10)), dated, rel_tol=1e-12)
