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


@pytest.mark.parametrize("run", [_fx_backtest, _fx_dynamic], ids=["backtest", "dynamic"])
@pytest.mark.parametrize("kind", FX_KINDS)
def test_fx_runs_x1_x2_x3g_x6(run, kind):
    q = kind == "quanto"
    booked = F.fx_ra(kind, "dated")
    before = copy.deepcopy(booked)
    (eng_d, values_d, pf, spots), warns = F.count_missing(run, [booked], env=F.fx_env(0, quanto=q))
    eng_n, values_n, _, spots_n = run([F.fx_ra(kind, "numeric")], env=F.fx_env(0, quanto=q))
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
    assert pf.positions[list(pf.positions)[0]].product is booked
    assert booked.observation_records == before.observation_records             # X3g


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
    eng, values, *_ = _fx_dynamic([F.fx_ra("domestic", "dated")], start=None)
    assert eng.run_fixings.recorded == ()
    assert values[10] == F.quiet_price(F.FX_CLASSES["domestic"][1](), F.fx_ra("domestic", "dated"),
                                       F.fx_env(0, spot=F.fx_spot(10)))
