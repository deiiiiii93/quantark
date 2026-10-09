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
