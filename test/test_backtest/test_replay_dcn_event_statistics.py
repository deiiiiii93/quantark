"""A DCN replay records its event rows from the MC DCN ``EventStatistics`` (unified event statistics T11, spec D21).

T8 left the DCN replay rows reading the transition's legacy DCN statistics (KO summary NaN, no KI rows). With the MC
DCN engines producing ``EventStatistics`` natively, a DCN day records:

* the KO summary and KO rows from the observation table's knock-out rows only (a ``ko_lock_months`` row is a
  coupon-only row: no KO row), event dates the schedule's exact dates (the DCN's ACT/365F axis, spec D12);
* ``ki_probability_to_maturity`` = ``ki_settlement.probability`` (matured knocked in) and
  ``expected_discounted_maturity_cashflow`` = ``maturity.discounted_cash + ki_settlement.discounted_cash`` (a never
  knocked-in DCN pays nothing at maturity, so this is the loss leg, signed as ``price()``) -- the D21 mapping;
* the KI rows from the per-date KI table (the daily closes; once knocked in, one row on the replay date);
* nothing NaN: every field is native.

The run steps on another market's calendar (an SSE observation date is skipped and decided at the previous spot) and
crosses the knock-in day.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from dcn_run_fixtures import BOOKED, KI_DAY, UNDERLYING, booked_dcn, nyse_like_days, replay_market

from quantark.asset.equity.engine.mc.dcn_mc_engine import DCNMCEngine
from quantark.asset.equity.param import MCParams
from quantark.backtest.replay import (
    AutocallableEngineConfig,
    BookAutocallableBacktestConfig,
    BookAutocallableBacktestEngine,
    BookProduct,
    HedgeSpec,
)
from quantark.util.enum.engine_enums import EngineType

END = datetime(2023, 4, 14)
SUMMARY = ("next_ko_probability", "total_remaining_ko_probability", "ki_probability_to_maturity",
           "survival_probability", "expected_discounted_ko_cashflow", "expected_discounted_maturity_cashflow", "pv")


@pytest.fixture(scope="module")
def run():
    captured = []
    original = DCNMCEngine.calculate_event_stats

    def spy(self, product, pricing_env, **kw):
        stats = original(self, product, pricing_env, **kw)
        captured.append((pd.Timestamp(pricing_env.valuation_date).normalize(), product, stats))
        return stats

    mp = pytest.MonkeyPatch()
    mp.setattr(DCNMCEngine, "calculate_event_stats", spy)
    try:
        days = nyse_like_days(BOOKED, END)
        config = BookAutocallableBacktestConfig(
            products=[BookProduct(product=booked_dcn(), quantity=1.0, position_id=1, has_lifecycle=True)],
            market_data=replay_market(days), hedge=HedgeSpec(kind="spot", multiplier=1.0),
            engine_config=AutocallableEngineConfig(
                pricing_engine_type=EngineType.MONTE_CARLO, surface_engine_type=EngineType.MONTE_CARLO,
                event_stats_engine_type=EngineType.MONTE_CARLO,
                mc_params=MCParams(num_paths=1024, seed=5, use_qmc=True)),
            underlying=UNDERLYING, start_date=days[0], end_date=days[-1], fixed_dividend_yield=0.1406,
            calculate_event_probabilities=True, terminate_on_lifecycle_end=False,
        )
        results = BookAutocallableBacktestEngine(config).run()
    finally:
        mp.undo()
    return days, captured, _frame(results.daily_event_summary_df), _frame(results.event_probability_df)


def _frame(accessor) -> pd.DataFrame:
    frame = accessor() if callable(accessor) else accessor
    return frame.reset_index() if "date" not in frame.columns else frame


def test_every_run_day_records_native_statistics(run):
    days, captured, summary, _ = run
    assert [d for d, _, _ in captured] == [pd.Timestamp(d) for d in days]
    assert len(summary) == len(days)
    assert not summary[list(SUMMARY)].isna().any().any(), summary[list(SUMMARY)]
    assert set(summary["event_stats_engine"]) == {"primary"}


def test_summary_columns_follow_the_d21_mapping(run):
    _, captured, summary, _ = run
    for date, _, stats in captured:
        row = summary[pd.to_datetime(summary["date"]) == date].iloc[0]
        ko = stats.observations.is_ko_obs
        assert row["next_ko_probability"] == stats.observations.probability[ko][0]
        assert row["total_remaining_ko_probability"] == float(np.sum(stats.observations.probability[ko]))
        assert row["survival_probability"] == stats.observations.survival[ko][-1]
        assert row["expected_discounted_ko_cashflow"] == float(np.sum(stats.observations.discounted_cash[ko]))
        assert row["ki_probability_to_maturity"] == stats.ki_settlement.probability
        assert row["expected_discounted_maturity_cashflow"] == (
            stats.maturity.discounted_cash + stats.ki_settlement.discounted_cash)
        assert stats.maturity.discounted_cash == 0.0                         # a DCN pays nothing unless knocked in
        assert row["expected_discounted_maturity_cashflow"] <= 0.0           # the buyer's loss leg
        assert row["pv"] == stats.pv


def test_ko_rows_are_the_knock_out_rows_on_their_exact_dates(run):
    _, captured, _, rows = run
    for date, product, stats in captured:
        ko_rows = rows[(pd.to_datetime(rows["date"]) == date) & (rows["event_type"] == "KO")]
        ko = stats.observations.is_ko_obs
        expected = [pd.Timestamp(d) for d, k in zip(stats.observations.dates, ko) if k]
        assert [pd.Timestamp(d) for d in ko_rows["event_date"]] == expected
        np.testing.assert_array_equal(ko_rows["event_probability"], stats.observations.probability[ko])
        np.testing.assert_array_equal(ko_rows["expected_discounted_cashflow"], stats.observations.discounted_cash[ko])
        assert expected == [pd.Timestamp(m.observation_date) for m in product.schedule.monthly if m.is_ko_obs]


def test_ki_rows_are_the_daily_table_and_one_row_once_knocked_in(run):
    _, captured, _, rows = run
    for date, product, stats in captured:
        ki_rows = rows[(pd.to_datetime(rows["date"]) == date) & (rows["event_type"] == "KI")]
        np.testing.assert_array_equal(ki_rows["event_probability"], stats.ki_table.probability)
        assert [pd.Timestamp(d) for d in ki_rows["event_date"]] == [pd.Timestamp(d) for d in stats.ki_table.dates]
        if date >= pd.Timestamp(KI_DAY):
            # the knock-in close (and every day after it): decided, one row on the replay date
            assert list(ki_rows["event_probability"]) == [1.0] and list(ki_rows["event_date"]) == [date]
            assert stats.knock_in.probability == 1.0
        else:
            assert len(ki_rows) == stats.ki_table.size > 100
            assert [pd.Timestamp(d) for d in ki_rows["event_date"]] == [
                pd.Timestamp(d) for d in product.schedule.daily_ki_dates]
