"""Replay event rows read the unified ``EventStatistics`` (unified event statistics T8, spec D21).

* ``ki_probability_to_maturity`` is ``ki_settlement.probability`` (matured knocked in);
  ``expected_discounted_maturity_cashflow`` is ``maturity.discounted_cash + ki_settlement.discounted_cash``;
* KO rows come from the observation table, KI rows from the per-date KI table;
* every field is read through ``EventStatistics.require``: a statistic the producer flags in ``unavailable``
  is recorded as NaN (or, for the KI table, no KI rows), never replaced by a stand-in — in particular no
  KI row is invented at maturity from a scalar probability;
* the T8 consumer modules name no legacy statistics type or converter.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from golden.replay_golden import fixtures  # noqa: E402

from quantark.asset.equity.engine.base_engine import BaseEngine  # noqa: E402
from quantark.asset.equity.engine.event_stats import (  # noqa: E402
    EventStatistics,
    KnockInStatistics,
    KnockInTable,
    ObservationTable,
    OutcomeRow,
    ReceivablesTable,
    StatisticsProvenance,
    build_cash_ledger,
)
from quantark.asset.equity.param import EngineParams  # noqa: E402
from quantark.backtest.replay.single import AutocallableBacktestEngine  # noqa: E402

REASON = "not computed on this route"
DATE = pd.Timestamp("2024-01-02")


KO_TIMES = (2.0 / 365.0, 5.0 / 365.0)
KO_PAY = (3.0 / 365.0, 6.0 / 365.0)
KO_P = (0.05, 0.15)
KO_CASH = (510.0, 1490.0)
KI_TIMES = (1.0 / 365.0, 3.0 / 365.0)
KI_P = (0.02, 0.03)


def _stats(*, unavailable=(), ki_table=True, ko=None) -> EventStatistics:
    """``ko``: ``(times, payment times, dates, probabilities, cash)`` of the knock-out rows (default: two rows)."""
    ko_times, ko_pay, ko_dates, ko_p, ko_cash = ko or (KO_TIMES, KO_PAY, None, KO_P, KO_CASH)
    n = len(ko_times)
    ko_p = np.asarray(ko_p)
    observations = ObservationTable(
        times=ko_times, payment_times=ko_pay, dates=ko_dates, payment_dates=ko_dates, is_ko_obs=[True] * n,
        is_coupon_obs=[False] * n, probability=ko_p, survival=1.0 - np.cumsum(ko_p), discounted_cash=ko_cash,
        cash_to_pay=np.asarray(ko_cash) * 1.001, payment_is_path_dependent=[False] * n,
    )
    maturity = OutcomeRow(time=6.0 / 365.0, payment_time=6.0 / 365.0, probability=0.76, discounted_cash=7600.0,
                          cash_to_pay=7601.0)
    ki_settlement = OutcomeRow(time=6.0 / 365.0, payment_time=6.0 / 365.0, probability=0.04,
                               discounted_cash=350.0, cash_to_pay=350.1)
    ki_p = np.asarray(KI_P)
    table = KnockInTable(times=KI_TIMES, payment_times=KI_TIMES, probability=ki_p,
                         survival=np.array([0.98, 0.9]), interpolated=[False, False])
    flags = {path: REASON for path in unavailable}
    if not ki_table:
        flags["ki_table"] = REASON
    receivables = ReceivablesTable.empty()
    return EventStatistics(
        product_family="snowball", pv=9950.0, observations=observations, coupons=None, maturity=maturity,
        ki_settlement=ki_settlement,
        knock_in=KnockInStatistics(probability=float(ki_p.sum()), probability_no_ko=ki_settlement.probability,
                                   expected_time=0.006, expected_loss=40.0),
        ki_table=table if ki_table else None, receivables=receivables,
        ledger=build_cash_ledger(observations=observations, coupons=None, maturity=maturity,
                                 ki_settlement=ki_settlement, receivables=receivables),
        expected_lifetime=0.015, provenance=StatisticsProvenance("hand_built", "hand_built"), unavailable=flags,
    )


class _Producer(BaseEngine):
    def __init__(self, stats) -> None:
        super().__init__(EngineParams())
        self._stats = stats

    def price(self, product, pricing_env) -> float:
        return float(self._stats.pv)

    def calculate_event_stats(self, product, pricing_env):
        return self._stats


def _record(stats):
    config = fixtures.make_scalar_bsm_config()
    config.engine_config.event_stats_fallback = "none"
    replay = AutocallableBacktestEngine(config)._inner._replays[0]
    replay.event_stats_engine = _Producer(stats)
    replay.daily_event_sink.clear()
    replay.event_prob_sink.clear()
    replay.record_event_probabilities(DATE, replay.product, None)
    assert len(replay.daily_event_sink) == 1
    return replay.daily_event_sink[0], list(replay.event_prob_sink)


def test_daily_row_maps_the_statistics_per_d21():
    stats = _stats()
    daily, _rows = _record(stats)
    assert daily["ki_probability_to_maturity"] == stats.ki_settlement.probability
    assert daily["expected_discounted_maturity_cashflow"] == (
        stats.maturity.discounted_cash + stats.ki_settlement.discounted_cash)
    assert daily["expected_discounted_ko_cashflow"] == float(np.sum(stats.observations.discounted_cash))
    assert daily["next_ko_probability"] == KO_P[0]
    assert daily["total_remaining_ko_probability"] == float(np.sum(stats.observations.probability))
    assert daily["survival_probability"] == stats.observations.survival[-1]
    assert daily["next_ko_date"] == DATE + pd.Timedelta(days=2)
    assert daily["pv"] == stats.pv


#: a knock-out pair on ONE instant (two contract records, e.g. 07-03 and the 07-04 holiday on a business-day clock:
#: one env time, their own dates and rows), then a third record
PAIR_TIMES = (2.0 / 365.0, 2.0 / 365.0, 5.0 / 365.0)
PAIR_DATES = tuple(pd.Timestamp(d).to_pydatetime() for d in ("2024-01-04", "2024-01-05", "2024-01-07"))
PAIR_P = (0.05, 0.02, 0.15)
PAIR = (PAIR_TIMES, PAIR_TIMES, PAIR_DATES, PAIR_P, (510.0, 180.0, 1490.0))


def test_next_ko_probability_sums_every_record_of_the_first_instant():
    """0.16.0 DCN/dated (T5b ruling): the next knock-out probability is P(knock out at the first knock-out instant) --
    the sum over every record on that instant (each keeps its own row); before, the first row alone (r1's share).
    The KO event rows stay one per record."""
    stats = _stats(ko=PAIR)
    daily, rows = _record(stats)
    assert daily["next_ko_probability"] == PAIR_P[0] + PAIR_P[1]
    assert daily["total_remaining_ko_probability"] == float(np.sum(PAIR_P))
    assert daily["next_ko_date"] == DATE + pd.Timedelta(days=2)
    assert [r["event_probability"] for r in rows if r["event_type"] == "KO"] == list(PAIR_P)


def test_next_ko_probability_sums_a_real_engines_same_instant_pair():
    """On a real engine's statistics (the dated AC2 pair alone: 07-03 at 103, the 07-04 holiday at 100, one env
    instant, 1D PDE): the next knock-out probability is the two records' rows together."""
    from test_asset.equity.engine_pde.test_same_instant_ko import env, pair, snowball
    from quantark.asset.equity.engine.pde import SnowballPDESolver
    from quantark.asset.equity.param import PDEParams

    stats = SnowballPDESolver(PDEParams()).calculate_event_stats(snowball(pair(), ki=None), env())
    p = list(stats.observations.probability)
    assert stats.observations.times[0] == stats.observations.times[1] and len(p) == 2 and p[1] > 0.01
    daily, _rows = _record(stats)
    assert daily["next_ko_probability"] == p[0] + p[1]


def test_ko_and_ki_rows_come_from_their_tables():
    stats = _stats()
    _daily, rows = _record(stats)
    ko = [r for r in rows if r["event_type"] == "KO"]
    ki = [r for r in rows if r["event_type"] == "KI"]
    assert [r["event_date"] for r in ko] == [DATE + pd.Timedelta(days=2), DATE + pd.Timedelta(days=5)]
    assert [r["event_probability"] for r in ko] == list(KO_P)
    assert [r["expected_discounted_cashflow"] for r in ko] == list(KO_CASH)
    assert ko[1]["conditional_probability"] == KO_P[1] / stats.observations.survival[0]
    assert [r["event_date"] for r in ki] == [DATE + pd.Timedelta(days=1), DATE + pd.Timedelta(days=3)]
    assert [r["event_probability"] for r in ki] == list(KI_P)
    assert [r["survival_probability"] for r in ki] == [0.98, 0.9]
    assert all(math.isnan(r["expected_discounted_cashflow"]) for r in ki)


def test_flagged_cash_is_recorded_as_nan_not_as_the_flagged_number():
    stats = _stats(unavailable=("observations.discounted_cash", "maturity.discounted_cash"))
    daily, rows = _record(stats)
    assert math.isnan(daily["expected_discounted_ko_cashflow"])
    assert math.isnan(daily["expected_discounted_maturity_cashflow"])
    assert daily["ki_probability_to_maturity"] == stats.ki_settlement.probability     # not flagged: kept
    assert all(math.isnan(r["expected_discounted_cashflow"]) for r in rows if r["event_type"] == "KO")
    assert [r["event_probability"] for r in rows if r["event_type"] == "KO"] == list(KO_P)


def test_an_unavailable_ki_table_records_no_ki_rows_and_invents_none_at_maturity():
    stats = _stats(ki_table=False)
    assert stats.ki_settlement.probability > 0.0
    daily, rows = _record(stats)
    assert [r for r in rows if r["event_type"] == "KI"] == []
    assert daily["ki_probability_to_maturity"] == stats.ki_settlement.probability


def test_a_flagged_ki_settlement_probability_is_nan():
    daily, _rows = _record(_stats(unavailable=("ki_settlement.probability",)))
    assert math.isnan(daily["ki_probability_to_maturity"])


@pytest.mark.parametrize("run", ["scalar_bsm", "localvol", "book"])
def test_banked_replay_frames_carry_every_statistic(run):
    """Final state (T8 note, T22): the replay goldens -- reproduced live by ``test/golden/replay`` -- record every
    statistic of their migrated engines: no NaN in the daily summary, none on a KO row; a KI row leaves blank only the
    two columns a KI row never has (conditional probability, cash)."""
    data = Path(__file__).parents[1] / "golden" / "replay_golden" / "data"
    summary = pd.read_csv(data / f"{run}_daily_event_summary.csv")
    assert len(summary) > 0 and not summary.isna().any().any(), summary[summary.isna().any(axis=1)]
    events = pd.read_csv(data / f"{run}_event_probabilities.csv")
    ko = events[events["event_type"] != "KI"]
    ki = events[events["event_type"] == "KI"]
    assert len(ko) > 0 and len(ki) > 0
    assert not ko.isna().any().any(), ko[ko.isna().any(axis=1)]
    assert not ki.drop(columns=["conditional_probability", "expected_discounted_cashflow"]).isna().any().any()
    assert ki["conditional_probability"].isna().all() and ki["expected_discounted_cashflow"].isna().all()
