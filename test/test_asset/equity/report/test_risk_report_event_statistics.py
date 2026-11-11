"""The Snowball risk report reads the unified ``EventStatistics`` (unified event statistics T8).

* The engine's statistics are read through the transition module and ``EventStatistics.require``: a field
  the producer flags in ``unavailable`` is shown as "unavailable (<reason>)" (NaN / null in the dashboard
  payload), never as the flagged number;
* "P(KI before any KO)" is ``knock_in.probability`` and "P(matured knocked in)" is ``ki_settlement.probability``;
* an engine without statistics falls back to the path analyzer, which reports the same type.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.engine.event_stats import (
    EventStatistics,
    KnockInStatistics,
    KnockInTable,
    ObservationTable,
    OutcomeRow,
    ReceivablesTable,
    StatisticsProvenance,
    build_cash_ledger,
)
from quantark.asset.equity.product.option.snowball_helpers import create_standard_snowball
from quantark.asset.equity.report import autocallable_risk_report as report
from quantark.asset.equity.report.surfaces import GridSpec
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment

REASON = "transition: legacy producer"
KO_CASH = (5.125, 9.875)


def _stats(*, unavailable=(), pv=1.0) -> EventStatistics:
    probability = np.array([0.1, 0.2])
    observations = ObservationTable(
        times=[0.5, 1.0], payment_times=[0.5, 1.0], is_ko_obs=[True, True], is_coupon_obs=[False, False],
        probability=probability, survival=1.0 - np.cumsum(probability), discounted_cash=KO_CASH,
        cash_to_pay=KO_CASH, payment_is_path_dependent=[False, False],
    )
    maturity = OutcomeRow(time=1.0, payment_time=1.0, probability=0.66, discounted_cash=0.0, cash_to_pay=0.0)
    ki_settlement = OutcomeRow(time=1.0, payment_time=1.0, probability=0.04, discounted_cash=-3.5,
                               cash_to_pay=-3.6)
    receivables = ReceivablesTable.empty()
    return EventStatistics(
        product_family="snowball", pv=pv, observations=observations, coupons=None, maturity=maturity,
        ki_settlement=ki_settlement,
        knock_in=KnockInStatistics(probability=0.05, probability_no_ko=0.04, expected_time=0.5, expected_loss=3.5),
        ki_table=KnockInTable(times=[0.5], payment_times=[0.5], probability=[0.05], survival=[0.85],
                              interpolated=[False]),
        receivables=receivables,
        ledger=build_cash_ledger(observations=observations, coupons=None, maturity=maturity,
                                 ki_settlement=ki_settlement, receivables=receivables),
        expected_lifetime=0.9, provenance=StatisticsProvenance("hand_built", "hand_built"),
        unavailable={path: REASON for path in unavailable},
    )


FLAGGED = ("observations.discounted_cash", "knock_in.probability")


class _StatsEngine(BaseEngine):
    def __init__(self, stats) -> None:
        super().__init__()
        self._stats = stats

    def price(self, product, pricing_env) -> float:
        return float(pricing_env.spot) * 0.01

    def calculate_event_stats(self, product, pricing_env):
        return self._stats


class _NoStatsEngine(BaseEngine):
    def price(self, product, pricing_env) -> float:
        return float(pricing_env.spot) * 0.01


def _env() -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0), vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.02), div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=datetime(2024, 1, 1),
    )


def _product():
    return create_standard_snowball(initial_price=100.0, strike=100.0, maturity=1.0, ko_barrier=103.0,
                                    ko_rate=0.15, ki_barrier=75.0, num_observations=4, include_principal=False)


GRID = GridSpec(spot_nodes=5, q_nodes=5, vol_nodes=5, time_bump_years=1.0 / 252.0)


def _report_text(tmp_path, monkeypatch, engine) -> str:
    monkeypatch.setattr(report, "_select_snowball_pricing_engine", lambda **kwargs: (engine, "stub"))
    result = report.generate_snowball_risk_report(product=_product(), pricing_env=_env(), output_dir=tmp_path,
                                                  grid_spec=GRID, engine_preference=["stub"])
    return result.report_path.read_text()


def test_snapshot_conditional_cashflows_show_flagged_ko_cash_as_null():
    snapshot = report.build_snowball_risk_snapshot(product=_product(), pricing_env=_env(),
                                                   engine=_StatsEngine(_stats(unavailable=FLAGGED)),
                                                   label="t", grid_spec=GridSpec(spot_nodes=5, q_nodes=5, vol_nodes=5))
    rows = snapshot["conditional_cashflows"]
    assert [row["p_ko"] for row in rows] == [0.1, 0.2]
    assert all(row["ed_ko_cf"] is None and row["conditional_ed_ko_cf"] is None for row in rows)


def test_snapshot_conditional_cashflows_read_available_ko_cash():
    snapshot = report.build_snowball_risk_snapshot(product=_product(), pricing_env=_env(),
                                                   engine=_StatsEngine(_stats()), label="t",
                                                   grid_spec=GridSpec(spot_nodes=5, q_nodes=5, vol_nodes=5))
    rows = snapshot["conditional_cashflows"]
    assert [row["ed_ko_cf"] for row in rows] == list(KO_CASH)
    assert rows[0]["conditional_ed_ko_cf"] == pytest.approx(KO_CASH[0] / 0.1)


def test_report_shows_flagged_statistics_as_unavailable(tmp_path, monkeypatch):
    content = _report_text(tmp_path, monkeypatch, _StatsEngine(_stats(unavailable=FLAGGED)))
    assert f"P(KI before any KO): unavailable ({REASON})" in content
    assert "P(matured knocked in): 0.040000" in content
    assert f"PV reconciliation error (PV - sum(expected discounted cashflows)): unavailable ({REASON})" in content
    assert f"observations.discounted_cash ({REASON})" in content
    for cash in KO_CASH:
        assert f"{cash}" not in content


def test_report_shows_available_statistics(tmp_path, monkeypatch):
    stats = _stats()
    content = _report_text(tmp_path, monkeypatch, _StatsEngine(stats))
    assert "P(KI before any KO): 0.050000" in content
    assert "P(matured knocked in): 0.040000" in content
    assert f"PV reconciliation error (PV - sum(expected discounted cashflows)): {stats.reconciliation_error:.6g}" \
        in content
    assert "Unavailable statistics" not in content


def test_an_engine_without_statistics_falls_back_to_the_path_analyzer(tmp_path, monkeypatch):
    stats = _stats()
    monkeypatch.setattr(report.AutocallablePathAnalyzer, "analyze_snowball_risk_neutral",
                        lambda self, product, pricing_env: stats)
    content = _report_text(tmp_path, monkeypatch, _NoStatsEngine())
    assert "statistics: hand_built" in content
    assert "P(KI before any KO): 0.050000" in content
