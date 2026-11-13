"""The legacy event-statistics types are gone (unified event statistics T21, spec AC24).

* No tracked ``*.py`` under ``quantark/``, ``test/`` or ``example/`` names a legacy statistics type, its
  ambiguous KI fields or the legacy converters (the pattern is assembled from fragments so this file does
  not match itself).
* The transition module, the legacy ledger helper and the dead PDE legacy builders no longer exist.
* A consumer handed something other than ``EventStatistics`` refuses it instead of converting it.
"""
from __future__ import annotations

import importlib
import re
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from quantark.asset.equity.engine import event_stats
from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.cashleg.event_distribution import EventDistribution
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.exceptions import ValidationError

ROOT = Path(__file__).resolve().parents[1]

_STATS = "Event" + "Stats"
_TYPES = ("Autocallable", "Phoenix", "KOReset", "DCN", "RiskNeutralSnowball")
_NAMES = tuple(t + _STATS for t in _TYPES) + (
    "ki_ever" + "_probability",
    "ki_survive_knocked_in" + "_probability",
    "_as_event" + "_statistics",
    "from_autocallable" + "_stats",
    "from_dcn" + "_stats",
)
LEGACY = re.compile("|".join(re.escape(n) for n in _NAMES))


def _tracked_python() -> list:
    out = subprocess.run(
        ["git", "ls-files", "quantark/*.py", "test/*.py", "example/*.py"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout
    return [p for p in out.splitlines() if p]


def test_no_tracked_python_names_a_legacy_statistics_type_field_or_converter():
    paths = _tracked_python()
    assert any(p.startswith("quantark/") for p in paths)
    hits = {}
    for rel in paths:
        path = ROOT / rel
        if not path.exists():
            continue
        found = sorted({m.group(0) for m in LEGACY.finditer(path.read_text(encoding="utf-8", errors="replace"))})
        if found:
            hits[rel] = found
    assert hits == {}


def test_the_transition_module_is_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("quantark.asset.equity.engine._event_stats_transition")


def test_the_legacy_ledger_helper_and_converters_are_gone():
    assert not hasattr(event_stats, "payment_aware_cashflow_fields")
    assert "payment_aware_cashflow_fields" not in event_stats.__all__
    for name in _NAMES[:5]:
        assert not hasattr(event_stats, name)
    assert not hasattr(EventDistribution, "from_autocallable" + "_stats")
    assert not hasattr(EventDistribution, "from_dcn" + "_stats")


@pytest.mark.parametrize("module, cls", [
    ("quantark.asset.equity.engine.pde.snowball_pde_solver", "SnowballPDESolver"),
    ("quantark.asset.equity.engine.pde.phoenix_pde_solver", "PhoenixPDESolver"),
    ("quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver", "KOResetSnowballPDESolver"),
])
def test_the_dead_pde_legacy_builders_are_gone(module, cls):
    assert not hasattr(getattr(importlib.import_module(module), cls), "_make_event_stats")


class _NotStatistics(BaseEngine):
    """A producer whose ``calculate_event_stats`` returns something that is not ``EventStatistics``."""

    def price(self, product, pricing_env, **kwargs):
        return 1.0

    def calculate_event_stats(self, product, pricing_env, **kwargs):
        return {"pv": 1.0}


def _env() -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0), vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.03), div_yield=ContinuousDividendYield(div_yield=0.0),
        valuation_date=datetime(2024, 1, 1),
    )


def test_price_with_events_refuses_a_producer_that_does_not_return_event_statistics():
    product = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    with pytest.raises(ValidationError, match="EventStatistics"):
        _NotStatistics().price_with_events(product, _env())


def test_checked_event_statistics_passes_the_new_type_and_refuses_anything_else():
    with pytest.raises(ValidationError, match="EventStatistics"):
        event_stats.checked_event_statistics(object())
    s = _minimal_statistics()
    assert event_stats.checked_event_statistics(s) is s


def _minimal_statistics() -> event_stats.EventStatistics:
    import numpy as np

    from quantark.asset.equity.engine.event_stats import (
        KnockInStatistics, KnockInTable, ObservationTable, OutcomeRow, ReceivablesTable, StatisticsProvenance,
        build_cash_ledger,
    )

    times = np.array([0.5, 1.0])
    p = np.array([0.3, 0.2])
    n = times.size
    observations = ObservationTable(
        times=times, payment_times=times + 0.01, is_ko_obs=np.ones(n, dtype=bool),
        is_coupon_obs=np.zeros(n, dtype=bool), probability=p, survival=1.0 - np.cumsum(p),
        discounted_cash=p * 99.0, cash_to_pay=p * 100.0, payment_is_path_dependent=np.zeros(n, dtype=bool),
    )
    maturity = OutcomeRow(time=1.0, payment_time=1.02, probability=0.4, discounted_cash=38.8, cash_to_pay=40.0)
    ki_settlement = OutcomeRow(time=1.0, payment_time=1.02, probability=0.1, discounted_cash=9.7, cash_to_pay=10.0)
    ki_table = KnockInTable(times=times, payment_times=times, probability=np.array([0.05, 0.05]),
                            survival=np.array([0.95, 0.9]), interpolated=np.zeros(n, dtype=bool))
    receivables = ReceivablesTable.empty()
    return event_stats.EventStatistics(
        product_family="snowball", pv=100.0, observations=observations, coupons=None, maturity=maturity,
        ki_settlement=ki_settlement,
        knock_in=KnockInStatistics(probability=0.1, probability_no_ko=0.1, expected_time=0.7, expected_loss=1.0),
        ki_table=ki_table, receivables=receivables,
        ledger=build_cash_ledger(observations=observations, coupons=None, maturity=maturity,
                                 ki_settlement=ki_settlement, receivables=receivables),
        expected_lifetime=0.6,
        provenance=StatisticsProvenance(statistics_model="hand_built", price_model="hand_built"),
        unavailable={},
    )
