"""Facades and execution adapters on the unified ``EventStatistics`` (unified event statistics T7).

* The facade/adapter modules name no legacy statistics type and no legacy converter: they read statistics
  as ``EventStatistics`` (through the branch transition module while producers migrate).
* A producer that returns ``EventStatistics`` directly works through every facade unchanged:
  ``price_with_events`` converts it with ``EventDistribution.from_statistics`` (npv is its ``pv``), the
  EVENT_STATS adapters and the PDE session pass the object through and read only ``pv``.
* A terminated note and a flagged stream reach the caller as the converter states them, never replaced.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

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
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.cashleg import EventDistribution, EventType, FixedPayoffLeg, LegDirection, PaymentTrigger
from quantark.cashleg.leg_valuator import value_leg
from quantark.execution.contracts import OutputKind, PricingOperation, PricingRequest
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.exceptions import ValidationError

ROOT = Path(__file__).resolve().parents[2]

#: the T7 facade/adapter modules (spec D15 consumer order: they migrate before any producer)
T7_FILES = (
    "quantark/asset/equity/engine/base_engine.py",
    "quantark/asset/equity/engine/pde_engine.py",
    "quantark/asset/equity/engine/pde/pde_execution_adapters.py",
    "quantark/asset/equity/engine/quad/v2/execution_adapter.py",
    "quantark/execution/greeks.py",
    "quantark/execution/legacy_adapter.py",
    "quantark/asset/equity/engine/settlement_support.py",
)
#: the legacy statistics types, their ambiguous fields and converters (spec AC24's list less the shim itself)
LEGACY = re.compile(
    r"AutocallableEventStats|PhoenixEventStats|KOResetEventStats|DCNEventStats|RiskNeutralSnowballEventStats"
    r"|ki_ever_probability|ki_survive_knocked_in_probability|from_autocallable_stats|from_dcn_stats"
)

PROV = StatisticsProvenance(statistics_model="hand_built", price_model="hand_built")
KO_TIMES = (0.25, 0.5, 0.75, 1.0)
KO_P = (0.30, 0.10, 0.05, 0.05)
#: a KI table on its own axis, longer than the KO grid (the stream the legacy converter dropped)
KI_TIMES = (0.1, 0.2, 0.3, 0.45, 0.6, 0.8)
KI_P = (0.01, 0.02, 0.03, 0.02, 0.01, 0.01)


def _env() -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0), vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.03), div_yield=ContinuousDividendYield(div_yield=0.0),
        valuation_date=datetime(2024, 1, 1),
    )


def _product() -> EuropeanVanillaOption:
    return EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)


def _observations(times, probability) -> ObservationTable:
    times = np.asarray(times, dtype=float)
    probability = np.asarray(probability, dtype=float)
    n = times.size
    return ObservationTable(
        times=times, payment_times=times + 0.01, is_ko_obs=np.ones(n, dtype=bool),
        is_coupon_obs=np.zeros(n, dtype=bool), probability=probability, survival=1.0 - np.cumsum(probability),
        discounted_cash=probability * 99.0, cash_to_pay=probability * 100.0,
        payment_is_path_dependent=np.zeros(n, dtype=bool),
    )


def _row(time, probability) -> OutcomeRow:
    return OutcomeRow(time=time, payment_time=time + 0.02, probability=probability,
                      discounted_cash=probability * 97.0, cash_to_pay=probability * 100.0)


def _ki_table(times, probability) -> KnockInTable:
    times = np.asarray(times, dtype=float)
    probability = np.asarray(probability, dtype=float)
    return KnockInTable(times=times, payment_times=times, probability=probability,
                        survival=1.0 - np.cumsum(probability), interpolated=np.zeros(times.size, dtype=bool))


def _stats(*, observations, maturity, ki_settlement, ki_table, p_ki, receivables=None, unavailable=None,
           pv=101.25) -> EventStatistics:
    receivables = ReceivablesTable.empty() if receivables is None else receivables
    unavailable = {} if unavailable is None else dict(unavailable)
    if p_ki == 0.0:
        unavailable["knock_in.expected_time"] = "undefined: P(knock in) = 0"
    return EventStatistics(
        product_family="snowball", pv=pv, observations=observations, coupons=None, maturity=maturity,
        ki_settlement=ki_settlement,
        knock_in=KnockInStatistics(probability=p_ki, probability_no_ko=ki_settlement.probability,
                                   expected_time=0.4 if p_ki > 0.0 else None, expected_loss=1.0),
        ki_table=ki_table, receivables=receivables,
        ledger=build_cash_ledger(observations=observations, coupons=None, maturity=maturity,
                                 ki_settlement=ki_settlement, receivables=receivables),
        expected_lifetime=0.6, provenance=PROV, unavailable=unavailable,
    )


def _snowball() -> EventStatistics:
    return _stats(observations=_observations(KO_TIMES, KO_P), maturity=_row(1.0, 0.42),
                  ki_settlement=_row(1.0, 0.08), ki_table=_ki_table(KI_TIMES, KI_P), p_ki=float(sum(KI_P)))


def _terminated() -> EventStatistics:
    receivables = ReceivablesTable(source=("ko",), decision_times=[-0.02], payment_times=[0.03],
                                   discounted_cash=[104.0], cash_to_pay=[105.0])
    return _stats(observations=_observations((), ()), maturity=_row(1.0, 0.0), ki_settlement=_row(1.0, 0.0),
                  ki_table=KnockInTable.empty(), p_ki=0.0, receivables=receivables, pv=104.0)


class _Producer(BaseEngine):
    """An engine whose producer already returns ``EventStatistics`` (a migrated producer, T9-T17)."""

    def __init__(self, stats):
        super().__init__()
        self._stats = stats

    def price(self, product, pricing_env, *, lifecycle_state=None):
        return float(self._stats.pv)

    def calculate_event_stats(self, product, pricing_env, *, lifecycle_state=None, **options):
        return self._stats


# --- the facade modules read no legacy statistics -------------------------------------------------------


@pytest.mark.parametrize("path", T7_FILES)
def test_facade_module_names_no_legacy_statistics_type_or_converter(path):
    text = (ROOT / path).read_text(encoding="utf-8")
    found = sorted({m.group(0) for m in LEGACY.finditer(text)})
    assert found == [], f"{path} still names {found}"


# --- price_with_events on a producer that returns EventStatistics ---------------------------------------


def test_price_with_events_passes_event_statistics_through_from_statistics():
    stats = _snowball()
    result = _Producer(stats).price_with_events(_product(), _env())
    assert result.npv == stats.pv                                    # the statistics' pv, bitwise
    dist = result.event_distribution
    expected = EventDistribution.from_statistics(stats)
    assert set(dist.probabilities) == set(expected.probabilities)
    for event in expected.stream_times:
        assert np.array_equal(dist.stream_times[event], expected.stream_times[event])
        assert np.array_equal(dist.probabilities[event], expected.probabilities[event])
    # the KI stream keeps its own axis (the legacy converter dropped a KI table off the KO grid)
    assert np.array_equal(dist.stream_times[EventType.KI], np.asarray(KI_TIMES))
    assert dist.flagged_streams == {}


def test_price_with_events_carries_a_flagged_stream_as_flagged():
    reason = "transition: legacy producer"
    stats = _stats(observations=_observations(KO_TIMES, KO_P), maturity=_row(1.0, 0.42),
                   ki_settlement=_row(1.0, 0.08), ki_table=_ki_table(KO_TIMES, (0.01, 0.02, 0.03, 0.04)),
                   p_ki=0.1, unavailable={"ki_table": reason})
    dist = _Producer(stats).price_with_events(_product(), _env()).event_distribution
    assert dist.flagged_streams == {EventType.KI: reason}


def test_price_with_events_on_a_terminated_note_returns_its_pv_and_a_terminated_distribution():
    stats = _terminated()
    assert stats.terminated
    result = _Producer(stats).price_with_events(_product(), _env())
    assert result.npv == stats.pv
    assert result.event_distribution.terminated
    assert result.event_distribution.probabilities == {}
    leg = FixedPayoffLeg(direction=LegDirection.BUYER_RECEIVES, amount=1.0, trigger=PaymentTrigger.AT_KO)
    with pytest.raises(ValidationError, match="terminated"):
        value_leg(leg, result.event_distribution, _env(), 1.0)


# --- execution adapters pass the statistics object through, reading only pv -----------------------------


def _event_stats_request(**kw) -> PricingRequest:
    return PricingRequest(product=_product(), pricing_env=_env(), operation=PricingOperation.EVENT_STATS, **kw)


def test_legacy_adapter_event_stats_passes_event_statistics_through():
    from quantark.execution.legacy_adapter import LegacyPriceAdapter

    stats = _snowball()
    adapter = LegacyPriceAdapter(call_shape="product_env")
    request = _event_stats_request()
    adapter.validate(_Producer(stats), request)
    value, economics = adapter.execute_native(_Producer(stats), request, None, None)
    assert value is stats
    assert economics == (("pv", stats.pv),)


def test_quad_v2_adapter_event_stats_passes_event_statistics_through():
    from quantark.asset.equity.engine.quad.v2.execution_adapter import QuadV2ExecutionAdapter

    stats = _snowball()
    request = _event_stats_request(outputs=frozenset({OutputKind.PV, OutputKind.EVENT_STATS}))
    value, economics = QuadV2ExecutionAdapter().execute_native(_Producer(stats), request, None, None)
    assert value is stats
    assert economics == (("pv", stats.pv),)


def test_pde_session_passes_the_solver_statistics_and_distribution_through():
    from quantark.asset.equity.engine.pde.pde_execution_adapters import (
        EquityPDEAutocallableSessionAdapter,
        PDESessionValue,
    )

    stats = _snowball()
    dist = EventDistribution.from_statistics(stats)

    class _Solver:
        def _session_outputs(self, product, env, *, want_events, want_grid, streams):
            return SimpleNamespace(npv=stats.pv, event_stats=stats, event_distribution=dist, solution=None)

    request = PricingRequest(product=_product(), pricing_env=_env(),
                             outputs=frozenset({OutputKind.PV, OutputKind.EVENT_STATS}))
    value, economics = EquityPDEAutocallableSessionAdapter().execute_native(_Solver(), request, None, None)
    assert isinstance(value, PDESessionValue)
    assert value.event_stats is stats and value.event_distribution is dist
    assert value.pv == stats.pv and economics == (("pv", stats.pv),)
