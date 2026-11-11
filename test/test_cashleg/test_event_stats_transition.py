"""The D15 transition module: legacy statistics read as ``EventStatistics`` while producers migrate.

``_as_event_statistics`` passes the new type through and converts each legacy type by its own named
conversion. A number the legacy producer does not compute directly under the new definition is listed in
``unavailable`` with ``"transition: legacy producer"``; an optional value the legacy object does not carry
at all is ``None`` (no stand-in). The legacy ``EventDistribution`` converters route through the module and
``EventDistribution.from_statistics``. The module is deleted in T21.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from quantark.asset.equity.analysis.autocallable_path_analyzer import RiskNeutralSnowballEventStats
from quantark.asset.equity.engine._event_stats_transition import (
    LEGACY_REASON,
    _as_event_statistics,
    _from_legacy_dcn,
    _from_legacy_ko_reset,
    _from_legacy_phoenix,
    _from_legacy_snowball,
)
from quantark.asset.equity.engine.event_stats import (
    AutocallableEventStats,
    DCNEventStats,
    EventStatistics,
    KOResetEventStats,
    PhoenixEventStats,
    StatisticsProvenance,
)
from quantark.cashleg import EventDistribution, EventType
from quantark.util.exceptions import ValidationError

KO = np.array([0.25, 0.5, 0.75, 1.0])
P_KO = np.array([0.3, 0.1, 0.05, 0.05])
KO_CASH = P_KO * 105.0 * 0.99


def _ledger(det, pay, disc, undisc=None):
    disc = np.asarray(disc, dtype=float)
    return dict(determination_times=np.asarray(det, dtype=float), payment_times=np.asarray(pay, dtype=float),
                expected_discounted_cashflows=disc,
                expected_undiscounted_cashflows=disc / 0.99 if undisc is None else np.asarray(undisc, dtype=float))


def _legacy_snowball(**overrides) -> AutocallableEventStats:
    kw = dict(
        pv=98.0, ko_times=KO, ko_probability=P_KO, survival_probability=1.0 - np.cumsum(P_KO),
        expected_discounted_ko_cashflow=KO_CASH, ki_probability=0.12, expected_discounted_maturity_cashflow=45.0,
        ki_ever_probability=0.2, ki_survive_knocked_in_probability=0.12,
        **_ledger([*KO, 1.0], [*KO + 0.01, 1.02], [*KO_CASH, 45.0]),
    )
    kw.update(overrides)
    return AutocallableEventStats(**kw)


def _ki_arrays(times, probability):
    times = np.asarray(times, dtype=float)
    probability = np.asarray(probability, dtype=float)
    return dict(ki_times=times, ki_event_probability=probability,
                ki_survival_probability=1.0 - np.cumsum(probability))


# --- pass-through and refusals -----------------------------------------------------------------------


def test_the_new_type_passes_through_unchanged():
    stats = _as_event_statistics(_legacy_snowball())
    assert _as_event_statistics(stats) is stats


def test_an_unknown_type_is_refused():
    with pytest.raises(ValidationError, match="dict"):
        _as_event_statistics({"pv": 1.0})


def test_the_path_analyzer_record_is_refused_not_guessed():
    """No payment times, no knock-in split, no ledger: T8 replaces the analyzer's type outright."""
    record = RiskNeutralSnowballEventStats(
        pv_mc=98.0, std_error=0.1, num_paths=100, ko_times=KO, ko_prob=P_KO, survive_prob=1.0 - np.cumsum(P_KO),
        expected_discounted_ko_cf=KO_CASH, ki_probability=0.2, expected_discounted_maturity_cf=45.0,
        reconciliation_error=0.0)
    with pytest.raises(ValidationError, match="RiskNeutralSnowballEventStats"):
        _as_event_statistics(record)


# --- snowball ---------------------------------------------------------------------------------------


def test_legacy_snowball_conversion_marks_what_it_cannot_compute():
    legacy = _legacy_snowball()
    stats = _from_legacy_snowball(legacy)
    assert isinstance(stats, EventStatistics) and stats.product_family == "snowball"
    assert stats.provenance == StatisticsProvenance(statistics_model="legacy", price_model="legacy")
    np.testing.assert_array_equal(stats.observations.probability, P_KO)
    np.testing.assert_array_equal(stats.observations.payment_times, KO + 0.01)
    assert stats.ki_settlement.probability == 0.12 == stats.knock_in.probability_no_ko
    # residual-derived legacy maturity: the old converter's 1 - ΣKO - ki
    assert stats.maturity.probability == pytest.approx(1.0 - P_KO.sum() - 0.12, abs=1e-15)
    assert stats.maturity.payment_time == 1.02
    # no stand-ins: values the legacy object does not carry are None, with the transition reason
    assert stats.ki_table is None and stats.knock_in.expected_time is None and stats.knock_in.expected_loss is None
    for key in ("ki_table", "knock_in.expected_time", "knock_in.expected_loss", "maturity.probability",
                "knock_in.probability", "maturity.discounted_cash", "ki_settlement.discounted_cash",
                "receivables", "expected_lifetime"):
        assert stats.unavailable[key] == LEGACY_REASON, key
    # the legacy producer's own payment ledger, verbatim
    np.testing.assert_array_equal(stats.ledger.discounted_cash, legacy.expected_discounted_cashflows)
    assert stats.ledger.total_discounted_cash == pytest.approx(stats.total_discounted_cash, rel=1e-12)


def test_legacy_provenance_is_kept():
    prov = StatisticsProvenance(statistics_model="flat_vol_1d", price_model="flat_vol_1d")
    assert _from_legacy_snowball(_legacy_snowball(provenance=prov)).provenance == prov


def test_legacy_ki_arrays_become_a_ki_table_on_their_own_axis():
    """The base drop: a KI stream whose length differed from the KO grid vanished from the distribution."""
    times = np.linspace(0.1, 1.0, 10)
    legacy = _legacy_snowball(**_ki_arrays(times, np.full(10, 0.02)))
    stats = _from_legacy_snowball(legacy)
    np.testing.assert_array_equal(stats.ki_table.times, times)
    np.testing.assert_array_equal(stats.ki_table.payment_times, times)          # D8
    assert stats.unavailable["ki_table"] == LEGACY_REASON                         # legacy KI definition
    dist = EventDistribution.from_autocallable_stats(legacy)
    np.testing.assert_array_equal(dist.stream_times[EventType.KI], times)


def test_legacy_converter_routes_through_the_transition_module():
    legacy = _legacy_snowball()
    routed = EventDistribution.from_autocallable_stats(legacy)
    direct = EventDistribution.from_statistics(_as_event_statistics(legacy))
    for event in direct.probabilities:
        np.testing.assert_array_equal(np.asarray(routed.probabilities[event]), np.asarray(direct.probabilities[event]))
    np.testing.assert_array_equal(routed.survival_probability, direct.survival_probability)
    # the legacy distribution's maturity split: 1 - ΣKO - ki, ki
    assert routed.probabilities[EventType.MATURITY_NO_KO] == pytest.approx(1.0 - P_KO.sum() - 0.12, abs=1e-15)
    assert routed.probabilities[EventType.MATURITY_WITH_KI] == 0.12


# --- phoenix ----------------------------------------------------------------------------------------


def _legacy_phoenix(path_dependent=False, coupon_probability=(0.9, 0.6, 0.5, 0.4)) -> PhoenixEventStats:
    coupon_cash = np.array([0.5, 0.4, 0.3, 0.2]) if coupon_probability else np.array([])
    rows = [*KO_CASH, *coupon_cash, 44.0]
    det = [*KO, *(KO if coupon_cash.size else []), 1.0]
    pay = [*KO + 0.01, *(KO + 0.015 if coupon_cash.size else []), 1.02]
    return PhoenixEventStats(
        pv=98.0, ko_times=KO, ko_probability=P_KO, survival_probability=1.0 - np.cumsum(P_KO),
        expected_discounted_ko_cashflow=KO_CASH, ki_probability=0.12, expected_discounted_maturity_cashflow=44.0,
        ki_survive_knocked_in_probability=0.12, coupon_probability=np.asarray(coupon_probability, dtype=float),
        expected_discounted_coupon_cashflow=coupon_cash, coupon_payment_is_path_dependent=path_dependent,
        **_ledger(det, pay, rows),
    )


def test_legacy_phoenix_conversion_keeps_the_coupon_grid():
    stats = _from_legacy_phoenix(_legacy_phoenix())
    assert stats.product_family == "phoenix"
    np.testing.assert_array_equal(stats.coupons.probability, [0.9, 0.6, 0.5, 0.4])
    np.testing.assert_array_equal(stats.coupons.payment_times, KO + 0.015)
    assert stats.unavailable["coupons.survival"] == LEGACY_REASON
    dist = EventDistribution.from_statistics(stats)
    np.testing.assert_array_equal(dist.payment_times_for(EventType.COUPON), KO + 0.015)


def test_legacy_phoenix_path_dependent_coupons_stay_path_dependent():
    stats = _from_legacy_phoenix(_legacy_phoenix(path_dependent=True))
    assert stats.coupons.payment_is_path_dependent.all()
    assert EventType.COUPON in EventDistribution.from_statistics(stats).path_dependent_payment_events


def test_legacy_phoenix_without_coupon_probabilities_has_no_coupon_stream():
    stats = _from_legacy_phoenix(_legacy_phoenix(coupon_probability=()))
    assert stats.coupons.probability is None
    assert stats.unavailable["coupons.probability"] == LEGACY_REASON
    assert EventType.COUPON not in EventDistribution.from_statistics(stats).probabilities


# --- KO-reset ---------------------------------------------------------------------------------------

PRE = np.array([0.5, 1.0])
POST = np.array([0.5, 1.0, 1.5, 2.0])
P_PRE = np.array([0.3, 0.1])
P_POST = np.array([0.0, 0.02, 0.03, 0.02])


def _legacy_ko_reset(buckets: bool, **overrides) -> KOResetEventStats:
    n = PRE.size + POST.size
    kw = dict(
        pv=95.0, ko_times=PRE, ko_probability=P_PRE, survival_probability=1.0 - np.cumsum(P_PRE),
        expected_discounted_ko_cashflow=P_PRE * 104.0, ki_probability=0.2, expected_discounted_maturity_cashflow=50.0,
        ki_survive_knocked_in_probability=0.2, pre_ko_times=PRE, pre_ko_probability=P_PRE, post_ko_times=POST,
        post_ko_probability=P_POST, pre_ko_payment_times=PRE + 0.01, post_ko_payment_times=POST + 0.01,
        **_ledger([*PRE, *POST, 1.0, 2.0], [*PRE + 0.01, *POST + 0.01, 1.01, 2.01], [*(np.ones(n)), 30.0, 20.0]),
    )
    if buckets:
        kw.update(pre_maturity_probability=0.33, post_maturity_probability=0.2, pre_maturity_time=1.0,
                  post_maturity_time=2.0, pre_maturity_payment_time=1.01, post_maturity_payment_time=2.01,
                  expected_discounted_pre_maturity_cashflow=30.0, expected_discounted_post_maturity_cashflow=20.0)
    kw.update(overrides)
    return KOResetEventStats(**kw)


@pytest.mark.parametrize("buckets", [True, False])
def test_legacy_ko_reset_merges_both_schedules(buckets):
    stats = _from_legacy_ko_reset(_legacy_ko_reset(buckets))
    assert stats.product_family == "ko_reset"
    np.testing.assert_array_equal(stats.observations.times, [0.5, 0.5, 1.0, 1.0, 1.5, 2.0])
    assert stats.observations.schedule == ("pre", "post", "pre", "post", "post", "post")
    if buckets:
        # direct maturity buckets: never knocked in at the pre maturity, knocked in at the post maturity
        assert (stats.maturity.probability, stats.maturity.time) == (0.33, 1.0)
        assert (stats.ki_settlement.probability, stats.ki_settlement.time) == (0.2, 2.0)
        assert "maturity.probability" not in stats.unavailable
        # F5: the pre-maturity mass leaves the survival at its own determination
        assert stats.observations.survival[2] == pytest.approx(1.0 - 0.4 - 0.33)
    else:
        assert stats.unavailable["maturity.probability"] == LEGACY_REASON
        assert stats.maturity.probability == pytest.approx(1.0 - 0.47 - 0.2, abs=1e-15)


def test_legacy_rebased_offsets_and_missing_payment_times_are_refused_as_before():
    with pytest.raises(NotImplementedError, match="offsets"):
        _from_legacy_ko_reset(_legacy_ko_reset(False, post_ko_times_are_offsets=True))
    with pytest.raises(ValidationError, match="payment times"):
        _from_legacy_ko_reset(_legacy_ko_reset(False, post_ko_payment_times=None))


# --- DCN --------------------------------------------------------------------------------------------


def _legacy_dcn() -> DCNEventStats:
    obs = np.array([0.1, 0.2, 0.3])
    ko = (0.0, 0.2, 0.1)
    coupon_cash, ko_cash, loss = [0.8, 0.6, 0.5], [0.0, 2.0, 1.0], -3.0
    return DCNEventStats(
        ki_probability=0.25, ko_probability=0.3, ko_timing_distribution=ko, coupon_probability=(0.9, 0.7, 0.6),
        expected_life_years=0.25, prob_survive_no_ki=0.5, prob_survive_ki=0.2, expected_discounted_loss_leg=loss,
        pv=0.9, **_ledger([*obs, *obs, 0.3], [*obs + 0.005, *obs + 0.006, 0.31], [*coupon_cash, *ko_cash, loss]),
    )


def test_legacy_dcn_conversion():
    stats = _from_legacy_dcn(_legacy_dcn())
    assert stats.product_family == "dcn"
    np.testing.assert_array_equal(stats.observations.probability, [0.0, 0.2, 0.1])
    np.testing.assert_allclose(stats.observations.payment_times, [0.106, 0.206, 0.306], rtol=0, atol=1e-15)
    np.testing.assert_allclose(stats.coupons.payment_times, [0.105, 0.205, 0.305], rtol=0, atol=1e-15)
    assert (stats.maturity.probability, stats.ki_settlement.probability) == (0.5, 0.2)
    assert stats.ki_settlement.discounted_cash == -3.0 and stats.maturity.discounted_cash == 0.0
    assert stats.expected_lifetime == 0.25
    # legacy P(coupon paid) excludes knock-out rows: not the new coupon-table definition
    assert stats.coupons.probability is None and stats.unavailable["coupons.probability"] == LEGACY_REASON
    dist = EventDistribution.from_dcn_stats(_legacy_dcn())
    assert EventType.COUPON not in dist.probabilities
    assert dist.probabilities[EventType.MATURITY_NO_KO] == pytest.approx(0.5, abs=1e-15)
    assert dist.payment_times_for(EventType.MATURITY_WITH_KI) == 0.31


def test_a_legacy_dcn_ledger_must_be_laid_out_as_before():
    bad = dataclasses.replace(_legacy_dcn(), **_ledger([0.1], [0.1], [1.0]))
    with pytest.raises(ValidationError, match="ledger"):
        _from_legacy_dcn(bad)
