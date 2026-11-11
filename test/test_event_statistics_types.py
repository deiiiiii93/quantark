"""Unified event-statistics family (spec 2026-11-11 §Contracts, AC6).

Hand-built instances with known answers: shape/finite/negative-probability/monotone-survival
validation, residuals computed by the type, `unavailable` and `standard_errors` rules, the KI
table (D8 payment time, D18 interpolation label), REBASED offsets + termination (D13),
receivables outside the probability identity, immutability, and the direct-cash ledger builder.
"""

import dataclasses
import math
import pickle
from datetime import datetime, timedelta

import numpy as np
import pytest

from quantark.asset.equity.engine.event_stats import (
    CashLedger,
    CouponTable,
    EventStatistics,
    KnockInStatistics,
    KnockInTable,
    ObservationTable,
    OutcomeRow,
    PostKIOffsetTable,
    ReceivablesTable,
    StatisticsProvenance,
    TerminationDistribution,
    build_cash_ledger,
    build_rebased_cash_ledger,
)
from quantark.util.exceptions import ValidationError

PROV = StatisticsProvenance("flat_vol_1d", "flat_vol_1d")
D0 = datetime(2026, 1, 5)


def _dates(times):
    return tuple(D0 + timedelta(days=round(365 * t)) for t in times)


# --------------------------------------------------------------------------- builders


def _observations(**kw):
    base = dict(
        times=[0.25, 0.5, 0.75, 1.0],
        payment_times=[0.26, 0.51, 0.76, 1.01],
        is_ko_obs=[False, True, True, True],
        is_coupon_obs=[True, True, True, True],
        probability=[0.0, 0.25, 0.125, 0.0625],
        survival=[1.0, 0.75, 0.625, 0.5625],
        discounted_cash=[0.0, 0.25, 0.125, 0.0625],
        cash_to_pay=[0.0, 0.25, 0.125, 0.0625],
        payment_is_path_dependent=[False, False, False, False],
    )
    base.update(kw)
    return ObservationTable(**base)


def _coupons(**kw):
    base = dict(
        probability=[0.5, 0.5, 0.25, 0.25],
        survival=[1.0, 1.0, 0.75, 0.625],
        payment_times=[0.26, 0.51, 0.76, 1.01],
        discounted_cash=[0.03125, 0.03125, 0.015625, 0.015625],
        cash_to_pay=[0.03125, 0.03125, 0.015625, 0.015625],
        payment_is_path_dependent=[False, False, False, False],
    )
    base.update(kw)
    return CouponTable(**base)


def _row(probability, cash, time=1.0, payment_time=1.01, **kw):
    return OutcomeRow(time=time, payment_time=payment_time, probability=probability,
                      discounted_cash=cash, cash_to_pay=cash, **kw)


def _ki_table(**kw):
    base = dict(times=[0.5, 1.0], payment_times=[0.5, 1.0], probability=[0.1875, 0.125],
                survival=[0.5625, 0.375], interpolated=[False, False], interpolation_scheme=None)
    base.update(kw)
    return KnockInTable(**base)


def _knock_in(**kw):
    base = dict(probability=0.3125, probability_no_ko=0.25, expected_time=(0.5 * 0.1875 + 1.0 * 0.125) / 0.3125,
                expected_loss=0.0625)
    base.update(kw)
    return KnockInStatistics(**base)


def _ledger_for(observations, coupons, maturity, ki_settlement, receivables):
    return build_cash_ledger(observations=observations, coupons=coupons, maturity=maturity,
                             ki_settlement=ki_settlement, receivables=receivables)


def _phoenix(**kw):
    """Phoenix: KO 0.4375, maturity 0.3125, KI settlement 0.25 -> residual 0; cash known."""
    observations = kw.pop("observations", _observations())
    coupons = kw.pop("coupons", _coupons())
    maturity = kw.pop("maturity", _row(0.3125, 0.3125))
    ki_settlement = kw.pop("ki_settlement", _row(0.25, 0.125))
    receivables = kw.pop("receivables", ReceivablesTable.empty())
    ledger = kw.pop("ledger", None)
    if ledger is None:
        ledger = _ledger_for(observations, coupons, maturity, ki_settlement, receivables)
    base = dict(
        product_family="phoenix", pv=1.0, observations=observations, coupons=coupons, maturity=maturity,
        ki_settlement=ki_settlement, knock_in=_knock_in(), ki_table=_ki_table(), receivables=receivables,
        ledger=ledger, expected_lifetime=0.8, provenance=PROV,
    )
    base.update(kw)
    return EventStatistics(**base)


# sum of the hand-built Phoenix cash: KO 0.4375 + coupons 0.09375 + maturity 0.3125 + KI 0.125
PHOENIX_CASH = 0.4375 + 0.09375 + 0.3125 + 0.125


# --------------------------------------------------------------------------- identities


def test_residuals_are_computed_with_known_answers():
    stats = _phoenix(pv=1.0)
    assert stats.probability_residual == pytest.approx(0.0, abs=1e-15)
    assert stats.reconciliation_error == pytest.approx(1.0 - PHOENIX_CASH, abs=1e-15)
    assert stats.total_discounted_cash == pytest.approx(PHOENIX_CASH, abs=1e-15)
    assert stats.total_cash_to_pay == pytest.approx(PHOENIX_CASH, abs=1e-15)


def test_probability_residual_reports_a_missing_mass_and_never_forces_it():
    stats = _phoenix(maturity=_row(0.3, 0.3125))
    assert stats.probability_residual == pytest.approx(0.0125, abs=1e-15)
    assert stats.maturity.probability == 0.3          # not rescaled


@pytest.mark.parametrize("name", ["probability_residual", "reconciliation_error",
                                  "total_discounted_cash", "total_cash_to_pay"])
def test_computed_fields_cannot_be_passed_in(name):
    with pytest.raises(TypeError):
        _phoenix(**{name: 0.0})


def test_replace_recomputes_the_residuals():
    stats = dataclasses.replace(_phoenix(), pv=PHOENIX_CASH)
    assert stats.reconciliation_error == pytest.approx(0.0, abs=1e-15)


def test_ledger_sum_must_equal_the_outcome_cash():
    good = _phoenix()
    bad_ledger = CashLedger(determination_times=[1.0], payment_times=[1.01],
                            discounted_cash=[good.total_discounted_cash + 1e-3], cash_to_pay=[0.0])
    with pytest.raises(ValidationError, match="ledger"):
        _phoenix(ledger=bad_ledger)


def test_ki_settlement_probability_must_equal_knock_in_probability_no_ko_exactly():
    with pytest.raises(ValidationError, match="probability_no_ko"):
        _phoenix(knock_in=_knock_in(probability_no_ko=0.25 + 1e-15))


def test_provenance_is_required():
    with pytest.raises(ValidationError, match="provenance"):
        _phoenix(provenance=None)


def test_unknown_product_family_is_rejected():
    with pytest.raises(ValidationError, match="product_family"):
        _phoenix(product_family="accumulator")


# --------------------------------------------------------------------------- table validation


@pytest.mark.parametrize("field,value", [
    ("probability", [0.0, 0.25, 0.125]),               # misaligned
    ("survival", [[1.0, 0.75], [0.625, 0.5625]]),      # 2-D
    ("discounted_cash", [0.0, np.nan, 0.125, 0.0625]),
    ("cash_to_pay", [0.0, 0.25, np.inf, 0.0625]),
    ("times", [0.25, 0.5, np.nan, 1.0]),
    ("probability", [0.0, -0.01, 0.125, 0.0625]),      # negative probability
    ("survival", [1.0, 0.75, 0.8, 0.5625]),            # survival increases
    ("survival", [1.0, 0.75, 0.625, -0.01]),           # negative survival
    ("times", [0.25, 0.75, 0.5, 1.0]),                 # unsorted
    ("times", [0.25, 0.5, 0.5, 1.0]),                  # duplicate time, no schedule tags
    ("times", [-0.25, 0.5, 0.75, 1.0]),                # before valuation: belongs in receivables
    ("payment_times", [0.26, 0.4, 0.76, 1.01]),        # paid before determined
    ("is_ko_obs", [0, 1, 1, 1]),                       # flags must be bool
    ("dates", _dates([0.25, 0.5, 0.75])),              # misaligned dates
    ("dates", ("2026-04-06",) * 4),                    # not datetime
])
def test_observation_table_rejects(field, value):
    with pytest.raises(ValidationError):
        _observations(**{field: value})


def test_observation_rows_carry_no_ko_mass_on_coupon_only_rows():
    with pytest.raises(ValidationError, match="is_ko_obs"):
        _observations(probability=[0.01, 0.25, 0.125, 0.0625])
    with pytest.raises(ValidationError, match="at least one"):
        _observations(is_coupon_obs=[False, True, True, True])


def test_observation_table_allows_pre_post_ties_only():
    table = _observations(schedule=("pre", "post", "pre", "post"), times=[0.25, 0.25, 0.5, 0.5],
                          payment_times=[0.26, 0.26, 0.51, 0.51], is_ko_obs=[True] * 4,
                          is_coupon_obs=[True] * 4)
    assert table.schedule == ("pre", "post", "pre", "post")
    with pytest.raises(ValidationError, match="duplicate"):
        _observations(schedule=("pre", "pre", "post", "post"), times=[0.25, 0.25, 0.5, 0.5],
                      payment_times=[0.26, 0.26, 0.51, 0.51], is_ko_obs=[True] * 4)
    with pytest.raises(ValidationError, match="schedule"):
        _observations(schedule=("pre", "mid", "pre", "post"))


@pytest.mark.parametrize("field,value", [
    ("probability", [0.5, -0.5, 0.25, 0.25]),
    ("probability", [0.5, 0.5, 0.25]),
    ("survival", [1.0, 1.0, 0.75, 0.8]),
    ("discounted_cash", [0.03125, np.nan, 0.015625, 0.015625]),
    ("payment_is_path_dependent", [False, False, False]),
])
def test_coupon_table_rejects(field, value):
    with pytest.raises(ValidationError):
        _coupons(**{field: value})


def test_coupon_table_aligns_with_the_observation_grid():
    with pytest.raises(ValidationError, match="observation"):
        _phoenix(coupons=_coupons(probability=[0.5, 0.5, 0.25, 0.25, 0.0],
                                  survival=[1.0, 1.0, 0.75, 0.625, 0.5],
                                  payment_times=[0.26, 0.51, 0.76, 1.01, 1.01],
                                  discounted_cash=[0.0] * 5, cash_to_pay=[0.0] * 5,
                                  payment_is_path_dependent=[False] * 5))
    with pytest.raises(ValidationError, match="payment"):    # coupon paid before it is earned
        _phoenix(coupons=_coupons(payment_times=[0.2, 0.51, 0.76, 1.01]))


@pytest.mark.parametrize("kw", [
    dict(probability=-0.01, discounted_cash=0.0),
    dict(probability=math.nan, discounted_cash=0.0),
    dict(probability=0.1, discounted_cash=math.inf),
    dict(probability=0.1, discounted_cash=0.1, time=1.0, payment_time=0.9),
    dict(probability=0.1, discounted_cash=0.1, date="2027-01-05"),
])
def test_outcome_row_rejects(kw):
    with pytest.raises(ValidationError):
        _row(kw.pop("probability"), kw.pop("discounted_cash"), **kw)


@pytest.mark.parametrize("kw", [
    dict(probability=-0.1),
    dict(probability_no_ko=-0.1),
    dict(expected_loss=-0.1),
    dict(expected_time=math.nan),
    dict(probability=0.0, expected_time=0.5),          # undefined when P(KI) = 0
])
def test_knock_in_statistics_rejects(kw):
    with pytest.raises(ValidationError):
        _knock_in(**kw)


def test_knock_in_with_zero_probability_has_undefined_expected_time():
    ki = KnockInStatistics(probability=0.0, probability_no_ko=0.0, expected_time=None, expected_loss=0.0)
    assert ki.expected_time is None


# --------------------------------------------------------------------------- KI table (D8, D18)


def test_ki_table_payment_time_is_the_observation_time_d8():
    dates = _dates([0.5, 1.0])
    table = _ki_table(dates=dates, payment_dates=dates)
    assert np.array_equal(table.payment_times, table.times)
    with pytest.raises(ValidationError, match="D8"):
        _ki_table(payment_times=[0.51, 1.01])
    with pytest.raises(ValidationError, match="D8"):
        _ki_table(dates=dates, payment_dates=_dates([0.6, 1.1]))


def test_ki_table_step_overlap_rows_are_labelled_d18():
    t = np.arange(1, 6) / 244.0
    table = _ki_table(times=t, payment_times=t, probability=[0.01] * 5,
                      survival=[0.99, 0.98, 0.97, 0.96, 0.95], interpolated=[True] * 5,
                      interpolation_scheme="step_overlap")
    assert table.interpolated.all() and table.interpolation_scheme == "step_overlap"
    with pytest.raises(ValidationError, match="interpolation_scheme"):
        _ki_table(interpolated=[True, False], interpolation_scheme=None)
    with pytest.raises(ValidationError, match="interpolation_scheme"):
        _ki_table(interpolation_scheme="step_overlap")


@pytest.mark.parametrize("kw", [
    dict(times=[1.0, 0.5], payment_times=[1.0, 0.5]),
    dict(times=[0.5, 0.5], payment_times=[0.5, 0.5]),
    dict(probability=[0.1875, -0.125]),
    dict(survival=[0.375, 0.5625]),
    dict(interpolated=[False]),
    dict(probability=[0.1875, np.nan]),
])
def test_ki_table_rejects(kw):
    with pytest.raises(ValidationError):
        _ki_table(**kw)


def test_empty_ki_table_for_no_ki_barrier():
    table = KnockInTable.empty()
    assert table.size == 0 and table.interpolation_scheme is None


# --------------------------------------------------------------------------- product shape


def test_snowball_and_ko_reset_have_no_coupon_table_ac11():
    with pytest.raises(ValidationError, match="coupons"):
        _phoenix(product_family="snowball")
    with pytest.raises(ValidationError, match="coupons"):
        _phoenix(coupons=None)


def test_schedule_tags_only_on_ko_reset():
    with pytest.raises(ValidationError, match="schedule"):
        _phoenix(observations=_observations(schedule=("pre",) * 4))


# --------------------------------------------------------------------------- unavailable


def test_pruned_values_need_an_unavailable_reason():
    pruned = dict(ki_table=None, knock_in=_knock_in(probability=None, expected_time=None, expected_loss=None),
                  coupons=_coupons(probability=None))
    with pytest.raises(ValidationError, match="unavailable"):
        _phoenix(**pruned)
    reasons = {key: "not requested" for key in ("ki_table", "knock_in.probability", "knock_in.expected_time",
                                                "knock_in.expected_loss", "coupons.probability")}
    stats = _phoenix(**pruned, unavailable=reasons)
    assert dict(stats.unavailable) == reasons
    assert stats.probability_residual == pytest.approx(0.0, abs=1e-15)   # never pruned


def test_undefined_expected_time_reason():
    stats = _phoenix(knock_in=_knock_in(probability=0.0, expected_time=None),
                     unavailable={"knock_in.expected_time": "undefined: P(knock in) = 0"})
    assert stats.unavailable["knock_in.expected_time"].startswith("undefined:")


@pytest.mark.parametrize("unavailable", [
    {"knock_in.expected_tiem": "not requested"},          # not a field path
    {"ki_table.probability.x": "not requested"},
    {"ki_table": ""},                                      # empty reason
    {"": "not requested"},
])
def test_unavailable_keys_are_field_paths_with_reasons(unavailable):
    with pytest.raises(ValidationError):
        _phoenix(unavailable=unavailable)


# --------------------------------------------------------------------------- standard errors


def test_standard_errors_keyed_by_dotted_field_paths():
    se = {"observations.probability": [0.0, 0.01, 0.01, 0.01], "maturity.discounted_cash": 0.002,
          "knock_in.expected_time": 0.03, "pv": 0.004}
    stats = _phoenix(standard_errors=se)
    assert stats.standard_errors["maturity.discounted_cash"] == 0.002
    assert stats.standard_errors["observations.probability"].shape == (4,)
    assert not stats.standard_errors["observations.probability"].flags.writeable


@pytest.mark.parametrize("se", [
    {"observations.probability": [0.01, 0.01]},           # misaligned with its table
    {"observations.probabilty": [0.0] * 4},               # not a field path
    {"observations": 0.1},                                 # not a numeric field
    {"observations.is_ko_obs": [0.0] * 4},                 # a flag has no SE
    {"maturity.discounted_cash": -0.1},
    {"maturity.discounted_cash": math.nan},
    {"maturity.discounted_cash": [0.1, 0.1]},              # scalar field, array SE
    {},                                                    # empty mapping: use None
])
def test_standard_errors_reject(se):
    with pytest.raises(ValidationError):
        _phoenix(standard_errors=se)


def test_standard_error_on_a_none_field_is_rejected():
    with pytest.raises(ValidationError):
        _phoenix(knock_in=_knock_in(probability=0.0, expected_time=None),
                 unavailable={"knock_in.expected_time": "undefined: P(knock in) = 0"},
                 standard_errors={"knock_in.expected_time": 0.1})


def test_single_batch_rule():
    stats = _phoenix(standard_errors=None, unavailable={"standard_errors": "undefined: single batch"})
    assert stats.standard_errors is None
    with pytest.raises(ValidationError, match="standard_errors"):
        _phoenix(standard_errors={"pv": 0.1}, unavailable={"standard_errors": "undefined: single batch"})


def test_undefined_standard_error_of_one_field():
    stats = _phoenix(standard_errors={"pv": 0.1},
                     unavailable={"standard_errors.knock_in.expected_time": "undefined: fewer than 2 KI paths"})
    assert "knock_in.expected_time" not in stats.standard_errors


# --------------------------------------------------------------------------- immutability


def test_instances_and_arrays_are_frozen_copies():
    probability = np.array([0.0, 0.25, 0.125, 0.0625])
    obs = _observations(probability=probability)
    probability[1] = 0.9
    assert obs.probability[1] == 0.25
    assert not obs.probability.flags.writeable
    with pytest.raises(ValueError):
        obs.survival[0] = 0.5
    stats = _phoenix(unavailable={"standard_errors": "undefined: single batch"})
    with pytest.raises(dataclasses.FrozenInstanceError):
        stats.pv = 2.0
    with pytest.raises(TypeError):
        stats.unavailable["x"] = "y"
    for array in (stats.coupons.discounted_cash, stats.ki_table.probability, stats.ledger.discounted_cash,
                  stats.receivables.payment_times, stats.observations.is_ko_obs):
        assert not array.flags.writeable


def test_pickle_round_trip():
    stats = _phoenix(standard_errors={"pv": 0.1}, unavailable={"standard_errors.knock_in.expected_time": "undefined: x"})
    back = pickle.loads(pickle.dumps(stats))
    assert back.reconciliation_error == stats.reconciliation_error
    assert dict(back.unavailable) == dict(stats.unavailable)


# --------------------------------------------------------------------------- receivables, terminated, DCN


def _receivables(cash, payment=0.02, source="ko"):
    return ReceivablesTable(source=(source,), decision_times=[-0.01], payment_times=[payment],
                            discounted_cash=[cash], cash_to_pay=[cash])


def test_receivables_are_in_the_cash_identity_not_the_probability_identity():
    receivables = _receivables(0.5)
    stats = _phoenix(receivables=receivables, pv=PHOENIX_CASH + 0.5)
    assert stats.reconciliation_error == pytest.approx(0.0, abs=1e-15)
    assert stats.probability_residual == pytest.approx(0.0, abs=1e-15)
    assert stats.total_cash_to_pay == pytest.approx(PHOENIX_CASH + 0.5, abs=1e-15)


@pytest.mark.parametrize("kw", [
    dict(source=("loan",), decision_times=[-0.01], payment_times=[0.02], discounted_cash=[1.0], cash_to_pay=[1.0]),
    dict(source=("ko",), decision_times=[0.5], payment_times=[0.6], discounted_cash=[1.0], cash_to_pay=[1.0]),
    dict(source=("ko",), decision_times=[-0.01], payment_times=[-0.02], discounted_cash=[1.0], cash_to_pay=[1.0]),
    dict(source=("ko", "coupon"), decision_times=[-0.01], payment_times=[0.02], discounted_cash=[1.0],
         cash_to_pay=[1.0]),
    dict(source=("ko",), decision_times=[-0.01], payment_times=[0.02], discounted_cash=[np.nan], cash_to_pay=[1.0]),
])
def test_receivables_table_rejects(kw):
    with pytest.raises(ValidationError):
        ReceivablesTable(**kw)


def _dcn_seller(receivables=None):
    """Seller DCN (direction -1): negative cash throughout, coupon-only lock-up row first."""
    obs = _observations(discounted_cash=[0.0, -0.25, -0.125, -0.0625], cash_to_pay=[0.0, -0.25, -0.125, -0.0625])
    coupons = _coupons(discounted_cash=[-0.03125, -0.03125, -0.015625, -0.015625],
                       cash_to_pay=[-0.03125, -0.03125, -0.015625, -0.015625])
    maturity, ki = _row(0.3125, -0.3125), _row(0.25, -0.125)
    receivables = receivables if receivables is not None else ReceivablesTable.empty()
    return _phoenix(product_family="dcn", pv=-PHOENIX_CASH + float(np.sum(receivables.discounted_cash)),
                    observations=obs, coupons=coupons, maturity=maturity, ki_settlement=ki,
                    receivables=receivables, knock_in=_knock_in(expected_loss=0.0625))


def test_seller_dcn_reports_negative_cash_with_identities_intact():
    stats = _dcn_seller(_receivables(-0.5, source="dcn_loss"))
    assert stats.reconciliation_error == pytest.approx(0.0, abs=1e-15)
    assert stats.probability_residual == pytest.approx(0.0, abs=1e-15)
    assert stats.total_discounted_cash < 0 and stats.knock_in.expected_loss > 0


def test_terminated_unpaid_note_reports_receivables_only():
    empty_obs = ObservationTable(times=[], payment_times=[], is_ko_obs=[], is_coupon_obs=[], probability=[],
                                 survival=[], discounted_cash=[], cash_to_pay=[], payment_is_path_dependent=[])
    empty_coupons = CouponTable(probability=[], survival=[], payment_times=[], discounted_cash=[], cash_to_pay=[],
                                payment_is_path_dependent=[])
    kw = dict(observations=empty_obs, coupons=empty_coupons, maturity=_row(0.0, 0.0), ki_settlement=_row(0.0, 0.0),
              knock_in=KnockInStatistics(probability=0.0, probability_no_ko=0.0, expected_time=None,
                                         expected_loss=0.0),
              ki_table=KnockInTable.empty(), expected_lifetime=0.0,
              unavailable={"knock_in.expected_time": "undefined: terminated"})
    stats = _phoenix(receivables=_receivables(0.75), pv=0.8, **kw)
    assert stats.terminated
    assert stats.probability_residual == 0.0
    assert stats.reconciliation_error == pytest.approx(0.05, abs=1e-15)
    with pytest.raises(ValidationError, match="settled"):
        _phoenix(receivables=ReceivablesTable.empty(), pv=0.0, **kw)
    assert not _phoenix().terminated


# --------------------------------------------------------------------------- REBASED KO-reset (D13)


def _rebased(**kw):
    """REBASED KO-reset on a deterministic candidate axis: KI dates 0.25/0.5, offsets +0.25/+0.5.

    MATURITY_WITH_KI is an array stream (one row per candidate settlement date, zeros kept).
    """
    obs = _observations(times=[0.25, 0.5, 0.75, 1.0], payment_times=[0.25, 0.5, 0.75, 1.0],
                        is_ko_obs=[True] * 4, is_coupon_obs=[False] * 4, schedule=("pre",) * 4,
                        probability=[0.125, 0.125, 0.0625, 0.0625], survival=[0.875, 0.75, 0.6875, 0.625],
                        discounted_cash=[0.125, 0.125, 0.0625, 0.0625], cash_to_pay=[0.125, 0.125, 0.0625, 0.0625],
                        payment_is_path_dependent=[False] * 4)
    termination = TerminationDistribution(
        times=[0.25, 0.5, 0.5, 0.75, 0.75, 0.75, 1.0, 1.0, 1.0],
        payment_times=[0.25, 0.5, 0.5, 0.75, 0.75, 0.75, 1.0, 1.0, 1.0],
        event=("ko_pre", "ko_pre", "ko_post", "ko_pre", "ko_post", "ki_settlement",
               "ko_pre", "maturity", "ki_settlement"),
        probability=[0.125, 0.125, 0.0625, 0.0625, 0.0, 0.125, 0.0625, 0.25, 0.1875],
        discounted_cash=[0.125, 0.125, 0.0625, 0.0625, 0.0, 0.0625, 0.0625, 0.25, 0.09375],
        cash_to_pay=[0.125, 0.125, 0.0625, 0.0625, 0.0, 0.0625, 0.0625, 0.25, 0.09375],
        survival_times=[0.25, 0.5, 0.75, 1.0], survival=[0.875, 0.6875, 0.5, 0.0],
    )
    offsets = PostKIOffsetTable(offsets=[0.25, 0.5, 0.5], kind=("ko_post", "ko_post", "ki_settlement"),
                                probability=[0.0625, 0.0, 0.3125], discounted_cash=[0.0625, 0.0, 0.15625],
                                cash_to_pay=[0.0625, 0.0, 0.15625])
    receivables = kw.pop("receivables", ReceivablesTable.empty())
    ledger = build_rebased_cash_ledger(termination=kw.pop("termination", termination), receivables=receivables)
    base = dict(
        product_family="ko_reset", pv=1.0, observations=obs, coupons=None,
        maturity=_row(0.25, 0.25), ki_settlement=_row(0.3125, 0.15625),
        knock_in=_knock_in(probability=0.375, probability_no_ko=0.3125, expected_time=0.4),
        ki_table=_ki_table(times=[0.25, 0.5], payment_times=[0.25, 0.5], probability=[0.25, 0.125],
                           survival=[0.625, 0.375]),
        post_ki_offsets=offsets, termination=termination, receivables=receivables, ledger=ledger,
        expected_lifetime=0.7, provenance=StatisticsProvenance("mc", "mc"),
    )
    base.update(kw)
    return EventStatistics(**base)


def test_rebased_residuals_come_from_the_termination_distribution():
    stats = _rebased()
    total = 0.125 + 0.125 + 0.0625 + 0.0625 + 0.0625 + 0.0625 + 0.25 + 0.09375
    assert stats.rebased
    assert stats.probability_residual == pytest.approx(0.0, abs=1e-15)
    assert stats.reconciliation_error == pytest.approx(1.0 - total, abs=1e-15)
    ki_rows = np.array(stats.termination.event) == "ki_settlement"
    assert ki_rows.sum() == 2                                  # array stream, not a scalar outcome
    assert float(stats.termination.probability[ki_rows].sum()) == stats.ki_settlement.probability


def test_rebased_zero_rows_are_kept_on_the_candidate_axis():
    stats = _rebased()
    assert 0.0 in stats.termination.probability.tolist()
    assert stats.termination.size == 9


def test_rebased_needs_both_tables_and_ko_reset():
    stats = _rebased()
    with pytest.raises(ValidationError, match="REBASED"):
        dataclasses.replace(stats, post_ki_offsets=None)
    with pytest.raises(ValidationError, match="REBASED"):
        dataclasses.replace(stats, product_family="snowball", coupons=None,
                            observations=dataclasses.replace(stats.observations, schedule=None))
    with pytest.raises(ValidationError, match="post"):
        dataclasses.replace(stats, observations=dataclasses.replace(
            stats.observations, schedule=("pre", "post", "pre", "pre")))


@pytest.mark.parametrize("kw", [
    dict(event=("ko_pre", "ko_pre", "ko_post", "ko_pre", "ko_post", "ki_settlement", "ko_pre", "maturity",
                "ki_settlement")[:8] + ("knock_in",)),
    dict(probability=[0.125, 0.125, 0.0625, 0.0625, -0.01, 0.125, 0.0625, 0.25, 0.1875]),
    dict(times=[0.25, 0.5, 0.5, 0.75, 0.75, 0.75, 1.0, 1.0, 0.9]),
    dict(event=("ko_pre", "ko_pre", "ko_pre", "ko_pre", "ko_post", "ki_settlement", "ko_pre", "maturity",
                "ki_settlement")),                                           # duplicate (time, event)
    dict(survival=[0.875, 0.6875, 0.7, 0.0]),
    dict(survival_times=[0.25, 0.5, 0.5, 1.0]),
    dict(survival=[0.875, 0.6875, 0.4375]),
    dict(payment_times=[0.25, 0.5, 0.5, 0.75, 0.75, 0.75, 1.0, 1.0, 0.95]),
])
def test_termination_distribution_rejects(kw):
    base = dict(
        times=[0.25, 0.5, 0.5, 0.75, 0.75, 0.75, 1.0, 1.0, 1.0],
        payment_times=[0.25, 0.5, 0.5, 0.75, 0.75, 0.75, 1.0, 1.0, 1.0],
        event=("ko_pre", "ko_pre", "ko_post", "ko_pre", "ko_post", "ki_settlement", "ko_pre", "maturity",
               "ki_settlement"),
        probability=[0.125, 0.125, 0.0625, 0.0625, 0.0, 0.125, 0.0625, 0.25, 0.1875],
        discounted_cash=[0.0] * 9, cash_to_pay=[0.0] * 9,
        survival_times=[0.25, 0.5, 0.75, 1.0], survival=[0.875, 0.6875, 0.5, 0.0],
    )
    base.update(kw)
    with pytest.raises(ValidationError):
        TerminationDistribution(**base)


@pytest.mark.parametrize("kw", [
    dict(kind=("ko_post", "ko_post", "maturity")),
    dict(probability=[0.0625, -0.1, 0.3125]),
    dict(offsets=[0.25, -0.5, 0.5]),
    dict(offsets=[0.25, 0.25, 0.5]),                                         # duplicate (offset, kind)
    dict(discounted_cash=[0.0625, 0.0]),
])
def test_post_ki_offset_table_rejects(kw):
    base = dict(offsets=[0.25, 0.5, 0.5], kind=("ko_post", "ko_post", "ki_settlement"),
                probability=[0.0625, 0.0, 0.3125], discounted_cash=[0.0625, 0.0, 0.15625],
                cash_to_pay=[0.0625, 0.0, 0.15625])
    base.update(kw)
    with pytest.raises(ValidationError):
        PostKIOffsetTable(**base)


# --------------------------------------------------------------------------- direct-cash ledger builder


def test_ledger_builder_rows_and_totals():
    obs, cpn = _observations(dates=_dates([0.25, 0.5, 0.75, 1.0]),
                             payment_dates=_dates([0.26, 0.51, 0.76, 1.01])), _coupons(
        payment_dates=_dates([0.26, 0.51, 0.76, 1.01]))
    mat = _row(0.3125, 0.3125, date=_dates([1.0])[0], payment_date=_dates([1.01])[0])
    ki = _row(0.25, 0.125, date=_dates([1.0])[0], payment_date=_dates([1.01])[0])
    rec = ReceivablesTable(source=("coupon",), decision_times=[-0.01], decision_dates=(D0 - timedelta(days=4),),
                           payment_times=[0.02], payment_dates=(D0 + timedelta(days=7),),
                           discounted_cash=[0.5], cash_to_pay=[0.51])
    ledger = build_cash_ledger(observations=obs, coupons=cpn, maturity=mat, ki_settlement=ki, receivables=rec)
    # 3 KO rows (the lock-up row is coupon-only), 4 coupon rows, maturity, KI settlement, 1 receivable
    assert ledger.size == 3 + 4 + 1 + 1 + 1
    assert ledger.total_discounted_cash == pytest.approx(PHOENIX_CASH + 0.5, abs=1e-15)
    assert ledger.total_cash_to_pay == pytest.approx(PHOENIX_CASH + 0.51, abs=1e-15)
    assert ledger.determination_dates is not None and len(ledger.determination_dates) == ledger.size
    assert ledger.determination_times[-1] == -0.01 and ledger.payment_times[-1] == 0.02
    stats = _phoenix(observations=obs, coupons=cpn, maturity=mat, ki_settlement=ki, receivables=rec, ledger=ledger)
    assert stats.total_discounted_cash == pytest.approx(ledger.total_discounted_cash, abs=1e-15)


def test_ledger_builder_drops_dates_unless_every_row_has_them():
    ledger = _ledger_for(_observations(dates=_dates([0.25, 0.5, 0.75, 1.0]),
                                       payment_dates=_dates([0.26, 0.51, 0.76, 1.01])),
                         _coupons(), _row(0.3125, 0.3125), _row(0.25, 0.125), ReceivablesTable.empty())
    assert ledger.determination_dates is None and ledger.payment_dates is None


def test_ledger_builder_takes_path_dependent_cash_at_actual_payment_events():
    # EXPIRY coupons: earned per row, paid at the KO settlement or maturity that ends the note
    cpn = _coupons(payment_times=[1.01] * 4, payment_is_path_dependent=[True] * 4)
    with pytest.raises(ValidationError, match="path-dependent"):
        _ledger_for(_observations(), cpn, _row(0.3125, 0.3125), _row(0.25, 0.125), ReceivablesTable.empty())
    paid = CashLedger(determination_times=[0.5, 0.75, 1.0, 1.0], payment_times=[0.51, 0.76, 1.01, 1.01],
                      discounted_cash=[0.03125, 0.015625, 0.03125, 0.015625],
                      cash_to_pay=[0.032, 0.016, 0.032, 0.016])
    ledger = build_cash_ledger(observations=_observations(), coupons=cpn, maturity=_row(0.3125, 0.3125),
                               ki_settlement=_row(0.25, 0.125), receivables=ReceivablesTable.empty(),
                               path_dependent_cash=paid)
    assert ledger.size == 3 + 4 + 2
    assert ledger.total_cash_to_pay == pytest.approx(0.4375 + 0.096 + 0.3125 + 0.125, abs=1e-15)
    stats = _phoenix(coupons=cpn, ledger=ledger)
    assert stats.reconciliation_error == pytest.approx(1.0 - PHOENIX_CASH, abs=1e-15)
    with pytest.raises(ValidationError, match="path-dependent"):
        build_cash_ledger(observations=_observations(), coupons=_coupons(), maturity=_row(0.3125, 0.3125),
                          ki_settlement=_row(0.25, 0.125), receivables=ReceivablesTable.empty(),
                          path_dependent_cash=paid)


def test_pruned_ledger_cash_to_pay_d17():
    cpn = _coupons(payment_times=[1.01] * 4, payment_is_path_dependent=[True] * 4)
    paid = CashLedger(determination_times=[1.0], payment_times=[1.01], discounted_cash=[0.09375], cash_to_pay=None)
    ledger = build_cash_ledger(observations=_observations(), coupons=cpn, maturity=_row(0.3125, 0.3125),
                               ki_settlement=_row(0.25, 0.125), receivables=ReceivablesTable.empty(),
                               path_dependent_cash=paid)
    assert ledger.cash_to_pay is None and ledger.total_cash_to_pay is None
    with pytest.raises(ValidationError, match="unavailable"):
        _phoenix(coupons=cpn, ledger=ledger)
    stats = _phoenix(coupons=cpn, ledger=ledger, unavailable={"ledger.cash_to_pay": "not requested"})
    assert stats.total_cash_to_pay == pytest.approx(PHOENIX_CASH, abs=1e-15)   # from the tables


def test_rebased_ledger_from_termination_and_receivables():
    stats = _rebased(receivables=_receivables(0.25), pv=1.25)
    assert stats.ledger.size == 9 + 1
    assert stats.ledger.total_discounted_cash == pytest.approx(stats.total_discounted_cash, abs=1e-15)


@pytest.mark.parametrize("kw", [
    dict(determination_times=[1.0, 0.5], payment_times=[1.0], discounted_cash=[1.0, 1.0], cash_to_pay=[1.0, 1.0]),
    dict(determination_times=[1.0], payment_times=[0.9], discounted_cash=[1.0], cash_to_pay=[1.0]),
    dict(determination_times=[1.0], payment_times=[1.0], discounted_cash=[np.inf], cash_to_pay=[1.0]),
    dict(determination_times=[1.0], payment_times=[1.0], discounted_cash=[1.0], cash_to_pay=[1.0],
         payment_dates=(D0, D0)),
])
def test_cash_ledger_rejects(kw):
    with pytest.raises(ValidationError):
        CashLedger(**kw)
