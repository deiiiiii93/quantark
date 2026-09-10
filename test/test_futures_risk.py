"""Units, signs and held-book algebra of the currency futures risk records.

The reference numbers come from ``test/bucket_hedge_fixtures.py``, which
derives them from the revised design with the standard library only.
"""

from __future__ import annotations

import math
from datetime import datetime

import pytest

from bucket_hedge_fixtures import (
    LINEAR_QUOTES,
    LINEAR_SPOT,
    AnalyticQuote,
    linear_book_risk,
)
from quantark.backtest.futures_risk import (
    CarryRiskSettings,
    FuturesBookRisk,
    FuturesBucket,
    bucket_contract_equivalent,
    gross_contractual_notional,
    gross_of,
    held_book_risk,
    parallel_of,
    rhoq_bp_per_1pct,
    spot_1pct_bp,
    spot_delta_hands,
)
from quantark.util.exceptions import ValidationError

EXPIRIES = (datetime(2025, 3, 21), datetime(2025, 6, 20))


def make_linear_risk(*, spot: float = LINEAR_SPOT) -> FuturesBookRisk:
    """``V = S + F_1 - 0.5 F_2`` as a production risk record."""
    reference = linear_book_risk(spot=spot)
    buckets = tuple(
        FuturesBucket(
            contract=quote.contract,
            expiry_date=expiry,
            tenor_years=quote.tenor_years,
            price=quote.price,
            multiplier=quote.multiplier,
            bucket_currency=bucket,
        )
        for quote, expiry, bucket in zip(LINEAR_QUOTES, EXPIRIES, reference.buckets)
    )
    return FuturesBookRisk(spot=spot, delta_q=reference.delta_q, buckets=buckets)


# ---------------------------------------------------------------------------
# The design's section 3.3 linear counterexample
# ---------------------------------------------------------------------------


def test_linear_reference_reproduces_the_design_numbers():
    reference = linear_book_risk()
    assert reference.delta_q == pytest.approx(1.5)
    assert reference.delta_f == pytest.approx(1.0)
    assert reference.buckets == pytest.approx((1.0, -0.5))
    assert reference.nodal_rhoq == pytest.approx((-25.0, 25.0))
    # The unhedged book already has zero PARALLEL rhoq while carrying large
    # opposing nodal risks: section 1's warning, in one assertion.
    assert reference.parallel_rhoq == pytest.approx(0.0)
    assert reference.gross_nodal_rhoq == pytest.approx(50.0)


def test_record_reproduces_product_nodal_rhoq_and_pinned_delta():
    risk = make_linear_risk()
    assert risk.nodal_rhoq == pytest.approx(
        {"IF2503": -25.0, "IF2506": 25.0}
    )
    assert risk.parallel_rhoq == pytest.approx(0.0)
    assert risk.gross_nodal_rhoq == pytest.approx(50.0)
    assert risk.delta_f_derived == pytest.approx(1.0)


def test_spot_parallel_holdings_zero_spot_and_parallel_with_opposing_nodes():
    risk = make_linear_risk()
    delta, rho = held_book_risk(risk, {"IF2503": -3.0, "IF2506": 1.5})
    assert delta == pytest.approx(0.0, abs=1e-12)
    assert rho["IF2503"] == pytest.approx(50.0)
    assert rho["IF2506"] == pytest.approx(-50.0)
    assert parallel_of(rho) == pytest.approx(0.0, abs=1e-12)
    # K = D_F S T_a T_b / (T_b - T_a) = 1 * 100 * 0.25 * 0.5 / 0.25 = 50
    assert gross_of(rho) == pytest.approx(100.0)


def test_nodes_holdings_zero_every_node_and_leave_pinned_delta():
    risk = make_linear_risk()
    delta, rho = held_book_risk(risk, {"IF2503": -1.0, "IF2506": 0.5})
    assert delta == pytest.approx(risk.delta_f_derived)
    assert parallel_of(rho) == pytest.approx(0.0, abs=1e-12)
    assert gross_of(rho) == pytest.approx(0.0, abs=1e-12)


def test_spot_far_holdings_zero_delta_and_leave_the_far_node():
    risk = make_linear_risk()
    delta, rho = held_book_risk(risk, {"IF2503": -1.0, "IF2506": -0.5})
    assert delta == pytest.approx(0.0, abs=1e-12)
    assert rho["IF2503"] == pytest.approx(0.0, abs=1e-12)
    # D_F * S * T_n = 1 * 100 * 0.5
    assert rho["IF2506"] == pytest.approx(50.0)


def test_unheld_nodes_still_report_the_product_rhoq():
    risk = make_linear_risk()
    delta, rho = held_book_risk(risk, {})
    assert delta == pytest.approx(risk.delta_q)
    assert rho == pytest.approx({"IF2503": -25.0, "IF2506": 25.0})


# ---------------------------------------------------------------------------
# Multipliers, signs and quantities
# ---------------------------------------------------------------------------


def test_distinct_multipliers_scale_hedge_terms_but_not_product_terms():
    quotes = (
        AnalyticQuote("IC2503", 0.25, 6000.0, 200.0),
        AnalyticQuote("IC2506", 0.50, 5900.0, 300.0),
    )
    reference = linear_book_risk(
        spot=6100.0, quotes=quotes, futures_coefficients=(2.0, -3.0)
    )
    buckets = tuple(
        FuturesBucket(
            contract=q.contract,
            expiry_date=None,
            tenor_years=q.tenor_years,
            price=q.price,
            multiplier=q.multiplier,
            bucket_currency=b,
        )
        for q, b in zip(quotes, reference.buckets)
    )
    risk = FuturesBookRisk(spot=6100.0, delta_q=reference.delta_q, buckets=buckets)

    # Exact nodal neutrality needs -B_i / m_i, which now differs per contract.
    holdings = {"IC2503": -2.0 / 200.0, "IC2506": 3.0 / 300.0}
    delta, rho = held_book_risk(risk, holdings)
    assert parallel_of(rho) == pytest.approx(0.0, abs=1e-9)
    assert gross_of(rho) == pytest.approx(0.0, abs=1e-9)
    assert delta == pytest.approx(reference.delta_f, rel=1e-12)
    assert bucket_contract_equivalent(buckets[0]) == pytest.approx(2.0 / 200.0)
    assert buckets[1].contract_equivalent == pytest.approx(-3.0 / 300.0)


def test_negative_book_quantity_flips_every_currency_greek():
    long_side = linear_book_risk()
    short_side = linear_book_risk(
        futures_coefficients=tuple(-c for c in (1.0, -0.5)),
        spot_coefficient=-1.0,
    )
    assert short_side.delta_q == pytest.approx(-long_side.delta_q)
    assert short_side.nodal_rhoq == pytest.approx(
        tuple(-r for r in long_side.nodal_rhoq)
    )


def test_signed_holdings_are_applied_once():
    risk = make_linear_risk()
    single, _ = held_book_risk(risk, {"IF2503": 1.0})
    double, _ = held_book_risk(risk, {"IF2503": 2.0})
    hedge_leg = risk.bucket("IF2503")
    step = hedge_leg.multiplier * hedge_leg.price / risk.spot
    assert double - single == pytest.approx(step)


# ---------------------------------------------------------------------------
# Reporting notional and unit conversions
# ---------------------------------------------------------------------------


def test_rhoq_bp_per_one_percentage_point_is_not_a_relative_yield_change():
    # 50,000 CNY per unit yield on 50,000,000 CNY of notional is 0.1 bp per
    # +1 percentage point of q, not 10,000 bp and not 0.001 bp.
    assert rhoq_bp_per_1pct(50_000.0, 50_000_000.0) == pytest.approx(0.1)
    assert rhoq_bp_per_1pct(-50_000.0, 50_000_000.0) == pytest.approx(-0.1)


def test_spot_shock_and_hand_conversions():
    # A 1% spot shock on delta 5,000 CNY/point at spot 4,000 moves
    # 200,000 CNY, which is 40 bp of a 50m notional.
    assert spot_1pct_bp(5_000.0, 4_000.0, 50_000_000.0) == pytest.approx(40.0)
    assert spot_delta_hands(5_000.0, 200.0) == pytest.approx(25.0)


def test_zero_signed_value_book_still_has_a_positive_reporting_notional():
    # Long and short the same product: the signed book value is zero but the
    # gross contractual notional that scales the report is not.
    positions = ((1.0, 4_700.0, 200.0), (-1.0, 4_700.0, 200.0))
    assert gross_contractual_notional(positions) == pytest.approx(2 * 4_700.0 * 200.0)


def test_reporting_notional_rejects_an_empty_or_zero_book():
    with pytest.raises(ValidationError):
        gross_contractual_notional(())
    with pytest.raises(ValidationError):
        gross_contractual_notional(((0.0, 4_700.0, 200.0),))
    with pytest.raises(ValidationError):
        rhoq_bp_per_1pct(1.0, 0.0)
    with pytest.raises(ValidationError):
        rhoq_bp_per_1pct(1.0, float("nan"))


# ---------------------------------------------------------------------------
# Record validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field, value",
    [
        ("contract", ""),
        ("tenor_years", 0.0),
        ("tenor_years", -0.25),
        ("price", 0.0),
        ("multiplier", 0.0),
        ("multiplier", -200.0),
        ("tenor_years", float("nan")),
        ("price", float("inf")),
        ("bucket_currency", float("nan")),
    ],
)
def test_bucket_rejects_invalid_coordinates(field, value):
    kwargs = dict(
        contract="IF2503",
        expiry_date=None,
        tenor_years=0.25,
        price=100.0,
        multiplier=1.0,
        bucket_currency=1.0,
    )
    kwargs[field] = value
    with pytest.raises(ValidationError):
        FuturesBucket(**kwargs)


def test_bucket_accepts_a_missing_expiry_in_a_mathematical_fixture():
    bucket = FuturesBucket(
        contract="IF2503",
        expiry_date=None,
        tenor_years=0.25,
        price=100.0,
        multiplier=1.0,
        bucket_currency=-3.5,
    )
    assert bucket.expiry_date is None
    assert bucket.nodal_rhoq == pytest.approx(3.5 * 100.0 * 0.25)


def test_book_rejects_duplicate_contracts_and_unsorted_tenors():
    def bucket(contract, tenor):
        return FuturesBucket(
            contract=contract,
            expiry_date=None,
            tenor_years=tenor,
            price=100.0,
            multiplier=1.0,
            bucket_currency=0.0,
        )

    with pytest.raises(ValidationError):
        FuturesBookRisk(spot=100.0, delta_q=0.0, buckets=(bucket("A", 0.25), bucket("A", 0.5)))
    with pytest.raises(ValidationError):
        FuturesBookRisk(spot=100.0, delta_q=0.0, buckets=(bucket("A", 0.5), bucket("B", 0.25)))
    with pytest.raises(ValidationError):
        FuturesBookRisk(spot=100.0, delta_q=0.0, buckets=(bucket("A", 0.25), bucket("B", 0.25)))


def test_book_rejects_invalid_spot_or_delta_and_an_empty_node_set():
    good = make_linear_risk().buckets
    with pytest.raises(ValidationError):
        FuturesBookRisk(spot=0.0, delta_q=1.0, buckets=good)
    with pytest.raises(ValidationError):
        FuturesBookRisk(spot=100.0, delta_q=float("nan"), buckets=good)
    with pytest.raises(ValidationError):
        FuturesBookRisk(spot=100.0, delta_q=1.0, buckets=())


def test_book_snapshots_the_bucket_sequence():
    buckets = list(make_linear_risk().buckets)
    risk = FuturesBookRisk(spot=100.0, delta_q=1.5, buckets=buckets)
    buckets.clear()
    assert len(risk.buckets) == 2
    assert isinstance(risk.buckets, tuple)


def test_holdings_outside_the_risk_coordinates_must_be_flat():
    risk = make_linear_risk()
    delta, rho = held_book_risk(risk, {"IF2509": 0.0})
    assert "IF2509" not in rho
    assert delta == pytest.approx(risk.delta_q)
    with pytest.raises(ValidationError):
        held_book_risk(risk, {"IF2509": 1.0})


def test_holdings_must_be_finite():
    risk = make_linear_risk()
    with pytest.raises(ValidationError):
        held_book_risk(risk, {"IF2503": float("nan")})
    with pytest.raises(ValidationError):
        held_book_risk(risk, {"IF2503": float("inf")})


def test_unknown_bucket_lookup_fails_closed():
    with pytest.raises(ValidationError):
        make_linear_risk().bucket("IF2509")


# ---------------------------------------------------------------------------
# CarryRiskSettings
# ---------------------------------------------------------------------------


def test_settings_defaults_match_the_plan_table():
    settings = CarryRiskSettings()
    assert settings.reference_notional is None
    assert settings.reference_multiplier == 200.0
    assert settings.futures_bump_points == 1.0
    assert settings.audit_spot_bump_rel is None
    assert settings.audit_yield_bump == 1e-4
    assert settings.delta_tolerance_hands == 0.01
    assert settings.rhoq_tolerance_bp == 0.01
    assert settings.stress_dates == ()
    assert settings.tail_shifts == (-0.01, 0.01)
    assert settings.shape_shifts == (-0.01, 0.01)
    assert not settings.is_resolved


def test_settings_resolution_records_the_effective_values():
    resolved = CarryRiskSettings().resolved(
        reference_notional=50_000_000.0, audit_spot_bump_rel=0.005
    )
    assert resolved.reference_notional == 50_000_000.0
    assert resolved.audit_spot_bump_rel == 0.005
    assert resolved.is_resolved
    # Resolution never mutates the input settings object.
    assert CarryRiskSettings().reference_notional is None
    resolved.require_resolved()


def test_unresolved_settings_cannot_be_used_for_reporting():
    with pytest.raises(ValidationError):
        CarryRiskSettings().require_resolved()
    with pytest.raises(ValidationError):
        CarryRiskSettings(reference_notional=50_000_000.0).require_resolved()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reference_multiplier": 0.0},
        {"reference_multiplier": -200.0},
        {"futures_bump_points": 0.0},
        {"audit_yield_bump": 0.0},
        {"audit_yield_bump": -1e-4},
        {"audit_spot_bump_rel": 0.0},
        {"delta_tolerance_hands": -0.01},
        {"rhoq_tolerance_bp": -0.01},
        {"reference_notional": 0.0},
        {"reference_notional": float("nan")},
        {"tail_shifts": (float("nan"),)},
        {"shape_shifts": (float("inf"),)},
        {"futures_bump_points": float("nan")},
    ],
)
def test_settings_reject_invalid_numbers(kwargs):
    with pytest.raises(ValidationError):
        CarryRiskSettings(**kwargs)


def test_settings_normalise_schedules():
    stamps = (datetime(2025, 3, 5), datetime(2025, 3, 3), datetime(2025, 3, 5))
    settings = CarryRiskSettings(stress_dates=stamps)
    assert settings.stress_dates == (datetime(2025, 3, 3), datetime(2025, 3, 5))
    assert isinstance(settings.tail_shifts, tuple)


def test_zero_tolerances_are_allowed_but_a_zero_bump_is_not():
    exact = CarryRiskSettings(delta_tolerance_hands=0.0, rhoq_tolerance_bp=0.0)
    assert exact.delta_tolerance_hands == 0.0
    with pytest.raises(ValidationError):
        CarryRiskSettings(futures_bump_points=-1.0)


def test_settings_are_hashable_and_frozen():
    settings = CarryRiskSettings()
    assert hash(settings) == hash(CarryRiskSettings())
    with pytest.raises(Exception):
        settings.reference_multiplier = 300.0  # type: ignore[misc]


def test_gross_and_parallel_aggregations_differ_on_cancelling_nodes():
    rho = {"a": 50.0, "b": -50.0}
    assert parallel_of(rho) == pytest.approx(0.0)
    assert gross_of(rho) == pytest.approx(100.0)
    assert math.isclose(gross_of({}), 0.0)
    assert math.isclose(parallel_of({}), 0.0)
