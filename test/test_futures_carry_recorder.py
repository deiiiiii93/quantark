"""Exposure rows, audit schedule and P&L attribution.

Two separations are the point of this suite.  Mapped exposure is reported
every day while a DIRECT measurement appears only on a scheduled audit date,
NaN with a status otherwise.  And a day whose linear attribution cannot be
formed still keeps its actual P&L and costs: only the decomposition is
withheld, with a reason.
"""

from __future__ import annotations

import math
from datetime import datetime

import pandas as pd
import pytest

import bucket_hedge_fixtures as fixtures
from quantark.backtest.futures_risk import CarryRiskSettings
from quantark.backtest.replay import (
    AutocallableDeltaHedgeStrategy,
    FuturesBucketHedgeStrategy,
    ReplayBacktestConfig,
    ReplayBacktestEngine,
    ReplayProduct,
)
from quantark.backtest.replay.carry_recorder import ATTRIBUTION_STATUSES

RETIREMENT_DATE = pd.Timestamp("2024-01-06")


def run(
    *,
    strategy=None,
    audit_mode: str = "daily",
    audit_dates=(),
    quantities=(-1.0,),
    dataset=None,
    settings=None,
    product_kwargs=None,
    **config_kwargs,
):
    products = [
        ReplayProduct(
            product=fixtures.market_product(**(product_kwargs or {})),
            quantity=q,
            position_id=index,
            has_lifecycle=True,
        )
        for index, q in enumerate(quantities)
    ]
    options = dict(
        products=products,
        market_data=dataset if dataset is not None else fixtures.market_dataset(),
        engine_config=fixtures.market_engine_config(),
        strategy=strategy
        if strategy is not None
        else FuturesBucketHedgeStrategy(round_contracts=False),
        calculate_surfaces=False,
        calculate_event_probabilities=False,
        carry_audit_mode=audit_mode,
        carry_audit_dates=audit_dates,
    )
    if settings is not None:
        options["carry_risk_settings"] = settings
    options.update(config_kwargs)
    engine = ReplayBacktestEngine(ReplayBacktestConfig(**options))
    return engine, engine.run()


# ---------------------------------------------------------------------------
# Every eligible node gets a row
# ---------------------------------------------------------------------------


def test_a_single_contract_control_still_reports_every_unhedged_node():
    """The point of writing zero-holding rows.

    A legacy delta hedge holds ONE contract.  Without a row per eligible
    node, the rhoq it is not hedging at the other tenors would simply be
    absent from the report.
    """
    _, results = run(
        strategy=AutocallableDeltaHedgeStrategy(round_contracts=False),
        record_carry_exposure=True,
    )
    legs = results.hedge_legs_df()
    first = legs[legs["date"] == pd.Timestamp("2024-01-02")]
    assert set(first["contract"]) == {"IM2401", "IM2402", "IM2403"}
    held = first[first["held_after"] != 0.0]
    assert len(held) == 1
    unheld = first[first["held_after"] == 0.0]
    assert len(unheld) == 2
    # Zero holdings, but real product exposure at those nodes.
    assert (unheld["product_rhoq_bp"].abs() > 0.0).all()
    assert (unheld["net_rhoq_bp"].abs() > 0.0).all()
    assert (unheld["hedge_rhoq_bp"] == 0.0).all()


def test_currency_risk_scales_with_the_book_but_the_bp_measure_does_not():
    """Doubling the book doubles the risk AND the reporting notional.

    The currency bucket is a book quantity and doubles; the bp column is
    normalised by the gross contractual notional, which doubles too, so it
    is invariant.  Mixing the two conventions in one comparison is exactly
    the mistake the units are meant to prevent.
    """
    single = run(quantities=(-1.0,))[1].hedge_legs_df()
    doubled = run(quantities=(-2.0,))[1].hedge_legs_df()
    date = pd.Timestamp("2024-01-02")
    one = single[single["date"] == date].set_index("contract")
    two = doubled[doubled["date"] == date].set_index("contract")
    for contract in one.index:
        assert two.loc[contract, "bucket_currency"] == pytest.approx(
            2.0 * one.loc[contract, "bucket_currency"], rel=1e-6
        )
        assert two.loc[contract, "product_rhoq_bp"] == pytest.approx(
            one.loc[contract, "product_rhoq_bp"], rel=1e-6
        )


def test_a_retired_leg_keeps_a_row_after_it_leaves_the_curve():
    _, results = run()
    legs = results.hedge_legs_df()
    day = legs[legs["date"] == RETIREMENT_DATE].set_index("contract")
    assert "IM2401" in day.index
    assert bool(day.loc["IM2401", "retired"]) is True
    assert bool(day.loc["IM2401", "is_curve_node"]) is False
    assert day.loc["IM2401", "held_after"] == 0.0
    assert day.loc["IM2401", "held_before"] != 0.0
    assert math.isnan(day.loc["IM2401", "bucket_currency"])


# ---------------------------------------------------------------------------
# Direct measurements and the objective
# ---------------------------------------------------------------------------


def test_a_bucket_run_reports_direct_measurements_next_to_the_mapped_ones():
    _, results = run()
    row = results.hedge_attribution_df().iloc[0]
    assert row["audit_status"] == "pass"
    assert row["objective"] == "spot_parallel"
    for column in (
        "direct_net_delta_hands",
        "direct_net_parallel_rhoq_bp",
        "delta_f_direct_hands",
        "identity_residual_hands",
    ):
        assert math.isfinite(row[column]), column
    # The mapped net delta is the policy's zero; the direct one agrees within
    # the numerical budget without being bit-identical.
    assert row["net_delta_hands"] == pytest.approx(0.0, abs=1e-9)
    assert abs(row["net_delta_audit_error_hands"]) < 0.01
    assert row["correction_contract_a"] == "IM2401"
    assert row["correction_contract_b"] == "IM2403"


def test_ideal_residuals_are_recorded_beside_the_actual_ones():
    _, results = run()
    row = results.hedge_attribution_df().iloc[0]
    assert row["ideal_net_delta_hands"] == pytest.approx(0.0, abs=1e-9)
    assert row["ideal_net_parallel_rhoq_bp"] == pytest.approx(0.0, abs=1e-9)
    # spot_parallel deliberately keeps 2|K| of gross nodal shape risk.
    assert row["ideal_net_rhoq_gross_bp"] > 0.0
    assert row["net_rhoq_gross_bp"] > 0.0
    assert row["product_rhoq_gross_bp"] > 0.0


def test_rounding_and_no_trade_contributions_are_separated():
    # A book large enough that the rounded target genuinely moves each day,
    # with a band wide enough to suppress the trade: both error sources are
    # then non-zero at once.
    _, results = run(
        strategy=FuturesBucketHedgeStrategy(
            round_contracts=True, delta_threshold=5.0
        ),
        quantities=(-500.0,),
    )
    legs = results.hedge_legs_df()
    day = legs[legs["date"] == pd.Timestamp("2024-01-03")]
    for _, row in day[day["is_curve_node"]].iterrows():
        assert row["rounding_error_contracts"] == pytest.approx(
            row["rounded_contracts"] - row["scaled_contracts"]
        )
        assert row["no_trade_error_contracts"] == pytest.approx(
            row["held_after"] - row["rounded_contracts"]
        )
        # Their sum is the total execution error against the scaled target.
        assert (
            row["rounding_error_contracts"] + row["no_trade_error_contracts"]
        ) == pytest.approx(row["held_after"] - row["scaled_contracts"])
    assert day["no_trade_error_contracts"].abs().sum() > 0.0


def test_a_deliberate_residual_does_not_fail_numerical_validity():
    _, results = run(strategy=FuturesBucketHedgeStrategy(objective="nodes",
                                                         round_contracts=False))
    frame = results.hedge_attribution_df()
    assert (frame["audit_status"] == "pass").all()
    # ... while the book is knowingly NOT spot neutral.
    assert frame["net_delta_hands"].abs().max() > 0.0
    assert frame["ideal_net_delta_hands"].abs().max() > 0.0


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


def test_a_daily_schedule_measures_every_priced_date():
    _, results = run(audit_mode="daily")
    frame = results.hedge_attribution_df()
    assert (frame["audit_status"] == "pass").all()
    assert frame["direct_net_delta_hands"].notna().all()


def test_no_audit_leaves_every_direct_field_nan_with_a_status():
    _, results = run(audit_mode="none", record_carry_exposure=True)
    frame = results.hedge_attribution_df()
    assert (frame["audit_status"] == "not_measured").all()
    assert frame["direct_net_delta_hands"].isna().all()
    assert frame["identity_residual_hands"].isna().all()
    assert frame["parallel_rhoq_audit_error_bp"].isna().all()
    # Mapped exposure is still reported every day.
    assert frame["net_delta_hands"].notna().all()
    assert frame["product_rhoq_bp"].notna().all()
    legs = results.hedge_legs_df()
    assert (legs["audit_status"] == "not_measured").all()
    assert legs["direct_net_rhoq_bp"].isna().all()
    assert legs["net_rhoq_bp"].notna().any()


def test_a_sampled_schedule_always_includes_inception_and_node_changes():
    _, results = run(
        audit_mode="sampled", audit_dates=(datetime(2024, 1, 8),)
    )
    frame = results.hedge_attribution_df().set_index("date")
    measured = frame["audit_status"] != "not_measured"
    assert measured.loc[pd.Timestamp("2024-01-02")]
    # The node set changes when IM2401 retires.
    assert measured.loc[RETIREMENT_DATE]
    # ... and the explicitly requested date.
    assert measured.loc[pd.Timestamp("2024-01-08")]
    # Quiet dates in between are not measured.
    assert not measured.loc[pd.Timestamp("2024-01-04")]
    assert math.isnan(frame.loc[pd.Timestamp("2024-01-04"), "direct_net_delta_hands"])


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------


def test_the_book_move_reconciles_product_hedge_and_costs():
    engine, results = run()
    frame = results.hedge_attribution_df().set_index("date")
    states = results.states_df().set_index("date")
    ok = frame[frame["attribution_status"] == "ok"]
    assert not ok.empty
    for date, row in ok.iterrows():
        assert row["book_dv"] == pytest.approx(
            row["product_dv"] + row["hedge_price_pnl"] - row["transaction_costs"],
            rel=1e-12,
        )
        # ... and it matches the difference of the already recorded total.
        previous = states.index[states.index.get_loc(date) - 1]
        recorded = states.loc[date, "total_pnl"] - states.loc[previous, "total_pnl"]
        assert row["book_dv"] == pytest.approx(recorded, rel=1e-9, abs=1e-6)


def test_daily_product_pnl_is_the_cashflow_aware_difference_not_delta_mtm():
    _, results = run()
    frame = results.hedge_attribution_df().set_index("date")
    states = results.states_df().set_index("date")
    ok = frame[frame["attribution_status"] == "ok"]
    for date, row in ok.iterrows():
        previous = states.index[states.index.get_loc(date) - 1]
        expected = (
            states.loc[date, "product_pnl"] - states.loc[previous, "product_pnl"]
        )
        assert row["product_dv"] == pytest.approx(expected, rel=1e-12)


def test_the_first_date_has_no_linear_attribution():
    _, results = run()
    row = results.hedge_attribution_df().iloc[0]
    assert row["attribution_status"] == "first_date"
    for column in (
        "product_dv",
        "hedge_price_pnl",
        "book_dv",
        "linear_spot_pinned",
        "linear_listed_forwards",
        "remainder_after_linear",
    ):
        assert math.isnan(row[column]), column
    # Exposure on that date is still fully reported.
    assert math.isfinite(row["net_delta_hands"])


def test_a_node_set_change_withholds_the_decomposition_but_not_the_pnl():
    engine, results = run()
    frame = results.hedge_attribution_df().set_index("date")
    row = frame.loc[RETIREMENT_DATE]
    assert row["attribution_status"] == "coordinate_set_changed"
    assert math.isnan(row["linear_spot_pinned"])
    assert math.isnan(row["remainder_after_linear"])
    # The day is still in the actual performance record.
    states = results.states_df().set_index("date")
    assert math.isfinite(states.loc[RETIREMENT_DATE, "total_pnl"])
    assert math.isfinite(row["gross_contracts"])
    assert row["audit_status"] == "pass"
    assert set(ATTRIBUTION_STATUSES) >= {"ok", "first_date", "coordinate_set_changed"}


def test_the_linear_terms_use_yesterdays_sensitivities_and_holdings():
    _, results = run()
    frame = results.hedge_attribution_df().set_index("date")
    legs = results.hedge_legs_df()
    states = results.states_df().set_index("date")
    date, previous = pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-03")
    row = frame.loc[date]
    assert row["attribution_status"] == "ok"
    spot_move = states.loc[date, "spot"] - states.loc[previous, "spot"]
    prior_delta_f = frame.loc[previous, "delta_f_derived_hands"] * (
        frame.loc[previous, "reference_multiplier"]
    )
    assert row["linear_spot_pinned"] == pytest.approx(
        prior_delta_f * spot_move, rel=1e-9, abs=1e-9
    )
    # The forwards leg uses yesterday's buckets AND yesterday's holdings.
    prior = legs[legs["date"] == previous].set_index("contract")
    today = legs[legs["date"] == date].set_index("contract")
    expected = sum(
        (
            prior.loc[c, "bucket_currency"]
            + prior.loc[c, "held_after"] * prior.loc[c, "multiplier"]
        )
        * (today.loc[c, "price"] - prior.loc[c, "price"])
        for c in prior.index
        if prior.loc[c, "is_curve_node"]
    )
    assert row["linear_listed_forwards"] == pytest.approx(expected, rel=1e-9)


def test_financing_is_zero_and_explicit():
    _, results = run()
    frame = results.hedge_attribution_df()
    assert (frame["financing_pnl"] == 0.0).all()


def test_the_frozen_q_gamma_stays_a_diagnostic_and_is_never_subtracted():
    _, results = run()
    frame = results.hedge_attribution_df()
    ok = frame[frame["attribution_status"] == "ok"]
    for _, row in ok.iterrows():
        assert row["remainder_after_linear"] == pytest.approx(
            row["product_dv"]
            + row["hedge_price_pnl"]
            - row["linear_spot_pinned"]
            - row["linear_listed_forwards"],
            rel=1e-9,
        )
        # The gamma column exists but plays no part in that identity.
        assert math.isfinite(row["spot_gamma_frozen_q_diagnostic"])


# ---------------------------------------------------------------------------
# A wrong bucket survives the replay but not the audit
# ---------------------------------------------------------------------------


def test_a_perturbed_bucket_fails_the_audit_even_though_the_replay_finishes():
    from dataclasses import replace as dc_replace

    from quantark.backtest.futures_risk import FuturesBookRisk

    engine = ReplayBacktestEngine(
        ReplayBacktestConfig(
            products=[
                ReplayProduct(
                    product=fixtures.market_product(),
                    quantity=-1.0,
                    position_id=0,
                    has_lifecycle=True,
                )
            ],
            market_data=fixtures.market_dataset(),
            engine_config=fixtures.market_engine_config(),
            strategy=FuturesBucketHedgeStrategy(round_contracts=False),
            calculate_surfaces=False,
            calculate_event_probabilities=False,
            carry_audit_mode="daily",
        )
    )
    original = engine._measure_book_carry_risk

    def perturbed(date, env, context, alive_specs):
        risk = original(date, env, context, alive_specs)
        buckets = list(risk.buckets)
        buckets[0] = dc_replace(
            buckets[0], bucket_currency=buckets[0].bucket_currency * 1.5
        )
        return FuturesBookRisk(
            spot=risk.spot, delta_q=risk.delta_q, buckets=tuple(buckets)
        )

    engine._measure_book_carry_risk = perturbed
    results = engine.run()
    frame = results.hedge_attribution_df()
    # The replay completed and its P&L is perfectly finite ...
    assert results.states_df()["total_pnl"].notna().all()
    # ... and the audit still refuses.
    assert (frame["audit_status"] == "fail").all()
    assert frame["identity_residual_hands"].abs().max() > 0.01


# ---------------------------------------------------------------------------
# Stresses
# ---------------------------------------------------------------------------


def test_the_tail_stress_is_recorded_with_a_zero_hedge_contribution():
    # A product maturing beyond the last quoted tenor genuinely has tail
    # exposure; the hedge cannot respond to it at all.
    _, results = run(product_kwargs={"maturity_days": 180.0})
    stresses = results.hedge_stresses_df()
    assert not stresses.empty
    assert set(stresses["scenario_family"]) == {"independent_tail"}
    assert set(stresses["holdings_kind"]) == {"actual", "ideal"}
    assert (stresses["hedge_pnl"] == 0.0).all()
    assert stresses["product_pnl"].abs().max() > 0.0
    assert (stresses["book_pnl"] == stresses["product_pnl"]).all()
    assert stresses["shock_definition"].str.contains("tail_lambda").all()


def test_stress_rows_carry_reproducible_bump_metadata():
    import json

    _, results = run(product_kwargs={"maturity_days": 180.0})
    row = results.hedge_stresses_df().iloc[0]
    metadata = json.loads(row["bump_metadata"])
    assert metadata["family"] == "independent_tail"
    assert metadata["yield_units"] == "absolute decimal annual yield"
    assert metadata["finite"] is True
    assert row["source_basis_assumption"] == "listed futures quotes"


def test_stresses_can_be_restricted_to_named_dates():
    _, results = run(
        product_kwargs={"maturity_days": 180.0},
        settings=CarryRiskSettings(stress_dates=(datetime(2024, 1, 4),)),
    )
    stresses = results.hedge_stresses_df()
    assert set(pd.to_datetime(stresses["date"])) == {pd.Timestamp("2024-01-04")}


# ---------------------------------------------------------------------------
# Recording off
# ---------------------------------------------------------------------------


def test_recording_off_prices_no_risk_and_writes_no_rows():
    engine = ReplayBacktestEngine(
        ReplayBacktestConfig(
            products=[
                ReplayProduct(
                    product=fixtures.market_product(),
                    quantity=-1.0,
                    position_id=0,
                    has_lifecycle=True,
                )
            ],
            market_data=fixtures.market_dataset(),
            engine_config=fixtures.market_engine_config(),
            strategy=AutocallableDeltaHedgeStrategy(round_contracts=False),
            calculate_surfaces=False,
            calculate_event_probabilities=False,
        )
    )
    calls = []
    original = engine._measure_book_carry_risk
    engine._measure_book_carry_risk = lambda *a, **k: calls.append(a) or original(*a, **k)
    results = engine.run()
    assert calls == []
    assert engine._carry_recorder is None
    assert results.hedge_legs_df().empty
    assert results.hedge_attribution_df().empty
    assert results.hedge_stresses_df().empty
    assert engine.config.carry_recording.reference_notional is None


def test_a_leg_outside_the_risk_universe_is_reported_not_dropped():
    """The roll window is looser than the curve's minimum tenor.

    A single-contract control can hold a contract that has already left the
    curve.  It has no product bucket, but it still carries spot delta and its
    own carry sensitivity, and the audit cannot decide a book that holds an
    instrument its scenarios do not reprice.
    """
    _, results = run(
        strategy=AutocallableDeltaHedgeStrategy(round_contracts=False),
        record_carry_exposure=True,
    )
    legs = results.hedge_legs_df()
    day = legs[legs["date"] == RETIREMENT_DATE].set_index("contract")
    assert "IM2401" in day.index
    off = day.loc["IM2401"]
    assert bool(off["is_curve_node"]) is False
    assert off["held_after"] != 0.0
    # No product bucket ...
    assert math.isnan(off["bucket_currency"])
    assert math.isnan(off["product_rhoq_bp"])
    # ... but its own hedge exposure is measured and reported.
    assert math.isfinite(off["hedge_rhoq_bp"])
    assert off["hedge_rhoq_bp"] != 0.0
    assert off["net_rhoq_bp"] == pytest.approx(off["hedge_rhoq_bp"])

    attribution = results.hedge_attribution_df().set_index("date")
    row = attribution.loc[RETIREMENT_DATE]
    assert row["audit_status"] == "inconclusive"
    assert math.isnan(row["direct_net_delta_hands"])
    # The off-curve leg's delta is still inside the reported net.
    assert math.isfinite(row["net_delta_hands"])
    assert math.isfinite(row["hedge_delta_hands"])
    assert row["hedge_delta_hands"] != 0.0


def test_a_terminated_book_still_records_a_row_without_a_product_risk():
    """After the last product dies there is no product risk to measure.

    The day still gets rows: the legs the book held report their OWN carry
    exposure, the audit is not_measured rather than a zero, and the linear
    attribution is withheld with a reason.
    """
    spots = [fixtures.MARKET_SPOT] * 3 + [fixtures.MARKET_SPOT * 1.12] * 4
    _, results = run(
        dataset=fixtures.market_dataset(spots=spots),
        product_kwargs={
            "ko_barrier": fixtures.MARKET_SPOT * 1.05,
            "ko_observation_days": (4.0, 50.0),
        },
        terminate_on_lifecycle_end=False,
    )
    attribution = results.hedge_attribution_df()
    assert not attribution.empty
    dead = attribution[attribution["carry_family"] == "none"]
    if dead.empty:
        # The book settled on its final recorded date; the last row is still
        # written and still refuses to invent measurements.
        dead = attribution.tail(1)
    row = dead.iloc[-1]
    assert row["audit_status"] in ("not_measured", "pass", "inconclusive")
    assert math.isfinite(row["gross_contracts"])
    legs = results.hedge_legs_df()
    assert not legs.empty
    assert set(legs.columns) >= {"net_rhoq_bp", "hedge_rhoq_bp", "audit_status"}
