"""Bucket hedging is accepted only where its risk coordinates actually exist.

Every coordinate a bucket policy sizes against must be a tradable futures
contract with a quote, an expiry and a multiplier.  A flat carry channel has
no nodes; an option-implied forward is a different instrument from the future
the hedge trades.  Neither can inherit the policy formulas without a separate
quote Jacobian and cross-market basis reporting, so both are refused here
rather than silently repurposed.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from replay_golden import fixtures  # noqa: E402

from quantark.backtest.futures_risk import CarryRiskSettings  # noqa: E402
from quantark.backtest.replay import (  # noqa: E402
    AutocallableBacktestConfig,
    AutocallableEngineConfig,
    FuturesBucketHedgeStrategy,
    HedgeSpec,
    ProportionalFuturesDeltaHedgeStrategy,
    ReplayBacktestConfig,
    ReplayProduct,
)
from quantark.backtest.futures_ledger import FuturesRollPolicy  # noqa: E402
from quantark.util.exceptions import ValidationError  # noqa: E402


def engine_config(**kwargs) -> AutocallableEngineConfig:
    options = dict(
        dividend_source="futures_curve",
        futures_curve_extrapolation="flat_q",
        futures_curve_min_tenor_days=1,
    )
    options.update(kwargs)
    return AutocallableEngineConfig(**options)


def book_config(**kwargs) -> ReplayBacktestConfig:
    options = dict(
        products=[
            ReplayProduct(
                product=fixtures._snowball_product(),
                quantity=-1.0,
                position_id=0,
                has_lifecycle=True,
            )
        ],
        market_data=fixtures._market_data(),
        engine_config=engine_config(),
        strategy=FuturesBucketHedgeStrategy(),
        calculate_surfaces=False,
    )
    options.update(kwargs)
    return ReplayBacktestConfig(**options)


def single_config(**kwargs) -> AutocallableBacktestConfig:
    options = dict(
        product=fixtures._snowball_product(),
        market_data=fixtures._market_data(),
        engine_config=engine_config(),
        strategy=FuturesBucketHedgeStrategy(),
        product_quantity=-1.0,
        calculate_surfaces=False,
    )
    options.update(kwargs)
    return AutocallableBacktestConfig(**options)


# ---------------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("extrapolation", ["flat_q", "flat_forward_carry"])
def test_both_configs_accept_the_bucket_hedge_on_actual_quotes(extrapolation):
    for config in (
        book_config(engine_config=engine_config(
            futures_curve_extrapolation=extrapolation
        )),
        single_config(engine_config=engine_config(
            futures_curve_extrapolation=extrapolation
        )),
    ):
        plan = config.carry_recording
        assert plan.record is True
        assert plan.audit_mode == "none"
        assert plan.reference_notional is not None
        assert plan.reference_notional > 0.0


def test_a_bucket_strategy_turns_recording_on_by_itself():
    assert book_config(record_carry_exposure=False).carry_recording.record is True


def test_a_requested_audit_turns_recording_on_rather_than_disappearing():
    plan = book_config(
        strategy=None, record_carry_exposure=False, carry_audit_mode="daily"
    ).carry_recording
    assert plan.record is True
    assert plan.audit_mode == "daily"
    assert plan.as_metadata()["record_carry_exposure"] is True


def test_a_legacy_run_records_nothing_and_resolves_no_notional():
    plan = ReplayBacktestConfig(
        products=[
            ReplayProduct(
                product=fixtures._snowball_product(),
                quantity=-1.0,
                position_id=0,
                has_lifecycle=True,
            )
        ],
        market_data=fixtures._market_data(),
    ).carry_recording
    assert plan.record is False
    assert plan.audit_mode == "none"
    assert plan.reference_notional is None
    assert plan.settings is None


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_a_spot_hedge_cannot_carry_futures_buckets():
    with pytest.raises(ValidationError, match="futures instruments"):
        book_config(hedge=HedgeSpec(kind="spot"))


def test_a_flat_carry_source_has_no_nodes_to_hedge():
    with pytest.raises(ValidationError, match="actual futures_curve"):
        book_config(engine_config=AutocallableEngineConfig())
    with pytest.raises(ValidationError, match="actual futures_curve"):
        single_config(engine_config=AutocallableEngineConfig())


def test_an_option_implied_forward_is_not_the_future_being_traded():
    with pytest.raises(ValidationError, match="actual futures_curve"):
        book_config(
            engine_config=AutocallableEngineConfig(
                dividend_source="surface_forwards"
            ),
            calculate_surfaces=False,
        )
    with pytest.raises(ValidationError, match="external tail"):
        book_config(
            engine_config=engine_config(
                futures_curve_extrapolation="surface_forward_carry"
            )
        )


def test_the_source_restriction_is_tested_directly_not_through_the_helper():
    # uses_term_dividend_source() is True for surface_forwards too, so a check
    # written against it would have accepted an option-implied forward.
    surface = AutocallableEngineConfig(dividend_source="surface_forwards")
    assert surface.uses_term_dividend_source()
    assert surface.dividend_source != "futures_curve"
    with pytest.raises(ValidationError):
        book_config(engine_config=surface, calculate_surfaces=False)


def test_dividend_rolls_and_single_leg_explain_are_incompatible():
    # A term dividend source already refuses a roll policy one layer earlier,
    # so the combination cannot be built at all; the message comes from
    # whichever check runs first.
    with pytest.raises(ValidationError, match="dividend_roll_policy"):
        book_config(dividend_roll_policy=FuturesRollPolicy())
    with pytest.raises(ValidationError, match="single-leg explain"):
        book_config(pnl_explain=object())


def test_the_bucket_check_refuses_a_roll_policy_on_its_own():
    from quantark.backtest.replay.config import _validate_carry_hedge

    with pytest.raises(ValidationError, match="dividend rolls"):
        _validate_carry_hedge(
            FuturesBucketHedgeStrategy(),
            hedge_kind="futures",
            engine_config=engine_config(),
            dividend_roll_policy=FuturesRollPolicy(),
            pnl_explain=None,
            record_carry_exposure=False,
            carry_audit_mode="none",
            carry_audit_dates=(),
            carry_risk_settings=None,
            products=[],
        )


def test_invalid_audit_modes_and_settings_are_rejected():
    with pytest.raises(ValidationError, match="carry_audit_mode"):
        book_config(carry_audit_mode="weekly")
    with pytest.raises(ValidationError, match="CarryRiskSettings"):
        book_config(carry_risk_settings={"reference_notional": 1.0})
    with pytest.raises(ValidationError, match="date-like"):
        book_config(carry_audit_dates=("2025-03-03",))


def test_an_audited_book_needs_unique_position_ids_and_finite_quantities():
    product = fixtures._snowball_product()
    duplicated = [
        ReplayProduct(product=product, quantity=-1.0, position_id=0, has_lifecycle=True),
        ReplayProduct(product=product, quantity=1.0, position_id=0, has_lifecycle=True),
    ]
    with pytest.raises(ValidationError, match="unique position ids"):
        book_config(products=duplicated)

    infinite = [
        ReplayProduct(
            product=product,
            quantity=float("inf"),
            position_id=0,
            has_lifecycle=True,
        )
    ]
    with pytest.raises(ValidationError, match="finite signed quantities"):
        book_config(products=infinite)


def test_the_simulation_engine_rejects_the_bucket_strategy():
    from quantark.backtest.simulation.config import EnsembleConfig

    import inspect

    source = inspect.getsource(EnsembleConfig.__post_init__)
    assert "FuturesBucketHedgeStrategy" in source
    # The refusal is placed before the scalar sizing path it would break.
    assert source.index("FuturesBucketHedgeStrategy") < source.index(
        "dividend_source"
    )


# ---------------------------------------------------------------------------
# Audit schedule and notional
# ---------------------------------------------------------------------------


def test_audit_dates_are_normalised_unique_and_sorted():
    dates = (
        datetime(2025, 3, 5),
        datetime(2025, 3, 3),
        datetime(2025, 3, 5),
    )
    plan = book_config(carry_audit_mode="sampled", carry_audit_dates=dates).carry_recording
    assert plan.audit_dates == (datetime(2025, 3, 3), datetime(2025, 3, 5))
    assert plan.audits_on(datetime(2025, 3, 3))
    assert not plan.audits_on(datetime(2025, 3, 4))


def test_a_daily_schedule_audits_every_priced_date():
    plan = book_config(carry_audit_mode="daily").carry_recording
    assert plan.audits_on(datetime(2025, 3, 4))
    assert plan.audits_on(datetime(2099, 1, 1))
    none_plan = book_config(carry_audit_mode="none").carry_recording
    assert not none_plan.audits_on(datetime(2025, 3, 4))


def test_the_reference_notional_is_gross_contractual_and_resolved_once():
    product = fixtures._snowball_product()
    expected = abs(-1.0) * product.initial_price * product.contract_multiplier
    plan = book_config().carry_recording
    assert plan.reference_notional == pytest.approx(expected)
    assert plan.settings.reference_notional == pytest.approx(expected)

    # Long and short the same product: signed value cancels, gross does not.
    both_ways = book_config(
        products=[
            ReplayProduct(product=product, quantity=-1.0, position_id=0, has_lifecycle=True),
            ReplayProduct(product=product, quantity=1.0, position_id=1, has_lifecycle=True),
        ]
    ).carry_recording
    assert both_ways.reference_notional == pytest.approx(2 * expected)


def test_an_explicit_notional_wins_over_the_contractual_rule():
    plan = book_config(
        carry_risk_settings=CarryRiskSettings(reference_notional=12_345_678.0)
    ).carry_recording
    assert plan.reference_notional == pytest.approx(12_345_678.0)


def test_a_product_without_a_documented_notional_needs_an_explicit_one():
    class Undocumented:
        pass

    products = [
        ReplayProduct(
            product=Undocumented(), quantity=-1.0, position_id=0, has_lifecycle=False
        )
    ]
    with pytest.raises(ValidationError, match="reference notional"):
        book_config(products=products, strategy=None, carry_audit_mode="daily")
    # ... and supplying one makes the same book acceptable.
    ok = book_config(
        products=products,
        strategy=None,
        carry_audit_mode="daily",
        carry_risk_settings=CarryRiskSettings(reference_notional=1_000.0),
    )
    assert ok.carry_recording.reference_notional == 1_000.0


# ---------------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------------


def test_the_new_fields_are_appended_after_dividend_roll_policy():
    import dataclasses

    for config_class in (AutocallableBacktestConfig, ReplayBacktestConfig):
        names = [f.name for f in dataclasses.fields(config_class)]
        assert names.index("dividend_roll_policy") < names.index(
            "record_carry_exposure"
        )
        assert names[-4:] == [
            "record_carry_exposure",
            "carry_audit_mode",
            "carry_audit_dates",
            "carry_risk_settings",
        ]


def test_the_single_wrapper_forwards_every_new_setting_unchanged():
    from quantark.backtest.replay.single import AutocallableBacktestEngine

    settings = CarryRiskSettings(
        reference_notional=9_000_000.0, futures_bump_points=0.5
    )
    config = single_config(
        strategy=None,
        record_carry_exposure=True,
        carry_audit_mode="sampled",
        carry_audit_dates=(datetime(2025, 3, 3),),
        carry_risk_settings=settings,
    )
    engine = AutocallableBacktestEngine(config)
    inner = engine._book_config
    assert inner.record_carry_exposure is True
    assert inner.carry_audit_mode == "sampled"
    assert inner.carry_audit_dates == (datetime(2025, 3, 3),)
    assert inner.carry_risk_settings is settings
    assert inner.carry_recording.reference_notional == pytest.approx(9_000_000.0)
    assert inner.carry_recording.audit_dates == config.carry_recording.audit_dates


def test_the_proportional_control_needs_no_bucket_source():
    # It is a single-contract strategy, so it stays available on any source.
    config = ReplayBacktestConfig(
        products=[
            ReplayProduct(
                product=fixtures._snowball_product(),
                quantity=-1.0,
                position_id=0,
                has_lifecycle=True,
            )
        ],
        market_data=fixtures._market_data(),
        strategy=ProportionalFuturesDeltaHedgeStrategy(),
    )
    assert config.carry_recording.record is False
