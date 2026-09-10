"""The revised study's grid, strategy factory and static/replay parity.

The study registry is loaded through the same importlib pattern the existing
study tests use, because the stage scripts are numbered files rather than a
package.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
STUDY = REPO / "example" / "snowball_q_term_structure"


def _load(name: str, filename: str):
    if str(STUDY) not in sys.path:
        sys.path.insert(0, str(STUDY))
    spec = importlib.util.spec_from_file_location(name, STUDY / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


C = _load("_common", "_common.py")
FLEET = _load("fleet_stage", "02_backtest_fleet.py")

from quantark.backtest.strategy import (  # noqa: E402
    AutocallableDeltaHedgeStrategy,
    FuturesBucketHedgeStrategy,
    ProportionalFuturesDeltaHedgeStrategy,
)
from quantark.backtest.futures_ledger import FuturesRollPolicy  # noqa: E402


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------


def test_the_primary_grid_is_two_models_by_seven_policies():
    cells = C.PRIMARY_BUCKET_CELLS
    assert len(cells) == 14
    assert len(set(cells)) == 14
    assert {model for model, _ in cells} == {"term_flat_q", "term_flat_fwd"}
    assert {hedge for _, hedge in cells} == {
        "front",
        "far",
        "front_scaled",
        "far_scaled",
        "buckets_nodes",
        "buckets_far",
        "buckets_spot_parallel",
    }


def test_the_legacy_cells_and_their_identity_control_survive():
    assert ("flat_from_far", "front") in FLEET.DEFAULT_CELLS
    assert ("flat_from_hedge", "front") in FLEET.DEFAULT_CELLS
    # flat_from_far hedged with the far contract IS flat_from_hedge__far.
    far_model = C.Q_MODELS["flat_from_far"]
    assert far_model.dividend_policy == "far"
    assert C.Q_MODELS["flat_from_hedge"].dividend_policy is None


def test_no_label_calls_the_primary_policy_neutral_to_every_node():
    label = C.HEDGE_STRATEGY_LABELS["buckets_spot_parallel"]
    assert "spot and parallel" in label
    assert "every" not in label
    # The one policy that IS neutral at every node says so, and only it.
    assert "every modelled node" in C.HEDGE_STRATEGY_LABELS["buckets_nodes"]
    for name, text in C.HEDGE_STRATEGY_LABELS.items():
        assert text != "bucket-neutral", name


def test_the_grid_selector_keeps_old_commands_stable():
    legacy = FLEET.resolve_cells(FLEET.parse_args([]))
    assert legacy == list(FLEET.DEFAULT_CELLS)
    quick = FLEET.resolve_cells(FLEET.parse_args(["--quick"]))
    assert quick == list(FLEET.QUICK_CELLS)
    buckets = FLEET.resolve_cells(FLEET.parse_args(["--study-grid", "buckets"]))
    assert buckets == list(C.PRIMARY_BUCKET_CELLS)
    combined = FLEET.resolve_cells(FLEET.parse_args(["--study-grid", "all"]))
    assert set(combined) >= set(legacy) | set(C.PRIMARY_BUCKET_CELLS)
    assert len(combined) == len(set(combined))


def test_quick_limits_inceptions_without_removing_bucket_controls():
    args = FLEET.parse_args(["--study-grid", "buckets", "--quick"])
    assert FLEET.resolve_cells(args) == list(C.PRIMARY_BUCKET_CELLS)
    # An explicit --cells subset overrides the grid, as a named subset.
    subset = FLEET.resolve_cells(
        FLEET.parse_args(
            ["--study-grid", "buckets", "--cells", "term_flat_q:buckets_nodes"]
        )
    )
    assert subset == [("term_flat_q", "buckets_nodes")]


# ---------------------------------------------------------------------------
# The strategy factory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, kind, objective",
    [
        ("front", AutocallableDeltaHedgeStrategy, None),
        ("far", AutocallableDeltaHedgeStrategy, None),
        ("front_scaled", ProportionalFuturesDeltaHedgeStrategy, None),
        ("far_scaled", ProportionalFuturesDeltaHedgeStrategy, None),
        ("buckets_nodes", FuturesBucketHedgeStrategy, "nodes"),
        ("buckets_far", FuturesBucketHedgeStrategy, "spot_far"),
        ("buckets_spot_parallel", FuturesBucketHedgeStrategy, "spot_parallel"),
    ],
)
def test_each_policy_builds_its_own_strategy(name, kind, objective):
    strategy = C.hedge_strategy_for(
        name, delta_threshold=0.25, round_contracts=False
    )
    assert type(strategy) is kind
    if objective is not None:
        assert strategy.objective == objective
    assert strategy.delta_threshold == 0.25
    assert strategy.round_contracts is False


def test_an_unknown_policy_fails_closed():
    with pytest.raises(C.StudyDataError):
        C.hedge_strategy_for("buckets_everything", delta_threshold=0.0,
                             round_contracts=True)
    with pytest.raises(C.StudyDataError):
        C.hedge_roll_policy_for("buckets_everything")


def test_scaled_policies_follow_their_own_roll_selector():
    front = C.hedge_roll_policy_for("front_scaled")
    far = C.hedge_roll_policy_for("far_scaled")
    assert type(front) is FuturesRollPolicy
    assert type(far) is C.FarContractRollPolicy
    # Bucket policies use the FRONT selector for their reference columns only.
    for name in C.BUCKET_OBJECTIVES:
        assert type(C.hedge_roll_policy_for(name)) is FuturesRollPolicy


def test_the_hedge_ratio_reaches_the_bucket_strategy():
    strategy = C.hedge_strategy_for(
        "buckets_spot_parallel",
        delta_threshold=0.0,
        round_contracts=True,
        hedge_ratio=1.2,
    )
    assert strategy.hedge_ratio == 1.2
    assert strategy.get_parameters()["hedge_ratio"] == 1.2


# ---------------------------------------------------------------------------
# Source restrictions
# ---------------------------------------------------------------------------


def chain_slice(valuation="2024-01-02"):
    valuation = pd.Timestamp(valuation)
    rows = []
    for contract, expiry, carry in (
        ("IM2401", "2024-02-16", 0.03),
        ("IM2402", "2024-03-15", 0.06),
        ("IM2403", "2024-06-21", 0.09),
    ):
        expiry_stamp = pd.Timestamp(expiry)
        tenor = (expiry_stamp - valuation).days / 365.0
        rows.append(
            {
                "date": valuation,
                "contract": contract,
                "futures_price": 5000.0 * math.exp((0.02 - carry) * tenor),
                "expiry_date": expiry_stamp,
                "multiplier": 200.0,
            }
        )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("model_name", ["term_flat_q", "term_flat_fwd"])
def test_the_supported_term_models_build_a_carry_context(model_name):
    context = C.carry_context_for(
        C.Q_MODELS[model_name],
        valuation=pd.Timestamp("2024-01-02"),
        spot=5000.0,
        rate=0.02,
        chain_slice=chain_slice(),
    )
    assert context.extrapolation == C.Q_MODELS[model_name].extrapolation
    assert context.spot == 5000.0
    assert context.contracts == ("IM2401", "IM2402", "IM2403")


@pytest.mark.parametrize(
    "model_name", ["flat_from_hedge", "flat_from_far", "surface_fwd", "term_opt_tail"]
)
def test_flat_and_external_tail_models_have_no_bucket_coordinates(model_name):
    with pytest.raises(C.StudyDataError):
        C.carry_context_for(
            C.Q_MODELS[model_name],
            valuation=pd.Timestamp("2024-01-02"),
            spot=5000.0,
            rate=0.02,
            chain_slice=chain_slice(),
        )


def test_the_static_context_and_the_replay_dividend_agree():
    """Stage 01 and the replay share ONE builder, so they cannot drift."""
    for model_name in ("term_flat_q", "term_flat_fwd"):
        model = C.Q_MODELS[model_name]
        context = C.carry_context_for(
            model,
            valuation=pd.Timestamp("2024-01-02"),
            spot=5000.0,
            rate=0.02,
            chain_slice=chain_slice(),
        )
        replay_dividend = C.dividend_for(
            model,
            valuation=pd.Timestamp("2024-01-02"),
            spot=5000.0,
            rate=0.02,
            chain_slice=chain_slice(),
        )
        for tenor in (0.05, 0.12, 0.25, 0.5, 1.0):
            assert context.dividend().get_yield(tenor) == pytest.approx(
                replay_dividend.get_yield(tenor), abs=1e-15
            )


def test_a_single_eligible_quote_still_builds_a_diagnostic_context():
    frame = chain_slice().iloc[:1]
    context = C.carry_context_for(
        C.Q_MODELS["term_flat_q"],
        valuation=pd.Timestamp("2024-01-02"),
        spot=5000.0,
        rate=0.02,
        chain_slice=frame,
    )
    assert context.contracts == ("IM2401",)
    # ... and the primary policy still refuses to size against one node.
    from quantark.backtest.futures_risk import FuturesBookRisk, FuturesBucket
    from quantark.backtest.strategy import ideal_targets
    from quantark.util.exceptions import ValidationError

    risk = FuturesBookRisk(
        spot=5000.0,
        delta_q=1.0,
        buckets=(
            FuturesBucket(
                contract="IM2401",
                expiry_date=None,
                tenor_years=0.12,
                price=4990.0,
                multiplier=200.0,
                bucket_currency=1.0,
            ),
        ),
    )
    with pytest.raises(ValidationError):
        ideal_targets(risk, "spot_parallel")


# ---------------------------------------------------------------------------
# The run profile and its fingerprint
# ---------------------------------------------------------------------------


def test_the_buckets_profile_records_and_audits_daily_by_default():
    profile = FLEET.resolve_risk_profile(
        FLEET.parse_args(["--study-grid", "buckets"])
    )
    assert profile["record_carry_exposure"] is True
    assert profile["carry_audit_mode"] == "daily"
    assert profile["exploratory_audit_override"] is False
    assert profile["reference_notional"] == C.NOTIONAL


def test_a_legacy_run_records_nothing_by_default():
    profile = FLEET.resolve_risk_profile(FLEET.parse_args([]))
    assert profile["record_carry_exposure"] is False
    assert profile["carry_audit_mode"] == "none"


def test_an_audit_override_is_marked_exploratory():
    profile = FLEET.resolve_risk_profile(
        FLEET.parse_args(
            ["--study-grid", "buckets", "--carry-audit-mode", "sampled"]
        )
    )
    assert profile["carry_audit_mode"] == "sampled"
    assert profile["exploratory_audit_override"] is True
    # ... and that marking is inside the fingerprint, so it cannot be reused
    # as a full daily-audit certificate.
    base = dict(_task(), risk=FLEET.resolve_risk_profile(
        FLEET.parse_args(["--study-grid", "buckets"])
    ))
    override = dict(_task(), risk=profile)
    assert FLEET.fingerprint(base) != FLEET.fingerprint(override)


def _task():
    return {
        "model": "term_flat_q",
        "hedge": "buckets_spot_parallel",
        "quad_grid": 2000,
        "cost_bp": 1.0,
        "rate": 0.02,
        "vol_tenor": 1.0,
        "coupon": 0.08,
        "delta_threshold": 0.0,
        "round_contracts": True,
        "inception": "2021-01-04",
        "trade_end": "2022-01-04",
    }


def test_a_legacy_task_fingerprint_is_unchanged_by_the_new_block():
    task = _task()
    assert "risk" not in task
    before = FLEET.fingerprint(task)
    assert FLEET.fingerprint(dict(task)) == before
    # Only a task that actually carries a risk block gets a new fingerprint.
    with_risk = dict(task, risk=FLEET.resolve_risk_profile(
        FLEET.parse_args(["--study-grid", "buckets"])
    ))
    assert FLEET.fingerprint(with_risk) != before


@pytest.mark.parametrize(
    "flag",
    [
        ["--hedge-ratio", "0.5"],
        ["--futures-bump-points", "0.5"],
        ["--audit-spot-bump-rel", "0.002"],
        ["--audit-yield-bump", "5e-5"],
        ["--delta-tolerance-hands", "0.02"],
        ["--rhoq-tolerance-bp", "0.02"],
        ["--carry-audit-mode", "sampled"],
        ["--carry-audit-dates", "2025-03-03"],
        ["--stress-dates", "2025-03-03"],
    ],
)
def test_every_pricing_relevant_option_changes_the_fingerprint(flag):
    base = dict(_task(), risk=FLEET.resolve_risk_profile(
        FLEET.parse_args(["--study-grid", "buckets"])
    ))
    changed = dict(_task(), risk=FLEET.resolve_risk_profile(
        FLEET.parse_args(["--study-grid", "buckets"] + flag)
    ))
    assert FLEET.fingerprint(base) != FLEET.fingerprint(changed), flag


def test_the_objective_is_part_of_the_fingerprint():
    profile = FLEET.resolve_risk_profile(
        FLEET.parse_args(["--study-grid", "buckets"])
    )
    nodes = dict(_task(), hedge="buckets_nodes", risk=profile)
    parallel = dict(_task(), hedge="buckets_spot_parallel", risk=profile)
    assert FLEET.fingerprint(nodes) != FLEET.fingerprint(parallel)


def test_carry_settings_are_none_when_nothing_is_recorded():
    assert FLEET._carry_settings({}) is None
    assert FLEET._carry_settings(
        {"record_carry_exposure": False, "carry_audit_mode": "none"}
    ) is None
    settings = FLEET._carry_settings(
        FLEET.resolve_risk_profile(FLEET.parse_args(["--study-grid", "buckets"]))
    )
    assert settings.reference_notional == C.NOTIONAL
    assert settings.futures_bump_points == 1.0
    assert settings.audit_yield_bump == 1e-4


def test_stress_and_audit_dates_round_trip_into_settings():
    profile = FLEET.resolve_risk_profile(
        FLEET.parse_args(
            [
                "--study-grid",
                "buckets",
                "--stress-dates",
                "2025-03-03",
                "2025-03-04",
            ]
        )
    )
    settings = FLEET._carry_settings(profile)
    assert settings.stress_dates == (
        datetime(2025, 3, 3),
        datetime(2025, 3, 4),
    )
