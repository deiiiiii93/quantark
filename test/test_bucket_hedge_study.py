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


# ---------------------------------------------------------------------------
# Versioned persistence
# ---------------------------------------------------------------------------


class FakeSingleResults:
    """The single-result API: frames are PROPERTIES."""

    def __init__(self, frames):
        self._frames = frames

    @property
    def states_df(self):
        return self._frames["states"]

    @property
    def greeks_df(self):
        return self._frames["greeks"]

    @property
    def trades_df(self):
        return self._frames["trades"]

    @property
    def rebalance_df(self):
        return self._frames["rebalances"]

    @property
    def actions_df(self):
        return self._frames["actions"]

    @property
    def hedge_legs_df(self):
        return self._frames["hedge_legs"]

    @property
    def hedge_attribution_df(self):
        return self._frames["hedge_attribution"]

    @property
    def hedge_stresses_df(self):
        return self._frames["hedge_stresses"]


class FakeBookResults:
    """The book-result API: frames are METHODS."""

    def __init__(self, frames):
        self._frames = frames

    def states_df(self):
        return self._frames["states"]

    def greeks_df(self):
        return self._frames["greeks"]

    def trades_df(self):
        return self._frames["trades"]

    def rebalances_df(self):
        return self._frames["rebalances"]

    def actions_df(self):
        return self._frames["actions"]

    def hedge_legs_df(self):
        return self._frames["hedge_legs"]

    def hedge_attribution_df(self):
        return self._frames["hedge_attribution"]

    def hedge_stresses_df(self):
        return self._frames["hedge_stresses"]


def _frames(*, empty_stresses=True):
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    base = pd.DataFrame({"total_pnl": [1.0, 2.0]}, index=index)
    base.index.name = "date"
    legs = pd.DataFrame(
        {
            "contract": ["IM2401", "IM2401"],
            "net_rhoq_bp": [1.0, 1.1],
            "audit_status": ["pass", "pass"],
        },
        index=index,
    )
    legs.index.name = "date"
    attribution = pd.DataFrame(
        {"audit_status": ["pass", "fail"], "attribution_status": ["first_date", "ok"]},
        index=index,
    )
    attribution.index.name = "date"
    stresses = pd.DataFrame(
        columns=["scenario_id", "product_pnl"],
        index=pd.DatetimeIndex([], name="date"),
    )
    if not empty_stresses:
        stresses = pd.DataFrame(
            {"scenario_id": ["tail+0.01"], "product_pnl": [-1.0]}, index=index[:1]
        )
        stresses.index.name = "date"
    return {
        "states": base,
        "greeks": base,
        "trades": base,
        "rebalances": base,
        "actions": base,
        "hedge_legs": legs,
        "hedge_attribution": attribution,
        "hedge_stresses": stresses,
    }


@pytest.mark.parametrize("api", [FakeSingleResults, FakeBookResults])
def test_the_frame_adapter_handles_both_result_apis(api):
    results = api(_frames())
    for name in C.LEGACY_RUN_FRAMES + C.CARRY_RUN_FRAMES:
        assert isinstance(C.result_frame(results, name), pd.DataFrame)
    # An EMPTY frame is falsy; the adapter must not read that as the wrong API.
    assert C.result_frame(results, "hedge_stresses").empty


def test_a_new_run_round_trips_all_eight_frames(tmp_path):
    results = FakeSingleResults(_frames())
    run_dir = tmp_path / "runs" / "2024-01" / "term_flat_q__buckets_spot_parallel"
    C.write_run(
        run_dir,
        results,
        {"fingerprint": "abc", "status": "completed"},
        run_format=C.RUN_FORMAT_CARRY,
        run_config={"objective": "spot_parallel"},
        audit_summary={"audit_coverage": "measured"},
    )
    for name in C.LEGACY_RUN_FRAMES + C.CARRY_RUN_FRAMES:
        assert (run_dir / f"{name}.csv").exists(), name
    assert (run_dir / "run_config.json").exists()
    assert (run_dir / "audit_summary.json").exists()

    loaded = C.load_run(run_dir)
    assert loaded["run_format"] == C.RUN_FORMAT_CARRY
    for name in C.LEGACY_RUN_FRAMES + C.CARRY_RUN_FRAMES:
        assert name in loaded, name
    # The empty stress file keeps its fixed columns.
    assert list(loaded["hedge_stresses"].columns) == ["scenario_id", "product_pnl"]
    assert loaded["run_config"]["objective"] == "spot_parallel"
    assert loaded["audit_summary"]["audit_coverage"] == "measured"


def test_a_legacy_run_stays_readable_and_reports_no_audit_coverage(tmp_path):
    results = FakeSingleResults(_frames())
    run_dir = tmp_path / "runs" / "2024-01" / "flat_from_hedge__front"
    C.write_run(run_dir, results, {"fingerprint": "abc", "status": "completed"})
    assert not (run_dir / "hedge_legs.csv").exists()
    loaded = C.load_run(run_dir)
    assert loaded["run_format"] == C.RUN_FORMAT_LEGACY
    assert loaded["audit_summary"]["audit_coverage"] == "not_available"
    assert set(C.LEGACY_RUN_FRAMES) <= set(loaded)


def test_a_new_format_run_missing_a_required_frame_is_an_error(tmp_path):
    results = FakeSingleResults(_frames())
    run_dir = tmp_path / "runs" / "2024-01" / "term_flat_q__buckets_nodes"
    C.write_run(
        run_dir,
        results,
        {"fingerprint": "abc", "status": "completed"},
        run_format=C.RUN_FORMAT_CARRY,
        run_config={},
        audit_summary={},
    )
    (run_dir / "hedge_attribution.csv").unlink()
    with pytest.raises(C.StudyDataError, match="hedge_attribution"):
        C.load_run(run_dir)


def test_audit_coverage_counts_statuses_rather_than_returning_a_boolean():
    frames = _frames()
    coverage = C.audit_coverage(frames["hedge_legs"], frames["hedge_attribution"])
    assert coverage["audit_coverage"] == "measured"
    assert coverage["dates"] == 2
    assert coverage["by_status"]["pass"] == 1
    assert coverage["by_status"]["fail"] == 1
    # One failed audit means the run cannot serve as a passing gate result.
    assert coverage["all_measured_passed"] is False

    unmeasured = frames["hedge_attribution"].copy()
    unmeasured["audit_status"] = "not_measured"
    quiet = C.audit_coverage(frames["hedge_legs"], unmeasured)
    assert quiet["audit_coverage"] == "not_measured"
    assert quiet["all_measured_passed"] is False
    # No frame at all is a THIRD thing: the run predates carry recording.
    assert C.audit_coverage(None, None)["audit_coverage"] == "not_available"


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def _write_run(tmp_path, task, *, run_format, audit_coverage=None):
    run_dir = C.run_dir_for(
        tmp_path, task["inception_tag"], task["model"], task["hedge"]
    )
    summary = {
        "fingerprint": FLEET.fingerprint(task),
        "status": "completed",
        "run_format": run_format,
    }
    if audit_coverage is not None:
        summary["audit_coverage"] = audit_coverage
    C.write_run(
        run_dir,
        FakeSingleResults(_frames()),
        summary,
        run_format=run_format,
        run_config={},
        audit_summary={"audit_coverage": audit_coverage or "not_available"},
    )
    return run_dir, summary


def _audited_task():
    task = dict(_task())
    task["inception_tag"] = "2021-01"
    task["risk"] = FLEET.resolve_risk_profile(
        FLEET.parse_args(["--study-grid", "buckets"])
    )
    return task


def test_a_legacy_artifact_cannot_resume_an_audited_task(tmp_path):
    task = _audited_task()
    run_dir, summary = _write_run(tmp_path, task, run_format=C.RUN_FORMAT_LEGACY)
    assert FLEET.resumable(summary, task, run_dir) is False


def test_a_complete_audited_artifact_resumes(tmp_path):
    task = _audited_task()
    run_dir, summary = _write_run(
        tmp_path, task, run_format=C.RUN_FORMAT_CARRY, audit_coverage="measured"
    )
    assert FLEET.resumable(summary, task, run_dir) is True


def test_an_audited_task_will_not_resume_an_unmeasured_run(tmp_path):
    task = _audited_task()
    run_dir, summary = _write_run(
        tmp_path, task, run_format=C.RUN_FORMAT_CARRY, audit_coverage="not_measured"
    )
    assert FLEET.resumable(summary, task, run_dir) is False


def test_a_missing_new_frame_blocks_the_resume(tmp_path):
    task = _audited_task()
    run_dir, summary = _write_run(
        tmp_path, task, run_format=C.RUN_FORMAT_CARRY, audit_coverage="measured"
    )
    (run_dir / "hedge_stresses.csv").unlink()
    assert FLEET.resumable(summary, task, run_dir) is False


def test_a_failed_summary_never_resumes(tmp_path):
    task = _audited_task()
    run_dir, summary = _write_run(
        tmp_path, task, run_format=C.RUN_FORMAT_CARRY, audit_coverage="measured"
    )
    summary["status"] = "failed"
    assert FLEET.resumable(summary, task, run_dir) is False


def test_a_legacy_task_still_resumes_a_legacy_artifact(tmp_path):
    task = dict(_task(), inception_tag="2021-01")
    run_dir, summary = _write_run(tmp_path, task, run_format=C.RUN_FORMAT_LEGACY)
    assert FLEET.resumable(summary, task, run_dir) is True


# ---------------------------------------------------------------------------
# Failure records
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message, category",
    [
        ("bucket hedge infeasible on 2025-03-03: objective=...", "infeasible_hedge"),
        ("no tradable mark for IM2401 on 2025-03-03", "missing_price"),
        ("nodal rhoq IM2401 +0.5 bp vs tolerance 0.01", "numeric_audit"),
        ("something else entirely", "other"),
    ],
)
def test_failure_categories_stay_distinct(message, category):
    assert FLEET.error_category(Exception(message)) == category


def test_a_failure_is_written_beside_a_completed_run_not_over_it(tmp_path):
    task = dict(_audited_task())
    task["out_dir"] = str(tmp_path)
    task["history_dir"] = str(tmp_path)
    run_dir, summary = _write_run(
        tmp_path, task, run_format=C.RUN_FORMAT_CARRY, audit_coverage="measured"
    )
    before = (run_dir / "run_summary.json").read_text()

    original = FLEET.run_cell
    def boom(_task):
        raise RuntimeError("bucket hedge infeasible on 2025-03-03")

    FLEET.run_cell = boom
    try:
        outcome = FLEET._run_cell_guarded(task)
    finally:
        FLEET.run_cell = original
    assert outcome["status"] == "failed"
    assert outcome["error_category"] == "infeasible_hedge"
    assert outcome["objective"] == "spot_parallel"
    assert outcome["input_fingerprint"] == FLEET.fingerprint(task)
    assert (run_dir / "failure.json").exists()
    # The completed summary is untouched.
    assert (run_dir / "run_summary.json").read_text() == before


def test_the_source_digest_follows_content_not_a_revision_name():
    digest = FLEET.source_digest()
    assert len(digest) == 16
    assert digest == FLEET.source_digest()
