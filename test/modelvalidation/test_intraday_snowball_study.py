"""Wiring test for the daily-KI intraday study. Soundness, not outcome: the verdict comes from the full run."""
import dataclasses
from pathlib import Path

import pytest

from quantark.modelvalidation.anchors import extract_anchors
from quantark.modelvalidation.builders.intraday_common import build_request, make_snowball
from quantark.modelvalidation.evidence import identity_hash
from quantark.modelvalidation.pipeline import certify, quick_policy, validate_payload
from quantark.modelvalidation.reference import reference_targets
from quantark.modelvalidation.study import reference_kind
from quantark.modelvalidation.yaml_loader import load_study
from quantark.intraday.context import resolve_context
from quantark.intraday.timestamp import to_utc

STUDY_PATH = Path("example/modelvalidation/snowball_intraday_daily_ki_bsm.yaml")
CANDIDATES = ("equity.snowball.intraday.quad_v2", "equity.snowball.intraday.pde")
CASES = [
    "ordinary", "near_ki_above", "near_ki_below", "near_ki_1s", "near_ki_10s", "near_ki_5m",
    "near_ko_on_ko_day", "near_ko_on_ko_day_1s", "ko_level_on_ki_day", "pre_open", "lunch_break",
    "lunch_break_zero_variance", "holiday_eve", "maturity_day", "already_ki", "terminated_pending_cash",
    "provisional_ki_after_close", "low_vol", "high_vol", "carry",
    "uniform_profile", "sessions_only_profile", "on_ki_barrier_at_close",
]


@pytest.fixture(scope="module")
def study():
    return load_study(STUDY_PATH)


def test_study_declares_schema_2_with_the_frozen_budgets(study):
    assert study.schema == 2 and study.name == "snowball-intraday-daily-ki-bsm"
    assert study.quantities == ("pv", "desk_delta", "desk_gamma", "desk_theta", "point_delta", "point_gamma")
    qb = study.quantity_bounds
    assert (qb["pv"].abs_floor, qb["pv"].rel) == (1e-6, 0.0)
    assert (qb["point_delta"].abs_floor, qb["point_delta"].rel) == (1e-5, 1e-4) == (qb["desk_delta"].abs_floor, qb["desk_delta"].rel)
    assert (qb["point_gamma"].abs_floor, qb["point_gamma"].rel) == (1e-4, 1e-3) == (qb["desk_gamma"].abs_floor, qb["desk_gamma"].rel)
    assert (qb["desk_theta"].abs_floor, qb["desk_theta"].rel) == (1e-6, 1e-4)
    assert study.bounds.cell == 1.0 and study.bounds.mean_signed_bias == 0.2
    assert study.bounds.se_budget_fraction == 0.25 and study.bounds.envelope_fraction == 0.5
    assert study.bounds.radius_budget_fraction == 0.25                        # the deterministic reference's allowance
    assert study.sampling.min_batches == study.sampling.max_batches == 32     # frozen budget, no stopping rule
    assert study.scale.notional == 100.0 and study.scale.spot_scale == 100.0


def test_case_list_is_the_declared_one(study):
    assert [case.name for case in study.cases] == CASES and len(CASES) == 23
    assert study.sampling.seed_scheme == "substream"
    assert tuple(c.name() for c in study.candidates) == CANDIDATES
    on_barrier = next(c for c in study.cases if c.name == "on_ki_barrier_at_close")
    assert on_barrier.expected == {"point_delta": "undefined", "point_gamma": "undefined", "desk_theta": "undefined"}


def test_the_reference_is_deterministic_targets_every_quantity_and_is_qualified_by_rqmc(study):
    assert reference_kind(study.reference) == "deterministic" and study.reference.levels == (2001, 4001, 8001, 16001, 32001)
    targets = reference_targets(study.reference, study.quantities)
    assert all(targets[q] is not None for q in study.quantities)              # the point Greeks are certified here
    assert targets["desk_delta"]["bump"] == study.sampling.bump
    qualifier = study.qualification.builder
    assert reference_kind(qualifier) == "stochastic" and study.qualification.max_z == 4.0
    assert qualifier.config()["engine"] == "SnowballMCEngine"
    assert reference_targets(qualifier, study.quantities)["point_gamma"] is None   # it qualifies PV and the desk moves only
    for candidate in study.candidates:
        assert candidate.params()["desk_bump"] == study.sampling.bump


def test_the_contract_is_the_daily_ki_fixture(study):
    product = make_snowball(study.reference.product_params)
    assert product.initial_price == 100.0 and product.barrier_config.ko_barrier == 103.0
    assert product.barrier_config.ki_barrier == 75.0 and product.payoff_config.rebate_rate == 0.12
    assert len(product.barrier_config.ko_observation_schedule.records) == 12
    assert len(product.barrier_config.ki_observation_schedule.records) > 200


def _context_of(study, name):
    env, product, context = study.reference.specs(next(c for c in study.cases if c.name == name))
    return resolve_context(build_request(env, product, context))


def test_holiday_eve_still_observes_its_own_close_and_then_waits_out_the_holiday(study):
    """Review R11: the 15:00 close of 2026-09-30 is itself a KI observation; the holiday is the gap AFTER it."""
    ctx = _context_of(study, "holiday_eve")
    ahead = sorted({to_utc(e.timestamp) for e in ctx.numerical.remaining_events if to_utc(e.timestamp) > to_utc(ctx.valuation_timestamp)})
    assert (ahead[0] - to_utc(ctx.valuation_timestamp)).total_seconds() == 3600.0
    assert (ahead[1] - ahead[0]).days >= 7


def test_a_future_fixing_is_not_provisional_and_a_due_missing_one_is(study):
    """Spec correction 1: below the barrier an hour before the close the claim is alive, not knocked in."""
    below = _context_of(study, "near_ki_below")
    assert not below.provisional and not below.numerical.knocked_in and below.numerical.lifecycle_state.alive
    due = _context_of(study, "provisional_ki_after_close")
    assert due.provisional and due.numerical.knocked_in


def test_the_desk_lunch_carries_variance_and_the_sessions_only_lunch_does_not(study):
    """Spec correction 2: the zero-variance control is the sessions_only case, not the desk one."""
    one_second = 1.0 / (365.0 * 86400.0)
    desk = _context_of(study, "lunch_break")
    assert float(desk.pricing_env.vol_surface.total_variance(100.0, one_second, 76.0)) > 0.0
    flat = _context_of(study, "lunch_break_zero_variance")
    assert float(flat.pricing_env.vol_surface.total_variance(100.0, one_second, 76.0)) == 0.0


def test_quick_certification_runs_end_to_end(tmp_path_factory, study):
    by_name = {case.name: case for case in study.cases}
    small = dataclasses.replace(study, cases=(by_name["ordinary"], by_name["on_ki_barrier_at_close"]))
    certificate = certify(small, out_dir=tmp_path_factory.mktemp("intraday"), quick=True)
    payload = certificate.payload
    validate_payload(payload)
    assert set(payload["decisions"]) == set(CANDIDATES)
    # Review R10: every arm ran under the policy the payload records -- the quick one, wiring ladder included
    sampling = quick_policy(small.sampling)
    assert payload["study"]["sampling"]["paths_per_batch"] == sampling.paths_per_batch == 4096
    bound = small.reference.bind(sampling, quick=True)
    qualifier = small.qualification.builder.bind(sampling)
    assert payload["contract"]["reference_kind"] == "deterministic"
    assert payload["contract"]["reference_error_model"]["levels"] == [101, 201, 401, 801, 1601]
    assert payload["contract"]["reference_error_model"]["ladder_policy"]["radius_kind"] == "calibrated_numerical_estimate"
    for case in small.cases:
        block = payload["references"][case.name]
        assert block["kind"] == "deterministic" and block["identity_hash"] == identity_hash(bound.identity(case))
        assert payload["qualification"][case.name]["identity_hash"] == identity_hash(qualifier.identity(case))
        assert payload["qualification"][case.name]["batches"] == sampling.max_batches
    assert set(payload["qualification"]["ordinary"]["checks"]) == {"pv", "desk_delta", "desk_gamma", "desk_theta"}
    assert set(payload["qualification"]["on_ki_barrier_at_close"]["checks"]) == {"pv", "desk_delta", "desk_gamma"}
    assert set(payload["references"]["on_ki_barrier_at_close"]["undefined"]) == {"desk_theta", "point_delta", "point_gamma"}
    assert payload["study"]["uncertified_quantities"] == []
    assert not [c for c in payload["cells"] if c["verdict"] == "ERROR"], [c["error"] for c in payload["cells"] if c["verdict"] == "ERROR"]
    for cell in payload["cells"]:
        if cell["case"] == "ordinary":
            assert cell["kind"] == "numeric" and cell["reference"]["kind"] == "deterministic"
            assert cell["reference"]["basis"] == "calibrated_estimate"
            # The wiring ladder is pre-asymptotic on purpose, so a radius may be unbounded or uncalibrated there
            # (desk theta is: the rule one level down did not cover it). A cell is then ungated WITH its reason;
            # a gated one is typed by its radius, never by a standard error.
            if cell["gate"] is None:
                assert cell["verdict"] == "UNRESOLVED" and ("could not bound" in cell["reason"] or "not qualified" in cell["reason"])
            else:
                assert cell["gate"]["se_c"] is None and cell["gate"]["radius_c"] is not None
        if cell["case"] == "on_ki_barrier_at_close" and cell["quantity"] in ("point_delta", "point_gamma", "desk_theta"):
            assert cell["kind"] == "semantic" and cell["verdict"] == "PASS" and cell["reference"]["value"] is None
    rules = {q: rec["rule"] for q, rec in payload["references"]["ordinary"]["evidence"]["quantities"].items()}
    assert set(rules.values()) <= {"geometric", "correction", "unextrapolated", "unbounded", "uncalibrated"}
    assert (certificate.path.parent / "report.md").exists()
    anchors = extract_anchors(payload, small)["anchors"]
    assert {a["candidate"] for a in anchors} == set(CANDIDATES)
