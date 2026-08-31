"""Tests for example/mo_volmodels/12_snowball_volmodel_backtest.py (Phase 4).

Covers the pieces that decide what the fleet actually runs: term-sheet parity
with the gate-certified stage 11 product, the inception scheduler, Gate G2
routing, the fair-coupon root finder, engine-config wiring, cost model, and
run classification (KO / matured / censored).

Deliberately excluded: an end-to-end replay. One replay day of the production
3Y snowball costs ~29s of PDE (measured), so an honest end-to-end run belongs
to the fleet, not the test suite. The seams that end-to-end coverage would
protect are covered here directly.
"""

import importlib.util
import hashlib
import json
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "example/mo_volmodels/12_snowball_volmodel_backtest.py"
HISTORY_DIR = ROOT / "example/mo_volmodels/data/history"

spec = importlib.util.spec_from_file_location(
    "snowball_volmodel_backtest_12", MODULE_PATH
)
s12 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = s12  # dataclasses resolve cls.__module__ here
spec.loader.exec_module(s12)

s11 = s12.stage11()

from quantark.util.enum.engine_enums import EngineType, MonteCarloMethod  # noqa: E402
from quantark.util.exceptions import ValidationError  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _weekday_calendar(start: date, end: date):
    days = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    return s11.TradingCalendar(days)


@pytest.fixture(scope="module")
def calendar():
    return _weekday_calendar(date(2023, 1, 2), date(2030, 12, 31))


@pytest.fixture(scope="module")
def terms(calendar):
    return s11.build_snowball_terms(date(2023, 5, 4), calendar)


# ---------------------------------------------------------------------------
# Term-sheet parity with the gate-certified product
# ---------------------------------------------------------------------------


def test_backtest_product_matches_gate_term_sheet(terms):
    """The backtest must price the SAME contract stage 11's gate certified.

    Only three things may differ: position size, the solved coupon, and
    initial_date (which the lifecycle tracker needs and a pricing gate does
    not).  Everything contractual must be identical.
    """
    s0 = 6733.97
    gate_product = s11.build_snowball_product(terms, s0)
    bt_product = s12.build_backtest_product(
        terms, initial_spot=s0, coupon=0.15, notional=50_000_000.0
    )

    assert bt_product.initial_price == gate_product.initial_price
    assert bt_product.strike == gate_product.strike
    assert bt_product.maturity == gate_product.maturity
    assert bt_product.is_reverse == gate_product.is_reverse

    gb, bb = gate_product.barrier_config, bt_product.barrier_config
    assert bb.ko_barrier == gb.ko_barrier
    assert bb.ki_barrier == gb.ki_barrier
    assert list(bb.ko_observation_dates) == list(gb.ko_observation_dates)
    assert list(bb.ki_observation_dates) == list(gb.ki_observation_dates)
    assert bb.ko_observation_type == gb.ko_observation_type
    assert bb.ki_observation_type == gb.ki_observation_type
    assert bb.ki_continuous == gb.ki_continuous

    gp, bp = gate_product.payoff_config, bt_product.payoff_config
    assert bp.include_principal == gp.include_principal
    assert bp.protection_type == gp.protection_type
    assert bp.participation_rate == gp.participation_rate


def test_backtest_product_scales_to_notional(terms):
    s0 = 6733.97
    notional = 50_000_000.0
    product = s12.build_backtest_product(
        terms, initial_spot=s0, coupon=0.15, notional=notional
    )
    assert product.contract_multiplier == pytest.approx(notional / s0)
    assert product.initial_price * product.contract_multiplier == pytest.approx(
        notional
    )


def test_backtest_product_carries_the_solved_coupon(terms):
    product = s12.build_backtest_product(terms, initial_spot=6733.97, coupon=0.1234)
    assert product.barrier_config.ko_rate == pytest.approx(0.1234)
    assert product.payoff_config.rebate_rate == pytest.approx(0.1234)


def test_backtest_product_sets_initial_date_for_the_lifecycle(terms):
    product = s12.build_backtest_product(terms, initial_spot=6733.97, coupon=0.15)
    assert product.initial_date.date() == terms.inception


# ---------------------------------------------------------------------------
# Inception scheduler
# ---------------------------------------------------------------------------


def test_schedule_is_monthly_and_ordered(calendar):
    out = s12.schedule_inceptions(
        calendar=calendar,
        data_start=date(2023, 5, 4),
        data_end=date(2026, 7, 22),
        first_admitted_surface=date(2023, 5, 4),
        min_observable_months=12,
    )
    assert out == sorted(out)
    assert len(out) == len(set(out)), "no duplicate inceptions"
    months = {(d.year, d.month) for d in out}
    assert len(months) == len(out), "one inception per calendar month"
    assert all(calendar.is_trading_day(d) for d in out)


def test_schedule_honours_min_observable_months(calendar):
    data_end = date(2026, 7, 22)
    for min_months in (0, 12, 24):
        out = s12.schedule_inceptions(
            calendar=calendar,
            data_start=date(2023, 5, 4),
            data_end=data_end,
            first_admitted_surface=date(2023, 5, 4),
            min_observable_months=min_months,
        )
        assert out, f"expected inceptions at min_observable_months={min_months}"
        assert s11.add_months(out[-1], min_months) <= data_end


def test_schedule_shrinks_as_the_requirement_grows(calendar):
    kwargs = dict(
        calendar=calendar,
        data_start=date(2023, 5, 4),
        data_end=date(2026, 7, 22),
        first_admitted_surface=date(2023, 5, 4),
    )
    n0 = len(s12.schedule_inceptions(min_observable_months=0, **kwargs))
    n12 = len(s12.schedule_inceptions(min_observable_months=12, **kwargs))
    n24 = len(s12.schedule_inceptions(min_observable_months=24, **kwargs))
    assert n0 > n12 > n24


def test_schedule_skips_dates_without_an_admitted_surface(calendar):
    out = s12.schedule_inceptions(
        calendar=calendar,
        data_start=date(2023, 5, 4),
        data_end=date(2026, 7, 22),
        first_admitted_surface=date(2024, 1, 2),
        min_observable_months=12,
    )
    assert out and min(out) >= date(2024, 1, 2)


def test_schedule_rejects_negative_requirement(calendar):
    with pytest.raises(ValidationError):
        s12.schedule_inceptions(
            calendar=calendar,
            data_start=date(2023, 5, 4),
            data_end=date(2026, 7, 22),
            first_admitted_surface=date(2023, 5, 4),
            min_observable_months=-1,
        )


@pytest.mark.skipif(
    not (HISTORY_DIR / "surface_manifest.json").exists(),
    reason="IV surface history not built",
)
def test_schedule_on_the_real_window(calendar):
    """The production window must yield the 27 monthly inceptions of record.

    ``data_end`` comes from the FROZEN cohort, exactly as ``--data-end`` now
    defaults, not from the tail of the spot cache.  A live launchd job extends
    that cache every weekday, and it had already drifted far enough to admit a
    28th inception (measured 2026-08-25: spot tail 2026-08-24 -> 28 vs the 27
    of record), which would have silently resized the fleet and voided the
    banked coupons.
    """
    from quantark.param.vol.surface_history import VolSurfaceHistory

    history = VolSurfaceHistory(HISTORY_DIR)
    spot = s12.load_spot_frame(HISTORY_DIR)
    import pandas as pd

    real_calendar = s11.TradingCalendar.from_spot_csv(HISTORY_DIR / "csi1000_spot.csv")
    out = s12.schedule_inceptions(
        calendar=real_calendar,
        data_start=pd.Timestamp(spot["date"].iloc[0]).date(),
        data_end=s12.COHORT_ASOF,
        first_admitted_surface=history.admitted_dates[0],
        min_observable_months=12,
    )
    assert len(out) == 27
    assert out[0] == date(2023, 5, 4)
    assert out[-1] == date(2025, 7, 1)


@pytest.mark.skipif(
    not (HISTORY_DIR / "surface_manifest.json").exists(),
    reason="IV surface history not built",
)
def test_data_end_defaults_to_the_frozen_cohort_not_the_spot_tail():
    """Forgetting --data-end must not resize the fleet."""
    import pandas as pd

    spot = s12.load_spot_frame(HISTORY_DIR)
    spot_tail = pd.Timestamp(spot["date"].iloc[-1]).date()

    assert s12.parse_args([]).data_end == s12.COHORT_ASOF.isoformat()
    # The guard only matters while the cache really has run past the pin.
    assert spot_tail > s12.COHORT_ASOF


# ---------------------------------------------------------------------------
# Gate G2 routing
# ---------------------------------------------------------------------------


def _routing(**overrides):
    """Test double for a gate decision, keyed by study VARIANT name.

    flat_bsm/flat_bsm_quad/ts_bsm/localvol default to "pde" so tests that
    are not specifically exercising 1D/quad routing don't have to spell out
    all four every time; heston/heston_slv are left unset unless a test
    passes them explicitly, since those are what most routing tests vary.

    mc_params carries a representative gate-certified MC config for
    heston/heston_slv regardless of whether a given test actually routes
    them to "mc" -- make_engine_config now fails closed (Task 10) when an
    MC-routed variant has no entry here, so every test that reprices heston
    or heston_slv needs one available even if routing to MC isn't the point
    of that particular test.
    """
    routes = {
        "flat_bsm": "pde",
        "flat_bsm_quad": "pde",
        "ts_bsm": "pde",
        "localvol": "pde",
    }
    routes.update(overrides)
    gated_mc = {
        "paths_per_batch": 8192,
        "batches": 16,
        "seed": 20260723,
        "substeps_per_interval": 4,
        "scheme": "QUADEXP_M",
    }
    return s12.GateRouting(
        decision_path="test",
        evidence_sha256="deadbeef",
        routes=routes,
        pde_params={
            "heston": {
                "n_x": 200,
                "n_v": 60,
                "n_t": 1202,
                "scheme": "cs",
                **s12.ADI_2D_PRODUCTION_ENGINE_CONTROLS,
            },
            "heston_slv": {
                "n_x": 200,
                "n_v": 60,
                "n_t": 1202,
                "scheme": "cs",
                **s12.ADI_2D_PRODUCTION_ENGINE_CONTROLS,
            },
        },
        mc_params={"heston": gated_mc, "heston_slv": gated_mc},
    )


def test_one_dimensional_variants_now_read_their_own_route_from_the_decision():
    """After Task 4, flat_bsm/flat_bsm_quad/ts_bsm/localvol are inside the
    gate's scope too. They share vol_model across the three bsm variants,
    but the decision keys routes by the STUDY VARIANT name, so each can get
    an independent verdict -- proven here by giving them different routes."""
    routing = _routing(flat_bsm="mc", ts_bsm="pde", localvol="mc")
    assert routing.solver_for("flat_bsm") == "mc"
    assert routing.solver_for("ts_bsm") == "pde"
    assert routing.solver_for("localvol") == "mc"


def test_one_dimensional_variant_fails_closed_when_its_route_is_missing():
    """No more short-circuit: an absent 1D/quad entry must raise, never
    silently default to PDE."""
    routing = s12.GateRouting("t", None, {"heston": "mc"}, {})
    with pytest.raises(ValidationError, match="no usable route"):
        routing.solver_for("flat_bsm")
    with pytest.raises(ValidationError, match="no usable route"):
        routing.solver_for("localvol")


def test_one_dimensional_variant_fails_closed_on_an_unrecognised_route_string():
    """Present-but-garbage is just as fatal as absent -- no partial trust."""
    routing = s12.GateRouting("t", None, {"flat_bsm": "quadrature"}, {})
    with pytest.raises(ValidationError, match="no usable route"):
        routing.solver_for("flat_bsm")


def test_routing_reads_the_decision_file_not_a_hardcoded_default():
    assert _routing(heston="mc", heston_slv="mc").solver_for("heston") == "mc"
    assert _routing(heston="pde", heston_slv="mc").solver_for("heston") == "pde"


def test_routing_fails_closed_on_a_missing_or_bad_route():
    with pytest.raises(ValidationError, match="no usable route"):
        _routing(heston_slv="mc").solver_for("heston")
    with pytest.raises(ValidationError, match="no usable route"):
        _routing(heston="quadrature").solver_for("heston")


def test_pde_grid_options_only_flow_on_the_pde_route():
    assert _routing(heston="mc").engine_options_for("heston") == {}
    assert _routing(heston="pde").engine_options_for("heston") == {
        "n_x": 200,
        "n_v": 60,
        "n_t": 1202,
        **s12.ADI_2D_PRODUCTION_ENGINE_CONTROLS,
    }, "only supported numerical grid controls are forwarded to the solver"


def test_pde_route_rejects_the_legacy_power_grid_override():
    routing = _routing(heston="pde")
    routing.pde_params["heston"].pop("variance_grid_mode")
    routing.pde_params["heston"]["v_grid_power"] = 2.5

    with pytest.raises(ValidationError, match="stale 2-D production controls"):
        routing.engine_options_for("heston")


def test_load_gate_routing_is_actionable_when_absent(tmp_path):
    with pytest.raises(ValidationError, match="run stage 11"):
        s12.load_gate_routing(tmp_path / "nope.json")


def test_load_gate_routing_rejects_malformed_files(tmp_path):
    bad = tmp_path / "gate.json"
    bad.write_text("{not json")
    with pytest.raises(ValidationError, match="not valid JSON"):
        s12.load_gate_routing(bad)
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"schema_version": 1}))
    with pytest.raises(ValidationError, match="no 'variants' map"):
        s12.load_gate_routing(empty)


def test_load_gate_routing_requires_stage16_as_adi_greek_authority(tmp_path):
    path = tmp_path / "gate.json"
    payload = {
        "evidence_sha256": "abc",
        "variants": {
            "heston": {
                "route": "pde",
                "pde_params": {},
                "gate": {
                    "delta_authority": "stage11",
                    "delta_required": True,
                },
            }
        },
    }
    path.write_text(json.dumps(payload))

    with pytest.raises(ValidationError, match="delegate ADI Greek admission"):
        s12.load_gate_routing(path)

    payload["variants"]["heston"]["gate"] = {
        "delta_authority": "stage16",
        "delta_required": False,
    }
    path.write_text(json.dumps(payload))
    assert s12.load_gate_routing(path).solver_for("heston") == "pde"


def _write_adi_greek_decision(tmp_path, *, quick=False, routes=None):
    routes = routes or {
        "heston": "pde",
        "heston_slv": "excluded_greek_unresolved",
    }
    runtime = {"python_version": "test", "numpy_version": "test"}
    implementation_hash = "1" * 64
    reference_seeds = {
        "schema11_parent": {"parent_schema9": 20260807},
        "aggregate_primary_refresh": 20260811,
        "aggregate_middle_control": 20260812,
    }
    run_configuration = {
        "schema_version": s12.ADI_GREEK_DECISION_SCHEMA_VERSION,
        "certification_mode": "aggregate_only_amendment",
        "implementation_sha256": implementation_hash,
        "runtime_environment": runtime,
        "reference_seeds": reference_seeds,
        "slv_spot_bridge_strata": 8,
    }

    class TestCertification:
        SCHEMA_VERSION = s12.ADI_GREEK_DECISION_SCHEMA_VERSION

        @staticmethod
        def _canonical_sha256(value):
            text = json.dumps(value, sort_keys=True, separators=(",", ":"))
            return hashlib.sha256(text.encode()).hexdigest()

        @classmethod
        def _projected_evidence_sha256(cls, payload):
            unsigned = dict(payload)
            unsigned.pop("evidence_sha256", None)
            return cls._canonical_sha256(unsigned)

        @staticmethod
        def implementation_sha256():
            return implementation_hash

        @classmethod
        def validate_payload(cls, payload):
            if payload.get("implementation_sha256") != implementation_hash:
                raise ValueError("evidence does not match the live implementation")
            if cls._canonical_sha256(payload.get("run_configuration")) != payload.get(
                "run_configuration_sha256"
            ):
                raise ValueError("run configuration hash mismatch")
            if payload.get("reference_seeds") != reference_seeds or payload.get(
                "run_configuration", {}
            ).get("reference_seeds") != reference_seeds:
                raise ValueError("reference seed metadata mismatch")

        @classmethod
        def build_decision_payload(cls, payload):
            cls.validate_payload(payload)
            decision = {
                "schema_version": cls.SCHEMA_VERSION,
                "study": payload["study"],
                "certification_mode": payload["certification_mode"],
                "profile": payload["profile"],
                "quick": payload["quick"],
                "evidence_sha256": payload["evidence_sha256"],
                "implementation_sha256": payload["implementation_sha256"],
                "run_configuration_sha256": payload["run_configuration_sha256"],
                "run_configuration": payload["run_configuration"],
                "runtime_environment": payload["runtime_environment"],
                "production_engine_controls": payload[
                    "production_engine_controls"
                ],
                "parent_certificate": {},
                "aggregate_reference_sha256": cls._canonical_sha256({}),
                "aggregate_rows_sha256": cls._canonical_sha256({}),
                "added_work": payload["added_work"],
                "decisions": payload["decisions"],
            }
            decision["decision_sha256"] = cls._canonical_sha256(decision)
            return decision

    certification = TestCertification
    s12._STAGE17 = certification
    decisions = {
        variant: {
            "route": route,
            "reason": f"{variant} reason",
            "cell_status": "PASS",
            "anchor_status": "PASS",
            "evidence_complete": route == "pde",
            "missing_anchors": [],
            "missing_cases": [],
            "sampling_complete": True,
            "aggregate_common_scrambles": 128,
            "delta_bias": {
                "status": "PASS",
                "estimate_difference": 0.0,
                "interval": [0.0, 0.0],
            },
        }
        for variant, route in routes.items()
    }
    evidence = {
        "schema_version": s12.ADI_GREEK_DECISION_SCHEMA_VERSION,
        "study": "adi_2d_snowball_greek_certification",
        "certification_mode": "aggregate_only_amendment",
        "profile": "production test fixture",
        "quick": False,
        "implementation_sha256": implementation_hash,
        "run_configuration_sha256": certification._canonical_sha256(run_configuration),
        "run_configuration": run_configuration,
        "runtime_environment": runtime,
        "reference_seeds": reference_seeds,
        "production_engine_controls": s12.ADI_2D_PRODUCTION_ENGINE_CONTROLS,
        "added_work": {"pde_solves": 0},
        "decisions": decisions,
    }
    evidence["evidence_sha256"] = certification._projected_evidence_sha256(evidence)
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "adi_greek_certification.json").write_text(json.dumps(evidence))
    decision = certification.build_decision_payload(evidence)
    (tmp_path / "adi_greek_certification_decision.json").write_text(
        json.dumps(decision)
    )
    path = tmp_path / "adi_greek_certification_decision.json"
    if quick:
        payload = json.loads(path.read_text())
        payload["quick"] = True
        payload.pop("decision_sha256")
        payload["decision_sha256"] = certification._canonical_sha256(payload)
        path.write_text(json.dumps(payload))
    return path


def _reseal_adi_greek_decision(path):
    payload = json.loads(path.read_text())
    payload.pop("decision_sha256", None)
    payload["decision_sha256"] = s12.stage17()._canonical_sha256(payload)
    path.write_text(json.dumps(payload))


def test_adi_greek_routing_requires_sibling_full_evidence(tmp_path):
    path = _write_adi_greek_decision(tmp_path)
    path.with_name("adi_greek_certification.json").unlink()

    with pytest.raises(ValidationError, match="sibling full evidence"):
        s12.load_adi_greek_routing(path)


def test_adi_greek_routing_rejects_resealed_decision_profile_tampering(
    tmp_path,
):
    path = _write_adi_greek_decision(tmp_path)
    payload = json.loads(path.read_text())
    payload["run_configuration"]["slv_spot_bridge_strata"] = 1
    payload["run_configuration_sha256"] = s12.stage17()._canonical_sha256(
        payload["run_configuration"]
    )
    path.write_text(json.dumps(payload))
    _reseal_adi_greek_decision(path)

    with pytest.raises(ValidationError, match="validated sibling evidence"):
        s12.load_adi_greek_routing(path)


def test_adi_greek_routing_rejects_rehashed_full_evidence_tampering(tmp_path):
    path = _write_adi_greek_decision(tmp_path)
    certification = s12.stage17()
    evidence_path = path.with_name("adi_greek_certification.json")
    evidence = json.loads(evidence_path.read_text())
    evidence["run_configuration"]["reference_seeds"][
        "aggregate_middle_control"
    ] = 1
    evidence["run_configuration_sha256"] = certification._canonical_sha256(
        evidence["run_configuration"]
    )
    evidence.pop("evidence_sha256")
    evidence["evidence_sha256"] = certification._projected_evidence_sha256(evidence)
    evidence_path.write_text(json.dumps(evidence))
    decision = json.loads(path.read_text())
    decision["evidence_sha256"] = evidence["evidence_sha256"]
    path.write_text(json.dumps(decision))
    _reseal_adi_greek_decision(path)

    with pytest.raises(ValidationError, match="reference seed metadata"):
        s12.load_adi_greek_routing(path)


def test_adi_greek_routing_rejects_stale_economic_scale_schema(tmp_path):
    path = _write_adi_greek_decision(tmp_path)
    payload = json.loads(path.read_text())
    payload["schema_version"] = 1
    path.write_text(json.dumps(payload))

    with pytest.raises(ValidationError, match="stale schema"):
        s12.load_adi_greek_routing(path)


def test_adi_greek_routing_rejects_quick_evidence(tmp_path):
    path = _write_adi_greek_decision(tmp_path, quick=True)

    with pytest.raises(ValidationError, match="quick/non-production"):
        s12.load_adi_greek_routing(path)


def test_adi_greek_routing_rejects_incomplete_pde_admission(tmp_path):
    path = _write_adi_greek_decision(tmp_path)
    payload = json.loads(path.read_text())
    payload["decisions"]["heston"]["evidence_complete"] = False
    path.write_text(json.dumps(payload))
    _reseal_adi_greek_decision(path)

    with pytest.raises(ValidationError, match="validated sibling evidence"):
        s12.load_adi_greek_routing(path)


def test_adi_greek_routing_rejects_decision_tampering(tmp_path):
    path = _write_adi_greek_decision(tmp_path)
    payload = json.loads(path.read_text())
    payload["decisions"]["heston_slv"]["route"] = "pde"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValidationError, match="validated sibling evidence"):
        s12.load_adi_greek_routing(path)


def test_adi_greek_routing_rejects_stale_live_implementation(tmp_path):
    path = _write_adi_greek_decision(tmp_path)
    certification = s12.stage17()
    evidence_path = path.with_name("adi_greek_certification.json")
    evidence = json.loads(evidence_path.read_text())
    evidence["implementation_sha256"] = "d" * 64
    evidence["run_configuration"]["implementation_sha256"] = "d" * 64
    evidence["run_configuration_sha256"] = certification._canonical_sha256(
        evidence["run_configuration"]
    )
    evidence.pop("evidence_sha256")
    evidence["evidence_sha256"] = certification._projected_evidence_sha256(evidence)
    evidence_path.write_text(json.dumps(evidence))
    decision = json.loads(path.read_text())
    decision["evidence_sha256"] = evidence["evidence_sha256"]
    path.write_text(json.dumps(decision))
    _reseal_adi_greek_decision(path)

    with pytest.raises(ValidationError, match="live implementation"):
        s12.load_adi_greek_routing(path)


def test_adi_greek_admission_never_substitutes_an_mc_2d_route(tmp_path):
    greek_gate = s12.load_adi_greek_routing(_write_adi_greek_decision(tmp_path))
    pv_gate = _routing(heston="mc", heston_slv="pde")

    admitted, excluded = s12.apply_adi_greek_admission(
        ["flat_bsm", "heston", "heston_slv"],
        pv_gate,
        greek_gate,
    )

    assert admitted == ["flat_bsm"]
    assert "Stage 11 PV route is 'mc'" in excluded["heston"]
    assert excluded["heston_slv"] == "heston_slv reason"


def test_adi_greek_admission_requires_both_pv_and_greek_pde_pass(tmp_path):
    path = _write_adi_greek_decision(
        tmp_path,
        routes={"heston": "pde", "heston_slv": "pde"},
    )
    greek_gate = s12.load_adi_greek_routing(path)
    pv_gate = _routing(heston="pde", heston_slv="pde")

    admitted, excluded = s12.apply_adi_greek_admission(
        ["heston", "heston_slv"],
        pv_gate,
        greek_gate,
    )

    assert admitted == ["heston", "heston_slv"]
    assert excluded == {}


MV_CERTIFICATE = (
    ROOT
    / "docs/modelvalidation/certificates/adi2d-snowball-greeks"
    / "2026-08-19/certificate.json"
)


def test_greek_routing_reads_the_committed_modelvalidation_certificate():
    """Stage 12's Greek authority is the certificate, not the raw stage-17 rows.

    The stage-17 payloads are 14.4 MB of Monte-Carlo row dumps and are
    deliberately NOT committed, so routing from them only ever worked on a
    machine that happened to hold them. The modelvalidation certificate IS
    committed, carries its own recomputable digest, and is re-verified on every
    commit by test_banked_certificates.py -- which re-runs both ADI solvers
    over all fourteen banked cells and checks they still produce the certified
    numbers. That is a strictly better trust root for a routing decision.
    """
    routing = s12.load_adi_greek_routing_from_certificate(MV_CERTIFICATE)

    assert routing.route_for("heston") == "pde"
    assert routing.route_for("heston_slv") == "pde"
    assert routing.evidence_sha256


def test_a_tampered_certificate_is_refused(tmp_path):
    """Fail closed: the digest is recomputed, not trusted."""
    payload = json.loads(MV_CERTIFICATE.read_text())
    payload["cells"][0]["verdict"] = "FAIL"  # digest no longer describes content
    tampered = tmp_path / "certificate.json"
    tampered.write_text(json.dumps(payload))

    with pytest.raises(Exception):
        s12.load_adi_greek_routing_from_certificate(tampered)


@pytest.mark.skipif(
    not s12.DEFAULT_GATE_DECISION.exists(), reason="gate decision not produced"
)
def test_recorded_gate_decision_admits_both_2d_pv_ladders():
    routing = s12.load_gate_routing(s12.DEFAULT_GATE_DECISION)
    assert routing.solver_for("heston") == "pde"
    assert routing.solver_for("heston_slv") == "pde"
    assert routing.evidence_sha256, "run manifest must be able to cite the evidence"


# ---------------------------------------------------------------------------
# Fair-coupon root finder (Gate G4)
# ---------------------------------------------------------------------------


def test_affine_root_is_found_in_one_step():
    """PV is affine in the coupon, so false position must land immediately."""
    calls = []

    def pv(coupon):
        calls.append(coupon)
        return -3_606_847.0 + coupon * 23_890_371.0

    result = s12.solve_affine_root(
        pv, lower=0.0, upper=0.8, tolerance=50.0, max_iterations=60
    )
    assert result.solved
    assert result.iterations == 1
    assert len(calls) == 3  # two anchors + the (exact) affine step
    assert abs(pv(result.coupon)) <= 50.0


def test_nonlinear_root_still_converges():
    """A non-affine PV must still converge - the method stays bracketing."""
    result = s12.solve_affine_root(
        lambda x: x**3 - 0.02,
        lower=0.0,
        upper=0.8,
        tolerance=1e-9,
        max_iterations=200,
    )
    assert result.solved
    assert result.coupon == pytest.approx(0.02 ** (1 / 3), rel=1e-4)


def test_unbracketed_root_raises_rather_than_returning_a_boundary():
    with pytest.raises(ValidationError, match="not bracketed"):
        s12.solve_affine_root(
            lambda x: 1.0e9, lower=0.0, upper=0.8, tolerance=1.0, max_iterations=10
        )


def test_flat_function_raises_instead_of_dividing_by_zero():
    """Equal endpoint values would make the false-position step divide by 0."""
    with pytest.raises(ValidationError, match="degenerate"):
        s12.solve_affine_root(
            lambda x: 0.0 if x > 0 else -0.0,  # brackets, but with zero slope
            lower=0.0,
            upper=0.8,
            tolerance=-1.0,  # unsatisfiable, so the solver must reach the guard
            max_iterations=5,
        )


def test_exhausted_budget_raises_and_names_gate_g4():
    with pytest.raises(ValidationError, match="Gate G4"):
        s12.solve_affine_root(
            lambda x: x**3 - 0.02,
            lower=0.0,
            upper=0.8,
            tolerance=1e-18,
            max_iterations=2,
        )


def test_endpoint_root_is_accepted_without_iterating():
    result = s12.solve_affine_root(
        lambda x: x - 0.0, lower=0.0, upper=0.8, tolerance=1e-6, max_iterations=10
    )
    assert result.solved and result.iterations == 0


# ---------------------------------------------------------------------------
# Engine configuration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", s12.VARIANTS)
def test_engine_config_matches_the_variant_spec(variant):
    routing = _routing(heston="mc", heston_slv="mc")
    cfg = s12.make_engine_config(variant, routing=routing)
    spec_ = s12.VARIANT_SPECS[variant]
    assert cfg.vol_model == spec_.vol_model
    assert cfg.vol_source == spec_.vol_source
    assert cfg.surface_vol_mode == spec_.surface_vol_mode


def test_every_variant_keeps_the_same_deterministic_reporting_engines():
    """Surface / event-stats engines must be comparable across variants."""
    routing = _routing(heston="mc", heston_slv="mc")
    types = {
        v: s12.make_engine_config(v, routing=routing).resolve_event_stats_engine_type()
        for v in s12.VARIANTS
    }
    assert set(types.values()) == {EngineType.PDE}


def test_mc_method_does_not_leak_into_the_pde_engine_slot():
    """The RQMC method belongs to the vol-model engine, not the PDE engine.

    Putting MonteCarloMethod in ``method`` makes the PDE surface/event-stats
    engines unconstructible ("Invalid method type"), which is why the config
    carries a separate ``vol_model_mc_method`` slot.
    """
    cfg = s12.make_engine_config("heston", routing=_routing(heston="mc"))
    assert cfg.method is None
    assert cfg.resolve_vol_model_mc_method() == MonteCarloMethod.RANDOMIZED_QUASI


def test_bsm_variants_request_no_calibration():
    routing = _routing(heston="mc", heston_slv="mc")
    for variant in ("flat_bsm", "ts_bsm"):
        assert not s12.VARIANT_SPECS[variant].uses_calibration()
    for variant in ("localvol", "heston", "heston_slv"):
        assert s12.VARIANT_SPECS[variant].uses_calibration()
        cfg = s12.make_engine_config(
            variant, routing=routing, calibration_cache_dir=Path("/tmp/cache")
        )
        assert cfg.vol_model_calibration.cache_dir == "/tmp/cache"


def test_quick_mode_only_shrinks_monte_carlo_effort():
    routing = _routing(heston="mc")
    full = s12.make_engine_config("heston", routing=routing, quick=False)
    quick = s12.make_engine_config("heston", routing=routing, quick=True)
    assert quick.mc_params.num_paths < full.mc_params.num_paths
    assert quick.vol_model == full.vol_model
    assert quick.surface_vol_mode == full.surface_vol_mode


def test_mc_params_are_fixed_cost_and_fixed_seed():
    """The gate reference is a fixed batch count at a fixed seed, not adaptive."""
    params = s12.make_mc_params(8192, 16, 20260723)
    assert params.num_paths == 8192
    assert params.rqmc_min_batches == params.rqmc_max_batches == 16
    assert params.seed == 20260723
    assert not params.use_antithetic


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


def test_cost_model_uses_the_agreed_cffex_parameters():
    model = s12.make_cost_model(True)
    assert model.proportional_rate == pytest.approx(5e-5)
    assert model.spread_bps == pytest.approx(1.0)
    assert model.fixed_commission == 0.0


def test_costs_can_be_disabled_for_a_no_cost_comparison():
    from quantark.backtest.transaction_costs import ZeroCostModel

    assert isinstance(s12.make_cost_model(False), ZeroCostModel)


def test_costs_are_actually_charged_on_a_futures_trade():
    model = s12.make_cost_model(True)
    cost = model.calculate_cost(
        quantity=10.0,
        price=6000.0,
        notional=10 * 6000.0 * 200.0,
        instrument_type="futures",
        trade_type="buy",
    )
    assert cost > 0.0


# ---------------------------------------------------------------------------
# Run classification
# ---------------------------------------------------------------------------


class _StubResults:
    """Mirrors the engine's real frames: lifecycle flags live in states_df.

    ``actions_df`` carries ``action_type`` with LifecycleEventType VALUES
    ("KO"/"KI"/"COUPON"/"MATURITY") - not a free-text ``action`` column.
    """

    def __init__(self, actions, *, knocked_out=False, knocked_in=False, matured=False):
        import pandas as pd

        self.actions_df = pd.DataFrame(actions)
        self.states_df = pd.DataFrame(
            {
                "total_pnl": [0.0, 1.0],
                "knocked_out": [False, knocked_out],
                "knocked_in": [False, knocked_in],
                "matured": [False, matured],
                "alive": [True, not (knocked_out or matured)],
            }
        )

    def get_summary(self):
        return {"total_pnl": 1.0}


def _summarize(actions, last_date, terms, **flags):
    import pandas as pd

    return s12.summarize_run(
        results=_StubResults(actions, **flags),
        terms=terms,
        inception=terms.inception,
        variant="flat_bsm",
        engine_config=s12.make_engine_config("flat_bsm", routing=_routing()),
        dates=pd.DatetimeIndex(
            [pd.Timestamp(terms.inception), pd.Timestamp(last_date)]
        ),
        coupon=0.15,
        notional=50_000_000.0,
        calibration_records=[],
        elapsed=1.0,
    )


def test_run_reaching_data_end_before_maturity_is_censored(terms):
    summary = _summarize([], date(2024, 6, 3), terms)
    assert summary["lifecycle"]["censored_at_data_end"]
    assert not summary["lifecycle"]["matured"]
    assert not summary["lifecycle"]["knocked_out"]


def test_run_reaching_maturity_is_not_censored(terms):
    summary = _summarize([], terms.maturity_date, terms, matured=True)
    assert summary["lifecycle"]["matured"]
    assert not summary["lifecycle"]["censored_at_data_end"]


def test_knocked_out_run_is_not_censored_and_not_matured(terms):
    """KO ends the trade early; it must not be double-counted as matured."""
    actions = [{"action_type": "KO", "payoff": 1.0}]
    summary = _summarize(actions, date(2023, 9, 4), terms, knocked_out=True)
    assert summary["lifecycle"]["knocked_out"]
    assert not summary["lifecycle"]["censored_at_data_end"]
    assert not summary["lifecycle"]["matured"]


def test_knock_out_date_is_read_from_the_action_log(terms):
    import pandas as pd

    actions = pd.DataFrame(
        [{"action_type": "COUPON"}, {"action_type": "KO"}],
        index=pd.to_datetime(["2023-08-04", "2023-09-04"]),
    )
    summary = s12.summarize_run(
        results=_StubResults(actions, knocked_out=True),
        terms=terms,
        inception=terms.inception,
        variant="flat_bsm",
        engine_config=s12.make_engine_config("flat_bsm", routing=_routing()),
        dates=pd.DatetimeIndex(
            [pd.Timestamp("2023-05-04"), pd.Timestamp("2023-09-04")]
        ),
        coupon=0.15,
        notional=50_000_000.0,
        calibration_records=[],
        elapsed=1.0,
    )
    assert "2023-09-04" in str(summary["lifecycle"]["ko_date"])


def test_lifecycle_flags_come_from_states_not_action_labels(terms):
    """An empty action log must not silently mean 'never knocked out'."""
    summary = _summarize([], date(2023, 9, 4), terms, knocked_out=True)
    assert summary["lifecycle"]["knocked_out"]
    assert summary["lifecycle"]["ko_date"] is None  # no action row to date it


def test_summary_records_the_routing_actually_used(terms):
    summary = _summarize([], date(2024, 6, 3), terms)
    assert summary["vol_model"] == "bsm"
    assert summary["vol_model_solver"] in ("pde", "mc")
    assert summary["surface_vol_mode"] == "flat_atm_remaining"
    assert summary["coupon"] == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# Task fan-out
# ---------------------------------------------------------------------------


def test_tasks_are_the_full_inception_by_variant_cross_product():
    prepared = [
        {
            "inception": "2023-05-04",
            "initial_spot": 6733.97,
            "coupon": 0.15,
            "maturity_date": "2026-05-04",
        },
        {
            "inception": "2023-06-01",
            "initial_spot": 6500.0,
            "coupon": 0.16,
            "maturity_date": "2026-06-01",
        },
    ]
    tasks = s12.build_tasks(
        prepared=prepared,
        variants=["flat_bsm", "heston"],
        routing=_routing(heston="mc"),
        history_dir=HISTORY_DIR,
        out_dir=Path("/tmp/out"),
        data_end=date(2026, 7, 22),
        rate=0.02,
        notional=50_000_000.0,
        costs_enabled=True,
        quick=False,
        calculate_surfaces=False,
        calculate_event_probabilities=True,
    )
    assert len(tasks) == 4
    assert {(t["inception"], t["variant"]) for t in tasks} == {
        ("2023-05-04", "flat_bsm"),
        ("2023-05-04", "heston"),
        ("2023-06-01", "flat_bsm"),
        ("2023-06-01", "heston"),
    }
    # Every variant of an inception must share ONE coupon (apples-to-apples).
    by_inception = {}
    for task in tasks:
        by_inception.setdefault(task["inception"], set()).add(task["coupon"])
    assert all(len(coupons) == 1 for coupons in by_inception.values())


def test_window_end_is_clipped_to_the_data_end():
    prepared = [
        {
            "inception": "2025-07-01",
            "initial_spot": 6000.0,
            "coupon": 0.15,
            "maturity_date": "2028-07-03",
        }
    ]
    tasks = s12.build_tasks(
        prepared=prepared,
        variants=["flat_bsm"],
        routing=_routing(),
        history_dir=HISTORY_DIR,
        out_dir=Path("/tmp/out"),
        data_end=date(2026, 7, 22),
        rate=0.02,
        notional=50_000_000.0,
        costs_enabled=True,
        quick=False,
        calculate_surfaces=False,
        calculate_event_probabilities=True,
    )
    assert tasks[0]["window_end"] == "2026-07-22"


def test_every_task_carries_the_gate_provenance():
    tasks = s12.build_tasks(
        prepared=[
            {
                "inception": "2023-05-04",
                "initial_spot": 6733.97,
                "coupon": 0.15,
                "maturity_date": "2026-05-04",
            }
        ],
        variants=["heston"],
        routing=_routing(heston="mc"),
        history_dir=HISTORY_DIR,
        out_dir=Path("/tmp/out"),
        data_end=date(2026, 7, 22),
        rate=0.02,
        notional=50_000_000.0,
        costs_enabled=True,
        quick=False,
        calculate_surfaces=False,
        calculate_event_probabilities=True,
    )
    assert tasks[0]["gate"]["evidence_sha256"] == "deadbeef"
    assert tasks[0]["gate"]["routes"]["heston"] == "mc"


def test_failed_runs_are_recorded_not_swallowed():
    def boom(task):
        raise RuntimeError("engine exploded")

    original = s12.run_one
    s12.run_one = boom
    try:
        summaries, failures = s12.execute_tasks(
            [{"inception": "2023-05-04", "variant": "flat_bsm"}], workers=1
        )
    finally:
        s12.run_one = original
    assert summaries == []
    assert len(failures) == 1
    assert failures[0]["error_type"] == "RuntimeError"
    assert "engine exploded" in failures[0]["error"]
    assert failures[0]["traceback"]


# ---------------------------------------------------------------------------
# Resume checkpointing
#
# The fleet is a ~4-day run whose cells each write run_summary.json as they
# finish, so an interruption must cost the driver's bookkeeping and nothing
# else.  What these tests protect is not "does it skip work" -- that part is a
# dict lookup -- but the fail-closed half: runs/ accumulates cells from every
# past fleet, and reusing one that a superseded gate decision produced would
# put un-computed, un-certified numbers into a fresh manifest.
# ---------------------------------------------------------------------------


def _resume_tasks(out_dir, *, routing=None, **overrides):
    kwargs = dict(
        prepared=[
            {
                "inception": "2023-05-04",
                "initial_spot": 6733.97,
                "coupon": 0.15,
                "maturity_date": "2026-05-04",
            }
        ],
        variants=["localvol"],
        routing=routing if routing is not None else _routing(),
        history_dir=HISTORY_DIR,
        out_dir=Path(out_dir),
        data_end=date(2026, 7, 22),
        rate=0.02,
        notional=50_000_000.0,
        costs_enabled=True,
        quick=False,
        calculate_surfaces=False,
        calculate_event_probabilities=True,
        code_sha256="code-v1",
        data_sha256="data-v1",
    )
    kwargs.update(overrides)
    return s12.build_tasks(**kwargs)


def _write_cell(out_dir, task, *, fingerprint="match", **summary_overrides):
    """Drop a completed cell on disk the way a finished worker leaves one."""
    run_dir = Path(out_dir) / "runs" / task["inception"] / task["variant"]
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": s12.SCHEMA_VERSION,
        "inception": task["inception"],
        "variant": task["variant"],
        "lifecycle": {
            "knocked_out": True,
            "knocked_in": False,
            "ko_date": "2024-01-02",
            "matured": False,
            "censored_at_data_end": False,
        },
        "metrics": {"total_pnl": 1.0},
        "elapsed_seconds": 12.5,
    }
    if fingerprint == "match":
        summary["provenance"] = s12._provenance_payload(task)
    elif fingerprint is not None:
        summary["provenance"] = dict(
            s12._provenance_payload(task), fingerprint_sha256=fingerprint
        )
    summary.update(summary_overrides)
    (run_dir / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return run_dir / "run_summary.json"


def test_every_task_carries_a_result_fingerprint(tmp_path):
    tasks = _resume_tasks(tmp_path)
    assert tasks[0]["fingerprint"]
    assert len(tasks[0]["fingerprint"]) == 64


def test_resume_reuses_a_cell_whose_fingerprint_matches(tmp_path):
    tasks = _resume_tasks(tmp_path)
    _write_cell(tmp_path, tasks[0])
    plan = s12.plan_resume(tasks, out_dir=tmp_path, resume=True)
    assert plan.todo == []
    assert len(plan.reused) == 1
    assert plan.reused[0]["variant"] == "localvol"
    assert plan.recomputed == []


def test_resume_recomputes_a_cell_from_a_superseded_gate_decision(tmp_path):
    """The defect this exists to prevent: silently banking stale numbers."""
    tasks = _resume_tasks(tmp_path)
    _write_cell(tmp_path, tasks[0])

    base = _routing()
    resealed = s12.GateRouting(
        decision_path=base.decision_path,
        evidence_sha256="a-newer-gate",
        routes=base.routes,
        pde_params=base.pde_params,
        mc_params=base.mc_params,
    )
    fresh = _resume_tasks(tmp_path, routing=resealed)
    assert fresh[0]["fingerprint"] != tasks[0]["fingerprint"]

    plan = s12.plan_resume(fresh, out_dir=tmp_path, resume=True)
    assert plan.reused == []
    assert len(plan.todo) == 1
    assert "stale" in plan.recomputed[0]["reason"]


def test_resume_recomputes_when_the_code_changed(tmp_path):
    tasks = _resume_tasks(tmp_path)
    _write_cell(tmp_path, tasks[0])
    fresh = _resume_tasks(tmp_path, code_sha256="code-v2")
    plan = s12.plan_resume(fresh, out_dir=tmp_path, resume=True)
    assert plan.reused == []
    assert len(plan.todo) == 1


def test_resume_recomputes_when_the_market_data_changed(tmp_path):
    tasks = _resume_tasks(tmp_path)
    _write_cell(tmp_path, tasks[0])
    fresh = _resume_tasks(tmp_path, data_sha256="data-v2")
    plan = s12.plan_resume(fresh, out_dir=tmp_path, resume=True)
    assert plan.reused == []
    assert len(plan.todo) == 1


def test_the_fingerprint_ignores_where_results_are_written(tmp_path):
    """Moving the output tree must not invalidate a cohort."""
    here = _resume_tasks(tmp_path / "a")
    there = _resume_tasks(tmp_path / "b")
    assert here[0]["fingerprint"] == there[0]["fingerprint"]


def test_an_unstamped_cell_is_recomputed_by_default(tmp_path):
    tasks = _resume_tasks(tmp_path)
    _write_cell(tmp_path, tasks[0], fingerprint=None)
    plan = s12.plan_resume(tasks, out_dir=tmp_path, resume=True)
    assert plan.reused == []
    assert len(plan.todo) == 1
    assert "unstamped" in plan.recomputed[0]["reason"]


def test_adopt_unstamped_reuses_the_cell_and_names_it(tmp_path):
    tasks = _resume_tasks(tmp_path)
    path = _write_cell(tmp_path, tasks[0], fingerprint=None)
    plan = s12.plan_resume(
        tasks,
        out_dir=tmp_path,
        resume=True,
        adopt_unstamped_since=os.path.getmtime(path) - 60,
    )
    assert plan.todo == []
    assert len(plan.reused) == 1
    assert plan.adopted[0]["inception"] == "2023-05-04"
    assert plan.adopted[0]["variant"] == "localvol"
    assert plan.adopted[0]["run_summary_mtime"]
    provenance = plan.reused[0]["provenance"]
    assert provenance["adopted_without_fingerprint"] is True
    assert provenance["fingerprint_sha256"] is None
    assert provenance["expected_fingerprint_sha256"] == tasks[0]["fingerprint"]


def test_adoption_will_not_reach_back_past_its_bound(tmp_path):
    """The defect a bare --adopt-unstamped shipped with.

    runs/ holds unstamped cells from every fleet that ever wrote there.  An
    unbounded flag adopted two cells a month older than the current stack --
    priced under a superseded gate decision -- because nothing on disk tells
    them apart.  The bound is what the operator actually knows.
    """
    tasks = _resume_tasks(tmp_path)
    path = _write_cell(tmp_path, tasks[0], fingerprint=None)
    stack_started = os.path.getmtime(path) + 60  # cell predates this stack
    plan = s12.plan_resume(
        tasks, out_dir=tmp_path, resume=True, adopt_unstamped_since=stack_started
    )
    assert plan.reused == []
    assert plan.adopted == []
    assert len(plan.todo) == 1
    assert "before the adoption bound" in plan.recomputed[0]["reason"]


def test_adopting_never_stamps_the_cell_on_disk(tmp_path):
    """We did not verify it, so we must not leave an artifact claiming we did."""
    tasks = _resume_tasks(tmp_path)
    path = _write_cell(tmp_path, tasks[0], fingerprint=None)
    s12.plan_resume(
        tasks,
        out_dir=tmp_path,
        resume=True,
        adopt_unstamped_since=os.path.getmtime(path) - 60,
    )
    assert "provenance" not in json.loads(path.read_text())


def test_a_cell_written_for_another_run_is_recomputed(tmp_path):
    tasks = _resume_tasks(tmp_path)
    _write_cell(tmp_path, tasks[0], variant="heston")
    plan = s12.plan_resume(tasks, out_dir=tmp_path, resume=True)
    assert plan.reused == []
    assert len(plan.todo) == 1


def test_a_corrupt_summary_is_recomputed_not_raised(tmp_path):
    tasks = _resume_tasks(tmp_path)
    run_dir = tmp_path / "runs" / "2023-05-04" / "localvol"
    run_dir.mkdir(parents=True)
    (run_dir / "run_summary.json").write_text("{ truncated", encoding="utf-8")
    plan = s12.plan_resume(tasks, out_dir=tmp_path, resume=True)
    assert len(plan.todo) == 1


def test_resume_without_a_code_fingerprint_fails_closed(tmp_path):
    tasks = _resume_tasks(tmp_path, code_sha256=None)
    _write_cell(tmp_path, tasks[0])
    with pytest.raises(ValidationError, match="code fingerprint"):
        s12.plan_resume(tasks, out_dir=tmp_path, resume=True)


def test_disabled_resume_runs_everything_but_says_what_it_could_reuse(tmp_path):
    tasks = _resume_tasks(tmp_path)
    _write_cell(tmp_path, tasks[0])
    plan = s12.plan_resume(tasks, out_dir=tmp_path, resume=False)
    assert len(plan.todo) == 1
    assert plan.reused == []
    assert plan.resumable_when_disabled == [
        {"inception": "2023-05-04", "variant": "localvol"}
    ]


def test_manifest_records_the_resume_split(tmp_path):
    tasks = _resume_tasks(tmp_path)
    path = _write_cell(tmp_path, tasks[0], fingerprint=None)
    plan = s12.plan_resume(
        tasks,
        out_dir=tmp_path,
        resume=True,
        adopt_unstamped_since=os.path.getmtime(path) - 60,
    )
    manifest = s12.build_run_manifest(
        cfg={
            "variants": ["localvol"],
            "notional": 50_000_000.0,
            "costs_enabled": True,
            "resume": True,
            "adopt_unstamped_since": "2026-08-27T12:00",
            "code_sha256": "code-v1",
            "data_sha256": "data-v1",
        },
        routing=_routing(),
        prepared=[{"inception": "2023-05-04"}],
        summaries=plan.reused,
        failures=[],
        elapsed=1.0,
        plan=plan,
    )
    resume = manifest["resume"]
    assert resume["enabled"] is True
    assert resume["reused"] == 1
    assert resume["computed_now"] == 0
    assert resume["adopted_without_fingerprint"][0]["variant"] == "localvol"
    assert resume["adopt_unstamped_since"] == "2026-08-27T12:00"
    assert resume["code_sha256"] == "code-v1"


def test_manifest_of_a_plain_run_reports_no_reuse(tmp_path):
    manifest = s12.build_run_manifest(
        cfg={"variants": ["localvol"], "notional": 1.0, "costs_enabled": True},
        routing=_routing(),
        prepared=[{"inception": "2023-05-04"}],
        summaries=[],
        failures=[],
        elapsed=1.0,
    )
    assert manifest["resume"]["enabled"] is False
    assert manifest["resume"]["reused"] == 0


# --- the data fingerprint, on the real history tree -------------------------


def test_data_fingerprint_ignores_rows_beyond_the_pinned_window():
    """The property that makes resume usable at all.

    The daily calibration pipeline appends a spot row every weekday.  If the
    data digest covered the whole cache it would change every morning, every
    cell would read as stale, and --resume would never reuse anything.
    """
    import pandas as pd

    spot = s12.load_spot_frame(HISTORY_DIR)
    futures = s12.load_futures_frame(HISTORY_DIR)
    pinned = date(2026, 7, 31)
    before = s12.compute_data_fingerprint(
        history_dir=HISTORY_DIR, spot=spot, futures=futures, data_end=pinned
    )
    tomorrow = spot.tail(1).assign(date=pd.Timestamp("2027-03-01"), spot=1.0)
    after = s12.compute_data_fingerprint(
        history_dir=HISTORY_DIR,
        spot=pd.concat([spot, tomorrow], ignore_index=True),
        futures=futures,
        data_end=pinned,
    )
    assert before == after


def test_data_fingerprint_notices_a_rewritten_row_inside_the_window():
    import pandas as pd

    spot = s12.load_spot_frame(HISTORY_DIR)
    futures = s12.load_futures_frame(HISTORY_DIR)
    pinned = date(2026, 7, 31)
    before = s12.compute_data_fingerprint(
        history_dir=HISTORY_DIR, spot=spot, futures=futures, data_end=pinned
    )
    tampered = spot.copy()
    tampered.loc[0, "spot"] = float(tampered.loc[0, "spot"]) + 1.0
    after = s12.compute_data_fingerprint(
        history_dir=HISTORY_DIR, spot=tampered, futures=futures, data_end=pinned
    )
    assert before != after


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=str(repo), check=True, capture_output=True)


@pytest.fixture
def code_repo(tmp_path):
    """A miniature repo shaped like this one: pricing code, data, docs."""
    repo = tmp_path / "repo"
    (repo / "quantark").mkdir(parents=True)
    (repo / "example/mo_volmodels/data").mkdir(parents=True)
    (repo / "docs").mkdir(parents=True)
    (repo / "quantark" / "engine.py").write_text("VERSION = 1\n", encoding="utf-8")
    (repo / "example/mo_volmodels/12_fleet.py").write_text("X = 1\n", encoding="utf-8")
    (repo / "example/mo_volmodels/data/spot.csv").write_text("d,s\n", encoding="utf-8")
    (repo / "docs" / "note.md").write_text("hello\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    return repo


def test_code_fingerprint_sees_an_uncommitted_edit(code_repo):
    """Stage 11/12 are routinely run dirty; committed content alone is blind."""
    before = s12.compute_code_fingerprint(code_repo)
    assert before is not None and len(before) == 64
    (code_repo / "example/mo_volmodels/12_fleet.py").write_text(
        "X = 2\n", encoding="utf-8"
    )
    assert s12.compute_code_fingerprint(code_repo) != before


def test_committing_the_running_edit_does_not_strand_the_fleet(code_repo):
    """The fingerprint is content, not commit state.

    A fleet takes days.  Committing the very edit it is running -- or landing
    unrelated work beside it -- must not turn every finished cell stale.
    """
    (code_repo / "quantark" / "engine.py").write_text("VERSION = 2\n", encoding="utf-8")
    dirty = s12.compute_code_fingerprint(code_repo)
    _git(code_repo, "add", "-A")
    _git(code_repo, "commit", "-qm", "land the edit")
    assert s12.compute_code_fingerprint(code_repo) == dirty


def test_an_unrelated_commit_does_not_move_the_fingerprint(code_repo):
    """docs/ and certificates land on this branch while the fleet runs."""
    before = s12.compute_code_fingerprint(code_repo)
    (code_repo / "docs" / "certificate.md").write_text("banked\n", encoding="utf-8")
    _git(code_repo, "add", "-A")
    _git(code_repo, "commit", "-qm", "docs: bank a certificate")
    assert s12.compute_code_fingerprint(code_repo) == before


def test_market_data_does_not_move_the_code_fingerprint(code_repo):
    """data/ is rewritten daily and has its own window-clipped fingerprint."""
    before = s12.compute_code_fingerprint(code_repo)
    (code_repo / "example/mo_volmodels/data/spot.csv").write_text(
        "d,s\n2026-08-28,1.0\n", encoding="utf-8"
    )
    assert s12.compute_code_fingerprint(code_repo) == before


def test_ignored_files_do_not_move_the_code_fingerprint(code_repo):
    """__pycache__ and local scratch must not invalidate a fleet."""
    (code_repo / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    _git(code_repo, "add", "-A")
    _git(code_repo, "commit", "-qm", "ignore pycache")
    before = s12.compute_code_fingerprint(code_repo)
    cache = code_repo / "quantark" / "__pycache__"
    cache.mkdir()
    (cache / "engine.cpython-311.py").write_text("junk\n", encoding="utf-8")
    assert s12.compute_code_fingerprint(code_repo) == before


def test_a_deleted_source_file_moves_the_fingerprint(code_repo):
    before = s12.compute_code_fingerprint(code_repo)
    (code_repo / "quantark" / "engine.py").unlink()
    assert s12.compute_code_fingerprint(code_repo) != before


def test_code_fingerprint_is_none_outside_a_repository(tmp_path):
    """No git answer means resume cannot tie a cell to its code -- fail closed."""
    assert s12.compute_code_fingerprint(tmp_path) is None


# ---------------------------------------------------------------------------
# Declared scope exclusions
#
# Four consecutive inceptions are declared out of scope for the 2-D variants
# (see SCOPE_EXCLUSIONS).  The fleet must not spend compute attempting them,
# must not record them as failures, and must fail closed if the declaration
# has drifted away from the schedule it describes.
# ---------------------------------------------------------------------------


def _prepared(*inceptions):
    return [
        {
            "inception": inception,
            "initial_spot": 6733.97,
            "coupon": 0.15,
            "maturity_date": "2027-05-04",
        }
        for inception in inceptions
    ]


def _scope_tasks(inceptions, variants):
    return _resume_tasks(
        Path("/tmp/unused"), prepared=_prepared(*inceptions), variants=list(variants)
    )


def test_the_declaration_names_only_two_d_variants():
    """A 1-D variant here would silently shrink an arm the gate fully covers."""
    for exclusion in s12.SCOPE_EXCLUSIONS:
        assert set(exclusion.variants) <= set(s12.TWO_D_VARIANTS)
        # The measurement that justifies it travels with it.
        assert exclusion.achieved > exclusion.target
        assert exclusion.reason


def test_declared_cells_are_dropped_and_recorded(tmp_path):
    declared = s12.SCOPE_EXCLUSIONS[0].inception
    inceptions = ["2023-05-04", declared]
    variants = ["flat_bsm", "heston", "heston_slv"]
    tasks = _scope_tasks(inceptions, variants)
    assert len(tasks) == 6

    scope = s12.apply_scope_exclusions(tasks, prepared=_prepared(*inceptions))
    kept, records = scope.tasks, scope.excluded
    assert scope.unmatched == sorted(
        e.inception for e in s12.SCOPE_EXCLUSIONS if e.inception != declared
    )
    assert len(kept) == 4
    assert {(t["inception"], t["variant"]) for t in kept} == {
        ("2023-05-04", "flat_bsm"),
        ("2023-05-04", "heston"),
        ("2023-05-04", "heston_slv"),
        (declared, "flat_bsm"),          # 1-D arm still runs on that date
    }
    assert {(r["inception"], r["variant"]) for r in records} == {
        (declared, "heston"),
        (declared, "heston_slv"),
    }
    for record in records:
        assert record["achieved_spacing"] > record["target_eps_crit"]
        assert record["reason"]


def test_a_one_d_only_fleet_is_untouched_by_the_declaration():
    """The declaration is about the 2-D grid, not about those dates."""
    inceptions = [s12.SCOPE_EXCLUSIONS[0].inception, "2023-05-04"]
    tasks = _scope_tasks(inceptions, ["flat_bsm", "ts_bsm", "localvol"])
    scope = s12.apply_scope_exclusions(tasks, prepared=_prepared(*inceptions))
    assert len(scope.tasks) == len(tasks)
    assert scope.excluded == []


def test_an_unscheduled_declaration_is_reported_not_fatal():
    """A subset run legitimately misses declared inceptions.

    Failing here would break --max-inceptions and the G3 smoke.  A declared
    entry that matches nothing simply excludes nothing, so the cell is
    attempted and the grid check fails closed as it always did -- visible,
    which is the safe direction.  It is surfaced so a FULL fleet can notice.
    """
    tasks = _scope_tasks(["2023-05-04"], ["heston"])
    scope = s12.apply_scope_exclusions(tasks, prepared=_prepared("2023-05-04"))
    assert scope.excluded == []
    assert len(scope.tasks) == 1
    assert scope.unmatched == sorted(e.inception for e in s12.SCOPE_EXCLUSIONS)


def test_a_declaration_naming_a_one_d_variant_fails_closed():
    tasks = _scope_tasks(["2023-05-04"], ["localvol"])
    bad = (
        s12.ScopeExclusion(
            inception="2023-05-04",
            variants=("localvol",),
            achieved=0.007,
            target=0.003,
            reason="wrong arm",
        ),
    )
    with pytest.raises(ValidationError, match="non-2-D"):
        s12.apply_scope_exclusions(
            tasks, prepared=_prepared("2023-05-04"), exclusions=bad
        )


def test_the_manifest_separates_out_of_scope_from_failed():
    """A declared gap and a broken run are different facts about the result."""
    prepared = _prepared("2023-05-04", "2023-06-01")
    records = [
        {
            "inception": "2023-06-01",
            "variant": "heston",
            "reason": "declared",
            "achieved_spacing": 0.0074,
            "target_eps_crit": 0.003,
        }
    ]
    manifest = s12.build_run_manifest(
        cfg={
            "notional": 50_000_000.0,
            "costs_enabled": True,
            "variants": ["flat_bsm", "heston"],
            "out_dir": "/tmp/out",
        },
        routing=_routing(),
        prepared=prepared,
        summaries=[
            {
                "inception": i,
                "variant": v,
                "lifecycle": {
                    "knocked_out": True,
                    "matured": False,
                    "censored_at_data_end": False,
                },
            }
            for i, v in (
                ("2023-05-04", "flat_bsm"),
                ("2023-05-04", "heston"),
                ("2023-06-01", "flat_bsm"),
            )
        ],
        failures=[],
        elapsed=1.0,
        scope_exclusions=records,
    )
    counts = manifest["counts"]
    assert counts["runs_expected"] == 4          # the full grid stays visible
    assert counts["runs_out_of_scope"] == 1
    assert counts["runs_in_scope"] == 3
    assert counts["runs_completed"] == 3
    assert counts["runs_failed"] == 0
    assert manifest["scope_exclusions"] == records


def test_a_manifest_without_exclusions_still_reports_full_scope():
    prepared = _prepared("2023-05-04")
    manifest = s12.build_run_manifest(
        cfg={
            "notional": 50_000_000.0,
            "costs_enabled": True,
            "variants": ["flat_bsm"],
            "out_dir": "/tmp/out",
        },
        routing=_routing(),
        prepared=prepared,
        summaries=[{
            "inception": "2023-05-04",
            "variant": "flat_bsm",
            "lifecycle": {
                "knocked_out": True, "matured": False,
                "censored_at_data_end": False,
            },
        }],
        failures=[],
        elapsed=1.0,
    )
    assert manifest["scope_exclusions"] == []
    assert manifest["counts"]["runs_out_of_scope"] == 0
    assert manifest["counts"]["runs_in_scope"] == 1
