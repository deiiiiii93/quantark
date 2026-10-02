"""The PDE candidates execute the params they record (spec 2 §4.6, decisions 1, 15)."""
import json
from pathlib import Path

import pytest

from quantark.asset.equity.engine.pde.core.components import CrankNicolson
from quantark.asset.equity.engine.pde.grid.config import resolved_grid_evidence
from quantark.asset.equity.param import PDEParams
from quantark.modelvalidation.builders.equity_snowball import SnowballPDECandidate
from quantark.util.exceptions import ValidationError

from modelvalidation.test_variant_case_builders import REPO_ROOT


def _banked_pde_grid(cert_path):
    d = json.loads(Path(cert_path).read_text())
    for cand in d["study"]["candidates"]:
        if str(cand["params"].get("engine", "")).endswith("PDESolver"):
            return cand["params"]["grid"]
    raise AssertionError(f"no PDE candidate in {cert_path}")


@pytest.mark.parametrize("study", ["snowball-flat-bsm", "phoenix-flat-bsm", "ko-reset-flat-bsm"])
def test_resolved_grid_evidence_is_the_banked_record(study):
    certs = sorted((REPO_ROOT / "docs" / "modelvalidation" / "certificates" / study).glob("*/certificate.json"))
    banked = _banked_pde_grid(certs[-1])
    # JSON round trip: bounds is a tuple in memory and a list in the certificate
    assert json.loads(json.dumps(resolved_grid_evidence(PDEParams(accuracy="standard"), day_count=252))) == banked


def _arm(params):
    return SnowballPDECandidate(environment_params={}, product_params={}, quantities=("pv",), params=params)


def test_a_declared_stepper_is_executed_and_recorded():
    cand = _arm({"accuracy": "standard", "stepper": {"theta": 1.0}})
    assert cand._engine("standard").params.stepper == CrankNicolson(theta=1.0)
    assert cand.params()["stepper"] == {"theta": 1.0}


def test_pdeparams_evidence_records_the_components_it_runs_with():
    from quantark.asset.equity.engine.pde.grid import GridConfig
    from quantark.modelvalidation.engine_config import engine_config

    from pde_core.components_fixtures import NearestNodeReadout, PlainStepper

    rec = engine_config(PDEParams(grid=GridConfig(points=300), stepper=CrankNicolson(theta=1.0)))
    assert rec["accuracy"] == "standard" and rec["grid"] == {"points": 300}
    assert rec["stepper"] == {"theta": 1.0, "event_theta": 1.0}
    assert "operator" not in rec and "readout" not in rec          # the built-ins
    assert rec["event_projection"] is None and rec["cache_enabled"] is True
    assert {"bump_size", "bump_config", "bus_days_in_year"} <= set(rec)
    custom = engine_config(PDEParams(stepper=PlainStepper(), readout=NearestNodeReadout()))
    assert custom["stepper"] == "uncertifiable"
    assert isinstance(custom["readout"], str) and custom["readout"] != "uncertifiable"


def test_a_key_from_config_rejects_fails_the_study_at_load():
    with pytest.raises(ValidationError):
        _arm({"accuracy": "standard", "operator": {}})


def test_a_half_step_stepper_records_its_form():
    rec = resolved_grid_evidence(PDEParams(stepper=CrankNicolson(damping="half_step")), day_count=252)
    # continuous KI keeps its (1, 2) counts in the half-step form; the core row resolves (1, 1) (flip f)
    assert rec["damping"] == "half_step" and rec["discrete_monitoring_damping"] == ["half_step", 1, 1]


def test_the_default_records_the_continuous_ki_counts():
    rec = resolved_grid_evidence(PDEParams(), day_count=252)
    assert (rec["terminal_damping_steps"], rec["event_damping_steps"]) == (1, 2) and "damping" not in rec


def test_a_builder_on_a_244_environment_records_244():
    arm = SnowballPDECandidate(environment_params={"bus_days_in_year": 244}, product_params={},
                               quantities=("pv",), params={"accuracy": "standard"})
    assert arm.params()["grid"]["day_count"] == 244


def test_the_banked_record_keeps_its_key_order():
    rec = resolved_grid_evidence(PDEParams(), day_count=252)
    keys = list(rec)
    assert keys.index("day_count") == keys.index("max_steps") + 1


def test_the_default_record_carries_the_discrete_monitoring_row():
    rec = resolved_grid_evidence(PDEParams(), day_count=252)
    assert rec["discrete_monitoring_damping"] == ["half_step", 1, 1]
    assert (rec["terminal_damping_steps"], rec["event_damping_steps"]) == (1, 2) and "damping" not in rec


def test_the_070_configuration_serializes_to_the_banked_record():
    # the 2026-09-28 certificates are the 0.7.0 records (Task 17 banks newer ones beside them)
    from quantark.asset.equity.engine.pde.grid.config import GridConfig
    params = PDEParams(grid=GridConfig(max_steps=5000), stepper=CrankNicolson(
        damping="full_step", terminal_damping_steps=1, event_damping_steps=2))
    for study in ("snowball-flat-bsm", "phoenix-flat-bsm", "ko-reset-flat-bsm"):
        cert = REPO_ROOT / "docs" / "modelvalidation" / "certificates" / study / "2026-09-28" / "certificate.json"
        banked = _banked_pde_grid(cert)
        assert json.loads(json.dumps(resolved_grid_evidence(params, day_count=252))) == banked
