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
    assert json.loads(json.dumps(resolved_grid_evidence(PDEParams(accuracy="standard")))) == banked


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
