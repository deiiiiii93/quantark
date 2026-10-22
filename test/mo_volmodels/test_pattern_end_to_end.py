"""The mo_volmodels reference pattern, end to end, offline, in a temp directory.

``example/case_studies/mo_volmodels`` is what a user copies to calibrate vol models on a
new market, so it must keep working. This is the one test that guards it. It replaces the
per-stage tests (stages 02-09 and the suite smoke test), which re-ran stage 02 for every
stage and wrote ``*_sample.*`` artifacts next to the scripts, so they raced under xdist and
left the working tree dirty.

The study resolves every input and output through ``HERE`` (its own directory), so the test
copies the scripts and the committed sample snapshot into ``tmp_path`` and runs them there.
Nothing under ``example/`` is read for output or written. Stages 08 and 09 (snowball MC/ADI,
delta hedging) are expensive and are not part of the offline pattern.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "example" / "case_studies" / "mo_volmodels"
SAMPLE_INPUTS = ("mo_snapshot_sample.json", "mo_calibration_diagnostics_sample.json")


def _git_status(path: Path) -> str:
    out = subprocess.run(
        ["git", "status", "--porcelain", "--", str(path)],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    return out.stdout


@pytest.fixture(scope="module")
def study_copy(tmp_path_factory):
    work = tmp_path_factory.mktemp("mo_volmodels")
    for script in STUDY.glob("*.py"):
        shutil.copy(script, work / script.name)
    (work / "data").mkdir()
    (work / "data" / "plots").mkdir()
    for name in SAMPLE_INPUTS:
        shutil.copy(STUDY / "data" / name, work / "data" / name)
    return work


def _run(work: Path, stage: str, *args: str) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT), env.get("PYTHONPATH")) if p)
    proc = subprocess.run(
        [sys.executable, str(work / stage), *args],
        cwd=work, env=env, capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"{stage} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"


def test_offline_pipeline_runs_in_a_temp_directory(study_copy):
    work, data = study_copy, study_copy / "data"
    before = _git_status(STUDY)

    # stage 02: market snapshot -> implied-vol surface
    _run(work, "02_build_iv_surface.py", "--snapshot", "sample")
    surf = json.loads((data / "mo_iv_surface_sample.json").read_text())
    assert surf["s0"] == 6000.0
    assert len(surf["maturities"]) >= 3 and len(surf["strikes"]) >= 5
    grid = np.array(surf["iv_grid"])
    assert grid.shape == (len(surf["maturities"]), len(surf["strikes"]))
    assert np.all(grid > 0) and np.all(grid < 2.0)
    # real exchange data arrives unsorted
    assert all(a < b for a, b in zip(surf["maturities"], surf["maturities"][1:]))
    # recovered rate/carry close to the sample's ground truth
    for pe in surf["per_expiry"]:
        assert abs(pe["r"] - 0.02) < 2e-3 and abs(pe["q"] - 0.01) < 2e-3

    # stage 03: Dupire local vol reprices the surface (regression guard, not a physics claim)
    _run(work, "03_dupire_localvol.py", "--tag", "sample")
    lv = json.loads((data / "mo_reprice_localvol_sample.json").read_text())
    assert lv["overall_rmse_iv"] < 0.03

    # stage 04: Heston calibration with a small seeded bootstrap
    _run(work, "04_heston_calibration.py", "--tag", "sample",
         "--bootstrap-reps", "2", "--bootstrap-seed", "12345")
    he = json.loads((data / "mo_calib_heston_sample.json").read_text())
    p = he["params"]
    assert 0 < p["v0"] < 0.5 and p["kappa"] > 0 and p["theta"] > 0 and -1 < p["rho"] < 0
    assert he["overall_rmse_iv"] < 0.03
    spec = he["calibration_spec"]
    assert spec["node_count"] == len(he["node_rows"])
    provenance = spec["surface_provenance"]
    assert provenance["kind"] == "file_sha256"
    assert provenance["sha256"] == hashlib.sha256((data / provenance["filename"]).read_bytes()).hexdigest()
    assert spec["bootstrap"]["same_start_and_constraints_as_main_fit"] is True
    jacobian = he["jacobian"]
    assert jacobian["shape"] == [spec["node_count"], 5]
    assert set(jacobian["svd"]) == {"raw", "fit_relative", "fixed_economic", "bound_span"}
    assert jacobian["excludes_feller_penalty"] is True
    bootstrap = he["bootstrap"]
    assert bootstrap["requested_replicates"] == 2
    assert bootstrap["successful_replicates"] + bootstrap["failed_replicates"] == 2
    assert len(bootstrap["replicates"]) == 2
    assert bootstrap["is_statistical_confidence_interval"] is False

    # stage 05: stochastic-local-vol leverage is positive and bounded
    _run(work, "05_slv_calibration.py", "--tag", "sample")
    slv = json.loads((data / "mo_reprice_slv_sample.json").read_text())
    assert 0 < slv["overall_rmse_iv"] < 0.08
    assert slv["leverage_min"] > 0 and slv["leverage_max"] < 10

    # stage 07: barrier option across models
    _run(work, "07_barrier_exotic.py", "--tag", "sample")
    models = json.loads((data / "mo_barrier_sample.json").read_text())["models"]
    assert set(models) >= {"BSM (flat ATM)", "Local Vol", "Heston", "SLV"}
    for d in models.values():
        assert d["mc"] > 0
        if "pde" in d:
            assert d["pde"] > 0
    bsm = models["BSM (flat ATM)"]
    # identical discrete monitoring: MC and PDE agree closely
    assert abs(bsm["mc"] - bsm["pde"]) < 0.15 * bsm["mc"]
    # the model matters on the exotic although all fit the same vanilla smile
    assert abs(models["Heston"]["mc"] - models["Local Vol"]["mc"]) > 1e-6

    # stage 06: lecture document, self-contained
    _run(work, "06_lecture.py", "--tag", "sample")
    html = (data / "mo_volmodels_lecture_sample.html").read_text()
    assert "<html" in html.lower() and "</html>" in html.lower()
    for token in ("Local Vol", "Heston", "SLV", "put-call parity", "Dupire", "Feller"):
        assert token in html, f"missing lecture content: {token}"
    assert "data:image/png;base64," in html
    assert (data / "comparison_summary_sample.csv").exists()

    # stage 10: calibration explainer
    _run(work, "10_explainer.py", "--tag", "sample")
    assert (data / "mo_calibration_explainer_sample.html").exists()

    # the repo tree was not touched
    assert _git_status(STUDY) == before
