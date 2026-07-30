import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "example/mo_volmodels/data/mo_calib_heston_sample.json"


def test_stage04_heston_calibrates():
    subprocess.run(
        [sys.executable, str(ROOT / "example/mo_volmodels/02_build_iv_surface.py"),
         "--snapshot", "sample"],
        check=True, cwd=ROOT,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "example/mo_volmodels/04_heston_calibration.py"),
            "--tag",
            "sample",
            "--bootstrap-reps",
            "2",
            "--bootstrap-seed",
            "12345",
        ],
        check=True, cwd=ROOT,
    )
    res = json.loads(OUT.read_text())
    p = res["params"]
    assert 0 < p["v0"] < 0.5 and p["kappa"] > 0 and p["theta"] > 0 and -1 < p["rho"] < 0
    # Heston fits the whole surface to a handful of vol points.
    assert res["overall_rmse_iv"] < 0.03
    spec = res["calibration_spec"]
    assert spec["node_count"] == len(res["node_rows"])
    provenance = spec["surface_provenance"]
    surface_path = OUT.with_name(provenance["filename"])
    assert provenance["kind"] == "file_sha256"
    assert provenance["sha256"] == hashlib.sha256(surface_path.read_bytes()).hexdigest()
    assert spec["bootstrap"]["same_start_and_constraints_as_main_fit"] is True
    jacobian = res["jacobian"]
    assert jacobian["shape"] == [spec["node_count"], 5]
    assert set(jacobian["svd"]) == {
        "raw", "fit_relative", "fixed_economic", "bound_span",
    }
    assert jacobian["excludes_feller_penalty"] is True
    bootstrap = res["bootstrap"]
    assert bootstrap["requested_replicates"] == 2
    assert bootstrap["successful_replicates"] + bootstrap["failed_replicates"] == 2
    assert len(bootstrap["replicates"]) == 2
    assert bootstrap["is_statistical_confidence_interval"] is False
