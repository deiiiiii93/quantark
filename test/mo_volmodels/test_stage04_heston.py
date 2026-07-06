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
        [sys.executable, str(ROOT / "example/mo_volmodels/04_heston_calibration.py"), "--tag", "sample"],
        check=True, cwd=ROOT,
    )
    res = json.loads(OUT.read_text())
    p = res["params"]
    assert 0 < p["v0"] < 0.5 and p["kappa"] > 0 and p["theta"] > 0 and -1 < p["rho"] < 0
    # Heston fits the whole surface to a handful of vol points.
    assert res["overall_rmse_iv"] < 0.03
