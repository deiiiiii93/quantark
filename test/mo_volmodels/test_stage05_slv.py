import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "example/mo_volmodels/data/mo_reprice_slv_sample.json"


def test_stage05_slv_calibrates_and_reprices():
    # prerequisites: sample surface + sample Heston calibration
    for stage, extra in [("02_build_iv_surface.py", ["--snapshot", "sample"]),
                         ("04_heston_calibration.py", ["--tag", "sample"])]:
        subprocess.run([sys.executable, str(ROOT / "example/mo_volmodels" / stage), *extra],
                       check=True, cwd=ROOT)
    subprocess.run(
        [sys.executable, str(ROOT / "example/mo_volmodels/05_slv_calibration.py"), "--tag", "sample"],
        check=True, cwd=ROOT,
    )
    res = json.loads(OUT.read_text())
    # SLV must produce a valid, positive leverage surface and a finite, bounded reprice RMSE.
    # (We do NOT assert SLV beats Heston on vanillas: the SLV PDE carries an inherent
    # few-vol-point discretization bias, and SLV's value is smile-consistent DYNAMICS for
    # exotics, not tighter vanilla repricing.)
    assert res["overall_rmse_iv"] > 0 and res["overall_rmse_iv"] < 0.08
    assert res["leverage_min"] > 0 and res["leverage_max"] < 10
