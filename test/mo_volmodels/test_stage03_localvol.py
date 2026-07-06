import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "example/mo_volmodels/data/mo_reprice_localvol_sample.json"


def test_stage03_localvol_reprices_within_tol():
    # ensure the sample surface exists, then reprice it
    subprocess.run(
        [sys.executable, str(ROOT / "example/mo_volmodels/02_build_iv_surface.py"),
         "--snapshot", "sample"],
        check=True, cwd=ROOT,
    )
    subprocess.run(
        [sys.executable, str(ROOT / "example/mo_volmodels/03_dupire_localvol.py"), "--tag", "sample"],
        check=True, cwd=ROOT,
    )
    res = json.loads(OUT.read_text())
    # Dupire reprices the surface it was built from up to (a) PDE discretization and
    # (b) the crude butterfly estimate on a coarse strike grid. A few vol-points is
    # normal and grows with maturity; this bound is a regression guard, not a physics claim.
    assert res["overall_rmse_iv"] < 0.03
