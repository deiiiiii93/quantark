import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "example/mo_volmodels/data/mo_iv_surface_latest.json"


def test_stage02_builds_surface():
    subprocess.run(
        [sys.executable, str(ROOT / "example/mo_volmodels/02_build_iv_surface.py"),
         "--snapshot", "sample"],
        check=True, cwd=ROOT,
    )
    surf = json.loads(OUT.read_text())
    assert surf["s0"] == 6000.0
    assert len(surf["maturities"]) >= 3 and len(surf["strikes"]) >= 5
    grid = np.array(surf["iv_grid"])
    assert grid.shape == (len(surf["maturities"]), len(surf["strikes"]))
    assert np.all(grid > 0) and np.all(grid < 2.0)
    # maturities strictly increasing (real exchange data arrives unsorted)
    assert all(surf["maturities"][i] < surf["maturities"][i + 1] for i in range(len(surf["maturities"]) - 1))
    # recovered rate/carry close to the sample ground truth
    for pe in surf["per_expiry"]:
        assert abs(pe["r"] - 0.02) < 2e-3 and abs(pe["q"] - 0.01) < 2e-3
