import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "example/mo_volmodels/data/mo_barrier_sample.json"


def test_stage07_barrier_divergence():
    for stage, extra in [
        ("02_build_iv_surface.py", ["--snapshot", "sample"]),
        ("04_heston_calibration.py", ["--tag", "sample"]),
    ]:
        subprocess.run([sys.executable, str(ROOT / "example/mo_volmodels" / stage), *extra],
                       check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(ROOT / "example/mo_volmodels/07_barrier_exotic.py"),
                    "--tag", "sample"], check=True, cwd=ROOT)
    res = json.loads(OUT.read_text())
    models = res["models"]
    assert set(models) >= {"BSM (flat ATM)", "Local Vol", "Heston", "SLV"}
    for m, d in models.items():
        assert d["mc"] > 0
        if "pde" in d:
            assert d["pde"] > 0
    # BSM MC and PDE use the identical discrete monitoring -> should agree closely.
    bsm = models["BSM (flat ATM)"]
    assert abs(bsm["mc"] - bsm["pde"]) < 0.15 * bsm["mc"]
    # Model choice matters on the exotic: stochastic-vol barrier price differs from local vol
    # even though all models fit the SAME vanilla smile.
    assert abs(models["Heston"]["mc"] - models["Local Vol"]["mc"]) > 1e-6
