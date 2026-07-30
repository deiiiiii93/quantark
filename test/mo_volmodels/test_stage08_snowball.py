import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "example/mo_volmodels/data/mo_snowball_sample.json"
PLOT = ROOT / "example/mo_volmodels/data/plots/08_snowball_sample.png"


def test_stage08_snowball_exotic_runs():
    for stage, extra in [
        ("02_build_iv_surface.py", ["--snapshot", "sample"]),
        ("04_heston_calibration.py", ["--tag", "sample"]),
    ]:
        subprocess.run(
            [sys.executable, str(ROOT / "example/mo_volmodels" / stage), *extra],
            check=True,
            cwd=ROOT,
        )

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "example/mo_volmodels/08_snowball_exotic.py"),
            "--tag",
            "sample",
            "--fast",
            "--mc-paths",
            "512",
            "--time-steps",
            "12",
            "--grid-size",
            "60",
            "--adi-x",
            "224",
            "--adi-v",
            "16",
            "--leverage-steps",
            "8",
            "--leverage-x",
            "41",
            "--leverage-z",
            "21",
        ],
        check=True,
        cwd=ROOT,
    )

    result = json.loads(OUT.read_text())
    assert set(result["models"]) >= {
        "BSM (flat ATM)",
        "Local Vol",
        "Heston QE",
        "SLV",
        "SLV QE",
    }
    assert result["spec"]["type"] == "standard snowball"
    assert result["spec"]["T"] == 2.0
    assert result["spec"]["principal"] == "excluded"
    assert result["spec"]["include_principal"] is False
    assert result["spec"]["ko_monitoring"] == "monthly discrete"
    assert result["spec"]["ki_monitoring"] == "continuous"
    assert result["spec"]["pde_grid_focus"] == "auto"
    assert "centers at KI" in result["spec"]["pde_grid_policy"]

    notional = result["spec"]["notional"]
    for row in result["models"].values():
        assert abs(row["mc"]) < notional
        assert abs(row["mc_pct_initial"]) < 100.0
        if row["pde"] is not None:
            assert abs(row["pde"]) < notional
            assert abs(row["pde_pct_initial"]) < 100.0

    assert result["models"]["SLV QE"]["pde"] is None
    assert result["models"]["SLV QE"]["cross_check"] is None
    assert PLOT.exists()
