import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "example/mo_volmodels/data/mo_hedging_sample.json"
PLOT = ROOT / "example/mo_volmodels/data/plots/09_delta_hedging_sample.png"


def test_stage09_delta_hedging_runs():
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
            str(ROOT / "example/mo_volmodels/09_delta_hedging.py"),
            "--tag",
            "sample",
            "--fast",
            "--rebalances",
            "5",
            "--grid-size",
            "60",
            "--time-steps",
            "12",
            "--adi-x",
            "36",
            "--adi-v",
            "16",
            "--leverage-steps",
            "6",
            "--leverage-x",
            "41",
            "--leverage-z",
            "17",
        ],
        check=True,
        cwd=ROOT,
    )

    result = json.loads(OUT.read_text())
    assert result["spec"]["type"] == "ATM European call delta-neutral hedge"
    assert result["spec"]["rebalances"] == 5
    assert "previous_delta" in result["spec"]["residual_pnl_rule"]
    assert set(result["models"]) == {"BSM (flat vol)", "Local Vol", "Heston", "SLV"}
    assert result["spec"]["flat_vol"] > 0.0

    for row in result["models"].values():
        assert len(row["path"]) == 5
        assert abs(row["initial_delta"]) < 2.0
        assert row["total_rebalance_units"] > 0.0
        for point in row["path"]:
            assert point["tau"] > 0.0
            assert point["spot"] > 0.0
            assert abs(point["hedge_units"] + point["delta"]) < 1e-12

    assert PLOT.exists()
