"""The shipped demo must keep running, and its config must stay loadable.

A demo that has rotted is worse than none: it is the first thing a new user
runs, and the module's own README points at it.
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "example/volcalibration/01_end_to_end.py"
CONFIG = ROOT / "example/volcalibration/usdcny_calibration.yaml"


def test_the_demo_config_loads_and_is_fx_shaped():
    from quantark.volcalibration.yaml_loader import load_run_config

    config = load_run_config(CONFIG)
    assert config.name == "usdcny-demo"
    assert config.underlying.convention == "fx_delta"
    assert config.underlying.price_field == "mid_iv"
    assert config.calibration.variants == ("localvol",)


@pytest.mark.slow
def test_the_demo_runs_end_to_end(tmp_path):
    """Run it for real, into a throwaway store."""
    config = tmp_path / "usdcny.yaml"
    config.write_text(
        CONFIG.read_text(encoding="utf-8").replace(
            "root: ../../output/volcalibration_demo", f"root: {tmp_path / 'store'}"
        ),
        encoding="utf-8",
    )
    demo = tmp_path / "demo.py"
    demo.write_text(
        DEMO.read_text(encoding="utf-8").replace(
            'CONFIG = HERE / "usdcny_calibration.yaml"', f"CONFIG = Path({str(config)!r})"
        ).replace(
            "HERE = Path(__file__).resolve().parent",
            f"HERE = Path({str(ROOT / 'example/volcalibration')!r})",
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(demo)],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert "6 date(s) calibrated" in result.stdout
    # The forward reconstruction is the demo's own numerical check.
    assert "diff=0.00e+00" in result.stdout
