import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Stage 01 is skipped (needs akshare); the committed sample snapshot drives the offline suite.
STAGES = [
    ("02_build_iv_surface.py", ["--snapshot", "sample"]),
    ("03_dupire_localvol.py", ["--tag", "sample"]),
    (
        "04_heston_calibration.py",
        ["--tag", "sample", "--bootstrap-reps", "2", "--bootstrap-seed", "12345"],
    ),
    ("05_slv_calibration.py", ["--tag", "sample"]),
    ("07_barrier_exotic.py", ["--tag", "sample"]),
    ("06_lecture.py", ["--tag", "sample"]),
    ("10_explainer.py", ["--tag", "sample"]),
]


def test_full_offline_pipeline():
    for stage, extra in STAGES:
        r = subprocess.run(
            [sys.executable, str(ROOT / "example/mo_volmodels" / stage), *extra],
            cwd=ROOT, capture_output=True, text=True,
        )
        assert r.returncode == 0, f"{stage} failed:\n{r.stdout}\n{r.stderr}"
    assert (ROOT / "example/mo_volmodels/data/mo_volmodels_lecture_sample.html").exists()
    assert (ROOT / "example/mo_volmodels/data/mo_calibration_explainer_sample.html").exists()
