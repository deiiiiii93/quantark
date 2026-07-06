import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HTML = ROOT / "example/mo_volmodels/data/mo_volmodels_lecture_sample.html"


def test_stage06_lecture_builds():
    # depends on stages 02-05 sample artifacts existing; the smoke test runs the full chain.
    for stage, extra in [
        ("02_build_iv_surface.py", ["--snapshot", "sample"]),
        ("03_dupire_localvol.py", ["--tag", "sample"]),
        ("04_heston_calibration.py", ["--tag", "sample"]),
        ("05_slv_calibration.py", ["--tag", "sample"]),
    ]:
        subprocess.run([sys.executable, str(ROOT / "example/mo_volmodels" / stage), *extra],
                       check=True, cwd=ROOT)
    subprocess.run(
        [sys.executable, str(ROOT / "example/mo_volmodels/06_lecture.py"), "--tag", "sample"],
        check=True, cwd=ROOT,
    )
    html = HTML.read_text()
    assert "<html" in html.lower() and "</html>" in html.lower()
    for token in ("Local Vol", "Heston", "SLV", "put-call parity", "Dupire", "Feller"):
        assert token in html, f"missing lecture content: {token}"
    assert "data:image/png;base64," in html  # self-contained figures
    assert (ROOT / "example/mo_volmodels/data/comparison_summary_sample.csv").exists()
