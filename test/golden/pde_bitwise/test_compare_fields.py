"""compare.py's per-field gates (spec 3B §7.1, F3): absolute field bounds, movable fields, added keys."""
import json
import subprocess
import sys
from pathlib import Path

COMPARE = Path(__file__).resolve().parent / "compare.py"


def _run(tmp_path, base, branch, *args):
    b, n = tmp_path / "base.json", tmp_path / "branch.json"
    b.write_text(json.dumps(base))
    n.write_text(json.dumps(branch))
    allow = tmp_path / "allow.txt"
    allow.write_text("r\n")
    proc = subprocess.run([sys.executable, str(COMPARE), str(b), str(n), "--allow-file", str(allow), *args],
                          capture_output=True, text=True)
    return proc.returncode, proc.stdout


def h(x):
    return float(x).hex()


def test_bound_field_is_absolute_and_per_field(tmp_path):
    base = {"r": {"stats_ko_probability": [h(0.5), h(0.25)], "price": h(100.0)}}
    branch = {"r": {"stats_ko_probability": [h(0.5 + 5e-8), h(0.25)], "price": h(100.0)}}
    code, out = _run(tmp_path, base, branch, "--bound-field", "r:stats_ko_*=1e-7")
    assert code == 0, out
    assert "MAXDELTA stats_ko_probability" in out
    code, out = _run(tmp_path, base, branch, "--bound-field", "r:stats_ko_*=1e-8")
    assert code == 1 and "beyond bound" in out


def test_the_first_matching_bound_applies(tmp_path):
    """Specific globs first, the catch-all last (Task 1): a row the specific glob covers is not also held
    to the catch-all's tighter bound."""
    base = {"r": {"stats_reconciliation_error": h(0.0)}}
    branch = {"r": {"stats_reconciliation_error": h(0.25)}}
    code, out = _run(tmp_path, base, branch, "--bound-field", "r:stats_reconciliation_error=10",
                     "--bound-field", "*:stats_reconciliation_error=1e-10")
    assert code == 0, out
    code, out = _run(tmp_path, base, branch, "--bound-field", "*:stats_reconciliation_error=1e-10",
                     "--bound-field", "r:stats_reconciliation_error=10")
    assert code == 1 and "beyond bound 1e-10" in out


def test_bound_field_row_list_from_a_file(tmp_path):
    """``@PATH`` names exact row ids, one per line (spec 3B stage 2 (b): the plain Phoenix movers)."""
    base = {"r": {"stats_x": h(1.0)}}
    branch = {"r": {"stats_x": h(1.5)}}
    listed, other = tmp_path / "listed.txt", tmp_path / "other.txt"
    listed.write_text("r\n")
    other.write_text("q\n")
    code, out = _run(tmp_path, base, branch, "--bound-field", f"@{listed}:stats_x=1e-3")
    assert code == 1 and "beyond bound 0.001" in out
    code, out = _run(tmp_path, base, branch, "--bound-field", f"@{other}:stats_x=1e-3")
    assert code == 0, out


def test_movable_fields_reject_a_price_move(tmp_path):
    base = {"r": {"stats_pv": h(1.0), "price": h(100.0)}}
    branch = {"r": {"stats_pv": h(1.5), "price": h(100.0 + 1e-13)}}
    code, out = _run(tmp_path, base, branch, "--movable-fields", "stats_*")
    assert code == 1 and "price" in out


def test_added_key_allowed_only_by_glob(tmp_path):
    base = {"r2": {"stats_pv": h(1.0)}}
    branch = {"r2": {"stats_pv": h(1.0), "stats_provenance": "repr:StatisticsProvenance('a', 'a')"}}
    b, n = tmp_path / "b.json", tmp_path / "n.json"
    b.write_text(json.dumps(base)); n.write_text(json.dumps(branch))
    ok = subprocess.run([sys.executable, str(COMPARE), str(b), str(n), "--allow-added-key", "stats_provenance"],
                        capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout
    bad = subprocess.run([sys.executable, str(COMPARE), str(b), str(n)], capture_output=True, text=True)
    assert bad.returncode == 1


def test_a_key_dropped_by_the_branch_is_a_diff(tmp_path):
    base = {"r2": {"stats_pv": h(1.0), "stats_x": h(2.0)}}
    branch = {"r2": {"stats_pv": h(1.0)}}
    b, n = tmp_path / "b.json", tmp_path / "n.json"
    b.write_text(json.dumps(base)); n.write_text(json.dumps(branch))
    out = subprocess.run([sys.executable, str(COMPARE), str(b), str(n), "--allow-added-key", "*"],
                         capture_output=True, text=True)
    assert out.returncode == 1


def test_tagged_strings_pass_the_manifest(tmp_path):
    book = {"r3": {"stats_flag": "bool:1", "stats_dates": "repr:(1, 2)", "stats_opt": "none", "x": h(1.0)}}
    m = tmp_path / "m.json"
    m.write_text(json.dumps({"r3": {"none": [], "error": None}}))
    b = tmp_path / "b.json"
    b.write_text(json.dumps(book))
    out = subprocess.run([sys.executable, str(COMPARE), str(b), str(b), "--manifest", str(m)],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout


def test_base_manifest_lets_a_row_stop_raising(tmp_path):
    base = {"r": {"__error__": "KeyError: 0.5014"}}
    branch = {"r": {"price": h(1.0)}}
    bm, nm = tmp_path / "bm.json", tmp_path / "nm.json"
    bm.write_text(json.dumps({"r": {"none": [], "error": "KeyError"}}))
    nm.write_text(json.dumps({"r": {"none": [], "error": None}}))
    code, out = _run(tmp_path, base, branch, "--manifest", str(nm), "--base-manifest", str(bm))
    assert code == 0, out
