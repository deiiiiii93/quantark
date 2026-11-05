"""Smoke test of example/param/curve_construction_demo.py (spec 2026-11-05, T9, AC16 demo part)."""
from __future__ import annotations

import importlib.util
import re
import time
from pathlib import Path

import pytest

DEMO_PATH = Path(__file__).resolve().parents[2] / "example" / "param" / "curve_construction_demo.py"
SCHEME_NAMES = ["LINEAR_ZERO", "LOG_LINEAR_DF", "CUBIC_ZERO", "LOG_CUBIC_DF", "MONOTONE_CONVEX"]
MARKERS = [
    "position: sticky",
    "scroll-padding-top",
    "IntersectionObserver",
    "data-theme",
    "localStorage",
    "prefers-color-scheme",
    "<svg",
    "<table",
]


def _load_demo():
    spec = importlib.util.spec_from_file_location("curve_construction_demo", DEMO_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    demo = _load_demo()
    out = tmp_path_factory.mktemp("report") / "report.html"
    t0 = time.perf_counter()
    path = demo.main(output_path=out, quiet=True)
    elapsed = time.perf_counter() - t0
    return demo, Path(path), elapsed, out


def test_runtime_and_output_path(run):
    _, path, elapsed, out = run
    assert elapsed < 60.0
    assert path == out and out.exists()


def test_html_markers(run):
    html = run[1].read_text(encoding="utf-8")
    for marker in MARKERS:
        assert marker in html, marker
    assert "<link" not in html and "http://" not in html and "https://" not in html
    assert html.count("<svg") >= 4


def test_table_follows_every_chart(run):
    html = run[1].read_text(encoding="utf-8")
    svgs = [m.start() for m in re.finditer(r"<svg", html)]
    tables = [m.start() for m in re.finditer(r"<table", html)]
    assert len(svgs) >= 4
    for i, s in enumerate(svgs):
        nxt_svg = svgs[i + 1] if i + 1 < len(svgs) else len(html)
        assert any(s < t < nxt_svg for t in tables), f"no table after chart {i}"


def test_true_minus_only(run):
    html = run[1].read_text(encoding="utf-8")
    body = re.sub(r"<(style|script)\b.*?</\1>", "", html, flags=re.S)
    # attribute values (tooltips, aria labels) are rendered text too
    attrs = " ".join(re.findall(r'(?:data-tip|aria-label)="([^"]*)"', body))
    text = re.sub(r"<[^>]*>", " ", body) + " " + attrs
    assert not re.search(r"-\s?\d", text), re.search(r".{20}-\s?\d.{10}", text)
    assert "−" in text
    assert "−0.000" not in text and "−0.00%" not in text


def test_deterministic(run, tmp_path):
    demo, path, _, _ = run
    again = demo.main(output_path=tmp_path / "again.html", quiet=True)
    assert Path(again).read_bytes() == path.read_bytes()


def test_stdout_tables_name_all_schemes(tmp_path, capsys):
    demo = _load_demo()
    demo.main(output_path=tmp_path / "r.html", quiet=False)
    out = capsys.readouterr().out
    for name in SCHEME_NAMES:
        assert name in out, name
    assert "residual" in out.lower()
    assert "carry" in out.lower() and "basis" in out.lower()


def test_quiet_prints_nothing(tmp_path, capsys):
    demo = _load_demo()
    demo.main(output_path=tmp_path / "q.html", quiet=True)
    assert capsys.readouterr().out == ""
