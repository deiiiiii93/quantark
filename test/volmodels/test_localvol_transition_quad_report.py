"""Smoke test for the local-vol transition QUAD report renderer (spec D15, AC11).

``docs/`` is not a package, so the renderer is loaded by path. The renderer is
pure stdlib: these tests read the committed result JSONs and price nothing.
"""

from __future__ import annotations

import importlib.util
import re
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "docs" / "research" / "localvol_transition_quad"
SCRIPT = FOLDER / "render_report.py"

pytestmark = pytest.mark.skipif(not (FOLDER / "reference.json").exists(),
                                reason="study results not present (docs/ is force-added per clone)")


def _renderer():
    spec = importlib.util.spec_from_file_location("lvq_render_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sections(page: str) -> dict[str, str]:
    """Section id -> its HTML, split on the section tags."""
    out = {}
    for match in re.finditer(r'<section id="(s\d+)">(.*?)</section>', page, flags=re.S):
        out[match.group(1)] = match.group(2)
    return out


def _visible_text(page: str) -> str:
    page = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", " ", page, flags=re.S)
    return re.sub(r"<[^>]+>", " ", page)


@pytest.fixture(scope="module")
def page() -> str:
    return _renderer().render(FOLDER)


def test_required_chrome(page):
    # frozen navigator: function bar + numbered menu in one sticky block
    assert '<div class="top">' in page and '<nav class="menu" id="menu">' in page
    assert re.search(r"\.top\{position:sticky;top:0", page)
    assert "scroll-padding-top" in page and "IntersectionObserver" in page
    # light/dark switcher with the pre-paint theme script and the light token block
    assert 'id="theme"' in page and "localStorage" in page and ':root[data-theme="light"]' in page
    assert "prefers-color-scheme" in page
    # self-contained: no external assets
    assert not re.search(r'<(script|link|img)[^>]+(src|href)="https?://', page)


def test_all_sections_and_charts(page):
    sections = _sections(page)
    assert sorted(sections, key=lambda s: int(s[1:])) == [f"s{i}" for i in range(1, 11)]
    for i in range(1, 11):
        assert f'href="#s{i}"' in page
    assert page.count("<svg") >= 1
    for sid, body in sections.items():
        if "<svg" in body:
            last_svg = body.rfind("<svg")
            assert "<table" in body[last_svg:], f"{sid}: a chart without a table under it"
    assert "data-tip=" in page              # hover layer


def test_no_missing_values_leak(page):
    text = _visible_text(page)
    for token in ("nan", "NaN", "None", "inf", "null", "undefined"):
        assert not re.search(rf"(?<![\w.]){token}(?![\w])", text), token
    assert not re.search(r"−0\.0+(?![0-9])", text)          # no signed zero


def test_pending_without_benchmark(tmp_path):
    for f in FOLDER.glob("*.json"):
        if f.name != "benchmark.json":
            shutil.copy(f, tmp_path / f.name)
    html = _renderer().render(tmp_path)
    sections = _sections(html)
    assert "PENDING" in sections["s6"] and "benchmark.json" in sections["s6"]
    assert "PENDING" in sections["s1"]
    assert len(sections) == 10


def test_survives_missing_inputs(tmp_path):
    """Partial runs (only the reference written) still render every section."""
    shutil.copy(FOLDER / "reference.json", tmp_path / "reference.json")
    (tmp_path / "tuning.json").write_text('{"cases": {"snowball_1y": {"quad": {}')    # mid-write
    html = _renderer().render(tmp_path)
    assert len(_sections(html)) == 10 and "PENDING" in html


def test_cli_writes_report(tmp_path):
    for f in FOLDER.glob("*.json"):
        shutil.copy(f, tmp_path / f.name)
    assert _renderer().main(["--dir", str(tmp_path)]) == 0
    assert (tmp_path / "report.html").read_text(encoding="utf-8").startswith("<!doctype html>")
