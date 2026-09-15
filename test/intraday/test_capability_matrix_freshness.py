"""CI freshness gate for the generated intraday capability matrix document."""
import pathlib

from quantark.intraday.publish import render_document

DOC_PATH = pathlib.Path(__file__).parents[2] / "docs" / "execution" / "intraday-capability-matrix.md"
REGENERATE = "python -m quantark.intraday.publish docs/execution/intraday-capability-matrix.md"


def test_checked_in_intraday_matrix_is_fresh():
    assert DOC_PATH.exists(), f"docs/execution/intraday-capability-matrix.md is missing; regenerate with {REGENERATE}"
    assert DOC_PATH.read_text(encoding="utf-8") == render_document(), f"intraday capability matrix is stale; {REGENERATE}"


def test_rendering_is_deterministic():
    assert render_document() == render_document()
