"""CI freshness + content gates for the generated capability matrix (Phase 6)."""
import pathlib

from quantark.execution.capability_matrix import render_capability_matrix
from quantark.execution.inventory import ENGINE_INVENTORY

DOC_PATH = (
    pathlib.Path(__file__).parents[2] / "docs" / "execution"
    / "capability-matrix.md"
)


def test_checked_in_matrix_is_fresh():
    assert DOC_PATH.exists(), (
        "docs/execution/capability-matrix.md is missing; regenerate with "
        "python -m quantark.execution.capability_matrix docs/execution/capability-matrix.md"
    )
    assert DOC_PATH.read_text(encoding="utf-8") == render_capability_matrix(), (
        "capability matrix is stale; regenerate with "
        "python -m quantark.execution.capability_matrix docs/execution/capability-matrix.md"
    )


def test_matrix_covers_every_inventory_row():
    text = render_capability_matrix()
    for record in ENGINE_INVENTORY:
        assert f"`{record.name}`" in text, record.name


def test_matrix_is_deterministic():
    assert render_capability_matrix() == render_capability_matrix()
