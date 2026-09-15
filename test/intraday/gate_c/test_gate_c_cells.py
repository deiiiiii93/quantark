"""Gate C: every intraday route against the independent reference on the time-to-fixing ladder.

The full ladder (3432 cells) runs only when asked for, because the repository's default selection does not
exclude ``slow``:

    QUANTARK_GATE_C=1 QUANTARK_WRITE_GATE_C=<dir> python -m pytest -n auto -m slow test/intraday/gate_c

``QUANTARK_WRITE_GATE_C`` names a directory; workers append ``gate_c_results.jsonl`` there and
``harness.aggregate`` turns it into ``quantark/intraday/evidence/gate_c_results.json``. One representative
cell per snowball engine always runs.
"""
import os

import pytest

from intraday.gate_c import cells as C
from intraday.gate_c.harness import append_result, run_cell

FULL = os.environ.get("QUANTARK_GATE_C") == "1"
WRITE_DIR = os.environ.get("QUANTARK_WRITE_GATE_C")
ACCEPTED = ("passed", "unqualified", "inconclusive", "unsupported")


def _record(result):
    if WRITE_DIR:
        append_result(result, os.path.join(WRITE_DIR, "gate_c_results.jsonl"))


@pytest.mark.slow
@pytest.mark.skipif(not FULL, reason="the full Gate C ladder runs with QUANTARK_GATE_C=1")
@pytest.mark.parametrize("cell", C.all_cells(), ids=lambda c: c.id)
def test_gate_c_cell(cell):
    result = run_cell(cell)
    _record(result)
    assert result.status in ACCEPTED, result


@pytest.mark.parametrize("cell", C.fast_cells(), ids=lambda c: c.id)
def test_gate_c_fast_cell(cell):
    result = run_cell(cell)
    assert result.status in ACCEPTED, result
    if cell.engine == "quad_v2":
        assert result.passed, result          # the exact Gaussian route meets the budget one hour before a fixing
