"""Gate C for Greeks: point derivatives, desk moves and bump-limit ladders against the independent reference.

The full ladder runs only when asked for (memory: at most 4 workers; see harness.LADDER_MAX_GRID_CELLS):

    QUANTARK_GATE_C_GREEKS=1 QUANTARK_WRITE_GATE_C=<dir> python -m pytest -n 4 -m slow test/intraday/gate_c/test_greek_ladders.py

Workers append ``gate_c_greeks.jsonl`` under ``QUANTARK_WRITE_GATE_C``; ``greek_harness.aggregate_greeks`` writes
``quantark/intraday/evidence/gate_c_greeks.json``. One group per product always runs as a smoke test on a coarser
reference (its statuses are not evidence).
"""
import os

import pytest

from intraday.gate_c.greek_harness import append_greek_results, fast_greek_groups, greek_groups, run_greek_group

FULL = os.environ.get("QUANTARK_GATE_C_GREEKS") == "1"
WRITE_DIR = os.environ.get("QUANTARK_WRITE_GATE_C")
ACCEPTED = ("passed", "unqualified", "undefined", "inconclusive", "unsupported")


@pytest.mark.slow
@pytest.mark.skipif(not FULL, reason="the full Gate C greek ladder runs with QUANTARK_GATE_C_GREEKS=1")
@pytest.mark.parametrize("group", greek_groups(), ids=lambda g: g.id)
def test_greek_group(group):
    results = run_greek_group(group)
    if WRITE_DIR:
        append_greek_results(results, os.path.join(WRITE_DIR, "gate_c_greeks.jsonl"))
    bad = [(r.cell.id, m.measure, m.status, m.reason) for r in results for m in r.measures if m.status not in ACCEPTED]
    assert not bad, bad


@pytest.mark.parametrize("group", fast_greek_groups(), ids=lambda g: g.id)
def test_greek_group_smoke(group):
    results = run_greek_group(group, reference_points=(1001, 2001, 4001), proxies=group.product == "digital")
    measures = {(r.route_name, m.measure): m for r in results for m in r.measures}
    assert all(m.status in ACCEPTED + ("failed",) for m in measures.values())
    if group.product == "digital":
        assert measures[("AnalyticalDigitalRoute", "point_delta")].status == "passed"
        assert measures[("AnalyticalDigitalRoute", "point_vega")].status == "passed"
        assert measures[("MCRoute", "point_delta")].status == "unsupported"
    else:
        assert measures[("QuadV2Route", "point_delta")].status in ("passed", "inconclusive")
