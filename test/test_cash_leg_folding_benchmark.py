"""Smoke test of example/cash_leg_folding_benchmark.py on tiny grids (ship spec 2026-10-21 AC13)."""
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "example" / "cash_leg_folding_benchmark.py"


def _load():
    spec = importlib.util.spec_from_file_location("cash_leg_folding_benchmark", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quick_benchmark_runs_every_engine_and_the_folded_totals_agree():
    rows = _load().run(quick=True, repeat=1)
    assert [r["engine"] for r in rows] == ["QUAD v1", "QUAD v2", "PDE 1D", "MC"]
    for r in rows:
        assert r["event"] > 0.0 and r["total"] > 0.0 and r["drilldown"] > 0.0
        assert r["drill_total_gap"] <= 1e-6, r            # TOTAL and DRILLDOWN totals: one valuation rule
