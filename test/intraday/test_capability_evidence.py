"""The published capability rows agree with the packaged Gate C price evidence."""
from datetime import timedelta

import pytest

from quantark.intraday.capability import INTRADAY_CAPABILITIES, capability_evidence

#: Gate C (product, engine) -> the capability row it certifies: (product class, engine class, monitoring)
GATE_C_ROWS = {
    ("snowball_discrete_ki", "quad_v2"): ("SnowballOption", "SnowballQuadEngineV2", "discrete"),
    ("snowball_discrete_ki", "pde"): ("SnowballOption", "SnowballPDESolver", "discrete"),
    ("snowball_discrete_ki", "mc_rqmc"): ("SnowballOption", "SnowballMCEngine", "discrete"),
    ("digital", "analytical"): ("CashOrNothingDigitalOption", "DigitalOptionAnalyticalEngine", "terminal"),
    ("digital", "mc_rqmc"): ("CashOrNothingDigitalOption", "DigitalOptionMCEngine", "terminal"),
    ("barrier_uo_zero_carry", "analytical"): ("BarrierOption", "BarrierAnalyticalEngine", "continuous"),
    ("barrier_uo_zero_carry", "pde"): ("BarrierOption", "BarrierPDESolver", "continuous"),
    ("barrier_uo_zero_carry", "mc_rqmc"): ("BarrierOption", "BarrierOptionMCEngine", "continuous"),
    ("one_touch_zero_carry", "analytical"): ("OneTouchOption", "OneTouchAnalyticalEngine", "continuous"),
    ("one_touch_zero_carry", "pde"): ("OneTouchOption", "OneTouchPDESolver", "continuous"),
}


def _row_key(row):
    return row.product_type.__name__, row.engine_class_path.rsplit(".", 1)[-1], row.monitoring


@pytest.fixture(scope="module")
def evidence():
    payload = capability_evidence()
    assert payload, "quantark/intraday/evidence/gate_c_results.json must be packaged"
    return payload


def test_the_evidence_file_parses_and_names_its_run(evidence):
    assert evidence["schema"] == "intraday-gate-c/1" and evidence["git_sha"]
    assert len(evidence["cells"]) == 3432
    assert {c["status"] for c in evidence["cells"]} <= {"passed", "unqualified", "inconclusive", "unsupported", "failed"}


def test_every_evidence_pair_names_a_published_row(evidence):
    published = {_row_key(r) for r in INTRADAY_CAPABILITIES}
    pairs = {(c["cell"]["product"], c["cell"]["engine"]) for c in evidence["cells"]}
    assert pairs == set(GATE_C_ROWS)
    assert set(GATE_C_ROWS.values()) <= published


def test_qualified_rows_are_backed_by_passing_cells_at_and_above_their_horizon(evidence):
    by_row = {}
    for c in evidence["cells"]:
        by_row.setdefault(GATE_C_ROWS[(c["cell"]["product"], c["cell"]["engine"])], []).append(c)
    for row in INTRADAY_CAPABILITIES:
        cells = by_row.get(_row_key(row), [])
        if row.status != "qualified":
            continue
        assert cells, f"{_row_key(row)} is qualified without Gate C cells"
        assert all(c["status"] != "failed" for c in cells), f"{_row_key(row)} has a failed cell"
        horizon = row.qualified_horizon.total_seconds()
        above = [c for c in cells if c["cell"]["horizon"] >= horizon]
        assert above and all(c["status"] == "passed" for c in above), _row_key(row)
        profiles = {c["cell"]["profile"] for c in above}
        offsets = {c["cell"]["offset"] for c in above}
        assert profiles == {"uniform", "desk", "sessions_only"} and len(offsets) == 11


def test_the_qualified_horizon_is_the_shortest_the_evidence_supports(evidence):
    horizons = sorted({c["cell"]["horizon"] for c in evidence["cells"]})
    for row in INTRADAY_CAPABILITIES:
        if row.status != "qualified":
            continue
        cells = [c for c in evidence["cells"] if GATE_C_ROWS[(c["cell"]["product"], c["cell"]["engine"])] == _row_key(row)]
        shorter = [h for h in horizons if h < row.qualified_horizon.total_seconds()]
        if shorter:
            next_down = max(shorter)
            assert any(c["status"] != "passed" for c in cells if c["cell"]["horizon"] >= next_down), \
                f"{_row_key(row)} could claim {timedelta(seconds=next_down)}"


def test_rows_without_evidence_are_not_qualified(evidence):
    certified = set(GATE_C_ROWS.values())
    assert all(_row_key(r) in certified for r in INTRADAY_CAPABILITIES if r.status == "qualified")


def _backing_cells(payload, row):
    """The cells a demonstrated row rests on: its whole family, economic identity and monitoring included, in its window.

    Product, route, profile and settings alone would let the daily-KI cells back (and break) the monthly rows.
    """
    from intraday.gate_c.cells import MONITORING
    return [c for c in payload["cells"] if (c["product"], c["route"]) == (row["product"], row["route"])
            and c["cell"]["profile"] == row["profile"] and c.get("settings") == row["settings"]
            and MONITORING[c["cell"]["product"]] == row["monitoring"]
            and c.get("economic_identity", "") == row.get("economic_identity", "")
            and row["horizon_s"] <= c["cell"]["horizon"] <= row["horizon_max_s"]]


def test_greek_evidence_parses_and_every_demonstration_is_backed_by_passing_cells():
    import importlib
    from quantark.intraday.capability import greek_evidence
    payload = greek_evidence()
    assert payload and payload["schema"] == "intraday-gate-c-greeks/2" and payload["git_sha"]
    for row in payload["demonstrated"]:
        module = importlib.import_module(f"quantark.intraday.engines.{_route_module(row['route'])}")
        assert hasattr(module, row["route"]), row["route"]
        assert row["settings"]["engine"] and row["horizon_s"] <= row["horizon_max_s"]
        # a window is backed by EVERY cell of its family -- same profile, engine settings and measure settings --
        # at every swept horizon inside it, and by nothing outside it
        cells = _backing_cells(payload, row)
        measures = [m for c in cells for m in c["measures"]
                    if m["measure"] == row["measure"] and (m.get("measure_settings") or {}) == row["measure_settings"]]
        assert measures and {m["status"] for m in measures} <= {"passed", "undefined"}, row
        assert sorted({c["cell"]["horizon"] for c in cells}) == row["swept_horizons_s"], row


def _route_module(route: str) -> str:
    return {"QuadV2Route": "quad_v2", "AnalyticalDigitalRoute": "analytical_digital", "AnalyticalBarrierRoute": "analytical_barrier",
            "PDERoute": "pde", "MCRoute": "mc"}[route]


def test_backing_cells_never_mix_economic_families():
    row = {"product": "SnowballOption", "route": "QuadV2Route", "profile": "desk", "settings": {"engine": "E"},
           "monitoring": "discrete", "economic_identity": "monthly", "horizon_s": 1, "horizon_max_s": 3600}

    def cell(fixture, identity, status):
        return {"product": "SnowballOption", "route": "QuadV2Route", "settings": {"engine": "E"},
                "economic_identity": identity, "cell": {"product": fixture, "profile": "desk", "horizon": 60},
                "measures": [{"measure": "point_delta", "status": status}]}

    payload = {"cells": [cell("snowball_discrete_ki", "monthly", "passed"),
                         cell("snowball_daily_ki", "daily", "unqualified")]}
    assert [c["economic_identity"] for c in _backing_cells(payload, row)] == ["monthly"]
