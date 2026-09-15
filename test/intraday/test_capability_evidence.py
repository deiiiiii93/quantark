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


def test_greek_evidence_parses_and_every_demonstration_is_backed_by_passing_cells():
    import importlib
    from quantark.intraday.capability import greek_evidence
    payload = greek_evidence()
    assert payload and payload["schema"] == "intraday-gate-c-greeks/1" and payload["git_sha"]
    for row in payload["demonstrated"]:
        module = importlib.import_module(f"quantark.intraday.engines.{_route_module(row['route'])}")
        assert hasattr(module, row["route"]), row["route"]
        cells = [c for c in payload["cells"] if (c["product"], c["route"]) == (row["product"], row["route"])
                 and c["cell"]["horizon"] >= row["horizon_s"]]
        statuses = {m["status"] for c in cells for m in c["measures"] if m["measure"] == row["measure"]}
        assert cells and statuses <= {"passed", "undefined"}, row


def _route_module(route: str) -> str:
    return {"QuadV2Route": "quad_v2", "AnalyticalDigitalRoute": "analytical_digital", "AnalyticalBarrierRoute": "analytical_barrier",
            "PDERoute": "pde", "MCRoute": "mc"}[route]
