"""The base-family event helper (spec 3A §5.4; decisions 8, 14)."""
import json
from pathlib import Path

import numpy as np
import pytest

from golden_compare import assert_close
from quantark.asset.equity.engine.pde import BarrierPDESolver
from quantark.asset.equity.engine.pde.base_pde_solver import BasePDESolver
from quantark.asset.equity.engine.pde.core.prepare import Operation, OutputRequest, prepare, run
from quantark.asset.equity.engine.pde.grid import DiscreteEvent, EventOp, EventRecord, Nodes, apply_discrete_event
from quantark.asset.equity.engine.pde.grid.events import _dual_cell_edges
from quantark.asset.equity.param import PDEParams
from quantark.util.enum.engine_enums import EventProjectionMode as P

from pde_3a.semantics import cases
from term_structure_benchmarks import make_term_env

V070 = json.loads((Path(__file__).parent / "pde_3a" / "fixtures" / "v070_event_semantics.json").read_text())


def _nodes(n=41, lo=80.0, hi=125.0):
    x = np.linspace(np.log(lo), np.log(hi), n)
    return Nodes(x, np.exp(x))


def test_nodal_set_and_add_are_the_masks():
    nd = _nodes()
    col = np.linspace(1.0, 2.0, len(nd.s))
    ev = DiscreteEvent((EventRecord(None, 110.0, 5.0, EventOp.SET), EventRecord(None, 115.0, 1.0, EventOp.ADD)))
    out = apply_discrete_event(nd, col, ev, P.NODAL)
    want = col.copy()
    want[nd.s >= 110.0] = 5.0
    want[nd.s >= 115.0] += 1.0
    assert out.tobytes() == want.tobytes()
    assert not np.shares_memory(out, col)


def _event_at(ev, s, v):
    for rec in ev.records:
        if rec.breached(np.array([s]))[0]:
            v = rec.value if rec.op is EventOp.SET else v + rec.value
    return v


def _exact_cell_average(nd, col, ev):
    """The dual-cell average of the composite event function, integrated exactly.

    The column is its linear interpolant; every piece between consecutive nodes and thresholds is
    linear and wholly breached or not, so the trapezoid rule on it is exact. A cell no threshold
    crosses keeps the nodal value (the projection averages crossed cells only).
    """
    edges = _dual_cell_edges(nd.x)
    cuts_x = sorted({float(np.log(t)) for r in ev.records for t in (r.lower, r.upper) if t is not None})
    out = np.empty_like(col)
    for i in range(len(col)):
        lo, hi = float(edges[i]), float(edges[i + 1])
        inner = [c for c in cuts_x if lo < c < hi]
        if not inner:
            out[i] = _event_at(ev, float(nd.s[i]), float(col[i]))
            continue
        pts = sorted({lo, hi, *inner, *(float(x) for x in nd.x if lo < x < hi)})
        total = 0.0
        for a, c in zip(pts[:-1], pts[1:]):
            mid_s = float(np.exp(0.5 * (a + c)))                    # the piece's breach state
            fa = _event_at(ev, mid_s, float(np.interp(a, nd.x, col)))
            fc = _event_at(ev, mid_s, float(np.interp(c, nd.x, col)))
            total += 0.5 * (fa + fc) * (c - a)
        out[i] = total / (hi - lo)
    return out


@pytest.mark.parametrize("gap_cells", [0.3, 1.5, 8.0])          # one dual cell, neighbouring cells, far apart
def test_coincident_records_are_projected_once(gap_cells):
    nd = _nodes()
    dx = nd.x[1] - nd.x[0]
    u1 = float(np.exp(nd.x[20] + 0.1 * dx))
    u2 = float(np.exp(np.log(u1) + gap_cells * dx))
    col = np.linspace(1.0, 2.0, len(nd.s))
    ev = DiscreteEvent((EventRecord(None, u1, 0.4, EventOp.ADD), EventRecord(None, u2, 0.7, EventOp.ADD)))
    got = apply_discrete_event(nd, col, ev, P.CELL_AVERAGE)
    np.testing.assert_allclose(got, _exact_cell_average(nd, col, ev), rtol=0, atol=1e-7)
    if gap_cells < 1.0:
        seq = col
        for rec in ev.records:                                   # the per-record implementation
            seq = apply_discrete_event(nd, seq, DiscreteEvent((rec,)), P.CELL_AVERAGE)
        assert np.max(np.abs(seq - got)) > 1e-3


def test_a_corridor_sharing_a_cell_is_exact():
    nd = _nodes()
    dx = nd.x[1] - nd.x[0]
    lo, hi = float(np.exp(nd.x[20] + 0.1 * dx)), float(np.exp(nd.x[20] + 0.4 * dx))
    col = np.full(len(nd.s), 1.0)
    ev = DiscreteEvent((EventRecord(lo, hi, 0.0, EventOp.SET),))
    np.testing.assert_allclose(apply_discrete_event(nd, col, ev, P.CELL_AVERAGE),
                               _exact_cell_average(nd, col, ev), rtol=0, atol=1e-7)


@pytest.mark.parametrize("name", sorted(cases()))
def test_nodal_keeps_each_solvers_semantics(name, monkeypatch):
    cls, factory, ops = cases()[name]
    env = make_term_env("flat")
    seen = []
    orig = BasePDESolver._apply_discrete_event

    def spy(self, column, x, s, event, *, nodal):
        seen.append(tuple(r.op.value for r in event.records))
        return orig(self, column, x, s, event, nodal=nodal)

    orig_at = BasePDESolver._apply_discrete_event_at

    def spy_at(self, grid, t_idx, x, s, event, *, nodal):  # the per-step entry (nodal: no inner call)
        seen.append(tuple(r.op.value for r in event.records))
        return orig_at(self, grid, t_idx, x, s, event, nodal=nodal)

    monkeypatch.setattr(BasePDESolver, "_apply_discrete_event", spy)
    monkeypatch.setattr(BasePDESolver, "_apply_discrete_event_at", spy_at)
    solver = cls(PDEParams(event_projection="nodal"))
    g = solver.calculate_greeks(factory(), env)
    banked = V070[name]
    assert_close({"price": g["price"], "delta": g["delta"], "gamma": g["gamma"]},
                 {k: banked[k] for k in ("price", "delta", "gamma")})
    if "column" in banked:
        out = run(prepare(cls(PDEParams(event_projection="nodal")), factory(), env,
                          OutputRequest(Operation.SESSION_OUTPUTS, grid=True)))
        assert_close(list(map(float, out.solution.solution_vec)), [float.fromhex(h) for h in banked["column"]])
    if ops is not None:
        terminal, interior = seen[0], seen[1]          # set_terminal_condition runs first, then the 0.5 node
        assert set(terminal) == {ops[1]} and set(interior) == {ops[0]}


@pytest.mark.parametrize("name", ["no_touch|acc", "double_no_touch|acc"])
def test_an_accumulating_no_touch_ignores_its_interior_observation(name):
    cls, factory, _ = cases()[name]
    env = make_term_env("flat")
    acc = cls(PDEParams(event_projection="nodal")).price(factory(), env)
    stop = cls(PDEParams(event_projection="nodal")).price(cases()[name.replace("acc", "stop")][1](), env)
    single = V070[name]["single_observation_price"]
    assert abs(acc - single) < V070[name]["single_observation_tolerance"]
    assert abs(acc - stop) > 10 * V070[name]["single_observation_tolerance"]


def test_a_valuation_date_observation_stays_nodal_under_projection():
    # Review Focus 2: an observation at t = 0 is deterministic; only interior observations project
    _, factory, _ = cases()["valuation_date_only"]
    env = make_term_env("flat")
    cols = []
    for mode in ("nodal", "cell_average"):
        out = run(prepare(BarrierPDESolver(PDEParams(event_projection=mode)), factory(), env,
                          OutputRequest(Operation.SESSION_OUTPUTS, grid=True)))
        cols.append(out.solution.solution_vec.tobytes())
    assert cols[0] == cols[1]
