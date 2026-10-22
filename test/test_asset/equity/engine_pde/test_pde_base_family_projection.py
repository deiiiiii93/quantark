"""The base-family event helper (spec 3A §5.4; decisions 8, 14)."""
import json
from pathlib import Path

import numpy as np
import pytest

from golden_compare import assert_close
from quantark.asset.equity.engine.pde import BarrierPDESolver
from quantark.asset.equity.engine.pde.base_pde_solver import BasePDESolver
from quantark.asset.equity.engine.pde.core.components import CrankNicolson
from quantark.asset.equity.engine.pde.core.prepare import Operation, OutputRequest, prepare, run
from quantark.asset.equity.engine.pde.grid import DiscreteEvent, EventOp, EventRecord, Nodes, apply_discrete_event
from quantark.asset.equity.engine.pde.grid.events import _dual_cell_edges
from quantark.asset.equity.param import PDEParams
from quantark.util.enum.engine_enums import EventProjectionMode as P
from quantark.util.exceptions import ValidationError

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


#: The 0.7.0 core stepper: flip (f) made half-step (1, 1) the default where monitoring is discrete.
V070_STEPPER = CrankNicolson(damping="full_step", terminal_damping_steps=1, event_damping_steps=2)


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
    solver = cls(PDEParams(event_projection="nodal", stepper=V070_STEPPER))
    if name in ("no_touch|acc", "double_no_touch|acc"):
        # user ruling 2026-10-07: ACCUMULATE has no meaning for a no-touch; the 0.7.0 product is refused
        with pytest.raises(ValidationError, match="never hit up to and including expiry"):
            factory()
        return
    if name.startswith(("barrier_di_put|", "double_ki|")):
        # spec 3C decision 8: a knock-in has one rebate and first-hit monitoring; these 0.7.0 knock-ins carry per-record
        # payoffs on a rebate of 0 (and '|acc' accumulates), so they are refused rather than priced
        with pytest.raises(ValidationError):
            solver.calculate_greeks(factory(), env)
        return
    if name == "no_touch|stop":
        # user decision 2026-10-15: a no-touch has one rebate; this 0.7.0 no-touch's records pay 2 and 3 on a rebate
        # of 1, so it is refused rather than priced
        with pytest.raises(ValidationError, match="no-touch has one rebate"):
            solver.calculate_greeks(factory(), env)
        return
    g = solver.calculate_greeks(factory(), env)
    banked = V070[name]
    # flip (g) moves every put's far-field row (forward over [t, T], spec 3A §5.10): a put keeps its
    # 0.7.0 event semantics (the ops below) but not its 0.7.0 values
    put = "_put" in name
    # spec 3C decision 9: a discretely monitored single knock-out's far-field row beyond its barrier is the certain
    # knock-out at the next observation. It moves these 0.7.0 values, toward the exact recursion: '|dates' by
    # -4.3e-8 / -4.3e-9 / -2.6e-10 (price / delta / gamma), '|stop' (next observation half a year away) by
    # -1.94e-4 / -1.86e-5 / -8.8e-7 (0.7.0 +2.225e-2 off exact, now +2.206e-2: the nodal stepper's own error)
    far_field = {"barrier_uo_call|dates": (1e-7, 1e-8, 1e-9), "barrier_uo_call|stop": (3e-4, 3e-5, 2e-6)}.get(name)
    if far_field:
        for k, bound in zip(("price", "delta", "gamma"), far_field):
            assert abs(g[k] - banked[k]) <= bound, (k, g[k], banked[k])
    elif not put:
        assert_close({"price": g["price"], "delta": g["delta"], "gamma": g["gamma"]},
                     {k: banked[k] for k in ("price", "delta", "gamma")})
    if "column" in banked and not put and not far_field:      # the far-field row reshapes the surface's edge
        out = run(prepare(cls(PDEParams(event_projection="nodal", stepper=V070_STEPPER)), factory(), env,
                          OutputRequest(Operation.SESSION_OUTPUTS, grid=True)))
        live, golden = list(map(float, out.solution.solution_vec)), [float.fromhex(h) for h in banked["column"]]
        if name == "valuation_date_only":
            # spec 3C decision 25: the valuation-date observation is the readout's decision, never written into
            # the nodes; the surface is the continuation, equal to 0.7.0's where its record is not breached
            keep = out.solution.s_vec < factory().barrier
            live = [v for v, k in zip(live, keep) if k]
            golden = [v for v, k in zip(golden, keep) if k]
        assert_close(live, golden)
    if ops is not None:
        terminal, interior = seen[0], seen[1]          # set_terminal_condition runs first, then the 0.5 node
        assert set(terminal) == {ops[1]} and set(interior) == {ops[0]}


@pytest.mark.parametrize("name", ["no_touch|acc", "double_no_touch|acc"])
def test_an_accumulating_no_touch_is_refused(name):
    """0.7.0 priced it ignoring its interior observations (only maturity ended it); the mode has no meaning on a
    no-touch, which pays only if never hit up to and including expiry (user ruling 2026-10-07)."""
    _, factory, _ = cases()[name]
    with pytest.raises(ValidationError, match="never hit up to and including expiry"):
        factory()


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


# ---- flip (c): the dual-cell projection is the base family's default (spec 3A §5.4, §4.6) ----------

from pde_3a.products import PRODUCTS, HORIZONS, dates, env as study_env  # noqa: E402
from quantark.asset.equity.engine import PDEEngine  # noqa: E402
from quantark.asset.equity.riskmeasures import GreeksCalculator  # noqa: E402

REF = json.loads((Path(__file__).parent / "pde_3a" / "fixtures" / "base_family_reference.json").read_text())
V070_GREEKS = json.loads((Path(__file__).parent / "pde_3a" / "fixtures" / "base_family_nodal_v070.json").read_text())
QUANTITIES = ("pv", "delta", "gamma", "bump_delta", "bump_gamma")

#: Twice the largest error spec §4.6 reports at 400 points, per horizon and quantity. "projection" is
#: flip (c) alone; "projection_half_step" is the 3A default after flip (f). Task 14 switches STAGE.
SPEC_46 = {
    "projection": {
        "1m": dict(pv=1.5e-4, delta=1.8e-5, gamma=1.9e-6, bump_delta=1.7e-5, bump_gamma=2.8e-5),
        "5d": dict(pv=3.2e-4, delta=1.8e-4, gamma=7.2e-5, bump_delta=1.2e-4, bump_gamma=1.2e-4),
        "1d": dict(pv=7.9e-3, delta=3.4e-3, gamma=9.1e-3, bump_delta=2.0e-3, bump_gamma=6.4e-3),
    },
    "projection_half_step": {
        "1m": dict(pv=1.7e-4, delta=9.8e-6, gamma=1.7e-6, bump_delta=1.9e-5, bump_gamma=2.8e-5),
        "5d": dict(pv=2.1e-4, delta=4.9e-5, gamma=2.2e-5, bump_delta=5.3e-5, bump_gamma=5.6e-5),
        "1d": dict(pv=1.1e-3, delta=8.5e-4, gamma=1.8e-3, bump_delta=4.9e-4, bump_gamma=8.5e-4),
    },
}
STAGE = "projection_half_step"
KO_AND_TOUCH = sorted(n for n in PRODUCTS if n != "di_put")
CELLS = [(r["product"], r["horizon"], r["spot"]) for r in REF["refs"] if r["product"] in KO_AND_TOUCH]


def _engine_values(name, horizon, spot, params):
    product = PRODUCTS[name][1](dates(HORIZONS[horizon]))
    env = study_env(spot=spot)
    g = PDEEngine(params=params).calculate_greeks(product, env)
    b = GreeksCalculator().calculate_numerical_greeks(product=product, pricing_env=env,
                                                      engine=PDEEngine(params=params), greeks=["delta", "gamma"])
    return dict(pv=g["price"], delta=g["delta"], gamma=g["gamma"], bump_delta=b["delta"], bump_gamma=b["gamma"])


@pytest.mark.parametrize("name,horizon,spot", CELLS)
def test_the_default_meets_the_deterministic_reference(name, horizon, spot):
    ref = next(r for r in REF["refs"] if (r["product"], r["horizon"], r["spot"]) == (name, horizon, spot))
    got = _engine_values(name, horizon, spot, PDEParams())
    tol = SPEC_46[STAGE][horizon]
    for q in QUANTITIES:
        assert abs(got[q] - ref[q]) <= 2.0 * tol[q], (q, got[q] - ref[q])


def _first_per_product_horizon(rows, n_products=4):
    """The first banked row per (product, horizon), for the first ``n_products`` products in banked
    order: four products x three horizons (the first-twelve slice covers only two products)."""
    seen, products = {}, []
    for r in rows:
        if r["product"] not in products:
            products.append(r["product"])
        if r["product"] in products[:n_products]:
            seen.setdefault((r["product"], r["horizon"]), r)
    return list(seen.values())


@pytest.mark.parametrize("row", _first_per_product_horizon([r for r in V070_GREEKS if r["product"] in KO_AND_TOUCH]),
                         ids=lambda r: f"{r['product']}|{r['horizon']}|{r['spot']}")
def test_nodal_full_step_is_the_070_value(row):
    from quantark.asset.equity.engine.pde.core.components import CrankNicolson
    params = PDEParams(event_projection="nodal", stepper=CrankNicolson(
        damping="full_step", terminal_damping_steps=1, event_damping_steps=2))
    got = _engine_values(row["product"], row["horizon"], row["spot"], params)
    assert_close(got, {q: row[q] for q in QUANTITIES})


#: capture_book._coincident_product (Task 1, measured): log(130) and log(130.01) share one dual cell of
#: the standard layout (a cell 2.6e-3 wide in log spot against a 7.7e-5 gap).
COINCIDENT_GAP = 0.01
#: Twice the error measured on the flip (c) commit against the closed form below (+8.873e-05).
COINCIDENT_TOL = 1.8e-4


def test_coincident_records_through_the_product_meet_the_closed_form():
    from math import exp, log, sqrt

    from scipy.special import ndtr

    from quantark.asset.equity.product.option import BarrierOption
    from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
    from quantark.util.enum import BarrierType, ObservationAggregation, ObservationType, OptionType

    g = COINCIDENT_GAP                        # the gap Task 1 measured: both thresholds in one dual cell
    recs = [(0.5, 130.0, 2.0), (0.5, 130.0 + g, 3.0), (1.0, 130.0, 1.0)]
    product = BarrierOption(
        strike=100.0, option_type=OptionType.CALL, barrier=130.0, barrier_type=BarrierType.UP_OUT,
        maturity=1.0, rebate=0.0, pay_at_hit=False, observation_type=ObservationType.DISCRETE,
        observation_schedule=ObservationSchedule(
            records=[ObservationRecord(observation_time=t, barrier=b, payoff=c) for t, b, c in recs],
            aggregation_mode=ObservationAggregation.ACCUMULATE))
    s, vol, r, q = 100.0, 0.22, 0.025, 0.03
    env = study_env(spot=s)

    def d2(k, t):
        return (log(s / k) + (r - q - 0.5 * vol * vol) * t) / (vol * sqrt(t))

    call = s * exp(-q) * ndtr(d2(100.0, 1.0) + vol) - 100.0 * exp(-r) * ndtr(d2(100.0, 1.0))
    digitals = sum(c * exp(-r * 1.0) * ndtr(d2(b, t)) for t, b, c in recs)   # paid at maturity
    exact = call + digitals
    got = BarrierPDESolver(PDEParams()).price(product, env)
    assert abs(got - exact) <= COINCIDENT_TOL


DI_PUT_CELLS = [(r["product"], r["horizon"], r["spot"]) for r in REF["refs"] if r["product"] == "di_put"]


@pytest.mark.parametrize("name,horizon,spot", DI_PUT_CELLS)
def test_the_down_and_in_put_meets_the_reference_with_a_forward_far_field_row(name, horizon, spot):
    # flip (g): the vanilla leg's put row gains its forward term; within twice spec 3A §4.10's patched errors.
    # Bump Greeks are not asserted: the knock-in bump context raises on this tree (spec 2 handoff).
    ref = next(r for r in REF["refs"] if (r["product"], r["horizon"], r["spot"]) == (name, horizon, spot))
    product = PRODUCTS[name][1](dates(HORIZONS[horizon]))
    g = PDEEngine(params=PDEParams()).calculate_greeks(product, study_env(spot=spot))
    # the larger of twice spec 4.10's patched errors and twice the base family's own bound at this
    # horizon: one day out, the knock-out leg's projection error dominates (1d 87.55: delta 4.8e-4)
    tol = SPEC_46[STAGE][horizon]
    assert abs(g["price"] - ref["pv"]) <= max(2.0 * 2.0e-4, 2.0 * tol["pv"])
    assert abs(g["delta"] - ref["delta"]) <= max(2.0 * 1e-4, 2.0 * tol["delta"])
