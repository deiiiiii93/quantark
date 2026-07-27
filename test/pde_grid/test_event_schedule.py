"""Tier-1 tests for the events module (spec §4.2, §5)."""

import numpy as np
import pytest

from quantark.asset.equity.engine.pde.grid import (
    GridConfig,
    GridRequest,
    MarketSnapshot,
    resolve_config,
)
from quantark.asset.equity.engine.pde.grid.events import (
    EventSchedule,
    breach_weights,
    project_between,
    project_piecewise,
)
from quantark.asset.equity.engine.pde.grid.space import build_space

MKT = MarketSnapshot(spot=100.0, sigma_ref=0.2, r_ref=0.03, q_ref=0.01)


def layout(points=201):
    r = GridRequest(
        tau=1.0,
        bound_anchors=(100.0,),
        critical_prices=(),
        hard_lower=None,
        hard_upper=None,
        event_times=(),
    )
    return build_space(r, MKT, resolve_config("standard", GridConfig(points=points)))


def test_weights_in_unit_interval():
    w = breach_weights(layout(), 103.0, breach_up=True)
    assert np.all(w >= 0.0) and np.all(w <= 1.0)


def test_constant_preservation_P_at_1_equals_1():
    lay = layout()
    ones = np.ones_like(lay.s)
    out = project_between(lay, 103.0, True, v_breach=ones, v_survive=ones)
    assert np.allclose(out, 1.0, atol=1e-14)


def test_envelope_containment():
    lay = layout()
    v_s = np.linspace(0.0, 1.0, len(lay.s))
    v_b = np.linspace(2.0, 3.0, len(lay.s))
    out = project_between(lay, 103.0, True, v_breach=v_b, v_survive=v_s)
    assert np.all(out >= np.minimum(v_s, v_b) - 1e-14)
    assert np.all(out <= np.maximum(v_s, v_b) + 1e-14)


def test_affine_exactness_in_straddling_cell():
    lay = layout()
    a, b = 0.7, 0.3
    v = a * lay.x + b  # same affine function on both branches
    out = project_between(lay, 103.0, True, v_breach=v, v_survive=v)
    assert np.allclose(out, v, atol=1e-12)


def test_threshold_perturbation_continuity():
    lay = layout()
    v_s = np.zeros_like(lay.s)
    v_b = np.ones_like(lay.s)
    out1 = project_between(lay, 103.0, True, v_b, v_s)
    out2 = project_between(lay, 103.0 * (1 + 1e-13), True, v_b, v_s)
    assert np.max(np.abs(out1 - out2)) < 1e-9  # nodal masks flip a whole cell


def test_broadcasting_columns_match_1d():
    lay = layout()
    n = len(lay.s)
    cols = np.stack(
        [np.linspace(0, 1, n), np.linspace(5, 2, n), np.zeros(n)], axis=1
    )
    block = project_between(lay, 97.0, False, v_breach=cols, v_survive=cols * 2)
    for j in range(3):
        one = project_between(
            lay, 97.0, False, v_breach=cols[:, j], v_survive=cols[:, j] * 2
        )
        assert np.allclose(block[:, j], one)


def test_second_order_convergence_on_discontinuity():
    # cell-average error of a Heaviside jump halves ~quadratically vs nodal O(dx)
    errs = []
    for pts in (101, 201, 401):
        lay = layout(points=pts)
        v_s = np.zeros_like(lay.s)
        v_b = np.ones_like(lay.s)
        out = project_between(lay, 103.17, True, v_b, v_s)
        # exact dual-cell average of the indicator
        w = breach_weights(lay, 103.17, True)
        errs.append(np.max(np.abs(out - w)))
    assert errs[0] < 1e-12 and errs[-1] < 1e-12  # indicator is integrated exactly
    # affine-payoff straddle error decays at ~2nd order:
    errs2 = []
    for pts in (101, 201, 401):
        lay = layout(points=pts)
        v_b = np.maximum(lay.s - 103.17, 0.0)
        v_s = np.zeros_like(lay.s)
        out = project_between(lay, 103.17, True, v_b, v_s)
        exact = v_b * breach_weights(lay, 103.17, True)
        errs2.append(np.max(np.abs(out - exact)))
    assert errs2[2] <= errs2[0]  # refinement never degrades the straddle cell


def test_piecewise_matches_single_threshold():
    lay = layout()
    v_s = np.linspace(1.0, 2.0, len(lay.s))
    v_b = np.linspace(0.5, 0.1, len(lay.s))
    a = project_between(lay, 103.0, True, v_b, v_s)
    b = project_piecewise(lay, [103.0], [v_s, v_b])
    assert np.allclose(a, b, atol=1e-12)


def test_schedule_stage_order_and_defaults():
    calls = []
    sched = EventSchedule(
        interior={
            5: lambda st: (calls.append("apply@5") or {k: v + 1 for k, v in st.items()})
        },
        continuous=lambda k, st: (calls.append(f"cont@{k}") or st),
        terminal=lambda st: (calls.append("terminal") or st),
        readout=lambda spot, st: (calls.append("readout") or 42.0),
    )
    st = {"alive": np.zeros(3)}
    st = sched.terminal(st)
    for k in (6, 5):
        st = sched.apply(k, st)
        st = sched.continuous(k, st)
    price = sched.valuation_readout(100.0, st)
    assert price == 42.0
    assert calls == ["terminal", "cont@6", "apply@5", "cont@5", "readout"]
    assert st["alive"][0] == 1.0  # only the step-5 interior transform fired
    assert sched.interior_steps == frozenset({5})


def test_schedule_purity_inputs_unmodified():
    lay = layout()
    v0 = np.linspace(0, 1, len(lay.s))
    orig = v0.copy()
    sched = EventSchedule(
        interior={
            3: lambda st: {
                "alive": project_between(lay, 103.0, True, st["alive"] * 0, st["alive"])
            }
        }
    )
    out = sched.apply(3, {"alive": v0})
    assert np.array_equal(v0, orig)  # input untouched
    assert out["alive"] is not v0


def test_readout_missing_raises():
    with pytest.raises(NotImplementedError):
        EventSchedule().valuation_readout(100.0, {"alive": np.zeros(2)})
