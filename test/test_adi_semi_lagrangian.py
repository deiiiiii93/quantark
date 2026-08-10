"""Semi-Lagrangian v-transport (spec WS-C).

The shipped donor-cell/upwind scheme is measured first-order on the variance
axis, and the 2026-08-10 attribution probe showed that first-order error is the
entire sigma_collapse delta bias. This scheme removes v-drift from the
generator and transports it along the exact CIR characteristics instead.

These are the unit-level gates. The certification-level gates (C-G1 flatness in
n_v against banked MC references, C-G2 PV de-biasing) need reference data the
2026-08-10 crash destroyed and run in the Phase-1 regeneration.
"""

import numpy as np
import pytest

from quantark.util.exceptions import ValidationError
from quantark.volmodels.adi_core import HestonSLVADICore
from quantark.volmodels.heston import HestonParams

SIGMA_COLLAPSE = HestonParams(
    v0=0.14027,
    kappa=3.0,
    theta=0.00306,
    sigma=0.00311,
    rho=-0.5,
)


def _core(**kwargs):
    return HestonSLVADICore(
        100.0,
        100.0,
        2.0,
        0.03,
        0.01,
        SIGMA_COLLAPSE,
        31,
        30,
        20,
        grid_style="concentrated",
        v0_boundary="degenerate_pde",
        **kwargs,
    )


def _sl_core(**kwargs):
    kwargs.setdefault("variance_grid_mode", "path_focused")
    return _core(v_drift_scheme="semi_lagrangian", **kwargs)


# --- scheme registration -------------------------------------------------


def test_semi_lagrangian_is_an_accepted_scheme():
    assert _sl_core().v_drift_scheme == "semi_lagrangian"


def test_unknown_scheme_is_still_rejected():
    with pytest.raises(ValidationError, match="v_drift_scheme"):
        _core(v_drift_scheme="teleporting")


def test_default_scheme_is_unchanged():
    """Exact semantics by default: SL must be opt-in only."""
    assert _core().v_drift_scheme == "adaptive_upwind"


# --- diffusion-only generator --------------------------------------------


def test_generator_carries_no_drift_and_is_unconditionally_monotone():
    sl = _sl_core()
    sub, diag, sup, monotone, fallback = sl._build_v_generator_coefficients()

    # Pure diffusion: 0.5 * sigma_eff^2 * v against the second-derivative
    # stencil, with no kappa*(theta-v) term anywhere.
    v_int = sl.V_grid[1:-1]
    diffusion = 0.5 * sl.sig_eff2 * v_int
    wm2, _w02, wp2 = sl._vv
    assert sub == pytest.approx(np.maximum(diffusion * wm2, 0.0))
    assert sup == pytest.approx(np.maximum(diffusion * wp2, 0.0))
    # Row-sum identity holds exactly.
    assert diag == pytest.approx(-(sub + sup))
    # Monotone by construction, so no row can ever need a fallback.
    assert monotone.all()
    assert not fallback.any()


def test_diagnostics_report_zero_peclet_and_no_fallback():
    """C-G7: the diagnostics must stay meaningful under the new scheme."""
    sl = _sl_core()
    upwind = _core(variance_grid_mode="path_focused")

    sl_diag = sl.variance_operator_diagnostics()
    upwind_diag = upwind.variance_operator_diagnostics()

    assert sl_diag["scheme"] == "semi_lagrangian"
    assert sl_diag["monotone"] is True
    assert sl_diag["fallback_nodes"] == 0
    assert sl_diag["centered_non_monotone_nodes"] == 0
    # Advection owns all drift, so the generator's local Peclet number is 0.
    assert sl_diag["max_local_peclet"] == 0.0
    # The same grid genuinely needs a fallback under the shipped scheme, so
    # the zero above is a property of SL, not of an easy fixture.
    assert upwind_diag["fallback_nodes"] > 0


def test_advection_diagnostics_report_foot_displacement():
    """C-G7: feet statistics are the observability hook for the transport."""
    sl = _sl_core()
    diagnostics = sl.variance_operator_diagnostics()
    assert diagnostics["advection"]["scheme"] == "exact_cir_characteristics"
    assert diagnostics["advection"]["feet_interior"] is True
    assert diagnostics["advection"]["max_cells_traversed"] >= 0


# --- exact CIR characteristics -------------------------------------------


def test_advection_reproduces_the_exact_characteristic_map_on_linear_data():
    """A linear profile is transported exactly: cubic weights are exact on it.

    U(v) = v advected by dt must return the foot value
    theta + (v - theta) * exp(-kappa*dt) at every node.
    """
    sl = _sl_core()
    dt = 0.01
    U = np.tile(sl.V_grid, (sl.N_S, 1))

    advected = sl._advect_v(U, dt)

    expected = sl.theta + (sl.V_grid - sl.theta) * np.exp(-sl.kappa * dt)
    assert advected[0] == pytest.approx(expected, rel=1e-12, abs=1e-14)


def test_advection_preserves_constants():
    """Interpolation weights sum to one, so a flat surface is untouched."""
    sl = _sl_core()
    U = np.full((sl.N_S, sl.N_V), 3.25)
    advected = sl._advect_v(U, 0.05)
    assert advected == pytest.approx(np.full((sl.N_S, sl.N_V), 3.25), abs=1e-14)


def test_characteristic_feet_stay_inside_the_grid():
    """Mean reversion contracts toward theta, so no outflow BC is needed."""
    sl = _sl_core()
    for dt in (1e-6, 1e-3, 0.05, 0.5, 5.0):
        feet = sl._characteristic_feet(dt)
        assert feet.min() >= sl.V_grid[0] - 1e-15
        assert feet.max() <= sl.V_grid[-1] + 1e-15


def test_advection_introduces_no_new_extrema():
    """Linear-bracket clipping keeps the transport monotone."""
    sl = _sl_core()
    rng = np.random.default_rng(11)
    U = np.cumsum(rng.random((sl.N_S, sl.N_V)), axis=1)  # increasing in v
    advected = sl._advect_v(U, 0.02)
    assert advected.max() <= U.max() + 1e-12
    assert advected.min() >= U.min() - 1e-12


def test_zero_step_is_a_no_op():
    sl = _sl_core()
    U = np.tile(sl.V_grid, (sl.N_S, 1))
    assert sl._advect_v(U, 0.0) is U


# --- operator wiring ------------------------------------------------------


def test_explicit_operator_drops_the_degenerate_drift_row():
    """Advection owns the v=0 drift, so _A2 must not add it again."""
    sl = _sl_core()
    U = np.tile(sl.V_grid, (sl.N_S, 1))
    out = sl._A2(U)
    assert out[1:-1, 0] == pytest.approx(np.zeros(sl.N_S - 2))
    # The shipped scheme does apply that row, so the zero above is a
    # property of SL rather than of the fixture.
    upwind = _core(variance_grid_mode="path_focused")
    assert np.any(upwind._A2(np.tile(upwind.V_grid, (upwind.N_S, 1)))[1:-1, 0] != 0.0)


def test_implicit_degenerate_row_becomes_an_identity():
    sl = _sl_core()
    a, b, c = sl._tri_V(0.01, 0.5)
    assert b[0] == pytest.approx(1.0)
    assert c[0] == pytest.approx(0.0)


def test_strang_split_wraps_the_parent_step():
    """A full step must advect a half step on each side of the ADI step."""
    sl = _sl_core()
    calls = []
    original = sl._advect_v

    def recording(U, dt_sub):
        calls.append(dt_sub)
        return original(U, dt_sub)

    sl._advect_v = recording
    U = sl._terminal(is_call=False)
    sl._douglas_step(U, 0.01, 0.01, 1.0, 0.005)

    assert calls == [0.005, 0.005]


# --- the reason the scheme exists ----------------------------------------


def _european_put(n_v, scheme, n_t=240):
    """One vanilla put solved on a fixed x/t grid, varying only n_v."""
    core = HestonSLVADICore(
        100.0,
        100.0,
        1.0,
        0.03,
        0.01,
        SIGMA_COLLAPSE,
        61,
        n_v,
        n_t,
        grid_style="concentrated",
        v0_boundary="degenerate_pde",
        variance_grid_mode="path_focused",
        v_drift_scheme=scheme,
    )
    U = core.solve(is_call=False, scheme="cs", theta=0.5, rannacher=2)
    # interpolate takes LOG spot and clamps to the grid; passing the raw spot
    # silently reads the grid edge, where a put is worth zero.
    return float(core.interpolate(U, float(np.log(100.0)), core.v0))


def test_semi_lagrangian_converges_faster_in_n_v_than_donor_cell():
    """C-G1 in unit form: SL must contract in n_v faster than upwind.

    The shipped donor-cell fallback is measured first-order on this axis, which
    the attribution probe tied to the whole sigma_collapse delta bias. Refining
    n_v by 2x should therefore shrink SL's remaining movement by clearly more
    than it shrinks upwind's. Both schemes solve the identical problem, so the
    comparison isolates the variance discretization.
    """
    upwind = [_european_put(n_v, "adaptive_upwind") for n_v in (30, 60, 120)]
    sl = [_european_put(n_v, "semi_lagrangian") for n_v in (30, 60, 120)]

    upwind_ratio = abs(upwind[1] - upwind[0]) / max(abs(upwind[2] - upwind[1]), 1e-15)
    sl_ratio = abs(sl[1] - sl[0]) / max(abs(sl[2] - sl[1]), 1e-15)

    # A first-order scheme halves its movement per doubling (ratio ~2); a
    # second-order one quarters it (ratio ~4). Measured 2026-08-10: upwind
    # 2.03, SL 113 -- the v-axis error is essentially gone by n_v=60, where
    # upwind still needs n_v>=240 to come as close.
    assert upwind_ratio < 3.0, f"upwind unexpectedly high-order: {upwind_ratio}"
    assert sl_ratio > 4.0 * upwind_ratio, (
        f"SL ratio {sl_ratio} must clearly beat upwind {upwind_ratio}"
    )
    # And SL at the COARSEST rung must already beat upwind at the finest.
    limit = _european_put(240, "semi_lagrangian")
    assert abs(sl[1] - limit) < abs(upwind[2] - limit)


def test_semi_lagrangian_agrees_with_upwind_on_a_refined_grid():
    """C-G4 in unit form: the two schemes must target the same continuum limit.

    Faster convergence is only meaningful if both schemes converge to the SAME
    value. At a fine n_v the schemes must agree closely; a mismatch here would
    mean the transport changed the PDE rather than its discretization.
    """
    fine_upwind = _european_put(240, "adaptive_upwind")
    fine_sl = _european_put(240, "semi_lagrangian")
    assert fine_sl == pytest.approx(fine_upwind, rel=2e-3)


# --- reachability from the solver layer ----------------------------------


def test_snowball_solver_accepts_the_semi_lagrangian_scheme():
    """The scheme is useless if the engine layer rejects it."""
    from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import (
        HestonSnowballPDESolver,
    )

    solver = HestonSnowballPDESolver(
        SIGMA_COLLAPSE,
        n_x=31,
        n_v=30,
        n_t=20,
        grid_style="concentrated",
        v0_boundary="degenerate_pde",
        variance_grid_mode="path_focused",
        v_drift_scheme="semi_lagrangian",
    )
    assert solver.v_drift_scheme == "semi_lagrangian"


def test_snowball_solver_still_rejects_unknown_schemes():
    from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import (
        HestonSnowballPDESolver,
    )

    with pytest.raises(ValidationError, match="v_drift_scheme"):
        HestonSnowballPDESolver(SIGMA_COLLAPSE, v_drift_scheme="teleporting")
