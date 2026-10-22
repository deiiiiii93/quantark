"""Regime-aware v_drift_scheme selection (decision C-G6).

The 2026-08-10 cell matrix measured something the node counts alone hide: the
donor-cell fallback engages for two completely different reasons.

In every benign regime it engages on exactly two nodes, at v = 2e-06 and
1.4e-05 -- the nodes nearest zero, where diffusion 0.5*sig^2*v vanishes while
drift kappa*theta stays finite, so the local Peclet number diverges *at any
resolution*. With v0 = theta = 0.04 the process never visits that corner, and
the measured cost of a donor cell there is below 5e-05 futures contracts:
`centered` and `adaptive_upwind` agree to every printed digit.

In sigma_collapse it engages on 132 of 133 nodes, spanning v = 0.00129 to 0.482
-- the whole live domain, containing both theta and v0 as grid nodes. That costs
0.115 contracts against a +/-0.10 aggregate bound.

So a node-count or Peclet threshold would conflate an unavoidable coordinate
singularity with genuine convection dominance. `"auto"` asks the separating
question instead: does the non-monotone region reach the variance scale the
process actually occupies? On the certification matrix that criterion clears by
2857x on one side and 158x on the other, so nothing here is tuned to the data.

Selecting the more accurate scheme per regime beats any single scheme: upwind is
strictly closer to the centered reference wherever the fallback stays in the
corner, and SL is the only correct answer where it does not.
"""

import numpy as np
import pytest

from quantark.util.exceptions import ValidationError
from quantark.volmodels.adi_core import HestonSLVADICore
from quantark.volmodels.heston import HestonParams

# theta far below v0 with a near-zero vol-of-vol: CIR drift transports the state
# across the grid while diffusion is negligible.
SIGMA_COLLAPSE = HestonParams(
    v0=0.14027, kappa=3.0, theta=0.00306, sigma=0.00311, rho=-0.5
)
# v0 == theta, ordinary vol-of-vol: the fallback is confined to the v->0 corner.
ORDINARY = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.30, rho=-0.55)
# Strongly Feller-violated; measured to need no fallback at all.
LOW_FELLER = HestonParams(v0=0.04, kappa=0.6, theta=0.09, sigma=1.4, rho=-0.55)


def _core(params, scheme, *, n_v=30, n_t=20, **kwargs):
    kwargs.setdefault(
        "variance_grid_mode",
        "path_focused" if params is SIGMA_COLLAPSE else "power",
    )
    return HestonSLVADICore(
        100.0,
        100.0,
        2.0,
        0.03,
        0.01,
        params,
        31,
        n_v,
        n_t,
        grid_style="concentrated",
        v0_boundary="degenerate_pde",
        v_drift_scheme=scheme,
        **kwargs,
    )


# --- registration ---------------------------------------------------------


def test_auto_is_an_accepted_scheme():
    assert _core(ORDINARY, "auto").v_drift_scheme in (
        "adaptive_upwind",
        "semi_lagrangian",
    )


def test_unknown_scheme_is_still_rejected():
    with pytest.raises(ValidationError, match="v_drift_scheme"):
        _core(ORDINARY, "teleporting")


def test_default_is_still_adaptive_upwind():
    """Auto must be opt-in: the library default cannot start selecting."""
    core = HestonSLVADICore(
        100.0, 100.0, 2.0, 0.03, 0.01, ORDINARY, 31, 30, 20,
        grid_style="concentrated", v0_boundary="degenerate_pde",
    )
    assert core.v_drift_scheme == "adaptive_upwind"
    assert core.requested_v_drift_scheme == "adaptive_upwind"


# --- the selection itself -------------------------------------------------


def test_auto_keeps_upwind_when_the_fallback_stays_in_the_v0_corner():
    core = _core(ORDINARY, "auto")
    assert core.v_drift_scheme == "adaptive_upwind"
    assert core.requested_v_drift_scheme == "auto"


def test_auto_keeps_upwind_when_no_row_needs_a_fallback():
    core = _core(LOW_FELLER, "auto")
    assert core.v_drift_scheme == "adaptive_upwind"


def test_auto_selects_semi_lagrangian_when_convection_reaches_the_live_scale():
    core = _core(SIGMA_COLLAPSE, "auto")
    assert core.v_drift_scheme == "semi_lagrangian"


def test_the_separating_fact_is_where_the_fallback_sits_not_how_many():
    """Guard the criterion against being re-derived as a node count.

    Both regimes below have non-monotone rows. What distinguishes them is
    whether those rows reach min(v0, theta), so assert that geometry directly.
    """
    ordinary = _core(ORDINARY, "adaptive_upwind")
    collapse = _core(SIGMA_COLLAPSE, "adaptive_upwind")

    for core, params, expected_intrusion in (
        (ordinary, ORDINARY, False),
        (collapse, SIGMA_COLLAPSE, True),
    ):
        *_, fallback = core._build_v_generator_coefficients()
        assert np.any(fallback), "both regimes must have some non-monotone row"
        live_scale = min(params.v0, params.theta)
        reaches = bool(np.any(core.V_grid[1:-1][fallback] >= live_scale))
        assert reaches is expected_intrusion


def test_selection_is_deterministic():
    assert (
        _core(SIGMA_COLLAPSE, "auto").v_drift_scheme
        == _core(SIGMA_COLLAPSE, "auto").v_drift_scheme
        == "semi_lagrangian"
    )


def test_auto_refuses_when_the_live_variance_scale_is_degenerate():
    """No invented semantics: with v0 = theta = 0 the criterion has no scale."""
    pinned = HestonParams(v0=0.0, kappa=2.0, theta=0.0, sigma=0.30, rho=-0.55)
    with pytest.raises(ValidationError, match="v_drift_scheme='auto'"):
        _core(pinned, "auto")


# --- auto must be exactly its resolution, not an approximation of it ------


def _march(params, scheme):
    core = _core(params, scheme, n_v=40, n_t=60)
    return core.solve(is_call=False, scheme="cs", theta=0.5, rannacher=2)


def test_auto_is_bitwise_identical_to_the_scheme_it_resolves_to():
    """The benign path especially must not move by a single bit."""
    for params, expected in (
        (ORDINARY, "adaptive_upwind"),
        (LOW_FELLER, "adaptive_upwind"),
        (SIGMA_COLLAPSE, "semi_lagrangian"),
    ):
        auto = _march(params, "auto")
        explicit = _march(params, expected)
        assert auto.tobytes() == explicit.tobytes(), expected


def test_auto_reports_both_the_request_and_the_resolution():
    diagnostics = _core(SIGMA_COLLAPSE, "auto").variance_operator_diagnostics()
    assert diagnostics["scheme"] == "semi_lagrangian"
    assert diagnostics["requested_scheme"] == "auto"
    # The resolution must carry its own justification into the evidence.
    assert diagnostics["auto_selection"]["live_variance_scale"] == pytest.approx(
        0.00306
    )
    assert diagnostics["auto_selection"]["fallback_reaches_live_scale"] is True


def test_explicit_schemes_report_themselves_as_requested():
    diagnostics = _core(ORDINARY, "adaptive_upwind").variance_operator_diagnostics()
    assert diagnostics["scheme"] == "adaptive_upwind"
    assert diagnostics["requested_scheme"] == "adaptive_upwind"
    assert diagnostics["auto_selection"] is None


# --- reachability from the solver layer ----------------------------------


def test_solver_layer_accepts_auto():
    """The solver keeps its own whitelist; WS-C proved it can go stale."""
    from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import (
        HestonSnowballPDESolver,
    )

    solver = HestonSnowballPDESolver(
        SIGMA_COLLAPSE,
        grid_style="concentrated",
        v0_boundary="degenerate_pde",
        v_drift_scheme="auto",
    )
    assert solver.v_drift_scheme == "auto"
