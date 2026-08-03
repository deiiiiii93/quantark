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


def test_adaptive_variance_operator_falls_back_only_on_nonmonotone_rows():
    centered = _core(
        variance_grid_mode="path_focused",
        v_drift_scheme="centered",
    )
    adaptive = _core(
        variance_grid_mode="path_focused",
        v_drift_scheme="adaptive_upwind",
    )

    centered_diag = centered.variance_operator_diagnostics()
    adaptive_diag = adaptive.variance_operator_diagnostics()

    assert centered_diag["centered_non_monotone_nodes"] > 0
    assert centered_diag["fallback_nodes"] == 0
    assert centered_diag["monotone"] is False
    assert adaptive_diag["fallback_nodes"] == centered_diag[
        "centered_non_monotone_nodes"
    ]
    assert adaptive_diag["monotone"] is True
    assert adaptive_diag["min_lower_offdiag"] >= -1e-12
    assert adaptive_diag["min_upper_offdiag"] >= -1e-12


def test_explicit_and_implicit_v_stages_use_identical_coefficients():
    core = _core(
        variance_grid_mode="path_focused",
        v_drift_scheme="adaptive_upwind",
    )
    rng = np.random.default_rng(20260803)
    values = rng.normal(size=(core.N_S, core.N_V))
    sub, diag, sup, _, _ = core._build_v_generator_coefficients()

    explicit = core._A2(values)[1:-1, 1:-1]
    expected = (
        values[1:-1, :-2] * sub
        + values[1:-1, 1:-1] * diag
        + values[1:-1, 2:] * sup
    )
    assert explicit == pytest.approx(expected)

    theta_loc = 0.5
    dt_step = 0.013
    lower, centre, upper = core._tri_V(dt_step, theta_loc)
    assert lower[1:-1] == pytest.approx(-theta_loc * dt_step * sub)
    assert centre[1:-1] == pytest.approx(1.0 - theta_loc * dt_step * diag)
    assert upper[1:-1] == pytest.approx(-theta_loc * dt_step * sup)


def test_path_focused_grid_pins_theta_and_v0():
    core = _core(variance_grid_mode="path_focused")

    assert np.any(np.isclose(core.V_grid, SIGMA_COLLAPSE.theta))
    assert np.any(np.isclose(core.V_grid, SIGMA_COLLAPSE.v0))
    assert np.all(np.diff(core.V_grid) > 0.0)


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("variance_grid_mode", "unknown", "variance_grid_mode"),
        ("v_drift_scheme", "unknown", "v_drift_scheme"),
    ],
)
def test_variance_operator_policy_is_validated(keyword, value, message):
    with pytest.raises(ValidationError, match=message):
        _core(**{keyword: value})
