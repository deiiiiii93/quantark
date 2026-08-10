"""Shared QE variance step, with an optional Numba backend.

The Andersen QE variance update is the hottest elementwise block in the SLV
Monte Carlo path (`simulate` carried ~38% of reference-stack tottime on
2026-08-10) and it was duplicated inline in two places. This module shares it
and lets a Numba kernel take over when the package is installed.

Numba is optional: absent it, the NumPy reference runs and nothing changes.
Whichever backend is live, results must be bit-identical -- the accelerator is a
speed change only.
"""

import numpy as np
import pytest

import quantark.asset  # noqa: F401
from quantark.montecarlo import qe_kernels


def _inputs(n=512, seed=0, theta=0.04):
    rng = np.random.default_rng(seed)
    return (
        np.abs(rng.normal(theta, theta * 0.5, n)),
        rng.normal(0.0, 1.0, n),
        rng.random(n),
    )


REGIMES = {
    "ordinary": dict(kappa=1.5, theta=0.04, sigma2=0.5**2, dt=1.0 / 252.0),
    "sigma_collapse": dict(kappa=3.0, theta=0.00306, sigma2=0.00311**2, dt=1.0 / 252.0),
    "low_feller": dict(kappa=0.6, theta=0.09, sigma2=1.4**2, dt=1.0 / 252.0),
    "zero_kappa": dict(kappa=0.0, theta=0.04, sigma2=0.5**2, dt=1.0 / 252.0),
}


def test_backend_is_reported():
    assert qe_kernels.qe_backend() in ("numpy", "numba")


@pytest.mark.parametrize("regime", sorted(REGIMES))
def test_dispatcher_matches_the_numpy_reference_bitwise(regime):
    var, zv, uv = _inputs(theta=REGIMES[regime]["theta"])
    got = qe_kernels.qe_variance_step(var, zv, uv, **REGIMES[regime])
    ref = qe_kernels.qe_variance_step_numpy(var, zv, uv, **REGIMES[regime])
    for field in got._fields:
        assert np.array_equal(getattr(got, field), getattr(ref, field)), field


def test_outputs_satisfy_the_scheme_invariants():
    var, zv, uv = _inputs()
    step = qe_kernels.qe_variance_step(var, zv, uv, **REGIMES["ordinary"])
    assert np.all(step.v_np >= 0.0)          # variance cannot go negative
    assert np.all(step.v_bar >= 0.0)
    assert np.all(step.prob_zero >= 0.0) and np.all(step.prob_zero <= 0.999999)
    assert step.quad_mask.dtype == np.bool_
    # v_bar is the average of the old and new variance, both floored at zero.
    expected = np.maximum(0.5 * (step.v_np + np.maximum(var, 0.0)), 0.0)
    assert np.array_equal(step.v_bar, expected)


def test_high_psi_paths_take_the_exponential_branch():
    """low_feller drives psi above psi_c, so the Bernoulli branch must engage."""
    var, zv, uv = _inputs(theta=REGIMES["low_feller"]["theta"], seed=5)
    step = qe_kernels.qe_variance_step(var, zv, uv, **REGIMES["low_feller"])
    assert not step.quad_mask.all(), "expected some psi > psi_c paths"


def test_zero_kappa_limit_is_finite():
    var, zv, uv = _inputs()
    step = qe_kernels.qe_variance_step(var, zv, uv, **REGIMES["zero_kappa"])
    assert np.all(np.isfinite(step.v_np))


def test_numba_backend_when_present_is_bitwise():
    if qe_kernels.qe_backend() != "numba":
        pytest.skip("numba not installed on this machine")
    for regime, prm in sorted(REGIMES.items()):
        var, zv, uv = _inputs(n=2048, seed=3, theta=prm["theta"])
        ref = qe_kernels.qe_variance_step_numpy(var, zv, uv, **prm)
        got = qe_kernels._qe_variance_step_numba(var, zv, uv, **prm)
        for field in ref._fields:
            assert np.array_equal(getattr(got, field), getattr(ref, field)), (
                f"{regime}/{field}"
            )
