"""Tests for BackwardOperator (shared factorization + theta_by_step) [§11.2]."""

import numpy as np
import pytest

from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.engine.pde.backward_operator import BackwardOperator
from quantark.asset.equity.param import PDEParams


@pytest.fixture
def lcu():
    # Representative interior tridiagonal coefficients (constant-vol BSM).
    rng = np.linspace(0.8, 1.2, 40)
    l = -0.5 * rng
    c = -2.0 * rng - 0.03
    u = 0.5 * rng
    return l, c, u


def test_banded_system_matches_legacy(lcu):
    l, c, u = lcu
    op = BackwardOperator(l, c, u, use_banded=True, cache_enabled=True)
    solver = SnowballPDESolver(PDEParams(cache_enabled=False))
    for dt, theta in [(0.01, 0.5), (0.004, 1.0), (0.02, 0.5)]:
        got = op.banded_system(dt, theta)
        ref = solver._get_banded_system(l, c, u, dt, theta)
        for a, b in zip(got, ref):
            assert np.allclose(a, b, rtol=0, atol=0)


def test_banded_system_cache_returns_equal(lcu):
    l, c, u = lcu
    op = BackwardOperator(l, c, u, cache_enabled=True)
    first = op.banded_system(0.01, 0.5)
    second = op.banded_system(0.01, 0.5)
    for a, b in zip(first, second):
        assert np.array_equal(a, b)


def _theta(t_vec, params, discontinuity_times):
    return BackwardOperator.theta_by_step(
        np.asarray(t_vec), np.diff(t_vec), params, discontinuity_times
    )


def test_theta_default_is_cn_away_from_events():
    t_vec = np.linspace(0.0, 1.0, 11)  # 10 uniform steps
    params = PDEParams()  # theta=0.5, rannacher_steps=1, event_theta=1.0, ers=1
    theta = _theta(t_vec, params, discontinuity_times=None)
    assert len(theta) == len(t_vec) - 1
    # rannacher_steps=1 => exactly one implicit-Euler step at the terminal
    # payoff discontinuity; CN everywhere else.
    assert theta[-1] == 1.0
    assert np.allclose(theta[:-1], 0.5)


def test_theta_event_step_uses_event_theta():
    t_vec = np.linspace(0.0, 1.0, 11)
    params = PDEParams()  # event_theta=1.0
    event = t_vec[6]  # aligned node
    theta = _theta(t_vec, params, discontinuity_times=[event])
    # step j = idx-1 = 5 (the step immediately before the event node) is damped;
    # the terminal step is the default rannacher_steps=1 implicit-Euler step.
    assert theta[5] == params.event_theta == 1.0
    assert theta[-1] == 1.0
    others = np.delete(theta, [5, len(theta) - 1])
    assert np.allclose(others, 0.5)


def test_theta_terminal_rannacher_when_steps_gt_1():
    t_vec = np.linspace(0.0, 1.0, 11)
    params = PDEParams(rannacher_steps=2)
    theta = _theta(t_vec, params, discontinuity_times=None)
    # rannacher_steps=2 => the last TWO backward steps are backward-Euler.
    assert theta[-1] == 1.0
    assert theta[-2] == 1.0
    assert np.allclose(theta[:-2], 0.5)


def test_theta_is_stream_independent():
    # theta_by_step depends ONLY on discontinuity_times, not on any stream
    # selection: passing the same disc times yields the same schedule [§11.2].
    t_vec = np.linspace(0.0, 1.0, 13)
    params = PDEParams()
    disc = [t_vec[4], t_vec[8]]
    a = _theta(t_vec, params, disc)
    b = _theta(t_vec, params, list(disc))
    assert np.array_equal(a, b)


def test_theta_no_rannacher_is_flat():
    t_vec = np.linspace(0.0, 1.0, 11)
    params = PDEParams(use_rannacher=False)
    theta = _theta(t_vec, params, discontinuity_times=[t_vec[5]])
    assert np.allclose(theta, params.theta)
