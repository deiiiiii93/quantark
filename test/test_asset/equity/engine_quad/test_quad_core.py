"""Unit tests for quadrature core edge cases."""

import math

import numpy as np
import pytest

from quantark.asset.equity.engine.quad.quad_core import QuadratureCore


def _make_core() -> QuadratureCore:
    return QuadratureCore(
        grid_x=101,
        spot=100.0,
        observation_times=[1.0],
        rate=0.05,
        div=0.01,
        vol=0.2,
    )


def test_factor_value_zero_strike_returns_zero_for_negative_epsilon():
    core = _make_core()
    value = core._factor_value_at_m(100.0, 0.0, -1, "a", 1)
    assert float(np.asarray(value)) == pytest.approx(0.0, abs=1e-12)


def test_factor_value_zero_strike_returns_base_for_positive_epsilon():
    core = _make_core()
    dt = float(core.dt[1])
    base = 100.0 * math.exp(-float(core.q[1]) * dt)
    value = core._factor_value_at_m(100.0, 0.0, 1, "a", 1)
    assert float(np.asarray(value)) == pytest.approx(base, rel=1e-12, abs=1e-12)


def test_factor_value_infinite_strike_returns_base_for_negative_epsilon():
    core = _make_core()
    dt = float(core.dt[1])
    base = 100.0 * math.exp(-float(core.q[1]) * dt)
    value = core._factor_value_at_m(100.0, math.inf, -1, "a", 1)
    assert float(np.asarray(value)) == pytest.approx(base, rel=1e-12, abs=1e-12)


def test_factor_value_infinite_strike_returns_zero_for_positive_epsilon():
    core = _make_core()
    value = core._factor_value_at_m(100.0, math.inf, 1, "a", 1)
    assert float(np.asarray(value)) == pytest.approx(0.0, abs=1e-12)
