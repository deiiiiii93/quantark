"""Forward-density event-stats mode (spec 2026-08-24). Battery grows task by task."""
import math

import numpy as np
import pytest
from scipy.stats import norm

from quantark.asset.equity.engine.quad.quad_math import QuadratureMath
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import QuadParams
from quantark.util.exceptions import ValidationError

# Provisional gate-(c) tolerances; tightened + banked by the Task 9 pilot.
MASS_TOL = 1e-6
MOMENT_RTOL = 1e-4
# Measured floor at grid 2001 is 1.36e-5 and is IDENTICAL when the kinked
# call payoff is integrated against the exact analytic density (i.e. it is
# Simpson-on-kink quadrature error, not a marching defect; contracts to
# 2.9e-7 at grid 8001).
FWD_VALUE_RTOL = 3e-5

SPOT, VOL, R, Q, T = 100.0, 0.2, 0.03, 0.05, 1.0
ALPHA = (R - Q - 0.5 * VOL * VOL) / (VOL * VOL)


def test_event_stats_mode_default_is_stacked():
    assert QuadParams().event_stats_mode == "stacked"


def test_event_stats_mode_accepts_forward_density():
    assert QuadParams(event_stats_mode="forward_density").event_stats_mode == "forward_density"


def test_event_stats_mode_rejects_unknown():
    with pytest.raises(ValidationError):
        QuadParams(event_stats_mode="fwd")


def _math_utils(grid_x=2001):
    return QuadratureMath(
        grid_x=grid_x, spot=SPOT, maturity=T, vol_max=VOL,
        num_std_devs=10, align_log=None, integration_rule="simpson",
        fft_padding_factor=2, fft_filter_alpha=0.0, fft_filter_power=2,
    )


def _march_free_density(engine, mu, n_steps=50):
    dt = T / n_steps
    tau_step = 0.5 * VOL * VOL * dt
    p = engine._forward_seed(mu, tau_step, ALPHA)
    omega, pref = engine._forward_kernel(mu, tau_step, ALPHA)
    p_lr, p_ur, p0 = 0, mu.grid.size - 1, (mu.grid.size - 1) % 2
    for _ in range(n_steps - 1):
        p = engine._diffuse_density(p, mu, omega, pref, p_lr, p_ur, p0)
    return p


def test_forward_density_mass_mean_variance():
    engine = SnowballQuadEngine()
    mu = _math_utils()
    p = _march_free_density(engine, mu)
    mass = engine._density_integral(mu, p)
    mean = engine._density_integral(mu, mu.grid * p)
    var = engine._density_integral(mu, (mu.grid - mean) ** 2 * p)
    m = R - Q - 0.5 * VOL * VOL
    assert abs(mass - 1.0) < MASS_TOL
    # Sign of the drift is the kernel-orientation detector.
    assert mean * m > 0.0
    assert abs(mean - m * T) < MOMENT_RTOL * max(abs(m * T), 1e-3)
    assert abs(var - VOL * VOL * T) / (VOL * VOL * T) < MOMENT_RTOL


def test_forward_density_undiscounted_call_value():
    engine = SnowballQuadEngine()
    mu = _math_utils()
    p = _march_free_density(engine, mu)
    strike = 105.0
    payoff = np.maximum(SPOT * np.exp(mu.grid) - strike, 0.0)
    got = engine._density_integral(mu, payoff * p)
    fwd = SPOT * math.exp((R - Q) * T)
    sig = VOL * math.sqrt(T)
    d1 = (math.log(fwd / strike) + 0.5 * sig * sig) / sig
    want = fwd * norm.cdf(d1) - strike * norm.cdf(d1 - sig)
    assert abs(got - want) / want < FWD_VALUE_RTOL
