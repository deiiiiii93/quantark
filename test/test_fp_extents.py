import numpy as np
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.slv.fokkerplanck.coordinates import z_extents, x_extents


def test_z_extents_envelope_v0_and_theta_strong_mean_reversion():
    # strong mean reversion: terminal law near theta, but v0 must still be inside
    p = HestonParams(v0=0.25, kappa=8.0, theta=0.04, sigma=0.3, rho=-0.5)
    t_nodes = np.linspace(0.0, 1.0, 13)          # includes t=0
    v_min, v_max = z_extents(p, eta=1.0, t_nodes=t_nodes, cir_quantile=1e-4, v_floor=1e-8)
    assert v_min <= min(p.v0, p.theta)
    assert v_max >= max(p.v0, p.theta)
    assert v_min > 0.0


def test_z_extents_skip_t0_no_nan():
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)
    t_nodes = np.array([0.0, 0.25, 0.5, 1.0])
    v_min, v_max = z_extents(p, eta=1.0, t_nodes=t_nodes, cir_quantile=1e-4, v_floor=1e-8)
    assert np.isfinite(v_min) and np.isfinite(v_max) and v_max > v_min


def test_x_extents_widen_with_local_vol_level_and_carry():
    # higher sup local vol -> wider x span; positive carry shifts the drift envelope
    lo1, hi1 = x_extents(s0=100.0, b_fwd=np.zeros(10), step_dt=np.full(10, 0.1),
                         vbar2=0.04, x_span_stds=6.0)
    lo2, hi2 = x_extents(s0=100.0, b_fwd=np.zeros(10), step_dt=np.full(10, 0.1),
                         vbar2=0.16, x_span_stds=6.0)
    assert (hi2 - lo2) > (hi1 - lo1)
    assert lo1 < np.log(100.0) < hi1
