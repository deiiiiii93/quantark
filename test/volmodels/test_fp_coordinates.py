import numpy as np
from quantark.volmodels.slv.fokkerplanck.coordinates import concentrated_grid, trapezoid_weights


def test_concentrated_grid_endpoints_and_monotone():
    g = concentrated_grid(lo=-2.0, hi=3.0, center=0.5, n=51, concentration=0.1)
    assert g.size == 51
    assert np.isclose(g[0], -2.0) and np.isclose(g[-1], 3.0)
    assert np.all(np.diff(g) > 0)


def test_concentration_clusters_nodes_near_center():
    g = concentrated_grid(lo=-2.0, hi=2.0, center=0.0, n=101, concentration=0.05)
    spacings = np.diff(g)
    i_center = int(np.argmin(np.abs(g)))
    # spacing near the center node is smaller than spacing at the boundary
    assert spacings[i_center] < spacings[0]


def test_trapezoid_weights_integrate_constant_to_length():
    g = concentrated_grid(lo=0.0, hi=4.0, center=1.0, n=80, concentration=0.2)
    w = trapezoid_weights(g)
    assert np.isclose(np.sum(w), 4.0)              # integral of 1 over [0,4]
    assert np.isclose(np.sum(w * g), 0.5 * 16.0, rtol=1e-3)   # integral of x = 8
