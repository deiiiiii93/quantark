import numpy as np
import pytest

from quantark.util.exceptions import ValidationError
from quantark.volmodels.localvol.surface import LocalVolSurface

K = np.array([80.0, 100.0, 125.0])
T = np.array([0.5, 1.0])
G = np.array([[0.20, 0.22, 0.25], [0.21, 0.23, 0.26]])


def test_default_is_linear_in_strike_unchanged():
    surf = LocalVolSurface(K, T, G)  # default interp="linear_s"
    # midpoint 90 between 80,100 at t=0.5: linear-in-S weight = (90-80)/(100-80)=0.5
    assert surf.local_vol(90.0, 0.5) == pytest.approx(0.5 * 0.20 + 0.5 * 0.22, abs=1e-12)


def test_linear_logs_uses_log_strike_weight():
    surf = LocalVolSurface(K, T, G, interp="linear_logs")
    wK = (np.log(90.0) - np.log(80.0)) / (np.log(100.0) - np.log(80.0))
    assert surf.local_vol(90.0, 0.5) == pytest.approx((1 - wK) * 0.20 + wK * 0.22, abs=1e-12)


def test_interp_modes_agree_at_grid_nodes():
    lin = LocalVolSurface(K, T, G, interp="linear_s")
    log = LocalVolSurface(K, T, G, interp="linear_logs")
    for k in K:
        assert lin.local_vol(float(k), 0.5) == pytest.approx(log.local_vol(float(k), 0.5), abs=1e-12)


def test_invalid_interp_raises():
    with pytest.raises(ValidationError):
        LocalVolSurface(K, T, G, interp="bogus")
