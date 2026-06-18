import pytest
from quantark.util.enum.engine_enums import ADIScheme
from quantark.util.exceptions import ValidationError
from quantark.volmodels.slv.fokkerplanck.config import FpCalibrationConfig


def test_defaults_are_valid():
    c = FpCalibrationConfig()
    assert c.n_x >= 3 and c.n_z >= 3
    assert 3 <= c.n_strike_nodes <= c.n_x
    assert 0.0 < c.cir_quantile < 0.5
    assert c.scheme is ADIScheme.CRAIG_SNEYD
    assert c.leverage_clip[0] < c.leverage_clip[1]


@pytest.mark.parametrize("kwargs", [
    {"n_x": 2}, {"n_z": 2}, {"n_strike_nodes": 999},        # n_strike_nodes > n_x
    {"cir_quantile": 0.0}, {"cir_quantile": 0.5},
    {"x_span_stds": 0.0}, {"v_floor": 0.0}, {"rannacher_steps": -1},
    {"leverage_clip": (2.0, 1.0)}, {"scheme": ADIScheme.DOUGLAS},
])
def test_invalid_configs_raise(kwargs):
    with pytest.raises(ValidationError):
        FpCalibrationConfig(**kwargs)
