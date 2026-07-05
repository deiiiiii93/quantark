import pytest
from quantark.util.exceptions import ValidationError
from quantark.volmodels.slv.fokkerplanck.config import FpCalibrationConfig


def test_defaults_are_valid():
    c = FpCalibrationConfig()
    assert c.n_x >= 3 and c.n_z >= 3
    assert 3 <= c.n_strike_nodes <= c.n_x
    assert 0.0 < c.cir_quantile < 0.5
    assert c.leverage_clip[0] < c.leverage_clip[1]


@pytest.mark.parametrize("kwargs", [
    {"n_x": 2}, {"n_z": 2}, {"n_strike_nodes": 999},        # n_strike_nodes > n_x
    {"cir_quantile": 0.0}, {"cir_quantile": 0.5},
    {"x_span_stds": 0.0}, {"v_floor": 0.0},
    {"leverage_clip": (2.0, 1.0)},
])
def test_invalid_configs_raise(kwargs):
    with pytest.raises(ValidationError):
        FpCalibrationConfig(**kwargs)


def test_removed_fields_raise_typeerror():
    # rannacher_steps / scheme were validated-but-never-consumed dead config; removed
    # 2026-07 (volmodels spec WS-A3). Constructor-signature errors are TypeError by design.
    with pytest.raises(TypeError):
        FpCalibrationConfig(rannacher_steps=2)
    with pytest.raises(TypeError):
        FpCalibrationConfig(scheme=None)


def test_new_switch_defaults_preserve_current_behavior():
    cfg = FpCalibrationConfig()
    assert cfg.flux_scheme == "central"
    assert cfg.linear_solver == "direct"
    assert cfg.time_scheme == "backward_euler"
    assert cfg.refactor_every == 5


@pytest.mark.parametrize("field,bad", [
    ("flux_scheme", "upwind"),
    ("linear_solver", "gmres"),
    ("time_scheme", "crank_nicolson"),
])
def test_invalid_scheme_strings_raise(field, bad):
    with pytest.raises(ValidationError):
        FpCalibrationConfig(**{field: bad})


@pytest.mark.parametrize("bad", [0, -1, 2.5])
def test_refactor_every_must_be_positive_integer(bad):
    with pytest.raises(ValidationError):
        FpCalibrationConfig(refactor_every=bad)
