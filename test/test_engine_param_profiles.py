import json

import pytest

from asset.equity.param import PDEParams, QuadParams, make_pde_params, make_quad_params
from util.exceptions import ValidationError


def test_make_quad_params_profile_fast():
    params = make_quad_params(profile="fast")
    assert params.grid_points == 401
    assert params.num_std_devs == 8.0


def test_make_pde_params_profile_accurate():
    params = make_pde_params(profile="accurate")
    assert params.grid_size == 800
    assert params.time_steps == 400


def test_override_wins():
    params = make_quad_params(profile="accurate", num_std_devs=10.0)
    assert params.num_std_devs == 10.0


def test_yaml_config_loader(tmp_path):
    try:
        import yaml
    except ImportError:
        pytest.skip("PyYAML not installed")

    data = {"engine": "quad", "profile": "fast", "overrides": {"grid_points": 501}}
    path = tmp_path / "params.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, default_flow_style=False, sort_keys=False)

    params = QuadParams.from_config(path)
    assert params.grid_points == 501


def test_json_config_loader(tmp_path):
    data = {
        "engine": "pde",
        "profile": "balanced",
        "overrides": {"grid_size": 300, "time_steps": 150},
    }
    path = tmp_path / "params.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle)

    params = PDEParams.from_config(path)
    assert params.grid_size == 300
    assert params.time_steps == 150


def test_unknown_profile_raises():
    with pytest.raises(ValidationError):
        make_pde_params(profile="ultra_fast")


def test_unknown_key_raises():
    with pytest.raises(ValidationError):
        make_quad_params(profile="balanced", unknown_param=1)

