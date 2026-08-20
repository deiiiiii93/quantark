"""Tests for recording resolved engine configuration in the evidence."""

import dataclasses
from enum import Enum

import pytest

from quantark.modelvalidation.engine_config import engine_config, flatten


class _Mode(str, Enum):
    CELL_AVERAGE = "cell_average"


@dataclasses.dataclass
class _Inner:
    spot_bump: float = 0.01


@dataclasses.dataclass
class _Params:
    grid_points: int = 1001
    num_std_devs: float = 10.0
    mode: _Mode = _Mode.CELL_AVERAGE
    bounds: tuple = (None, None)
    nested: _Inner = dataclasses.field(default_factory=_Inner)
    cache_size: int = 512


def test_serializes_a_params_dataclass():
    config = engine_config(_Params())
    assert config["grid_points"] == 1001
    assert config["num_std_devs"] == 10.0


def test_enums_become_their_value():
    """An enum repr in evidence is unreadable and unstable across versions."""
    assert engine_config(_Params())["mode"] == "cell_average"


def test_nested_dataclasses_expand():
    assert engine_config(_Params())["nested"] == {"spot_bump": 0.01}


def test_tuples_become_lists_for_json():
    assert engine_config(_Params())["bounds"] == [None, None]


def test_exclusions_are_honoured():
    config = engine_config(_Params(), exclude=("cache_size",))
    assert "cache_size" not in config
    assert "grid_points" in config


def test_rejects_a_non_dataclass():
    with pytest.raises(TypeError):
        engine_config({"grid_points": 1001})


def test_config_is_json_safe():
    import json

    json.dumps(engine_config(_Params()))


def test_flatten_produces_dotted_paths():
    flat = flatten({"grid": {"points": 400, "steps_per_day": 4.0}, "engine": "PDE"})
    assert flat == {
        "grid.points": 400,
        "grid.steps_per_day": 4.0,
        "engine": "PDE",
    }


def test_flatten_handles_deep_nesting():
    flat = flatten({"a": {"b": {"c": 1}}})
    assert flat == {"a.b.c": 1}


def test_flatten_leaves_lists_alone():
    flat = flatten({"grid": {"bounds": [None, None]}})
    assert flat == {"grid.bounds": [None, None]}
