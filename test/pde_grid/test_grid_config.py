"""Tier-1 tests for GridConfig + accuracy profiles (spec §4.7)."""

import pytest

from quantark.asset.equity.engine.pde.grid import GridConfig, resolve_config
from quantark.util.exceptions import ValidationError


def test_resolve_standard_fully_populated():
    c = resolve_config("standard", None)
    assert c.points == 400 and c.steps_per_day == 4.0 and c.day_count == 252
    assert c.terminal_damping_steps == 1 and c.event_damping_steps == 2
    assert c.eps_crit == 0.003 and c.num_std == 4.0
    assert c.max_points == 2000 and c.max_steps == 5000
    assert c.bounds == (None, None)


def test_profiles_differ_as_specified():
    fast, high = resolve_config("fast", None), resolve_config("high", None)
    assert fast.points == 200 and fast.steps_per_day == 4.0
    assert high.points == 800 and high.steps_per_day == 8.0 and high.eps_crit == 0.002


def test_field_by_field_override():
    c = resolve_config("standard", GridConfig(steps_per_day=8.0))
    assert c.steps_per_day == 8.0 and c.points == 400  # only that field moved


def test_unknown_accuracy_rejected():
    with pytest.raises(ValidationError):
        resolve_config("ultra", None)


def test_points_over_max_rejected():
    with pytest.raises(ValidationError):
        resolve_config("standard", GridConfig(points=4000))  # > max_points 2000


def test_negative_counts_rejected():
    with pytest.raises(ValidationError):
        resolve_config("standard", GridConfig(event_damping_steps=-1))
    with pytest.raises(ValidationError):
        resolve_config("standard", GridConfig(points=0))


def test_per_side_bounds_kept():
    c = resolve_config("standard", GridConfig(bounds=(50.0, None)))
    assert c.bounds == (50.0, None)
    with pytest.raises(ValidationError):
        resolve_config("standard", GridConfig(bounds=(120.0, 80.0)))


def test_key_fingerprint():
    assert resolve_config("fast", None).key == resolve_config("fast", None).key
    assert resolve_config("fast", None).key != resolve_config("high", None).key
    assert (
        resolve_config("standard", GridConfig(points=444)).key
        != resolve_config("standard", None).key
    )
