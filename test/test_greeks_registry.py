"""Registry as the single source of greek validation/aliases/defaults."""

import pytest

from quantark.asset.equity.riskmeasures.greeks.registry import (
    ALIASES,
    DEFAULT_SET,
    REGISTRY,
    normalize_greeks,
)
from quantark.util.exceptions import ValidationError


def test_registry_matches_incumbent_surface():
    assert DEFAULT_SET == {
        "price",
        "delta",
        "gamma",
        "vega",
        "theta",
        "rho",
        "dividend_rho",
        "convexity_theta",
        "r_theta",
        "q_theta",
    }
    assert ALIASES["deltadq"] == "delta_q"
    assert ALIASES["rhoq"] == "dividend_rho"
    assert "vanna" in REGISTRY
    assert REGISTRY["delta"].linear_value == 1.0
    assert REGISTRY["gamma"].linear_value == 0.0


def test_normalize_resolves_aliases_to_canonical_keys():
    requests = normalize_greeks(["deltadq", "div_rho"])
    assert {req.key for req in requests} == {"delta_q", "dividend_rho"}
    assert all(req.clock is None for req in requests)


def test_normalize_none_and_empty():
    assert normalize_greeks(None) is None
    assert normalize_greeks([]) == set()


def test_normalize_rejects_unknown():
    with pytest.raises(ValidationError):
        normalize_greeks(["not_a_greek"])


def test_normalize_rejects_bad_type():
    with pytest.raises(ValidationError):
        normalize_greeks([3.14])
