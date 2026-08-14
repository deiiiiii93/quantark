"""Tests for the per-family builder registry."""

import pytest

from quantark.util.exceptions import ValidationError
from quantark.modelvalidation import registry
from quantark.modelvalidation.registry import (
    BUILDER_KINDS,
    get_builder,
    list_builders,
    register_builder,
)


@pytest.fixture(autouse=True)
def _isolated_registry():
    """Snapshot and restore the module registry around every test."""
    saved = dict(registry._REGISTRY)
    try:
        yield
    finally:
        registry._REGISTRY.clear()
        registry._REGISTRY.update(saved)


def test_register_and_get():
    @register_builder("flat_bsm_x", kind="environment")
    def make_env(params):
        return {"env": params}

    assert get_builder("flat_bsm_x", kind="environment") is make_env
    assert get_builder("flat_bsm_x", kind="environment")({"spot": 1.0}) == {
        "env": {"spot": 1.0}
    }


def test_register_returns_the_function_unchanged():
    """The decorator must not wrap -- builders stay directly callable/testable."""

    def make_env(params):
        return params

    decorated = register_builder("passthrough_x", kind="environment")(make_env)
    assert decorated is make_env


def test_unknown_builder_lists_available():
    @register_builder("flat_bsm_x", kind="environment")
    def make_env(params):
        return {"env": params}

    with pytest.raises(ValidationError) as exc:
        get_builder("nope", kind="environment")
    message = str(exc.value)
    assert "flat_bsm_x" in message
    assert "environment" in message
    assert "nope" in message


def test_duplicate_registration_rejected():
    @register_builder("dupe_x", kind="candidate")
    def first(params):
        return params

    with pytest.raises(ValidationError):

        @register_builder("dupe_x", kind="candidate")
        def second(params):
            return params


def test_same_name_different_kind_is_allowed():
    """Kinds are separate namespaces: a product and a candidate may share a name."""

    @register_builder("equity.snowball_x", kind="product")
    def product(params):
        return "product"

    @register_builder("equity.snowball_x", kind="candidate")
    def candidate(params):
        return "candidate"

    assert get_builder("equity.snowball_x", kind="product")({}) == "product"
    assert get_builder("equity.snowball_x", kind="candidate")({}) == "candidate"


def test_bad_kind_rejected_at_registration():
    with pytest.raises(ValidationError):

        @register_builder("whatever_x", kind="not_a_kind")
        def builder(params):
            return params


def test_bad_kind_rejected_at_lookup():
    with pytest.raises(ValidationError):
        get_builder("whatever_x", kind="not_a_kind")


def test_empty_name_rejected():
    with pytest.raises(ValidationError):

        @register_builder("", kind="product")
        def builder(params):
            return params


def test_list_builders_groups_by_kind_sorted():
    @register_builder("zeta_x", kind="product")
    def zeta(params):
        return params

    @register_builder("alpha_x", kind="product")
    def alpha(params):
        return params

    listed = list_builders()
    assert set(listed) == set(BUILDER_KINDS)
    products = listed["product"]
    assert products.index("alpha_x") < products.index("zeta_x")


def test_unknown_builder_with_empty_kind_still_reports_kind():
    with pytest.raises(ValidationError) as exc:
        get_builder("nothing", kind="reference")
    assert "reference" in str(exc.value)
