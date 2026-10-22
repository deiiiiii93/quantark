"""Public equity option engine lifecycle-state signature tests."""

import importlib
import inspect
import pkgutil

import pytest

from quantark.asset.equity.engine.analytical import (
    BlackScholesEngine,
    DigitalOptionAnalyticalEngine,
)
from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.engine.mc import (
    DigitalOptionMCEngine,
    EuropeanMCEngine,
    SnowballMCEngine,
)
from quantark.asset.equity.engine.pde import (
    EuropeanPDESolver,
    SnowballPDESolver,
)
from quantark.asset.equity.engine.quad import (
    EuropeanQuadEngine,
    SnowballQuadEngine,
)
import quantark.asset.equity.engine.analytical as analytical_package
import quantark.asset.equity.engine.mc as mc_package
import quantark.asset.equity.engine.pde as pde_package
import quantark.asset.equity.engine.quad as quad_package


PUBLIC_ENGINE_CLASSES = (
    BaseEngine,
    BlackScholesEngine,
    DigitalOptionAnalyticalEngine,
    EuropeanMCEngine,
    DigitalOptionMCEngine,
    SnowballMCEngine,
    EuropeanPDESolver,
    SnowballPDESolver,
    EuropeanQuadEngine,
    SnowballQuadEngine,
)


@pytest.mark.parametrize("engine_class", PUBLIC_ENGINE_CLASSES)
def test_price_exposes_keyword_only_lifecycle_state(engine_class):
    parameter = inspect.signature(engine_class.price).parameters[
        "lifecycle_state"
    ]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


@pytest.mark.parametrize(
    "method_name",
    ["price_with_events", "calculate_greeks"],
)
def test_base_derived_operations_expose_lifecycle_state(method_name):
    parameter = inspect.signature(
        getattr(BaseEngine, method_name)
    ).parameters["lifecycle_state"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


def test_all_discoverable_concrete_equity_engine_prices_expose_lifecycle_state():
    for package in (
        analytical_package,
        mc_package,
        pde_package,
        quad_package,
    ):
        for module in pkgutil.walk_packages(
            package.__path__,
            package.__name__ + ".",
        ):
            importlib.import_module(module.name)

    checked = []
    for engine_class in _descendants(BaseEngine):
        if not engine_class.__module__.startswith(
            "quantark.asset.equity.engine."
        ):
            continue
        signature = inspect.signature(engine_class.price)
        if "product" not in signature.parameters:
            continue
        checked.append(engine_class)
        parameter = signature.parameters["lifecycle_state"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is None

    assert len(checked) >= 50


def _descendants(base_class):
    seen = set()
    remaining = list(base_class.__subclasses__())
    while remaining:
        candidate = remaining.pop()
        if candidate in seen:
            continue
        seen.add(candidate)
        remaining.extend(candidate.__subclasses__())
        yield candidate
