import importlib.util
import sys
from pathlib import Path

import pytest

from quantark.util.enum.engine_enums import EngineType

REPO = Path(__file__).resolve().parents[2]


def _load(stem):
    """Import a numbered stage script (the stages are not a package)."""
    path = REPO / "example" / "mo_volmodels" / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem.split("_")[0], path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod        # @dataclass resolves cls.__module__ here
    spec.loader.exec_module(mod)
    return mod


def _load_gate():
    return _load("11_pde_convergence_gate")


def _load_stage12():
    return _load("12_snowball_volmodel_backtest")


def test_flat_bsm_quad_differs_from_flat_bsm_in_engine_only():
    """The engine control is only a control if the market data is identical."""
    s12 = _load_stage12()
    assert "flat_bsm_quad" in s12.VARIANTS
    bsm = s12.VARIANT_SPECS["flat_bsm"]
    quad = s12.VARIANT_SPECS["flat_bsm_quad"]
    assert quad.vol_source == bsm.vol_source
    assert quad.surface_vol_mode == bsm.surface_vol_mode
    assert quad.vol_model == bsm.vol_model == "bsm"
    assert bsm.pricing_engine_type == EngineType.PDE
    assert quad.pricing_engine_type == EngineType.QUADRATURE


def test_engine_config_honours_the_variant_pricing_engine_type():
    """A new VariantSpec field is inert until make_engine_config reads it."""
    s12 = _load_stage12()
    routing = s12.GateRouting("p", None, {}, {})
    cfg = s12.make_engine_config("flat_bsm_quad", routing=routing)
    assert cfg.pricing_engine_type == EngineType.QUADRATURE
    assert s12.make_engine_config(
        "flat_bsm", routing=routing
    ).pricing_engine_type == EngineType.PDE
