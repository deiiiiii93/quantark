from __future__ import annotations

import numpy as np
import pytest

from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.backtest.replay import AutocallableEngineConfig, HedgeSpec, ReplayProduct
from quantark.backtest.simulation.config import CacheConfig, EnsembleConfig, GateConfig, PricingProviderConfig
from quantark.util.enum import OptionType
from quantark.util.exceptions import ValidationError

from .conftest import SPOT, ensemble_config, short_snowball


def test_a_valid_config_exposes_its_quantities():
    cfg = ensemble_config(products=[
        ReplayProduct(product=short_snowball(), quantity=-1.0, position_id=1, has_lifecycle=True),
        ReplayProduct(product=short_snowball(), quantity=2.0, position_id=2, has_lifecycle=True),
    ])
    assert cfg.quantities == pytest.approx(np.array([-1.0, 2.0]))
    assert cfg.pricing.provider == "repricing"


def test_gate_and_cache_tolerances_are_required_and_validated():
    with pytest.raises(TypeError):
        GateConfig(sample_states=5)                       # tolerances have no default
    with pytest.raises(ValidationError):
        GateConfig(sample_states=-1, pv_tolerance_bp=1.0, delta_tolerance_hands=0.1)
    with pytest.raises(ValidationError):
        GateConfig(sample_states=5, pv_tolerance_bp=-1.0, delta_tolerance_hands=0.1)
    with pytest.raises(ValidationError):
        CacheConfig(memory_bytes=0)


def test_only_snowballs_and_only_a_futures_hedge_are_accepted():
    vanilla = EuropeanVanillaOption(strike=SPOT, option_type=OptionType.CALL, maturity=0.5)
    with pytest.raises(ValidationError):
        ensemble_config(products=[ReplayProduct(product=vanilla, quantity=1.0, position_id=1, has_lifecycle=True)])
    with pytest.raises(ValidationError):
        ensemble_config(hedge=HedgeSpec(kind="spot"))
    with pytest.raises(ValidationError):
        ensemble_config(products=[])


def test_unsupported_dividend_sources_and_surface_vol_fail_closed():
    with pytest.raises(ValidationError):
        ensemble_config(engine_config=AutocallableEngineConfig(dividend_source="surface_forwards"))
    with pytest.raises(ValidationError):
        ensemble_config(engine_config=AutocallableEngineConfig(vol_source="surface"))


def test_provider_literal_admits_only_what_is_implemented():
    with pytest.raises(ValidationError):
        PricingProviderConfig(provider="life_surface",
                              cache=CacheConfig(memory_bytes=1024),
                              gate=GateConfig(sample_states=0, pv_tolerance_bp=0.0, delta_tolerance_hands=0.0))
