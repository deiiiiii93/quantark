from __future__ import annotations

import numpy as np
import pytest

from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.backtest.replay import AutocallableEngineConfig, HedgeSpec, ReplayProduct
from quantark.backtest.simulation.config import CacheConfig, EnsembleConfig, GateConfig, PricingProviderConfig
from quantark.util.enum import OptionType
from quantark.util.exceptions import ValidationError

from quantark.util.enum.engine_enums import EngineType

from .conftest import (SPOT, ensemble_config, ladder_pricing, pde_engine_config, short_snowball,
                       surface_pricing)


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


def test_a_multi_leg_bucket_hedge_is_rejected_before_the_scalar_sizing_path():
    """The simulated path sizes ONE contract per day.

    A bucket strategy answers ``target_legs``, not a scalar contract count,
    and its risk coordinates come from an actual futures chain the simulated
    path does not carry.  Refusing it here beats reaching the sizing call.
    """
    from quantark.backtest.strategy import FuturesBucketHedgeStrategy

    with pytest.raises(ValidationError, match="multi-leg bucket hedge"):
        ensemble_config(strategy=FuturesBucketHedgeStrategy())
    with pytest.raises(ValidationError, match="multi-leg bucket hedge"):
        ensemble_config(strategy=FuturesBucketHedgeStrategy(objective="nodes"))


def test_the_proportional_single_contract_control_is_still_accepted():
    from quantark.backtest.strategy import ProportionalFuturesDeltaHedgeStrategy

    config = ensemble_config(strategy=ProportionalFuturesDeltaHedgeStrategy())
    assert isinstance(config.strategy, ProportionalFuturesDeltaHedgeStrategy)


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


def _gate():
    return GateConfig(sample_states=0, pv_tolerance_bp=0.0, delta_tolerance_hands=0.0)


def test_exact_mode_is_the_default_and_rejects_bucket_steps():
    exact = PricingProviderConfig(provider="repricing", cache=CacheConfig(memory_bytes=1024), gate=_gate())
    assert exact.mode == "exact"
    with pytest.raises(ValidationError):
        PricingProviderConfig(provider="repricing", cache=CacheConfig(memory_bytes=1024), gate=_gate(), vol_step=0.01)


def test_ladder_mode_requires_both_bucket_steps_and_a_positive_spot_step():
    assert ladder_pricing().mode == "ladder"
    with pytest.raises(ValidationError):
        PricingProviderConfig(provider="repricing", cache=CacheConfig(memory_bytes=1024), gate=_gate(), spot_step=0.002)
    with pytest.raises(ValidationError):
        ladder_pricing(spot_step=0.0)
    with pytest.raises(ValidationError):
        ladder_pricing(vol_step=-0.01)
    with pytest.raises(ValidationError):
        PricingProviderConfig(provider="repricing", cache=CacheConfig(memory_bytes=1024), gate=_gate(),
                              spot_step=0.002, vol_step=0.0, q_step=0.0, surface_cache_bytes=10)


def test_life_surface_requires_its_budget_and_a_pde_engine():
    assert surface_pricing().mode == "life_surface"
    with pytest.raises(ValidationError):
        PricingProviderConfig(provider="life_surface", cache=CacheConfig(memory_bytes=1024), gate=_gate(),
                              vol_step=0.01, q_step=0.0025)
    with pytest.raises(ValidationError):
        PricingProviderConfig(provider="life_surface", cache=CacheConfig(memory_bytes=1024), gate=_gate(),
                              vol_step=0.01, q_step=0.0025, surface_cache_bytes=1024, spot_step=0.002)
    with pytest.raises(ValidationError):
        ensemble_config(pricing=surface_pricing(),
                        engine_config=pde_engine_config(pricing_engine_type=EngineType.QUADRATURE))


def test_batching_fields_are_validated():
    cfg = ensemble_config(workers=2, batch_paths=8)
    assert (cfg.workers, cfg.batch_paths) == (2, 8)
    with pytest.raises(ValidationError):
        ensemble_config(workers=0)
    with pytest.raises(ValidationError):
        ensemble_config(batch_paths=0)


def test_a_disk_dir_is_optional_and_kept(tmp_path):
    cache = CacheConfig(memory_bytes=1024, disk_dir=str(tmp_path))
    assert cache.disk_dir == str(tmp_path)
    assert CacheConfig(memory_bytes=1024).disk_dir is None
