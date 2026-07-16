"""Adapter registry resolution and the serial compatibility adapter."""
import pytest

from quantark.execution.contracts import (
    OutputKind,
    PricingOperation,
    PricingRequest,
)
from quantark.execution.errors import CapabilityError
from quantark.execution.legacy_adapter import ADAPTER_ID, LegacyPriceAdapter
from quantark.execution.registry import AdapterRegistry, build_default_registry
from quantark.util.exceptions import ValidationError


class _FakeProductEnvEngine:
    """Stands in for an equity-style engine: price(product, env)."""

    def price(self, product, env):
        return 42.0


class _FakeEnvBoundEngine:
    """Stands in for a convertible-bond-style engine: price(product)."""

    def price(self, product):
        return 7.0


def _register_fakes(registry):
    registry.register(
        f"{_FakeProductEnvEngine.__module__}.{_FakeProductEnvEngine.__qualname__}",
        lambda: LegacyPriceAdapter(call_shape="product_env"),
    )
    registry.register(
        f"{_FakeEnvBoundEngine.__module__}.{_FakeEnvBoundEngine.__qualname__}",
        lambda: LegacyPriceAdapter(call_shape="env_bound"),
    )


def test_resolution_matches_exact_class_then_mro():
    registry = AdapterRegistry()
    _register_fakes(registry)
    registry.freeze()

    adapter = registry.resolve(_FakeProductEnvEngine())
    assert adapter.capabilities().adapter_id == ADAPTER_ID

    class Sub(_FakeProductEnvEngine):
        pass

    assert registry.resolve(Sub()) is not None  # nearest registered base


def test_unregistered_engine_raises_capability_error():
    registry = AdapterRegistry()
    registry.freeze()
    with pytest.raises(CapabilityError):
        registry.resolve(object())


def test_duplicate_registration_and_frozen_registration_fail():
    registry = AdapterRegistry()
    registry.register("a.B", lambda: None)
    with pytest.raises(ValidationError):
        registry.register("a.B", lambda: None)
    registry.freeze()
    with pytest.raises(ValidationError):
        registry.register("c.D", lambda: None)


def test_default_registry_covers_engine_family_roots():
    registry = build_default_registry()
    expected = {
        "quantark.asset.equity.engine.base_engine.BaseEngine",
        "quantark.asset.equity.engine.mc.sabr_mc_engine.SABRMCEngine",
        "quantark.asset.fx.engine.base_fx_engine.BaseFxEngine",
        "quantark.asset.credit.engine.base_credit_engine.BaseCreditEngine",
        "quantark.asset.bond.engine.pde.convertible."
        "jump_diffusion_engine.ConvertibleBondJumpDiffusionEngine",
        "quantark.asset.bond.engine.pde.convertible.tf_engine.ConvertibleBondTFEngine",
        "quantark.asset.bond.engine.convertible."
        "convertible_bond_engine.ConvertibleBondEngine",
        "quantark.asset.equity.engine.mc.dcn_vol_mc_engines.LocalVolDCNMCEngine",
        "quantark.asset.equity.engine.pde.dcn_vol_pde_solvers.LocalVolDCNPDEEngine",
        # Phase 2: batch capability; Phase 3: Heston-family batch adapters
        "quantark.asset.equity.engine.mc.dcn_mc_engine.DCNMCEngine",
        "quantark.asset.equity.engine.mc.dcn_vol_mc_engines.HestonDCNMCEngine",
        "quantark.asset.equity.engine.mc.dcn_vol_mc_engines.QEDCNMCEngine",
        "quantark.asset.equity.engine.mc.dcn_vol_mc_engines."
        "CoupledCoarseHestonDCNMCEngine",
        # Phase 3: adaptive RQMC compatibility (non-exact: subclass-safe)
        "quantark.asset.equity.engine.mc.snowball_mc_engine.SnowballMCEngine",
        "quantark.asset.equity.engine.mc.phoenix_mc_engine.PhoenixMCEngine",
    }
    assert set(registry.registered_paths()) == expected


def test_legacy_adapter_price_dispatch_both_shapes():
    ctx = None  # execute_native does not need the context in Phase 0
    adapter_pe = LegacyPriceAdapter(call_shape="product_env")
    req = PricingRequest(product="P", pricing_env="E")
    norm = adapter_pe.normalize(_FakeProductEnvEngine(), req)
    assert norm.snapshot_complete is False and norm.fingerprint is None
    value, economics = adapter_pe.execute_native(
        _FakeProductEnvEngine(), req, norm, ctx
    )
    assert value == 42.0
    assert dict(economics)["pv"] == 42.0

    adapter_eb = LegacyPriceAdapter(call_shape="env_bound")
    req_eb = PricingRequest(product="bond")
    norm_eb = adapter_eb.normalize(_FakeEnvBoundEngine(), req_eb)
    value_eb, economics_eb = adapter_eb.execute_native(
        _FakeEnvBoundEngine(), req_eb, norm_eb, ctx
    )
    assert value_eb == 7.0


def test_legacy_adapter_rejects_unsupported_operation_and_output():
    adapter = LegacyPriceAdapter(call_shape="product_env")
    engine = _FakeProductEnvEngine()

    req_detailed = PricingRequest(
        product="P", pricing_env="E", operation=PricingOperation.PRICE_DETAILED
    )
    norm = adapter.normalize(engine, req_detailed)
    with pytest.raises(CapabilityError):
        adapter.execute_native(engine, req_detailed, norm, None)

    req_grid = PricingRequest(
        product="P", pricing_env="E",
        outputs=frozenset({OutputKind.PV, OutputKind.GRID}),
    )
    with pytest.raises(CapabilityError):
        adapter.validate(engine, req_grid)
