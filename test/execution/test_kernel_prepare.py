"""Kernel prepare lifecycle, session-owned services, DCN adapter (spec section 7)."""
import dataclasses

import pytest

from quantark.execution import PricingRequest, PricingSession
from quantark.execution.contracts import PreparedState


class _PrepEngine:
    """price(product, env) engine whose adapter prepares state."""

    def price(self, product, env):
        return 1.0


class _PrepAdapter:
    """Minimal specialized adapter with a prepare step."""

    def __init__(self):
        from quantark.execution.legacy_adapter import LegacyPriceAdapter

        self._legacy = LegacyPriceAdapter(call_shape="product_env")
        self.prepared_seen = []

    def capabilities(self):
        return self._legacy.capabilities()

    def validate(self, engine, request):
        return self._legacy.validate(engine, request)

    def normalize(self, engine, request):
        return self._legacy.normalize(engine, request)

    def prepare(self, engine, request, context):
        return PreparedState(
            payload={"k": 1}, descriptors=(), fingerprint="prep-fp",
            byte_estimate=8,
        )

    def execute_native(self, engine, request, normalized, context, prepared=None):
        self.prepared_seen.append(prepared)
        return 1.0, (("pv", 1.0),)


def _session_with(adapter):
    from quantark.execution.context import default_context
    from quantark.execution.registry import AdapterRegistry

    registry = AdapterRegistry()
    registry.register(
        f"{_PrepEngine.__module__}.{_PrepEngine.__qualname__}", lambda: adapter
    )
    ctx = dataclasses.replace(default_context(), adapter_registry=registry)
    return PricingSession(ctx)


def test_kernel_calls_prepare_and_stamps_manifest():
    adapter = _PrepAdapter()
    with _session_with(adapter) as session:
        outcome = session.execute(
            _PrepEngine(), PricingRequest(product="P", pricing_env="E")
        )
    assert adapter.prepared_seen and adapter.prepared_seen[0].fingerprint == "prep-fp"
    assert outcome.manifest.preparation_fingerprint == "prep-fp"


def test_session_owns_cache_and_lease_manager():
    with PricingSession() as session:
        ctx = session.context
        assert ctx.artifact_cache is not None
        assert ctx.lease_manager is not None
        cache = ctx.artifact_cache
    # closed with the session: further use raises
    from quantark.execution.errors import PreparationError

    with pytest.raises(PreparationError):
        cache.get_or_build(None, lambda: 1, size_bytes=1)


def test_legacy_engines_still_work_without_prepare():
    from datetime import datetime

    from quantark.asset.equity.engine.mc import EuropeanMCEngine
    from quantark.asset.equity.param import MCParams
    from quantark.asset.equity.product.option import EuropeanVanillaOption
    from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
    from quantark.priceenv import PricingEnvironment
    from quantark.util.enum import OptionType

    env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.05),
        valuation_date=datetime(2024, 1, 1),
    )
    opt = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    engine = EuropeanMCEngine(params=MCParams(num_paths=64, seed=1))
    direct = engine.price(opt, env)
    with PricingSession() as session:
        outcome = session.execute(engine, PricingRequest(product=opt, pricing_env=env))
    assert outcome.value == direct
    assert outcome.manifest.preparation_fingerprint is None
