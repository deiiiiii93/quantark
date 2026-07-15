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


class TestDCNLocalVolAdapter:
    @pytest.fixture()
    def dcn_case(self):
        import pathlib
        import sys

        import numpy as np

        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
        from dcn_fixtures import DCN_A, FLAT, flat_env, make_dcn
        from quantark.param import GridVolSurface

        def grid_env():
            env = flat_env(**FLAT)
            env.vol_surface = GridVolSurface(
                strikes=[3000.0, 4500.0, 6000.0, 7500.0, 9000.0],
                maturities=[0.25, 0.5, 1.0, 1.5, 2.0, 2.5],
                iv_grid=np.full((6, 5), FLAT["sigma"]),
            )
            return env

        return make_dcn(DCN_A), grid_env

    def test_session_parity_and_single_build(self, dcn_case, monkeypatch):
        import quantark.asset.equity.engine.mc.dcn_execution_adapters as mod
        from quantark.asset.equity.engine.mc import LocalVolDCNMCEngine

        product, grid_env = dcn_case
        engine = LocalVolDCNMCEngine(num_paths=2**9, seed=42)
        direct = engine.price(product, grid_env())

        builds = []
        real_build = mod.build_dupire_local_vol

        def counting_build(*args, **kwargs):
            builds.append(1)
            return real_build(*args, **kwargs)

        monkeypatch.setattr(mod, "build_dupire_local_vol", counting_build)
        with PricingSession() as session:
            v1 = session.price(LocalVolDCNMCEngine(num_paths=2**9, seed=42),
                               product, grid_env())
            v2 = session.price(LocalVolDCNMCEngine(num_paths=2**9, seed=42),
                               product, grid_env())  # equal-VALUED new env
        assert v1 == direct and v2 == direct   # bit-identical
        assert len(builds) == 1                # second call hit the cache

    def test_changed_vol_rebuilds(self, dcn_case, monkeypatch):
        import numpy as np

        import quantark.asset.equity.engine.mc.dcn_execution_adapters as mod
        from quantark.asset.equity.engine.mc import LocalVolDCNMCEngine
        from quantark.param import GridVolSurface

        product, grid_env = dcn_case
        builds = []
        real_build = mod.build_dupire_local_vol
        monkeypatch.setattr(
            mod, "build_dupire_local_vol",
            lambda *a, **k: builds.append(1) or real_build(*a, **k),
        )
        env2 = grid_env()
        env2.vol_surface = GridVolSurface(
            strikes=env2.vol_surface.strikes,
            maturities=env2.vol_surface.maturities,
            iv_grid=np.full((6, 5), 0.30),
        )
        with PricingSession() as session:
            session.price(LocalVolDCNMCEngine(num_paths=2**8), product, grid_env())
            session.price(LocalVolDCNMCEngine(num_paths=2**8), product, env2)
        assert len(builds) == 2

    def test_disabled_cache_still_exact(self, dcn_case):
        from quantark.asset.equity.engine.mc import LocalVolDCNMCEngine
        from quantark.execution import ResourceBudget
        from quantark.execution.context import default_context

        product, grid_env = dcn_case
        engine = LocalVolDCNMCEngine(num_paths=2**8, seed=3)
        direct = engine.price(product, grid_env())
        ctx = dataclasses.replace(
            default_context(),
            resource_budget=ResourceBudget(artifact_cache_bytes=0),
        )
        with PricingSession(ctx) as session:
            assert session.price(engine, product, grid_env()) == direct

    def test_pde_adapter_parity(self, dcn_case):
        from quantark.asset.equity.engine.pde import LocalVolDCNPDEEngine

        product, grid_env = dcn_case
        engine = LocalVolDCNPDEEngine(num_space_nodes=301)
        direct = engine.price(product, grid_env())
        with PricingSession() as session:
            assert session.price(
                LocalVolDCNPDEEngine(num_space_nodes=301), product, grid_env()
            ) == direct

    def test_mutation_during_prepare_raises_determinism_violation(
        self, dcn_case, monkeypatch
    ):
        """Env mutated mid-build -> loud failure, never a cached surface
        that mismatches its key."""
        import quantark.asset.equity.engine.mc.dcn_execution_adapters as mod
        from quantark.asset.equity.engine.mc import LocalVolDCNMCEngine
        from quantark.execution.errors import DeterminismViolation

        product, grid_env = dcn_case
        env = grid_env()
        real_build = mod.build_dupire_local_vol

        def mutating_build(*args, **kwargs):
            surface = real_build(*args, **kwargs)
            env.vol_surface.iv_grid = env.vol_surface.iv_grid + 0.01
            return surface

        monkeypatch.setattr(mod, "build_dupire_local_vol", mutating_build)
        with PricingSession() as session:
            with pytest.raises(DeterminismViolation):
                session.price(
                    LocalVolDCNMCEngine(num_paths=2**8), product, env
                )

    def test_partial_service_injection_rejected(self):
        """Cache/lease-manager come as a validated pair."""
        from quantark.execution.cache.artifacts import PreparedArtifactCache
        from quantark.execution.context import default_context
        from quantark.execution.leases import ResourceLeaseManager
        from quantark.execution.policy import ResourceBudget
        from quantark.util.exceptions import ValidationError

        leases = ResourceLeaseManager(ResourceBudget(artifact_cache_bytes=100))
        cache = PreparedArtifactCache(leases)
        base = default_context()
        with pytest.raises(ValidationError):
            PricingSession(dataclasses.replace(base, artifact_cache=cache))
        with pytest.raises(ValidationError):
            PricingSession(dataclasses.replace(base, lease_manager=leases))
        other = ResourceLeaseManager(ResourceBudget(artifact_cache_bytes=100))
        with pytest.raises(ValidationError):
            PricingSession(dataclasses.replace(
                base, artifact_cache=cache, lease_manager=other,
            ))
        with PricingSession(dataclasses.replace(
            base, artifact_cache=cache, lease_manager=leases,
        )) as session:  # matched pair accepted; borrowed, not closed
            assert session.context.artifact_cache is cache
        cache.get_or_build(  # still open after session close: borrowed
            __import__("quantark.execution.cache.artifacts",
                       fromlist=["ArtifactDescriptor"]).ArtifactDescriptor(
                kind="k", fingerprint="f",
                dependency_tags=frozenset(), builder_version="1",
            ),
            lambda: 1, size_bytes=1,
        ).close()

    def test_prebuilt_surface_bypasses_cache(self, dcn_case):
        from quantark.asset.equity.engine.mc import LocalVolDCNMCEngine
        from quantark.volmodels.localvol import build_dupire_local_vol

        product, grid_env = dcn_case
        env = grid_env()
        surface = build_dupire_local_vol(
            env.vol_surface, spot=env.spot,
            rate_curve=env.rate_curve, div_yield=env.get_div_yield,
        )
        engine = LocalVolDCNMCEngine(
            local_vol_surface=surface, num_paths=2**8, seed=5
        )
        direct = engine.price(product, env)
        with PricingSession() as session:
            outcome = session.execute(
                engine, PricingRequest(product=product, pricing_env=env)
            )
        assert outcome.value == direct
        assert outcome.manifest.preparation_fingerprint is None
