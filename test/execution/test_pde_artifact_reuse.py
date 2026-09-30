"""Phase 4 artifact-state and reuse gates (spec sections 9.2/10.1/21)."""
import pytest

from quantark.asset.equity.engine.pde.core.groups import Backend
from quantark.asset.equity.engine.pde.core.prepare import SessionArtefacts
from quantark.asset.equity.engine.pde.pde_session_prep import (
    factorization_state,
    grid_state,
    step_coefficients_state,
)
from quantark.execution import PricingRequest, PricingSession
from quantark.execution.errors import DeterminismViolation

from execution.matrix_fixtures import FIXTURE_BUILDERS


def _case(name):
    engine, product, env, _shape = FIXTURE_BUILDERS[name]()
    return engine, product, env


class TestArtifactStates:
    def test_descriptor_determinism_and_bump_sensitivity(self):
        from quantark.param import FlatRateCurve

        engine, product, env = _case("SnowballPDESolver")
        with PricingSession() as session:
            ctx = session.context
            first = grid_state(engine, product, env, ctx)
            second = grid_state(engine, product, env, ctx)
            assert first.fingerprint == second.fingerprint
            assert first.descriptor == second.descriptor
            assert ctx.artifact_cache.stats()["hits"] >= 1
            env.rate_curve = FlatRateCurve(rate=0.06)
            bumped = grid_state(engine, product, env, ctx)
            assert bumped.fingerprint != first.fingerprint
            for state in (first, second, bumped):
                state.handle.close()

    def test_disable_strategy_builds_fresh_without_descriptor(self):
        engine, product, env = _case("SnowballPDESolver")
        engine._cache_enabled = False
        with PricingSession() as session:
            state = grid_state(engine, product, env, session.context)
            assert state.descriptor is None and state.handle is None
            assert state.value is not None

    def test_chain_fingerprints_compose(self):
        engine, product, env = _case("SnowballPDESolver")
        with PricingSession() as session:
            ctx = session.context
            grid = grid_state(engine, product, env, ctx)
            coeff = step_coefficients_state(engine, product, env, grid.value, ctx)
            fact = factorization_state(
                engine, product, env, grid.value, coeff.value, coeff.fingerprint, ctx
            )
            assert grid.fingerprint and coeff.fingerprint and fact.fingerprint
            packs = fact.value
            assert len(packs[Backend.SPARSE_LU]) + len(packs[Backend.BANDED]) > 0
            with pytest.raises(TypeError):
                packs[Backend.BANDED]["x"] = 1  # published packs are immutable
            with pytest.raises(TypeError):
                packs["x"] = 1
            for state in (grid, coeff, fact):
                if state.handle is not None:
                    state.handle.close()

    def test_mutation_during_build_raises_determinism_violation(self):
        from quantark.param import FlatRateCurve

        engine, product, env = _case("EuropeanPDESolver")
        with PricingSession() as session:
            ctx = session.context

            class SwappingCache:
                def __init__(self, inner):
                    self._inner = inner

                def get_or_build(self, descriptor, builder, size_bytes, measure=None):
                    handle = self._inner.get_or_build(
                        descriptor, builder, size_bytes, measure
                    )
                    env.rate_curve = FlatRateCurve(rate=0.061)  # concurrent swap
                    return handle

                def invalidate_tags(self, tags):
                    self._inner.invalidate_tags(tags)

            class Ctx:
                artifact_cache = SwappingCache(ctx.artifact_cache)

            with pytest.raises(DeterminismViolation):
                grid_state(engine, product, env, Ctx())

    def test_injected_state_values_reprice_bitwise(self):
        engine, product, env = _case("SnowballPDESolver")
        baseline = engine.price(product, env)
        clone = type(engine)(params=engine.params)
        with PricingSession() as session:
            ctx = session.context
            grid = grid_state(clone, product, env, ctx)
            coeff = step_coefficients_state(clone, product, env, grid.value, ctx)
            fact = factorization_state(
                clone, product, env, grid.value, coeff.value, coeff.fingerprint, ctx
            )
            clone._session_artefacts = SessionArtefacts(
                layout=grid.value, coefficients=coeff.value,
                lu_pack=fact.value[Backend.SPARSE_LU],
                banded_pack=fact.value[Backend.BANDED],
            )
            assert clone.price(product, env) == baseline
            for state in (grid, coeff, fact):
                if state.handle is not None:
                    state.handle.close()


class TestBuildCountGates:
    """Phase 4 exit-gate evidence: preparation is built once per session
    across CRN repricings (spec section 21)."""

    def test_dupire_built_once_across_crn_repricings(self, monkeypatch):
        import quantark.asset.equity.engine.pde.pde_execution_adapters as adapters_mod
        from quantark.volmodels import localvol

        engine, product, env = _case("LocalVolSnowballPDESolver")
        counts = {"n": 0}
        original = localvol.build_dupire_local_vol

        def counting(*args, **kwargs):
            counts["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(localvol, "build_dupire_local_vol", counting)
        with PricingSession() as session:
            values = [
                session.execute(
                    engine, PricingRequest(product=product, pricing_env=env)
                ).value
                for _ in range(3)
            ]
        assert counts["n"] == 1
        assert values[0] == values[1] == values[2] == engine.price(product, env)

    def test_grid_and_factorizations_built_once_across_crn_repricings(
        self, monkeypatch
    ):
        from quantark.asset.equity.engine.pde.core import backends

        engine, product, env = _case("SnowballPDESolver")
        direct = engine.price(product, env)

        import quantark.asset.equity.engine.pde.grid.space as grid_space

        grid_counts = {"n": 0}
        original_build_space = grid_space.build_space

        def counting_build_space(*args, **kwargs):
            grid_counts["n"] += 1
            return original_build_space(*args, **kwargs)

        monkeypatch.setattr(grid_space, "build_space", counting_build_space)
        monkeypatch.setattr(
            "quantark.asset.equity.engine.pde.grid.binder.build_space",
            counting_build_space,
        )

        banded_counts = {"n": 0}
        original_banded = backends.banded_factor

        def counting_banded(*args, **kwargs):
            # the one construction site: pack hits and cache hits never reach it
            banded_counts["n"] += 1
            return original_banded(*args, **kwargs)

        monkeypatch.setattr(backends, "banded_factor", counting_banded)

        with PricingSession() as session:
            values = [
                session.execute(
                    engine, PricingRequest(product=product, pricing_env=env)
                ).value
                for _ in range(3)
            ]
            stats = session.context.artifact_cache.stats()
        assert values == [direct] * 3
        assert grid_counts["n"] == 1  # one spatial-grid construction, ever
        # Banded systems are constructed ONLY inside the single warm-up of the
        # factorization pack (once per unique (set, dt, theta) key across the
        # whole session); the three marches hit the injected pack.
        assert 0 < banded_counts["n"] <= 4
        assert stats["misses"] >= 3      # grid + coeffs + pack built once
        assert stats["hits"] >= 6        # ...and reused by dispatches 2 and 3

    def test_cold_cache_lv_autocallable_first_dispatch(self):
        # Plan-gate finding: a fresh session's very FIRST LV dispatch (nothing
        # cached anywhere) must succeed and match direct bitwise.
        for name in ("LocalVolSnowballPDESolver", "LocalVolPhoenixPDESolver"):
            engine, product, env = _case(name)
            direct = engine.price(product, env)
            with PricingSession() as session:
                outcome = session.execute(
                    engine, PricingRequest(product=product, pricing_env=env)
                )
            assert outcome.value == direct, name

    def test_bumped_env_never_reuses_stale_artifacts(self):
        from quantark.param import FlatRateCurve

        engine, product, env = _case("SnowballPDESolver")
        with PricingSession() as session:
            base = session.execute(
                engine, PricingRequest(product=product, pricing_env=env)
            ).value
            env.rate_curve = FlatRateCurve(rate=0.06)  # field REPLACEMENT
            bumped_direct = engine.price(product, env)
            bumped_session = session.execute(
                engine, PricingRequest(product=product, pricing_env=env)
            ).value
        assert bumped_session == bumped_direct
        assert bumped_session != base

    def test_tiny_budget_bypass_stays_bitwise(self):
        import dataclasses

        from quantark.execution.cache.artifacts import PreparedArtifactCache
        from quantark.execution.context import default_context
        from quantark.execution.leases import ResourceLeaseManager

        engine, product, env = _case("SnowballPDESolver")
        direct = engine.price(product, env)
        ctx = default_context()
        budget = dataclasses.replace(
            ctx.resource_budget,
            artifact_cache_bytes=1024,   # too small to admit anything real
            draw_cache_bytes=1024,
        )
        leases = ResourceLeaseManager(budget)
        cache = PreparedArtifactCache(leases)
        ctx = dataclasses.replace(
            ctx, resource_budget=budget, lease_manager=leases,
            artifact_cache=cache,
        )
        with PricingSession(context=ctx) as session:
            values = [
                session.execute(
                    engine, PricingRequest(product=product, pricing_env=env)
                ).value
                for _ in range(2)
            ]
        assert values == [direct] * 2
        assert cache.stats()["bytes_in_use"] == 0  # nothing retained
