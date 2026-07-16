"""Phase 4 artifact-state and reuse gates (spec sections 9.2/10.1/21)."""
import pytest

from quantark.asset.equity.engine.pde.pde_session_prep import (
    factorization_state,
    grid_state,
    market_scalars,
    step_coefficients_state,
)
from quantark.execution import PricingRequest, PricingSession
from quantark.execution.cache.fingerprint import try_fingerprint
from quantark.execution.errors import DeterminismViolation

from execution.matrix_fixtures import FIXTURE_BUILDERS


def _case(name):
    engine, product, env, _shape = FIXTURE_BUILDERS[name]()
    return engine, product, env


def _curves_fp(env):
    return try_fingerprint((env.rate_curve, env.div_yield, env.vol_surface))


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
        engine._cache_strategy = "disable"
        with PricingSession() as session:
            state = grid_state(engine, product, env, session.context)
            assert state.descriptor is None and state.handle is None
            assert state.value is not None

    def test_chain_fingerprints_compose(self):
        engine, product, env = _case("SnowballPDESolver")
        with PricingSession() as session:
            ctx = session.context
            grid = grid_state(engine, product, env, ctx)
            coeff = step_coefficients_state(
                engine, product, env, grid.value, grid.fingerprint,
                _curves_fp(env), ctx,
            )
            engine._session_grids = grid.value
            engine._session_step_coefficients = coeff.value
            try:
                fact = factorization_state(
                    engine, product, env, grid.value, coeff.fingerprint, ctx
                )
                assert grid.fingerprint and coeff.fingerprint and fact.fingerprint
                matrix_pack, banded_pack = fact.value
                assert len(matrix_pack) + len(banded_pack) > 0
                with pytest.raises(TypeError):
                    matrix_pack["x"] = 1  # published packs are immutable
                for state in (grid, coeff, fact):
                    if state.handle is not None:
                        state.handle.close()
            finally:
                engine._session_grids = None
                engine._session_step_coefficients = None

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
            coeff = step_coefficients_state(
                clone, product, env, grid.value, grid.fingerprint,
                _curves_fp(env), ctx,
            )
            clone._session_grids = grid.value
            clone._session_step_coefficients = coeff.value
            fact = factorization_state(
                clone, product, env, grid.value, coeff.fingerprint, ctx
            )
            clone._session_matrix_pack, clone._session_banded_pack = fact.value
            assert clone.price(product, env) == baseline
            for state in (grid, coeff, fact):
                if state.handle is not None:
                    state.handle.close()

    def test_market_scalars_match_solve_preamble(self):
        engine, product, env = _case("EuropeanPDESolver")
        spot, strike, tau, r, q, sigma = market_scalars(product, env)
        assert spot == env.spot
        assert tau == product.get_maturity(env)
        assert r == env.get_rate(tau) and q == env.get_div_yield(tau)
        assert sigma == env.get_vol(strike, tau)
