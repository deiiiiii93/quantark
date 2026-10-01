"""Seam-refactor equivalence (Phase 4 Task 1/2).

price / price_with_events / calculate_spot_greeks_curve now drive the session
seams, so seam outputs and public outputs must be the same values — and the
independent pre-refactor goldens (test_pde_pre_refactor_goldens) pin the
public outputs themselves.
"""
import numpy as np
from quantark.asset.equity.engine.pde.core.components import CrankNicolson

from execution.matrix_fixtures import FIXTURE_BUILDERS


def _case(name):
    engine, product, env, _shape = FIXTURE_BUILDERS[name]()
    return engine, product, env


def _dist_equal(a, b) -> bool:
    if not np.array_equal(a.event_times, b.event_times):
        return False
    if not np.array_equal(a.survival_probability, b.survival_probability):
        return False
    if set(a.probabilities) != set(b.probabilities):
        return False
    for key, pa in a.probabilities.items():
        pb = b.probabilities[key]
        if not np.array_equal(np.asarray(pa), np.asarray(pb)):
            return False
    return True


class TestPriceWithSolution:
    def test_price_equals_seam_pv_vanilla(self):
        from quantark.asset.equity.engine.pde.core.march import march
        from quantark.asset.equity.engine.pde.core.prepare import (
            Operation, OutputRequest, prepare,
        )

        engine, product, env = _case("EuropeanPDESolver")
        out = engine._session_outputs(product, env, want_grid=True)
        assert engine.price(product, env) == out.npv
        assert out.solution is not None
        assert np.array_equal(
            out.solution.solution_vec,
            march(prepare(engine, product, env, OutputRequest(Operation.PRICE)))
            .states["value"][:, 0],
        )

    def test_price_equals_seam_pv_snowball(self):
        engine, product, env = _case("SnowballPDESolver")
        out = engine._session_outputs(product, env, want_grid=True)
        assert engine.price(product, env) == out.npv
        assert out.solution is not None


class TestSessionOutputs:
    def test_events_match_price_with_events(self):
        engine, product, env = _case("SnowballPDESolver")
        direct = engine.price_with_events(product, env, emit_distribution=True)
        out = engine._session_outputs(product, env, want_events=True)
        assert out.npv == direct.npv
        assert _dist_equal(out.event_distribution, direct.event_distribution)

    def test_grid_matches_spot_greeks_curve(self):
        engine, product, env = _case("SnowballPDESolver")
        levels = [float(env.spot) * m for m in (0.9, 1.0, 1.1)]
        direct = engine.calculate_spot_greeks_curve(product, env, levels)
        out = engine._session_outputs(product, env, want_grid=True)
        assert engine._grid_projection_from_solution(out.solution, levels) == direct

    def test_grid_default_levels_are_the_grid_nodes(self):
        engine, product, env = _case("EuropeanPDESolver")
        out = engine._session_outputs(product, env, want_grid=True)
        rows = engine._grid_projection_from_solution(out.solution)
        assert len(rows) == len(out.solution.s_vec)
        assert rows[0]["spot"] == float(out.solution.s_vec[0])

    def test_lv_snowball_events_match(self):
        engine, product, env = _case("LocalVolSnowballPDESolver")
        direct = engine.price_with_events(product, env, emit_distribution=True)
        out = engine._session_outputs(product, env, want_events=True)
        assert out.npv == direct.npv
        assert _dist_equal(out.event_distribution, direct.event_distribution)

    def test_heston_2d_events_match(self):
        engine, product, env = _case("HestonSnowballPDESolver")
        direct = engine.price_with_events(product, env, emit_distribution=True)
        out = engine._session_outputs(product, env, want_events=True)
        assert out.npv == direct.npv
        assert out.solution is None  # 2D never exposes a 1D grid
        assert _dist_equal(out.event_distribution, direct.event_distribution)

    def test_phoenix_2d_events_match(self):
        engine, product, env = _case("HestonPhoenixPDESolver")
        direct = engine.price_with_events(product, env, emit_distribution=True)
        out = engine._session_outputs(product, env, want_events=True)
        assert out.npv == direct.npv
        assert _dist_equal(out.event_distribution, direct.event_distribution)

    def test_no_emit_matches_plain_price(self):
        engine, product, env = _case("PhoenixPDESolver")
        direct = engine.price_with_events(product, env, emit_distribution=False)
        assert direct.npv == engine.price(product, env)


class TestInjectionSeams:
    """The session artefacts (spec 2 §5.10): the bound Layout, the pre-flat-exact
    step sets, the warmed factorizations -- injected as ``_session_artefacts``."""

    @staticmethod
    def _artifacts(clone, product, env, max_entries=None):
        from quantark.asset.equity.engine.pde.core.groups import Backend
        from quantark.asset.equity.engine.pde.core.march import warm_factorizations
        from quantark.asset.equity.engine.pde.core.prepare import (
            Operation, OutputRequest, SessionArtefacts, bind_layout, prepare, session_step_sets,
        )

        layout = bind_layout(clone, product, env, OutputRequest(Operation.PRICE))
        coeffs = session_step_sets(clone, product, env, layout)
        prepared = prepare(clone, product, env, OutputRequest(Operation.PRICE),
                           session=SessionArtefacts(layout=layout, coefficients=coeffs))
        packs = warm_factorizations(prepared, max_entries=max_entries)
        return SessionArtefacts(layout=layout, coefficients=coeffs,
                                lu_pack=packs[Backend.SPARSE_LU], banded_pack=packs[Backend.BANDED])

    @staticmethod
    def _stores_after_price(clone, product, env):
        """The prepared problem's own factorization stores after its march."""
        from quantark.asset.equity.engine.pde.core.groups import Backend
        from quantark.asset.equity.engine.pde.core.prepare import Operation, OutputRequest, prepare, run

        prepared = prepare(clone, product, env, OutputRequest(Operation.PRICE),
                           session=clone._session_artefacts)
        run(prepared)
        return {b: len(prepared.factorizations.store(b)) for b in Backend}

    def test_injected_artifacts_reproduce_snowball_bitwise(self):
        engine, product, env = _case("SnowballPDESolver")
        baseline = engine.price(product, env)
        clone = type(engine)(params=engine.params)
        clone._session_artefacts = self._artifacts(clone, product, env)
        assert clone.price(product, env) == baseline

    def test_injected_artifacts_reproduce_european_bitwise(self):
        engine, product, env = _case("EuropeanPDESolver")
        baseline = engine.price(product, env)
        clone = type(engine)(params=engine.params)
        clone._session_artefacts = self._artifacts(clone, product, env)
        assert clone.price(product, env) == baseline

    def test_pack_covers_every_step_key(self):
        """Every (set, dt, theta) the march requests is prebuilt: the march's own
        stores stay EMPTY when the warmed packs are injected."""
        engine, product, env = _case("SnowballPDESolver")
        clone = type(engine)(params=engine.params)
        clone._session_artefacts = self._artifacts(clone, product, env)
        assert set(self._stores_after_price(clone, product, env).values()) == {0}

    def test_banded_pack_covers_two_surface_march(self):
        from quantark.asset.equity.engine.pde import SnowballPDESolver
        from execution.matrix_fixtures import _pdep, _snowball, _eq_flat_env

        engine = SnowballPDESolver(
            _pdep()
        )
        product, env = _snowball(), _eq_flat_env()
        baseline = engine.price(product, env)
        clone = type(engine)(params=engine.params)
        artifacts = self._artifacts(clone, product, env)
        assert artifacts.banded_pack, "banded pack expected for the two-surface family"
        clone._session_artefacts = artifacts
        assert clone.price(product, env) == baseline
        assert set(self._stores_after_price(clone, product, env).values()) == {0}


class TestSessionPreparation:
    """pde-prep/v2 (Task 10): the grid artefact is the Layout execute binds."""

    def test_builder_version(self):
        from quantark.asset.equity.engine.pde import pde_session_prep

        assert pde_session_prep._BUILDER_VERSION == "pde-prep/v3"

    def test_factorization_fingerprint_moves_with_the_stepper(self):
        from quantark.asset.equity.engine.pde import SnowballPDESolver
        from quantark.asset.equity.param import PDEParams
        from quantark.execution import PricingSession

        from execution.matrix_fixtures import _snowball, _eq_flat_env
        from quantark.asset.equity.engine.pde.pde_execution_adapters import (
            EquityPDEAutocallableSessionAdapter,
        )
        from quantark.execution import PricingRequest

        product, env = _snowball(), _eq_flat_env()
        fps = []
        for params in (PDEParams(), PDEParams(stepper=CrankNicolson(theta=1.0))):
            engine = SnowballPDESolver(params)
            with PricingSession() as session:
                state = EquityPDEAutocallableSessionAdapter().prepare(
                    engine, PricingRequest(product=product, pricing_env=env), session.context)
                kinds = {d.kind: d.fingerprint for d in state.descriptors}
                for handle in state.handles:
                    handle.close()
            fps.append(kinds)
        assert fps[0]["pde-grid"] == fps[1]["pde-grid"]
        assert fps[0]["pde-factorization-pack"] != fps[1]["pde-factorization-pack"]

    def test_stored_layout_is_the_one_execute_binds(self, monkeypatch):
        from quantark.asset.equity.engine.pde import base_pde_solver
        from quantark.execution import PricingRequest, PricingSession
        from quantark.execution.cache.artifacts import PreparedArtifactCache

        engine, product, env = _case("SnowballPDESolver")
        stored, bound = {}, []
        original_get = PreparedArtifactCache.get_or_build

        def get_spy(self, descriptor, builder, size_bytes, measure=None):
            handle = original_get(self, descriptor, builder, size_bytes, measure)
            stored[descriptor.kind] = handle.value
            return handle

        original_prepare = base_pde_solver.prepare   # the solver's _run prepares through it

        def prepare_spy(*args, **kwargs):
            problem = original_prepare(*args, **kwargs)
            bound.append(problem)
            return problem

        monkeypatch.setattr(PreparedArtifactCache, "get_or_build", get_spy)
        with PricingSession() as session:
            session.execute(engine, PricingRequest(product=product, pricing_env=env))
            bound.clear()
            monkeypatch.setattr(base_pde_solver, "prepare", prepare_spy)
            session.execute(engine, PricingRequest(product=product, pricing_env=env))
        executed = [p for p in bound if getattr(p, "layout", None) is not None]
        assert executed and all(p.layout is stored["pde-grid"] for p in executed)

    def test_phoenix_and_ko_reset_honour_injected_coefficients(self, monkeypatch):
        from quantark.asset.equity.engine.pde import base_pde_solver
        from quantark.asset.equity.engine.pde.core.components import CentralDifference
        from quantark.asset.equity.engine.pde.pde_execution_adapters import _EquityPDESessionBase
        from quantark.execution import PricingRequest, PricingSession

        calls = {"n": 0, "executing": False}
        original_assemble = CentralDifference.assemble_sets

        def assemble_spy(self, *args, **kwargs):
            if calls["executing"]:
                calls["n"] += 1
            return original_assemble(self, *args, **kwargs)

        original_execute = _EquityPDESessionBase.execute_native

        def execute_spy(self, *args, **kwargs):
            base_pde_solver._ENV_STEP_COEFF_MEMO.clear()   # no memo can stand in for the injection
            calls["executing"] = True
            try:
                return original_execute(self, *args, **kwargs)
            finally:
                calls["executing"] = False

        monkeypatch.setattr(CentralDifference, "assemble_sets", assemble_spy)
        monkeypatch.setattr(_EquityPDESessionBase, "execute_native", execute_spy)
        for name in ("PhoenixPDESolver", "KOResetSnowballPDESolver"):
            engine, product, env = _case(name)
            with PricingSession() as session:
                value = session.execute(engine, PricingRequest(product=product, pricing_env=env)).value
            assert value == engine.price(product, env), name
        assert calls["n"] == 0
