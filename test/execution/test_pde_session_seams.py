"""Seam-refactor equivalence (Phase 4 Task 1/2).

price / price_with_events / calculate_spot_greeks_curve now drive the session
seams, so seam outputs and public outputs must be the same values — and the
independent pre-refactor goldens (test_pde_pre_refactor_goldens) pin the
public outputs themselves.
"""
import numpy as np

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
        engine, product, env = _case("EuropeanPDESolver")
        pv, solution = engine._price_with_solution(product, env)
        assert engine.price(product, env) == pv
        assert solution is not None
        assert np.array_equal(
            solution.solution_vec, engine._solve(product, env).solution_vec
        )

    def test_price_equals_seam_pv_snowball(self):
        engine, product, env = _case("SnowballPDESolver")
        pv, solution = engine._price_with_solution(product, env)
        assert engine.price(product, env) == pv
        assert solution is not None


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
    @staticmethod
    def _artifacts(clone, product, env):
        spot = env.spot
        tau = product.get_maturity(env)
        strike = getattr(product, "strike", spot)
        r, q = env.get_rate(tau), env.get_div_yield(tau)
        sigma = env.get_vol(strike, tau)
        grids = clone._build_grids(product, env, spot, sigma, tau, r, q)
        coeffs = clone._build_step_coefficients(
            env, strike, grids[3], grids[2], len(grids[0])
        )
        matrix_pack, banded_pack = clone._session_factorization_packs(
            product, env, grids
        )
        return grids, coeffs, matrix_pack, banded_pack

    def _inject(self, clone, grids, coeffs, matrix_pack, banded_pack):
        clone._session_grids = grids
        clone._session_step_coefficients = coeffs
        clone._session_matrix_pack = matrix_pack
        if hasattr(clone, "_session_banded_pack"):
            clone._session_banded_pack = banded_pack

    def test_injected_artifacts_reproduce_snowball_bitwise(self):
        engine, product, env = _case("SnowballPDESolver")
        baseline = engine.price(product, env)
        clone = type(engine)(params=engine.params)
        self._inject(clone, *self._artifacts(clone, product, env))
        assert clone.price(product, env) == baseline

    def test_injected_artifacts_reproduce_european_bitwise(self):
        engine, product, env = _case("EuropeanPDESolver")
        baseline = engine.price(product, env)
        clone = type(engine)(params=engine.params)
        self._inject(clone, *self._artifacts(clone, product, env))
        assert clone.price(product, env) == baseline

    def test_pack_covers_every_step_key(self):
        """Every (coeff_key, dt, theta) the march requests is prebuilt: the
        per-solve lazy caches stay EMPTY when packs are injected."""
        engine, product, env = _case("SnowballPDESolver")
        clone = type(engine)(params=engine.params)
        self._inject(clone, *self._artifacts(clone, product, env))
        clone.price(product, env)
        assert len(clone._matrix_cache) == 0
        assert len(clone._banded_cache) == 0

    def test_banded_pack_covers_two_surface_march(self):
        from quantark.asset.equity.engine.pde import SnowballPDESolver
        from execution.matrix_fixtures import _pdep, _snowball, _eq_flat_env

        engine = SnowballPDESolver(
            _pdep(grid_size=90, time_steps=48,
                  use_banded_solver=True)
        )
        product, env = _snowball(), _eq_flat_env()
        baseline = engine.price(product, env)
        clone = type(engine)(params=engine.params)
        artifacts = self._artifacts(clone, product, env)
        assert artifacts[3], "banded pack expected under use_banded_solver"
        self._inject(clone, *artifacts)
        assert clone.price(product, env) == baseline
        assert len(clone._banded_cache) == 0
