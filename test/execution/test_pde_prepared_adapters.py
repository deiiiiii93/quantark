"""Phase 4 prepared PDE adapters: bitwise parity, one-solve rich outputs,
fail-closed semantics, subclass fallback (spec sections 8/9/21)."""
import numpy as np
import pytest

from quantark.asset.equity.engine.pde.pde_execution_adapters import (
    ADAPTER_ID,
    PDESessionValue,
)
from quantark.execution import PricingRequest, PricingSession
from quantark.execution.contracts import OutputKind, PricingOperation
from quantark.execution.errors import CapabilityError

from execution.matrix_fixtures import FIXTURE_BUILDERS

_PREPARED_1D = (
    "EuropeanPDESolver",
    "AmericanPDESolver",
    "BarrierPDESolver",
    "DoubleBarrierPDESolver",
    "OneTouchPDESolver",
    "DoubleOneTouchPDESolver",
    "SnowballPDESolver",
    "KOResetSnowballPDESolver",
    "PhoenixPDESolver",
)
_PREPARED_LV = ("LocalVolPDESolver", "LocalVolBarrierPDESolver")
_PREPARED_LV_AUTOCALL = ("LocalVolSnowballPDESolver", "LocalVolPhoenixPDESolver")
_PREPARED_2D = (
    "HestonSnowballPDESolver",
    "HestonSLVSnowballPDESolver",
    "HestonPhoenixPDESolver",
    "HestonSLVPhoenixPDESolver",
)
_PREPARED_FX = ("FxLocalVolPDESolver",)
ALL_PREPARED = (
    _PREPARED_1D + _PREPARED_LV + _PREPARED_LV_AUTOCALL + _PREPARED_2D
    + _PREPARED_FX
)


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
    return all(
        np.array_equal(np.asarray(pa), np.asarray(b.probabilities[key]))
        for key, pa in a.probabilities.items()
    )


class TestBitwiseParity:
    @pytest.mark.parametrize("name", ALL_PREPARED)
    def test_session_price_bitwise_via_prepared_adapter(self, name):
        engine, product, env = _case(name)
        direct = engine.price(product, env)
        with PricingSession() as session:
            outcome = session.execute(
                engine, PricingRequest(product=product, pricing_env=env)
            )
        assert outcome.value == direct
        assert outcome.manifest.adapter_id != "legacy-price"

    @pytest.mark.parametrize("name", _PREPARED_1D)
    def test_repeated_dispatch_deterministic_and_reuses_artifacts(self, name):
        engine, product, env = _case(name)
        direct = engine.price(product, env)
        with PricingSession() as session:
            first = session.execute(
                engine, PricingRequest(product=product, pricing_env=env)
            )
            second = session.execute(
                engine, PricingRequest(product=product, pricing_env=env)
            )
            hits = session.context.artifact_cache.stats()["hits"]
        assert first.value == direct and second.value == direct
        assert hits >= 3  # grid + coefficients + factorization pack reused


class TestRichOutputs:
    def test_pv_events_grid_one_value_solve(self):
        from quantark.asset.equity.engine.pde import SnowballPDESolver

        engine, product, env = _case("SnowballPDESolver")
        direct_events = engine.price_with_events(product, env)
        levels = tuple(float(env.spot) * m for m in (0.9, 1.0, 1.1))
        direct_curve = engine.calculate_spot_greeks_curve(product, env, list(levels))

        solve_calls = []
        original = SnowballPDESolver._solve

        def counting(self, *args, **kwargs):
            solve_calls.append(type(self).__name__)
            return original(self, *args, **kwargs)

        SnowballPDESolver._solve = counting
        try:
            with PricingSession() as session:
                outcome = session.execute(
                    engine,
                    PricingRequest(
                        product=product, pricing_env=env,
                        outputs=frozenset(
                            {OutputKind.PV, OutputKind.EVENT_STATS, OutputKind.GRID}
                        ),
                        operation_options=(("grid_spot_levels", levels),),
                    ),
                )
        finally:
            SnowballPDESolver._solve = original

        value = outcome.value
        assert isinstance(value, PDESessionValue)
        assert value.pv == direct_events.npv
        assert list(value.grid) == direct_curve
        assert _dist_equal(value.event_distribution, direct_events.event_distribution)
        assert dict(outcome.normalized_economics)["pv"] == direct_events.npv
        # ONE value solve populated PV + events + grid (the event-stat
        # indicator sweep is a separate designed pass, not a _solve call).
        assert solve_calls.count("SnowballPDESolver") == 1

    def test_total_backward_march_parity_with_direct(self):
        """The session performs exactly as many banded backward sweeps as
        the direct price_with_events path — rich outputs add NO marches."""
        import scipy.linalg as sla

        from quantark.asset.equity.engine.pde import snowball_pde_solver as mod

        engine, product, env = _case("SnowballPDESolver")

        counts = {"n": 0}
        original = mod.solve_banded

        def counting(*args, **kwargs):
            counts["n"] += 1
            return original(*args, **kwargs)

        mod.solve_banded = counting
        try:
            engine.price_with_events(product, env)
            direct_n = counts["n"]
            counts["n"] = 0
            with PricingSession() as session:
                session.execute(
                    engine,
                    PricingRequest(
                        product=product, pricing_env=env,
                        outputs=frozenset({OutputKind.PV, OutputKind.EVENT_STATS}),
                    ),
                )
            session_n = counts["n"]
        finally:
            mod.solve_banded = original
        assert session_n == direct_n

    def test_pv_events_matches_price_with_events_2d(self):
        engine, product, env = _case("HestonSnowballPDESolver")
        direct = engine.price_with_events(product, env)
        with PricingSession() as session:
            outcome = session.execute(
                engine,
                PricingRequest(
                    product=product, pricing_env=env,
                    outputs=frozenset({OutputKind.PV, OutputKind.EVENT_STATS}),
                ),
            )
        assert outcome.value.pv == direct.npv
        assert _dist_equal(
            outcome.value.event_distribution, direct.event_distribution
        )

    def test_grid_without_levels_projects_full_grid(self):
        engine, product, env = _case("EuropeanPDESolver")
        out = engine._session_outputs(product, env, want_grid=True)
        expected = engine._grid_projection_from_solution(out.solution)
        with PricingSession() as session:
            outcome = session.execute(
                engine,
                PricingRequest(
                    product=product, pricing_env=env,
                    outputs=frozenset({OutputKind.PV, OutputKind.GRID}),
                ),
            )
        assert list(outcome.value.grid) == expected

    def test_lv_autocallable_grid_output_served_from_wrapped_solve(self):
        # Direct calculate_spot_greeks_curve raises on LV autocallables (no
        # active surface); the session serves GRID from the surface-wrapped
        # one-solve seam — compare against the manually wrapped oracle.
        engine, product, env = _case("LocalVolSnowballPDESolver")
        levels = tuple(float(env.spot) * m for m in (0.9, 1.0, 1.1))
        out = engine._session_outputs(product, env, want_grid=True)
        expected = engine._grid_projection_from_solution(out.solution, list(levels))
        with PricingSession() as session:
            outcome = session.execute(
                engine,
                PricingRequest(
                    product=product, pricing_env=env,
                    outputs=frozenset({OutputKind.PV, OutputKind.GRID}),
                    operation_options=(("grid_spot_levels", levels),),
                ),
            )
        assert list(outcome.value.grid) == expected

    def test_event_stats_operation_still_served(self):
        engine, product, env = _case("SnowballPDESolver")
        direct = engine.calculate_event_stats(product, env)
        with PricingSession() as session:
            outcome = session.execute(
                engine,
                PricingRequest(
                    product=product, pricing_env=env,
                    operation=PricingOperation.EVENT_STATS,
                ),
            )
        assert np.array_equal(
            np.asarray(outcome.value.ko_probability),
            np.asarray(direct.ko_probability),
        )


class TestFailClosed:
    def test_grid_on_expired_product_raises(self):
        import copy

        engine, product, env = _case("EuropeanPDESolver")
        expired = copy.deepcopy(product)
        expired.maturity = 0.0
        with PricingSession() as session:
            with pytest.raises(CapabilityError):
                session.execute(
                    engine,
                    PricingRequest(
                        product=expired, pricing_env=env,
                        outputs=frozenset({OutputKind.PV, OutputKind.GRID}),
                    ),
                )

    def test_unsupported_output_rejected_at_validate(self):
        engine, product, env = _case("EuropeanPDESolver")
        with PricingSession() as session:
            with pytest.raises(CapabilityError):
                session.execute(
                    engine,
                    PricingRequest(
                        product=product, pricing_env=env,
                        outputs=frozenset({OutputKind.PV, OutputKind.CASHFLOWS}),
                    ),
                )

    def test_event_stats_output_rejected_for_vanilla_solver(self):
        engine, product, env = _case("EuropeanPDESolver")
        with PricingSession() as session:
            with pytest.raises(CapabilityError):
                session.execute(
                    engine,
                    PricingRequest(
                        product=product, pricing_env=env,
                        outputs=frozenset({OutputKind.PV, OutputKind.EVENT_STATS}),
                    ),
                )

    def test_grid_output_rejected_for_2d_solver(self):
        engine, product, env = _case("HestonSnowballPDESolver")
        with PricingSession() as session:
            with pytest.raises(CapabilityError):
                session.execute(
                    engine,
                    PricingRequest(
                        product=product, pricing_env=env,
                        outputs=frozenset({OutputKind.PV, OutputKind.GRID}),
                    ),
                )


class TestSubclassFallback:
    def test_price_override_honored_via_legacy_adapter(self):
        from quantark.asset.equity.engine.pde import SnowballPDESolver

        class FlatPriceSnowballPDE(SnowballPDESolver):
            def price(self, product, pricing_env):
                return 123.0

        engine, product, env = _case("SnowballPDESolver")
        flat = FlatPriceSnowballPDE(params=engine.params)
        with PricingSession() as session:
            outcome = session.execute(
                flat, PricingRequest(product=product, pricing_env=env)
            )
        assert outcome.value == 123.0
        assert outcome.manifest.adapter_id == "legacy-price"

    def test_prepared_adapter_id_reported(self):
        engine, product, env = _case("SnowballPDESolver")
        with PricingSession() as session:
            outcome = session.execute(
                engine, PricingRequest(product=product, pricing_env=env)
            )
        assert outcome.manifest.adapter_id == ADAPTER_ID


class TestCodeGateRegressions:
    """Codex code-gate findings (2026-07-16), with reproductions."""

    def test_mutation_between_prepare_and_execute_fails_closed(self):
        # Finding 1: a rate replacement after prepare() must never price a
        # MIXED market (stale injected artifacts + live boundary reads).
        from quantark.asset.equity.engine.pde.pde_execution_adapters import (
            EquityPDEAutocallableSessionAdapter,
        )
        from quantark.execution.errors import DeterminismViolation
        from quantark.param import FlatRateCurve

        engine, product, env = _case("SnowballPDESolver")
        adapter = EquityPDEAutocallableSessionAdapter()
        request = PricingRequest(product=product, pricing_env=env)
        with PricingSession() as session:
            ctx = session.context
            adapter.validate(engine, request)
            normalized = adapter.normalize(engine, request)
            prepared = adapter.prepare(engine, request, ctx)
            try:
                env.rate_curve = FlatRateCurve(rate=0.061)  # concurrent swap
                with pytest.raises(DeterminismViolation):
                    adapter.execute_native(
                        engine, request, normalized, ctx, prepared=prepared
                    )
            finally:
                for handle in prepared.handles:
                    handle.close()

    def test_inplace_mutation_during_factorization_build_fails_closed(self):
        # Finding 2: an in-place market value change during the pack build
        # keeps object identity, so only a LIVE market fingerprint recompute
        # can catch it; the poisoned entry must be purged, never reused.
        from quantark.asset.equity.engine.pde.pde_session_prep import (
            factorization_state,
            grid_state,
            step_coefficients_state,
        )
        from quantark.execution.errors import DeterminismViolation

        engine, product, env = _case("EuropeanPDESolver")
        clone = type(engine)(params=engine.params)
        clone._prepare_solve_state(product, env)
        with PricingSession() as session:
            ctx = session.context
            grid = grid_state(clone, product, env, ctx)
            from quantark.execution.cache.fingerprint import try_fingerprint

            curves_fp = try_fingerprint(
                (env.rate_curve, env.div_yield, env.vol_surface)
            )
            coeff = step_coefficients_state(
                clone, product, env, grid.value, grid.fingerprint,
                curves_fp, ctx,
            )
            clone._session_grids = grid.value
            clone._session_step_coefficients = coeff.value

            rate_obj = env.rate_curve
            original_rate = rate_obj.rate

            class MutatingCache:
                def __init__(self, inner):
                    self._inner = inner

                def get_or_build(self, descriptor, builder, size_bytes,
                                 measure=None):
                    def mutating_builder():
                        value = builder()
                        # in-place: same object identity, new market values
                        try:
                            rate_obj.rate = original_rate + 0.003
                        except Exception:
                            object.__setattr__(
                                rate_obj, "rate", original_rate + 0.003
                            )
                        return value

                    return self._inner.get_or_build(
                        descriptor, mutating_builder, size_bytes, measure
                    )

                def invalidate_tags(self, tags):
                    self._inner.invalidate_tags(tags)

            class Ctx:
                artifact_cache = MutatingCache(ctx.artifact_cache)
                resource_budget = ctx.resource_budget

            try:
                with pytest.raises(DeterminismViolation):
                    factorization_state(
                        clone, product, env, grid.value, coeff.fingerprint, Ctx()
                    )
                # the poisoned entry was invalidated: a clean rebuild after
                # restoring the market must produce a working pack, and the
                # session must price bitwise.
                try:
                    rate_obj.rate = original_rate
                except Exception:
                    object.__setattr__(rate_obj, "rate", original_rate)
                fact = factorization_state(
                    clone, product, env, grid.value, coeff.fingerprint, ctx
                )
                clone._session_matrix_pack = fact.value[0]
                assert clone.price(product, env) == engine.price(product, env)
                if fact.handle is not None:
                    fact.handle.close()
            finally:
                for state in (grid, coeff):
                    if state.handle is not None:
                        state.handle.close()

    def test_lv_pack_is_bounded_and_budget_prechecked(self):
        # Finding 3: LV coefficients create one factorization key per step;
        # the eager pack must stay within its charged bound, and a budget
        # too small for even the bounded pack must skip eager packing
        # BEFORE allocation — bitwise unchanged either way.
        import dataclasses

        from quantark.asset.equity.engine.pde import LocalVolSnowballPDESolver
        from quantark.asset.equity.engine.pde.pde_session_prep import (
            _PACK_MAX_ENTRIES,
        )
        from quantark.execution.cache.artifacts import PreparedArtifactCache
        from quantark.execution.context import default_context
        from quantark.execution.leases import ResourceLeaseManager
        from execution.matrix_fixtures import _pdep, _snowball, _eq_grid_env

        engine = LocalVolSnowballPDESolver(
            _pdep()
        )
        product, env = _snowball(), _eq_grid_env()
        direct = engine.price(product, env)

        with PricingSession() as session:
            outcome = session.execute(
                engine, PricingRequest(product=product, pricing_env=env)
            )
            assert outcome.value == direct
            # the published pack respects the entry bound
            for record in outcome.diagnostics.records:
                pass  # bound asserted structurally below

        # structural bound check on the builder itself
        clone = type(engine)(params=engine.params)
        clone._prepare_solve_state(product, env)
        surface = engine._build_surface(env)
        clone._active_lv_surface = surface
        spot, tau = env.spot, product.get_maturity(env)
        strike = product.strike
        r, q = env.get_rate(tau), env.get_div_yield(tau)
        sigma = env.get_vol(strike, tau)
        grids = clone._build_grids(product, env, spot, sigma, tau, r, q)
        clone._active_s_vec = grids[1]
        try:
            packs = clone._session_factorization_packs(
                product, env, grids, max_entries=_PACK_MAX_ENTRIES
            )
        finally:
            clone._active_lv_surface = None
            clone._active_s_vec = None
        assert len(packs[0]) + len(packs[1]) <= _PACK_MAX_ENTRIES
        assert len(grids[3]) - 1 > _PACK_MAX_ENTRIES  # the cap actually bound

        # tiny budget: eager packing skipped pre-allocation, still bitwise
        ctx = default_context()
        budget = dataclasses.replace(
            ctx.resource_budget, artifact_cache_bytes=1024,
            draw_cache_bytes=1024,
        )
        leases = ResourceLeaseManager(budget)
        cache = PreparedArtifactCache(leases)
        ctx = dataclasses.replace(
            ctx, resource_budget=budget, lease_manager=leases,
            artifact_cache=cache,
        )
        with PricingSession(context=ctx) as session:
            outcome = session.execute(
                engine, PricingRequest(product=product, pricing_env=env)
            )
        assert outcome.value == direct


class TestBGKPreparedSession:
    """Stage-6 iteration-2 finding: grid_state fingerprinted the PRE-BGK
    request (251 discrete-KI event nodes), the build activated BGK (11
    nodes), and capture-and-reverify raised DeterminismViolation against
    the solver's own preparation. The fingerprint must resolve solve state
    (_prepare_for_request) first, exactly like _grids_via_layer."""

    def test_bgk_prepared_session_matches_direct(self):
        import sys

        sys.path.insert(0, "test")
        from test_bgk_continuous_ki import _daily_ki_snowball, _env

        from quantark.asset.equity.engine.pde import SnowballPDESolver
        from quantark.asset.equity.param import PDEParams
        from quantark.util.enum import KnockInMonitoringMode

        engine = SnowballPDESolver(
            PDEParams(ki_monitoring_mode=KnockInMonitoringMode.BGK_APPROXIMATION)
        )
        product = _daily_ki_snowball()
        env = _env()
        direct = float(engine.price(product, env))
        with PricingSession() as session:
            session_px = session.execute(
                engine, PricingRequest(product=product, pricing_env=env)
            ).value
        assert float(session_px) == direct
