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
