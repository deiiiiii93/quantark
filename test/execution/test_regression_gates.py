"""Phase 1 regression gates (spec section 20 gates 1-2, CI smoke form)."""
import dataclasses
import time

from quantark.execution import PricingSession, ResourceBudget
from quantark.execution.context import default_context

from execution.matrix_fixtures import FIXTURE_BUILDERS


def test_disabled_cache_matches_enabled_cache():
    """Gate: caches off == caches on, exactly (spec section 20 gate 2)."""
    engine, product, env, _ = FIXTURE_BUILDERS["LocalVolDCNMCEngine"]()
    with PricingSession() as s_on:
        v_on = s_on.price(engine, product, env)
    ctx = dataclasses.replace(
        default_context(),
        resource_budget=ResourceBudget(artifact_cache_bytes=0),
    )
    with PricingSession(ctx) as s_off:
        v_off = s_off.price(engine, product, env)
    assert v_on == v_off


def test_serial_session_overhead_smoke():
    """CI smoke for spec section 20 gate 1 (the 3% gate runs on a controlled
    host; this catches gross regressions only)."""
    engine, product, env, _ = FIXTURE_BUILDERS["EuropeanMCEngine"]()

    def median_seconds(fn, n=15):
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return sorted(times)[n // 2]

    engine.price(product, env)  # warm-up
    direct = median_seconds(lambda: engine.price(product, env))
    with PricingSession() as session:
        session.price(engine, product, env)  # warm-up
        via = median_seconds(lambda: session.price(engine, product, env))
    # generous CI bound: framework layers must stay in the noise, not 2x
    assert via <= direct * 2.0 + 0.005
