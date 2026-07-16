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


# ---------------------------------------------------------------------------
# Phase 2 structural gates (deterministic stand-ins for the spec section 20
# wall-clock gates; the measured gates run via benchmark_phase2.py)
# ---------------------------------------------------------------------------

def _threads_context(workers, **budget_kw):
    from quantark.execution.policy import ExecutionPolicy, ExecutorSelection

    budget_kw.setdefault("max_threads", 8)
    budget_kw.setdefault("max_in_flight", 8)
    return dataclasses.replace(
        default_context(),
        execution_policy=ExecutionPolicy(
            batch=ExecutorSelection(backend="threads", workers=workers)
        ),
        resource_budget=ResourceBudget(**budget_kw),
    )


def _result_fields_equal(a, b):
    for field in type(a).__dataclass_fields__:
        if field in ("elapsed_seconds", "event_stats"):
            continue
        assert getattr(a, field) == getattr(b, field), field


def test_threads_any_worker_count_bitwise_equal_serial():
    """Gate: canonical reduction makes every worker count bit-identical."""
    from quantark.execution.contracts import PricingOperation, PricingRequest

    def build_engine():
        from quantark.asset.equity.engine.mc import DCNMCEngine

        return DCNMCEngine(num_paths=2048, num_batches=8, seed=11)

    _, product, env, _ = FIXTURE_BUILDERS["DCNMCEngine"]()
    direct = build_engine().price_detailed(product, env)
    request = PricingRequest(
        product=product, pricing_env=env,
        operation=PricingOperation.PRICE_DETAILED,
    )
    for workers in (1, 2, 3, 8):
        with PricingSession(_threads_context(workers)) as session:
            outcome = session.execute(build_engine(), request)
        _result_fields_equal(outcome.value, direct)


def test_no_unbounded_outcome_retention_in_scramble_means_mode():
    """Gate: compact outcomes carry event-vector-sized arrays, never paths
    (spec section 20 gate 8)."""
    from quantark.asset.equity.engine.mc.dcn_execution_adapters import (
        DCNBatchMCAdapter,
    )
    from quantark.execution.contracts import PricingRequest

    engine, product, env, _ = FIXTURE_BUILDERS["DCNMCEngine"]()
    adapter = DCNBatchMCAdapter()
    request = PricingRequest(product=product, pricing_env=env)
    context = default_context()
    state = adapter.prepare(engine, request, context)
    plan = adapter.plan_batches(engine, request, state, context)
    assert plan.stderr_mode == "scramble_means" or plan.num_batches == 1
    for task in plan.tasks:
        outcome = adapter.execute_batch(task, state, context)
        stats = outcome.payload
        if plan.stderr_mode == "scramble_means":
            assert stats.totals is None
        for name in ("fixed_by_period", "ko_by_period", "coupon_paid",
                     "ko_timing"):
            assert getattr(stats, name).size == state.payload.n_obs


def test_crn_repricing_reuses_draws_and_surfaces():
    """Gate: a 10-point CRN spot ladder in one session regenerates NO draw
    block (draw misses stay at num_batches) and builds each distinct Dupire
    surface exactly once (spec section 20 gate 4, structural form)."""
    from quantark.param import SpotQuote
    from quantark.execution.contracts import PricingRequest

    engine, product, env, _ = FIXTURE_BUILDERS["LocalVolDCNMCEngine"]()
    assert engine.use_sobol and engine.num_batches >= 1
    with PricingSession() as session:
        for bump in range(10):
            bumped = dataclasses.replace(
                env, spot_quote=SpotQuote(spot=env.spot * (1 + 0.001 * bump))
            )
            session.execute(
                engine, PricingRequest(product=product, pricing_env=bumped)
            )
        draws = session.context.draw_repository.stats()
        cache = session.context.artifact_cache.stats()
        assert draws["misses"] == engine.num_batches  # shared across ladder
        assert draws["hits"] == 9 * engine.num_batches
        assert cache["misses"] == 10   # one Dupire build per distinct spot
        # repricing the same ladder is all hits everywhere
        for bump in range(10):
            bumped = dataclasses.replace(
                env, spot_quote=SpotQuote(spot=env.spot * (1 + 0.001 * bump))
            )
            session.execute(
                engine, PricingRequest(product=product, pricing_env=bumped)
            )
        assert session.context.draw_repository.stats()["misses"] == (
            engine.num_batches
        )
        assert session.context.artifact_cache.stats()["misses"] == 10


def test_disabled_draw_and_artifact_caches_bitwise_exact():
    """Gate: correctness never depends on caching (spec section 3.3)."""
    engine, product, env, _ = FIXTURE_BUILDERS["LocalVolDCNMCEngine"]()
    direct = engine.price_detailed(product, env)
    ctx = dataclasses.replace(
        default_context(),
        resource_budget=ResourceBudget(
            artifact_cache_bytes=0, draw_cache_bytes=0
        ),
    )
    with PricingSession(ctx) as session:
        from quantark.execution.contracts import (
            PricingOperation, PricingRequest,
        )

        outcome = session.execute(
            engine,
            PricingRequest(
                product=product, pricing_env=env,
                operation=PricingOperation.PRICE_DETAILED,
            ),
        )
    _result_fields_equal(outcome.value, direct)
