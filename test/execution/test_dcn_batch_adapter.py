"""DCN batch adapter: plans, compact outcomes, exact reduction, kernel path."""
import dataclasses

import pytest

from quantark.execution.backends import serial
from quantark.execution.context import default_context
from quantark.execution.contracts import PricingOperation, PricingRequest

from dcn_fixtures import DCN_A, FLAT, flat_env, make_dcn


def make_dcn_product_env():
    return make_dcn(DCN_A), flat_env(**FLAT)


def make_engine(**kwargs):
    from quantark.asset.equity.engine.mc import DCNMCEngine

    kwargs.setdefault("seed", 7)
    return DCNMCEngine(**kwargs)


def make_adapter():
    from quantark.asset.equity.engine.mc.dcn_execution_adapters import (
        DCNBatchMCAdapter,
    )

    return DCNBatchMCAdapter()


def price_via_adapter(engine, product, env, context=None):
    adapter = make_adapter()
    request = PricingRequest(
        product=product, pricing_env=env,
        operation=PricingOperation.PRICE_DETAILED,
    )
    context = context or default_context()
    state = adapter.prepare(engine, request, context)
    try:
        plan = adapter.plan_batches(engine, request, state, context)
        outcomes = (
            o for _, o in serial.iter_ordered(
                plan, lambda t: adapter.execute_batch(t, state, context)
            )
        )
        value, economics = adapter.reduce_batches(
            outcomes, plan, state, context
        )
        return value, economics, plan, state
    finally:
        for handle in state.handles:
            handle.close()


def assert_results_bitwise_equal(actual, expected):
    for field in type(expected).__dataclass_fields__:
        if field in ("elapsed_seconds", "event_stats"):
            continue
        assert getattr(actual, field) == getattr(expected, field), field


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(num_paths=2048, num_batches=4, use_sobol=True),      # scramble_means
        dict(num_paths=2048, num_batches=1, use_sobol=True),      # pathwise, 1 batch
        dict(num_paths=2048, num_batches=4, use_sobol=False),     # pathwise IID
        dict(num_paths=2048, num_batches=4, use_sobol=False,
             use_antithetic=True),                                # antithetic
    ],
    ids=["sobol4", "sobol1", "prng4", "antithetic4"],
)
def test_bitwise_identical_to_direct(kwargs):
    product, env = make_dcn_product_env()
    direct = make_engine(**kwargs).price_detailed(product, env)
    value, economics, _, _ = price_via_adapter(
        make_engine(**kwargs), product, env
    )
    assert_results_bitwise_equal(value, direct)
    assert dict(economics)["pv"] == direct.pv
    assert dict(economics)["std_error"] == direct.std_error


def test_outcome_payload_is_compact():
    product, env = make_dcn_product_env()
    engine = make_engine(num_paths=2048, num_batches=4, use_sobol=True)
    adapter = make_adapter()
    request = PricingRequest(product=product, pricing_env=env)
    context = default_context()
    state = adapter.prepare(engine, request, context)
    plan = adapter.plan_batches(engine, request, state, context)
    outcome = adapter.execute_batch(plan.tasks[0], state, context)
    stats = outcome.payload
    assert stats.totals is None  # scramble_means: no per-path retention
    for name in ("fixed_by_period", "ko_by_period", "coupon_paid", "ko_timing"):
        assert getattr(stats, name).size == state.payload.n_obs


def test_plan_reflects_engine_configuration():
    product, env = make_dcn_product_env()
    engine = make_engine(num_paths=4096, num_batches=8, use_sobol=True)
    adapter = make_adapter()
    request = PricingRequest(product=product, pricing_env=env)
    context = default_context()
    state = adapter.prepare(engine, request, context)
    plan = adapter.plan_batches(engine, request, state, context)
    assert plan.num_batches == 8 and plan.paths_per_batch == 512
    assert plan.stderr_mode == "scramble_means"
    assert plan.tasks[3].batch_id == 3
    assert plan.est_task_peak_bytes > 0
    assert plan.est_outcome_bytes > 0


def test_clone_never_mutates_borrowed_engine():
    product, env = make_dcn_product_env()
    engine = make_engine(num_paths=1024, num_batches=2, num_workers=4)
    price_via_adapter(engine, product, env)
    assert engine._last_result is None
    assert engine._draw_provider is None
    assert engine.num_workers == 4


# ---------------------------------------------------------------------------
# Kernel batch dispatch (Task 7)
# ---------------------------------------------------------------------------

def make_batch_context(workers=1, backend="serial", **budget_kw):
    from quantark.execution.cache.artifacts import PreparedArtifactCache
    from quantark.execution.cache.draws import DrawRepository
    from quantark.execution.leases import ResourceLeaseManager
    from quantark.execution.policy import (
        ExecutionPolicy, ExecutorSelection, ResourceBudget,
    )

    budget_kw.setdefault("max_in_flight", 8)  # batch tasks hold per-task slots
    budget = ResourceBudget(
        max_threads=8, artifact_cache_bytes=64 * 2**20,
        draw_cache_bytes=64 * 2**20, **budget_kw,
    )
    leases = ResourceLeaseManager(budget)
    context = default_context()
    return dataclasses.replace(
        context,
        execution_policy=ExecutionPolicy(
            batch=ExecutorSelection(backend=backend, workers=workers)
        ),
        resource_budget=budget,
        lease_manager=leases,
        artifact_cache=PreparedArtifactCache(leases),
        draw_repository=DrawRepository(leases),
    )


def dispatch(engine, product, env, ctx, operation=PricingOperation.PRICE_DETAILED):
    from quantark.execution.kernel import ExecutionKernel

    request = PricingRequest(
        product=product, pricing_env=env, operation=operation
    )
    return ExecutionKernel.dispatch(engine, request, ctx)


def test_kernel_batch_path_serial_and_threads_bitwise_equal():
    product, env = make_dcn_product_env()
    direct = make_engine(
        num_paths=2048, num_batches=8
    ).price_detailed(product, env)

    serial_out = dispatch(
        make_engine(num_paths=2048, num_batches=8), product, env,
        make_batch_context(workers=1, backend="serial"),
    )
    threads_out = dispatch(
        make_engine(num_paths=2048, num_batches=8), product, env,
        make_batch_context(workers=4, backend="threads"),
    )
    assert_results_bitwise_equal(serial_out.value, direct)
    assert_results_bitwise_equal(threads_out.value, direct)
    assert serial_out.manifest.plan_fingerprint is not None
    assert (serial_out.manifest.plan_fingerprint
            == threads_out.manifest.plan_fingerprint)


def test_kernel_price_operation_projects_pv():
    product, env = make_dcn_product_env()
    engine = make_engine(num_paths=1024, num_batches=4)
    direct = make_engine(num_paths=1024, num_batches=4).price(product, env)
    outcome = dispatch(
        engine, product, env, make_batch_context(),
        operation=PricingOperation.PRICE,
    )
    assert outcome.value == direct


def test_kernel_records_worker_clamp():
    product, env = make_dcn_product_env()
    engine = make_engine(num_paths=1024, num_batches=4)
    ctx = make_batch_context(workers=64, backend="threads")
    outcome = dispatch(engine, product, env, ctx)
    assert any("clamp:batch.workers=64->" in r
               for r in outcome.diagnostics.records)


def test_kernel_max_in_flight_serializes_batches_and_stays_exact():
    # Codex plan-gate finding: admission control must bind on batch tasks.
    product, env = make_dcn_product_env()
    direct = make_engine(num_paths=1024, num_batches=4).price_detailed(
        product, env
    )
    ctx = make_batch_context(workers=4, backend="threads", max_in_flight=1)
    outcome = dispatch(
        make_engine(num_paths=1024, num_batches=4), product, env, ctx
    )
    assert_results_bitwise_equal(outcome.value, direct)
    assert any("clamp:batch.window=" in r for r in outcome.diagnostics.records)


def test_draws_pinned_bytes_released_and_reused_across_dispatches():
    product, env = make_dcn_product_env()
    ctx = make_batch_context(workers=2, backend="threads")
    engine = make_engine(num_paths=1024, num_batches=4)
    dispatch(engine, product, env, ctx)
    stats = ctx.draw_repository.stats()
    assert stats["misses"] == 4          # one Sobol block per batch
    assert stats["bytes_in_use"] > 0     # masters retained for CRN reuse
    dispatch(engine, product, env, ctx)  # identical repricing: all hits
    stats = ctx.draw_repository.stats()
    assert stats["misses"] == 4
    assert stats["hits"] == 4
