"""Heston DCN fixed-batch adapters (Phase 3): bit-identity + draw routing."""
import numpy as np
import pytest

from dcn_fixtures import DCN_A, FLAT, flat_env, make_dcn
from quantark.execution.contracts import PricingOperation, PricingRequest
from quantark.util.enum.engine_enums import HestonMCScheme


def _heston_params():
    from quantark.volmodels.heston import HestonParams

    return HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma=0.5, rho=-0.5)


def _dcn_product_env():
    return make_dcn(DCN_A), flat_env(**FLAT)


def make_heston_engine(**kwargs):
    from quantark.asset.equity.engine.mc.dcn_vol_mc_engines import (
        HestonDCNMCEngine,
    )

    kwargs.setdefault("model_params", _heston_params())
    kwargs.setdefault("seed", 7)
    return HestonDCNMCEngine(**kwargs)


def test_uniform_provider_hook_intercepts_three_stream_draws():
    from quantark.asset.equity.engine.mc.qmc_draws import qmc_uniforms

    engine = make_heston_engine(
        scheme=HestonMCScheme.QUADEXP, num_paths=128, use_sobol=True,
        num_batches=1,
    )
    calls = []

    def provider(n_dims, n_paths, batch_id):
        calls.append((n_dims, n_paths, batch_id))
        return qmc_uniforms(engine.seed, n_paths, n_dims,
                            batch_id=batch_id, writable=True)

    baseline = engine._heston_draws(4, 128, None)
    engine._uniform_provider = provider
    hooked = engine._heston_draws(4, 128, None)
    assert calls == [(12, 128, None)]
    for a, b in zip(baseline, hooked):
        assert np.array_equal(a, b)


def test_uniform_provider_ignored_for_two_stream_scheme():
    engine = make_heston_engine(
        scheme=HestonMCScheme.FULL_TRUNCATION_EULER, num_paths=64,
        use_sobol=True, num_batches=1,
    )
    engine._uniform_provider = lambda *a: pytest.fail("must not be called")
    z_var, z_ind, u_var = engine._heston_draws(4, 64, None)
    assert u_var is None
    assert z_var.shape == (64, 4) and z_ind.shape == (64, 4)


def test_gbm_and_lv_plans_unchanged_by_hook_refactor():
    """The mixin plan hooks must not alter the Phase-2 GBM/LV plan fields."""
    from execution.test_dcn_batch_adapter import make_engine
    from quantark.asset.equity.engine.mc.dcn_execution_adapters import (
        DCNBatchMCAdapter,
    )
    from quantark.execution.context import default_context

    product, env = _dcn_product_env()
    engine = make_engine(num_paths=4096, num_batches=8, use_sobol=True)
    adapter = DCNBatchMCAdapter()
    request = PricingRequest(product=product, pricing_env=env)
    context = default_context()
    state = adapter.prepare(engine, request, context)
    try:
        plan = adapter.plan_batches(engine, request, state, context)
    finally:
        for handle in state.handles:
            handle.close()
    sim = state.payload
    time_steps = int(sim.dt_array.size)
    assert plan.scheme == "gbm-term/v1"
    assert plan.dimension == time_steps
    assert plan.stream_kind == "sobol"
    assert plan.stream_layout == "batch-shifted-sobol/v1"
    assert plan.est_task_peak_bytes == (
        8 * 512 * (2 * time_steps + 1 + 6 * sim.n_obs) + (1 << 20)
    )
    assert plan.est_outcome_bytes == 8 * (6 * sim.n_obs + 16)


# ---------------------------------------------------------------------------
# Task 10: Heston + QE batch adapters (exact registrations)
# ---------------------------------------------------------------------------

def _assert_results_bitwise_equal(actual, expected):
    for field in type(expected).__dataclass_fields__:
        if field in ("elapsed_seconds", "event_stats"):
            continue
        assert getattr(actual, field) == getattr(expected, field), field


def make_qe_engine(**kwargs):
    from quantark.asset.equity.engine.mc.dcn_vol_mc_engines import QEDCNMCEngine

    kwargs.setdefault("model_params", _heston_params())
    kwargs.setdefault("seed", 7)
    return QEDCNMCEngine(**kwargs)


_MODES = {
    "sobol_multi": dict(num_paths=1024, num_batches=4, use_sobol=True),
    "sobol_single": dict(num_paths=1024, num_batches=1, use_sobol=True),
    "pseudo_iid": dict(num_paths=1024, num_batches=4, use_sobol=False),
}


@pytest.mark.parametrize("scheme", [
    "FULL_TRUNCATION_EULER", "EULERLOG", "QUADEXP", "QUADEXP_M",
])
@pytest.mark.parametrize("mode", sorted(_MODES))
def test_heston_session_bitwise_vs_direct(scheme, mode):
    from execution.test_dcn_batch_adapter import dispatch, make_batch_context

    kwargs = dict(_MODES[mode], scheme=HestonMCScheme[scheme])
    product, env = _dcn_product_env()
    direct = make_heston_engine(**kwargs).price_detailed(product, env)
    outcome = dispatch(
        make_heston_engine(**kwargs), product, env, make_batch_context(),
    )
    _assert_results_bitwise_equal(outcome.value, direct)
    assert outcome.manifest.adapter_id == "dcn-batch-mc"
    assert outcome.manifest.plan_fingerprint is not None


def test_qe_engine_bitwise_and_uniform_draws_from_repo():
    from execution.test_dcn_batch_adapter import dispatch, make_batch_context

    product, env = _dcn_product_env()
    kwargs = dict(num_paths=1024, num_batches=4, use_sobol=True,
                  martingale_correction=True)
    direct = make_qe_engine(**kwargs).price_detailed(product, env)
    ctx = make_batch_context()
    outcome = dispatch(make_qe_engine(**kwargs), product, env, ctx)
    _assert_results_bitwise_equal(outcome.value, direct)
    stats = ctx.draw_repository.stats()
    # QE draws one uniform block per batch (transformed in place); no
    # separate normal block is fetched on this path.
    assert stats["misses"] == 4
    dispatch(make_qe_engine(**kwargs), product, env, ctx)  # CRN repricing
    stats = ctx.draw_repository.stats()
    assert stats["misses"] == 4
    assert stats["hits"] == 4


def test_heston_two_stream_normals_from_repo():
    from execution.test_dcn_batch_adapter import dispatch, make_batch_context

    product, env = _dcn_product_env()
    kwargs = dict(num_paths=1024, num_batches=4, use_sobol=True,
                  scheme=HestonMCScheme.FULL_TRUNCATION_EULER)
    ctx = make_batch_context()
    dispatch(make_heston_engine(**kwargs), product, env, ctx)
    assert ctx.draw_repository.stats()["misses"] == 4
    dispatch(make_heston_engine(**kwargs), product, env, ctx)
    assert ctx.draw_repository.stats()["hits"] == 4


def test_heston_threads_backend_bitwise():
    from execution.test_dcn_batch_adapter import dispatch, make_batch_context

    product, env = _dcn_product_env()
    kwargs = dict(num_paths=1024, num_batches=4, use_sobol=True,
                  scheme=HestonMCScheme.QUADEXP)
    direct = make_heston_engine(**kwargs).price_detailed(product, env)
    outcome = dispatch(
        make_heston_engine(**kwargs), product, env,
        make_batch_context(workers=4, backend="threads"),
    )
    _assert_results_bitwise_equal(outcome.value, direct)


def test_unknown_heston_subclass_falls_to_legacy():
    from quantark.asset.equity.engine.mc.dcn_vol_mc_engines import (
        HestonDCNMCEngine,
    )
    from quantark.execution.registry import build_default_registry

    class _Tweaked(HestonDCNMCEngine):
        pass

    registry = build_default_registry()
    registry.freeze()
    adapter = registry.resolve_class(_Tweaked)
    assert not hasattr(adapter, "plan_batches")
    assert adapter.capabilities().adapter_id == "legacy-price"


def test_model_params_mutation_fails_closed():
    """HestonParams is FROZEN (in-place mutation is structurally
    impossible), and the adapter additionally fingerprints the captured
    model as defense-in-depth: a changed fingerprint fails closed."""
    import dataclasses

    from quantark.asset.equity.engine.mc.dcn_execution_adapters import (
        DCNHestonBatchMCAdapter,
    )
    from quantark.execution.context import default_context
    from quantark.execution.errors import DeterminismViolation

    hp = _heston_params()
    with pytest.raises(dataclasses.FrozenInstanceError):
        hp.v0 = 0.09

    product, env = _dcn_product_env()
    engine = make_heston_engine(num_paths=512, num_batches=2, use_sobol=True)
    adapter = DCNHestonBatchMCAdapter()
    request = PricingRequest(
        product=product, pricing_env=env,
        operation=PricingOperation.PRICE_DETAILED,
    )
    context = default_context()
    state = adapter.prepare(engine, request, context)
    try:
        sim = state.payload
        assert sim.model_fp is not None
        tampered = dataclasses.replace(sim, model_fp="tampered")
        with pytest.raises(DeterminismViolation):
            adapter._verify_captured_inputs(tampered)
        adapter._verify_captured_inputs(sim)  # untampered passes
    finally:
        for handle in state.handles:
            handle.close()


def test_heston_plan_metadata():
    from quantark.asset.equity.engine.mc.dcn_execution_adapters import (
        DCNHestonBatchMCAdapter,
    )
    from quantark.execution.context import default_context

    product, env = _dcn_product_env()
    engine = make_heston_engine(
        num_paths=1024, num_batches=4, use_sobol=True,
        scheme=HestonMCScheme.QUADEXP, substeps_per_interval=2,
    )
    adapter = DCNHestonBatchMCAdapter()
    request = PricingRequest(product=product, pricing_env=env)
    context = default_context()
    state = adapter.prepare(engine, request, context)
    try:
        plan = adapter.plan_batches(engine, request, state, context)
    finally:
        for handle in state.handles:
            handle.close()
    time_steps = int(state.payload.dt_array.size)
    assert plan.scheme == "heston-quadexp-sub2/v1"
    assert plan.dimension == 3 * time_steps * 2
    assert plan.stream_layout == "batch-shifted-sobol-3stream/v1"


# ---------------------------------------------------------------------------
# Task 11: CoupledCoarse Heston DCN batch adapter (pair-aware clone)
# ---------------------------------------------------------------------------

def _coupled_pair(scheme, **kwargs):
    from quantark.asset.equity.engine.mc.dcn_vol_mc_engines import (
        coupled_heston_ladder_pair,
    )

    kwargs.setdefault("num_paths", 512)
    kwargs.setdefault("seed", 7)
    kwargs.setdefault("use_sobol", True)
    kwargs.setdefault("num_batches", 2)
    return coupled_heston_ladder_pair(
        _heston_params(), 1, HestonMCScheme[scheme], **kwargs
    )


@pytest.mark.parametrize("scheme", ["FULL_TRUNCATION_EULER", "QUADEXP_M"])
def test_coupled_coarse_session_bitwise_vs_direct(scheme):
    from execution.test_dcn_batch_adapter import dispatch, make_batch_context

    product, env = _dcn_product_env()
    coarse, _ = _coupled_pair(scheme)
    direct = coarse.price_detailed(product, env)

    coarse2, _ = _coupled_pair(scheme)
    outcome = dispatch(coarse2, product, env, make_batch_context())
    _assert_results_bitwise_equal(outcome.value, direct)
    assert outcome.manifest.adapter_id == "dcn-batch-mc"


def test_coupled_pair_ladder_difference_preserved():
    from execution.test_dcn_batch_adapter import dispatch, make_batch_context

    product, env = _dcn_product_env()
    coarse, fine = _coupled_pair("QUADEXP_M")
    direct_diff = (
        coarse.price_detailed(product, env).pv
        - fine.price_detailed(product, env).pv
    )

    coarse2, fine2 = _coupled_pair("QUADEXP_M")
    ctx = make_batch_context()
    session_coarse = dispatch(coarse2, product, env, ctx).value.pv
    session_fine = dispatch(fine2, product, env, ctx).value.pv
    assert (session_coarse - session_fine) == direct_diff


def test_coupled_coarse_resolves_to_batch_adapter():
    from quantark.asset.equity.engine.mc import CoupledCoarseHestonDCNMCEngine
    from quantark.execution.registry import build_default_registry

    registry = build_default_registry()
    registry.freeze()
    adapter = registry.resolve_class(CoupledCoarseHestonDCNMCEngine)
    assert hasattr(adapter, "plan_batches")
    assert adapter.capabilities().adapter_id == "dcn-batch-mc"
