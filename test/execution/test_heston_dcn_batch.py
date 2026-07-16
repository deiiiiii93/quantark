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
