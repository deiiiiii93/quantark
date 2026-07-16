"""Adaptive RQMC session path: seams, adapter, kernel dispatch (Phase 3).

Bit-identity bar (kickoff decision 2026-07-16): session PRICE must equal the
direct engine call bitwise - price, std_error, batches_used, event probs.
"""
import pytest

from execution.matrix_fixtures import _eq_flat_env, _phoenix, _snowball
from quantark.montecarlo.qmc_rqmc_driver import RQMCRunSpec, run_rqmc


def _rqmc_params(**overrides):
    from quantark.asset.equity.param import MCParams

    kw = dict(
        seed=42, num_paths=2048, time_steps=64,
        rqmc_min_batches=2, rqmc_max_batches=6, rqmc_target_std=1e-9,
    )
    kw.update(overrides)
    return MCParams(**kw)


def _rqmc_snowball_engine(**overrides):
    from quantark.asset.equity.engine.mc import SnowballMCEngine
    from quantark.util.enum.engine_enums import MonteCarloMethod

    return SnowballMCEngine(
        params=_rqmc_params(**overrides),
        method=MonteCarloMethod.RANDOMIZED_QUASI,
    )


def _rqmc_phoenix_engine(**overrides):
    from quantark.asset.equity.engine.mc import PhoenixMCEngine
    from quantark.util.enum.engine_enums import MonteCarloMethod

    return PhoenixMCEngine(
        params=_rqmc_params(**overrides),
        method=MonteCarloMethod.RANDOMIZED_QUASI,
    )


class TestSnowballRQMCSpecSeam:
    def test_session_spec_none_for_non_rqmc_method(self):
        from quantark.asset.equity.engine.mc import SnowballMCEngine
        from quantark.util.enum.engine_enums import MonteCarloMethod

        engine = SnowballMCEngine(
            params=_rqmc_params(), method=MonteCarloMethod.QUASI,
        )
        assert engine.build_rqmc_session_spec(_snowball(), _eq_flat_env()) is None

    def test_session_spec_shape(self):
        engine = _rqmc_snowball_engine()
        spec = engine.build_rqmc_session_spec(_snowball(), _eq_flat_env())
        assert isinstance(spec, RQMCRunSpec)
        assert spec.max_batches == 6 and spec.min_batches == 2
        assert spec.paths_per_batch == spec.path_generator.num_paths
        assert spec.product is not None

    def test_spec_driven_run_equals_direct_price(self):
        product, env = _snowball(), _eq_flat_env()
        direct = _rqmc_snowball_engine()
        direct_price = direct.price(product, env)
        direct_result = direct.get_last_result()

        session_like = _rqmc_snowball_engine()
        spec = session_like.build_rqmc_session_spec(product, env)
        result = spec.finalize(run_rqmc(
            pricer_fn=spec.pricer_fn, path_generator=spec.path_generator,
            max_batches=spec.max_batches, target_std=spec.target_std,
            min_batches=spec.min_batches,
        ))
        price = session_like._complete_price(product, result)

        assert price == direct_price
        assert result.std_error == direct_result.std_error
        assert result.batches_used == direct_result.batches_used
        assert result.ko_probability == direct_result.ko_probability
        assert session_like.get_last_result() is result


class TestPhoenixRQMCSpecSeam:
    def test_session_spec_none_for_non_rqmc_method(self):
        from quantark.asset.equity.engine.mc import PhoenixMCEngine
        from quantark.util.enum.engine_enums import MonteCarloMethod

        engine = PhoenixMCEngine(
            params=_rqmc_params(), method=MonteCarloMethod.QUASI,
        )
        assert engine.build_rqmc_session_spec(_phoenix(), _eq_flat_env()) is None

    def test_session_spec_shape(self):
        engine = _rqmc_phoenix_engine()
        spec = engine.build_rqmc_session_spec(_phoenix(), _eq_flat_env())
        assert isinstance(spec, RQMCRunSpec)
        assert spec.max_batches == 6 and spec.min_batches == 2
        assert spec.paths_per_batch == spec.path_generator.num_paths
        assert spec.product is not None

    def test_spec_driven_run_equals_direct_price(self):
        product, env = _phoenix(), _eq_flat_env()
        direct = _rqmc_phoenix_engine()
        direct_price = direct.price(product, env)
        direct_result = direct.get_last_result()

        session_like = _rqmc_phoenix_engine()
        spec = session_like.build_rqmc_session_spec(product, env)
        result = spec.finalize(run_rqmc(
            pricer_fn=spec.pricer_fn, path_generator=spec.path_generator,
            max_batches=spec.max_batches, target_std=spec.target_std,
            min_batches=spec.min_batches,
        ))
        price = session_like._complete_price(product, result)

        assert price == direct_price
        assert result.std_error == direct_result.std_error
        assert result.batches_used == direct_result.batches_used
        assert result.ko_probability == direct_result.ko_probability
        assert session_like.get_last_result() is result


class TestAutocallableAdaptiveAdapter:
    def _adapter(self):
        from quantark.asset.equity.engine.mc.autocallable_execution_adapters import (
            AutocallableAdaptiveMCAdapter,
        )

        return AutocallableAdaptiveMCAdapter()

    def _context(self):
        from quantark.execution.context import default_context

        return default_context()

    def _close(self, prepared):
        for handle in prepared.handles:
            handle.close()

    def test_plan_shape(self):
        from quantark.execution.contracts import PricingRequest

        adapter, engine = self._adapter(), _rqmc_snowball_engine()
        request = PricingRequest(product=_snowball(), pricing_env=_eq_flat_env())
        context = self._context()
        prepared = adapter.prepare(engine, request, context)
        try:
            plan = adapter.plan_adaptive(engine, request, prepared, context)
            assert plan is not None
            assert plan.max_batches == 6 and plan.min_batches == 2
            assert plan.paths_per_batch == prepared.payload.paths_per_batch
            assert plan.stopping_rule == "welford-batch-means/v1"
            assert plan.engine_class_path.endswith("SnowballMCEngine")
        finally:
            self._close(prepared)

    def test_plan_none_for_non_rqmc(self):
        from quantark.asset.equity.engine.mc import SnowballMCEngine
        from quantark.execution.contracts import PricingRequest
        from quantark.util.enum.engine_enums import MonteCarloMethod

        adapter = self._adapter()
        engine = SnowballMCEngine(
            params=_rqmc_params(), method=MonteCarloMethod.QUASI,
        )
        request = PricingRequest(product=_snowball(), pricing_env=_eq_flat_env())
        context = self._context()
        prepared = adapter.prepare(engine, request, context)
        try:
            assert adapter.plan_adaptive(engine, request, prepared, context) is None
        finally:
            self._close(prepared)

    def test_execute_bitwise_vs_direct(self):
        from quantark.execution.contracts import PricingRequest

        product, env = _snowball(), _eq_flat_env()
        direct = _rqmc_snowball_engine()
        expected = direct.price(product, env)
        expected_result = direct.get_last_result()

        adapter, engine = self._adapter(), _rqmc_snowball_engine()
        request = PricingRequest(product=product, pricing_env=env)
        context = self._context()
        prepared = adapter.prepare(engine, request, context)
        try:
            plan = adapter.plan_adaptive(engine, request, prepared, context)
            value, economics, trace = adapter.execute_adaptive(
                engine, plan, prepared, context
            )
        finally:
            self._close(prepared)
        assert value == expected
        econ = dict(economics)
        assert econ["pv"] == expected
        assert econ["std_error"] == float(expected_result.std_error)
        assert len(trace) == expected_result.batches_used
        assert trace[-1].stopped

    def test_prepare_acquires_engine_lock_until_handles_closed(self):
        from quantark.asset.equity.engine.mc.autocallable_execution_adapters import (
            _engine_lock,
        )
        from quantark.execution.contracts import PricingRequest

        adapter, engine = self._adapter(), _rqmc_snowball_engine()
        request = PricingRequest(product=_snowball(), pricing_env=_eq_flat_env())
        prepared = adapter.prepare(engine, request, self._context())
        lock = _engine_lock(engine)
        assert lock.locked()
        self._close(prepared)
        assert not lock.locked()
