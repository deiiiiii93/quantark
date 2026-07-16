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


def _ko_reset_product():
    from quantark.asset.equity.product.option.ko_reset_snowball_option import (
        KnockOutResetSnowballOption,
        PostKOScheduleMode,
    )
    from quantark.asset.equity.product.option.snowball_config import BarrierConfig
    from quantark.util.enum import ObservationType

    pre = BarrierConfig(
        ko_barrier=105.0, ko_rate=0.15,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
        ki_barrier=85.0, ki_observation_type=ObservationType.CONTINUOUS,
        ki_continuous=True,
    )
    post = BarrierConfig(
        ko_barrier=98.0, ko_rate=0.15,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
    )
    return KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0,
        barrier_config=pre, post_barrier_config=post,
        contract_multiplier=1.0, maturity=1.0, is_reverse=False,
        post_ko_mode=PostKOScheduleMode.ABSOLUTE,
    )


class TestKernelAdaptiveDispatch:
    def _assert_session_bitwise(self, make_engine, product):
        from quantark.execution import PricingSession

        env = _eq_flat_env()
        direct = make_engine()
        expected = direct.price(product, env)
        expected_result = direct.get_last_result()

        from quantark.execution.contracts import PricingRequest
        with PricingSession() as session:
            outcome = session.execute(
                make_engine(),
                PricingRequest(product=product, pricing_env=env),
            )
        assert outcome.value == expected
        econ = dict(outcome.normalized_economics)
        assert econ["pv"] == expected
        assert econ["std_error"] == float(expected_result.std_error)
        assert outcome.manifest.adapter_id == "autocallable-adaptive-mc"
        assert outcome.manifest.plan_fingerprint is not None
        records = outcome.diagnostics.records
        assert (
            f"adaptive:batches_used={expected_result.batches_used}" in records
        )
        assert any(
            r.startswith("adaptive:trace_fingerprint=") for r in records
        )
        return outcome

    def test_session_price_bitwise_vs_direct_snowball(self):
        self._assert_session_bitwise(_rqmc_snowball_engine, _snowball())

    def test_session_price_bitwise_vs_direct_ko_reset(self):
        self._assert_session_bitwise(_rqmc_snowball_engine, _ko_reset_product())

    def test_session_price_bitwise_vs_direct_phoenix(self):
        self._assert_session_bitwise(_rqmc_phoenix_engine, _phoenix())

    def test_non_rqmc_method_falls_to_native(self):
        from quantark.asset.equity.engine.mc import SnowballMCEngine
        from quantark.execution import PricingSession
        from quantark.execution.contracts import PricingRequest
        from quantark.util.enum.engine_enums import MonteCarloMethod

        product, env = _snowball(), _eq_flat_env()

        def make_engine():
            return SnowballMCEngine(
                params=_rqmc_params(), method=MonteCarloMethod.QUASI,
            )

        expected = make_engine().price(product, env)
        with PricingSession() as session:
            outcome = session.execute(
                make_engine(),
                PricingRequest(product=product, pricing_env=env),
            )
        assert outcome.value == expected
        assert outcome.manifest.adapter_id == "autocallable-adaptive-mc"
        assert outcome.manifest.plan_fingerprint is None
        assert not any(
            r.startswith("adaptive:") for r in outcome.diagnostics.records
        )

    def test_event_stats_operation_unchanged(self):
        from quantark.execution import PricingSession
        from quantark.execution.contracts import PricingOperation, PricingRequest

        product, env = _snowball(), _eq_flat_env()
        direct = _rqmc_snowball_engine().calculate_event_stats(product, env)
        with PricingSession() as session:
            outcome = session.execute(
                _rqmc_snowball_engine(),
                PricingRequest(
                    product=product, pricing_env=env,
                    operation=PricingOperation.EVENT_STATS,
                ),
            )
        import numpy as np
        assert np.array_equal(
            np.asarray(outcome.value.ko_probability),
            np.asarray(direct.ko_probability),
        )
