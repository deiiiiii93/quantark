"""Session-vs-direct parity and the BaseEngine.execute seam (spec sections 5.4, 5.5)."""
import pytest

from quantark.asset.equity.engine.mc import EuropeanMCEngine
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.execution import (
    OutputKind,
    PricingRequest,
    PricingSession,
)
from quantark.execution.contracts import economics_mapping
from quantark.execution.errors import CapabilityError
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType


@pytest.fixture()
def equity_env():
    from datetime import datetime

    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.05),
        valuation_date=datetime(2024, 1, 1),
    )


@pytest.fixture()
def european_option():
    return EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )


def _mc_engine():
    return EuropeanMCEngine(params=MCParams(num_paths=2000, seed=42))


def test_session_price_equals_direct_price(equity_env, european_option):
    direct = _mc_engine().price(european_option, equity_env)
    with PricingSession() as session:
        via_session = session.price(_mc_engine(), european_option, equity_env)
    assert via_session == direct  # exact: same numerical plan, same code path


def test_execute_returns_outcome_with_manifest(equity_env, european_option):
    with PricingSession() as session:
        outcome = session.execute(
            _mc_engine(),
            PricingRequest(product=european_option, pricing_env=equity_env),
        )
    assert isinstance(outcome.value, float)
    assert economics_mapping(outcome)["pv"] == outcome.value
    assert outcome.manifest.adapter_id == "legacy-price"
    assert outcome.manifest.request_fingerprint is None
    assert dict(outcome.manifest.versions)["numpy"]
    assert outcome.diagnostics.adapter_id == "legacy-price"


def test_base_engine_execute_method(equity_env, european_option):
    from quantark.execution.context import default_context

    engine = _mc_engine()
    outcome = engine.execute(
        PricingRequest(product=european_option, pricing_env=equity_env),
        default_context(),
    )
    assert outcome.value == engine.price(european_option, equity_env)


def test_price_many_preserves_order_and_types(equity_env, european_option):
    put = EuropeanVanillaOption(
        strike=110.0, option_type=OptionType.PUT, maturity=1.0
    )
    items = [
        (_mc_engine(), PricingRequest(product=european_option, pricing_env=equity_env)),
        (_mc_engine(), PricingRequest(product=put, pricing_env=equity_env)),
    ]
    with PricingSession() as session:
        values = session.price_many(items)
    assert len(values) == 2 and all(isinstance(v, float) for v in values)
    assert values[0] == _mc_engine().price(european_option, equity_env)


def test_price_many_collect_errors(equity_env, european_option):
    from quantark.execution import PricingFailure

    class _Boom:
        def price(self, product, env):
            raise ValueError("boom")

    # _Boom is not a registered engine family -> CapabilityError, collected.
    items = [
        (_mc_engine(), PricingRequest(product=european_option, pricing_env=equity_env)),
        (_Boom(), PricingRequest(product=european_option, pricing_env=equity_env)),
    ]
    with PricingSession() as session:
        results = session.price_many(items, collect_errors=True)
    assert isinstance(results[0], float)
    assert isinstance(results[1], PricingFailure)
    assert results[1].error.error_type == "CapabilityError"


def test_non_serial_backend_raises_capability_error(equity_env, european_option):
    import dataclasses

    from quantark.execution import ExecutionPolicy
    from quantark.execution.context import default_context
    from quantark.execution.policy import ExecutorSelection

    ctx = dataclasses.replace(
        default_context(),
        execution_policy=ExecutionPolicy(
            batch=ExecutorSelection(backend="threads", workers=4)
        ),
    )
    with PricingSession(ctx) as session:
        with pytest.raises(CapabilityError):
            session.price(_mc_engine(), european_option, equity_env)


def test_run_scenarios_accepts_empty_specs(equity_env, european_option):
    # Phase 5 replaced the Phase 0 CapabilityError stub with the scenario
    # planner; an empty spec collection is a valid empty plan.
    with PricingSession() as session:
        outcomes = session.run_scenarios(
            PricingRequest(product=european_option, pricing_env=equity_env),
            scenario_specs=(),
            engine_factory=_mc_engine,
        )
    assert outcomes == []


def test_unsupported_output_raises(equity_env, european_option):
    req = PricingRequest(
        product=european_option,
        pricing_env=equity_env,
        outputs=frozenset({OutputKind.PV, OutputKind.GRID}),
    )
    with PricingSession() as session:
        with pytest.raises(CapabilityError):
            session.execute(_mc_engine(), req)


def test_requested_error_estimate_rejected_not_silently_omitted(
    equity_env, european_option
):
    """The legacy adapter guarantees only PV; a request for ERROR_ESTIMATE
    must fail loudly rather than return a success without the output
    (Codex code-gate finding 1)."""
    from quantark.execution import PricingOperation

    req = PricingRequest(
        product=european_option,
        pricing_env=equity_env,
        operation=PricingOperation.PRICE_DETAILED,
        outputs=frozenset({OutputKind.PV, OutputKind.ERROR_ESTIMATE}),
    )
    with PricingSession() as session:
        with pytest.raises(CapabilityError):
            session.execute(_mc_engine(), req)


def test_manifest_records_resolved_policy_values(equity_env, european_option):
    """Manifests carry effective configuration values, not source labels
    (Codex code-gate finding 2)."""
    with PricingSession() as session:
        outcome = session.execute(
            _mc_engine(),
            PricingRequest(product=european_option, pricing_env=equity_env),
        )
    resolved = dict(outcome.manifest.resolved_policy)
    assert resolved["batch.backend"] == "serial"
    assert resolved["batch.workers"] == "1"
    # Phase 2: the session-OWNED auto budget upgrades thread/in-flight
    # capacity to the machine (spec section 11.1); values are recorded.
    import os

    assert resolved["budget.max_threads"] == str(os.cpu_count() or 1)
    assert resolved["budget.max_in_flight"] == str(os.cpu_count() or 1)
    assert resolved["determinism.require_manifest"] == "True"
    # Source labels live in diagnostics, not the manifest.
    assert dict(outcome.diagnostics.policy_sources)["batch.backend"] in (
        "default", "env", "explicit", "env_invalid_default",
    )


def test_injected_registry_is_frozen_by_session(equity_env, european_option):
    """A caller-supplied registry cannot be mutated after session
    construction (Codex code-gate finding 3; spec section 6.2)."""
    import dataclasses

    from quantark.execution.context import default_context
    from quantark.execution.registry import build_default_registry
    from quantark.util.exceptions import ValidationError

    registry = build_default_registry()  # deliberately not frozen
    ctx = dataclasses.replace(default_context(), adapter_registry=registry)
    with PricingSession(ctx) as session:
        with pytest.raises(ValidationError):
            registry.register("x.Y", lambda: None)
        assert session.price(_mc_engine(), european_option, equity_env)


class TestCrossFamilyParity:
    """Phase 0 exit evidence: session == direct for one engine per family."""

    @pytest.fixture(scope="class")
    def cases(self):
        from execution.freeze_goldens import build_representative_cases

        return build_representative_cases()

    @pytest.mark.parametrize(
        "case_name",
        [
            "equity_mc_european",
            "equity_pde_european",
            "fx_mc_barrier",
            "credit_mc_basket_cds",
            "bond_pde_convertible_tf",
        ],
    )
    def test_session_price_equals_direct(self, cases, case_name):
        engine, product, env, call_shape = cases[case_name]
        if call_shape == "env_bound":
            direct = engine.price(product)
        else:
            direct = engine.price(product, env)
        with PricingSession() as session:
            via_session = session.price(engine, product, env)
        assert via_session == direct

    def test_legacy_internal_parallelism_preserved(self, cases):
        """Spec sections 12.4/17.1: engine-owned parallel settings (Snowball
        use_dask, DCN workers) behave identically through the session,
        including the missing-Dask UserWarning fallback."""
        import warnings

        from quantark.asset.equity.engine.mc import SnowballMCEngine
        from quantark.asset.equity.product.option import SnowballOption
        from quantark.asset.equity.product.option.snowball_config import (
            BarrierConfig,
        )

        _, _, eq_env, _ = cases["equity_mc_european"]
        snowball = SnowballOption(
            initial_price=100.0, strike=100.0,
            barrier_config=BarrierConfig(
                ko_barrier=103.0, ko_rate=0.15,
                ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
                ki_barrier=75.0, ki_continuous=True,
            ),
            contract_multiplier=10_000.0, maturity=1.0,
        )

        def _build():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # missing-Dask UserWarning ok
                return SnowballMCEngine(
                    params=MCParams(num_paths=4000, seed=11),
                    use_dask=True, num_batches=4,
                )

        direct = _build().price(snowball, eq_env)
        with PricingSession() as session:
            via_session = session.price(_build(), snowball, eq_env)
        assert via_session == direct

    def test_goldens_match_current_serial_results(self, cases):
        import json
        import pathlib

        golden_path = (
            pathlib.Path(__file__).parent / "goldens" / "phase0_goldens.json"
        )
        goldens = json.loads(golden_path.read_text())["values"]
        for name, (engine, product, env, call_shape) in cases.items():
            if call_shape == "env_bound":
                value = float(engine.price(product))
            else:
                value = float(engine.price(product, env))
            assert value == pytest.approx(goldens[name], abs=1e-10), name
