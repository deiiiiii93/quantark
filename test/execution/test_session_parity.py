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


def test_run_scenarios_is_phase5(equity_env, european_option):
    with PricingSession() as session:
        with pytest.raises(CapabilityError):
            session.run_scenarios(
                PricingRequest(product=european_option, pricing_env=equity_env),
                scenario_specs=(),
                engine_factory=_mc_engine,
            )


def test_unsupported_output_raises(equity_env, european_option):
    req = PricingRequest(
        product=european_option,
        pricing_env=equity_env,
        outputs=frozenset({OutputKind.PV, OutputKind.GRID}),
    )
    with PricingSession() as session:
        with pytest.raises(CapabilityError):
            session.execute(_mc_engine(), req)
