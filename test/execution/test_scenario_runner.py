"""Serial scenario execution through PricingSession.run_scenarios
(spec sections 13.3, 13.4, 15)."""
import dataclasses
from datetime import datetime

import pytest

from quantark.execution.api import PricingSession
from quantark.execution.contracts import (
    OutputKind,
    PricingFailure,
    PricingRequest,
    ScenarioOutcome,
    ScenarioSpec,
    economics_mapping,
)
from quantark.execution.errors import TaskExecutionError
from quantark.execution.scenario import registries


def _flat_env(spot):
    from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
    from quantark.param.div import ContinuousDividendYield
    from quantark.priceenv import PricingEnvironment

    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.02),
        valuation_date=datetime(2024, 1, 1),
    )


def _euro():
    from quantark.asset.equity.product.option import EuropeanVanillaOption
    from quantark.util.enum import OptionType

    return EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )


def euro_spot_bump(base_request, parameters):
    """Rebuild the request around a freshly-built bumped environment."""
    spot = base_request.pricing_env.spot_quote.spot + parameters["ds"]
    request = dataclasses.replace(base_request, pricing_env=_flat_env(spot))
    if parameters.get("poison"):
        request = dataclasses.replace(
            request, outputs=frozenset({OutputKind.ERROR_ESTIMATE})
        )
    return request


registries.register_transformer(
    "euro-spot-bump/v1", euro_spot_bump,
    allowed_tags=frozenset({"spot"}),
    components=(("spot", lambda r: r.pricing_env.spot_quote.spot),),
)


def _engine_factory(parameters):
    from quantark.asset.equity.engine.analytical import BlackScholesEngine

    return BlackScholesEngine()


def _spec(scenario_id, ds, **extra):
    params = {"ds": ds}
    params.update(extra)
    return ScenarioSpec(
        scenario_id=scenario_id,
        transformer_id="euro-spot-bump/v1",
        parameters=tuple(sorted(params.items())),
        mutation_tags=frozenset({"spot"}),
    )


def _base_request():
    return PricingRequest(product=_euro(), pricing_env=_flat_env(100.0))


def test_ordered_outcomes_match_direct_calls():
    from quantark.asset.equity.engine.analytical import BlackScholesEngine

    specs = [_spec("down", -5.0), _spec("flat", 0.0), _spec("up", 5.0)]
    with PricingSession() as session:
        outcomes = session.run_scenarios(_base_request(), specs, _engine_factory)
    assert [o.scenario_id for o in outcomes] == ["down", "flat", "up"]
    engine = BlackScholesEngine()
    product = _euro()
    for outcome, ds in zip(outcomes, (-5.0, 0.0, 5.0)):
        direct = engine.price(product, _flat_env(100.0 + ds))
        assert isinstance(outcome, ScenarioOutcome)
        assert economics_mapping(outcome)["pv"] == float(direct)
        assert outcome.value == float(direct)
        assert outcome.manifest_fingerprint is not None


def test_fail_fast_raises_native_error():
    from quantark.execution.errors import CapabilityError

    specs = [_spec("ok", 1.0), _spec("bad", 2.0, poison=True)]
    with PricingSession() as session:
        with pytest.raises(CapabilityError):
            session.run_scenarios(_base_request(), specs, _engine_factory)


def test_collect_errors_isolates_the_failing_cell():
    specs = [_spec("ok", 1.0), _spec("bad", 2.0, poison=True), _spec("ok2", 3.0)]
    with PricingSession() as session:
        outcomes = session.run_scenarios(
            _base_request(), specs, _engine_factory, collect_errors=True
        )
    assert isinstance(outcomes[0], ScenarioOutcome)
    assert isinstance(outcomes[1], PricingFailure)
    assert outcomes[1].item_id == "bad"
    assert outcomes[1].error.error_type == "CapabilityError"
    assert isinstance(outcomes[2], ScenarioOutcome)


def test_identical_cells_dedupe_but_keep_identity():
    specs = [_spec("a", 1.0), _spec("b", 1.0)]
    with PricingSession() as session:
        outcomes = session.run_scenarios(_base_request(), specs, _engine_factory)
        sink = session.context.diagnostics_sink
    assert outcomes[0].scenario_id == "a"
    assert outcomes[1].scenario_id == "b"
    assert outcomes[0].value == outcomes[1].value
    assert economics_mapping(outcomes[0]) == economics_mapping(outcomes[1])
    records = [
        r for d in sink.entries for r in getattr(d, "records", ())
        if r.startswith("scenario:")
    ]
    assert "scenario:deduped=1" in records


class _CancelAfterFirst:
    def __init__(self):
        self.calls = 0

    def cancelled(self):
        self.calls += 1
        return self.calls > 1


def test_cancellation_between_cells():
    import quantark.execution.context as context_mod

    token = _CancelAfterFirst()
    base_context = context_mod.default_context()
    ctx = dataclasses.replace(base_context, cancellation_token=token)
    specs = [_spec("a", 1.0), _spec("b", 2.0), _spec("c", 3.0)]
    with PricingSession(ctx) as session:
        with pytest.raises(TaskExecutionError):
            session.run_scenarios(_base_request(), specs, _engine_factory)


def test_run_scenarios_no_longer_raises_phase0_stub():
    with PricingSession() as session:
        outcomes = session.run_scenarios(
            _base_request(), [_spec("a", 1.0)], _engine_factory
        )
    assert len(outcomes) == 1
