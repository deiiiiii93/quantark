"""Core bitwise gate: greeks assembled from bump cells == the originals.

Any `!=` here is a parity bug (spec D2): diagnose with float.hex(), never
loosen the assertion.
"""
from datetime import datetime

import pytest

from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.cashleg.autocallable_leg import (
    AccrualBasis,
    AutocallableCashLeg,
    AutocallableLegType,
)
from quantark.cashleg.base import LegDirection
from quantark.cashleg.event_distribution import EventType
from quantark.execution.greeks import (
    TradeState,
    assemble_product_greeks,
    assemble_trade_greeks,
    greek_bump_cells,
    greek_bump_transform,
    run_greek_bump,
)
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.portfolio.equity.position import EquityPosition
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType, OptionType

GREEKS7 = ["delta", "gamma", "vega", "theta", "rho", "dividend_rho"]


def _env():
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.03),
        div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=datetime(2024, 1, 1),
    )


def _snowball():
    cfg = BarrierConfig(
        ko_barrier=103.0,
        ko_rate=0.15,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[i / 12 for i in range(1, 13)],
        ki_barrier=75.0,
        ki_observation_type=ObservationType.CONTINUOUS,
        ki_continuous=True,
    )
    return SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=cfg,
        contract_multiplier=10_000.0,
        maturity=1.0,
    )


def _terminal_leg():
    return AutocallableCashLeg(
        direction=LegDirection.BUYER_RECEIVES,
        leg_type=AutocallableLegType.REBATE,
        notional=1_000.0,
        rate=0.05,
        terminal_accrual_factor=1.0,
        terminal_settlement_time=1.0,
        accrual_basis=AccrualBasis.KO_MATURITY,
        terminal_events=frozenset({EventType.MATURITY_NO_KO}),
    )


def _cell_values(state, gc, greeks):
    values = {}
    for cell in greek_bump_cells(greeks):
        bumped = greek_bump_transform(state, {"bump_id": cell.bump_id})
        values[cell.bump_id] = run_greek_bump(cell.bump_id, state, bumped, gc)
    return values


@pytest.mark.parametrize("quantity", [1.0, -1.0])
def test_trade_greeks_bitwise_vs_get_trade_risk(quantity):
    product = _snowball()
    engine = SnowballPDESolver(PDEParams())
    env = _env()
    gc = GreeksCalculator()
    pos = EquityPosition(
        product=product,
        quantity=quantity,
        entry_price=100.0,
        underlying="TEST",
        engine=engine,
        entry_timestamp=datetime(2024, 1, 1),
        cash_legs=[_terminal_leg()],
    )
    expected = pos.get_trade_risk(env, gc, GREEKS7)

    state = TradeState(
        product=_snowball(),
        pricing_env=_env(),
        cash_legs=(_terminal_leg(),),
        engine=SnowballPDESolver(PDEParams()),
        streams=pos._required_streams(),
        quantity=quantity,
        greeks_params=gc.params,
    )
    values = _cell_values(state, gc, GREEKS7)
    got = assemble_trade_greeks(
        values["base"], values, quantity=quantity, spot=env.spot,
        bump_config=gc._bump_config, requested=set(GREEKS7),
    )
    for key in ["price"] + GREEKS7:
        assert got["product"][key] == expected.product[key], (
            key, got["product"][key].hex(), expected.product[key].hex()
        )
        assert got["total"][key] == expected.total[key], (
            key, got["total"][key].hex(), expected.total[key].hex()
        )
    (leg_pv_expected,) = expected.leg_pvs.values()
    ((leg_name, _leg_dir, leg_pv),) = got["leg_pvs"]
    assert leg_name == str(leg_pv_expected.name)
    assert leg_pv == leg_pv_expected.pv


def test_product_greeks_bitwise_vs_calculate_numerical_greeks():
    product = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )
    engine = BlackScholesEngine()
    env = _env()
    gc = GreeksCalculator()
    raw = float(engine.price(product, env))
    expected = gc.calculate_numerical_greeks(
        product, env, engine, base_price=raw, greeks=["price"] + GREEKS7
    )

    state = TradeState(
        product=product, pricing_env=env, cash_legs=(), engine=engine,
        streams=None, quantity=1.0, greeks_params=gc.params,
    )
    values = _cell_values(state, gc, GREEKS7)
    got = assemble_product_greeks(
        base_price=raw, bump_values=values, spot=env.spot,
        bump_config=gc._bump_config, requested=set(["price"] + GREEKS7),
    )
    for key in ["price"] + GREEKS7:
        assert got[key] == expected[key], (
            key, got[key].hex(), expected[key].hex()
        )


def test_transformer_and_runner_registered():
    import quantark.execution.greeks  # noqa: F401 - canonical registration
    from quantark.execution.scenario import registries

    reg = registries.get_transformer("greek-bump/v1")
    assert reg.allowed_tags == frozenset({"spot", "vol", "rate", "div", "time"})
    runner = registries.get_runner("greek-value/v1")
    assert runner.value_kind == "float"


# ------------------------------------ code-gate regressions (2026-07-20)
def test_linear_product_bitwise_vs_calculator_short_circuit():
    """Delta-one products must reproduce _greeks_for_linear (delta=1,
    rest 0) — never finite-difference bump arithmetic."""
    from quantark.asset.equity.engine.analytical import DeltaOneEngine
    from quantark.asset.equity.product.deltaone import Futures

    product = Futures(underlying="TEST", maturity=1.0, basis=2.0)
    engine = DeltaOneEngine()
    env = _env()
    gc = GreeksCalculator()
    raw = float(engine.price(product, env))
    expected = gc.calculate_numerical_greeks(
        product, env, engine, base_price=raw, greeks=["price"] + GREEKS7
    )

    state = TradeState(
        product=product, pricing_env=env, cash_legs=(), engine=engine,
        streams=None, quantity=1.0, greeks_params=gc.params,
    )
    values = _cell_values(state, gc, GREEKS7)
    assert all(v.get("linear") for v in values.values())
    got = assemble_product_greeks(
        base_price=raw, bump_values=values, spot=env.spot,
        bump_config=gc._bump_config, requested=set(["price"] + GREEKS7),
    )
    assert got == expected


def test_unknown_greek_names_fail_closed():
    from quantark.util.exceptions import ValidationError

    with pytest.raises(ValidationError, match="veag"):
        greek_bump_cells(["delta", "veag"])


def test_packed_bases_share_one_child_context():
    """Different bases with identical child budgets must share ONE child
    context (leases/caches) per worker — not one budget domain per base."""
    import execution.scenario_process_helpers  # noqa: F401 - registers toys
    from quantark.execution.context import default_context
    from quantark.execution.contracts import ScenarioSpec
    from quantark.execution.scenario import worker as worker_mod
    from quantark.execution.scenario.contracts import BaseInputsRef
    from quantark.execution.scenario.planner import plan_scenarios, resolve_base

    def build(vol):
        base = BaseInputsRef(
            factory_id="toy-inputs/v1",
            payload=(("spot", 100.0), ("vol", vol)),
        )
        spec = ScenarioSpec(
            scenario_id="probe",
            transformer_id="toy-bump/v1",
            parameters=(("ds", 1.0),),
            mutation_tags=frozenset({"spot"}),
            required_capabilities=frozenset({"runner:toy/v1"}),
        )
        _, resolved, _ = resolve_base(base)
        plan = plan_scenarios(base, [spec], "toy-engine/v1", resolved=resolved)
        worker_spec = worker_mod.build_worker_spec(
            plan, base, default_context(), workers=1
        )
        payload = worker_mod.worker_spec_to_payload(worker_spec)
        cell_payload = worker_mod._cell_payload(plan.cells[0])
        return payload, cell_payload

    worker_mod._CHILD_CONTEXTS.clear()
    for vol in (0.25, 0.30):
        payload, cell_payload = build(vol)
        result = worker_mod.run_worker_cell(payload, cell_payload,
                                            "toy-engine/v1")
        assert result["error"] is None, result["error"]
    assert len(worker_mod._CHILD_CONTEXTS) == 1


def _legless_state():
    product = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )
    return TradeState(
        product=product, pricing_env=_env(), cash_legs=(),
        engine=BlackScholesEngine(), streams=None, quantity=1.0,
        greeks_params=GreeksCalculator().params,
    )


@pytest.mark.parametrize("bump_id,tag", [
    ("spot_up", "spot"), ("spot_down", "spot"), ("vol_up", "vol"),
    ("rate_up", "rate"), ("div_up", "div"), ("theta", "time"),
])
def test_planner_attributes_each_bump_to_its_tag(bump_id, tag):
    """POSITIVE: each bump plans cleanly with its declared tag (component
    extractors read the real class fields). NEGATIVE: the same bump with an
    empty declared footprint is rejected — attribution really detected the
    change."""
    from quantark.execution.contracts import ScenarioSpec
    from quantark.execution.errors import ValidationGateError
    from quantark.execution.scenario.contracts import BaseInputsRef
    from quantark.execution.scenario.planner import plan_scenarios

    state = _legless_state()
    base_ref = BaseInputsRef(factory_id="unused/v1", payload=())

    def spec(tags):
        return ScenarioSpec(
            scenario_id=f"cell::{bump_id}",
            transformer_id="greek-bump/v1",
            parameters=(("bump_id", bump_id),),
            mutation_tags=frozenset(tags),
            required_capabilities=frozenset({"runner:greek-value/v1"}),
        )

    plan = plan_scenarios(base_ref, [spec({tag})], None, resolved=state)
    assert len(plan.cells) == 1

    with pytest.raises(ValidationGateError):
        plan_scenarios(base_ref, [spec(set())], None, resolved=state)
