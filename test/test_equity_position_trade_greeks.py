"""EquityPosition.get_trade_risk: product + total greeks in one frozen loop.

Plan Task 3.3 [§11.4, §11.8]: one bump loop yields product and total greeks;
each bumped env reprices once via price_with_events; the frozen bump context is
resolved once; delta/gamma share the two spot bumps; the quantity/size contract
holds (product scales by signed quantity, legs are absolute); theta shifts the
product and the legs together.
"""

from datetime import datetime

from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.param import PDEParams
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
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.portfolio.equity.position import EquityPosition
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType


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
    # Terminal-only leg: PV depends on P(maturity, no KO); no schedule alignment.
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


def _position(engine, quantity=1.0):
    return EquityPosition(
        product=_snowball(),
        quantity=quantity,
        entry_price=100.0,
        underlying="TEST",
        engine=engine,
        entry_timestamp=datetime(2024, 1, 1),
        cash_legs=[_terminal_leg()],
    )


def _spy(engine):
    calls = {"ctx": 0, "pwe": 0}
    orig_ctx = engine.create_bump_context

    def ctx_spy(product, env):
        calls["ctx"] += 1
        clone = orig_ctx(product, env)
        orig_pwe = clone.price_with_events

        def pwe_spy(*a, **k):
            calls["pwe"] += 1
            return orig_pwe(*a, **k)

        clone.price_with_events = pwe_spy
        return clone

    engine.create_bump_context = ctx_spy
    return calls


def test_one_loop_product_and_total_with_expected_solves():
    engine = SnowballPDESolver(PDEParams(grid_size=150))
    calls = _spy(engine)
    pos = _position(engine)
    gc = GreeksCalculator()
    risk = pos.get_trade_risk(_env(), gc, ["delta", "gamma", "vega", "theta", "rho", "dividend_rho"])

    # Both product and total greeks produced.
    for k in ["delta", "gamma", "vega", "theta", "rho", "dividend_rho"]:
        assert k in risk.product and k in risk.total
    # Frozen bump context resolved exactly once.
    assert calls["ctx"] == 1
    # base + 2 spot (delta/gamma shared) + vega + rho + div + theta = 7 solves.
    assert calls["pwe"] == 7
    # Leg PV attribution present.
    assert len(risk.leg_pvs) == 1


def test_quantity_contract_product_scales_legs_absolute():
    env = _env()
    gc = GreeksCalculator()
    long = _position(SnowballPDESolver(PDEParams(grid_size=150)), quantity=1.0)
    short = _position(SnowballPDESolver(PDEParams(grid_size=150)), quantity=-1.0)
    rl = long.get_trade_risk(env, gc, ["delta"])
    rs = short.get_trade_risk(env, gc, ["delta"])

    # Product delta flips sign with quantity.
    assert abs(rl.product["delta"] + rs.product["delta"]) < 1e-6 * max(
        abs(rl.product["delta"]), 1.0
    )
    # The leg delta (total - product) is absolute (identical for long and short).
    leg_delta_long = rl.total["delta"] - rl.product["delta"]
    leg_delta_short = rs.total["delta"] - rs.product["delta"]
    assert abs(leg_delta_long - leg_delta_short) < 1e-6 * max(abs(leg_delta_long), 1.0)


def test_total_price_matches_trade_value_breakdown():
    engine = SnowballPDESolver(PDEParams(grid_size=150))
    env = _env()
    pos = _position(engine)
    gc = GreeksCalculator()
    risk = pos.get_trade_risk(env, gc, ["delta"])
    # total price == quantity*npv + Σ leg_pv (legs absolute) [§11.8]
    expected = risk.product["price"] + sum(lp.pv for lp in risk.leg_pvs.values())
    assert abs(risk.total["price"] - expected) < 1e-9
