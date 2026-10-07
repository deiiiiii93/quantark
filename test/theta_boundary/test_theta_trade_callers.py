"""Every theta caller takes the one step of ``theta_shift`` (spec 2026-10-07 R1, R4): the execution bump cells, the
position's trade risk with cash legs, and the bitwise parity between the two."""
from datetime import datetime

import pytest

from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.asset.equity.riskmeasures.greeks import bump_envs
from quantark.asset.equity.riskmeasures.greeks.theta_shift import (carried_ended_value, shift_cash_legs,
                                                                     shift_for_theta)
from quantark.cashleg.autocallable_leg import AccrualBasis, AutocallableCashLeg, AutocallableLegType
from quantark.cashleg.base import LegDirection
from quantark.cashleg.event_distribution import EventType
from quantark.cashleg.leg_valuator import value_leg
from quantark.execution.greeks import (TradeState, assemble_product_greeks, assemble_trade_greeks,
                                       greek_bump_cells, greek_bump_transform, run_greek_bump)
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.portfolio.equity.position import EquityPosition
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType, OptionType

DISC = ObservationType.DISCRETE


def _env(spot=100.0):
    return PricingEnvironment(spot_quote=SpotQuote(spot=spot), vol_surface=FlatVolSurface(volatility=0.2),
                              rate_curve=FlatRateCurve(rate=0.03), div_yield=ContinuousDividendYield(div_yield=0.01),
                              valuation_date=datetime(2024, 1, 1))


def _snowball(first_ko):
    cfg = BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=DISC,
                        ko_observation_dates=[first_ko] + [i / 12 for i in range(1, 13)],
                        ki_barrier=75.0, ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True)
    return SnowballOption(initial_price=100.0, strike=100.0, barrier_config=cfg, contract_multiplier=10_000.0,
                          maturity=1.0)


def _leg(first_ko):
    dates = [first_ko] + [i / 12 for i in range(1, 13)]
    return AutocallableCashLeg(direction=LegDirection.BUYER_RECEIVES, leg_type=AutocallableLegType.REBATE,
                               notional=1_000.0, rate=0.05, observation_schedule=tuple(dates),
                               accrual_factors=tuple(dates), settlement_schedule=tuple(dates),
                               terminal_accrual_factor=1.0, terminal_settlement_time=1.0,
                               accrual_basis=AccrualBasis.KO_MATURITY,
                               terminal_events=frozenset({EventType.MATURITY_NO_KO}))


def _position(product, quantity=1.0, legs=None):
    return EquityPosition(product=product, quantity=quantity, entry_price=100.0, underlying="TEST",
                          engine=SnowballPDESolver(PDEParams()), entry_timestamp=datetime(2024, 1, 1),
                          cash_legs=list(legs or []))


def _cells(state, gc, greeks):
    out = {}
    for cell in greek_bump_cells(greeks):
        bumped = greek_bump_transform(state, {"bump_id": cell.bump_id})
        out[cell.bump_id] = run_greek_bump(cell.bump_id, state, bumped, gc)
    return out


def _execution_trade_theta(product, env, legs, quantity, streams):
    gc = GreeksCalculator()
    state = TradeState(product=product, pricing_env=env, cash_legs=tuple(legs),
                       engine=SnowballPDESolver(PDEParams()), streams=streams, quantity=quantity,
                       greeks_params=gc.params)
    values = _cells(state, gc, ["theta"])
    return assemble_trade_greeks(values["base"], values, quantity=quantity, spot=env.spot,
                                 bump_config=gc._bump_config, requested={"theta"})


def test_execution_product_theta_is_numerical_theta_on_the_landing_case():
    from quantark.asset.equity.engine.analytical.accumulator_analytical_engine import AccumulatorAnalyticalEngine
    from quantark.asset.equity.product.option import AccumulatorOption
    p = AccumulatorOption(strike=95.0, knock_out_barrier=105.0, initial_price=100.0, maturity=63 / 365,
                          option_type=OptionType.CALL, notional=1_000_000.0, observation_type=DISC,
                          observation_dates=[k / 365 for k in range(1, 64)])
    eng, env, gc = AccumulatorAnalyticalEngine(), _env(), GreeksCalculator()
    want = gc.calculate(product=p, pricing_env=env, engine=eng, greeks=["theta"])["theta"]
    state = TradeState(product=p, pricing_env=env, cash_legs=(), engine=eng, streams=None, quantity=1.0,
                       greeks_params=gc.params)
    values = _cells(state, gc, ["theta"])
    got = assemble_product_greeks(base_price=float(eng.price(p, env)), bump_values=values, spot=env.spot,
                                  bump_config=gc._bump_config, requested={"theta"})
    assert got["theta"] == want


def test_trade_theta_keeps_the_ko_observation_landing_on_t1():
    first_ko = 1 / 365                                   # the calendar-day bump lands on it
    env, gc = _env(), GreeksCalculator()
    pos = _position(_snowball(first_ko), legs=[_leg(first_ko)])
    risk = pos.get_trade_risk(env, gc, ["theta"])
    bumped_date, tb, _ = bump_envs.advance_theta_bump(env, 1, "calendar_days")
    shift = shift_for_theta(_snowball(first_ko), env, tb, bumped_date)
    assert shift.product.barrier_config.ko_observation_dates[0] == 0.0
    # every trade-risk cell prices on the bump engine frozen from the base state
    engine = gc._resolve_bump_engine(_snowball(first_ko), env, SnowballPDESolver(PDEParams()))
    streams = pos._required_streams()
    base = engine.price_with_events(_snowball(first_ko), env, streams=streams)
    adv = engine.price_with_events(shift.product, shift.env, streams=streams)
    unit = pos._get_unit_notional(env)
    legs_0 = value_leg(_leg(first_ko), base.event_distribution, env, unit)
    (moved,) = shift_cash_legs([_leg(first_ko)], tb)
    legs_1 = value_leg(moved, adv.event_distribution, shift.env, pos._get_unit_notional(shift.env))
    assert risk.product["theta"] == pytest.approx(float(adv.npv) + shift.fixed_pv - float(base.npv), abs=1e-9)
    assert risk.total["theta"] == pytest.approx(risk.product["theta"] + legs_1 - legs_0, abs=1e-9)


@pytest.mark.parametrize("spot", [100.0, 104.0])
def test_execution_trade_cells_match_trade_risk_bitwise_at_the_boundary(spot):
    first_ko = 0.0 if spot > 103.0 else 1 / 365          # ended at t0 / landing on t1
    env, gc = _env(spot), GreeksCalculator()
    pos = _position(_snowball(first_ko), quantity=-1.0, legs=[_leg(first_ko)])
    risk = pos.get_trade_risk(env, gc, ["theta"])
    got = _execution_trade_theta(_snowball(first_ko), _env(spot), [_leg(first_ko)], -1.0, pos._required_streams())
    assert got["product"]["theta"] == risk.product["theta"], (got["product"]["theta"], risk.product["theta"])
    assert got["total"]["theta"] == risk.total["theta"], (got["total"]["theta"], risk.total["theta"])


def test_trade_knocked_out_at_t0_carries_its_base_value():
    env, gc = _env(104.0), GreeksCalculator()
    pos = _position(_snowball(0.0), legs=[_leg(0.0)])
    risk = pos.get_trade_risk(env, gc, ["price", "theta"])
    bumped_date, tb, _ = bump_envs.advance_theta_bump(env, 1, "calendar_days")
    shift = shift_for_theta(_snowball(0.0), env, tb, bumped_date)
    assert shift.terminated
    base_npv = risk.product["price"]
    base_legs = risk.total["price"] - base_npv
    assert risk.product["theta"] == pytest.approx(carried_ended_value(shift, base_npv) - base_npv, abs=1e-9)
    assert risk.total["theta"] == pytest.approx(
        carried_ended_value(shift, base_npv + base_legs) - base_npv - base_legs, abs=1e-9)
    assert risk.product["theta"] != 0.0
