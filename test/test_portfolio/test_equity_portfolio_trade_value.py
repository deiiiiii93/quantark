"""EquityPortfolio trade-value mode (ship spec 2026-10-23 portfolio-trade-value D1–D6, D19, D21; AC1, AC2, AC5, AC13).

``EquityPortfolio(trade_value=True)`` values every EquityPosition at ``get_trade_value`` with Greeks from
``get_trade_risk(...).total`` and PnL from ``get_trade_pnl``; a TRS keeps its own path. Flag off is today's
product path, bitwise.
"""
import copy
from datetime import datetime

import pytest

from quantark.asset.equity.engine import BlackScholesEngine
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import QuadParams
from quantark.asset.equity.product import EuropeanVanillaOption
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.cashleg.autocallable_leg import AccrualBasis, AutocallableCashLeg, AutocallableLegType
from quantark.cashleg.base import LegDirection
from quantark.cashleg.deterministic_leg import DeterministicLeg
from quantark.cashleg.event_distribution import EventType
from quantark.dynamicscenario.engine import DynamicScenarioEngine
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.portfolio import EquityPosition, Portfolio
from quantark.portfolio.equity import valuation as shared
from quantark.priceenv import PricingEnvironment
from quantark.stresstest import StressTestConfig, StressTestEngine
from quantark.stresstest.equity import valuation as stress_valuation
from quantark.util.enum import OptionType
from quantark.util.exceptions import ValidationError
from test_stresstest.test_trs_stress import _make_trs

VAL = datetime(2026, 3, 16)
LEGACY_KEYS = ("market_value", "delta", "gamma", "vega", "theta", "rho")


def _env():
    return PricingEnvironment(spot_quote=SpotQuote(spot=100.0, asset_name="X"), vol_surface=FlatVolSurface(0.2),
                              rate_curve=FlatRateCurve(0.03), div_yield=ContinuousDividendYield(0.01),
                              valuation_date=VAL)


def _trs_env():
    return PricingEnvironment(spot_quote=SpotQuote(spot=4000.0, asset_name="IDX"), vol_surface=FlatVolSurface(0.2),
                              rate_curve=FlatRateCurve(0.048), valuation_date=datetime(2024, 1, 30))


def _snowball():
    return SnowballOption(initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=10_000.0,
                          barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15,
                                                       ko_observation_dates=[i / 12 for i in range(1, 13)],
                                                       ki_barrier=75.0, ki_continuous=True))


def _legs():
    rebate = AutocallableCashLeg(direction=LegDirection.BUYER_RECEIVES, leg_type=AutocallableLegType.REBATE,
                                 notional=1_000.0, rate=0.05, terminal_accrual_factor=1.0,
                                 terminal_settlement_time=1.0, accrual_basis=AccrualBasis.KO_MATURITY,
                                 terminal_events=frozenset({EventType.MATURITY_NO_KO}))
    fee = DeterministicLeg(direction=LegDirection.BUYER_PAYS, amount=250.0, payment_time=0.25)
    return [rebate, fee]


def _book(trade_value=None, entry_trade_value=-1_000.0):
    kwargs = {} if trade_value is None else {"trade_value": trade_value}
    pf = Portfolio(portfolio_name="book", pricing_environments={"X": _env(), "IDX": _trs_env()},
                   creation_date=VAL, **kwargs)
    vanilla = EuropeanVanillaOption(strike=100.0, maturity=0.5, option_type=OptionType.CALL)
    pf.positions["vanilla"] = EquityPosition(product=vanilla, quantity=10, entry_price=7.5, underlying="X",
                                             engine=BlackScholesEngine(), entry_timestamp=VAL,
                                             position_id="vanilla")
    pf.positions["note"] = EquityPosition(product=_snowball(), quantity=-2.0, entry_price=0.0, underlying="X",
                                          engine=SnowballQuadEngine(params=QuadParams()), entry_timestamp=VAL,
                                          position_id="note", cash_legs=_legs(),
                                          entry_trade_value=entry_trade_value)
    swap = pf.positions.pop(pf.add_swap_position(product=_make_trs(), quantity=1.0).position_id)
    swap.position_id = "trs"
    pf.positions["trs"] = swap
    return pf


def _trade_greeks(position, env, gc):
    total = copy.deepcopy(position).get_trade_risk(env, gc).total
    return shared.trade_greeks(position, env, gc, total, market_value=copy.deepcopy(position).get_trade_value(env))


# --- AC1: flag off is today's product path, bitwise ------------------------------------------------------------

def test_flag_defaults_off():
    assert _book().trade_value is False


def test_flag_off_value_pnl_and_greeks_are_the_product_path():
    pf = _book()
    gc = GreeksCalculator()
    envs = pf.pricing_environments
    ref = copy.deepcopy(pf)
    assert pf.get_portfolio_value() == sum(p.get_market_value(envs[p.underlying]) for p in ref.positions.values())
    assert pf.get_portfolio_pnl() == sum(p.get_pnl(envs[p.underlying]) for p in ref.positions.values())
    want = {k: 0.0 for k in LEGACY_KEYS}
    for p in ref.positions.values():
        g = p.get_greeks(envs[p.underlying], gc, True)
        for k in want:
            if k in g:
                want[k] += g[k]
    got = pf.get_portfolio_greeks(gc)
    assert list(got) == list(LEGACY_KEYS)
    assert got == want


# --- AC2: flag on ----------------------------------------------------------------------------------------------

def test_flag_on_value_is_the_trade_value():
    pf = _book(trade_value=True)
    envs = pf.pricing_environments
    ref = copy.deepcopy(pf)
    want = 0.0
    for p in ref.positions.values():
        want += p.get_trade_value(envs[p.underlying]) if shared.has_trade_api(p) else p.get_market_value(envs[p.underlying])
    assert pf.get_portfolio_value() == want
    note = ref.positions["note"]
    assert pf.position_value(pf.positions["note"]) == note.get_trade_value(envs["X"])
    assert pf.position_value(pf.positions["note"]) != note.get_market_value(envs["X"])  # the legs count


def test_flag_on_greeks_are_trade_risk_for_every_option_position():
    pf = _book(trade_value=True)
    gc = GreeksCalculator()
    envs = pf.pricing_environments
    ref = copy.deepcopy(pf)
    per = {
        "vanilla": _trade_greeks(ref.positions["vanilla"], envs["X"], gc),
        "note": _trade_greeks(ref.positions["note"], envs["X"], gc),
        "trs": ref.positions["trs"].get_greeks(envs["IDX"], gc, use_analytical=True),
    }
    got = pf.get_portfolio_greeks(gc)
    assert list(got) == list(shared.AGGREGATE_GREEK_KEYS)
    for key in shared.AGGREGATE_GREEK_KEYS:
        assert got[key] == sum(g.get(key, 0.0) for g in per.values()), key
    assert got["market_value"] == pf.get_portfolio_value()
    # the vanilla's delta is the bump delta, not the analytical one
    analytical = ref.positions["vanilla"].get_greeks(envs["X"], gc, use_analytical=True)["delta"]
    assert pf.position_greeks(pf.positions["vanilla"], gc)["delta"] == per["vanilla"]["delta"]
    assert per["vanilla"]["delta"] != analytical


def test_flag_on_greeks_by_underlying_and_risk_measures():
    pf = _book(trade_value=True)
    gc = GreeksCalculator()
    envs = pf.pricing_environments
    ref = copy.deepcopy(pf)
    by_x = pf.get_greeks_by_underlying("X", gc)
    assert list(by_x) == list(shared.AGGREGATE_GREEK_KEYS)
    for key in shared.AGGREGATE_GREEK_KEYS:
        assert by_x[key] == sum(_trade_greeks(ref.positions[p], envs["X"], gc)[key] for p in ("vanilla", "note"))
    assert pf.get_portfolio_risk_measures(gc) == pf.get_portfolio_greeks(gc)


# --- AC5: trade PnL --------------------------------------------------------------------------------------------

def test_trade_pnl_uses_entry_trade_value():
    pf = _book(trade_value=True, entry_trade_value=-1_000.0)
    env = pf.pricing_environments["X"]
    note = pf.positions["note"]
    assert note.resolved_entry_trade_value() == -1_000.0
    assert note.get_trade_pnl(env) == copy.deepcopy(note).get_trade_value(env) + 1_000.0
    assert pf.position_pnl(note) == note.get_trade_pnl(env)


def test_no_leg_position_entry_is_entry_price_times_quantity_and_pnl_is_bitwise_product():
    pf = _book(trade_value=True)
    vanilla = pf.positions["vanilla"]
    env = pf.pricing_environments["X"]
    assert vanilla.resolved_entry_trade_value() == 7.5 * 10
    assert pf.position_pnl(vanilla) == copy.deepcopy(vanilla).get_pnl(env)


@pytest.mark.parametrize("quantity,entry_price", [(3, 7.3), (-7, 2.9), (11, 6.1)])
def test_no_leg_trade_pnl_is_bitwise_product_pnl_for_inexact_inputs(quantity, entry_price):
    """CP1: price·q − entry·q is not (price − entry)·q in floating point; the no-leg trade PnL is get_pnl."""
    pf = _book(trade_value=True)
    vanilla = pf.positions["vanilla"]
    vanilla.quantity, vanilla.entry_price = quantity, entry_price
    env = pf.pricing_environments["X"]
    assert vanilla.get_trade_pnl(env) == copy.deepcopy(vanilla).get_pnl(env)
    assert pf.position_pnl(vanilla) == copy.deepcopy(vanilla).get_pnl(env)


def test_to_dataframe_follows_the_mode():
    pf = _book(trade_value=True)
    frame = pf.to_dataframe().set_index("position_id")
    assert frame.loc["note", "market_value"] == copy.deepcopy(pf).position_value(pf.positions["note"])
    assert frame["market_value"].sum() == pytest.approx(copy.deepcopy(pf).get_portfolio_value(), rel=1e-15)
    assert frame.loc["note", "pnl"] == copy.deepcopy(pf).position_pnl(pf.positions["note"])


def test_cash_leg_position_without_entry_trade_value_raises_in_trade_mode():
    pf = _book(trade_value=True, entry_trade_value=None)
    with pytest.raises(ValidationError, match="entry_trade_value"):
        pf.get_portfolio_pnl()
    with pytest.raises(ValidationError, match="entry_trade_value"):
        pf.position_pnl(pf.positions["note"])
    # product mode never asks for it
    assert isinstance(_book(trade_value=False, entry_trade_value=None).get_portfolio_pnl(), float)


def test_quantity_change_refused_when_entry_trade_value_is_set():
    pf = _book(trade_value=True)
    with pytest.raises(ValidationError, match="entry_trade_value"):
        pf.update_position("note", quantity=-3.0)
    pf.update_position("vanilla", quantity=5)  # no entry_trade_value: allowed
    assert pf.positions["vanilla"].quantity == 5


def test_to_dict_carries_entry_trade_value_only_when_set():
    pf = _book(trade_value=True)
    assert pf.positions["note"].to_dict()["entry_trade_value"] == -1_000.0
    assert "entry_trade_value" not in pf.positions["vanilla"].to_dict()


# --- D3: one shared switch --------------------------------------------------------------------------------------

def test_stress_valuation_reexports_the_shared_switch():
    for name in ("value_position", "value_positions", "trade_greeks", "aggregate_greeks", "has_trade_api",
                 "supports_native_batch", "PositionValuation", "AGGREGATE_GREEK_KEYS", "POSITION_GREEK_KEYS",
                 "TRADE_GREEKS"):
        assert getattr(stress_valuation, name) is getattr(shared, name), name


def test_value_position_product_and_trade_modes():
    pf = _book()
    gc = GreeksCalculator()
    env = pf.pricing_environments["X"]
    note = pf.positions["note"]
    product = shared.value_position(copy.deepcopy(note), env, gc, trade=False, use_analytical=False)
    assert product.value == copy.deepcopy(note).get_market_value(env)
    assert product.greeks == copy.deepcopy(note).get_greeks(env, gc, use_analytical=False)
    trade = shared.value_position(copy.deepcopy(note), env, gc, trade=True)
    assert trade.value == copy.deepcopy(note).get_trade_value(env)
    assert trade.greeks == _trade_greeks(note, env, gc)
    assert trade.greeks["market_value"] == trade.value


# --- AC13 / D19: clone sites keep the flag ---------------------------------------------------------------------

@pytest.mark.parametrize("flag", [True, False])
def test_dynamic_scenario_clone_keeps_the_flag(flag):
    pf = _book(trade_value=flag)
    assert DynamicScenarioEngine.__new__(DynamicScenarioEngine)._clone_portfolio(pf).trade_value is flag


@pytest.mark.parametrize("flag", [True, False])
def test_stress_portfolio_copies_keep_the_flag(flag):
    pf = _book(trade_value=flag)
    engine = StressTestEngine(StressTestConfig(calculate_greeks=False))
    assert engine._view(pf, pf.pricing_environments).trade_value is flag
    assert engine._create_stressed_portfolio(pf, pf.pricing_environments).trade_value is flag
