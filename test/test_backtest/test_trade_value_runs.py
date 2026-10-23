"""Trade-value backtests and dynamic scenarios (ship spec 2026-10-23 portfolio-trade-value D16, D17; AC3, AC10, AC13,
AC14 run level).

The backtest carries cash legs / folding / entry_trade_value through ``initial_positions`` and books the legs'
flows; a dynamic scenario books them through its lifecycle manager. Day 0 equals the static trade value of the
same book, and the cash the legs pay over a run is their realized flows.
"""
import copy
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from quantark.asset.equity.engine.analytical import BarrierAnalyticalEngine
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import QuadParams
from quantark.asset.equity.product.option.barrier_option import BarrierOption
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.backtest import BacktestConfig, BacktestEngine, ZeroCostModel
from quantark.backtest.strategy import DeltaNeutralStrategy
from quantark.cashleg import (
    AccrualLeg,
    BaseAmount,
    BaseAmountMode,
    DeterministicLeg,
    EventType,
    KOBehavior,
    LegDirection,
    LegSchedule,
)
from quantark.cashleg.accrual_leg import SurvivalBasis
from quantark.cashleg.autocallable_leg import AccrualBasis, AutocallableCashLeg, AutocallableLegType
from quantark.cashleg.fixed_payoff_leg import FixedPayoffLeg, PaymentTrigger
from quantark.dynamicscenario import DynamicScenarioConfig, DynamicScenarioEngine, PathLibrary
from quantark.execution.errors import CapabilityError
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.portfolio import EquityPosition, Portfolio
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.enum.option_enums import BarrierType
from quantark.util.exceptions import ValidationError
from quantark.util.marketdata.models import MarketDataSet, TimeSeriesData

UNDERLYING = "EQUITY_INDEX"
START = datetime(2024, 1, 1)          # a Monday: whole-week offsets land on business days
D = 1 / 365.0
OBS = [28, 56, 84]
KO_DAY = 56
VOL, RATE, DIV = 0.25, 0.05, 0.02


class StepAdapter:
    """Spot 100 until ``KO_DAY`` (calendar days from START), 110 from then on; flat vol, rate and dividend."""

    def __init__(self, num_days=70):
        self.dates = pd.bdate_range(start=START, periods=num_days)
        self._spots = [110.0 if (d - pd.Timestamp(START)).days >= KO_DAY else 100.0 for d in self.dates]

    def get_market_data_set(self, asset_name, start_date, end_date, currency, frequency):
        idx = self.dates

        def series(column, values):
            return TimeSeriesData(pd.DataFrame({column: values}, index=idx), asset_name, column)

        return MarketDataSet(spot_data=series("spot", self._spots),
                             vol_data=series("volatility", [VOL] * len(idx)),
                             rate_data=series("rate", [RATE] * len(idx)),
                             div_yield_data=series("div_yield", [DIV] * len(idx)))


def _snowball():
    return SnowballOption(initial_price=100.0, strike=100.0, maturity=OBS[-1] * D, contract_multiplier=1_000.0,
                          barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15,
                                                       ko_observation_dates=[d * D for d in OBS],
                                                       ki_barrier=75.0, ki_continuous=True))


def _legs():
    obs = tuple(d * D for d in OBS)
    starts = np.array([0] + OBS[:-1]) * D
    ends = np.array(OBS) * D
    return [
        DeterministicLeg(direction=LegDirection.BUYER_PAYS, amount=250.0, payment_time=7 * D),
        AccrualLeg(direction=LegDirection.BUYER_RECEIVES, rate=0.04,
                   base=BaseAmount(value=100_000.0, mode=BaseAmountMode.ABSOLUTE),
                   schedule=LegSchedule(period_starts=starts, period_ends=ends, payment_times=ends.copy()),
                   ko_behavior=KOBehavior.TRUNCATE_AT_KO, survival_basis=SurvivalBasis.ENTER_PERIOD),
        FixedPayoffLeg(direction=LegDirection.BUYER_RECEIVES, amount=1_000.0, trigger=PaymentTrigger.AT_KO),
        AutocallableCashLeg(direction=LegDirection.BUYER_RECEIVES, leg_type=AutocallableLegType.REBATE,
                            notional=10_000.0, rate=0.05, observation_schedule=obs, accrual_factors=obs,
                            settlement_schedule=obs, terminal_accrual_factor=OBS[-1] * D,
                            terminal_settlement_time=OBS[-1] * D, accrual_basis=AccrualBasis.KO_MATURITY,
                            terminal_events=frozenset({EventType.MATURITY_NO_KO})),
    ]


#: fee + two accrual periods starting before the KO + KO payoff + rebate at the KO observation
REALIZED_LEG_CASH = (-250.0 + 2 * 0.04 * 100_000.0 * 28 * D + 1_000.0 + 10_000.0 * 0.05 * KO_DAY * D)


def _position(underlying=UNDERLYING, legs=True):
    return EquityPosition(product=_snowball(), quantity=1.0, entry_price=0.0, underlying=underlying,
                          engine=SnowballQuadEngine(params=QuadParams()), entry_timestamp=START,
                          cash_legs=_legs() if legs else [], entry_trade_value=-500.0)


def _config(trade_value=True, positions=None, adapter=None):
    adapter = adapter or StepAdapter()
    return BacktestConfig(strategy=DeltaNeutralStrategy(delta_threshold=1e12), start_date=START,
                          end_date=adapter.dates[-1].to_pydatetime(), underlying=UNDERLYING,
                          initial_positions=positions if positions is not None else [_position()],
                          market_data_adapter=adapter, transaction_cost_model=ZeroCostModel(),
                          calculate_greeks=False, trade_value=trade_value)


def _static_value(trade_value, spot=100.0, valuation_date=START, underlying=UNDERLYING):
    env = PricingEnvironment(spot_quote=SpotQuote(spot=spot, asset_name=underlying), vol_surface=FlatVolSurface(VOL),
                             rate_curve=FlatRateCurve(RATE), div_yield=ContinuousDividendYield(DIV),
                             valuation_date=valuation_date)
    pf = Portfolio(portfolio_name="static", pricing_environments={underlying: env}, creation_date=START,
                   trade_value=trade_value)
    pf.positions["note"] = _position(underlying)
    return pf.get_portfolio_value()


# --- backtest ----------------------------------------------------------------------------------------------------

def test_backtest_carries_the_trade_terms_through_initial_positions():
    engine = BacktestEngine(_config())
    engine._initialize()
    (position,) = engine.portfolio.positions.values()
    assert engine.portfolio.trade_value is True
    assert [type(l) for l in position.cash_legs] == [type(l) for l in _legs()]
    assert position.entry_trade_value == -500.0


def test_backtest_books_the_legs_and_starts_at_the_static_trade_value():
    engine = BacktestEngine(_config())
    results = engine.run()
    states = results.states_df
    assert states["portfolio_value"].iloc[0] == _static_value(trade_value=True)      # AC3, bitwise
    assert engine.lifecycle_manager.leg_paid_cash == pytest.approx(REALIZED_LEG_CASH, rel=1e-13)
    assert states["num_positions"].iloc[-1] == 0                                       # knocked out


def test_backtest_flag_off_books_no_legs_and_starts_at_the_product_value():
    engine = BacktestEngine(_config(trade_value=False))
    results = engine.run()
    assert results.states_df["portfolio_value"].iloc[0] == _static_value(trade_value=False)
    assert engine.lifecycle_manager.leg_paid_cash == 0.0


def test_backtest_refuses_legs_on_a_barrier_position():
    barrier = BarrierOption(strike=100.0, option_type=OptionType.PUT, barrier=85.0, barrier_type=BarrierType.DOWN_OUT,
                            rebate=2.0, pay_at_hit=True, contract_multiplier=1.0, maturity=1.0)
    position = EquityPosition(product=barrier, quantity=1.0, entry_price=0.0, underlying=UNDERLYING,
                              engine=BarrierAnalyticalEngine(), entry_timestamp=START,
                              cash_legs=[DeterministicLeg(direction=LegDirection.BUYER_PAYS, amount=1.0,
                                                          payment_time=7 * D)], entry_trade_value=0.0)
    with pytest.raises(CapabilityError):
        BacktestEngine(_config(positions=[position]))._initialize()


# --- dynamic scenarios -------------------------------------------------------------------------------------------

def _dynamic_portfolio(trade_value):
    env = PricingEnvironment(spot_quote=SpotQuote(spot=100.0, asset_name="IDX"), vol_surface=FlatVolSurface(VOL),
                             rate_curve=FlatRateCurve(RATE), div_yield=ContinuousDividendYield(DIV),
                             valuation_date=START)
    pf = Portfolio(portfolio_name="dyn", pricing_environments={"IDX": env}, creation_date=START,
                   trade_value=trade_value)
    pf.positions["note"] = _position("IDX")
    return pf


def _rally():
    path = PathLibrary.consecutive_rally(days=60, daily_pct=0.004, vol_change_pct=0.0)
    path.start_date = START
    return path


def test_dynamic_scenario_books_the_legs_and_starts_at_the_static_trade_value():
    trade = DynamicScenarioEngine(DynamicScenarioConfig(calculate_greeks=False, handle_lifecycle_events=True)).run(
        _dynamic_portfolio(True), _rally())
    product = DynamicScenarioEngine(DynamicScenarioConfig(calculate_greeks=False, handle_lifecycle_events=True)).run(
        _dynamic_portfolio(False), _rally())
    assert trade.baseline_value == _static_value(trade_value=True, underlying="IDX")      # AC3, bitwise
    ko = [d.day_index for d in trade.day_results if any(e.event_type == "KO" for e in d.lifecycle_events)]
    assert ko, "the rally must knock the note out"
    # every leg flow is realized by the end of the run: the trade run's cash exceeds the product run's by them
    assert trade.day_results[-1].paid_cash - product.day_results[-1].paid_cash == pytest.approx(
        _realized_leg_cash_for_ko(ko[0]), rel=1e-12)


def _realized_leg_cash_for_ko(ko_day):
    """The legs' realized cash when the note knocks out at the observation on ``ko_day``."""
    k = OBS.index(ko_day)
    periods = sum(0.04 * 100_000.0 * 28 * D for start in [0] + OBS[:-1] if start < ko_day)
    return -250.0 + periods + 1_000.0 + 10_000.0 * 0.05 * OBS[k] * D


def test_dynamic_trade_mode_with_legs_needs_lifecycle_handling_and_a_dated_path():
    with pytest.raises(ValidationError, match="handle_lifecycle_events"):
        DynamicScenarioEngine(DynamicScenarioConfig(calculate_greeks=False, handle_lifecycle_events=False)).run(
            _dynamic_portfolio(True), _rally())
    undated = PathLibrary.consecutive_rally(days=5, daily_pct=0.004)
    with pytest.raises(ValidationError, match="dated"):
        DynamicScenarioEngine(DynamicScenarioConfig(calculate_greeks=False, handle_lifecycle_events=True)).run(
            _dynamic_portfolio(True), undated)


def test_dynamic_position_rows_follow_the_mode():
    results = DynamicScenarioEngine(DynamicScenarioConfig(calculate_greeks=False, handle_lifecycle_events=True)).run(
        _dynamic_portfolio(True), _rally())
    first = results.day_results[0]
    (row,) = first.positions
    assert row.pnl == pytest.approx(row.market_value + 500.0, rel=1e-14)          # trade PnL vs entry_trade_value
