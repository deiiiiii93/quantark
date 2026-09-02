"""PnL explain demo: a vanilla, a barrier and a snowball across a Friday -> Monday gap,
a knock-out day with its event row, and a portfolio explain with reconciliation.

Run:  PYTHONPATH=. python example/pnl_explain_demo.py
"""
from copy import deepcopy
from datetime import datetime, timedelta

import pandas as pd

from quantark.asset.equity.engine.analytical import BarrierAnalyticalEngine
from quantark.asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.lifecycle.autocallable import AutocallableLifecycleTracker
from quantark.asset.equity.lifecycle.events import LifecycleEventType
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option.barrier_option import BarrierOption
from quantark.asset.equity.product.option.european_vanilla_option import EuropeanVanillaOption
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.pnlexplain import (
    BookSnapshot, ExplainMethod, LifecycleTransition, PositionSnapshot, ValuationSnapshot,
    explain, explain_portfolio,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType, OptionType
from quantark.util.enum.option_enums import BarrierType

pd.set_option("display.width", 160)
pd.set_option("display.max_rows", 80)
COLS = ["method", "kind", "factor", "term", "pnl", "greek", "cash_greek", "spot_return", "vol_pts", "days"]
FRI, MON = datetime(2026, 6, 26), datetime(2026, 6, 29)


def env(spot, vol, rate, date):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot), vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate), div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=date)


def snowball():
    dates = [0.25, 0.5, 0.75, 1.0]
    return SnowballOption(
        initial_price=100.0, strike=100.0,
        barrier_config=BarrierConfig(ko_barrier=[105.0, 104.0, 103.0, 102.0], ko_rate=0.10,
                                     ko_observation_type=ObservationType.DISCRETE, ko_observation_dates=dates,
                                     ki_barrier=75.0, ki_observation_type=ObservationType.DISCRETE,
                                     ki_observation_dates=dates, ki_continuous=False),
        payoff_config=PayoffConfig(rebate_rate=0.0, include_principal=True),
        accrual_config=AccrualConfig(is_annualized=True),
        contract_multiplier=1.0, maturity=1.0, is_reverse=False)


def show(title, res):
    print(f"\n=== {title}: total {res.total_pnl:+.4f}  "
          f"(waterfall gap {res.reconcile(ExplainMethod.WATERFALL):+.2e}, "
          f"taylor gap {res.reconcile(ExplainMethod.TAYLOR):+.2e}, unexplained {res.unexplained:+.4f})")
    print(res.to_frame()[COLS].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# --- 1. Friday -> Monday market move on three products ------------------------------------
e0, e1 = env(100.0, 0.20, 0.03, FRI), env(103.0, 0.22, 0.032, MON)
bs, bar, pde = BlackScholesEngine(), BarrierAnalyticalEngine(), SnowballPDESolver(PDEParams(accuracy="fast"))
days = (MON - FRI).days

vanilla0 = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
vanilla1 = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0 - days / 365)
show("vanilla call x10", explain(ValuationSnapshot(vanilla0, bs, e0, date=FRI, quantity=10.0),
                                 ValuationSnapshot(vanilla1, bs, e1, date=MON, quantity=10.0)))

barrier0 = BarrierOption(strike=100.0, option_type=OptionType.PUT, barrier=85.0,
                         barrier_type=BarrierType.DOWN_OUT, maturity=1.0)
barrier1 = BarrierOption(strike=100.0, option_type=OptionType.PUT, barrier=85.0,
                         barrier_type=BarrierType.DOWN_OUT, maturity=1.0 - days / 365)
show("down-and-out put x-5", explain(ValuationSnapshot(barrier0, bar, e0, date=FRI, quantity=-5.0),
                                     ValuationSnapshot(barrier1, bar, e1, date=MON, quantity=-5.0)))

# A snowball's contract rolls schedule and maturity together: use the tracker, never assign maturity.
tracker = AutocallableLifecycleTracker(product=snowball(), quantity=-3.0, start_date=pd.Timestamp(FRI))
sb1 = tracker.product_for_pricing(pd.Timestamp(MON), e1)
show("snowball x-3 (PDE fast)", explain(ValuationSnapshot(snowball(), pde, e0, date=FRI, quantity=-3.0),
                                        ValuationSnapshot(sb1, pde, e1, date=MON, quantity=-3.0),
                                        transition=None))

# --- 2. A knock-out day: the event row -----------------------------------------------------
START = datetime(2024, 1, 1)


def find_event_day(tracker, event_type, spot_quiet, spot_trigger, max_days=400):
    d = START
    for _ in range(max_days):
        d = d + timedelta(days=1)
        probe = deepcopy(tracker)
        ev = probe.observe(pd.Timestamp(d), probe.product_for_lifecycle(), env(spot_trigger, 0.22, 0.02, d), spot_trigger)
        if any(e.event_type is event_type for e in ev):
            return d - timedelta(days=1), d
        tracker.observe(pd.Timestamp(d), tracker.product_for_lifecycle(), env(spot_quiet, 0.22, 0.02, d), spot_quiet)
    raise RuntimeError("no event")


ko_tracker = AutocallableLifecycleTracker(product=snowball(), quantity=-3.0, start_date=pd.Timestamp(START))
t0, t1 = find_event_day(ko_tracker, LifecycleEventType.KNOCK_OUT, spot_quiet=100.0, spot_trigger=110.0)
ke0, ke1 = env(100.0, 0.22, 0.02, t0), env(110.0, 0.22, 0.02, t1)
state_before = deepcopy(ko_tracker.lifecycle)
product_t0 = ko_tracker.product_for_pricing(pd.Timestamp(t0), ke0)
product_alive = ko_tracker.product_for_pricing(pd.Timestamp(t1), ke1)
events = ko_tracker.observe(pd.Timestamp(t1), ko_tracker.product_for_lifecycle(), ke1, 110.0)
state_after = deepcopy(ko_tracker.lifecycle)
transition = LifecycleTransition(product_alive_t1=product_alive, engine_alive_t1=pde,
                                 state_before=state_before, state_after=state_after, events=tuple(events))
ko0 = ValuationSnapshot(product_t0, pde, ke0, date=t0, quantity=-3.0, lifecycle_state=state_before,
                        valuation_point=state_before.valuation_point)
ko1 = ValuationSnapshot(ko_tracker.product_for_pricing(pd.Timestamp(t1), ke1), pde, ke1, date=t1, quantity=-3.0,
                        lifecycle_state=state_after, valuation_point=state_after.valuation_point)
show(f"snowball KO day {t1.date()}", explain(ko0, ko1, transition=transition))

# --- 3. A three-position book ------------------------------------------------------------------
book0 = BookSnapshot(date=FRI, environments={"IDX": e0}, positions={
    "vanilla": PositionSnapshot("vanilla", "IDX", ValuationSnapshot(vanilla0, bs, e0, date=FRI, quantity=10.0)),
    "barrier": PositionSnapshot("barrier", "IDX", ValuationSnapshot(barrier0, bar, e0, date=FRI, quantity=-5.0)),
    "snowball": PositionSnapshot("snowball", "IDX", ValuationSnapshot(snowball(), pde, e0, date=FRI, quantity=-3.0)),
})
book1 = BookSnapshot(date=MON, environments={"IDX": e1}, positions={
    "vanilla": PositionSnapshot("vanilla", "IDX", ValuationSnapshot(vanilla1, bs, e1, date=MON, quantity=10.0)),
    "barrier": PositionSnapshot("barrier", "IDX", ValuationSnapshot(barrier1, bar, e1, date=MON, quantity=-5.0)),
    "snowball": PositionSnapshot("snowball", "IDX", ValuationSnapshot(sb1, pde, e1, date=MON, quantity=-3.0)),
})
port = explain_portfolio(book0, book1)
print(f"\n=== portfolio: total {port.total_pnl:+.4f}")
print(port.to_frame().query("position_id == 'portfolio' and underlying == '*'")[COLS]
      .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nreconciliation:", port.metadata["reconciliation"])
