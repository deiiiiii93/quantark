"""F1-F2: a dated FX range accrual (domestic and foreign coupon) through the FX backtest and the FX dynamic scenario;
the numeric twins alongside.

Same method as probe_equity_runs.py: the run's value on day k vs the engine's price at day k's environment of the
contract with the fixings dated strictly before day k recorded by hand at their own day's spot. The quanto range
accrual is a subclass sharing the records and hooks (its engine needs a quanto environment, not built here).
"""
import contextlib
import io
import logging
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd

import _fixtures as F
from quantark.asset.fx.engine.analytical import FxRangeAccrualAnalyticalEngine
from quantark.asset.fx.engine.analytical.fx_foreign_range_accrual_engine import FxForeignRangeAccrualAnalyticalEngine
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import FxRangeAccrualConfig, FxRangeAccrualObservationRecord, FxRangeAccrualOption
from quantark.asset.fx.product.option.fx_foreign_range_accrual_option import FxForeignRangeAccrualOption
from quantark.backtest import FXBacktestConfig, FXBacktestEngine
from quantark.backtest.strategy import FXDeltaNeutralStrategy
from quantark.backtest.transaction_costs import ZeroCostModel
from quantark.dynamicscenario import FXDynamicScenarioConfig, FXDynamicScenarioEngine
from quantark.dynamicscenario.path.day_path import DayPath, DayStep, ParameterChange
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.portfolio.fx import FXPortfolio
from quantark.priceenv import FxPricingEnvironment
from quantark.stresstest.stress.stress_types import StressType

PAIR = CurrencyPair("EUR", "USD")
CLASSES = {"domestic": (FxRangeAccrualOption, FxRangeAccrualAnalyticalEngine),
           "foreign": (FxForeignRangeAccrualOption, FxForeignRangeAccrualAnalyticalEngine)}
SHOW = [0, 3, 4, 5, 10, 14, 18, 20, 24]


def fx_spot(day):
    return 1.25 if day <= F.FIX_DAY else 1.25 + 0.01 * (day - F.FIX_DAY)   # out of [1.10, 1.30] from day 10


def fx_env(day, spot=None):
    return FxPricingEnvironment(
        valuation_date=F.START + timedelta(days=day), spot_quote=SpotQuote(spot=fx_spot(day) if spot is None else spot),
        domestic_curve=FlatRateCurve(rate=0.05), foreign_curve=FlatRateCurve(rate=0.03),
        vol_surface=FlatVolSurface(volatility=0.10),
    )


def make(kind, form, outcomes=None):
    outcomes = outcomes or {}
    cls, _ = CLASSES[kind]
    cfg = FxRangeAccrualConfig(upper_barrier=1.30, lower_barrier=1.10, accrual_rate=0.04)
    if form == "dated":
        recs = [FxRangeAccrualObservationRecord(observation_date=d, observed_in_range=outcomes.get(i))
                for i, d in enumerate(F.fixing_dates())]
        return cls(notional=1_000_000.0, range_config=cfg, currency_pair=PAIR,
                   expiry_date=F.START + timedelta(days=365), initial_date=F.START, observation_records=recs)
    recs = [FxRangeAccrualObservationRecord(observation_time=(d - F.START).days / 365.0,
                                            observed_in_range=outcomes.get(i))
            for i, d in enumerate(F.fixing_dates())]
    return cls(notional=1_000_000.0, range_config=cfg, currency_pair=PAIR, maturity=1.0, observation_records=recs)


def book(product, kind):
    pf = FXPortfolio(portfolio_name="fx", pricing_environments={"EURUSD": fx_env(0)})
    pf.add_position(product=product, quantity=1.0, entry_price=0.0, underlying="EURUSD", engine=CLASSES[kind][1]())
    return pf


def backtest(product, kind, n):
    idx = pd.date_range(F.START, periods=n, freq="D")
    path = pd.DataFrame({"EURUSD_spot": [fx_spot(k) for k in range(n)], "EURUSD_vol": np.full(n, 0.10),
                         "EURUSD_dom_rate": np.full(n, 0.05), "EURUSD_for_rate": np.full(n, 0.03)}, index=idx)
    cfg = FXBacktestConfig(portfolio=book(product, kind), market_path=path,
                           strategy=FXDeltaNeutralStrategy(delta_threshold=1e12, rebalance_frequency="on_threshold"),
                           transaction_cost_model=ZeroCostModel())
    return [r["portfolio_value"] for r in FXBacktestEngine(cfg).run().rows]


def dynamic(product, kind, n):
    steps = [DayStep(day_index=k, changes=[ParameterChange("spot", StressType.VALUE, fx_spot(k))]) for k in range(n)]
    path = DayPath(name="fx", steps=steps, start_date=F.START)
    with contextlib.redirect_stdout(io.StringIO()):
        res = FXDynamicScenarioEngine(FXDynamicScenarioConfig(calculate_greeks=True)).run(book(product, kind), path)
    return [d.portfolio_value for d in res.day_results]


def reference(kind, k):
    prod = make(kind, "dated")
    recs = prod.get_observation_records()
    outcomes = {i: prod.is_in_range(fx_spot((d - F.START).days), i, recs[i])
                for i, d in enumerate(F.fixing_dates()) if (d - F.START).days < k}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CLASSES[kind][1]().price(make(kind, "dated", outcomes), fx_env(k))


def report(name, run):
    print(f"\n=== {name} ===")
    for kind in CLASSES:
        dated, n_d = F.count_missing(run, make(kind, "dated"), kind, F.DAYS)
        numeric, n_n = F.count_missing(run, make(kind, "numeric"), kind, F.DAYS)
        print(f"[{kind}] MissingFixingWarning count over the run: dated {n_d}, numeric {n_n}")
        print(f"{'day':>4} {'spot':>6} {'run dated':>14} {'by hand':>14} {'run-hand':>12} {'run numeric':>14}")
        for k in SHOW:
            ref = reference(kind, k)
            print(f"{k:>4} {fx_spot(k):>6.3f} {dated[k]:>14.4f} {ref:>14.4f} {dated[k] - ref:>12.4f} {numeric[k]:>14.4f}")


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    report("F1 FXBacktestEngine (daily calendar path, quantity 1, notional 1e6)", backtest)
    report("F2 FXDynamicScenarioEngine (dated DayPath)", dynamic)
