"""W4b B6 byte gate (plan 2026-10-09-fx-tarn-lifecycle, Tasks 1 and 12).

capture <json>: run on the implementation branch BEFORE any library change (== main) and store the FX backtest and FX
dynamic-scenario rows of a book holding every M7 FX product except TARN/TARF.
compare <json>: run on the finished branch; every stored number must match to the bit, and the new cash columns must
be exactly 0.0.

Run: env PYTHONPATH=<WT>:<WT>/test <venv>/bin/python <this file> capture|compare <WT>/.superpowers/w4b/baseline.json
"""
import json
import math
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from quantark.asset.fx.engine.analytical import (
    FxDigitalOptionAnalyticalEngine, FxForeignRangeAccrualAnalyticalEngine, FxQuantoDigitalAnalyticalEngine,
    FxQuantoRangeAccrualAnalyticalEngine, GarmanKohlhagenEngine, GarmanKohlhagenQuantoEngine,
)
from quantark.asset.fx.engine.analytical.fx_range_accrual_engine import FxRangeAccrualAnalyticalEngine
from quantark.asset.fx.engine.mc import FxBarrierMCEngine, FxSharkfinMCEngine
from quantark.asset.fx.engine.mc.fx_mc_params import FxMCParams
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import (
    FxBarrierOption, FxDigitalOption, FxForeignRangeAccrualOption, FxQuantoDigitalOption, FxQuantoRangeAccrualOption,
    FxQuantoVanillaOption, FxRangeAccrualConfig, FxRangeAccrualOption, FxSharkfinOption, FxVanillaOption,
)
from quantark.backtest.fx import FXBacktestConfig, FXBacktestEngine
from quantark.backtest.strategy.fx_delta_neutral_strategy import FXDeltaNeutralStrategy
from quantark.dynamicscenario import FXDynamicScenarioConfig, FXDynamicScenarioEngine
from quantark.dynamicscenario.path.day_path import DayPath, DayStep, ParameterChange
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.portfolio.fx import FXPortfolio
from quantark.priceenv import FxPricingEnvironment
from quantark.priceenv.fx_pricing_environment import FxQuantoMarketData, QuantoConversionOrientation
from quantark.stresstest.stress.stress_types import StressType
from quantark.util.enum import FxBarrierType, OptionType

warnings.simplefilter("ignore")
PAIR = CurrencyPair("EUR", "USD")
V = datetime(2026, 6, 15)
FIX = [k / 12 for k in range(1, 13)]
NEW_BACKTEST_KEYS = {"paid_cash", "pending_pv"}


def env():
    return FxPricingEnvironment(
        valuation_date=V, spot_quote=SpotQuote(spot=1.20), domestic_curve=FlatRateCurve(rate=0.05),
        foreign_curve=FlatRateCurve(rate=0.03), vol_surface=FlatVolSurface(volatility=0.10),
        quanto=FxQuantoMarketData(settlement_curve=FlatRateCurve(rate=0.02), quanto_vol=0.08, correlation=0.3,
                                  conversion_orientation=QuantoConversionOrientation.SETTLEMENT_PER_DOMESTIC))


def book():
    mc = FxMCParams(num_paths=5_000, seed=7, time_steps=24)
    cfg = FxRangeAccrualConfig(upper_barrier=1.30, lower_barrier=1.10, accrual_rate=0.04)
    rows = [
        (FxVanillaOption(strike=1.20, option_type=OptionType.CALL, maturity=1.0, notional_foreign=1.0,
                         currency_pair=PAIR), GarmanKohlhagenEngine()),
        (FxVanillaOption(strike=1.20, option_type=OptionType.CALL, expiry_date=V + timedelta(days=365),
                         notional_foreign=1.0, currency_pair=PAIR), GarmanKohlhagenEngine()),
        (FxDigitalOption(strike=1.20, option_type=OptionType.CALL, payout=1.0, maturity=1.0, currency_pair=PAIR),
         FxDigitalOptionAnalyticalEngine()),
        (FxQuantoVanillaOption(strike=1.20, option_type=OptionType.CALL, quanto_fx_rate=1.0, maturity=1.0,
                               currency_pair=PAIR, notional_foreign=1.0), GarmanKohlhagenQuantoEngine()),
        (FxBarrierOption(strike=1.20, barrier=1.35, is_up=True, knock_type=FxBarrierType.KNOCK_OUT,
                         option_type=OptionType.CALL, maturity=1.0, currency_pair=PAIR), FxBarrierMCEngine(mc)),
        (FxSharkfinOption(strike=1.20, barrier=1.35, is_up=True, option_type=OptionType.CALL, maturity=1.0,
                          currency_pair=PAIR), FxSharkfinMCEngine(mc)),
        (FxRangeAccrualOption(notional=1.0, range_config=cfg, maturity=1.0, observation_times=FIX,
                              currency_pair=PAIR), FxRangeAccrualAnalyticalEngine()),
        (FxQuantoDigitalOption(strike=1.20, option_type=OptionType.CALL, payout=1.0, quanto_fx_rate=1.0,
                               maturity=1.0, currency_pair=PAIR), FxQuantoDigitalAnalyticalEngine()),
        (FxQuantoRangeAccrualOption(notional=1.0, range_config=cfg, quanto_fx_rate=1.0, maturity=1.0,
                                    observation_times=FIX, currency_pair=PAIR),
         FxQuantoRangeAccrualAnalyticalEngine()),
        (FxForeignRangeAccrualOption(notional=1.0, range_config=cfg, maturity=1.0, observation_times=FIX,
                                     currency_pair=PAIR), FxForeignRangeAccrualAnalyticalEngine()),
    ]
    pf = FXPortfolio(portfolio_name="b6", pricing_environments={"EURUSD": env()},
                     creation_date=datetime(2026, 1, 1))
    for prod, eng in rows:
        pf.add_position(product=prod, quantity=1.0, entry_price=0.0, underlying="EURUSD", engine=eng,
                        entry_timestamp=datetime(2026, 1, 1))
    return pf


def _plain(v):
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.isoformat()
    if isinstance(v, (np.floating, np.bool_)):
        return v.item()
    return v


def backtest(days, greeks):
    rng = np.random.default_rng(3)
    idx = pd.date_range(V, periods=days, freq="D")
    spot = 1.20 * np.exp(np.cumsum(rng.normal(0.0, 0.005, days)))
    path = pd.DataFrame({"EURUSD_spot": spot, "EURUSD_vol": 0.10, "EURUSD_dom_rate": 0.05,
                         "EURUSD_for_rate": 0.03}, index=idx)
    res = FXBacktestEngine(FXBacktestConfig(portfolio=book(), market_path=path, strategy=FXDeltaNeutralStrategy(),
                                            calculate_greeks=greeks)).run()
    return {"rows": [{k: _plain(v) for k, v in r.items()} for r in res.rows],
            "initial_value": res.initial_value, "final_value": res.final_value, "num_hedges": res.num_hedges,
            "total_hedge_pnl": res.total_hedge_pnl}


def dynamic(start):
    steps = [DayStep(day_index=i, changes=[ParameterChange("spot", StressType.PERCENTAGE, 0.001)])
             for i in range(30)]
    r = FXDynamicScenarioEngine(FXDynamicScenarioConfig(calculate_greeks=True)).run(
        book(), DayPath(name="b6", steps=steps, start_date=start))
    return {"baseline": r.baseline_value, "final": r.final_value, "total_pnl": r.total_pnl,
            "days": [{"portfolio_value": d.portfolio_value, "daily_pnl": d.daily_pnl,
                      "cumulative_pnl": d.cumulative_pnl, "net_pnl": d.net_pnl,
                      "greeks": {k: _plain(v) for k, v in d.greeks.items()},
                      "paid_cash": d.paid_cash, "realized_cash": d.realized_cash,
                      "pending_receivable_pv": d.pending_receivable_pv,
                      "n_lifecycle_events": len(d.lifecycle_events)} for d in r.day_results]}


CASES = {
    "backtest_92_no_greeks": lambda: backtest(92, False),
    "backtest_20_greeks": lambda: backtest(20, True),
    "dynamic_dated": lambda: dynamic(V),
    "dynamic_undated": lambda: dynamic(None),
}


def _same(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def _diff(path, base, got, out):
    if isinstance(base, dict):
        for k, v in base.items():
            if k not in got:
                out.append(f"{path}.{k}: missing")
            else:
                _diff(f"{path}.{k}", v, got[k], out)
        for k in set(got) - set(base):
            if k in NEW_BACKTEST_KEYS and got[k] == 0.0:
                continue
            out.append(f"{path}.{k}: new key with value {got[k]!r}")
    elif isinstance(base, list):
        if len(base) != len(got):
            out.append(f"{path}: length {len(base)} -> {len(got)}")
        for i, (x, y) in enumerate(zip(base, got)):
            _diff(f"{path}[{i}]", x, y, out)
    elif not _same(base, got):
        out.append(f"{path}: {base!r} -> {got!r}")


def main():
    mode, target = sys.argv[1], sys.argv[2]
    got = json.loads(json.dumps({name: fn() for name, fn in CASES.items()}))
    if mode == "capture":
        with open(target, "w") as fh:
            json.dump(got, fh, indent=1)
        print(f"captured {len(got)} cases to {target}")
        return
    with open(target) as fh:
        base = json.load(fh)
    diffs = []
    _diff("", base, got, diffs)
    if diffs:
        print("B6 FAIL\n" + "\n".join(diffs[:200]))
        sys.exit(1)
    print("B6 OK: every stored number bitwise equal; new cash columns 0.0")


if __name__ == "__main__":
    main()
