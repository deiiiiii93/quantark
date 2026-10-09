"""W4b rev 2 probe: which FX products does the FX backtest / FX dynamic scenario freeze (M5)?

Runs the REAL FXBacktestEngine and FXDynamicScenarioEngine over a flat market (constant spot / vol / rates) for 92
calendar days and reports each position's value on day 0 and day 91. A numeric product that does not age keeps its
day-0 value to the bit; a dated product (control) moves (theta).
"""
import warnings
from datetime import datetime, timedelta

import pandas as pd

from quantark.asset.fx.engine.analytical import (
    FxDigitalOptionAnalyticalEngine, GarmanKohlhagenEngine, GarmanKohlhagenQuantoEngine,
)
from quantark.asset.fx.engine.analytical.fx_range_accrual_engine import FxRangeAccrualAnalyticalEngine
from quantark.asset.fx.engine.analytical import (
    FxForeignRangeAccrualAnalyticalEngine, FxQuantoDigitalAnalyticalEngine, FxQuantoRangeAccrualAnalyticalEngine,
)
from quantark.asset.fx.product.option import (
    FxForeignRangeAccrualOption, FxOneTouchOption, FxQuantoDigitalOption, FxQuantoRangeAccrualOption,
)
from quantark.asset.fx.engine.mc import (
    FxBarrierMCEngine, FxSharkfinMCEngine, FxTargetRedemptionNoteMCEngine, FxTarnForwardMCEngine,
)
from quantark.asset.fx.engine.mc.fx_mc_params import FxMCParams
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import (
    FxBarrierOption, FxDigitalOption, FxQuantoVanillaOption, FxRangeAccrualConfig, FxRangeAccrualOption,
    FxSharkfinOption, FxTargetRedemptionForward, FxTargetRedemptionNote, FxVanillaOption,
)
from quantark.backtest.fx import FXBacktestConfig, FXBacktestEngine
from quantark.backtest.strategy.fx_delta_neutral_strategy import FXDeltaNeutralStrategy
from quantark.dynamicscenario import FXDynamicScenarioConfig, FXDynamicScenarioEngine
from quantark.dynamicscenario.path.day_path import DayPath, DayStep
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.portfolio.fx import FXPortfolio
from quantark.priceenv import FxPricingEnvironment
from quantark.priceenv.fx_pricing_environment import FxQuantoMarketData, QuantoConversionOrientation
from quantark.util.enum import FxBarrierType, OptionType

warnings.simplefilter("ignore")
PAIR = CurrencyPair("EUR", "USD")
V = datetime(2026, 6, 15)
FIX = [k / 12 for k in range(1, 13)]


def env():
    return FxPricingEnvironment(
        valuation_date=V, spot_quote=SpotQuote(spot=1.20), domestic_curve=FlatRateCurve(rate=0.05),
        foreign_curve=FlatRateCurve(rate=0.03), vol_surface=FlatVolSurface(volatility=0.10),
        quanto=FxQuantoMarketData(settlement_curve=FlatRateCurve(rate=0.02), quanto_vol=0.08, correlation=0.3,
                                  conversion_orientation=QuantoConversionOrientation.SETTLEMENT_PER_DOMESTIC))


def mc(seed=7, n=20_000, steps=None):
    return FxMCParams(num_paths=n, seed=seed, **({"time_steps": steps} if steps else {}))


def book():
    rows = [
        ("vanilla numeric", FxVanillaOption(strike=1.20, option_type=OptionType.CALL, maturity=1.0, notional_foreign=1.0,
                                            currency_pair=PAIR), GarmanKohlhagenEngine()),
        ("vanilla DATED (control)", FxVanillaOption(strike=1.20, option_type=OptionType.CALL,
                                                    expiry_date=V + timedelta(days=365), currency_pair=PAIR, notional_foreign=1.0),
         GarmanKohlhagenEngine()),
        ("digital numeric", FxDigitalOption(strike=1.20, option_type=OptionType.CALL, payout=1.0, maturity=1.0,
                                            currency_pair=PAIR), FxDigitalOptionAnalyticalEngine()),
        ("quanto vanilla numeric", FxQuantoVanillaOption(strike=1.20, option_type=OptionType.CALL,
                                                         quanto_fx_rate=1.0, maturity=1.0, currency_pair=PAIR, notional_foreign=1.0),
         GarmanKohlhagenQuantoEngine()),
        ("barrier numeric", FxBarrierOption(strike=1.20, barrier=1.35, is_up=True,
                                            knock_type=FxBarrierType.KNOCK_OUT, option_type=OptionType.CALL,
                                            maturity=1.0, currency_pair=PAIR), FxBarrierMCEngine(mc(steps=24))),
        ("sharkfin numeric", FxSharkfinOption(strike=1.20, barrier=1.35, is_up=True, option_type=OptionType.CALL,
                                              maturity=1.0, currency_pair=PAIR), FxSharkfinMCEngine(mc(steps=24))),
        ("range accrual numeric", FxRangeAccrualOption(
            notional=1.0, range_config=FxRangeAccrualConfig(upper_barrier=1.30, lower_barrier=1.10,
                                                            accrual_rate=0.04),
            maturity=1.0, observation_times=FIX, currency_pair=PAIR), FxRangeAccrualAnalyticalEngine()),
        ("quanto digital numeric", FxQuantoDigitalOption(strike=1.20, option_type=OptionType.CALL, payout=1.0,
                                                         quanto_fx_rate=1.0, maturity=1.0, currency_pair=PAIR),
         FxQuantoDigitalAnalyticalEngine()),
        ("quanto range acc. numeric", FxQuantoRangeAccrualOption(
            notional=1.0, range_config=FxRangeAccrualConfig(upper_barrier=1.30, lower_barrier=1.10,
                                                            accrual_rate=0.04),
            quanto_fx_rate=1.0, maturity=1.0, observation_times=FIX, currency_pair=PAIR),
         FxQuantoRangeAccrualAnalyticalEngine()),
        ("foreign range acc. numeric", FxForeignRangeAccrualOption(
            notional=1.0, range_config=FxRangeAccrualConfig(upper_barrier=1.30, lower_barrier=1.10,
                                                            accrual_rate=0.04),
            maturity=1.0, observation_times=FIX, currency_pair=PAIR), FxForeignRangeAccrualAnalyticalEngine()),
        ("TARN numeric", FxTargetRedemptionNote(fixing_times=FIX, coupon_rate=0.12, notional=100.0, strike=1.10,
                                                target=3.0, currency_pair=PAIR),
         FxTargetRedemptionNoteMCEngine(mc())),
        ("TARF numeric", FxTargetRedemptionForward(strike=1.15, fixing_times=FIX, target=0.05, currency_pair=PAIR),
         FxTarnForwardMCEngine(mc())),
    ]
    pf = FXPortfolio(portfolio_name="m5", pricing_environments={"EURUSD": env()})
    ids = {}
    for label, prod, eng in rows:
        p = pf.add_position(product=prod, quantity=1.0, entry_price=0.0, underlying="EURUSD", engine=eng)
        ids[p.position_id] = label
    return pf, ids


def per_position_values(pf):
    e = pf.pricing_environments["EURUSD"]
    return {pid: pos.get_market_value(e) for pid, pos in pf.positions.items()}


def one_touch_structural():
    """FxOneTouchOption prices only on a Vanna-Volga surface; check the clock it would be priced on instead."""
    ot = FxOneTouchOption(barrier=1.30, is_up=True, maturity=1.0, currency_pair=PAIR)
    e0 = env()
    e91 = env()
    e91.valuation_date = V + timedelta(days=91)
    print(f"one-touch numeric (structural, VV surface needed to price): get_maturity "
          f"{ot.get_maturity(e0):.6f} -> {ot.get_maturity(e91):.6f} after the env date moves 91 days")


def main():
    dates = pd.date_range(V, periods=92, freq="D")
    path = pd.DataFrame({"EURUSD_spot": 1.20, "EURUSD_vol": 0.10, "EURUSD_dom_rate": 0.05,
                         "EURUSD_for_rate": 0.03}, index=dates)

    # --- FX backtest: hook into the engine's own row application to read per-position values each day.
    pf, ids = book()
    captured = {}
    orig = FXBacktestEngine._apply_row

    def spy(portfolio, pairs, row, ts):
        orig(portfolio, pairs, row, ts)
        captured[pd.Timestamp(ts)] = (portfolio.pricing_environments["EURUSD"].valuation_date,
                                      per_position_values(portfolio))

    FXBacktestEngine._apply_row = staticmethod(spy)
    try:
        res = FXBacktestEngine(FXBacktestConfig(portfolio=pf, market_path=path,
                                                strategy=FXDeltaNeutralStrategy(), calculate_greeks=False)).run()
    finally:
        FXBacktestEngine._apply_row = staticmethod(orig)
    d0, d91 = dates[0], dates[91]
    print("FX backtest, flat market, valuation_date", captured[d0][0].date(), "->", captured[d91][0].date())
    for pid, label in ids.items():
        a, b = captured[d0][1][pid], captured[d91][1][pid]
        print(f"  {label:26s} day0 {a:14.6f}  day91 {b:14.6f}  frozen(bitwise) {a == b}")
    print(f"  portfolio_pnl on day 91: {res.rows[-1]['portfolio_pnl']:.6f}")

    # --- FX dynamic scenario: a dated DayPath of 92 empty steps, and an undated one.
    for start in (V, None):
        pf, ids = book()
        steps = [DayStep(day_index=i, changes=[]) for i in range(92)]
        dp = DayPath(name="flat", steps=steps, start_date=start)
        eng = FXDynamicScenarioEngine(FXDynamicScenarioConfig(calculate_greeks=False))
        seen = {}
        orig_val = FXPortfolio.get_portfolio_value

        def spy_val(self, as_of_date=None):
            e = self.pricing_environments["EURUSD"]
            seen.setdefault("dates", []).append(e.valuation_date)
            seen.setdefault("vals", []).append(per_position_values(self))
            return orig_val(self, as_of_date)

        FXPortfolio.get_portfolio_value = spy_val
        try:
            r = eng.run(pf, dp)
        finally:
            FXPortfolio.get_portfolio_value = orig_val
        # calls: baseline, day0..day91, final
        base_d, day91_d = seen["dates"][1], seen["dates"][92]
        print(f"FX dynamic scenario, start_date={start and start.date()}: day0 date {base_d.date()} "
              f"day91 date {day91_d.date()}; total_pnl {r.total_pnl:.6f}")
        for pid, label in ids.items():
            a, b = seen["vals"][1][pid], seen["vals"][92][pid]
            print(f"  {label:26s} day0 {a:14.6f}  day91 {b:14.6f}  frozen(bitwise) {a == b}")


if __name__ == "__main__":
    main()
    one_touch_structural()
