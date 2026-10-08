"""Item 9: an annualized range accrual's coupon shrinks as the contract ages.

``RangeAccrualOption.get_year_fraction`` (equity) and ``FxRangeAccrualOption.get_year_fraction`` (FX) return the
REMAINING maturity (``maturity``, or valuation -> exercise/expiry on a dated contract), while the accrual ratio they
multiply is the WHOLE contract's (past + future fixings over the total weight).

The control: a one-year contract, 10% annualized, notional 100, twelve monthly dated fixings, a band so wide that every
fixing is in range (lower 1, upper 1e6; spot 100, vol 20%). Its contractual coupon is 100 x 10% x 1 x YF(initial,
expiry) = 10.0 whatever the valuation date, so its price is 10 x DF(valuation -> payment). Priced at inception and
after 91, 182, 273 and 364 days with every past fixing recorded in range.

Also the numeric (``maturity``) contract under ``time_shift``: maturity shrinks, the fixing times do not move, and the
coupon follows the maturity.

Run from the repo root:  env PYTHONPATH=$PWD:$PWD/test <venv>/bin/python <this file>
"""
import copy
import math
from datetime import datetime, timedelta

from quantark.asset.equity.engine.analytical import RangeAccrualAnalyticalEngine
from quantark.asset.equity.product.option.range_accrual_config import (
    RangeAccrualConfig, RangeAccrualObservationRecord,
)
from quantark.asset.equity.product.option.range_accrual_option import RangeAccrualOption
from quantark.asset.fx.engine.analytical import FxRangeAccrualAnalyticalEngine
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import FxRangeAccrualConfig, FxRangeAccrualOption
from quantark.asset.fx.product.option.fx_range_accrual_config import FxRangeAccrualObservationRecord
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import FxPricingEnvironment, PricingEnvironment
from quantark.util.calendar.day_counter import DayCountConvention as DC

T0 = datetime(2024, 1, 2)
EXPIRY = T0 + timedelta(days=365)
FIX_DAYS = [round(365 * k / 12) for k in range(1, 13)]
R, RATE = 0.03, 0.10
AGES = (0, 91, 182, 273, 364)


def equity_product(age_days):
    records = [RangeAccrualObservationRecord(observation_date=T0 + timedelta(days=d),
                                             observed_in_range=True if d <= age_days else None) for d in FIX_DAYS]
    # a fixing ON the valuation date resolves to t = 0, which the product files as past
    return RangeAccrualOption(initial_price=100.0, range_config=RangeAccrualConfig(
        upper_barrier=1e6, lower_barrier=1.0, accrual_rate=RATE, is_rate_annualized=True),
        exercise_date=EXPIRY, initial_date=T0, observation_records=records)


def fx_product(age_days):
    records = [FxRangeAccrualObservationRecord(observation_date=T0 + timedelta(days=d),
                                               observed_in_range=True if d <= age_days else None) for d in FIX_DAYS]
    return FxRangeAccrualOption(notional=100.0, range_config=FxRangeAccrualConfig(
        upper_barrier=1e6, lower_barrier=1e-3, accrual_rate=RATE, is_rate_annualized=True),
        currency_pair=CurrencyPair("EUR", "USD"), expiry_date=EXPIRY, observation_records=records)


def main():
    print("equity RangeAccrualOption, dated, annualized 10%, every fixing in range; contractual coupon 10.0")
    print(f"{'age(d)':>6} {'year_frac':>10} {'price':>10} {'10*DF':>10} {'paid/10':>8}")
    for age in AGES:
        val = T0 + timedelta(days=age)
        env = PricingEnvironment(spot_quote=SpotQuote(spot=100.0, asset_name="X"), vol_surface=FlatVolSurface(0.2),
                                 rate_curve=FlatRateCurve(R), valuation_date=val)
        p = equity_product(age)
        yf = p.get_year_fraction(env)
        price = RangeAccrualAnalyticalEngine().price(p, env)
        df = math.exp(-R * (365 - age) / 365)
        print(f"{age:>6} {yf:>10.6f} {price:>10.6f} {10 * df:>10.6f} {price / (10 * df):>8.4f}")

    print("\nFX FxRangeAccrualOption, dated, annualized 10%, every fixing in range; contractual coupon 10.0")
    print(f"{'age(d)':>6} {'year_frac':>10} {'price':>10} {'10*DF':>10} {'paid/10':>8}")
    for age in AGES:
        val = T0 + timedelta(days=age)
        env = FxPricingEnvironment(valuation_date=val, spot_quote=SpotQuote(spot=1.10),
                                   domestic_curve=FlatRateCurve(rate=R), foreign_curve=FlatRateCurve(rate=0.01),
                                   vol_surface=FlatVolSurface(volatility=0.08))
        p = fx_product(age)
        yf = p.get_year_fraction(env)
        price = FxRangeAccrualAnalyticalEngine().price(p, env)
        df = math.exp(-R * (365 - age) / 365)
        print(f"{age:>6} {yf:>10.6f} {price:>10.6f} {10 * df:>10.6f} {price / (10 * df):>8.4f}")

    print("\nequity, the same dated contract with tenor=1.0 and the config's day count ACT/360, aged 182 days")
    val = T0 + timedelta(days=182)
    env = PricingEnvironment(spot_quote=SpotQuote(spot=100.0, asset_name="X"), vol_surface=FlatVolSurface(0.2),
                             rate_curve=FlatRateCurve(R), valuation_date=val)
    p = equity_product(182)
    p.tenor = 1.0
    p.range_config = RangeAccrualConfig(upper_barrier=1e6, lower_barrier=1.0, accrual_rate=RATE,
                                        is_rate_annualized=True, day_count_convention=DC.ACT_360)
    print(f"  year_fraction={p.get_year_fraction(env):.6f} (tenor 1.0 and ACT/360 365/360={365 / 360:.6f} unread)")

    print("\nequity, numeric maturity=1.0, 12 monthly observation_times, time_shift(0.5)")
    p = RangeAccrualOption(initial_price=100.0, range_config=RangeAccrualConfig(
        upper_barrier=1e6, lower_barrier=1.0, accrual_rate=RATE, is_rate_annualized=True),
        maturity=1.0, observation_times=[k / 12 for k in range(1, 13)])
    env = PricingEnvironment(spot_quote=SpotQuote(spot=100.0, asset_name="X"), vol_surface=FlatVolSurface(0.2),
                             rate_curve=FlatRateCurve(R), valuation_date=T0)
    aged = copy.deepcopy(p)
    aged.time_shift(0.5, T0 + timedelta(days=182), copy.deepcopy(env))
    for label, q in (("inception", p), ("aged 0.5", aged)):
        times = [round(r.observation_time, 4) for r in q.get_observation_records()]
        print(f"  {label:<10} maturity={q.maturity:.4f} year_fraction={q.get_year_fraction(env):.4f} "
              f"payoff(all in range)={q.get_payoff(100.0, 12.0, 12.0, env):.4f} first fixings={times[:3]}")


if __name__ == "__main__":
    main()
