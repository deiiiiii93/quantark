"""W4b rev 2 probe: the clocks a numeric product and its dated twin age by.

C1  FX theta (BaseFxEngine._fdm_theta, called directly): numeric shifts days/365, dated moves the env date by calendar days. Numeric vs
    dated vanilla theta under a CALENDAR_DAYS env and a BUSINESS_DAYS env (TARGET calendar), on a Friday.
C2  Additivity: the sum of daily year fractions over 91 days vs the direct year fraction, per convention.
C3  Dated-twin mapping of times: yf(V+k, D) vs yf(V, D) - yf(V, V+k) for a monthly schedule, k = 1..91 (ULPs).
C4  Contract accrual day count: ACT_365 period factors vs CALENDAR_DAYS factors on a monthly date schedule.
"""
from datetime import datetime, timedelta

from quantark.asset.fx.engine.analytical import GarmanKohlhagenEngine
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import FxVanillaOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import FxPricingEnvironment
from quantark.util.calendar import calculate_year_fraction
from quantark.util.calendar.business_calendar import CalendarType, create_calendar
from quantark.util.calendar.day_counter import DayCountConvention as DC
from quantark.util.enum import OptionType

PAIR = CurrencyPair("EUR", "USD")
V = datetime(2026, 6, 12)  # a Friday
TGT = create_calendar(CalendarType.TARGET, year_range=(2026, 2028))


def env(dc, cal=None):
    return FxPricingEnvironment(valuation_date=V, spot_quote=SpotQuote(spot=1.20),
                                domestic_curve=FlatRateCurve(rate=0.05), foreign_curve=FlatRateCurve(rate=0.03),
                                vol_surface=FlatVolSurface(volatility=0.10), day_count_convention=dc, calendar=cal)


def yf(a, b, dc, cal=None):
    if a == b:
        return 0.0
    return calculate_year_fraction(a, b, dc, 252, calendar=cal)


def c1():
    print("C1 theta (1 day) numeric vs dated vanilla, expiry V+365d, valuation Friday", V.date())
    eng = GarmanKohlhagenEngine()
    for dc, cal in ((DC.CALENDAR_DAYS, None), (DC.BUSINESS_DAYS, TGT)):
        e = env(dc, cal)
        dated = FxVanillaOption(strike=1.2, option_type=OptionType.CALL, expiry_date=V + timedelta(days=365),
                                currency_pair=PAIR, notional_foreign=1.0)
        T = dated.get_maturity(e)
        num = FxVanillaOption(strike=1.2, option_type=OptionType.CALL, maturity=T, currency_pair=PAIR,
                              notional_foreign=1.0)
        pn, pd_ = eng.price(num, e), eng.price(dated, e)
        # the shared finite-difference theta every MC FX engine (TARN/TARF included) uses
        tn, td = eng._fdm_theta(num, e, pn), eng._fdm_theta(dated, e, pd_)
        print(f"  {dc.name:13s} T={T:.6f}  price numeric {pn:.12f} dated {pd_:.12f}  "
              f"theta numeric {tn:.3e} dated {td:.3e}")


def c2():
    print("C2 sum of 91 daily year fractions vs the direct one")
    for dc, cal in ((DC.CALENDAR_DAYS, None), (DC.BUSINESS_DAYS, TGT), (DC.ACT_365, None), (DC.ACT_360, None),
                    (DC.THIRTY_360_US, None), (DC.ACT_ACT_ISDA, None)):
        s = 0.0
        for k in range(91):
            s += yf(V + timedelta(days=k), V + timedelta(days=k + 1), dc, cal)
        d = yf(V, V + timedelta(days=91), dc, cal)
        print(f"  {dc.name:15s} sum {s:.17f} direct {d:.17f} diff {s - d:+.3e}")


def c3():
    print("C3 dated time yf(V+k, D) vs numeric twin aged by sequential daily shifts (CALENDAR_DAYS)")
    dates = [datetime(2026, 7, 15) + timedelta(days=30 * i) for i in range(12)]
    worst, nbit = 0.0, 0
    num = [yf(V, d, DC.CALENDAR_DAYS) for d in dates]
    for k in range(1, 92):
        num = [t - 1 / 365 for t in num]
        vk = V + timedelta(days=k)
        for t, d in zip(num, dates):
            if d <= vk:
                continue
            dated = yf(vk, d, DC.CALENDAR_DAYS)
            worst = max(worst, abs(t - dated))
            nbit += t != dated
    print(f"  max |numeric - dated| over 91 days = {worst:.3e}; non-bitwise (day, fixing) pairs: {nbit}")


def c4():
    print("C4 accrual factors on monthly dates: ACT_365 vs CALENDAR_DAYS")
    dates = [datetime(2026, 6, 15)] + [datetime(2026 + (6 + m) // 12, (6 + m) % 12 + 1, 15) for m in range(12)]
    diffs = [yf(a, b, DC.ACT_365) - yf(a, b, DC.CALENDAR_DAYS) for a, b in zip(dates, dates[1:])]
    print(f"  max |ACT_365 - CALENDAR_DAYS| = {max(abs(x) for x in diffs):.3e}; "
          f"first factors {[round(yf(a, b, DC.ACT_365), 6) for a, b in list(zip(dates, dates[1:]))[:3]]}")


if __name__ == "__main__":
    c1()
    c2()
    c3()
    c4()
