"""Item 8: BaseEquityOption.get_tenor passes no calendar to a BUSINESS_DAYS year fraction.

1. A dated European option with ``annualization_day_count = BUSINESS_DAYS``: get_tenor raises, with or without an
   environment that carries a calendar and a denominator (get_tenor takes ``pricing_env`` and never reads it).
2. A dated snowball with the same basis and its frozen terms (``accrual_calendar`` / ``accrual_bus_days_in_year``):
   get_contract_tenor answers from the frozen terminal; the inherited get_tenor still raises.
3. Who would notice: every ``.get_tenor(`` call site in ``quantark/`` (production) and in ``test/`` / ``example/``.

On 8753894f (before the fix) every get_tenor line raises; after it, the European reads the environment's calendar
(243/244 here) and refuses without one, and the snowball's get_tenor is its frozen get_contract_tenor.

Run from the repo root:  env PYTHONPATH=$PWD:$PWD/test <venv>/bin/python <this file>
"""
import re
from datetime import datetime
from pathlib import Path

from quantark.asset.equity.product.option.european_vanilla_option import EuropeanVanillaOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.calendar.day_counter import DayCountConvention as DC
from quantark.util.enum import OptionType

ROOT = Path(__file__).resolve().parents[4]
T0, T1 = datetime(2026, 1, 5), datetime(2027, 1, 5)
CAL = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))


def attempt(label, fn):
    try:
        print(f"  {label}: {fn()!r}")
    except Exception as exc:
        print(f"  {label}: RAISES {type(exc).__name__}: {exc}")


def main():
    env = PricingEnvironment(spot_quote=SpotQuote(spot=100.0, asset_name="X"), vol_surface=FlatVolSurface(0.2),
                             rate_curve=FlatRateCurve(0.02), valuation_date=T0, calendar=CAL, bus_days_in_year=244,
                             day_count_convention=DC.BUSINESS_DAYS)
    print("1. European, BUSINESS_DAYS, initial_date -> exercise_date")
    euro = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, exercise_date=T1, initial_date=T0,
                                 annualization_day_count=DC.BUSINESS_DAYS)
    attempt("get_tenor()", lambda: euro.get_tenor())
    attempt("get_tenor(env with calendar, 244)", lambda: euro.get_tenor(env))
    days = CAL.count_business_days(T0, T1, include_start=False, include_end=True)
    print(f"  the business-day fraction the contract means: {days}/244 = {days / 244:.12f}")

    print("2. Snowball, BUSINESS_DAYS, frozen accrual_calendar / accrual_bus_days_in_year, annualized rebate")
    ko = ObservationSchedule(records=[ObservationRecord(observation_date=T1, barrier=103.0)])
    snow = SnowballOption(initial_price=100.0, strike=100.0,
                          barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.12, ko_observation_schedule=ko),
                          payoff_config=PayoffConfig(rebate_rate=0.05),
                          accrual_config=AccrualConfig(is_annualized_rebate=True, accrual_calendar=CAL,
                                                       accrual_bus_days_in_year=244.0),
                          initial_date=T0, exercise_date=T1, annualization_day_count=DC.BUSINESS_DAYS)
    attempt("get_contract_tenor(env)", lambda: snow.get_contract_tenor(env))
    attempt("get_tenor(env)", lambda: snow.get_tenor(env))

    print("3. .get_tenor( call sites (excluding definitions)")
    for area in ("quantark", "test", "example"):
        hits = []
        for path in sorted((ROOT / area).rglob("*.py")):
            for n, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
                if re.search(r"\.get_tenor\(", line) and "def get_tenor" not in line:
                    hits.append(f"{path.relative_to(ROOT)}:{n}")
        print(f"  {area}: {len(hits)}" + "".join(f"\n    {h}" for h in hits))


if __name__ == "__main__":
    main()
