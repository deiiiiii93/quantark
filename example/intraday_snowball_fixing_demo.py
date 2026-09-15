"""
Intraday valuation of a Snowball around its 15:00 knock-out fixing.

This script demonstrates:
1. A session calendar (SSE, fixed +08:00) and a desk variance profile
2. A dated monthly Snowball whose first five fixings are confirmed
3. Four valuations around the sixth fixing:
   - one second before it,
   - exactly at it, before the fixing is known ("before"),
   - exactly at it, with the confirmed fixing supplied ("after"),
   - thirty seconds later with no fixing supplied: the latest spot is assumed
     and the result says so (provisional)
4. The same four rows for a spot below the barrier, where the contract lives on
"""

from datetime import datetime, time, timedelta, timezone

from quantark.asset.equity.engine.quad.v2 import SnowballQuadEngineV2
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.snowball_config import BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.execution import PricingSession
from quantark.intraday import EventKind, Fixing, TradingSession, TradingSessionCalendar, VarianceProfile, resolve_context
from quantark.intraday.request import IntradayValuationRequest
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.enum.option_enums import ObservationType

SHANGHAI = timezone(timedelta(hours=8))
T0 = datetime(2026, 3, 16)


def monthly_trading_dates(calendar, start, months):
    dates = []
    for k in range(1, months + 1):
        month = (start.month - 1 + k) % 12 + 1
        year = start.year + (start.month - 1 + k) // 12
        d = datetime(year, month, min(start.day, 28))
        while not calendar.is_business_day(d):
            d += timedelta(days=1)
        dates.append(d)
    return dates


def build_snowball(calendar):
    dates = monthly_trading_dates(calendar, T0, 12)
    ko = ObservationSchedule(records=[ObservationRecord(observation_date=d, barrier=103.0) for d in dates])
    ki = ObservationSchedule(records=[ObservationRecord(observation_date=d, barrier=75.0) for d in dates])
    return SnowballOption(
        initial_price=100.0, strike=100.0, initial_date=T0, exercise_date=dates[-1],
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.12, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=ko, ki_barrier=75.0,
                                     ki_observation_type=ObservationType.DISCRETE, ki_observation_schedule=ki),
        payoff_config=PayoffConfig(rebate_rate=0.12, include_principal=False),
    )


def market(ts, spot):
    return PricingEnvironment(rate_curve=FlatRateCurve(0.03), valuation_date=ts,
                              spot_quote=SpotQuote(spot, timestamp=ts), vol_surface=FlatVolSurface(0.20),
                              div_yield=ContinuousDividendYield(0.01))


def main():
    calendar = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
    sse = TradingSessionCalendar(name="SSE", tz=SHANGHAI, calendar=calendar,
                                 sessions=(TradingSession(time(9, 30), time(11, 30)), TradingSession(time(13, 0), time(15, 0))))
    desk = VarianceProfile("desk", "1", days_per_year=244, overnight_weight=0.25,
                           session_weights=(0.35, 0.35), break_weights=(0.05,))
    product = build_snowball(calendar)
    probe = resolve_context(IntradayValuationRequest(product=product, pricing_env=market(datetime(2026, 9, 1, tzinfo=SHANGHAI), 100.0),
                                                     session_calendar=sse, variance_profile=desk))
    kos = [e for e in probe.timeline.events if e.kind is EventKind.KO]
    fixing = kos[5]
    confirmed = [Fixing(k.timestamp, 100.0) for k in kos[:5]]
    print(f"Sixth KO fixing: {fixing.timestamp.isoformat()}  barrier {fixing.barrier}  KO cash {fixing.cash:.6f}")

    engine = SnowballQuadEngineV2()
    header = f"{'row':<34}{'price':>12}{'contingent':>12}{'paid':>10}  {'provisional':<12}{'assumed':<9}{'state':<12}method"
    for spot in (103.5, 102.0):
        print(f"\nspot {spot}")
        print(header)
        rows = (
            ("1s before, before", fixing.timestamp - timedelta(seconds=1), "before", confirmed),
            ("at 15:00, before", fixing.timestamp, "before", confirmed),
            ("at 15:00, after, fixing = spot", fixing.timestamp, "after", confirmed + [Fixing(fixing.timestamp, spot)]),
            ("15:00:30, after, no fixing", fixing.timestamp + timedelta(seconds=30), "after", confirmed),
        )
        with PricingSession() as session:
            for label, ts, phase, fixings in rows:
                res = session.value_intraday(engine, product, market(ts, spot), session_calendar=sse,
                                             variance_profile=desk, fixings=fixings, event_phase=phase)
                state = "knocked out" if res.lifecycle["knocked_out"] else ("alive" if res.lifecycle["alive"] else "ended")
                assumed = ",".join(f"{a.assumed_value:g}" for a in res.assumptions) or "-"
                print(f"{label:<34}{res.price:>12.6f}{res.contingent_pv:>12.6f}{res.paid_cash:>10.6f}  "
                      f"{str(res.provisional):<12}{assumed:<9}{state:<12}{res.method}")
    print("\nRows 3 and 4 agree on the lifecycle and the paid cash: the assumed fixing equals the spot. Row 4 is\n"
          "flagged provisional until the real fixing arrives; supplying it replays history from the checkpoint.")


if __name__ == "__main__":
    main()
