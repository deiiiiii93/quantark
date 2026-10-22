"""
Intraday Greeks of a Snowball around its 15:00 knock-out fixing.

This script demonstrates:
1. Point Greeks (derivatives of the resolved price function) next to desk-bump Greeks
   (the configured finite moves) one hour and one second before the fixing
2. Exactly at the fixing under "before" with the spot on the barrier: point delta and gamma
   are undefined (the price function jumps there) — never a number
3. Thirty seconds after an assumed knock-out: zero delta, but a non-zero rho from the
   delayed KO payment
4. Local intraday theta with a declared step, clamped to land on the fixing
5. roll_through_events: a scenario valuation after an outcome for the fixing
Every Greek prints its status and reason columns.
"""

from datetime import datetime, time, timedelta, timezone

from quantark.asset.equity.engine.quad.v2 import SnowballQuadEngineV2
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.snowball_config import BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.intraday import (EventKind, Fixing, TradingSession, TradingSessionCalendar, VarianceProfile, resolve_context,
                               roll_through_events, value_intraday)
from quantark.intraday.request import IntradayValuationRequest
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.enum.option_enums import ObservationType

SHANGHAI = timezone(timedelta(hours=8))
T0 = datetime(2026, 3, 16)
GREEKS = ("delta", "gamma", "vega", "rho", "theta")


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


def build_snowball(calendar, settlement_lag_days=0):
    dates = monthly_trading_dates(calendar, T0, 12)
    ko = ObservationSchedule(records=[ObservationRecord(observation_date=d, barrier=103.0,
                                                        settlement_date=d + timedelta(days=settlement_lag_days) if settlement_lag_days else None)
                                      for d in dates])
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


def show(label, result):
    print(f"\n{label}: price {result.price:.6f}  provisional={result.provisional}  method={result.method}")
    print(f"  {'greek':<8}{'convention':<11}{'status':<12}{'value':>14}  {'unit':<34}reason")
    for g in result.greeks:
        value = f"{g.value:14.6f}" if g.value is not None else f"{'-':>14}"
        print(f"  {g.name:<8}{g.convention:<11}{g.status:<12}{value}  {g.unit:<34}{g.reason or ''}")


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
    confirmed = tuple(Fixing(k.timestamp, 100.0) for k in kos[:5])
    engine = SnowballQuadEngineV2()

    def request(ts, spot, convention, **kw):
        return IntradayValuationRequest(product=kw.pop("product", product), pricing_env=market(ts, spot), session_calendar=sse,
                                        variance_profile=desk, fixings=kw.pop("fixings", confirmed), greeks=GREEKS,
                                        greek_convention=convention, **kw)

    for label, ts in (("one hour before the fixing", fixing.timestamp - timedelta(hours=1)),
                      ("one second before the fixing", fixing.timestamp - timedelta(seconds=1))):
        for convention in ("point", "desk_bump"):
            show(f"{label}, spot 102.5, {convention}", value_intraday(engine, request(ts, 102.5, convention,
                                                                                      theta_step=timedelta(minutes=5),
                                                                                      theta_unit="minute")))

    show("at 15:00 under 'before', spot exactly on the 103 barrier, point",
         value_intraday(engine, request(fixing.timestamp, 103.0, "point", event_phase="before")))

    delayed = build_snowball(calendar, settlement_lag_days=5)
    show("15:00:30, no fixing supplied, spot 104 (assumed KO, payment in 5 days), desk_bump",
         value_intraday(engine, request(fixing.timestamp + timedelta(seconds=30), 104.0, "desk_bump", product=delayed,
                                        event_phase="after")))

    near = value_intraday(engine, request(fixing.timestamp - timedelta(seconds=15), 102.5, "point",
                                          theta_step=timedelta(minutes=1), theta_unit="minute"))
    print(f"\nTheta 15 s before the fixing with a 1-minute step: step taken {near.numerical['theta_step_actual_s']:.0f} s, "
          f"adjusted={near.numerical['theta_adjusted']} ({near.numerical['theta_side']}), "
          f"theta {near.greek('theta').value:.6f} {near.greek('theta').unit}")

    base = IntradayValuationRequest(product=product, pricing_env=market(fixing.timestamp - timedelta(minutes=1), 102.5),
                                    session_calendar=sse, variance_profile=desk, fixings=confirmed)
    for outcome in (104.0, 101.0):
        scenario = roll_through_events(engine, base, fixing.timestamp + timedelta(minutes=1),
                                       outcomes=(Fixing(fixing.timestamp, outcome),))
        state = "knocked out" if scenario.lifecycle["knocked_out"] else "alive"
        print(f"roll_through_events to 15:01 with fixing {outcome}: price {scenario.price:.6f} "
              f"(paid {scenario.paid_cash:.6f}), {state}, provisional={scenario.provisional}")


if __name__ == "__main__":
    main()
