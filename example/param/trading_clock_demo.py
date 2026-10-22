"""Trading-clock volatility demo: a term-structure snowball across CNY.

Docs: docs/trading-clock.md
Spec: docs/superpowers/specs/2026-09-01-trading-clock-vol-design.md

Three setups, one desk reality (sigma quoted per sqrt-trading-year at
trading-day pillars, r/q calendar ACT/365):

  A  NAIVE      calendar axis, surface used UNWRAPPED (axis misread)
  B  CORRECT    calendar axis + TradingClockVolSurface (the new default)
  C  CORRECT    trading axis + TradingClockRateCurve/DividendYield (opt-in)

Sections:
  A  the two clocks around the CNY 2026 closure (Feb 14-23)
  B  per-step vols the engines consume (exact zeros on holidays)
  C  the trading axis' mirror image (holiday carry in one fwd-rate tick)
  D  short-dated Europeans: the naive error flips sign at the holiday
  E  the snowball: model error vs discretization vs coupon-basis

Run:
  python example/trading_clock_demo.py
"""
from datetime import datetime, timedelta

import numpy as np
from dateutil.relativedelta import relativedelta

from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import SpotQuote
from quantark.param.div.dividend_yield import ContinuousDividendYield
from quantark.param.div.trading_clock_yield import TradingClockDividendYield
from quantark.param.rrf.rate_curve import FlatRateCurve
from quantark.param.rrf.trading_clock_curve import TradingClockRateCurve
from quantark.param.vol.vol_surface import TermStructureVolSurface
from quantark.param.vol.trading_clock_surface import TradingClockVolSurface
from quantark.priceenv import PricingEnvironment
from quantark.priceenv.term_sampling import TermCoefficients
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.calendar.trading_clock import BusinessTimeMap, TradingClock
from quantark.util.enum import ObservationType, OptionType

ANCHOR = datetime(2026, 1, 15)
R, Q, SPOT, D = 0.02, 0.01, 100.0, 244

cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
tmap = BusinessTimeMap(TradingClock(cal, D), ANCHOR, ANCHOR + timedelta(days=400))

# Desk quote: sigma per sqrt-trading-year at TRADING-DAY pillars (n_td/244),
# steep short end -- the shape a pre-holiday market actually shows.
PILLAR_TD_DAYS = [10, 30, 60, 90, 130]
PILLAR_VOLS = [0.28, 0.25, 0.23, 0.215, 0.205]
inner = TermStructureVolSurface(
    times=[n / D for n in PILLAR_TD_DAYS], vols=PILLAR_VOLS
)

env_naive = PricingEnvironment(
    rate_curve=FlatRateCurve(R), valuation_date=ANCHOR,
    spot_quote=SpotQuote(SPOT), vol_surface=inner,          # axis misread!
    div_yield=ContinuousDividendYield(Q),
)
env_cal = PricingEnvironment(
    rate_curve=FlatRateCurve(R), valuation_date=ANCHOR,
    spot_quote=SpotQuote(SPOT),
    vol_surface=TradingClockVolSurface(inner, tmap),        # correct, default
    div_yield=ContinuousDividendYield(Q),
)
env_td = PricingEnvironment(
    rate_curve=TradingClockRateCurve(FlatRateCurve(R), tmap),
    valuation_date=ANCHOR, spot_quote=SpotQuote(SPOT),
    vol_surface=inner,                                      # native on this axis
    div_yield=TradingClockDividendYield(ContinuousDividendYield(Q), tmap),
)

# ---------------------------------------------------------------- section A
print("=" * 78)
print("A. THE TWO CLOCKS AROUND CNY 2026  (anchor", ANCHOR.date(), ")")
print("=" * 78)
d, rows, in_block, shown = ANCHOR + timedelta(days=25), [], False, 0
while shown < 14:
    t_cal = (d - ANCHOR).days / 365.0
    t_td = tmap.to_trading(t_cal)
    is_td = cal.is_business_day(d)
    if not is_td and d.weekday() < 5:
        in_block = True
    if in_block:
        rows.append((d, is_td, t_cal, t_td))
        shown += 1
    d += timedelta(days=1)
first_holiday = rows[0][0]
print(f"{'date':>12} {'day':>4} {'trading?':>9} {'t_cal':>9} {'t_td':>9} {'dt_td':>8}")
prev_td = None
for d_, is_td, t_cal, t_td in rows:
    dtd = "" if prev_td is None else f"{t_td - prev_td:8.5f}"
    print(f"{str(d_.date()):>12} {d_.strftime('%a'):>4} {('yes' if is_td else 'NO'):>9}"
          f" {t_cal:9.5f} {t_td:9.5f} {dtd:>8}")
    prev_td = t_td

# ---------------------------------------------------------------- section B
print()
print("=" * 78)
print("B. STEP VOLS ON A DAILY CALENDAR GRID  (TermCoefficients, ref K=100)")
print("   naive = unwrapped surface (axis misread) vs correct = wrapper")
print("=" * 78)
n_days = 45
t_grid = np.array([(k) / 365.0 for k in range(n_days + 1)], dtype=float)
tc_naive = TermCoefficients.from_env(env_naive, t_grid, ref_strike=100.0)
tc_cal = TermCoefficients.from_env(env_cal, t_grid, ref_strike=100.0)
print(f"{'interval end':>13} {'trading?':>9} {'naive sv':>10} {'correct sv':>11}")
start = (first_holiday - ANCHOR).days - 4
for k in range(start, min(start + 12, n_days)):
    d_ = ANCHOR + timedelta(days=k + 1)
    is_td = cal.is_business_day(d_)
    print(f"{str(d_.date()):>13} {('yes' if is_td else 'NO'):>9}"
          f" {tc_naive.step_vols[k]:10.4f} {tc_cal.step_vols[k]:11.4f}")
zero_steps = int(np.sum(tc_cal.step_vols == 0.0))
print(f"\ncorrect axis: {zero_steps} of {n_days} daily steps carry EXACTLY 0.0 vol")
print(f"naive axis:   min step vol {tc_naive.step_vols.min():.4f} "
      f"(never zero -- holiday variance invented)")

# ---------------------------------------------------------------- section C
print()
print("=" * 78)
print("C. TRADING AXIS: WHERE THE JAGGEDNESS MOVES  (per-tick fwd rates)")
print("=" * 78)
n_ticks = 30
u_grid = np.array([k / D for k in range(n_ticks + 1)], dtype=float)
tc_td = TermCoefficients.from_env(env_td, u_grid, ref_strike=100.0)
print(f"{'tick':>5} {'ends on':>12} {'fwd rate':>9} {'step vol':>9}")
# the anchor day itself is the map's first tick when it is a trading day
td_dates, d_ = ([ANCHOR] if cal.is_business_day(ANCHOR) else []), ANCHOR
while len(td_dates) < n_ticks:
    d_ += timedelta(days=1)
    if cal.is_business_day(d_):
        td_dates.append(d_)
k0 = max(0, next(i for i, x in enumerate(td_dates) if x > first_holiday) - 3)
for k in range(k0, min(k0 + 7, n_ticks)):
    print(f"{k + 1:>5} {str(td_dates[k].date()):>12}"
          f" {tc_td.fwd_rates[k]:9.4f} {tc_td.step_vols[k]:9.4f}")
print(f"\nflat r = {R}: the CNY-crossing tick carries ~9 days of interest in "
      f"1/{D} of trading time\nstep vols on this axis stay smooth (the quote's "
      "native clock)")

# ---------------------------------------------------------------- section D
print()
print("=" * 78)
print("D. SHORT-DATED EUROPEAN OVER THE HOLIDAY  (10 trading days, K=100)")
print("=" * 78)
bs = BlackScholesEngine()
for n_td_expiry in (10, 25):
    d_, n = ANCHOR, 0
    while n < n_td_expiry:
        d_ += timedelta(days=1)
        if cal.is_business_day(d_):
            n += 1
    t_cal_e = (d_ - ANCHOR).days / 365.0
    t_td_e = tmap.to_trading(t_cal_e)
    spans = "SPANS the CNY block" if d_ > first_holiday else "before the holiday"
    p_naive = bs.price(EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=t_cal_e), env_naive)
    p_cal = bs.price(EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=t_cal_e), env_cal)
    p_td = bs.price(EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=t_td_e), env_td)
    print(f"\nexpiry {d_.date()}  ({n_td_expiry} trading days = "
          f"{(d_ - ANCHOR).days} calendar days, {spans})")
    print(f"  t_cal = {t_cal_e:.5f}   t_td = {t_td_e:.5f}")
    print(f"  A naive     (misread axis): {p_naive:9.5f}")
    print(f"  B correct   (cal + wrap):   {p_cal:9.5f}   naive error: "
          f"{100 * (p_naive / p_cal - 1):+.2f}%")
    print(f"  C correct   (trading axis): {p_td:9.5f}   B vs C rel: "
          f"{abs(p_cal / p_td - 1):.2e}")

# ---------------------------------------------------------------- section E
print()
print("=" * 78)
print("E. THE SNOWBALL  (6M, monthly KO 103, daily KI 75, PDE fast)")
print("=" * 78)
ko_dates = []
for k in range(1, 7):
    d2 = ANCHOR + relativedelta(months=k)
    while not cal.is_business_day(d2):
        d2 += timedelta(days=1)
    ko_dates.append(d2)
ko_dates = sorted(set(ko_dates))
maturity_date = ko_dates[-1]
ki_dates, d2 = [], ANCHOR + timedelta(days=1)
while d2 <= maturity_date:
    if cal.is_business_day(d2):
        ki_dates.append(d2)
    d2 += timedelta(days=1)

def t_cal_of(x):
    return (x - ANCHOR).days / 365.0

def t_td_of(x):
    return cal.count_business_days(ANCHOR, x, include_start=False,
                                   include_end=True) / float(D)

def snowball(times_of, ko_rate):
    kt = [times_of(x) for x in ko_dates]
    return SnowballOption(
        initial_price=100.0, strike=100.0,
        barrier_config=BarrierConfig(
            ko_barrier=103.0, ko_rate=ko_rate,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=kt,
            ki_barrier=75.0,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_dates=[times_of(x) for x in ki_dates],
            ki_continuous=False,
        ),
        contract_multiplier=1.0, maturity=kt[-1], is_reverse=False,
    )

def pde_price(product, env, spot=None):
    if spot is not None:
        env = PricingEnvironment(
            rate_curve=env.rate_curve, valuation_date=env.valuation_date,
            spot_quote=SpotQuote(spot), vol_surface=env.vol_surface,
            div_yield=env.div_yield,
        )
    return float(SnowballPDESolver(params=PDEParams(accuracy="fast")).price(product, env))

prod_cal = snowball(t_cal_of, 0.15)
pv_naive = pde_price(prod_cal, env_naive)
pv_cal = pde_price(prod_cal, env_cal)
h = 1.0
delta_naive = (pde_price(prod_cal, env_naive, SPOT + h)
               - pde_price(prod_cal, env_naive, SPOT - h)) / (2 * h)
delta_cal = (pde_price(prod_cal, env_cal, SPOT + h)
             - pde_price(prod_cal, env_cal, SPOT - h)) / (2 * h)
print(f"same product (coupon 15%), calendar axis, {len(ki_dates)} daily KI dates:")
print(f"  A naive   PV {pv_naive:9.5f}   delta {delta_naive:+.4f}")
print(f"  B correct PV {pv_cal:9.5f}   delta {delta_cal:+.4f}")
print(f"  PV gap A-B: {pv_naive - pv_cal:+.5f}  "
      f"({10000 * (pv_naive - pv_cal) / 100:+.1f} bp of notional)")

pv_cal_zc = pde_price(snowball(t_cal_of, 0.0), env_cal)
pv_td_zc = pde_price(snowball(t_td_of, 0.0), env_td)
print(f"\nzero-coupon variant across axes (same DATES, both correct):")
print(f"  B cal axis   PV {pv_cal_zc:9.5f}")
print(f"  C td  axis   PV {pv_td_zc:9.5f}   rel gap {abs(pv_cal_zc / pv_td_zc - 1):.2e}")

pv_cal_cpn = pde_price(snowball(t_cal_of, 0.15), env_cal)
pv_td_cpn = pde_price(snowball(t_td_of, 0.15), env_td)
print(f"\nWITH coupon 15% naively restated on each axis (a DIFFERENT contract):")
print(f"  B cal axis   PV {pv_cal_cpn:9.5f}")
print(f"  C td  axis   PV {pv_td_cpn:9.5f}   gap "
      f"{10000 * (pv_cal_cpn - pv_td_cpn) / 100:+.1f} bp  <- coupon basis, not engine error")
