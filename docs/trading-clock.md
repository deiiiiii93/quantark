# Trading-Clock Volatility — Usage Guide

Spec: `docs/superpowers/specs/2026-09-01-trading-clock-vol-design.md`.
Plan: `docs/superpowers/plans/2026-09-01-trading-clock-vol.md`.

## The problem in one equation

Vol is quoted per trading year (D = 244 CFFEX/CSI/SSE, 252 CFETS); rates and
dividend yield are quoted per calendar year (ACT/365). Every engine step
from date d_i to d_{i+1} therefore integrates two clocks at once:

```
S_{i+1} = S_i · exp[ (r_i − q_i)·Δτ_cal  −  ½·Δw_i  +  √Δw_i · Z ]
                     └─ calendar time ─┘    └─ trading time: Δw_i = σ_td²·Δn_td/D ─┘
```

Across Chinese New Year the step from the last pre-holiday trading day to
the first post-holiday one carries ~9/365 of carry but exactly 1/D of
variance. QuantArk supports this on either engine axis; the calendar axis
is the default.

## The shared clock map

```python
from quantark.util.calendar import (
    BusinessTimeMap, CalendarType, TradingClock, create_calendar,
)

cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
clock = TradingClock(calendar=cal, days_per_year=244)   # 252 for CFETS
time_map = BusinessTimeMap(clock, anchor_date=valuation_date,
                           horizon_date=product_horizon)
```

- `to_trading(τ_cal)` places **variance**: slope 365/D on trading days,
  exactly flat (bitwise-stable knot values) across holidays.
- `to_calendar(τ_td)` places **carry**: the continuous piecewise-linear map
  between consecutive trading-date knots. It is deliberately NOT the
  pointwise inverse of `to_trading`; the two agree exactly at every
  trading-date knot, which is where the DF invariance is asserted.
- The map's `anchor_date` is the environment's `valuation_date`: since 0.15.0 a `PricingEnvironment` re-anchors
  every trading-clock wrapper it holds (vol surface and cash curves) at its valuation date, at construction and
  whenever `valuation_date` is reassigned, keeping the map's clock and horizon.
- Queries beyond `horizon_date` raise — size the horizon from the product.

## Default: calendar axis + `TradingClockVolSurface`

Engine times stay ACT/365 calendar fractions of real dates (the mo pipeline
convention). Wrap the trading-quoted surface; curves stay native:

```python
from quantark.param.vol import FlatVolSurface, TradingClockVolSurface

env = PricingEnvironment(
    rate_curve=FlatRateCurve(0.02),                    # calendar-quoted, native
    valuation_date=valuation_date,
    spot_quote=SpotQuote(100.0),
    vol_surface=TradingClockVolSurface(
        inner=FlatVolSurface(0.20),                    # σ per √trading-year
        time_map=time_map,
    ),
    div_yield=ContinuousDividendYield(0.01),           # calendar-quoted, native
)
```

`get_vol(K, τ_cal) = σ_inner(K, τ_td)·√(τ_td/τ_cal)` preserves total
variance exactly; the wrapper's `total_variance` method (opt-in marker
`exposes_exact_total_variance`) feeds the engines' step-vol sampling so
holiday intervals carry σ_step == 0.0 **exactly**. The degenerate steps are
handled exactly: QUAD switches to a deterministic drift shift, the PDE runs
those steps upwinded and fully implicit, MC steps become drift-only. No vol
floors anywhere.

The inner surface's maturity axis must be quoted in trading time (pillar at
n_td/D for the pillar's expiry date).

## Opt-in: trading axis + curve wrappers

For desk replication where engine times are remaining-trading-days/D
(date-based products + `day_count_convention=BUSINESS_DAYS` + calendar —
see `example/case_studies/phoenix_engine_diagnostics/phoenix_external_case_compare.py`), the vol surface is native
and the **cash curves** are wrapped instead. Since 0.15.0 the environment does this itself when the curves are
declared `CALENDAR_YEAR` (next section), with the same wrappers over a `DateKnotTimeMap` (0.16.0). The hand recipe
below uses a `BusinessTimeMap`, which keeps the 0.15.0 lag after non-trading days:

```python
from quantark.param.rrf import TradingClockRateCurve
from quantark.param.div import TradingClockDividendYield

env = PricingEnvironment(
    rate_curve=TradingClockRateCurve(FlatRateCurve(0.02), time_map),
    vol_surface=FlatVolSurface(0.20),                  # native trading-quoted
    div_yield=TradingClockDividendYield(ContinuousDividendYield(0.01), time_map),
    day_count_convention=DayCountConvention.BUSINESS_DAYS,
    bus_days_in_year=244,                              # must match the clock
    calendar=cal,                                      # required — no fallback
    ...
)
```

- `TradingClockRateCurve`: `DF_td(u) = DF_cal(to_calendar(u))` — DF ratios
  telescope exactly and holiday carry spreads smoothly across the adjacent
  trading tick (never a jump between substeps).
- `TradingClockDividendYield` is **cumulative-yield preserving**:
  `q_td(u) = q_cal(c)·c/u`, because the engines difference `q(t)·t`.
- Dated products on this axis place a date that is a holiday of `calendar`
  by `PricingEnvironment.holiday_time_adjustment` (default `PRECEDING`: the
  previous business day's time, i.e. plain business-day counting;
  `FOLLOWING` / `MODIFIED_*` available). Contract dates never move, only
  their model time; dates sharing one instant are decided at that instant.
  In 0.14.0 the DCN reads it (`quantark.priceenv.date_axis.env_time`); every
  other product counts business days (= `PRECEDING`).
- One clock per configuration: the environment validates that every wrapped
  curve's `TradingClock` matches (`bus_days_in_year`, calendar identity)
  and raises on mismatch. `BUSINESS_DAYS` without a calendar now raises —
  the old fallback silently computed exactly ACT/365.
- 0.15.0: a native (unwrapped) cash curve on a `BUSINESS_DAYS` environment
  must be declared (`CALENDAR_YEAR` or `TRADING_YEAR`) unless it is exactly
  zero; see the next section.

Both axes price the same date schedule identically (analytical European to
1e-12; PDE/QUAD/MC to discretization tolerance —
`test/test_asset/equity/product_other/test_trading_clock_axis_equivalence.py`).

## Declaring how cash curves are quoted (0.15.0)

On the trading axis the environment cannot tell a calendar-quoted curve from a trading-quoted one by looking at it,
so it is told, per curve:

```python
from quantark.priceenv import CurveQuotingBasis, PricingEnvironment

env = PricingEnvironment(
    rate_curve=FlatRateCurve(0.02),                    # quoted per calendar year (ACT/365)
    div_yield=ContinuousDividendYield(0.01),
    vol_surface=FlatVolSurface(0.20),                  # native trading-quoted
    day_count_convention=DayCountConvention.BUSINESS_DAYS,
    bus_days_in_year=244,
    calendar=cal,
    rate_curve_quoting=CurveQuotingBasis.CALENDAR_YEAR,
    div_yield_quoting=CurveQuotingBasis.CALENDAR_YEAR,
    trading_clock_horizon=product_horizon + timedelta(days=1),
    ...
)
# env.rate_curve is now TradingClockRateCurve(FlatRateCurve(0.02), <the env's map>)
```

| Declaration | `BUSINESS_DAYS` environment | any other convention |
|---|---|---|
| `CALENDAR_YEAR` | the environment wraps the curve on its own `DateKnotTimeMap` (needs `calendar` and `trading_clock_horizon`) | read as is (a wrapper declared `CALENDAR_YEAR` is unwrapped to its inner) |
| `TRADING_YEAR` | read as is; a wrapper is refused | refused |
| none | refused unless the curve is exactly zero or already wrapped (a wrapper is re-anchored at the valuation date) | read as is (a wrapper is re-anchored at the valuation date) |

- **The environment's map reads the actual date (0.16.0).** The environment wraps declared curves
  (`TradingClockRateCurve` / `TradingClockDividendYield` / `TradingClockBasisYield`) over
  `DateKnotTimeMap(TradingClock(calendar, bus_days_in_year), valuation_date, trading_clock_horizon,
  holiday_time_adjustment)`. That map has one knot per business date: the date's env time against its actual
  ACT/365 calendar time, linear between knots.
  - A Monday, or the first day after a holiday, reads the calendar curve at that date.
  - 0.15.0 used `BusinessTimeMap`, whose knots sit at the end of each trading day. A Monday was discounted as the
    Saturday: +1.64e-4 on the DF at 3%, and +8.20e-4 after Spring Festival.
  - A hand-made recipe on a `BusinessTimeMap` keeps that lag. A declared curve is always moved onto the
    environment's map; an undeclared hand-wrapped curve is left as built.
  - An environment holding both kinds can refuse a relationship tenor as "not on one clock".
- `trading_clock_horizon` must lie strictly after the last date priced. The map reaches the first business date
  on or after the horizon, and a read past its last knot raises.
- A `FOLLOWING` environment valued on a holiday is refused, because the next business date would sit at u = 0
  with calendar carry already accrued. This applies at construction and on a `valuation_date` move.
- **The map follows the valuation date.** `dataclasses.replace(env, valuation_date=d)` and a plain
  `env.valuation_date = d` (what numerical theta, the product theta helpers, dynamic scenarios and lifecycle do on
  a copy) re-anchor every wrapped cash curve at `d`: an environment-made one on the environment's horizon, a
  hand-made one on its own map's horizon. The `TradingClockVolSurface` of the calendar-axis default re-anchors
  the same way, so a one-day theta that starts on a weekend or holiday is carry only (before 0.15.0 the bumped
  copy kept the old anchor and mis-stated the variance to each product date by [the start day is a trading day] -
  [the day before that date is a trading day] trading days: about 20x too large for a European across Chinese New
  Year, and wrong from trading-day starts too for dates that follow a non-trading day). A date at or past the horizon raises. An `IntradayTimeMap` (the intraday route's own clock) is never
  moved by the environment.
- Cash stresses and scenario moves are per calendar year: they shift the calendar-quoted curve inside a wrapper and
  keep its trading clock (`quantark.priceenv.clock_validation.calendar_quoted_view`).
- Basis/dividend relationship modes (0.16.0) work across wrappers. The tenor `T` (environment axis) is mapped once
  to calendar time, `T_cal = to_inner_time(T)`, before the shift (`relationship_tenor`). Every read at the tenor
  uses `T_cal`: the source curve's current value and the relationship's rate, dividend and basis. A curve derived
  into an empty slot (`None` or exactly zero, undeclared) is wrapped on the sources' clock. Three cases are refused:
  a non-zero native curve next to a wrapped one, a `TRADING_YEAR`-declared target slot, and wrappers mapping `T`
  to different calendar times. `INDEPENDENT` on such a mixed environment keeps reading at `T`, including a wrapped
  source (conservative).
- A curve a bump creates from nothing (0.16.0) is a calendar quote. A dividend bump of `div_yield=None` on a
  `BUSINESS_DAYS` environment declaring it `CALENDAR_YEAR` (dividend rho, the greek-conventions report, carry rho-q
  buckets) is wrapped on the environment's map (`place_created_cash_curve`).
- Bump curves composed over a wrapper (0.16.0) shift the calendar quote. Local-vol rho, Heston/SLV PDE rho and
  the convertible DV01/convexity shift the wrapper's inner and re-wrap (`clock_parallel_shift_rate`). Carry rho-q
  bucket windows are mapped onto the inner's calendar axis. A window edge past the clock's horizon is placed past
  the last calendar knot; the wrapper refuses reads there, so the placement is exact. FX environments have no
  trading clock and are unchanged.
- Exactly zero (`FlatRateCurve(0.0)`, `ContinuousDividendYield(0.0)`, `NoDividend`, `FlatBasisYield(0.0)`,
  `ZeroBasis`, `None`) means the same on both clocks and needs no declaration.

## Unit conventions (documented consequence, not a defect)

| Quantity | Calendar axis | Trading axis | Conversion |
|----------|---------------|--------------|------------|
| σ | σ_cal | σ_td | σ_td = σ_cal · √(τ_cal/τ_td) at the same expiry |
| theta | per calendar day | per trading day | not a constant factor: both step the same dates and divide the same date-step P&L by different Δt. A Fri→Mon or holiday-crossing step carries several calendar days of carry in one trading day; calendar theta on a holiday is carry-only (zero variance decay) |
| vega | per unit σ_cal | per unit σ_td | ×√(τ_td/τ_cal) |
| rho | per unit r (calendar-annual) | same | none — r stays calendar-quoted on both axes (DFs attach to dates) |

**Coupon and rebate rates are CONTRACT data, not engine properties.** A
snowball coupon accrues as `ko_rate × elapsed engine time`, so quoting the
same `ko_rate` on both axes produces two *different contracts* (0.15·τ_td ≠
0.15·τ_cal at the same date). When moving a term sheet between axes,
restate the accrual so the cash paid at each date is unchanged — the
axis-equivalence gate uses a zero-coupon contract for exactly this reason.

## PDE grid day

The PDE's `steps_per_day` density counts the pricing environment's business days
(0.8.0): an interval of length `L` years on the engine's time axis gets
`L · PricingEnvironment.bus_days_in_year · steps_per_day` steps, at least one day's worth.
Set `bus_days_in_year=244` for a CFFEX/CSI/SSE book and the grid follows it (a one-year
Snowball at `standard`: 972 steps instead of 1,008); nothing on `GridConfig` sets it any
more (`GridConfig(day_count=...)` raises, naming `bus_days_in_year`). On the calendar axis
a grid "day" is still `1 / bus_days_in_year` of a year of engine time, not a calendar
day. The intraday route prices on its context's numerical environment, whose
`bus_days_in_year` is the variance profile's `days_per_year`.

## Calendar-quoted vol on the trading axis: `CalendarQuotedVolSurface` (0.16.0)

On a `BUSINESS_DAYS` env the vol is read at trading time, so a calendar-quoted
surface (a market vol column, the IV-surface artifacts) must be converted.
`CalendarQuotedVolSurface(inner, DateKnotTimeMap(...))` presents a calendar-quoted
`inner` on the trading axis:

    σ_td(K, u) = σ_cal(K, c(u)) · √(c(u) / u)

`c(u)` comes from a date-knot map: one knot per business date d, at
(`env_time(d)`, ACT/365 calendar time of d), linear between knots. Every business
date therefore carries exactly the calendar variance σ_cal(c_d)² · c_d. This includes
a Monday and the first day after a holiday.

`BusinessTimeMap.to_calendar` is not used here. It maps a date after a non-trading
day back to the day after the previous trading day, so it would drop the weekend's
variance. The environment's declared cash curves use the same `DateKnotTimeMap`, so
vol and cash agree on every business date. The env re-anchors the wrapper like any
clock wrapper.
`parallel_shifted` and `bump_unit_vol` act on the calendar quote. QUAD V2 accepts
it when the inner is deterministic (flat or term structure).

## Replay on a business-day clock (0.16.0)

A replay backtest (single product or book; verified on Snowball and KO-reset with QUAD V1, PDE and MC, and on
DCN with MC) can price each day on
a `BUSINESS_DAYS` clock:

```python
from quantark.backtest.replay import AutocallableEngineConfig, TradingClockConfig
from quantark.util.calendar import CalendarType, create_calendar

sse = create_calendar(CalendarType.CHINA_SSE)
engine_config = AutocallableEngineConfig(
    ...,
    trading_clock=TradingClockConfig(calendar=sse, bus_days_in_year=244),
)
```

- **Cash curves.** Every day's env declares the flat rate, the implied or term
  dividend curve and the basis `CALENDAR_YEAR`, and wraps them on its own clock.
- **Vol.** The vol is wrapped in `CalendarQuotedVolSurface`
  (`vol_quoting=CALENDAR_YEAR`, the default), in every surface mode. With
  `vol_quoting=TRADING_YEAR` it is used as is.
- **Same clock everywhere.** The day env, the maturity probe, the scenario and carry
  callbacks, the book's spot-hedge carry and the surface grid all use one clock.
- **Horizon.** `trading_clock_horizon` is the later of the last market date and every
  product's last contract, settlement or payment date, plus 366 days.
- **Unchanged.** Lifecycle trackers still decide by date. Futures tenors and the carry
  inversion keep their ACT/365 market quotes. The state row's `pricing_q` stays the
  calendar quote.
- **Report dates and settlement.** Event dates in the reports map engine times back
  through business-day counts. Pending receivables are discounted at the
  settlement date's env time.
- **Greek units and the PDE grid.** Theta is per trading day (see the unit table). Vega
  stays per unit CALENDAR vol: with `vol_quoting=CALENDAR_YEAR` the bump moves the calendar
  quote, not σ_td. The PDE's default grid has `steps_per_day × bus_days_in_year` steps per
  env year.
- **Refused** (each would mix clocks silently):
  - vol models other than `bsm`;
  - `trade_value`;
  - `pnl_explain`;
  - the simulated-path / ensemble engines;
  - products with a numeric (float-time) schedule;
  - a replay day that is not a business day of the clock's calendar;
  - dashboard snapshot repricing of a clock run.
- **No silent fallback.** On the clock path the MC event-statistics fallback lets a
  `ValidationError` through.
- **Results.** The results' run info records the clock.

The default (`trading_clock=None`) is the calendar-day replay, unchanged.

## Scope

BSM autocallable stack (Snowball / Phoenix / KO-reset PDE, QUAD, MC +
analytical European). LV/Heston/SLV are deferred: Dupire differentiates
total variance in maturity, so the clock enters the calibrator itself. The
vol-calibration module's artifacts are calendar-clock listed-market
surfaces and stay unwrapped.
