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
declared `CALENDAR_YEAR` (next section); the hand recipe below is what it builds:

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
| `CALENDAR_YEAR` | the environment wraps the curve on its own `BusinessTimeMap` (needs `calendar` and `trading_clock_horizon`) | read as is (a wrapper declared `CALENDAR_YEAR` is unwrapped to its inner) |
| `TRADING_YEAR` | read as is; a wrapper is refused | refused |
| none | refused unless the curve is exactly zero or already wrapped (a wrapper is re-anchored at the valuation date) | read as is (a wrapper is re-anchored at the valuation date) |

- The environment's wrapping is exactly the hand recipe above (`TradingClockRateCurve` /
  `TradingClockDividendYield` / `TradingClockBasisYield` over `BusinessTimeMap(TradingClock(calendar,
  bus_days_in_year), valuation_date, trading_clock_horizon)`): the same prices bit for bit on every engine family.
- `trading_clock_horizon` must lie strictly after the last date priced. `BusinessTimeMap` counts whole days from the
  valuation instant, so with a time-of-day valuation (14:55) use last date + 1 day.
- **The map follows the valuation date.** `dataclasses.replace(env, valuation_date=d)` and a plain
  `env.valuation_date = d` (what numerical theta, the product theta helpers, dynamic scenarios and lifecycle do on
  a copy) re-anchor every wrapped cash curve at `d`: an environment-made one on the environment's horizon, a
  hand-made one on its own map's horizon. The `TradingClockVolSurface` of the calendar-axis default re-anchors
  the same way, so a one-day theta that starts on a weekend or holiday is carry only (before 0.15.0 the bumped
  copy kept the old anchor and decayed a full trading day of variance: about 20x too large across Chinese New
  Year). A date at or past the horizon raises. An `IntradayTimeMap` (the intraday route's own clock) is never
  moved by the environment.
- Cash stresses and scenario moves are per calendar year: they shift the calendar-quoted curve inside a wrapper and
  keep its trading clock (`quantark.priceenv.clock_validation.calendar_quoted_view`). Basis/dividend relationship
  modes other than `INDEPENDENT` are refused on wrapped curves.
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

## Scope

BSM autocallable stack (Snowball / Phoenix / KO-reset PDE, QUAD, MC +
analytical European). LV/Heston/SLV are deferred: Dupire differentiates
total variance in maturity, so the clock enters the calibrator itself. The
vol-calibration module's artifacts are calendar-clock listed-market
surfaces and stay unwrapped.
