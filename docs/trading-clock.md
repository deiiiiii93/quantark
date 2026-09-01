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
- The map's `anchor_date` must equal the environment's `valuation_date`.
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
see `example/phoenix_external_case_compare.py`), the vol surface is native
and the **cash curves** are wrapped instead:

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
- One clock per configuration: the environment validates that every wrapped
  curve's `TradingClock` matches (`bus_days_in_year`, calendar identity)
  and raises on mismatch. `BUSINESS_DAYS` without a calendar now raises —
  the old fallback silently computed exactly ACT/365.

Both axes price the same date schedule identically (analytical European to
1e-12; PDE/QUAD/MC to discretization tolerance —
`test/test_trading_clock_axis_equivalence.py`).

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

## Scope

BSM autocallable stack (Snowball / Phoenix / KO-reset PDE, QUAD, MC +
analytical European). LV/Heston/SLV are deferred: Dupire differentiates
total variance in maturity, so the clock enters the calibrator itself. The
vol-calibration module's artifacts are calendar-clock listed-market
surfaces and stay unwrapped.
