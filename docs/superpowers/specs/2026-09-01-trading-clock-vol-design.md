# Trading-Clock Volatility — Design

Date: 2026-09-01
Status: draft for review
Scope: BSM autocallable stack (Snowball / Phoenix / KO-reset PDE, QUAD, MC + analytical European). LV/Heston/SLV and FX are explicitly deferred (§10).

## 1. Problem

Market data for CN equity derivatives is quoted on two different clocks:

- **Vol** is quoted per trading year: a desk mark σ_td means total variance
  σ_td² · n_td/D to an expiry that is n_td trading days away, with D = 244
  (CFFEX/CSI/SSE) or D = 252 (CFETS). No variance accrues on holidays.
- **Rates and dividend yield** are quoted per calendar year (ACT/365 zero
  curves). Cash accrues interest through holidays.

Every engine step from date d_i to d_{i+1} therefore needs two time increments
simultaneously:

```
S_{i+1} = S_i · exp[ (r_i − q_i)·Δτ_cal  −  ½·Δw_i  +  √Δw_i · Z ]
                     └─ calendar time ─┘    └─ trading time: Δw_i = σ_td²·Δn_td/D ─┘
```

Across Chinese New Year the step from the last pre-holiday trading day to the
first post-holiday one carries Δτ_cal ≈ 9/365 of carry but exactly Δn_td = 1
day of variance. A single clock cannot be right for both halves: a calendar
clock invents holiday variance; a trading clock under-accrues holiday interest.

The library today has one clock per configuration and no way to split them.
This design adds the split without changing any engine.

## 2. Current state (verified 2026-09-01)

1. **Engines are clock-agnostic.** All engine families consume pre-computed
   year fractions. Per-step market coefficients come from one shared module:
   `TermCoefficients.from_env` (`quantark/priceenv/term_sampling.py`) samples
   r as DF-exact forwards `−ln(DF₁/DF₀)/Δτ` over each (possibly uneven)
   interval, q as zero-yield differences, and vol by total-variance
   differencing (`step_vols_on_grid`), all on the engine's own `t_grid`.
   Consequence: the *cash half* of the SDE in §1 is already exact on any
   date-anchored calendar grid, uneven holiday gaps included.
2. **The certified paths are calendar-clock.** The mo pipeline, replay, and
   all float-based callers compute times as `(d − t0).days / 365` of real
   trading dates. The trading calendar selects *which* dates exist;
   ACT/365 decides *where* they sit on the axis.
3. **A trading-clock mode exists but is broken on the rates side.**
   Date-based products resolve times through
   `BaseEquityOption.get_maturity(pricing_env)` and
   `ObservationRecord.resolve_time(pricing_env)`, which honor
   `PricingEnvironment.day_count_convention`. With `BUSINESS_DAYS` + a
   calendar, the engine axis genuinely becomes remaining-trading-days/D
   (used by `example/phoenix_external_case_compare.py`,
   `example/ko_reset_snowball_demo.py`). But nothing converts the curves:
   calendar-quoted r/q are queried at trading abscissae — the DF of a
   different date, accrued over the wrong measure (~3%·r·T PV error, lumpy
   around holidays).
4. **`BUSINESS_DAYS` without a calendar is a silent identity.** The fallback
   in `calculate_year_fraction` computes `num_days·(D/365)/D ≡ num_days/365`
   — exactly ACT/365, for every D. `test/test_european_option.py:443`
   exercises this path believing it tests business-day pricing.
5. **Zero forward variance is unguarded.** A holiday-flat total variance
   interval yields σ_step = 0. The QUAD transition kernel divides by σ√dt;
   the PDE operator degenerates to centrally-differenced pure advection.
   Un-triggered today only because pillar interpolation smears holiday
   variance; armed the moment vol becomes trading-clock-correct.

## 3. Decisions (from design review with the desk)

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | **Split clocks**: variance on trading time, cash/carry on calendar time. | Matches quoting: vol per trading year, r/q per calendar year. No single-clock setting can serve both (§1). |
| D2 | **Scope**: BSM autocallable stack first. | One shared sampling module (`TermCoefficients`) covers PDE/QUAD/MC; analytical European rides along via the env accessors. Dupire/Heston calibration would pull the clock into the calibrator (dw/dT) — deferred. |
| D3 | **Denominator binds to the calendar**: 244 for CFFEX/CSI/SSE, 252 for CFETS. | Desk convention. A `TradingClock` value object pairs them so the two can never be mixed accidentally. |
| D4 | **Support both engine axes; calendar is the default.** | Prices are identical either way (both integrate Δτ_cal for cash and Δn_td/D for variance per step). Calendar axis keeps every certified path, golden, and float-based caller valid; the trading axis remains available for desk-replication work. |
| D5 | **Discount factors attach to dates, not axes.** | The correctness invariant: on any axis, the discount between two dates must equal DF_cal(d₁,d₂) from the calendar-quoted curve. Both wrappers below are constructed to satisfy it, which is what makes D4's "identical prices" true and keeps the analytical engine consistent with PDE/QUAD (its `exp(−r·T)` evaluates the same date-anchored DF). |
| D6 | **Exact semantics, no floors.** | Degenerate σ_step = 0 intervals get exact handling (deterministic-shift kernel), never a vol floor. Holiday-flat variance is the *correct* answer, not an edge case to paper over. |

## 4. Architecture

Three small units. Engines, `TermCoefficients`, products, and the calibration
artifact format are untouched.

### 4.1 `quantark/util/calendar/trading_clock.py` — the shared clock map

```python
@dataclass(frozen=True)
class TradingClock:
    calendar: Calendar          # e.g. CHINA_SSE, or the mo CSV-backed calendar
    days_per_year: int          # 244 (CFFEX/CSI/SSE), 252 (CFETS)

class BusinessTimeMap:
    def __init__(self, clock: TradingClock, anchor_date: date, horizon_date: date): ...
    def to_trading(self, tau_cal): ...    # scalar or ndarray
    def to_calendar(self, tau_td): ...    # scalar or ndarray
```

- Precomputed knots, one per calendar day from `anchor_date` to
  `horizon_date`. `to_trading` is piecewise linear: slope 365/D across a
  trading day, slope 0 across a holiday. Continuous, monotone
  (non-strictly).
- `to_calendar` is the inverse restricted to increasing segments: exact at
  trading-day knots (the only points date-anchored consumers hit), linear
  within a trading day. A holiday plateau maps to the plateau's start.
  This convention is documented on the method.
- **No extrapolation by default.** Querying beyond `horizon_date` raises
  `ValidationError`. An explicit `extend_weekdays=True` constructor flag
  enables the mo-style weekday extension past the calendar's data, and the
  flag's use is recorded on the object (`repr` and an attribute) so a run
  can be audited.
- Both directions are the same knot table; the two wrappers below consume
  one shared map object, so the axes cannot drift apart numerically.

### 4.2 Calendar axis (default): `TradingClockVolSurface`

`quantark/param/vol/trading_clock_surface.py`:

```python
@dataclass(frozen=True)
class TradingClockVolSurface(BlackImpliedVolSurface):
    inner: BlackImpliedVolSurface   # axis quoted in trading time (pillars at n_td/D)
    time_map: BusinessTimeMap
```

- `get_vol(K, τ_cal, spot)` computes `τ_td = time_map.to_trading(τ_cal)` and
  returns `σ_inner(K, τ_td) · sqrt(τ_td / τ_cal)` — i.e. **total variance is
  preserved by construction**: w_cal(τ) = w_td(τ_td(τ)).
- τ_cal → 0 limit: return `σ_inner(K, 0⁺) · sqrt(slope(0))` where slope(0)
  is the map's initial derivative (0 if the anchor sits before a holiday —
  the correct statement that no variance accrues before the next trading
  day). Implemented via the knot table, not L'Hôpital at runtime.
- `is_smile` passes through from `inner`.
- Input contract (docstring + `asset/equity/CLAUDE.md`): a wrapped surface
  must only be used with engine times that are ACT/365 calendar fractions of
  real dates, and the map's `anchor_date` must equal the environment's
  `valuation_date`. The wrapper cannot detect a violation; the contract line
  is the defense, and the daily rebuild in backtests re-anchors the map each
  valuation date.

Because every engine consumes vol only through total-variance differencing,
this wrapper alone delivers the split-clock SDE of §1 on the default axis:
`Δw_i = w_td(τ_td(t_{i+1})) − w_td(τ_td(t_i))` is exact per step, zero across
holidays, and MC substeps inside a holiday-straddling interval subdivide Δw
correctly (the map is piecewise linear with daily knots).

### 4.3 Trading axis (opt-in): `TradingClockRateCurve`

`quantark/param/rrf/trading_clock_curve.py`:

```python
class TradingClockRateCurve(RateCurve):
    inner: RateCurve                # calendar-quoted (ACT/365)
    time_map: BusinessTimeMap
```

- `get_discount_factor(τ_td) = inner.get_discount_factor(time_map.to_calendar(τ_td))`;
  `get_rate` derives from the DF. Forwards telescope exactly, so holiday-
  crossing steps legitimately carry large forward-rate spikes (≈ 9 days of
  interest in one trading tick) while PV, parity and carry stay exact (D5).
- A matching thin wrapper `TradingClockDividendYield` adapts `DividendYield`
  the same way (same map object).
- Pairing rule (documented): on the trading axis the vol surface is native
  (unwrapped, quoted at n_td/D pillars) and the curves are wrapped — the
  mirror of the default axis. `phoenix_external_case_compare.py` gets a
  pointer to this as the correct construction.

### 4.4 `BUSINESS_DAYS` hardening

`calculate_year_fraction` with `BUSINESS_DAYS` and no calendar raises
`ValidationError` naming the missing calendar, instead of silently computing
ACT/365. `bus_days_in_year` keeps its meaning (the denominator D) and becomes
actually load-bearing. `test/test_european_option.py:443` is rewritten to
pass a calendar and assert genuinely business-day behavior.

### 4.5 Degenerate-interval guards (armed by this feature)

- **QUAD**: an interval with σ_step = 0 gets an exact deterministic-shift
  transition — values are shifted by the interval's log-drift
  `(r − q)·Δτ` and re-interpolated onto the grid, then discounted; no
  Gaussian kernel, no division by σ√dt. This is the σ → 0 limit of the
  kernel — the only numerical error left is the grid interpolation the QUAD
  step already carries; no new approximation is introduced.
- **PDE**: zero-diffusion steps (D = 0, central-differenced advection under
  the implicit θ-scheme) get a characterization test on a holiday-straddling
  case — stability and monotonicity are verified, not assumed. If the test
  shows oscillation at production settings, the fix is scheme-level (upwind
  those steps), decided on the test's evidence, not pre-emptively.
- **MC**: a unit test pins the zero-vol step (drift-only advance, no
  division hazard) for the GBM path generator and the snowball engines.

### 4.6 Relationship to the vol-calibration module (spec of the same day)

`2026-09-01-volcalibration-module-design.md` produces calendar-clock
artifacts from listed-market quotes; their implied vols already embed holiday
effects in the level, and the artifact bytes are sha-frozen. This design does
not touch that path: calibration artifacts stay unwrapped on the calendar
axis. `TradingClockVolSurface` is for *trading-quoted* desk/vendor marks — a
different input class. The two compose (a backtest can hold both kinds of
surface on the same env) because the wrapper is query-time only.

## 5. Unit conventions (documented consequence, not a defect)

Greeks and quotes carry axis-dependent units. The spec of record:

| Quantity | Calendar axis | Trading axis | Conversion |
|----------|---------------|--------------|------------|
| σ | σ_cal | σ_td | σ_td = σ_cal · sqrt(τ_cal/τ_td) at the same expiry |
| theta | per calendar day | per trading day | not a constant factor: both step the same dates; they divide the same date-step P&L by different Δt. A Fri→Mon or holiday-crossing step carries several calendar days of carry in one trading day; calendar theta on a holiday is carry-only (zero variance decay) |
| vega | per unit σ_cal | per unit σ_td | ×sqrt(τ_td/τ_cal) |
| rho | per unit r (calendar-annual) | same | none — r is calendar-quoted on both axes (D5) |

Desk comparisons must state units; the conversion table ships in the docs.

## 6. Error handling

- `BusinessTimeMap` query beyond horizon → `ValidationError` (unless
  `extend_weekdays=True`), negative time → `ValidationError`.
- `TradingClockVolSurface` construction rejects a map whose anchor/horizon
  cannot cover the inner surface's last pillar mapped to calendar time.
- `TradingClockRateCurve` propagates the inner curve's validation; wrapped
  DF must remain strictly positive (inherited check).
- `BUSINESS_DAYS` without calendar → `ValidationError` (§4.4).
- `step_vols_on_grid` behavior unchanged: dw < −1e-12 still raises
  (calendar arbitrage); dw = 0 is legal and now meaningful.

## 7. Validation

1. **Map properties** (exact): monotone; flat exactly on holidays;
   `to_trading(to_calendar(k/D)) = k/D` at every trading knot; slope 365/D
   on trading days; 244 vs 252 per calendar; horizon fail-closed.
2. **European parity** (exact): wrapper price on the calendar axis equals
   closed-form BS with (σ_td, τ_td) and DF_cal at the same expiry date.
3. **DF invariance** (exact): for every trading date to horizon,
   `DF_td(τ_td(d)) == DF_cal(τ_cal(d))` — the D5 invariant, byte-for-byte
   the same float where the arithmetic permits, else ≤ 1 ulp.
4. **Axis equivalence** (flagship, tolerance = discretization): one
   CNY-straddling daily-KI snowball priced both ways — calendar axis +
   `TradingClockVolSurface` vs trading axis + `TradingClockRateCurve` —
   agrees across analytical (European control), PDE, QUAD, and MC.
5. **Node marginals**: on the calendar axis, Var[ln S] at each observation
   node equals w_td(node) and E[S_T]·DF_cal(T) satisfies forward parity —
   pins each clock to its own quantity.
6. **Degenerate guards**: QUAD deterministic-shift interval vs analytic
   value; PDE zero-D stability/monotonicity characterization; MC zero-vol
   step unit test.

## 8. Phasing

- **Phase 1 — the need**: `TradingClock`, `BusinessTimeMap`,
  `TradingClockVolSurface`, degenerate guards (§4.5), tests 1/2/5/6,
  docs/contract lines. Delivers trading-clock vol on the default axis.
- **Phase 2 — the opt-in axis**: `TradingClockRateCurve` (+ dividend
  wrapper), `BUSINESS_DAYS` hardening (§4.4), tests 3/4, example fixes
  (`test_european_option.py:443`, pointer in
  `phoenix_external_case_compare.py`), unit-convention docs (§5).

Phase 2 lands second because the axis-equivalence test needs both halves.

## 9. Hygiene riding along

- `GridConfig.day_count` docstring corrected: it is a step-density constant
  in the units of the engine axis, not a day-count convention.
- `PricingEnvironment.day_count_convention` docstring states explicitly that
  float-based products bypass it and which resolvers honor it.

## 10. Out of scope (deferred, recorded)

- LV/Heston/SLV: Dupire differentiates total variance in maturity, so the
  clock enters the calibrator itself; the mo fleet keeps its calendar-clock
  listed-market surfaces meanwhile (§4.6).
- FX stack (GK engines, VV surfaces): different conventions (expiry cuts,
  T+2, CFETS D=252) — reuse `TradingClock` when taken up.
- Intraday anchoring (valuation morning vs close): the map's knots are
  daily; sub-day anchoring is a future refinement of `anchor_date`.
- Per-contract denominators ("actual count" quoting): needs per-quote
  metadata; not requested.
