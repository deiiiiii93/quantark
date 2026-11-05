# Curve Construction — Usage Guide

Spec and plan: `.ship/2026-11-05-curve-construction/` (run directory, untracked).
Demo: `python example/param/curve_construction_demo.py` writes
`example/output/curve_construction_report.html`.

Everything described here lives in `quantark/param/curve/` and is re-exported
from `quantark.param`. It is strictly additive: `FlatRateCurve`,
`InterpolatedRateCurve`, `ForwardCarryCurve`, `TermStructureDividendYield` and
every engine keep their old behaviour, with one deliberate exception — the
floating-rate projection convention of the rate engines, described at the end.

## One kernel, two curves

A rate curve and a carry curve are the same object seen through different
units: a *cumulative quantity* `Y(T)` with level `y(T) = Y(T)/T`.

| curve | cumulative `Y(T)` | level `y(T)` | forward `dY/dT` |
|---|---|---|---|
| `ZeroCurve` | `−log DF(T) = z(T)·T` | zero rate `z(T)` | instantaneous forward rate |
| `CarryCurve` | `B(T) = log F(T)/S` | zero carry yield `B(T)/T` | forward carry `b(T)` |

`TermInterpolator(times, levels, interpolation, long_end)` holds the pillars
`(T_i, y_i)` with `0 < T_1 < … < T_n` and exposes `cumulative(t)`, `level(t)`
and `forward(t)` (the right derivative). Scalars in give floats out, arrays in
give arrays out, evaluated by the identical formula.

### Interpolation schemes (`CurveInterpolation`)

| scheme | what is interpolated | short end `[0, T_1]` |
|---|---|---|
| `LINEAR_ZERO` | level `y`, piecewise linear | `y = y_1` |
| `LOG_LINEAR_DF` | cumulative `Y`, piecewise linear through `(0, 0)` | implied by the `(0, 0)` anchor (constant forward) |
| `CUBIC_ZERO` | level `y`, natural cubic spline on `T_1..T_n` | `y = y_1` |
| `LOG_CUBIC_DF` | cumulative `Y`, natural cubic spline on `0, T_1..T_n` | spline through `(0, 0)` |
| `MONOTONE_CONVEX` | Hagan–West (2006) forward interpolation with positivity enforcement | `f_0 = fd_1 − ½(f_1 − fd_1)` |

`LOG_LINEAR_DF` is the market-standard discount-curve scheme (piecewise-constant
forwards, local, exact on deposits). The cubic schemes are smooth but
non-local: a change at one pillar moves the whole curve. `MONOTONE_CONVEX`
produces continuous, positive forwards wherever the discrete forwards are
positive and reproduces every pillar exactly (`G(1) = 0` in each of the four
published regions). Its single documented limitation: forward continuity at a
knot is not guaranteed when one of `g0`, `g1` is exactly zero and the other is
not (two equal consecutive discrete forwards).

### Long-end rules

Beyond `T_n` the interpolator applies a `LongEndRule`; the public enums map to
it:

| public option | rule | beyond `T_n` |
|---|---|---|
| `RateExtrapolation.FLAT_ZERO_RATE` | `HOLD_LEVEL` | `z(T) = z_n` |
| `RateExtrapolation.FLAT_FORWARD_RATE` (default) | `HOLD_FORWARD` | `Y(T) = Y_n + fd_n·(T − T_n)`, last discrete forward |
| `CarryExtrapolation.FLAT_FORWARD_CARRY` (default) | `HOLD_FORWARD` | last discrete forward carry held |
| `CarryExtrapolation.ZERO_FORWARD_CARRY` | `HOLD_CUMULATIVE` | `B(T) = B_n` (no carry past the last node) |

## `ZeroCurve`

```python
from datetime import datetime
from quantark.param import ZeroCurve, CurveInterpolation
from quantark.param.extrapolation import RateExtrapolation

zc = ZeroCurve(
    [(0.25, 0.030), (0.5, 0.031), (1.0, 0.032), (2.0, 0.0345), (5.0, 0.036)],
    interpolation=CurveInterpolation.MONOTONE_CONVEX,
    extrapolation=RateExtrapolation.FLAT_FORWARD_RATE,
    valuation_date=datetime(2026, 11, 5),      # None = numeric axis
)
zc.get_rate(1.5); zc.get_discount_factor([0.0, 1.5, 7.0]); zc.instantaneous_forward(1.5)
zc.with_pillars([...])        # same configuration, node roles and anchor
zc.parallel_shifted(1e-4)     # exact DF·exp(−sT)
```

`ZeroCurve` is a `RateCurve`: every pricing engine, `TradingClockRateCurve`,
the key-rate bucketing (`key_rate_bumped_zero_curve` now calls `with_pillars`
when the curve has it) and every consumer that bumps a curve through
`parallel_shifted` accept it unchanged. The parts of the risk stack that
type-check for `InterpolatedRateCurve` do **not** yet (see Open items). Pillars at `T = 0`
are rejected (`DF(0) = 1` is implicit); `T < 0` raises `ValidationError`.

`parallel_shifted` returns a pillar-shifted `ZeroCurve` for the four linear
schemes (the shift is exact there) and an `AnchoredParallelShiftRateCurve`
wrapper for `MONOTONE_CONVEX`, whose clipping is not linear in the pillars.
The wrapper keeps the `valuation_date` anchor and accepts arrays.

## `CarryCurve` and the dividend-yield view

```python
from quantark.param import CarryCurve
from quantark.param.extrapolation import CarryExtrapolation

cc = CarryCurve([(0.25, 0.0025), (0.5, 0.0049), (1.0, 0.0098)],
                extrapolation=CarryExtrapolation.FLAT_FORWARD_CARRY)
cc.carry(0.75)                   # B(T)
cc.forward(s0=4000.0, T=0.75)    # S·exp(B)
cc.forward_carry(0.75)           # dB/dT⁺
cc.interval_carry(0.5, 1.0)      # (B(1) − B(0.5)) / 0.5
q = cc.to_dividend_yield(zc)     # CarryImpliedDividendYield: q(T) = r(T) − B(T)/T, a live view
```

With the defaults `CarryCurve(nodes)` is numerically identical to
`ForwardCarryCurve(nodes)`. `MONOTONE_CONVEX` is refused for carry (its
positivity clip has no meaning for a signed basis). The view's
`parallel_shifted(s)` shifts the nodes exactly under `FLAT_FORWARD_CARRY` and
falls back to the exact `ParallelShiftDividendYield` wrapper under
`ZERO_FORWARD_CARRY`, whose flat tail cannot represent `q + s`.

## Quotes and the clock rule

Quotes are frozen dataclasses. Each carries **either** `tenor_years` (numeric
run) **or** dates (dated run), never both; a run is numeric when
`valuation_date=None` and dated otherwise, and mixing raises
`ValidationError`. The dated axis is calendar ACT/365F,
`T = (date − valuation_date).days / 365`, the convention the bond and rate
engines use, so bootstrapped curves reprice through the engines by
construction.

| quote | instrument | pillar |
|---|---|---|
| `DepositQuote(rate, tenor_years \| start_date/maturity_date, day_count=ACT_365)` | money-market deposit / repo, simple rate | maturity |
| `SwapQuote(fixed_rate, index, tenor_years \| effective_date/maturity_date, payment_frequency=QUARTERLY, …)` | par IRS, payer, notional 1, spread 0 | last payment |
| `BondQuote(coupon_rate, clean_price, payment_frequency, tenor_years \| issue_date/maturity_date, day_count=ACT_ACT_ISDA)` | fixed coupon bond on 100 face | maturity |
| `FuturesQuote(price, tenor_years \| expiry_date)` | index futures | expiry |
| `OptionPairQuote(strike, call_price, put_price, tenor_years \| expiry_date)` | same-strike call/put pair | expiry |

Dated swaps and bonds are turned into the repo's own products —
`SwapQuote.to_swap(valuation_date)` builds an `InterestRateSwap` from
`FixedLeg`/`FloatingLeg`, `BondQuote.to_bond(valuation_date)` a `FixedBond` —
and the bootstrap reads *their* cash flows (payment dates, accrual windows,
day-count fractions, accrued interest). Numeric quotes are the idealized
twins: `n = tenor·frequency` whole periods, `dcf = 1/frequency`, bonds valued
on a coupon date. `FuturesQuote.from_index_futures_curve(curve, dated=False)`
converts an existing `IndexFuturesCurve`.

Curves carry the same clock: a `ZeroCurve`/`CarryCurve` with a
`valuation_date` handed to a numeric run, or to a run on another date, raises
`ValidationError`; curves without the attribute (`FlatRateCurve`) pass.

## Bootstrapping

```python
from quantark.param import (BootstrapConfig, DepositQuote, SwapQuote, BondQuote,
                            bootstrap_discount_curve, bootstrap_projection_curve,
                            build_carry_curve, reprice_quotes, FuturesQuote)
from quantark.param.index import SHIBOR_3M

quotes = [
    DepositQuote(rate=0.0180, tenor_years=1/12), DepositQuote(rate=0.0190, tenor_years=0.25),
    DepositQuote(rate=0.0205, tenor_years=0.5),
    SwapQuote(fixed_rate=0.0220, index=SHIBOR_3M, tenor_years=1.0),
    SwapQuote(fixed_rate=0.0262, index=SHIBOR_3M, tenor_years=3.0),
    BondQuote(coupon_rate=0.030, clean_price=100.9, payment_frequency=PaymentFrequency.SEMI_ANNUAL, tenor_years=5.0),
]
cfg = BootstrapConfig(interpolation=CurveInterpolation.MONOTONE_CONVEX)   # + extrapolation, max_sweeps
discount = bootstrap_discount_curve(quotes, cfg)                           # single-curve
residuals = reprice_quotes(quotes, discount)                               # ≤ 1e-12 each

# dual-curve: project SHIBOR_3M off an OIS-style discount curve
projection = bootstrap_projection_curve(deposits + swaps, discount_curve=ois, config=cfg)

# carry curve from futures (or option pairs), with the rate curve as input
carry = build_carry_curve(spot, FuturesQuote.from_index_futures_curve(fut_curve), discount)
q_view = carry.to_dividend_yield(discount)
```

The algorithm is an **iterative sequential bootstrap**. One pillar per quote,
at the quote's pillar time (duplicates raise `ValidationError` naming both
quotes). Sweep 1 is sequential-truncated: while pillar `i` is solved the curve
holds only pillars `1..i`, so no artificial guess enters the interpolant.
From sweep 2 on every pillar exists and is re-solved in maturity order;
sweeps stop when the largest pillar change is ≤ `SWEEP_TOLERANCE = 1e-14` and
raise `NumericalError` past `max_sweeps` (default 50). Each pillar is a
`brentq` root in its zero level (`PILLAR_XTOL = 1e-16`, `PILLAR_RTOL = 4·eps`,
`PILLAR_MAXITER = 200`, bracket `[−1, 1]` widened ×2 up to four times). Local
schemes converge in two sweeps; cubic and monotone-convex need a few more.
There is no fallback: a pillar that cannot be bracketed raises.

Residual definitions (`InstrumentStrip.residual(discount, projection)`):

- deposit: `P(t_start) − P(t_end)·(1 + r·τ)` on the **projection** curve (a
  deposit pins the projection pseudo-DF, like an index fixing; in a discount
  bootstrap both curves are the one under construction);
- swap: `Σ fixed_k·DF(t_pay_k) − Σ L_k·dcf_k·DF(t_pay_k)` with
  `L_k = simple_forward_rate(projection, t_start_k, t_end_k, dcf_k)`;
- bond: `Σ cf_k·DF(t_pay_k) − (clean + accrued)` per 100 face.

Bootstrapped curves set `node_roles = (CALIBRATED,)*n`,
`last_observable_tenor = T_n` and the run's `valuation_date`. Risk on them is
pillar-level: `with_pillars`, `parallel_shifted`, key-rate buckets.

Carry nodes are closed-form: futures `B = ln(F/S)`; option pairs
`F = K + (C − P)/DF(T)` (then `B = ln(F/S)`), `F ≤ 0` raising before any log.
One quote per expiry.

### QuantLib cross-check

`test/test_param/test_curve_bootstrap_quantlib.py` (skipped without QuantLib)
compares the dated bootstrap with `PiecewiseLogLinearDiscount` and
`PiecewiseLinearZero` under matched conventions — `Actual365Fixed`,
`NullCalendar`, `Unadjusted`, settlement 0, annual legs on a 1Y index — and
requires every pillar and mid-point discount factor to agree within 1e-10.
The conventions needed on our side are `DepositQuote(day_count=ACT_365)` and
`SwapQuote(payment_frequency=ANNUAL, calendar_type=CalendarType.NONE,
business_day_convention=UNADJUSTED, fixed/float day count ACT_365)`.

## The projection convention (engine change)

Every floating projection in the library is now the **simple forward on the
projection curve over the window the index covers**, in the leg's own
day-count basis:

```
L = (P(t_a)/P(t_b) − 1) / dcf_w
```

`simple_forward_rate(curve, t_start, t_end, dcf=None)` in
`quantark.param.rrf.projection` implements it and is the single call site for
`FloatingLeg`, `FloatingRateBond`, `FRAEngine`, `CapFloorEngine`, the
`SwaptionEngine` zero-annuity fallback and the bootstrap strips.

- `IN_ADVANCE` flows and every overnight (compounded) index: window = the
  accrual period `[t_s, t_e]`, `dcf_w` = the coupon's day-count fraction.
- A term index reset `IN_ARREARS`: window = the index period
  `[t_fix, t_fix + tenor]`, `dcf_w` = its axis length, no convexity adjustment.
- A running period (`t_start < 0`): `L = (1/P(t_e) − 1)/(dcf·t_e/(t_e − t_s))`,
  the pro-rata spot rate in the leg's basis.
- A still-payable coupon whose accrual has fully elapsed with no stored fixing
  raises `MarketDataError` naming the index, fixing date and accrual period;
  already-paid flows are skipped. With the fixing stored, it is used.

The previous code plugged the *continuously-compounded* forward over
`[t_fix, t_fix + tenor]` (for overnight indices: the zero rate at the fixing
date) into `N·L·dcf`. How much values move depends on the index:

| index type | old → new | typical size |
|---|---|---|
| term index on the axis day count (ACT/365) | compounding convexity only, `≈ f²τ/2` | ≈ 1 bp at 3 % / 3M |
| term index on ACT/360 | convexity plus the day-count basis term `≈ −f·5/365` | −3.8 bp at 5 % / 3M (+3.1 convexity, −6.9 basis) |
| overnight compounded leg on a sloped curve | zero rate to the fixing date → forward over the period; first-order in the slope | ≈ 200 bp on a 10Y quarterly coupon with `z(t) = 2 % + 0.2 %·t` |

Equity, FX, credit, bond-discount and all golden outputs have no floating
projection and are bitwise unchanged. Two pre-existing rate tests encoded the
shortcut and were reconciled to the exact formulas.

## Open items

- **`ZeroCurve` and the `InterpolatedRateCurve`-only risk paths.** Eight sites
  type-check for `InterpolatedRateCurve` and therefore reject or bypass a
  bootstrapped curve: `quantark/sacva/exposure/asof.py` silently skips the
  forward-curve roll-down for a `ZeroCurve` (with `include_ir_delta=False`,
  SA-CVA future-date exposures on a sloped bootstrapped curve are discounted
  on the un-rolled curve — the first one to fix); the stress applicators
  (`stresstest/stress/stress_applicator.py`, `stresstest/fx/fx_stress_applicator.py`,
  `stresstest/credit/credit_stress_applicator.py`), `var/fx/revaluation.py`,
  `sacva/exposure/curves.py` and `sacva/sacva_engine.py` raise `ValidationError`
  (the SA-CVA engine with a misleading "flat curve" message). Convert a
  `ZeroCurve` to `InterpolatedRateCurve(pillars)` for those paths until they
  accept any `RateCurve` with `with_pillars`.
- `FRAEngine` never consults `fra.fixing_store`; a FRA whose accrual has
  started is projected from the curve (`t_start < 0` branch).
- `SwaptionEngine.forward_swap_rate` on a swaption whose underlying swap has
  already ended returns 0.0 (it used to return a number from the old
  shortcut); `price` is unaffected (`is_expired` returns 0 first). The
  swaption par-rate numerator (`notional·(DF_start − DF_end)`) is untouched.
- `FloatingRateBond` discount-margin pricing still reads `get_rate(0.25)` as
  its reference-rate heuristic (pre-existing; not a projection).
- A term index reset in arrears is projected without convexity adjustment;
  the omitted term is roughly `F²σ²Tτ/(1 + Fτ)`: sub-bp short-dated, several
  bp on long-dated in-arrears legs.
- `create_calendar` holiday files cover 2020–2030; dated examples keep
  maturities ≤ 2030 or use `CalendarType.NONE`.
- Monotone-convex forward continuity at a knot with exactly equal consecutive
  discrete forwards (see above).
