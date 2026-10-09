# FX TARN / TARF mid-life booking, dated form and ageing — design

Status: **rev 2**, 2026-10-09, workstream W4b, branch `w4b-tarn-spec` from main `dcea77b6`. Rev 1 (`189df24a`)
applied decision **A1** and asked Q1–Q6. Rev 2 folds in the user's answers (§1.2), adds the dated TARN/TARF (§4) and
the ageing of both products in the FX backtest and FX dynamic scenario (§7), and replaces rev 1's open questions with
the new ones this investigation raised (§12). No library code is changed by this spec.

Evidence, all run on `dcea77b6` in this worktree (`env PYTHONPATH=<wt>:<wt>/test <main venv>/bin/python <script>`):

| Script (`2026-10-09-fx-tarn-lifecycle-evidence/`) | Measures |
|---|---|
| `measure_tarn_tarf_ageing.py` / `.out` (rev 1) | M1–M6 |
| `measure_backtest_freeze.py` / `.out` (rev 2) | M7, M8: the real `FXBacktestEngine` / `FXDynamicScenarioEngine` over a flat market |
| `measure_clocks.py` / `.out` (rev 2) | M9–M11: theta clock, year-fraction additivity, twin time ULPs, accrual day count |

Origin: findings `2026-10-08-accrual-scope-followups-findings.md` §10.

## 1. User decisions (binding)

### 1.1 A1 (2026-10-09)

- **A1.1** A mid-life TARN booking states its quoted periods and its realized state. A contract rebuilt mid-life
  prices exactly like the same contract aged from inception (numeric-vs-aged parity, §10 P-tests).
- **A1.2** Until a contract states these terms, rebuilding it mid-life is REFUSED. No start is invented.
- **A1.3** `time_shift` drops passed fixings after banking their effect in the realized state. For theta they fix
  at the held spot, as the FX range accrual does (`BaseFxProduct.record_passed_fixings`).
- **A1.4** The TARF gets the same treatment where measurement shows it needs it (§2.3).

### 1.2 Answers to rev 1's Q1–Q6 (2026-10-09)

| # | Decision | Where |
|---|---|---|
| Q1 | `state: TargetRedemptionState` is **mandatory** on TARN and TARF, both forms. Inception code writes `TargetRedemptionState(0.0)`. | §3, §4 |
| Q2 | (a) A **numeric** TARN must give `accrual_start` or explicit `accrual_factors`. A **dated** TARN derives its period starts from its dates (`initial_date`, then each previous fixing date). | §4.2, §5.1 |
| Q3 | (a) A fixing stepped over by theta or ageing is fixed at the **quoted** `fx_env.spot`, as the range accrual's `record_passed_fixings` and the W5 missing-fixing fill do, not at `effective_spot()`. | §6 |
| Q4 | Scope expanded: (1) build the **dated** TARN and TARF; a numeric contract must show no difference from its dated twin (same events, same cash timing, same PV to float noise). (2) Option (b): the FX backtest and FX dynamic scenario **age** numeric TARN/TARF each simulated day with `time_shift` + `record_passed_fixings` at that day's spot. Other numeric FX products also frozen by M5 are only listed (§9), not designed for. | §4, §7, §9 |
| Q5 | (a) No knocked-out flag. `is_redeemed` is derived: `accumulated >= target - hit_tol(target)`, with the engines' hit tolerance moved to the product. | §3.1 |
| Q6 | (a) An already-redeemed (terminal) booking is **allowed**: value = PV of `pending`, or 0 if nothing is pending, and it emits a `UserWarning` subclass. An aged copy that redeems inside a run does not warn. | §5.3 |

### 1.3 Standing rules this spec applies

- **Numeric == dated twin** (user rule 2026-10-09): identical events, ledger and PV; tested in both forms (§10).
- **Past observations without carried state:** a missing past fixing is filled at the current (quoted) spot with a
  warning and never raises (W5, branch `w5-missing-fixings` `bfb9b88f`: `quantark/util/fixings.py`,
  `MissingFixingWarning`, `warn_filled_fixings`). A contract that matured before a run's start warns and is valued
  at zero with no cash (W1, branch `w1-tracker-clock` `510bf6a7`: `MaturedBeforeRunWarning`).
- **Theta excludes only BOOKED cash:** cash paid within the bump is not in the value, so theta on a payment day is
  roughly minus the payment.
- Products stay engine-agnostic; tolerances come from `quantark/util/numerical`; errors are `ValidationError`.

## 2. Problem (measured on dcea77b6)

### 2.1 Today's code

- `FxTargetRedemptionNote.__init__` defaults the accrual factors to `t_i - t_{i-1}` with `t_0 = 0`: the first
  period is assumed to start at the valuation instant.
- Neither product has realized state; both engines start every path at `acc = 0`, `alive = True`.
- `time_shift` subtracts the bump from fixing and pay times and from `maturity`/`delivery`; it keeps past fixings.
- `record_passed_fixings` is the base-class no-op.
- Neither product has a dated form: both take only year-fraction `fixing_times` / `pay_times`.
- Both engines (`FxTargetRedemptionNoteMCEngine`, `FxTarnForwardMCEngine`) are constant-vol Garman-Kohlhagen Monte
  Carlo: they simulate every listed fixing (`FxGKPathGenerator` requires times > 0) and discount every pay time.
- Theta is `BaseFxEngine._fdm_theta` (`base_fx_engine.py:219-241`): a numeric product (`maturity is not None`) is
  `deepcopy`'d and `time_shift(days/365)`'d; a dated one keeps its terms and the env date moves by `days` calendar
  days; then `record_passed_fixings(env)` and reprice. A grep of `quantark/asset/fx`, `portfolio/fx`, `execution`,
  `var/fx`, `stresstest/fx`, `backtest/fx`, `dynamicscenario/fx` and `pnlexplain` finds no other caller of
  `time_shift`.
- The FX backtest and FX dynamic scenario move `env.valuation_date` only (§7.1).

### 2.2 Measurements

Fixture M1–M6: twelve monthly fixings at `k/12`. TARN: notional 100, coupon 12%, strike 1.10. TARF: strike 1.15,
notional 1. EURUSD spot 1.20, `r_d` 5%, `r_f` 3%, vol 10%, 200,000 PSEUDO paths, seed 7.

| # | What | Result |
|---|---|---|
| M1 | TARN rebuilt from its 11 remaining times after 0.125y | first accrual factor **0.041667** instead of 0.083333: first coupon **0.50 instead of 1.00**. `time_shift(0.125)` keeps the factors (correct) and all 12 fixings, the first at t = −0.041667. |
| M2 | Price after `time_shift(0.125)` + `record_passed_fixings` | TARN and TARF both **raise** `ValidationError: Time to maturity must be non-negative, got -0.0417` (DF of the past pay time, `fx_pricing_environment.py:187`). |
| M3 | Theta, first fixing 0.5/365 away | both **raise** (the same DF error, on t = −0.00137) |
| M3 | Theta, first fixing exactly 1/365 away | both **raise** `times must be strictly increasing and positive` |
| M3 | Theta, first fixing 2/365 away | finite: TARN 0.016861, TARF 0.003077 |
| M4 | Two months aged, two coupons paid; remaining 10 fixings priced with the full target vs the remaining target | TARN with principal 101.652635 vs 100.578131 (**+1.07**); without principal 2.957940 vs 0.995609 (**+1.96**); TARF (target 0.20, 0.10 realized) 0.091663 vs 0.027251 (**+0.064**). |
| M5 | `env.valuation_date` moved 91 days, product unchanged | `get_maturity` stays 1.000000; price unchanged to the bit (TARN 101.643148, TARF −0.011326). |
| M6 | Accrual factors derived from shifted times vs the inception factors (11 periods) | τ = 0.125 bitwise; τ = 0.1, 45/365, 91/365: 2, 4, 2 periods differ by ≤ **1.11e-16**. |
| M7 | The real `FXBacktestEngine` and `FXDynamicScenarioEngine` (dated `DayPath`, `start_date` 2026-06-15), flat market, 92 days, per-position value on day 0 vs day 91 | Valuation date moves 2026-06-15 → 2026-09-14. **Every numeric FX product is bitwise frozen** in both engines: vanilla 0.058424, digital 0.532325, quanto vanilla 0.058495, barrier 0.014386, sharkfin 0.014386, range accrual 0.029024, quanto digital 0.539241, quanto range accrual 0.029953, foreign range accrual 0.035244, TARN 101.645083, TARF −0.011336. The dated vanilla control moves 0.058424 → 0.049584. One-touch (needs a Vanna-Volga surface to price): `get_maturity` 1.0 → 1.0. |
| M8 | The same dynamic scenario with an **undated** `DayPath` (`start_date=None`) | The env date never moves (2026-06-15 on day 91); **the dated vanilla is frozen too**; `total_pnl` 0.000000. Every `FXPathLibrary` builder returns an undated path (`path_library.py:38, 59, 76, 90`). |
| M9 | `_fdm_theta` (1 day), numeric vs dated FX vanilla (expiry V+365d), valuation on a Friday | `CALENDAR_DAYS` env: numeric −9.193e-05, dated −9.193e-05. `BUSINESS_DAYS` env (TARGET calendar): numeric **−9.135e-05**, dated **0.000e+00** (Friday → Saturday adds no business day; the numeric step is always 1/365). |
| M10 | Sum of 91 daily year fractions vs the direct one | `CALENDAR_DAYS`, `BUSINESS_DAYS` (TARGET), `ACT_365`, `ACT_360`, `THIRTY_360_US`, `ACT_ACT_ISDA`: all differ by ≤ **5.0e-16**. Numeric times aged by 91 daily shifts of 1/365 vs dated times re-resolved each day (12 monthly dates): max difference **1.11e-15**, 988 (day, fixing) pairs not bitwise. |
| M11 | Monthly accrual factors `ACT_365` vs `CALENDAR_DAYS` | bitwise equal (max difference 0). |

Reading of the results:

- **Rebuilding mid-life (M1, M4).** A rebuilt TARN gets its first period's accrual wrong, and neither product can say
  how much has already accrued: a mid-life booking is wrong by about one coupon, or by the whole remaining target.
- **Theta (M2, M3).** Theta cannot step over a fixing; within one day of a fixing it raises.
- **No run ages these products (M5, M7, M8).** The FX backtest and dynamic scenario move only
  `env.valuation_date`, and every numeric FX product ignores it: each day is priced as if it were the booking date,
  at that day's market. A TARN in a 92-day backtest never fixes, never pays and never redeems. An undated dynamic
  scenario freezes dated products as well.
- **Twin arithmetic (M6, M10, M11).** A numeric twin aged day by day and its dated twin re-resolved each day agree to
  ~1e-15 in time, not bitwise; parity is therefore stated as bitwise at the twin's booking date and to a stated
  relative tolerance after ageing (§10).
- **Theta clock (M9).** Under a non-calendar env day count the numeric and dated FX theta steps differ. This is shared
  FX code; §12 Q7.

### 2.3 TARF: needs the realized state, not the accrual term (A1.4)

The TARF's favourable amount per fixing is `notional × max(±(S_i − K), 0)`: no period length, so M1 cannot occur.
The other defects do (M2, M3, M4: +0.064 on a value of 0.027). **Decision (rev 1, unchanged):** the TARF gets the
same realized state, dated form, ageing and refusals as the TARN, and no accrual term.

## 3. Shared state: `TargetRedemptionState`

New module `quantark/asset/fx/product/option/target_redemption_state.py`; both products import it.

```python
TARGET_HIT_RTOL: Final[float] = 1e-12        # moved from the two engines' _HIT_RTOL, value unchanged

@dataclass(frozen=True)
class PendingPayment:
    amount: float                            # undiscounted domestic cash a banked fixing has determined
    pay_time: Optional[float] = None         # numeric form: years from valuation
    pay_date: Optional[datetime] = None      # dated form

@dataclass(frozen=True)
class TargetRedemptionState:
    accumulated: float                             # undiscounted amount banked towards the target
    pending: Tuple[PendingPayment, ...] = ()       # determined, unpaid cash, ascending pay time / date
    as_of: Optional[datetime] = None               # dated form only: every fixing dated on/before it is banked

class TerminalBookingWarning(UserWarning): ...     # §5.3 (Q6)

class TargetRedemptionEventType(Enum):
    FIXING = "fixing"       # a fixing was banked
    PAYMENT = "payment"     # determined cash reached its pay time / date

@dataclass(frozen=True)
class TargetRedemptionEvent:
    event_type: TargetRedemptionEventType
    amount: float           # FIXING: cash it determined (coupon / net cashflow, principal excluded); PAYMENT: cash paid
    fixing: Optional[float] = None    # FIXING: the rate used
    time: float = 0.0       # FIXING: the fixing's signed time at banking (<= 0); PAYMENT: the pay time (<= 0)
    redeemed: bool = False  # FIXING: this fixing ended the contract (target hit, or the last fixing)
    label: str = ""         # W5 ``fixing_label`` style, informational only
```

- `accumulated`: the TARN's realized coupons or the TARF's realized favourable amounts, including those fixed but not
  yet paid.
- `pending`: cash a banked fixing has determined whose pay time is after valuation — a coupon or TARF net cashflow
  with a pay lag, or the principal of a TARN that redeemed or matured. Cash whose pay time is at or before valuation
  is booked and is not part of the value (theta-boundary R4).
- **Validation in the state** (`ValidationError`): `accumulated` finite and ≥ 0; each `amount` finite; each
  `PendingPayment` has exactly one of `pay_time` / `pay_date`; all pending entries use the same form; sorted
  ascending; a numeric `pay_time` is finite, > 0 and not `is_zero`; a dated `pay_date` is after `as_of`, and
  pending cash with `as_of=None` is refused (cash implies a banked fixing).
- **Form check in the product**: a numeric product refuses `as_of` or any `pay_date`; a dated product refuses any
  `pay_time`.
- The state is immutable; ageing replaces it, never mutates it.
- **No knocked-out flag (Q5).** The products derive `is_redeemed = accumulated >= target - hit_tol(target)` with
  `hit_tol(target) = 0` for an unbounded target, else `|target| × TARGET_HIT_RTOL`. Both engines import the same
  constant, so product and engines cannot disagree. The constant keeps today's value `1e-12` rather than
  `Tolerance.RELATIVE` (1e-9) because changing it would move redemption decisions and inception prices.

**Lifecycle log.** Each product carries `lifecycle_events: Tuple[TargetRedemptionEvent, ...]` (empty at
construction, not a booking term, ignored by the engines). `time_shift` and `record_passed_fixings` append to it;
the backtest/dynamic-scenario hook reads the entries appended in a step to book cash (§7). It survives `deepcopy`.

## 4. The two forms

Each product is one class with two mutually exclusive schedule forms, as `FxRangeAccrualOption` has
`maturity`/`observation_times` vs `expiry_date`/`observation_records` dates.

### 4.1 Numeric form (fixing times)

`FxTargetRedemptionNote(fixing_times, coupon_rate, ..., *, state)` and
`FxTargetRedemptionForward(strike, fixing_times, ..., *, state)`. `fixing_times` stays the first positional
schedule argument; `state` is keyword-only and has no default (Q1).

| Field | Product | Rule |
|---|---|---|
| `fixing_times` | both | Years from valuation, ascending, unique, finite. Lists **only the fixings not yet banked**. A time ≤ 0 or `is_zero` is an unbanked past fixing: accepted, and filled at pricing with a `MissingFixingWarning` (§5.4). May be empty: the contract is terminal (§5.3); a terminal TARN then states `accrual_factors=[]`. |
| `pay_times` | both | As today (default = fixing times; each ≥ its fixing; non-decreasing). |
| `accrual_start: Optional[float]` | TARN | Time the first LISTED period began (< 0 if before valuation). **Exactly one** of `accrual_start` / `accrual_factors` (Q2): neither → refused (A1.2), both → refused. `accrual_start < fixing_times[0]`. Factors `t_i − t_{i−1}` with `t_{−1} = accrual_start` are computed once at construction and are contract terms from then on. |
| `accrual_factors` | TARN | Explicit per-listed-period factors (as today). |
| `state` | both | Numeric `PendingPayment.pay_time` only; `as_of` must be None. |

Consistency rules (rev 1, kept): `accumulated <= target + hit_tol`; `accrual_start > 0` with `accumulated > 0` is
refused (periods are contiguous); a TARN pending amount must be > 0 (TARF pending may have either sign: a geared
unfavourable fixing is a liability); a redeemed state with listed fixings is refused (ageing drops dead fixings, so
the aged path never produces one). `maturity` = last listed fixing, or `max(pending pay times)` / `0.0` when the list
is empty; `delivery` = `max(pay_times)` or the same terminal value.

### 4.2 Dated form (fixing dates) — new

| Field | Product | Rule |
|---|---|---|
| `fixing_dates: List[datetime]` | both | The **whole contractual schedule**, past and future, ascending, unique, non-empty. Dates are taken as stated (already business-day adjusted); no rolling. |
| `pay_dates: Optional[List[datetime]]` | both | Default = fixing dates; 1:1, each ≥ its fixing date, non-decreasing. |
| `initial_date: Optional[datetime]` | TARN | Start of the first period, `< fixing_dates[0]`. **Exactly one** of `initial_date` / `accrual_factors` (Q2: the dated form derives every period start from its dates). |
| `accrual_factors` | TARN | Explicit factors, 1:1 with the whole schedule. |
| `day_count_convention: Optional[DayCountConvention]` | TARN | The coupon accrual day count, a contract term; `None` means `ACT_365` (the `FxRangeAccrualConfig` default). Refused on the numeric form and with explicit `accrual_factors`. |
| `state` | both | Dated `PendingPayment.pay_date` only; `as_of` = the date through which fixings are banked (`None` = nothing banked). |

- **Mixing forms is refused**: exactly one of `fixing_times` / `fixing_dates`; `pay_times`, `accrual_start` only with
  times; `pay_dates`, `initial_date`, `day_count_convention` only with dates.
- **Banked vs unbanked**: fixing `i` is banked iff `as_of is not None and fixing_dates[i] <= as_of`. The rest are
  unbanked; an unbanked fixing dated on or before the valuation date is a missing past fixing (§5.4).
- **Accrual factors**: period `i` runs from `initial_date` (i = 0) or `fixing_dates[i-1]` to `fixing_dates[i]`;
  factor = `calculate_year_fraction(start, end, day_count_convention, fx_env.bus_days_in_year,
  calendar=fx_env.calendar)`, evaluated at resolve time because a `BUSINESS_DAYS` count needs the environment's
  calendar (the range accrual's `get_year_fraction` precedent). They depend on dates only, so ageing never changes
  them.
- **Times**: each fixing and pay date resolves to a signed year fraction from `fx_env.valuation_date` in the
  **environment's** convention (`day_count_convention`, `bus_days_in_year`, `calendar`), exactly as
  `FxRangeAccrualObservationRecord.resolve_time` does: 0.0 on the valuation date, negative before it.
- **Consistency rules**: `accumulated <= target + hit_tol`; `accumulated > 0` or non-empty `pending` needs
  `as_of >= fixing_dates[0]`; `is_redeemed` needs the same; TARN pending amounts > 0. At resolve time
  `as_of > fx_env.valuation_date` is refused (the state claims fixings that have not happened). Fixings after `as_of`
  on a redeemed contract are dead, not refused: the dated form lists the whole term sheet.
- **Maturity / delivery**: `expiry_date = fixing_dates[-1]`, `delivery_date = max(pay_dates)`, `maturity = None`
  (so `_fdm_theta` takes its dated branch). `get_maturity(fx_env)` returns the base-class value while the valuation
  date is on or before `expiry_date` and `0.0` after it, instead of raising, so an aged dated contract still answers
  FX consumers such as SIMM's `option_tenor`.
- **Terminal**: `is_redeemed`, or `as_of >= fixing_dates[-1]`.

### 4.3 The numeric twin

The twin of a dated contract `D` at valuation date `V` (environment `E`) is the numeric contract `N` with:

- the unbanked fixings of `D` (dead fixings dropped when redeemed): `fixing_times = [t_E(V, F_i)]`,
  `pay_times = [t_E(V, P_i)]`;
- TARN: `accrual_factors` = `D`'s resolved factors for those periods (explicit, not `accrual_start`, so the coupons
  are bitwise equal whatever the two day counts are);
- `state = TargetRedemptionState(D.accumulated, pending=[PendingPayment(a, pay_time=t_E(V, d)) for (a, d) in
  D.pending if d > V])`, `as_of=None`;
- every economic term (coupon, notional, target, condition, principal flag, strike, gearing, side, pair) unchanged.

It is a test helper (`_numeric_twin(dated, env)` in the new test module), not public API: the mapping is the parity
contract the tests enforce, not a conversion users need. Under it the two forms resolve to the same engine input
(§5.1), bitwise at `V`.

## 5. Pricing

### 5.1 One resolved view

Both products expose `resolve(fx_env, *, warn_missing_fixings=True) -> ResolvedTargetRedemption`, a frozen record:

| Field | Meaning |
|---|---|
| `fixing_times`, `pay_times` | the unbanked fixings still ahead, all > 0 and not `is_zero` |
| `accrual_factors` | TARN: their factors; TARF: `None` |
| `accumulated` | after banking any missing past fixing |
| `pending` | `(pay_time, amount)` with `pay_time > 0`, not `is_zero` |
| `redeemed`, `terminal` | derived as in §3 / §4 |

`resolve` never changes the product. If it finds unbanked fixings at or before valuation (numeric `t <= 0` or
`is_zero(t)`; dated date ≤ valuation date), it banks them on a **copy** through `record_passed_fixings(fx_env)` at
the quoted spot (Q3) and emits one `MissingFixingWarning` naming them (W5's `warn_filled_fixings`). Pending cash
whose pay time is at or before valuation is excluded (booked). A numeric contract with no passed fixing skips the copy
and returns its own lists, so its inception arithmetic is unchanged.

The engines read only this view; the products' other attributes are not engine inputs. Numeric and dated contracts
therefore reach the same arithmetic, which is what makes the twin parity hold by construction.

### 5.2 Engines

Only two engines exist, `FxTargetRedemptionNoteMCEngine` and `FxTarnForwardMCEngine` (`execution/inventory.py:438-439`).

1. `view = product.resolve(fx_env)`; `pv_pending = Σ amount × DF_dom(pay_time)` over `view.pending`.
2. **Terminal** (`view.terminal`): return `pv_pending` without simulating and without a vol lookup.
   `FxTarnMCResult`: `num_paths = 0`, `std_error = 0.0`, `sigma = 0.0`, `mean_redemption_fraction = 1.0` if
   redeemed else `0.0`.
3. Otherwise `T = view.fixing_times[-1]` for the vol lookup (today `opt.get_maturity`, the same number for a
   numeric inception contract); the walk starts at `acc = np.full(n_paths, view.accumulated)`; `pv_pending` is added
   to every path only when `pending` is non-empty; the hit test uses the product's `hit_tol`.
   At inception (`accumulated = 0.0`, nothing pending) the arithmetic is identical to today's
   (`np.full(n, 0.0) == np.zeros(n)`, no extra add), so every inception price is unchanged to the bit (§8).
4. Rev 1's "refuse pricing an unsettled product" guard is **dropped**: under the W5 rule a passed, unbanked fixing is
   filled with a warning (step 1), never refused.

The `PricingError` on a negative TARN price stays (TARN pending cash is > 0). `BaseFxEngine` is unchanged: its spot,
vol and rate bumps leave `state` alone, and `_fdm_theta` already calls the hook.

### 5.3 Terminal bookings (Q6)

- A booking whose state is terminal is **allowed**: numeric with empty `fixing_times`, or a redeemed state; dated with
  `as_of >= fixing_dates[-1]`, or a redeemed state. Value = PV(`pending`), or 0 if nothing is pending.
- It emits `TerminalBookingWarning` (a `UserWarning` subclass) **once, from the constructor** — the booking. Pricing
  does not re-warn: a terminal product is priced on every Greek bump and every simulated day, and a warning per
  call would only repeat the booking's.
- An aged copy that becomes terminal inside a run is produced by `time_shift` / `record_passed_fixings` on an
  existing object (never by the constructor), so it never warns: redemption is the normal end of the trade.

### 5.4 Refusals and fills

Refused (`ValidationError`): an omitted `state`; an omitted period start on a TARN (numeric: neither `accrual_start`
nor `accrual_factors`; dated: neither `initial_date` nor `accrual_factors`) — M1's exact failure; both period
sources; mixed forms; an inconsistent state (§3, §4); a dated `as_of` after the valuation date.

Filled with a `MissingFixingWarning`, never refused: an unbanked fixing at or before valuation, in either form, at
pricing time. This replaces rev 1's refusal of a numeric fixing ≤ 0, which the W5 rule forbids.

Not detectable from numbers alone: a booking that states a wrong period start or a wrong `accumulated`. Both are
mandatory, so a caller has to write them down; nothing defaults them silently.

## 6. Lifecycle hooks and theta

Both hooks are product methods (the base-class contract; they return `None` and append to `lifecycle_events`).

**`time_shift(dt)`** — numeric form only (the dated form ages by its environment's date; the base class already
leaves `maturity is None` products alone):
- shifts fixing, pay and pending pay times and `maturity`/`delivery`;
- removes the pending payments whose pay time is now ≤ 0 or `is_zero` (paid within the bump), appending a `PAYMENT`
  event for each;
- keeps passed fixings unbanked.

**`record_passed_fixings(fx_env)`** — both forms. For each unbanked fixing at or before valuation, in order, stopping
once the contract is terminal, at `S = fx_env.spot` (Q3):
- **TARN**: `c = full_coupon(i)` if `coupon_condition(S)` else 0, clipped to `target − accumulated`;
  `accumulated += c`. A non-zero `c` paid at `pay_i`: pending if `pay_i` is after valuation, else a `PAYMENT` event.
  On a hit (`is_redeemed`), the principal (with `include_principal`) at `pay_i` is handled the same way and every later
  fixing is dead. If the last fixing banks without a hit, the final principal at its pay time is handled the same way.
- **TARF**: `fav = notional × max(sign × (S − K), 0)` clipped to the remaining target, `cf = fav_eff − gearing × unf`
  (either sign), `accumulated += fav_eff`; non-zero `cf` pending/paid and the hit rule as for the TARN. On a final hit
  the geared unfavourable leg is still paid, as in the engine.
- One `FIXING` event per banked fixing (`time` = its signed time ≤ 0, `redeemed` when it ends the contract).
- Numeric: banked fixings, their pay times and factors are removed. Dated: `as_of := fx_env.valuation_date`, and
  pending entries with `pay_date <= valuation_date` are removed with a `PAYMENT` event each.
- Then `validate()` runs again.

**Theta, then (unchanged from rev 1 except the dated branch):**
- A fixing inside the bump, or landing exactly on it, fixes at the held spot. Cash paid by the bumped date leaves the
  value (theta ≈ −payment); cash paid later stays as a receivable.
- A step past the last fixing prices the terminal contract (§5.2 step 2): theta is pure carry on `pending`.
- Dated: `_fdm_theta` moves the env date by `days` and calls `record_passed_fixings(env_future)`; same banking.
- The landing fixing is banked at the held spot rather than left at t = 0 for the engine (equity theta-boundary R1);
  the value is the same because the spot is held, and the mechanism follows the FX range accrual precedent.
- Under a `CALENDAR_DAYS` or `ACT_365` environment the numeric step (`days/365`) equals the dated step (M9, M11), so
  numeric and dated theta agree. Under other environment day counts they do not today (M9); §12 Q7.

## 7. Ageing in the FX backtest and FX dynamic scenario (Q4 b)

### 7.1 Where the clock moves today and the product does not

| File:line | What happens |
|---|---|
| `quantark/backtest/fx/engine.py:31` | `_clone_portfolio` deep-copies positions (products) and environments: the run works on copies. |
| `quantark/backtest/fx/engine.py:44-45` | `_apply_row(first row)` then `initial_value = portfolio.get_portfolio_value()`. |
| `quantark/backtest/fx/engine.py:57-58` | Each row: `_apply_row(portfolio, pairs, row, ts)`. |
| `quantark/backtest/fx/engine.py:138-157` | `_apply_row` replaces each pair's env with `valuation_date=timestamp` (line 141) and the row's spot / vol / rates. **The products are not touched.** |
| `quantark/backtest/fx/engine.py:66, 99, 117` | `value = get_portfolio_value()`; `net_pnl = portfolio_pnl + hedge − costs`; `final_value`. There is no cash ledger. |
| `quantark/dynamicscenario/fx/engine.py:79-80` | Clone; `baseline_value` at the envs' own valuation dates. |
| `quantark/dynamicscenario/fx/engine.py:84-89` | Each step: `_apply_day_changes`, then `env.valuation_date = day_date` **only if** `DayPath.start_date` is set (`day_path.py:298-310`). **The products are not touched.** |
| `quantark/dynamicscenario/fx/engine.py:91, 99-109, 112` | `value = get_portfolio_value()`; `DayResult` built without its cash fields (they keep their 0.0 defaults); `final_value`. |

M7 measures the consequence for every numeric FX product; M8 shows an undated path freezes dated products too.

### 7.2 The hook: `FxTargetRedemptionLifecycle`

New module `quantark/asset/fx/lifecycle/target_redemption.py` (mirroring `quantark/asset/equity/lifecycle/`). It acts
only on positions whose product is an `FxTargetRedemptionNote` or `FxTargetRedemptionForward`; every other position
is left exactly as today (§9).

- **`register(portfolio)`** — called on the run's working copy before the baseline is valued. Per ageable position:
  1. **Clock origin** = its pair environment's `valuation_date` at that moment (backtest: the first row's date,
     after line 44; dynamic scenario: the environment's own date, before line 80). A numeric product's times are read
     as years from this date — the date the run first values it, as the equity `FloatMaturityRoller` anchors a numeric
     maturity at first sight (`float_roll.py:51-71`).
  2. **Matured before the run** (W1 rule): the last fixing is strictly before the origin (numeric: last listed time
     < 0 and not `is_zero`; dated: last fixing date < origin) **and** is not banked by the state → one
     `MaturedBeforeRunWarning`; the position is removed from the working copy (value 0, no cash, no Greeks).
     A terminal booking (§5.3) is not this case: it is kept and valued at PV(pending).
  3. **Fixings before the run**: unbanked fixings strictly before the origin are banked at the origin's spot through
     `record_passed_fixings`, with one `MissingFixingWarning` per position naming them (W5 rule). A fixing on the
     origin date itself is banked at that spot **without** a warning: on the run's own dates the path's spot is the
     realized fixing.
  4. `PAYMENT` events produced by registration are **not** booked: that cash was paid on or before the baseline and
     is neither in the baseline value nor in the run's P&L.
- **`advance(portfolio, date) -> FxLifecycleDay`** — called each simulated day after the day's market is applied and
  the environment carries `date`. Per ageable position, with `prev` its last clock date:
  1. `date < prev` → `ValidationError` (a run's clock cannot go back).
  2. Numeric: `dt = calculate_year_fraction(prev, date, env.day_count_convention, env.bus_days_in_year,
     calendar=env.calendar)` (0.0 when equal); `time_shift(dt)` unless `is_zero(dt)`. The step uses the environment's
     convention because that is how the dated twin's times are resolved; daily steps telescope to the direct year
     fraction to ≤ 5e-16 (M10).
  3. `record_passed_fixings(env)`: every fixing at or before `date` is banked at **that day's spot** (the run's
     realized fixing), and dated `as_of` becomes `date`.
  4. The events appended in steps 2–3 are the day's: `paid_today += quantity × Σ PAYMENT.amount`.
  5. `prev := date`.

  `FxLifecycleDay` carries `paid_cash` (quantity-scaled, today), `events` (position id, event) and
  `pending_pv` (Σ quantity × PV of each position's pending at the day's environment).
- **Calendar gaps.** A fixing whose date (or time) falls between two path days is banked on the first path day on or
  after it, at that day's spot, without a warning; a payment due between path days is booked, undiscounted, on the
  first path day on or after its pay date. Numeric fixing times rarely fall on day boundaries (k/12 years is
  30.42 days), so a warning here would fire on ordinary trades; both forms get the identical rule, keeping parity.
  The FX backtest has no cash-interest account (its hedge P&L has none either), so booked cash earns nothing.
- **Pending cash** stays inside the product's value as PV(pending) until its pay date; on that day it leaves the
  value and enters paid cash. Run value = Σ position MTM (which includes pending PV) + cumulative paid cash — the
  equity convention (`dynamicscenario/lifecycle_manager.py:11-13`), so P&L does not jump on a payment.

### 7.3 Changes at the call sites

**FX backtest (`quantark/backtest/fx/engine.py`):**
- after line 44: `lifecycle = FxTargetRedemptionLifecycle(); lifecycle.register(portfolio)`; line 45 then values the
  aged book;
- after line 58: `day = lifecycle.advance(portfolio, ts)`; `cum_paid += day.paid_cash`. On the first row `ts` is the
  registration date, so the call is a no-op;
- line 66: `value = portfolio.get_portfolio_value() + cum_paid`; the row gains `paid_cash` (cumulative) and
  `pending_pv`; Greeks and hedging run on the aged products;
- line 117: `final_value = portfolio.get_portfolio_value() + cum_paid`.
- `FXBacktestResults` gains nothing else; for a book without TARN/TARF `cum_paid` stays 0.0 and every number is
  unchanged to the bit.

**FX dynamic scenario (`quantark/dynamicscenario/fx/engine.py`):**
- after line 79: register; line 80 values the baseline;
- after line 89, when `day_date` is not None: `day = lifecycle.advance(working, day_date)`;
- line 91: `value = working.get_portfolio_value() + cum_paid`; `DayResult` fills the existing `paid_cash`,
  `realized_cash`, `pending_receivable_pv` (`dynamic_results.py:224-227`) and `lifecycle_events`: one
  `LifecycleEventSnapshot` per `FIXING` event, `event_type="FIXING"` (a new value next to `'KO'`, `'COUPON'`, …),
  `spot` = the fixing, `payoff` = the amount it determined, `cashflow` = the quantity-scaled amount paid on the day,
  `terminates_position` = `redeemed`;
- line 112: `final_value` includes `cum_paid`.
- **Undated path** (`start_date is None`): no date to age by (M8). §12 Q8 decides; until then the spec's behaviour
  is Q8's recommendation: before `register`, `run` raises a `ValidationError` naming `DayPath.start_date` when the
  book holds a TARN/TARF.

**Dated TARN/TARF** need no clock of their own: the environment's date is the clock. The same `advance` call records
their passed fixings at the day's spot into the state (step 3), so numeric and dated run through identical events.

## 8. Compatibility (measured by grep and the gates on dcea77b6)

**Construction sites** — each needs `state=TargetRedemptionState(0.0)`; the TARN sites also `accrual_start=0.0`.
Numbers unchanged.

| Site | Product | Today |
|---|---|---|
| `test/test_fx_tarn_note_mc.py::_note` (every TARN test) | TARN | 27 tests |
| `test/test_fx_tarf_mc.py::_tarf` | TARF | 25 tests (52 pass together, 1.3 s, `-n 2`) |
| `test/execution/matrix_fixtures.py` `fx_tarn`, `fx_tarf` | both | `test_matrix_parity.py -k "Tarn or TargetRedemption"`: 2 pass |
| `example/fx_tarn_demo.py` (lines 74, 85, 97, 109) | 2 TARF + 2 TARN | demo stdout |

- `test_default_accruals_are_period_lengths` passes `accrual_start=0.0` and asserts the same `[0.5, 0.5, 0.5]`.
- `test_rejects_*` keep their messages. A non-positive fixing time and an empty `fixing_times` stop being refused
  (an unbanked past fixing, §5.4; a terminal booking, §5.3); no existing test asserts either refusal (grep of both
  test files), so none changes.
- No golden in `test/execution/goldens/` contains either engine (grep count 0).
- `test/test_fx_backtest.py` and `test/test_fx_dynamic_scenario.py` hold no TARN/TARF: unchanged and must pass
  unmodified (e.g. `test_flat_path_has_zero_pnl` keeps a frozen numeric vanilla, M7).
- **Expected to move:** theta within one day of a fixing (exception → number, M3); construction without the period
  start or `state` (accepted → refused); pricing after a bare `time_shift` (exception → filled with a warning, M2);
  a backtest or dated dynamic scenario holding a TARN/TARF (frozen → aged, with paid cash). That is the whole break;
  release notes must list it (both products are public via `quantark.asset.fx.product.option`).
- **Expected not to move:** every inception price and Greek, theta whenever no fixing is in the bump, and every
  backtest / dynamic-scenario number of a book without TARN/TARF — all bitwise. The gate is the 52 + 2 tests, the two
  FX run test files, and a byte diff of the demo's stdout against main.
- **Merge order:** this work imports W5's `quantark.util.fixings` and W1's `MaturedBeforeRunWarning`; both branches
  land first. W1 defines the warning in `quantark/asset/equity/lifecycle/autocallable.py`; the FX lifecycle imports
  it from there; it is not moved.

## 9. M5 finding: the other frozen FX products (for a later decision; not designed here)

Measured in M7 with the real engines over 92 days (dated `DayPath`, and the backtest): the following **numeric**
(`maturity=`) FX products keep their booking-date value to the bit while the valuation date moves, through
`FXBacktestEngine._apply_row` (`backtest/fx/engine.py:138-157`) and `FXDynamicScenarioEngine.run`
(`dynamicscenario/fx/engine.py:87-89`), because both move only `env.valuation_date` and a numeric product's times (its `maturity` through
`BaseFxProduct.get_maturity`, its observation or fixing times) never read that date:

`FxVanillaOption`, `FxDigitalOption`, `FxQuantoVanillaOption`, `FxQuantoDigitalOption`, `FxBarrierOption`,
`FxSharkfinOption`, `FxOneTouchOption` (structural: `get_maturity` unchanged; it prices only on a Vanna-Volga
surface), `FxRangeAccrualOption`, `FxQuantoRangeAccrualOption`, `FxForeignRangeAccrualOption`.

The FX delta-one products (`FxSpot`, `FxForward`, `FxSwap`) take dates only and are not affected. On an undated
`DayPath` (every `FXPathLibrary` path) the dated forms of all of these are frozen as well (M8). The equity side's
answer for numeric vanillas is `FloatMaturityRoller` plus a once-per-position warning for un-rollable numeric
products (`asset/equity/lifecycle/manager.py:168-184`); the FX side has neither.

## 10. Acceptance tests

New file `test/test_fx_target_redemption_lifecycle.py`; backtest/scenario tests in
`test/test_fx_target_redemption_runs.py`. MC tests use the same seed on both sides and the same remaining fixing
times. "Relative 1e-12" means `is_close(a, b, rel_tol=1e-12)` from `quantark.util.numerical`; the bound comes from
M6/M10 (times differ by ≤ 1.1e-15).

**Numeric-vs-aged parity (A1.1)**
- **P1 TARN.** Inception C0 (§2.2 fixture, `accrual_start=0.0`) aged by `time_shift(τ)` + `record_passed_fixings(env
  at S_h)` vs R rebuilt from the remaining times, the aged state and (a) C0's surviving `accrual_factors`: bitwise
  for τ ∈ {0.125, 91/365}, S_h above and below the strike, PSEUDO and QUASI; (b) `accrual_start = 1/12 − τ`: bitwise
  at τ = 0.125, relative 1e-12 at 91/365.
- **P2** pay lag 0.02: R's stated pending equals the aged pending; prices as P1.
- **P3** across redemption: P2 lag, target 1.5 coupons, two banked fixings hit it (second coupon clipped to 0.5);
  aged product terminal, value = PV(pending principal + clipped coupon); a rebuilt terminal booking with empty
  `fixing_times` prices equal (and raises its `TerminalBookingWarning`, WARN1).
- **P4 TARF**: P1–P3 including a negative pending cashflow (S_h below K, gearing 2).
- **P5** 91 daily `time_shift(1/365)` + record steps vs one rebuild: relative 1e-12.

**Numeric-vs-dated twin parity (Q4.1)** — every case builds the dated contract `D` and its twin `N = _numeric_twin(D,
env)` (§4.3) and compares price, `state.accumulated`, `pending` amounts, and the sequence of `(event_type, amount,
fixing, redeemed)`:
- **D1 at booking**: bitwise price, TARN (with and without principal) and TARF, PSEUDO and QUASI, `CALENDAR_DAYS` and
  `BUSINESS_DAYS` (TARGET calendar) environments.
- **D2 aged**: both forms aged in step by the §7.2 `advance` rule at ages {1 day before a fixing, on a fixing date,
  1 day after, 91 days, after redemption, after the last pay date}; quiet path (spot held below the TARN strike / at
  K for the TARF: no coupon, no hit) and redeeming path (spot above the strike: hit on the second fixing). Events and
  state amounts bitwise (same spots, same factors); price relative 1e-12; same day on which each payment is booked.
- **D3 rebuilt mid-life**: a dated booking stating `as_of` and the state at a mid-life date equals the dated contract
  aged from inception to that date (bitwise state, relative 1e-12 price); likewise for its numeric twin.
- **D4 theta**: numeric and dated theta agree (relative 1e-12) under a `CALENDAR_DAYS` environment, including a
  fixing inside the bump. (Under `BUSINESS_DAYS` they differ today, M9: pending Q7.)
- **D5 dated accrual**: `initial_date` + monthly dates give `ACT_365` factors; a `BUSINESS_DAYS` contract day count
  reads the environment's calendar; `initial_date` together with `accrual_factors` is refused, and neither is
  refused (R2).

**Backtest and dynamic-scenario ageing (Q4.2)**
- **B1** A numeric TARN in an `FXBacktestEngine` run over a flat market: its value on day 0 and day 91 **differ**, and
  day 91 equals (relative 1e-12) a fresh pricing of the contract rebuilt on day 91 from the run's banked state.
- **B2 redeeming path**: spot path above the strike from day 0, target 1.5 coupons, pay lag 2 days. The first coupon
  is booked as paid cash on its pay day, the clipped second coupon and the principal on the second pay day; the
  product is terminal from the second fixing day; from then its value is exactly Σ DF × pending (no simulation), and
  on each pay day it drops by that payment while paid cash rises by it; after the last payment the value is 0.0; no
  `TerminalBookingWarning` is raised (warnings-as-errors).
- **B3 quiet path**: no coupon; principal booked on the final pay day; TARF with gearing: negative cash booked.
- **B4 twins in one run**: the numeric and the dated TARN (and TARF) in the same backtest and the same dated dynamic
  scenario: identical paid cash per day (bitwise), values relative 1e-12, identical event days.
- **B5 registration**: fixings before the run start → one `MissingFixingWarning` per position, banked at the day-0
  spot, their payments not booked; last fixing before the run with unbanked fixings → one
  `MaturedBeforeRunWarning`, value 0 every day, no cash; a terminal booking in a run is kept at PV(pending); a path
  whose date goes backwards → `ValidationError`.
- **B6 no collateral change**: a book of the other M7 products (no TARN/TARF) produces bitwise the same rows as main
  in both engines; the existing FX backtest and dynamic-scenario tests pass unmodified.

**Warnings (Q6, W5)**
- **WARN1** Each terminal booking form (numeric empty `fixing_times`; numeric redeemed state; dated `as_of` ≥ last
  fixing; dated redeemed) warns `TerminalBookingWarning` once at construction; value = PV(pending), or exactly 0.0
  with nothing pending; pricing and Greeks do not warn again.
- **WARN2** An aged copy that redeems (P3, B2) does not warn.
- **WARN3** Pricing a contract with an unbanked passed fixing (numeric after a bare `time_shift`; dated with a fixing
  date before valuation and `as_of` before it) → exactly one `MissingFixingWarning` naming the fixings; the price
  equals the product after `record_passed_fixings` at the same quoted spot; the product is unchanged afterwards.

**Regression and refusals**
- **R1** (findings §10): rebuilding with neither period source raises; with `accrual_start = 1/12 − 0.125` the first
  factor is 1/12 and the coupon 1.00, not 0.50.
- **R2** each raises `ValidationError`: both period sources; missing `state`; `accumulated` < 0, NaN, or > target +
  tol; `accrual_start ≥ fixing_times[0]`; `accrual_start > 0` with `accumulated > 0`; numeric pending `pay_time` ≤ 0;
  TARN pending amount ≤ 0; redeemed numeric state with listed fixings; a dated TARN with neither or both of
  `initial_date` / `accrual_factors`; `day_count_convention` on the numeric form; mixed forms (each pair); dated pending
  `pay_date <= as_of`; dated `accumulated > 0` with `as_of` None or before the first fixing; dated `as_of` after the
  valuation date (at pricing).
- **T1 theta**: the M3 cases (0.5/365, 1/365) give finite theta; in the small-vol limit the landing-fixing theta
  equals the deterministic forward-path reference (`_det_pv`, `test_fx_tarn_note_mc.py:68`, extended with a start
  `acc`); the 2/365 case equals main's theta to the bit (TARN 0.016861…, TARF 0.003077… at full precision).
- **T2 matured**: a bump past the last fixing gives PV(pending); theta is carry only.
- **C1 inception unchanged**: the 52 + 2 tests pass unmodified apart from constructor arguments; demo stdout
  byte-identical to main.

## 11. Out of scope

- Ageing or warning for the other numeric FX products (§9 lists them for a later decision).
- Moving `AccrualPeriod` to a neutral package (findings §10: no resolver migration for the TARN).
- Local-vol, Heston or analytic TARN/TARF engines.
- Business-day rolling of stated dates; a cash-interest account in the FX backtest.

## 12. Open questions (rev 2)

- **Q7 Which clock does numeric FX theta step by?** `_fdm_theta` steps a numeric product by `days/365` whatever the
  environment's day count, while its dated twin moves the env date (M9: `BUSINESS_DAYS`, Friday: numeric
  −9.135e-05 vs dated 0). The ageing hook (§7.2) already uses the environment's convention, so in a run the two
  forms agree; only theta disagrees.
  - Recommended: step numeric products by `calculate_year_fraction(V, V + days, env convention)` in
    `_fdm_theta`. It is bitwise unchanged under `CALENDAR_DAYS`/`ACT_365` (1/365.0 either way) and restores
    parity elsewhere.
  - Cost if wrong: the change is in shared FX code, so every numeric FX product's theta moves under a non-calendar
    environment (Fridays/holidays go to 0, `ACT_360` steps 1/360). Not doing it leaves D4 limited to calendar
    environments and the standing parity rule unmet for theta.
- **Q8 What does an undated `DayPath` do with a TARN/TARF?** Today it moves no date (M8), so nothing ages — neither
  form. Every `FXPathLibrary` builder makes undated paths.
  - Recommended: refuse with a `ValidationError` naming `DayPath.start_date` when the book holds a TARN/TARF and the
    path is undated. Parity holds and no clock is invented.
  - Alternative: the equity lifecycle rule (`asset/equity/lifecycle/manager.py:211-215`): day k = the environment's
    date + k calendar days, and the FX scenario then sets every environment's date to it. Cost: every dated FX
    product in an undated scenario starts showing theta (today frozen, M8) — a change outside this workstream's
    two products.
  - Cost if the recommendation is wrong: callers of `FXPathLibrary` with a TARN/TARF must set `start_date` (the
    builders take no such argument today; the field is settable on the returned path).
