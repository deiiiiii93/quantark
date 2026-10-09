# FX TARN / TARF mid-life booking and ageing — design

Status: draft 2026-10-09 for workstream W4b, written on branch `w4b-tarn-spec` from main `dcea77b6`. It applies the
user's decision **A1** (below) to `FxTargetRedemptionNote` and measures whether `FxTargetRedemptionForward` needs the
same treatment (it does, except for the accrual term: §2.6). No library code is changed by this spec. Evidence:
`2026-10-09-fx-tarn-lifecycle-evidence/measure_tarn_tarf_ageing.py` and its saved output `.out` (M1–M6 below).
Origin: findings `2026-10-08-accrual-scope-followups-findings.md` §10.

## 1. User decision A1 (binding, 2026-10-09)

- **A1.1** A mid-life TARN booking states its quoted periods (fixing times plus a stated period start, or explicit
  accrual factors) and its realized state (what has already accrued towards the target, and so whether it has
  knocked out). A contract rebuilt mid-life prices exactly like the same contract aged from inception. This
  numeric-vs-aged parity is a standing user rule; §7 makes it an acceptance test.
- **A1.2** Until a contract states these terms, rebuilding it mid-life is REFUSED. No start is invented.
- **A1.3** `time_shift` drops passed fixings after banking their effect in the realized state. For theta they fix at
  the held spot, the same as the range accrual fix (findings §9: `BaseFxProduct.record_passed_fixings`).
- **A1.4** The spec decides, by measurement, whether the TARF needs the same treatment.

## 2. Problem (measured on dcea77b6)

### 2.1 Today's code

- `FxTargetRedemptionNote.__init__` sets the default accrual factors to `t_i - t_{i-1}`, with `t_0 = 0`. The first
  period is assumed to start at the valuation instant.
- Neither product has any realized state. The engines start every path at `acc = 0`, `alive = True`.
- `time_shift` subtracts the bump from the fixing and pay times and from `maturity`/`delivery`. It keeps past
  fixings and does not re-validate.
- `record_passed_fixings` is inherited from the base class, where it does nothing.
- Both engines (`FxTargetRedemptionNoteMCEngine`, `FxTarnForwardMCEngine`) are constant-vol Garman-Kohlhagen (GK)
  Monte Carlo. They simulate every listed fixing (`FxGKPathGenerator` requires times > 0) and discount every listed
  pay time.
- Theta is `BaseFxEngine._fdm_theta`: `deepcopy`, then `time_shift(days/365)`, then
  `record_passed_fixings(env)`, then reprice. It is the only FX theta path. The grep of `quantark/asset/fx`,
  `portfolio/fx`, `execution`, `var/fx`, `stresstest/fx`, `backtest/fx`, `dynamicscenario/fx` and `pnlexplain`
  finds no other caller of `time_shift`.

### 2.2 Measurements

Fixture: twelve monthly fixings at `k/12`. TARN: notional 100, coupon 12%, strike 1.10. TARF: strike 1.15,
notional 1. EURUSD spot 1.20, `r_d` 5%, `r_f` 3%, vol 10%, 200,000 PSEUDO paths, seed 7.

| # | What | Result |
|---|---|---|
| M1 | TARN rebuilt from its 11 remaining times after 0.125y | first accrual factor **0.041667** instead of 0.083333, so the first coupon is **0.50 instead of 1.00**. `time_shift(0.125)` keeps the factors (correct) and all 12 fixings, the first at t = −0.041667. |
| M2 | Price after `time_shift(0.125)` + `record_passed_fixings` | TARN and TARF both **raise** `ValidationError: Time to maturity must be non-negative, got -0.0417`. The raise comes from the DF of the past pay time (`fx_pricing_environment.py:187`). |
| M3 | Theta, first fixing 0.5/365 away | both **raise** (the same DF error, on t = −0.00137) |
| M3 | Theta, first fixing exactly 1/365 away | both **raise** `times must be strictly increasing and positive` (the fixing lands on t = 0) |
| M3 | Theta, first fixing 2/365 away | finite: TARN 0.016861, TARF 0.003077 |
| M4 | Two months aged, two coupons paid; the remaining 10 fixings priced with the full target 3.0 against the remaining target 1.0 | TARN with principal: 101.652635 vs 100.578131 (**+1.07**). TARN without principal: 2.957940 vs 0.995609 (**+1.96**). TARF (target 0.20, 0.10 realized): 0.091663 vs 0.027251 (**+0.064**). |
| M5 | `env.valuation_date` moved 91 days, product unchanged (the FX backtest and dynamic-scenario clock) | `get_maturity` stays 1.000000. The price is unchanged to the bit (TARN 101.643148, TARF −0.011326). |
| M6 | Accrual factors derived from shifted times vs the inception factors (11 periods) | τ = 0.125: bitwise equal. τ = 0.1, 45/365, 91/365: 2, 4 and 2 periods differ by ≤ **1.11e-16**. |

Reading of the results:

- **Rebuilding mid-life (M1, M4).** A rebuilt TARN gets its first period's accrual wrong. Neither product can say
  how much has already accrued, so a mid-life booking is wrong by about one coupon, or by the whole remaining
  target.
- **Theta (M2, M3).** Theta cannot step over a fixing. Within one day of a fixing, theta raises.
- **No lifecycle ages these products (M5).** No lifecycle, backtest or replay path ages a TARN or TARF today. The
  equity lifecycle tracker, replay and the simulated-path backtest do not know them. `grep` finds them only in
  their two test files, the demo and `execution/inventory.py` plus `test/execution/matrix_fixtures.py`. The FX
  backtest and the FX dynamic scenario move `env.valuation_date` only, and a numeric product ignores it. Every day
  is priced as if it were the inception date, at that day's market. This holds for every numeric FX product, not
  just these two (Q4).

### 2.3 TARF: needs the realized state, not the accrual term (A1.4)

The TARF's favourable amount per fixing is `notional × max(±(S_i − K), 0)`. It has no period length, so the M1
defect cannot occur. The other defects do occur on the TARF, measured:

- the past fixing that `time_shift` keeps (M2);
- theta raising near a fixing (M3);
- no realized accumulation towards the target (M4, +0.064 on a value of 0.027).

**Decision:** the TARF gets the same realized state, ageing and refusals as the TARN (§3.2–§5). It gets no accrual
term.

## 3. Data model

### 3.1 Shared state: `TargetRedemptionState`

New module `quantark/asset/fx/product/option/target_redemption_state.py`. Both products import it.

```python
@dataclass(frozen=True)
class PendingPayment:
    pay_time: float          # years from valuation, > 0 (not is_zero)
    amount: float            # undiscounted domestic cash already determined by a fixing

@dataclass(frozen=True)
class TargetRedemptionState:
    accumulated: float                             # undiscounted domestic amount banked towards the target
    pending: Tuple[PendingPayment, ...] = ()       # fixed but unpaid cash, ascending pay_time
```

- `accumulated` is the TARN's realized coupons, or the TARF's realized favourable amounts. It includes coupons
  that are fixed but not yet paid.
- `pending` holds cash a past fixing has determined but which has not been paid yet:
  - a coupon or TARF net cashflow with a pay lag;
  - the principal of a TARN that redeemed, or matured, with its pay date still ahead.

  Cash whose pay time is at or before valuation is booked and is not part of the contract's value. This is the
  ruling behind theta-boundary spec R4.
- Validation (`ValidationError`):
  - `accumulated` must be finite and ≥ 0.
  - Each `pay_time` must be finite and > 0, with `is_zero(pay_time)` refused.
  - Each `amount` must be finite.
  - `pending` must be sorted by `pay_time`.
- `TargetRedemptionState` is immutable, as `param/` objects are. `time_shift` replaces the state; it never mutates
  it.

**There is no knocked-out flag.** A1.1 names "any knocked-out flag", but a target redemption has exactly one
termination event: the accumulated amount reaching the target. `is_redeemed` is therefore derived on the product as
`accumulated >= target - hit_tol(target)`. The tolerance is the engines' own `_HIT_RTOL × |target|`, which moves to
the product so the product and the engines share it; it is 0 for an unbounded target. A stored flag could only
contradict the amount (Q5).

### 3.2 `FxTargetRedemptionNote`

New constructor arguments:

| Field | Meaning | Rule |
|---|---|---|
| `accrual_start: Optional[float]` | Numeric time at which the first LISTED period began. It is < 0 when that period started before valuation. | Exactly one of `accrual_start` and `accrual_factors` must be given. **Neither → refused** (A1.2: no invented `t_0 = 0`). Both → refused (one source of truth). `accrual_start < fixing_times[0]`. |
| `accrual_factors` | Unchanged: explicit per-period factors | as today |
| `state: TargetRedemptionState` | The realized state | **Required, no default** (Q1). At inception: `TargetRedemptionState(accumulated=0.0)`. |

- The derived factors become `t_i - t_{i-1}` with `t_{-1} = accrual_start`. They are computed once at
  construction and are terms of the contract from then on. `time_shift` never recomputes them; it drops the factor
  together with its fixing (§5).
- Validation added to today's rules:
  - **`fixing_times` lists only the remaining fixings, each > 0 and not `is_zero`.** A fixing at or before
    valuation belongs in the state, not in the schedule. Today the check is `t <= 0.0`; the `is_zero` guard is new.
  - **The terms must be consistent:**
    - `accumulated <= target + hit_tol` (the engines clip, so a realized overshoot is impossible);
    - `accrual_start > 0` with `accumulated > 0` is refused, because periods are contiguous: once a coupon has
      fixed, the current period began at or before valuation;
    - a TARN pending amount must be > 0.
  - **`fixing_times` may be empty** (recommended; Q6). The contract is then terminal: either redeemed, or
    matured with every fixing banked, so only `pending` is left. An empty list is what "every fixing banked"
    means; the state cannot say so on its own. `maturity` is then `max(pending pay times)`, or `0.0` if nothing is pending. Otherwise `maturity`
    stays the last fixing, as today.
- A non-terminal contract with a redeemed state is refused: `is_redeemed` with remaining fixings is a booking
  error. Ageing drops the dead fixings (§5), so the aged path never produces one.

### 3.3 `FxTargetRedemptionForward`

- It gains `state: TargetRedemptionState` (required), with the same validation, the same `is_redeemed` and the
  same terminal-empty rule.
- It gets no accrual term.
- TARF pending amounts may have either sign. A geared unfavourable fixing is a liability.

### 3.4 Refusals: what A1.2 can and cannot detect

**Refused:**
- an omitted period start (M1's exact failure becomes a `ValidationError`);
- an omitted state;
- a fixing at or before valuation;
- an inconsistent state (§3.2);
- pricing an aged product whose passed fixings are not yet banked (§5).

**Not detectable from numbers alone:** a booking that states a wrong `accrual_start` or a wrong `accumulated`. A1
makes these terms of the booking. Because both are mandatory, a caller must write them down; nothing defaults them
silently.

## 4. Engine impact

There are only two engines: `FxTargetRedemptionNoteMCEngine` and `FxTarnForwardMCEngine`. No analytical, PDE or
QUAD engine exists for either product (`execution/inventory.py:438-439`). Each engine changes as follows:

1. `pv_pending = Σ amount × DF_dom(pay_time)` over `state.pending`.
2. **Terminal contract** (`is_redeemed`, or no remaining fixings): return `pv_pending` and do not simulate.
   - The vol lookup is skipped. Today `get_vol(T)` with T ≤ 0 would be called.
   - `FxTarnMCResult`: `num_paths = 0`, `std_error = 0.0`, `sigma = 0.0`, `mean_redemption_fraction = 1.0` if
     redeemed else `0.0`.
3. Otherwise:
   - the pathwise walk starts at `acc = np.full(n_paths, state.accumulated)` instead of `np.zeros`;
   - `pv_pending` is added to every path, only when `pending` is non-empty;
   - the hit test uses `product.hit_tol()`.

   At inception (`accumulated = 0.0`, no pending) the arithmetic is identical (`np.full(n, 0.0) == np.zeros(n)`,
   with no extra add), so every inception price is unchanged to the bit (§6).
4. A guard before the simulation raises `ValidationError` on any remaining fixing ≤ 0 or `is_zero`. The message
   names `record_passed_fixings`. Today the error is the opaque generator/DF error of M2/M3.

The `PricingError` on a negative TARN price stays, since TARN pending cash is ≥ 0. `BaseFxEngine` is unchanged: its
spot, vol and rate bumps leave `state` alone, and `_fdm_theta` already calls the hook.

## 5. Lifecycle and theta

The ageing step follows the FX range accrual's two-call protocol:

**`time_shift(dt)`:**
- shifts the fixing times, pay times, pending pay times and `maturity`/`delivery`;
- drops the pending payments whose pay time is now ≤ 0 or `is_zero`, because they were paid within the bump;
- keeps the passed fixings, still unbanked. Until the next call the product is "unsettled", and pricing it is
  refused (§4 step 4).

**`record_passed_fixings(fx_env)`**, for each remaining fixing with `t <= 0` or `is_zero(t)`, in order. It stops
once the contract is terminal.
- **TARN:**
  - the coupon is `full_coupon(i)` if `coupon_condition(fx_env.spot)` holds (Q3), clipped to
    `target - accumulated`;
  - `accumulated += c`;
  - the cash `c` at `pay_i` goes to pending if `pay_i > 0` and is not `is_zero`, and is dropped as paid otherwise;
  - if the coupon hits the target: the principal (with `include_principal`) at `pay_i` is handled the same way,
    and every later fixing is dropped as dead;
  - if the last fixing banks without a hit, the final principal at its pay time is handled the same way.
- **TARF:**
  - `fav = notional × max(sign × (S − K), 0)`, clipped to the remaining target;
  - `cf = fav_eff − gearing × unf`, which may be negative;
  - `accumulated += fav_eff`;
  - pending/paid and the hit rule as for the TARN. On a final hit the geared unfavourable leg is still paid, as in
    the engine.
- The banked fixings, their pay times and their accrual factors are removed. Then `validate()` runs again.

**Theta, then:**
- **A fixing strictly inside the bump, or landing exactly on t1,** fixes at the held spot.
  - If its cash is paid by t1, it leaves the value: theta on a fixing day is roughly minus the payment, as in
    theta-boundary R4.
  - If it is paid after t1, it stays as a receivable.
- **A step past the last fixing** prices the terminal contract (§4 step 2): theta is pure carry on `pending`.
- **Same value as the equity rule.** The landing fixing is banked at the held spot rather than kept at time 0 and
  decided by the engine, as the equity theta-boundary R1 does. The value is the same, because the spot is held.
  The mechanism follows the FX precedent: the range accrual requires `t <= 0` fixings to be recorded.

Banking at the held spot is the theta convention, not market data. A booking or a lifecycle consumer that knows the
real fixing states it in `TargetRedemptionState`. No new lifecycle consumer is added here: M5's
`valuation_date`-only ageing is Q4.

## 6. Compatibility (measured by grep and by running the gates on dcea77b6)

**Construction sites.** Each needs `state=TargetRedemptionState(0.0)`; the TARN sites also need
`accrual_start=0.0`. The numbers are unchanged.

| Site | Product | Today |
|---|---|---|
| `test/test_fx_tarn_note_mc.py::_note` (every TARN test goes through it) | TARN | 27 tests |
| `test/test_fx_tarf_mc.py::_tarf` | TARF | 25 tests (52 pass together, 1.3 s, `-n 2`) |
| `test/execution/matrix_fixtures.py` `fx_tarn`, `fx_tarf` | both | `test_matrix_parity.py -k "Tarn or TargetRedemption"`: 2 pass |
| `example/fx_tarn_demo.py` (lines 74, 85, 97, 109) | 2 TARF + 2 TARN | demo stdout |

**Values and behaviour:**
- The test that pins the default factors (`test_default_accruals_are_period_lengths`, `[0.5, 0.5, 0.5]`) passes
  `accrual_start=0.0` and asserts the same values.
- `test_rejects_*` keep their messages. One new path: `fixing_times` containing `1e-12` is now refused (`is_zero`).
- No golden in `test/execution/goldens/` contains either engine (grep count 0).
- **Expected to move:**
  - theta within one day of a fixing changes from an exception to a number (M3);
  - a construction without `accrual_start`/`accrual_factors` or without `state` changes from accepted to refused.

  That is the whole API break. Release notes must say so: both products are public via
  `quantark.asset.fx.product.option`.
- **Expected not to move:** every inception price and Greek, and theta whenever no fixing is within the bump. All
  are bitwise. The gate is the 52 + 2 tests above plus a byte diff of the demo's stdout against main.

## 7. Acceptance tests

New file `test/test_fx_target_redemption_lifecycle.py`. The parity tests use the same seed and the same remaining
fixing times on both sides.

- **P1 TARN parity (A1.1, standing rule).** The inception contract C0 (§2.2 fixture, `accrual_start=0.0`) is aged
  by `time_shift(τ)` + `record_passed_fixings(env at S_h)`. It is compared with R, rebuilt from the remaining times
  `[t − τ]`, the aged state, and either:
  - (a) C0's surviving `accrual_factors`: the price must be **bitwise equal** for τ ∈ {0.125, 91/365}, S_h above
    and below the strike, and PSEUDO and QUASI;
  - (b) `accrual_start = 1/12 − τ`: bitwise at τ = 0.125, and within `1e-12` relative at τ = 91/365. M6 bounds the
    factor difference at 1.11e-16 per period.
- **P2 parity with pay lag.** Every pay time is its fixing + 0.02. The pending coupon must equal R's stated
  pending, and the prices must match as in P1.
- **P3 parity across redemption.** P2's pay lag, a target of 1.5 coupons, and two banked fixings that hit it (the
  second coupon is clipped to 0.5). The aged product is terminal, with value = PV(pending principal + clipped
  coupon). A rebuilt redeemed booking with empty
  `fixing_times` must give an equal price.
- **P4 TARF parity.** P1–P3 on the TARF, including a negative pending cashflow (S_h below K, gearing 2).
- **P5 multi-step ageing.** 91 daily `time_shift(1/365)` + record steps against one rebuild must match within
  `1e-12` relative. Sequential subtraction drifts by ULPs, so this tolerance is stated and is not bitwise.
- **R1 §10 regression.**
  - Rebuilding with neither accrual term raises `ValidationError`.
  - With `accrual_start = 1/12 − 0.125`, the first factor is `1/12` and the coupon is 1.00, not 0.50.
- **R2 refusals.** Each of these raises:
  - both accrual terms given; a missing `state`;
  - `accumulated` < 0, NaN, or > target + tol;
  - `accrual_start ≥ fixing_times[0]`; `accrual_start > 0` with `accumulated > 0`;
  - a fixing ≤ 0 or `is_zero`; a pending `pay_time` ≤ 0;
  - a TARN pending amount ≤ 0;
  - a redeemed state with remaining fixings;
  - pricing between `time_shift` and `record_passed_fixings`.
- **T1 theta.**
  - The M3 cases at 0.5/365 and 1/365 must give finite theta.
  - In the small-vol limit the existing tests use (the engines refuse σ ≤ 0), the landing-fixing theta must equal
    the closed-form deterministic forward-path reference: today's
    `_det_pv` (`test_fx_tarn_note_mc.py:68`), extended with a start `acc`.
  - The 2/365 case must equal main's theta to the bit (TARN 0.016861…, TARF 0.003077… at full precision).
- **T2 matured.** A bump past the last fixing gives value PV(pending), and theta is carry only.
- **C1 inception unchanged.** The existing 52 tests and the 2 matrix-parity cases pass unmodified apart from the
  constructor arguments, and the demo stdout is byte-identical to main.

## 8. Out of scope

- A dated TARN/TARF (fixing dates, `initial_date`).
- Ageing numeric FX products in the FX backtest and dynamic scenario (Q4).
- Moving `AccrualPeriod` to a neutral package: the TARN's per-period factor is already the explicit-fraction
  leg, and findings §10 recommends no resolver migration for it.
- Local-vol, Heston or analytic TARN engines.

## 9. Open questions for the user

- **Q1 Is the state mandatory?**
  - Recommended: `state` is required on both products (§3.2–§3.3). Inception code writes
    `TargetRedemptionState(0.0)` at 8 sites.
  - Alternative: default to the inception state, and refuse only when `accrual_start < 0`. This leaves the TARF,
    and a TARN rebuilt on a fixing day (`accrual_start = 0`), silently unbooked, which is the M4 error.
- **Q2 Is the period start mandatory?**
  - Recommended: `accrual_start` (or explicit `accrual_factors`) is always required. The measured M1 failure is
    exactly the `t_0 = 0` default, and requiring it costs 4 sites.
  - Alternative: keep `0.0` as a documented "inception booking" default. Under this alternative, A1.2's refusal
    cannot fire for the M1 case.
- **Q3 Which spot fixes a stepped-over fixing?** `fx_env.spot` (the quoted rate, as the range accrual's hook), or
  `effective_spot()` (the engines' simulation start). They differ only when `spot_days > 0`.
- **Q4 Which workstream ages FX products over calendar time?**
  - A dated TARN/TARF.
  - The FX backtest and dynamic scenario: today they freeze every numeric FX product (M5). Options: (a) refuse a
    numeric FX product there; (b) age it with `time_shift` + `record_passed_fixings` at each day's spot (the
    real fixing); (c) leave it to a later workstream.
- **Q5 Is the knocked-out flag derived?** Confirm that "knocked out" is derived from `accumulated` against the
  target (§3.1) rather than stored as a separate flag.
- **Q6 Should a terminal booking be allowed?** Should the library accept a booking that has already redeemed
  (empty `fixing_times`, value = pending only)? The aged path needs the terminal state either way (P3, T2). The
  question is only whether a desk may book one directly.
