# Last-day theta: proposal (I11)

Status: proposal only, no production code. Follow-up queue item I11 (spec `.ship/2026-10-13-followup-queue/spec.md`).
It picks up the W10 Q9 ruling (user, 2026-10-10): "penultimate-day theta = settled(final, today's spot) − V(today);
last-day theta = FUTURE STUDY". Every file:line below was read in this worktree (branch `worktree-ship-followup-queue`,
head 37458b5d). Every number comes from a probe script run on that tree (section 2.4).

## 1. Problem

A risk report run on the last days of a contract gets one of three things, depending on the product:

- theta = 0.0 (every equity product through `GreeksCalculator`, and that already happens on the day *before* expiry);
- an exception (DCN cash-Greeks report on the final date; FX Garman-Kohlhagen on the expiry date);
- minus the payoff (FX TARF on its last fixing, when nothing is left to pay later).

Who hits it: the daily risk run and any desk theta report on the books' expiring trades, every day.

Concrete symptom: an ATM equity call (S = K = 100, σ = 20%, r = 1%) one calendar day before expiry is worth 0.419.
That is its entire remaining time value, all of which goes on the last day. The reported theta is 0.0. Two days
before expiry it reports −0.174.

The question is what theta should mean when the bump reaches or passes maturity: on the penultimate day, on the
expiry/settlement day, at the close or intraday, with cash pending, and on the FX env clock (Q7). The question covers
equity vanillas and autocallables, the DCN, FX vanillas and FX TARN/TARF.

## 2. Current behaviour

### 2.1 Equity: generic numerical theta (vanillas, barriers, autocallables, accumulators)

- `numerical_theta` (`quantark/asset/equity/riskmeasures/greeks/numerical.py:204-254`) advances the date with
  `advance_theta_bump` (`greeks/bump_envs.py:206-233`, the env's own `calculate_year_fraction`). It returns 0.0 when:
  - the step is zero and the product is already at maturity (`:239-241`);
  - the step is zero on a calendar-day clock (`:248`);
  - **`current_maturity <= time_bump` (`:249-250`)**. The theta-boundary design kept this on purpose: "A bump at or
    past maturity keeps today's handling (`current_maturity <= time_bump` returns 0 before any shift)"
    (`docs/superpowers/specs/2026-10-07-theta-observation-boundary-design.md:76-77`).
- `time_scenario` (`numerical.py:545-601`) has the same cascade and sets `zero=True` (`:569-581`).
  `numerical_theta_from_scenario` (`:635`) and the cross-time greeks (`:690`, `:713`, `:737`) return 0.0 on it.
- `shift_for_theta` (`greeks/theta_shift.py:226-262`) is never reached past that guard. It is where a contract that
  ENDS at `t0` is handled: a KO today, a barrier hit today, or a KO-reset whose schedule 1 ends. It builds
  `ThetaShift.ended_at_t0`, and `bumped_value` (`:282-296`) prices the engine's own `t0` value and carries it to `t1`
  with `_carried_to_t1` (`:51-62`): re-discounted, or 0 once paid by `t1`. This is the R4 rule (user ruling
  2026-10-07, design `:44`): cash the engine decides at `t0` and pays within the bump leaves the value, "so theta on
  an event day is about minus the payment". Cash already BOOKED in the trading system is excluded.

Measured, BS dated call, valuation date at midnight, default bump of 1 calendar day:

| K | T−2d price / theta | T−1d price / theta | T=0 price / theta |
|---|---|---|---|
| 95 | 5.00530 / −0.00270 | 5.00260 / **0.0** | 5.0 / 0.0 |
| 100 | 0.59334 / −0.17435 | 0.41900 / **0.0** | 0.0 / 0.0 |
| 105 | 1.99e−4 / −1.99e−4 | 3.4e−7 / **0.0** | 0.0 / 0.0 |

A numeric `maturity = 1/730` (half a day) ATM call: price 0.296, theta 0.0. A numeric `maturity = 0.0` contract cannot
be constructed (`ValidationError: Either maturity/tenor or exercise_date/maturity_date must be provided`).

### 2.2 DCN: date-rolled theta of the cash-Greeks report

- `_date_rolled_theta` (`quantark/asset/equity/riskmeasures/greek_conventions_report.py:70-97`) rebuilds the schedule
  at the next SSE trading day (`:85` or `:89`) and returns `engine.price(rolled) − base_pv`. The env is not moved:
  DCN times come from the schedule.
- Penultimate day (W10 Q9, pinned by `test/test_dcn_final_observation.py:303-313`): the rolled product sits on its
  final date, which every route now settles from `dcn_valuation.settle_final_observation`
  (`quantark/asset/equity/engine/dcn_valuation.py:109-114`). So theta = settled(final, today's spot) − V(today).
  Measured, DCN_A on PDE at spot 5500: 0.97491.
- Final date: the roll lands after maturity and `DCNScheduleSpec.build` raises
  (`quantark/asset/equity/product/option/dcn_schedule.py:161`). The whole `build_cash_greeks_report` call then raises
  (`:132`), so delta, vega and rho are lost along with theta. Measured: `ValidationError: valuation_date after
  maturity_date`. Price on that date: 9996.10.
- This report never passes `lifecycle_state`. W14 ledger cash (pending receivables) is outside both sides.

### 2.3 FX

- Closed-form GK greeks (`quantark/asset/fx/engine/analytical/garman_kohlhagen_engine.py:86-196`) go through
  `_d1_d2`. That function raises `PricingError("Vanishing volatility-time term")` at τ = 0 (`:235-239`), even though
  `price` settles at τ = 0 (`:55-60`). Measured on the expiry date: price 0.05, theta RAISES.
- FD theta `BaseFxEngine._fdm_theta` (`quantark/asset/fx/engine/base_fx_engine.py:226-251`):
  - dated form: moves the env one calendar day. A dated product's `_year_fraction_to` then refuses a valuation after
    expiry (`quantark/asset/fx/product/base_fx_product.py:135-138`). Measured: RAISES on the expiry date.
  - numeric form: ages by the env year fraction (Q7, `:236-242`). `time_shift` (`base_fx_product.py:108-118`) can
    take maturity below zero, and GK `price` refuses τ < 0 (`garman_kohlhagen_engine.py:51-52`).
  - Result: (V(t+1d) − V)/days, not capped at maturity.
- The local-vol / Heston numeric theta (`quantark/asset/fx/engine/localvol_greeks.py:19-50`) caps the step at
  `eff = min(dt_env, 0.5·maturity)` and rescales linearly to a per-day figure. An env step of zero gives 0.0
  (`:40-41`). This is the only route that defines a near-expiry theta, and it is a clamped rate, not a one-day roll.
- TARN / TARF dated (`FxTarnForwardMCEngine`, `quantark/asset/fx/engine/mc/fx_tarf_mc_engine.py:72-118`):
  `resolve` banks the t = 0 fixing at the quoted spot and the contract becomes terminal. The price is then
  `pending_present_value`. In `time_shift` / `record_passed_fixings` (`target_redemption_base.py:373-404`), cash paid
  strictly before `t1` leaves and cash due today stays (OQ-1 b). Measured, TARF strike 1.15, spot 1.20:

| | day before last fixing | last fixing day |
|---|---|---|
| no pay lag: price / theta | 0.050059 / −5.87e−5 | 0.050000 / **−0.050000** |
| 2-day pay lag: price / theta | 0.050045 / −5.87e−5 | 0.049986 / **+6.85e−6** |

  TARF is already R4-consistent: payoff paid within the bump gives theta = −cash, and a payoff paid later gives theta
  = carry.

### 2.4 Intraday

`quantark/intraday/greeks.py` defines theta on the frozen market inside one event/clock segment:
- `resolve_theta_step` (`:312-353`) clamps the step to the next contractual event, so the last session's theta is
  the decay up to the expiry instant.
- AT an event instant, theta is `status="undefined"` (`:325-327`, `:444-445`).
- The desk roll adds back cash paid on the way: `_rolled_value` returns `price + received` (`:346-353`).

The daily theta (R4) does the opposite and lets cash paid within the bump leave the value. The two conventions
already differ on an event.

Probe: `scratchpad/t15/probe_theta.py` (session scratchpad, not tracked). It covers the equity BS rows, DCN_A PDE,
GK, and TARF with and without a lag.

### 2.5 The inconsistency in one line

On the same "last day" the codebase reports:
- 0.0 for equity, on both T−1d and T;
- settled − V for DCN on the penultimate day, and a raise on its final day;
- a raise for FX GK at T;
- −cash or carry for FX TARF.

Only TARF and the DCN penultimate rule follow R4/W10 Q9.

## 3. Options

All three keep every theta with `current_maturity > time_bump` bitwise. They differ only once the bump reaches
maturity.

### Option A: make "zero" uniform

Theta = 0.0 whenever the bump reaches or passes maturity, on every route:
- DCN final date: 0.0 instead of a raise. The penultimate DCN rule (W10 Q9) would also go back to 0.0 for
  consistency, or it stays as the one exception.
- FX GK at τ = 0 and dated FD past expiry: 0.0.
- TARF last fixing: 0.0. That needs a special case, because TARF reaches the value without any maturity guard.

- Cost: small. A guard in `_fdm_theta`, one in GK `calculate_greeks`, one in `_date_rolled_theta`.
- Risk: the largest decay of the contract's life (2.1: the whole 0.419) is never reported on any day. The P&L of the
  last day shows up as unexplained wherever theta is used to explain P&L. This contradicts R4, which the user ruled
  for event days, and it reverses the signed W10 Q9.
- What moves: DCN final day raise → 0.0; GK T=0 raise → 0.0; TARF last-fixing theta −cash/carry → 0.0, and W10 Q9
  if harmonised. No golden or cert should hold a raise; the TARF tests that pin last-fixing theta need checking.

### Option B: roll to settlement, with cash per R4 (generalises W10 Q9)

When the bump reaches maturity, the bumped value is **the contract's own settled value at maturity, on today's
spot, carried to `t1`**:

1. **Penultimate day** (0 < T ≤ bump): age the product by T exactly (`time_shift(T, …, keep_landing=True)`) and price
   it at maturity 0 on the frozen spot. Every route has done this since W8/W9/W10: equity expiry settlement,
   `settle_final_observation` for DCN, and TARN terminal resolve. Then carry from the maturity instant to `t1`
   (`_carried_to_t1`): 0 if paid by `t1`, re-discounted otherwise. Theta = carried settled − V(t0). For DCN this IS
   W10 Q9.
2. **Expiry / settlement day** (T = 0): the contract ends at `t0`, so this is exactly the existing `ended_at_t0` path
   of `shift_for_theta` (`theta_shift.py:248-250`, `bumped_value` `:287-290`) with the terminal payment time. Theta =
   carried(settled) − settled. That is −cash when the payment falls within the bump, and DF carry when it is paid
   later. TARF already behaves like this (2.3).
3. **Booked cash** (the W14 DCN ledger, TARN pending): R4 excludes cash flows already BOOKED. So pending receivables
   are held out of both sides, and only engine-decided cash at `t0` follows rule 2. This needs confirmation (Q3).
4. **FX**: the same rule in `_fdm_theta` and `fx_numeric_theta`. On the env clock (Q7), "reaches maturity" is
   decided on the env's year fraction. A zero env step (a BUSINESS_DAYS weekend) still gives 0.0 (Q7 rule). The GK
   closed form falls back to the FD route at τ = 0 instead of raising.
5. **Intraday**: unchanged. It already clamps the step to the expiry instant (2.4).

- Cost, medium:
  - `numerical.py:249-250` and `:580-581` build a settlement `ThetaShift` instead of returning 0.0. The cross greeks
    on the `zero` flag become the existing `frozen` path, where spot greeks of an ended contract are 0.
  - `_date_rolled_theta` handles the final date.
  - `_fdm_theta`, GK `calculate_greeks` and `fx_numeric_theta` (whose `0.5·maturity` cap and rescale stay below
    maturity) each get the settlement branch.
  - Products with no settled T=0 value (the routes W8 left refusing with `TODO(dated-expiry)`) still raise.
- Risk: on the expiry day a long ITM option reports theta ≈ −payoff (e.g. −5.0 for the K=95 call). That is R4 by
  ruling, but a desk may read it as a loss (Q2). The penultimate-day number mixes the last day's decay with the final
  decision on today's spot, which is the W10 Q9 trade-off.
- What moves:
  - Equity theta on every date with T ≤ bump: 0.0 → value (T−1d ATM call: 0.0 → −0.419 plus carry).
  - Equity T=0 theta: 0.0 → −cash or carry.
  - DCN final day: raise → value.
  - GK T=0: raise → value.
  - TARF: unchanged.
  - The DCN penultimate rule: unchanged.
  - Replay goldens with a theta column (`test/replay_golden/data/scalar_bsm_greeks.csv`, `localvol_greeks.csv`) and
    any modelvalidation anchor that holds a theta within one bump of maturity must be measured before
    implementation. A move there is a STOP (spec guardrail).

### Option C: clamped rate (the FX local-vol and intraday precedent)

Clamp the step at maturity, `eff = min(bump, T)`, and report (V(T) − V(t0))·(per-day scaling)/eff, as
`fx_numeric_theta` does with its `0.5·T` cap. At T = 0, theta is undefined: 0.0, or a status once a result type
exists (I12).

- Cost: medium. Same sites as B, plus a per-day rescale in the equity path.
- Risk: it is a derivative, not a one-day P&L. ATM it grows like 1/√T as T → 0, so a half-day-to-expiry figure
  scaled to "per day" overstates the remaining decay. It disagrees with W10 Q9 (signed) and with R4 on event days.
  The cash at T is never in theta.
- What moves: the same cells as B, with different numbers. TARF moves away from its current R4 values.

## 4. Recommendation

**Option B.** Reasons:

1. It is the only option consistent with the two rulings already signed: R4 (event-day cash) and W10 Q9 (penultimate
   DCN). TARF already implements it, so the rule becomes uniform with the least new semantics.
2. It reuses existing machinery: the T=0 settlement on every route (W8/W9/W10) and the `ended_at_t0` /
   `_carried_to_t1` carry. No new pricing path is invented ("no stupid fallbacks"). A route that cannot settle at
   T=0 keeps raising.
3. It reports the largest decay of the contract's life instead of hiding it (2.1).

Implementation shape, for when it is approved: one helper `settlement_shift(product, env, time_bump, bumped_date)`
in `theta_shift.py` returning a `ThetaShift` with `ended_at_t0` set at the maturity instant. Equity, DCN and FX theta
all call it. The moved-values list per route is measured on the fixtures in 2.1–2.3, and goldens and certs are
checked first.

## 5. Open questions for the user

1. **Definition.** Adopt Option B: when the theta bump reaches maturity, the bumped value is the contract settled at
   maturity on today's spot, carried to `t1`? (A = always 0.0; C = clamped per-day rate.)
2. **Expiry-day cash.** On the expiry/settlement day, with the payoff paid within the bump, theta ≈ −payoff (R4, as
   TARF does today). Confirm, or report expiry-day theta as carry only, i.e. add the payoff back the way intraday's
   desk roll does (`intraday/greeks.py:351-353`)?
3. **Booked pending cash.** Hold ledger receivables (W14 DCN ledger, TARN pending booked by a run) out of theta on
   both sides, per R4's "booked cash is excluded"? Yes/no.
4. **Penultimate-day equity.** Accept that equity theta on T−1d moves from 0.0 to settled-on-today's-spot − V(t0)
   (ATM example: 0.0 → about −0.419), the same rule as DCN W10 Q9? Yes/no.
5. **Close vs intraday.** Daily theta is a close-to-close roll (valuation date at midnight, the W10 Q1 close). Leave
   the intraday theta as it is (clamped to the expiry instant, undefined AT it, paid cash added back)? Or align its
   cash treatment with the answer to Q2?
6. **FX env clock.** "Reaches maturity" is decided on the env's year fraction (Q7). On a BUSINESS_DAYS env a Friday
   theta stays 0.0 even when expiry is Monday, and Monday's theta then carries the settlement. Accept?
7. **DCN report on its final date.** Should `build_cash_greeks_report` on the final date return the settled-row
   greeks (Q6 delta slope, gamma 0, vega 0) with the new theta, instead of raising as a whole? Yes/no.
