# Autocallable-Driven Cash Legs — Design Spec

**Date:** 2026-06-30
**Status:** Approved (design) — pending implementation plan
**Module:** `quantark/cashleg/`

## 1. Problem

In the `otc-price-adapter`, the legacy `OtcOptionPricer` models autocallable
*adjustment* cash legs (prepayment/margin, backend premium, backend interest,
rebate, minimum return) differently from fixed-maturity products. For
Snowball/Phoenix structures these legs are **not** deterministic fixed-maturity
cashflows — they are contingent on the **same KO observation schedule and
barriers** as the parent autocallable.

The current adapter workaround builds a **separate synthetic `SnowballOption`**
per cash leg and prices/Greeks it with QuantArk engines. This works but belongs
in QuantArk's `cashleg/` module, not in adapter code.

This spec adds a native `AutocallableCashLeg` so the adapter can attach legs to
the **one real parent autocallable trade** and delete its synthetic
`SnowballOption` workaround.

## 2. Key findings (existing architecture)

These facts make the feature small and surgical — no risk-engine changes are
required.

1. **Native event timing is engine-specific — NOT uniform (verified).** Only a
   subset of engines emit *native* per-observation event stats; the rest return
   a trivial maturity-only distribution or silently fall back to MC. Verified
   against the code:

   | Product | MC | PDE | QUAD |
   |---------|----|-----|------|
   | Snowball | native | native | native |
   | Phoenix | native | **returns `None` → trivial** | **delegates to MC (`MCParams()` defaults)** |

   - `PhoenixOption` is `class PhoenixOption(BaseEquityOption)` — **not** a
     `SnowballOption`.
   - `PhoenixPDESolver(SnowballPDESolver)` does not override
     `calculate_event_stats`; the inherited body guards
     `if not isinstance(product, SnowballOption): return None`
     (`snowball_pde_solver.py:335`). ⇒ a Phoenix parent priced with PDE yields
     `EventDistribution.trivial(...)` — **no KO/coupon probabilities**.
   - `PhoenixQuadEngine.calculate_event_stats` (`phoenix_quad_engine.py:486`)
     constructs `PhoenixMCEngine(params=MCParams())` and delegates. ⇒ Phoenix
     QUAD event legs are driven by a **fresh MC run with default params**,
     inconsistent with the QUAD parent PV and Greeks.

   **Design consequence (see §3a, §6a):** `AutocallableCashLeg` supports
   **Snowball on MC/PDE/QUAD** and **Phoenix on MC** natively; it must **fail
   loud** when the parent's `EventDistribution` lacks the KO (or COUPON)
   probabilities a nontrivial leg requires, rather than silently pricing against
   a trivial distribution. Native Phoenix PDE/QUAD event stats are a separate
   engine workstream (§9).

2. **`price_with_events` already bridges engine → leg.**
   `BaseEngine.price_with_events()`
   (`quantark/asset/equity/engine/base_engine.py:55`) wraps
   `calculate_event_stats()` into an `EventDistribution` carrying
   per-observation `ko_probability[]`, `survival_probability[]`, and terminal
   buckets (`MATURITY_NO_KO`, `MATURITY_WITH_KI`, optional `KI`, `COUPON`).

3. **Future KO-observation filtering is automatic.**
   `_filter_observations_by_tau` (`snowball_pde_solver.py:700`) restricts the
   event grid to `[0, tau]` measured from `pricing_env.valuation_date`. A leg
   that consumes the parent's `EventDistribution` inherits the *exact same*
   filtered future schedule — satisfying requirement #2 with no extra code.

4. **The risk path already gives risk-bearing legs their Greeks for free.**
   `EquityPosition.get_trade_value()`
   (`quantark/portfolio/equity/position.py:94`) sums product NPV + leg PVs via
   `price_with_events`. `EquityPosition.get_trade_greeks()` (line 150) bumps
   spot and re-invokes `get_trade_value()`, which **re-solves the parent and
   rebuilds the `EventDistribution` under the bumped spot**, then re-feeds it to
   every leg. Any leg whose PV derives from `EventDistribution` therefore
   acquires genuine delta/gamma — this is exactly how `AccrualLeg`'s
   KO-truncation gets its delta today.

**Consequence:** the leg needs no engine of its own. Method (PDE/QUAD/MC), grid/
path parameterization, and as-of filtering are all inherited from the parent.
The "No MC inside PDE" invariant is preserved — the leg selects no engine.

## 3. Architecture & data flow

`AutocallableCashLeg(CashLeg)` is a pure consumer of the parent autocallable's
`EventDistribution`, expressing a **per-KO-date accrual schedule** that no
current leg can.

```
EquityPosition(product=SnowballOption|PhoenixOption, engine=PDE|QUAD|MC,
               cash_legs=[AutocallableCashLeg(margin), ...(rebate), ...(interest)])
   │
   ├── get_trade_value(env)
   │     └── engine.price_with_events(product, env) → EventDistribution
   │           └── leg.value(event_dist, env, notional) → per-leg PV
   │
   └── get_trade_greeks(env, GreeksCalculator)
         └── bump spot → re-solve parent → new EventDistribution → re-value legs
             ⇒ leg delta/gamma FREE
```

A thin **`value_standalone(parent_product, engine, env)`** helper calls
`price_with_events` itself, so a leg can be priced directly without constructing
a position ("priced directly by QuantArk"). Greeks remain the position route
(the supported `GreeksCalculator` / `get_trade_greeks` path).

The adapter keeps **one real `SnowballOption`/`PhoenixOption`** (the actual
trade) and attaches legs to it, deleting per-leg synthetic `SnowballOption`s.

## 3a. Engine support & fail-loud contract

Supported parent/engine combinations for a nontrivial `AutocallableCashLeg`
(`KO_MATURITY` or `COUPON` basis):

| Parent | MC | PDE | QUAD |
|--------|----|-----|------|
| Snowball | ✅ | ✅ | ✅ |
| Phoenix  | ✅ | ❌ (engine emits trivial) | ⚠️ MC-derived stats (documented; not method-consistent) |

- **Fail-loud guard (mandatory).** `value()` requires the probability stream its
  `accrual_basis` needs:
  - `KO_MATURITY` → `EventType.KO` present in `event_dist.probabilities` as an
    array whose length equals `len(accrual_factors)`.
  - `COUPON` → `EventType.COUPON` present, same length rule.
  If the required stream is absent (e.g. a trivial distribution from Phoenix
  PDE, or any non-autocallable engine), raise `ValidationError` naming the
  engine/product — never price against a trivial distribution. This implements
  the "no stupid fallbacks / exact semantics" rule.
- **Phoenix QUAD caveat.** Because `PhoenixQuadEngine` delegates event stats to
  MC with default `MCParams()`, a Phoenix-QUAD leg's PV/Greeks are MC-derived
  (carry MC noise and are not consistent with the QUAD parent). The leg will
  *function*, but the spec treats this as ⚠️ documented, not first-class
  support; production use should prefer Phoenix MC until native QUAD event stats
  exist (§9).
- Native Phoenix PDE/QUAD event stats are **out of scope** here (§9); when added,
  the support matrix upgrades with no leg-side changes.

## 4. The `AutocallableCashLeg` dataclass

New file: `quantark/cashleg/autocallable_leg.py`.

```python
class AutocallableLegType(Enum):
    MARGIN          = "margin"            # prepayment / margin
    BACKEND_PREMIUM = "backend_premium"
    BACKEND_INTEREST= "backend_interest"
    REBATE          = "rebate"
    MINIMUM_RETURN  = "minimum_return"

class PvFormula(Enum):
    NORMAL                = "normal"                  # PV = sign · R
    NOTIONAL_MINUS_PAYOFF = "notional_minus_payoff"   # PV = sign · (notional − R)   margin/prepayment

class AccrualBasis(Enum):
    KO_MATURITY = "ko_maturity"   # default: KO observation dates + terminal branch
    COUPON      = "coupon"        # Phoenix coupon observations (survival/coupon_probability, 计算天数因子)

@dataclass(frozen=True)
class AutocallableCashLeg(CashLeg):          # inherits direction(sign), name, leg_id
    leg_type: AutocallableLegType
    notional: float                          # absolute, per-unit (position scales by quantity)
    rate: float
    # --- schedule identity & accrual (all REQUIRED, all sourced from the workbook) ---
    observation_schedule: Sequence[float]    # FUTURE KO (or coupon) observation year-fractions
    accrual_factors: Sequence[float]         # one per observation, aligned to observation_schedule
    settlement_schedule: Sequence[float]     # settlement year-fraction per observation (for discounting)
    terminal_accrual_factor: float           # a_T for the maturity branch
    terminal_settlement_time: float          # explicit maturity-branch settlement year-fraction
    # --- semantics ---
    pv_formula: PvFormula = PvFormula.NORMAL
    accrual_basis: AccrualBasis = AccrualBasis.KO_MATURITY
    terminal_events: frozenset[EventType] = frozenset({MATURITY_NO_KO, MATURITY_WITH_KI})  # default ALL
    notional_settlement_time: Optional[float] = None   # see PvFormula.NOTIONAL_MINUS_PAYOFF (§5)
```

`leg_type` carries **no hidden valuation behaviour**. It sets defaults (all
default to `terminal_events = {MATURITY_NO_KO, MATURITY_WITH_KI}`) and is a
label for attribution/reporting. Every input that affects the number is an
explicit field.

**Why schedule, settlement, and terminal time are explicit & required** (review
findings #2, #3, #4): the parent's `EventDistribution` carries only KO
*observation* year-fractions and `event_dates=None` — it stores **no maturity
time and no settlement times** (`AutocallableEventStats` has neither; `from_
autocallable_stats` discards `ResolvedObservationRecord.settlement_time`). The
earlier draft inferred the terminal time as `event_times[-1]` and settlement as
`observation + offset`; both are unbacked and silently wrong under irregular
calendars, lockouts, as-of filtering, or EXPIRY-settled coupons. Per requirement
#3 ("workbook-provided accrual factors/days are ground truth"), the leg instead
takes the future observation times, per-observation settlement times, terminal
accrual factor, and terminal settlement time **explicitly from the workbook**,
and validates them against the parent's filtered grid (§5). No silent defaults.

## 5. Valuation math (exact — no approximations)

Parent future KO observations `t₁…tₘ` with first-KO probabilities `p_ko[i]`,
settlement times `sᵢ`, accrual factors `a[i]`. Terminal probability
`P_term = Σ probabilities over terminal_events` at terminal settlement `s_T`.

```
R = notional · rate · [ Σᵢ a[i] · p_ko[i] · DF(sᵢ)  +  a_T · P_term · DF(s_T) ]   (unsigned, ≥ 0)

PvFormula.NORMAL                →  PV = sign · R
PvFormula.NOTIONAL_MINUS_PAYOFF →  PV = sign · (notional − R)        # margin / prepayment
```

- `sign` is `CashLeg.sign()` (`+1` BUYER_RECEIVES, `−1` BUYER_PAYS).
- `tᵢ = observation_schedule[i]`; `p_ko[i] = EventDistribution.probabilities[EventType.KO][i]`.
- `P_term = Σ_{e ∈ terminal_events} EventDistribution.probabilities[e]`.
- `DF(·) = env.get_discount_factor(·)`.
- **Settlement times are explicit, not inferred** (finding #2/#3):
  `sᵢ = settlement_schedule[i]` (required, workbook-sourced); the terminal
  branch discounts at `s_T = terminal_settlement_time` (required,
  workbook-sourced — the product's actual maturity settlement, **not**
  `event_times[-1]`). The leg never relies on the last KO observation equalling
  maturity.

**COUPON basis** replaces the KO sum with the coupon stream
`Σⱼ ac[j] · coupon_probability[j] · DF(settleⱼ)` (survival/coupon-conditional)
using `PhoenixEventStats.coupon_probability`. Opt-in; everything defaults to
`KO_MATURITY`. (Phoenix MC only — see §3a.)

### NOTIONAL_MINUS_PAYOFF convention (finding #5)

`PV = sign · (notional − R)` mixes an **undiscounted notional claim** with a
**discounted contingent repayment** `R`. This is intentional: it reproduces the
legacy adapter identity `pv_margin = margin_notional − cash_leg_value`
(requirement #4), and represents the funding/carry value of a margin/prepayment
that **remains a live claim at the valuation date** and is repaid (with
accrued return) on KO/maturity. Constraints, enforced in code:

- Valid only when the notional is outstanding as of `valuation_date`. The leg
  models *only the outstanding claim net of contingent repayment* — it does
  **not** re-inject the original upfront exchange. `notional_settlement_time`,
  if set, must be `≤ 0` (already settled / posted at/before as-of); a strictly
  positive value (a future-dated notional exchange) is rejected with
  `ValidationError`, since that case needs an explicit `DeterministicLeg` for
  the upfront, not this formula.
- We **keep** the closed `notional − R` form rather than the reviewer's
  "two-leg" decomposition because the acceptance target (requirement #3) is to
  match the legacy `notional − cash_leg_value` number; the two-leg model would
  only coincide under identical settlement calibration and adds surface area for
  no functional gain here. The validity domain above is the guardrail.

### Alignment contract (fail loud — finding #4)

For a nontrivial leg (`accrual_factors` non-empty), `value()` enforces **all** of:

- The required probability stream exists: `EventType.KO` (KO_MATURITY) or
  `EventType.COUPON` (COUPON) is present in `event_dist.probabilities` as an
  array. Absent ⇒ `ValidationError` naming the engine/product (§3a fail-loud).
- `len(observation_schedule) == len(accrual_factors) == len(settlement_schedule)
  == len(p_ko_or_coupon_array)`. Any mismatch ⇒ `ValidationError`.
- **Schedule identity is mandatory, not optional:** `observation_schedule` must
  match `event_dist.event_times` elementwise within `Tolerance`. This catches an
  equal-length **shifted** workbook slice (e.g. a dropped holiday-adjusted
  observation or a today/past-date boundary mismatch) that a length check alone
  cannot — the failure mode the reviewer flagged. (`event_dates=None` in the
  current `from_autocallable_stats`, so a date-keyed guard is unavailable; the
  year-fraction identity check is the strongest available and is required.)
- No padding, no truncation, no silent realignment.

### ZL496 margin sanity check

`GJZQ-ZL496-20260514-OPTION-01`, notional `N = 20,004,513.86`,
`rate = 365/735`, `terminal_accrual_factor a_T = 735/365` ⇒
`a_T · rate = 1`, so the return leg repays exactly the notional at termination ⇒
`R = N · E[DF]` ⇒ `PV = N · (1 − E[DF]) = 207,475.74`. Matches the target model
PV (`207,475.74`; Tongyu `207,730.75`).

## 6. Greeks & integration

No risk-engine changes. `get_trade_value()` already sums leg PVs;
`get_trade_greeks()` already bumps it. Because `R` depends on
`p_ko[]` / `P_term` / `DF` and the parent re-solves on each spot bump, the
margin / rebate / interest legs acquire real delta/gamma. Method and as-of
KO-filtering inherit from the parent.

**Caveat — Phoenix QUAD Greeks are MC-derived (finding #1).** When the parent is
a Phoenix priced with QUAD, each bump re-solves event stats via the delegated
`PhoenixMCEngine(MCParams())`, so the legs' delta/gamma carry MC noise and are
not consistent with the QUAD parent. For stable Phoenix leg Greeks, use the MC
engine end-to-end (or wait for native Phoenix QUAD event stats, §9). Snowball
PDE/QUAD/MC and Phoenix MC produce deterministic, method-consistent leg Greeks.

Registration: export `AutocallableCashLeg` + enums from
`quantark/cashleg/__init__.py`; register in `cashleg` serialization registry if
one is present for round-trip (follow existing `LegRegistry` pattern).

## 7. Testing & validation

1. **Per-leg-type unit tests** — margin (`NOTIONAL_MINUS_PAYOFF`),
   backend_premium, backend_interest, rebate, minimum_return — each verified
   against an **independent re-implementation** of the §5 sum (not the
   production code path).
2. **Invariants & adversarial cases** — fail-loud guards: missing `EventType.KO`/
   `EventType.COUPON` stream (trivial distribution from Phoenix PDE / non-
   autocallable engine) → `ValidationError`; length mismatch across
   `observation_schedule` / `accrual_factors` / `settlement_schedule` / prob
   array; **shifted-but-equal-length** `observation_schedule` vs
   `event_dist.event_times` → `ValidationError` (the finding-#4 case); explicit
   `settlement_schedule` + `terminal_settlement_time` honored in discounting
   (finding #2/#3); a parent whose **last KO observation ≠ maturity** still
   discounts the terminal branch at `terminal_settlement_time` (regression for
   the dropped `event_times[-1]` assumption); `NOTIONAL_MINUS_PAYOFF` with a
   strictly-positive `notional_settlement_time` → `ValidationError` (finding #5);
   sign/direction; `terminal_events` bucketing; `COUPON` basis against a Phoenix
   MC `EventDistribution`.
3. **Greeks** — attach legs to an `EquityPosition` and confirm
   `get_trade_greeks()` produces non-zero, finite delta/gamma for the margin
   leg, and that a deterministic leg's contribution stays zero. Cover Snowball
   on PDE, QUAD, and MC (method-consistent); assert the Phoenix-PDE leg path
   raises rather than returning trivial Greeks.
4. **Regression (requirement #6)** — `DeterministicLeg` and fixed-maturity
   products remain unchanged: explicit test that fixed-maturity products keep
   deterministic cashflow style.
5. **ZL496 numeric acceptance is REQUIRED, not optional (finding #6).** A
   **sanitized golden fixture** is committed to `test/test_cashleg/fixtures/`
   (parent autocallable params + per-future-observation accrual & settlement
   factors + curve) and a CI test asserts the native legs' PV **and** delta/gamma
   against the workaround targets across every supported engine method (Snowball
   PDE/QUAD/MC; Phoenix MC). If the literal ZL496 confirm cannot be committed,
   the fixture is a sanitized equivalent whose **own** golden numbers are
   produced by the legacy synthetic-`SnowballOption` workaround and frozen into
   the repo — so CI always proves native-vs-workaround parity, never an
   optionally-skipped check. Target numbers (literal ZL496):
   - `pv_margin`:   model PV `207,475.74`, delta `−1,183,832.92`, gamma `98,566.13`
   - `pv_interest`: model PV `−3,417.10`,  delta `17,000.95`,     gamma `−1,393.67`
   - `pv_rebate`:   model PV `−409,798.69`, delta `−24,505.34`,    gamma `2,040.32`
   - total trade delta target (product + legs): `−13,990,506.81` (Tongyu `−14,019,368.71`)

   **Open input dependency:** providing the sanitized fixture data (or
   authorizing a synthetic stand-in generated from the workaround) is a
   prerequisite for closing requirement #3.

## 8. Acceptance criteria → design mapping

| # | Criterion | Satisfied by |
|---|-----------|--------------|
| 1 | Native autocallable cash-leg API | `AutocallableCashLeg` (§4) |
| 2 | Adapter removes synthetic `SnowballOption` workaround | Consume-parent architecture (§3) |
| 3 | PV/Greeks match workaround within tight tolerance | §5 math + §7 **required** golden fixture |
| 4 | Works for Snowball and Phoenix | Snowball MC/PDE/QUAD; Phoenix MC (`KO_MATURITY` + `COUPON`); Phoenix PDE/QUAD gated on engine support (§3a, §9) |
| 5 | Supports PDE/QUAD/MC where parent supports them | Snowball: all three (native). Phoenix: MC native; PDE → fail-loud, QUAD → ⚠️ MC-derived (§3a) |
| 6 | Unit tests cover all 5 leg types + deterministic legs unchanged | §7 items 1, 4 |

**Note on #4/#5:** the verified engine reality (§2, §3a) means "works for Phoenix
on all methods" is **not** achievable without first adding native Phoenix
PDE/QUAD event stats. This spec delivers Phoenix on MC and fails loud elsewhere;
full Phoenix method coverage is the §9 follow-on.

## 9. Out of scope

- New pricing engines or changes to autocallable engines — **including native
  `PhoenixEventStats` for `PhoenixPDESolver` and `PhoenixQuadEngine`.** That is
  the tracked follow-on that upgrades the §3a matrix; until it lands, Phoenix PDE
  fails loud and Phoenix QUAD is ⚠️ MC-derived.
- Changes to `GreeksCalculator` / `EquityPosition` risk path.
- Extending `AutocallableEventStats` / `EventDistribution` to carry settlement
  times / maturity time / observation dates — the consumer-only design takes
  these explicitly from the workbook (§4); propagating them from the engine is a
  possible future robustness improvement but not required here.
- Basket / multi-asset autocallable legs.
- KO-reset / Phoenix memory-coupon leg variants beyond the `COUPON` basis above
  (deferred unless ZL496 validation requires them).
