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

1. **All autocallable engines emit event timing.** `SnowballMCEngine`,
   `PhoenixMCEngine`, `SnowballPDESolver`, `PhoenixPDESolver`,
   `SnowballQuadEngine`, `PhoenixQuadEngine` all implement
   `calculate_event_stats()` →
   `AutocallableEventStats` / `PhoenixEventStats`
   (`quantark/asset/equity/engine/event_stats.py`).

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
    accrual_factors: Sequence[float]         # one per FUTURE KO (or coupon) observation
    terminal_accrual_factor: float           # a_T for the maturity branch
    pv_formula: PvFormula = PvFormula.NORMAL
    accrual_basis: AccrualBasis = AccrualBasis.KO_MATURITY
    terminal_events: frozenset[EventType] = frozenset({MATURITY_NO_KO, MATURITY_WITH_KI})  # default ALL
    settlement_schedule: Optional[Sequence[float]] = None   # explicit settle year-fractions; else…
    settlement_offset_days: int = 0                          # …settle = observation + offset (0 ⇒ settle=observe)
    terminal_settlement_offset_days: int = 0
    observation_schedule: Optional[Sequence[float]] = None   # OPTIONAL consistency guard (§5)
```

`leg_type` carries **no hidden valuation behaviour**. It sets defaults (all
default to `terminal_events = {MATURITY_NO_KO, MATURITY_WITH_KI}`) and is a
label for attribution/reporting. Every input that affects the number is an
explicit field.

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
- `p_ko[i] = EventDistribution.probabilities[EventType.KO][i]`.
- `P_term = Σ_{e ∈ terminal_events} EventDistribution.probabilities[e]`.
- `DF(·) = env.get_discount_factor(·)`.
- Settlement times: `sᵢ = settlement_schedule[i]` if provided, else
  `tᵢ + settlement_offset_days/365`. The terminal time is
  `event_dist.event_times[-1]` (mirroring `FixedPayoffLeg`'s maturity handling,
  `fixed_payoff_leg.py:65,86` — the final KO observation sits at maturity), so
  `s_T = event_dist.event_times[-1] + terminal_settlement_offset_days/365`. The
  leg never needs a product/maturity handle inside `value()`.

**COUPON basis** replaces the KO sum with the coupon stream
`Σⱼ ac[j] · coupon_probability[j] · DF(settleⱼ)` (survival/coupon-conditional)
using `PhoenixEventStats.coupon_probability`. Opt-in; everything defaults to
`KO_MATURITY`.

### Alignment contract (fail loud)

- `len(accrual_factors)` **must** equal the number of future KO (or coupon)
  observations in the `EventDistribution`. Mismatch → `ValidationError` — no
  padding, no silent truncation (per the "no stupid fallbacks" / "exact
  semantics" rules). The caller (adapter) pre-filters its workbook factors to
  the future set, the same way the product filters observations.
- If `observation_schedule` is supplied, `value()` additionally asserts it
  matches `event_dist.event_times` within `Tolerance` — a guard that the leg's
  factors line up with the parent's *filtered* future grid.
- `settlement_schedule`, when provided, must match `accrual_factors` in length.

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

Registration: export `AutocallableCashLeg` + enums from
`quantark/cashleg/__init__.py`; register in `cashleg` serialization registry if
one is present for round-trip (follow existing `LegRegistry` pattern).

## 7. Testing & validation

1. **Per-leg-type unit tests** — margin (`NOTIONAL_MINUS_PAYOFF`),
   backend_premium, backend_interest, rebate, minimum_return — each verified
   against an **independent re-implementation** of the §5 sum (not the
   production code path).
2. **Invariants** — alignment-length `ValidationError`; `observation_schedule`
   guard; `settlement_schedule` length check; sign/direction; settlement-lag
   discounting; `terminal_events` bucketing; `COUPON` basis against a Phoenix
   `EventDistribution`.
3. **Greeks** — attach legs to an `EquityPosition` and confirm
   `get_trade_greeks()` produces non-zero, finite delta/gamma for the margin
   leg, and that a deterministic leg's contribution stays zero.
4. **Regression (requirement #6)** — `DeterministicLeg` and fixed-maturity
   products remain unchanged: explicit test that fixed-maturity products keep
   deterministic cashflow style.
5. **ZL496 numeric acceptance (PV + delta/gamma within tight tolerance)** —
   needs the workbook's parent `SnowballOption` params + per-future-date accrual
   factors + curve. **Resolution:** quant-ark holds the synthetic-equivalent
   math test (items 1–4) unconditionally. The ZL496 numeric match is gated on a
   fixture: if the ZL496 inputs are dropped into
   `test/test_cashleg/fixtures/`, a `pytest` test asserts the PV/delta/gamma of
   `pv_margin` / `pv_interest` / `pv_rebate` against the workaround targets;
   otherwise that numeric match lives in the adapter repo and quant-ark relies
   on the synthetic math test. Target numbers:
   - `pv_margin`:   model PV `207,475.74`, delta `−1,183,832.92`, gamma `98,566.13`
   - `pv_interest`: model PV `−3,417.10`,  delta `17,000.95`,     gamma `−1,393.67`
   - `pv_rebate`:   model PV `−409,798.69`, delta `−24,505.34`,    gamma `2,040.32`
   - total trade delta target (product + legs): `−13,990,506.81` (Tongyu `−14,019,368.71`)

## 8. Acceptance criteria → design mapping

| # | Criterion | Satisfied by |
|---|-----------|--------------|
| 1 | Native autocallable cash-leg API | `AutocallableCashLeg` (§4) |
| 2 | Adapter removes synthetic `SnowballOption` workaround | Consume-parent architecture (§3) |
| 3 | PV/Greeks match workaround within tight tolerance | §5 math + §7 fixture test |
| 4 | Works for Snowball and Phoenix | `KO_MATURITY` + `COUPON` basis (§5) |
| 5 | Supports PDE/QUAD/MC where parent supports them | Inherited from parent engine (§2, §6) |
| 6 | Unit tests cover all 5 leg types + deterministic legs unchanged | §7 items 1, 4 |

## 9. Out of scope

- New pricing engines or changes to autocallable engines.
- Changes to `GreeksCalculator` / `EquityPosition` risk path.
- Basket / multi-asset autocallable legs.
- KO-reset / Phoenix memory-coupon leg variants beyond the `COUPON` basis above
  (deferred unless ZL496 validation requires them).
