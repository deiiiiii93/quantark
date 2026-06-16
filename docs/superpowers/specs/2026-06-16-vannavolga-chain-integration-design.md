# Vanna-Volga Chain Integration — Design Spec

**Date:** 2026-06-16
**Status:** Approved for planning
**Scope owner:** fuxinyao

## 1. Goal

Make the Vanna-Volga (VV) FX engine a first-class citizen of the quant-ark
pricing convention so that VV-priced products flow through the entire risk
chain — Greeks, VaR, backtest, dynamic scenario — exactly like any other FX
product. Today the VV code is a standalone functional pricer that lives
outside the `BaseFxEngine` / `FxPricingEnvironment` convention and cannot ride
the chain.

This project covers two instrument families:

1. **One-touch** (already priced by `price_vv_one_touch`, needs wrapping).
2. **Vanilla single-barrier options** — knock-out *and* knock-in, all 8
   standard types (up/down × in/out × call/put), with optional rebate. The
   Black-Scholes baseline for these does **not** exist yet and must be
   implemented (Reiner-Rubinstein).

Out of scope (explicitly deferred, not fabricated): double-barrier / DKO
pricing (Ikeda-Kunitomo), window/partial barriers, and VV vanilla-barrier
correction for double barriers. The `enforce_double_barrier_arbitrage` clamp
stays unused until a future double-barrier sub-project.

## 2. Background — current state

Two VV modules exist, split across the param and engine layers:

| Module | Layer | Today |
|--------|-------|-------|
| `quantark/param/vol/vannavolga/` | Parameter | `VannaVolgaVolSurface` — a `VolatilitySurface` subclass, but stores its **own** frozen `FXEnv(spot, rd, rf, tau)` independent of any pricing environment |
| `quantark/asset/fx/engine/analytical/vannavolga/` | Engine | `price_vv_one_touch(env, quotes, ...)` — a **function** taking `FXEnv` + `SmileQuotes`, returning `VVBarrierResult`. Not a `BaseFxEngine`. |

Three structural gaps block chain integration:

1. **No product.** Only vanilla / digital / quanto FX products exist. No
   one-touch and no barrier product subclass `BaseFxProduct`.
2. **No engine class.** The standard contract is
   `BaseFxEngine.price(product, fx_env)` + bump-and-reprice Greeks on a
   deep-copied `FxPricingEnvironment`. The VV barrier is a free function in a
   different universe (`FXEnv` + `SmileQuotes`).
3. **The chain cannot bump a VV smile.** Vol bumping is done by `isinstance`
   checks at three sites — `base_fx_engine._vol_scaled_env`,
   `var/fx/revaluation._shift_vol`, and (transitively) anything built on them.
   All three only handle `FlatVolSurface` and `TermStructureVolSurface`; a VV
   surface currently **raises**.

Existing building blocks we reuse:

- `arbitrage.py` — `BarrierPrices` + `enforce_single_barrier_arbitrage`
  (already enforces `0 <= ko <= wko <= vanilla`). Built in anticipation of
  vanilla-KO, never wired.
- `barrier_bs.py` — `one_touch_hit_prob`, `no_touch_price`,
  `survival_probability_single` (Reiner-Rubinstein touch probabilities).
  **Missing:** the Reiner-Rubinstein vanilla knock-out/knock-in pricer.
- `attenuation.py` — `gamma_surv`, `gamma_fet`,
  `p_vanna_p_volga_from_gamma` (survival / first-exit-time attenuation).
- `vv_core.compute_omega` — vanilla VV Omega weights.

## 3. Decisions (resolved during brainstorming)

| Decision | Choice |
|----------|--------|
| Integration scope | Barrier engine **and** VV smile promoted to a first-class bumpable surface |
| Smile bump semantics | **Sticky-delta** spot (smile re-anchors to bumped spot), **full-quote** vega (bump ATM + RR + BF together) |
| Smile–environment coupling | **Approach A** — env-derived anchor; smile carries only quotes + convention as intrinsic data |
| Barrier family | Single-barrier **KO + KI + rebate**, all 8 standard types; double-barrier deferred |
| KI pricing | Priced directly by Reiner-Rubinstein; in-out parity (`KI + KO = vanilla`) used as a validation oracle |
| Canonical Greeks | Chain-consistent **bump-and-reprice** (now enabled by a bumpable smile). VV-native vanna/volga decomposition retained as supplementary diagnostics. |

## 4. Architecture

### 4.1 Product layer

New file `quantark/asset/fx/product/option/fx_one_touch_option.py`:

```
class FxOneTouchOption(BaseFxProduct):
    barrier: float
    is_up: bool                  # True = up-barrier, False = down-barrier
    payout: float = 1.0          # domestic units paid on touch (expiry-pay)
    # + inherited maturity/delivery/premium fields
```

New file `quantark/asset/fx/product/option/fx_barrier_option.py`:

```
class FxBarrierOption(BaseFxProduct):
    strike: float
    barrier: float
    is_up: bool
    knock_type: FxBarrierType    # KNOCK_IN | KNOCK_OUT
    option_type: OptionType      # CALL | PUT
    rebate: float = 0.0
    rebate_at_hit: bool = False  # KO only: pay rebate at hit vs at expiry
    # + inherited maturity/delivery/premium fields
```

Rebate semantics are knock-type dependent and must not be conflated:
- **KO:** rebate paid when the barrier *is* hit; `rebate_at_hit` selects
  pay-at-hit vs pay-at-expiry.
- **KI:** rebate paid at expiry only if the barrier is *never* hit (the
  option never knocked in); `rebate_at_hit` is ignored and rejected if set.

- New `FxBarrierType` enum in `quantark/util/enum/`.
- `get_payoff(spot)` returns the *unconditional terminal vanilla payoff*
  (barrier monitoring is an engine concern; payoff is used only by MC
  cross-checks, not the analytic engine — mirrors how the existing one-touch
  function keeps path logic in the pricer).
- `validate()` rejects non-positive strike/barrier/payout, validates enums,
  and calls `_validate_maturity_inputs()` (same pattern as `FxDigitalOption`).

### 4.2 Smile coupling — env-anchored `VannaVolgaVolSurface` (the crux)

Refactor `quantark/param/vol/vannavolga/vv_surface.py` so the smile's only
**intrinsic** data is the three quotes (ATM / RR25 / BF25) plus the delta
convention. The **market anchor** (spot, rd, rf, tau) is sourced from the live
environment rather than frozen at construction.

- Constructor stays backward-compatible: `VannaVolgaVolSurface(env, quotes,
  conv)` still works; the passed `env` becomes the *default anchor* used for
  standalone / analytic use, keeping the existing `test_vanna_volga_smile.py`
  suite green.
- New `rebound(spot, rd, rf, tau) -> VannaVolgaVolSurface` returns a **new,
  immutable** surface with the same quotes/conv and a new anchor. This is how
  the chain re-anchors the smile to a bumped environment.
- New `with_quotes(quotes) -> VannaVolgaVolSurface` returns a new surface with
  shifted ATM/RR/BF — the full-quote vega path.

How each chain bump is realized:

| Bump | Mechanism | Result |
|------|-----------|--------|
| Spot (delta/gamma) | env spot moves; engine builds `FXEnv` from the bumped env, `get_vol` uses the passed-in spot; 25d strikes recompute | **Sticky-delta, structural** |
| Rate (rho) | env curves move; anchor re-sourced via `rebound`; strikes/omega recompute | Re-anchored |
| Vol (vega) | `with_quotes` shifts all three quotes | **Full-quote, sticky-delta** |

Honest interface note: `VolatilitySurface.get_vol(strike, tau, spot)` supplies
spot but **not** rd/rf. A rate-dependent FX smile genuinely needs rd/rf to
locate 25d strikes, so the **barrier engine builds `FXEnv` directly from the
environment** (it has full env access) instead of routing through the narrow
`get_vol`. The generic `get_vol` path keeps the default anchor's rates,
refreshed by `rebound` at bump sites. Returning new surfaces (never mutating)
preserves the "Immutable Market Data" principle.

### 4.3 Math layer (the genuinely new work)

**`barrier_bs.py` extension — `reiner_rubinstein_barrier(...)`**

The standard Reiner-Rubinstein single-barrier analytic pricer: the A–F term
decomposition parameterized by η (barrier side, ±1) and φ (call/put, ±1),
covering all 8 KO/KI types plus rebate (paid at hit or at expiry). This is the
Black-Scholes baseline (BSTV) for `FxBarrierOption`.

- Formulas verified against Haug, *The Complete Guide to Option Pricing
  Formulas* (standard barrier chapter) and cross-checked numerically against
  QuantLib `AnalyticBarrierEngine` in the optional-QuantLib test path.
- **No approximation, no fallback.** If a configuration cannot be priced
  exactly it raises; it does not silently degrade.

**`vv_vanilla_barrier.py` (new) — `price_vv_barrier(...)`**

Castagna-Mercurio survival-weighted VV correction for vanilla barriers,
structurally identical to the one-touch path:

```
VV = BS_RR + p_vanna * vanna * Omega[vanna] + p_volga * volga * Omega[volga]
VV = enforce_single_barrier_arbitrage(BarrierPrices(vanilla=..., ko=VV)).ko
```

- BS baseline from `reiner_rubinstein_barrier`.
- Numeric vega/vanna/volga of the RR price via the same finite-difference
  pattern as `numeric_greeks_ot`.
- Omega weights from `compute_omega` (reused unchanged).
- Survival attenuation from `gamma_surv` + `p_vanna_p_volga_from_gamma`
  (reused unchanged).
- KO clamped to `[0, vanilla]` via `enforce_single_barrier_arbitrage`
  (finally wiring the existing clamp).
- KI priced directly via RR; `KI + KO = vanilla` checked as an invariant.
- Returns a `VVBarrierResult`-style record (bstv, vv, gamma, p_vanna,
  p_volga, omega, greeks).

### 4.4 Engine layer — `VannaVolgaBarrierEngine(BaseFxEngine)`

New `quantark/asset/fx/engine/analytical/vannavolga/vv_barrier_engine.py`.

- A shared adapter mixin builds `FXEnv(spot, rd, rf, tau)` from the
  `FxPricingEnvironment` (`fx_env.spot`, `get_domestic_rate(tau)`,
  `get_foreign_rate(tau)`, `product.get_maturity(fx_env)`) and sources
  `SmileQuotes` + `DeltaConvention` from `fx_env.vol_surface`.
- `price(product, fx_env) -> float` dispatches on product type:
  - `FxOneTouchOption` → `price_vv_one_touch(...).vv`
  - `FxBarrierOption` → `price_vv_barrier(...).vv`
- `price_details(product, fx_env) -> VVBarrierResult` preserves the full VV
  decomposition (mirrors `FxDeltaOneEngine`'s price + price_details).
- `calculate_greeks` inherits the `BaseFxEngine` bump-and-reprice machinery
  unchanged — it now *works* because the smile is bumpable and re-anchors
  sticky-delta. VV-native vanna/volga are surfaced as extra diagnostics, not
  the canonical Greeks.
- **No silent fallback:** if `fx_env.vol_surface` is not a
  `VannaVolgaVolSurface`, raise `MarketDataError` (do not fall back to flat
  vol).

### 4.5 Chain wiring

Add one `VannaVolgaVolSurface` branch at each of the three vol-bump sites:

- `base_fx_engine._vol_scaled_env` → `surface.with_quotes(scaled quotes)`.
- `var/fx/revaluation._shift_vol` → same VV branch (parametric + historical +
  MC FX VaR then revalue correctly under VV).
- Backtest / dynamicscenario consume `price` / `calculate_greeks`, so they
  inherit support once the engine is a `BaseFxEngine`. **No per-module change
  expected — to be verified, not assumed.**

Registration: export `FxOneTouchOption`, `FxBarrierOption`, `FxBarrierType`,
and `VannaVolgaBarrierEngine` through the FX `__init__` chain; wire into the
FX engine-type enum if one is used.

## 5. Error handling

- Non-VV surface where VV is required → `MarketDataError`.
- Non-positive barrier / strike / payout, negative tau → `ValidationError`
  (barrier engine already validates barrier/tau in `price_vv_one_touch`).
- Matured instrument (tau == 0) → existing immediate-settlement branch
  (one-touch) / intrinsic terminal payoff (barrier); no smile calibration.
- Unsupported double-barrier product → `ValidationError` with a clear message
  pointing at the deferred sub-project (never a wrong-but-quiet number).

## 6. Testing strategy

**Products** — validation, payoff, repr (mirror `test_fx_*.py` style).

**RR baseline** — vs Haug worked examples; numeric cross-check vs QuantLib
`AnalyticBarrierEngine` (skipped unless QuantLib installed, like the existing
`test_fx_quantlib_validation.py`).

**In-out parity** — `KI + KO = vanilla` (rebate = 0) for each of the 4
call/put × up/down combinations.

**No-arbitrage** — `0 <= KO <= vanilla`; limiting behavior (KO → vanilla as
the barrier moves far out-of-the-money region; KO → 0 as barrier → spot);
rebate limits.

**VV → BS reduction** — flat smile (RR = BF = 0) ⇒ VV correction = 0 for both
one-touch and vanilla barrier.

**Engine parity** — `VannaVolgaBarrierEngine.price` equals the underlying
`price_vv_one_touch` / `price_vv_barrier` `.vv` for the same market (adapter
faithfulness).

**Greeks** — bump-and-reprice delta/vega finite and sign-correct; sticky-delta
check (engine delta matches a manual re-anchored-smile reprice).

**Chain smokes** — a one-touch and a barrier position each run through
`calculate_greeks`, parametric + historical FX VaR, and a short backtest path
without raising on the VV surface.

**Regression** — existing `test_vanna_volga_smile.py` and `test_vv_barrier.py`
stay green after the surface refactor.

**MC cross-check (optional, gated)** — one-touch / barrier vs a barrier MC
under flat vol, coarse no-smile sanity bound.

## 7. References

- E. Reiner & M. Rubinstein, "Breaking Down the Barriers," *Risk* (1991).
- E. G. Haug, *The Complete Guide to Option Pricing Formulas*, 2nd ed.
  (single-barrier analytic formulas, rebate terms).
- A. Castagna & F. Mercurio, "The Vanna-Volga method for implied
  volatilities," *Risk* (2007).
- U. Wystup, *FX Options and Structured Products* (FX barrier VV survival
  weighting, market conventions).

## 8. Files touched (summary)

New:
- `quantark/asset/fx/product/option/fx_one_touch_option.py`
- `quantark/asset/fx/product/option/fx_barrier_option.py`
- `quantark/util/enum/` — `FxBarrierType`
- `quantark/asset/fx/engine/analytical/vannavolga/vv_vanilla_barrier.py`
- `quantark/asset/fx/engine/analytical/vannavolga/vv_barrier_engine.py`
- `test/test_fx_one_touch.py`, `test/test_fx_barrier_option.py`,
  `test/test_vv_barrier_engine.py`, `test/test_vv_chain_integration.py`

Modified:
- `quantark/param/vol/vannavolga/vv_surface.py` (env-anchored refactor +
  `rebound` / `with_quotes`)
- `quantark/asset/fx/engine/analytical/vannavolga/barrier_bs.py`
  (`reiner_rubinstein_barrier`)
- `quantark/asset/fx/engine/base_fx_engine.py` (`_vol_scaled_env` VV branch)
- `quantark/var/fx/revaluation.py` (`_shift_vol` VV branch)
- FX `__init__` exports; engine-type enum if applicable
- `quantark/asset/fx/engine/analytical/vannavolga/vv_barrier.py` (remove the
  `# TODO(vanilla-KO)` once wired)
