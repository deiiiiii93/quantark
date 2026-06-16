# Smile-Consistent FX Vanillas & Digitals — Design Spec

**Date:** 2026-06-16
**Status:** In review (ZenMux)
**Branch:** worktree-vv-chain-integration (continues the VV chain integration)

## 1. Goal

Make FX **vanilla** and **European digital** options fully smile-consistent —
price *and* Greeks — when priced under a `VannaVolgaVolSurface`, so they behave
correctly throughout the chain (Greeks, VaR, backtest) like the VV barrier
products already do.

Two gaps motivate this (see §2):
1. **Digital price** omits the smile-skew term (`−vega·∂σ/∂K`).
2. **Vanilla and digital Greeks** are sticky-strike closed forms at a single
   `σ(K)`; they ignore the smile's spot/quote dependence.

Vanilla *price* is already smile-consistent and is left unchanged.

## 2. Background — current state

| Path | File | Today |
|------|------|-------|
| Vanilla price | `garman_kohlhagen_engine.py:62` | `sigma = fx_env.get_vol(strike, tau)` then BS → under a VV surface this IS the full VV vanilla. **Already correct.** |
| Vanilla Greeks | `garman_kohlhagen_engine.py:85,111` | Closed forms at the single `σ(K)` → sticky-strike, not smile-consistent. |
| Digital price | `fx_digital_engine.py:64` | `BS_digital(σ(K))` = `N(d2)` at the strike vol → captures smile **level** only, **not** the skew term. |
| Digital Greeks | `fx_digital_engine.py:74,94` | Closed forms at `σ(K)` → sticky-strike + missing skew. |

The VV smile is already a first-class `VolatilitySurface` (env-anchored,
sticky-delta) and `BaseFxEngine._vol_scaled_env` already knows how to bump it
(full-quote vega). So bump-and-reprice on these engines would *already* produce
smile-consistent Greeks — except both engines **override** `calculate_greeks`
with closed forms, bypassing that path.

## 3. Decisions (resolved in brainstorming)

| Decision | Choice |
|----------|--------|
| Coverage | Price **and** Greeks, for both vanillas and digitals |
| Delivery | **Upgrade the existing engines** (`GarmanKohlhagenEngine`, `FxDigitalOptionAnalyticalEngine`) to be smile-aware; no new engines, no API/flag churn |
| Digital price method | **Static replication** (`−∂C/∂K`, the call-spread limit) off the smile-consistent vanilla |
| Non-VV surfaces | **Unchanged** — keep existing closed forms for `FlatVolSurface` / `TermStructureVolSurface`. This change is purely additive for the VV case. |

## 4. Architecture

### 4.1 Vanilla — `GarmanKohlhagenEngine`

- **Price:** unchanged.
- **Greeks:** override `calculate_greeks` to branch on surface type:
  - `isinstance(fx_env.vol_surface, VannaVolgaVolSurface)` → delegate to
    `BaseFxEngine.calculate_greeks` (bump-and-reprice; re-anchors the smile
    sticky-delta for spot/rate bumps and shifts all quotes for vega).
  - otherwise → existing closed forms (unchanged).

### 4.2 Digital — `FxDigitalOptionAnalyticalEngine`

- **Price under a VV surface — static replication.** The atom is the
  cash-or-nothing call paying one unit:

  ```
  cash_call(K) = −∂C/∂K ≈ (C(K − h) − C(K + h)) / (2h)
  ```

  where `C(·)` is the smile-consistent vanilla priced by `GarmanKohlhagenEngine`
  (so `C(K ± h)` use `σ(K ± h)` and the central difference captures the skew).
  All four digital variants derive by no-arbitrage identities:

  | Variant | Formula |
  |---------|---------|
  | cash-or-nothing call (pay 1 if S_T > K) | `cash_call` |
  | cash-or-nothing put (pay 1 if S_T < K) | `DF_dom(T_del) − cash_call` |
  | asset-or-nothing call | `C(K) + K · cash_call` |
  | asset-or-nothing put | `S·DF_for(T_del) − (C(K) + K · cash_call)` |

  Result scaled by `payout · participation_rate` and the chosen `payout_currency`
  (`DOMESTIC` = cash-or-nothing, `FOREIGN` = asset-or-nothing).

- **Price under flat/term surface:** existing closed form (no skew ⇒ identical
  to replication, but exact and cheaper).
- **Greeks:** same surface branch — bump-and-reprice when VV, closed form
  otherwise.

### 4.3 Replication bump size `h`

- Relative to the strike: `h = max(h_floor, h_rel · K)` with a small `h_rel`
  (e.g. `1e-4`) and an absolute floor, ensuring `K − h > 0`.
- Must stay well inside the surface's valid strike range; the VV surface caps
  the implied vol at extreme moneyness (degenerate-vega floor), so `C(K ± h)`
  stays well-posed.

## 5. Error handling

- No new error surfaces. Non-VV surfaces keep their documented behavior.
- The VV path needs the surface's quotes, guaranteed by the `isinstance` type
  check; no silent fallback to flat vol.
- Near-expiry / `tau == 0` keeps the existing terminal-payoff branch in both
  engines (replication is only invoked for `tau > 0`).

## 6. Testing

- **Digital replication reduces to closed form (flat smile):** under
  `FlatVolSurface` (and a flat VV smile, RR = BF = 0) the replicated digital
  equals the closed-form `N(d2)` digital to tight tolerance.
- **Skew direction:** under a negative `RR25`, the smile-consistent digital
  differs from the `σ(K)`-only digital with the correct sign.
- **Put-call / cash-asset parities** under the VV surface:
  `cash_call + cash_put = DF_dom`; `asset_call + asset_put = S·DF_for`.
- **Vanilla price regression:** unchanged under the VV surface.
- **Greeks:** VV-surface Greeks for both products match a manual bump-reprice
  and are finite; flat-surface Greeks unchanged (closed-form path still taken).
- **Chain smoke:** a vanilla and a digital under a VV surface run through
  `calculate_greeks` and FX VaR without raising.
- **Regression:** existing `test_fx_*` suites stay green (closed-form paths
  untouched).

## 7. References

- Castagna & Mercurio, "The Vanna-Volga method for implied volatilities,"
  *Risk* (2007).
- Wystup, *FX Options and Structured Products* (digital/barrier smile risk).
- Breeden & Litzenberger (1978) — digital = call-spread limit `−∂C/∂K`
  (static replication basis).

## 8. Files touched

Modify:
- `quantark/asset/fx/engine/analytical/garman_kohlhagen_engine.py`
  (smile-aware `calculate_greeks` branch)
- `quantark/asset/fx/engine/analytical/fx_digital_engine.py`
  (replication price under VV surface + smile-aware `calculate_greeks` branch)

Tests:
- `test/test_fx_digital_smile.py` (new), additions to existing FX vanilla/digital
  and chain-integration suites.
