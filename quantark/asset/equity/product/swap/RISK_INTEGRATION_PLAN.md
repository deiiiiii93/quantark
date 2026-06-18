# Equity TRS — Risk-Stack Integration Plan

Goal: wire the realized-cashflow equity Total Return Swap (TRS) module into the
QuantArk risk/analytics layers: Position, Portfolio, Backtest, Stress Test,
Dynamic Scenario, VaR, SIMM, SA-CCR, SA-CVA.

## The core problem & the bridge

The TRS engine (`TotalReturnSwapEngine`) prices a swap from an **observed price
path** and returns a **per-period cashflow DataFrame** — it never takes a
`PricingEnvironment` and produces no single NPV/greeks. Every downstream risk
layer, by contrast, is built on `value(product, pricing_env) -> NPV` plus
**deepcopy-and-bump** revaluation of a `PricingEnvironment`.

**Bridge** (`engine/cashflow/trs_valuation.py :: TRSValuationEngine`): map a
`PricingEnvironment`'s spot (and, opt-in, funding rate) onto the **valuation-date
slice** of the TRS price path, re-run the cashflow engine, and read the
valuation-date `present_value` as the swap mark-to-market. A TRS is delta-one on
its float leg, so:
- `delta = d(PV)/d(spot)` (≈ `asset_quantity × float_dir`) via bump-revaluation
- `gamma ≈ 0`, `vega = 0` (no vol dependence)
- `rho` via a direct `fix_leg.rate` bump (funding-rate exposure of remaining life)

Because the bridge consumes a standard `PricingEnvironment`, every revaluation
layer (VaR FD, stress bumps, dynamic scenario) works on a swap position with no
changes to those engines — the impedance mismatch is absorbed in one place.

### Financing-rate flow-through (design choice)
- Default (`funding_rate_ref=None`): financing leg is the **contractual fixed
  rate** `fix_leg.rate`; env rate shocks do not move base MtM. `rho` still
  reported by bumping `fix_leg.rate`. (Exact semantics for a fixed-financing TRS.)
- Opt-in (`funding_rate_ref=r0`): floating financing — `financing(env) =
  fix_leg.rate + (env.get_rate(tenor) - r0)`, so env rate shocks flow through.

## Gates (each gate ends with `/zenmux-codex-review-loop`, max 3 loops)

- **Gate 1 — Foundation: pricing bridge + Position + Portfolio**
  `TRSValuationEngine`; `EquitySwapPosition` (structural `BasePosition`);
  `EquityPortfolio` acceptance (value/pnl/greeks aggregation). Tests.
- **Gate 2 — VaR**: parametric/historical/MC revaluation of swap positions via
  equity risk factors (spot) + funding factor. Tests.
- **Gate 3 — Stress Test + Dynamic Scenario**: spot/rate stress bumps;
  multi-day path simulation + lifecycle (maturity). Tests.
- **Gate 4 — Backtest**: delta-hedged TRS backtest over a price path. Tests.
- **Gate 5 — SIMM**: `get_simm_sensitivities` → Equity delta (and IR for
  floating financing). Tests.
- **Gate 6 — SA-CCR**: TRS → equity-derivative trade (adjusted notional,
  supervisory delta/factor, maturity factor) in a netting set. Tests.
- **Gate 7 — SA-CVA**: consume supplied/derived equity + IR CVA sensitivities.
  Tests.

## Status
- [x] Gate 1 (review clean, 1 iter)
- [x] Gate 2 (review 3 iters; 1 known limitation below)
- [x] Gate 3 (no engine changes — stress + dynamic are already duck-typed)
- [x] Gate 4 (backtest: swap-aware position init; step loop already duck-typed)
- [x] Gate 5 (SIMM: EquitySwapPosition.get_simm_sensitivities -> EquityDelta)
- [x] Gate 6 (SA-CCR: EquitySwapPosition.to_saccr_trade -> EQUITY trade)
- [~] Gate 7 (SA-CVA: not applicable in this branch — see note)

### Gate 7 note (SA-CVA) — not integrated here
Two reasons, both blocking and both outside the scope of this branch:
1. **Module absent.** There is no `quantark/sacva` in this codebase; the Basel
   SA-CVA (MAR50) SBA engine lives only on the separate, unmerged
   `worktree-sacva` branch. Pulling it in would conflate two independent
   unmerged feature efforts in this diff.
2. **No per-instrument hook by design.** The SA-CVA SBA engine consumes
   *supplied* CVA and CVA-hedge sensitivities (delta/vega per risk factor); it
   does not derive them from instruments/positions. So unlike VaR/SIMM/SA-CCR
   there is no `position -> engine` seam to add. Producing a TRS's equity (and,
   for floating funding, IR) CVA sensitivities requires a CVA model (expected-
   exposure simulation over the swap's life), which is a separate workstream.

What *is* delivered toward counterparty/CVA capital: the TRS's
counterparty-exposure path — SA-CCR EAD (Gate 6) — which is the exposure input a
CVA framework builds on. A follow-up on top of `worktree-sacva` can feed
TRS-derived CVA sensitivities into the SBA engine.

### Gate 6 note
`EquitySwapPosition.to_saccr_trade(env, is_index=)` maps the TRS to a SA-CCR
`AssetClass.EQUITY` trade: adjusted notional = current spot x shares, supervisory
delta +/-1 by economic direction (quantity x float-leg direction), maturity =
remaining tenor (floored to the 10-business-day SA-CCR minimum), market value =
current MtM. Single-name (32%) vs index (20%) supervisory factor via `is_index`.
The trade flows through SACCRNettingSet -> SACCRCalculator to an EAD.

### Gate 5 note
`EquitySwapPosition.get_simm_sensitivities` mirrors `EquityPosition`'s: the
equity sensitivity engine is already duck-typed on `get_greeks`/`underlying`, so
it derives the SIMM EquityDelta (`0.01 * spot * delta_$`) from the swap's
delta-one greek. No vega (TRS has none). Flows through SIMMPortfolioAdapter ->
SIMMCalculator to an IM number. IR (financing) SIMM delta is out of scope here
(would require the opt-in floating-funding curve sensitivity).

### Gate 4 note
`BacktestEngine._initialize` now registers `EquitySwapPosition`s via
`add_swap_position` (they have no separate engine); the step/greeks/value loop
was already duck-typed. A swap is revalued against each step's spot (delta-one
PnL). Caveat: the engine marks a spot delta-hedge at market value without
booking the offsetting financing cash, so absolute PnL of a *hedged* swap run is
not meaningful — a pre-existing engine accounting characteristic, unrelated to
the TRS. Delta hedging of the swap itself works (hedges trigger and trade the
underlying).

### Gate 3 note
The `EquityStressEngine`, `StressApplicator`, and `DynamicScenarioEngine` are
already position-agnostic (they bump pricing environments and value through the
`BasePosition` interface). A swap drops in with no engine changes; the
dynamic-scenario `LifecycleManager` tracks only autocallable/barrier options and
silently skips a TRS (marked to market each day against that day's spot). A swap
marks against its fixed contractual valuation date with the day's spot
substituted — capturing the delta-one PnL evolution; intra-path financing carry
is not advanced (consistent with the mark-at-valuation-date semantics).

## Known limitations
- **Parametric component / incremental / marginal VaR attribution is spot-delta
  only** (pre-existing; the per-position P&L decomposition uses `factor_returns[
  :,0]·delta`). Portfolio-total VaR correctly includes vega/rho, but the
  per-position split understates non-spot factors. Immaterial for a TRS
  (vega=0; rate risk only under opt-in funding flow-through). Flagged by the
  Gate 2 review (iter 3); deferred — not a regression.
