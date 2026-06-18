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
- [ ] Gate 1  - [ ] Gate 2  - [ ] Gate 3  - [ ] Gate 4
- [ ] Gate 5  - [ ] Gate 6  - [ ] Gate 7
