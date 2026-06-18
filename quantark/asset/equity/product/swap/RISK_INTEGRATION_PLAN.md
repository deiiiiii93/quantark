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
- **Gate 7 — SA-CVA**: `to_cva_trade` -> CVATrade -> MC exposure -> SBA capital.
  Phase A exact (single-period); Phase B opt-in approx (market-value financing).
  Tests.

## Status
- [x] Gate 1 (review clean, 1 iter)
- [x] Gate 2 (review 3 iters; 1 known limitation below)
- [x] Gate 3 (no engine changes — stress + dynamic are already duck-typed)
- [x] Gate 4 (backtest: swap-aware position init; step loop already duck-typed)
- [x] Gate 5 (SIMM: EquitySwapPosition.get_simm_sensitivities -> EquityDelta)
- [x] Gate 6 (SA-CCR: EquitySwapPosition.to_saccr_trade -> EQUITY trade)
- [x] Gate 7 (SA-CVA: EquitySwapPosition.to_cva_trade -> CVATrade; two phases)

### Gate 7 note (SA-CVA) — integrated (supersedes the earlier "not applicable")
The earlier rationale (module absent; no per-instrument hook) is **obsolete**:
`quantark/sacva` is now merged to `main`, and the SA-CVA exposure engine does in
fact derive sensitivities from instruments — it reprices each `CVATrade` over
simulated paths to a discounted EPE profile, then bumps risk factors. So the seam
is the `CVATrade` (`product` + `engine`), exactly like VaR/SIMM/SA-CCR.

The hard part is that the SA-CVA MC exposure engine reprices each trade as a
**Markovian value surface** `engine.price(product, as_of_env(base, spot, t))` — it
hands a single spot at a future node, not the path. The realized-cashflow
`TotalReturnSwapEngine` is path-reading (it reads the whole daily series and
raises on missing pivots), so it cannot serve this directly.

**Phase A — exact, single-period TRS** (`engine/cashflow/trs_cva_repricer.py`):
the realized PV decomposes as `accrual_interest_cum + float_interest +
cash_div_accrual`, and only `float_interest = float_dir·q·(S−S0)` depends on the
valuation-date spot. So `V(S,t) = baseline(t) + float_dir·q·(S−S0)`, where
`baseline(t)` is the realized engine's PV on a **flat-S0 path** (the float term is
identically zero there, so this is the genuine spot-independent
accrual+dividends, computed with the engine's own conventions — not an
approximation). Validated `== TotalReturnSwapEngine` to 1e-4.
`EquitySwapPosition.to_cva_trade -> CVATrade(product=TRSCVAProduct,
engine=TRSCVARepricer)`; the baseline is cached by as-of date (invariant to
spot/vol/IR bumps). Guards (raise, never approximate): matured (judged on the
supplied env), forward-starting, dual-currency, and the path-dependent variants
below.

**Phase B — path-dependent, market-value financing** (opt-in APPROXIMATION):
under `MARKET_VALUE` / `LAST_MARKET_VALUE` accrual the financing leg accrues on
the daily market value, so the future financing at `t` is `fix_dir·rate·q·∫_today^t
S du` — path-dependent. `TRSPathwiseCVARepricer.value_paths` values the whole
path array: `V = V_today + float_dir·q·(S_t−S_today) + fix_dir·rate·q·∫S du`, with
`V_today` the realized MtM (no double-counting). The integral is a **coarse-grid
trapezoid** — an explicit, opt-in approximation (NOT byte-exact vs the engine's
daily left-endpoint accrual; exactness needs a daily exposure grid). Reached only
via `to_cva_trade(allow_approx_financing=True)` + `TRSPathwiseExposureEngine`
(a thin `MonteCarloExposureEngine` subclass overriding only the per-trade value
step). The pathwise repricer's Markovian `price()` raises, so a market-value TRS
on the plain engine fails loudly rather than silently dropping the path
dependence. Values are zeroed past maturity (a matured TRS netted on a longer
grid stops contributing). Capital path (netting/discounting/SBA) is unchanged.

**Still deferred (raise, not approximate):** intermediate redemptions, share
dividends/splits and future cash dividends (quantity / dividend-schedule
path-dependence), and dual-currency (the FX conversion is a second risk factor).

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
