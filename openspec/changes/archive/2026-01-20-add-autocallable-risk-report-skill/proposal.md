# Change: Add autocallable risk profile report skill (Snowball-first)

## Why
Autocallable products (Snowball/Phoenix) require a consistent risk profile report that combines event probabilities,
cashflow attribution, and multi-factor risk surfaces. In China index markets, dividend/carry (basis) risk is often
the dominant hedging dimension, so the report must prioritize dividend sensitivities and related cross terms.

## What Changes
- Add a Snowball-first risk profile report generator that produces a Markdown report with plots for PV, Greeks, and event risk.
- Add an `AutocallablePathAnalyzer` (Monte Carlo) to compute:
  - risk-neutral (pricing-measure) event probabilities and expected discounted cashflows
  - historical (real-world) replay statistics and parametric scenario distributions using spot and dividend yield series
- Add dividend/basis-focused surfaces:
  - `dividend_rho` surfaces vs `Spot×Vol` and `Spot×Dividend`
  - `Delta(Spot, Dividend)` and mixed partial `∂²V/(∂S∂q)` for spot–dividend cross exposure
- Add an engine-level API proposal for event stats / cashflow decomposition so QUAD/PDE engines can (optionally) provide
  per-observation probabilities faster than MC.
- Package a Codex skill that scaffolds and runs the report generation workflow.

## Default Report Grid Assumptions
- Single underlying only (initial scope).
- Dividend yield `q` uses a flat bump model (parallel shift).
- Spot range: `S ∈ [0.60, 1.20] * S0`
- Dividend range: `q ∈ [q0 - 500bp, q0 + 500bp]`
- Vol range: `σ ∈ σ0 * (1 ± 5%)` with 11 nodes

## Impact
- Affected specs: `autocallable-risk-report` (new)
- Affected code (planned):
  - New analysis/report modules under `asset/equity/` (Snowball-first)
  - Optional additions to engine interfaces for event stats
  - New Codex skill packaging (tooling)

## Non-Goals (This Change)
- Basket / worst-of underlyings
- Full Phoenix support (planned follow-up after Snowball)
- Term-structure dividend curve shocks (flat only for MVP)

