---
name: autocallable-risk-report
description: |
  Generate and validate risk profile analysis reports for autocallable products (Snowball first; Phoenix later),
  including risk-neutral event probabilities, cashflow attribution, and multi-factor risk surfaces with a strong
  China index-futures focus (dividend/basis risk, dividend_rho surfaces, Delta(Spot,Dividend), and mixed partial ∂²V/(∂S∂q)).
  Use when asked to create a Snowball/Phoenix risk profile report, add dividend/basis risk analysis, add event
  probability/cashflow attribution, or implement/report (Spot×Vol, Spot×Dividend) surfaces for autocallables.
---

# Autocallable Risk Report (Snowball-first)

## Quick Start

- Repo-scoped skill runner:
  - `python .codex/skills/autocallable-risk-report/scripts/generate_report.py --input example/snowball_risk_report_input.py --fast`
- User-scoped install runner (if installed into `~/.codex/skills`):
  - `python ~/.codex/skills/autocallable-risk-report/scripts/generate_report.py --input example/snowball_risk_report_input.py --fast`

## Workflow

1) Confirm inputs
- Product: `SnowballOption` (single underlying)
- Market: `spot`, `rate_curve`, `vol_surface`, `div_yield` (flat q for MVP)
- Historical inputs (optional, for historical replay / parametric shocks):
  - `spot_series` and `q_series` (same length, ordered, no gaps preferred)
- Report grids:
  - Spot: `S ∈ [0.60, 1.20] * S0`
  - Dividend: `q ∈ q0 ± 500bp` (flat shift; clipped at 0 lower bound)
  - Vol: `σ ∈ σ0 * (1±5%)` with 11 nodes

2) Choose pricing engine order
- Preferred: QUAD → PDE → MC (Snowball has all three)
- Use MC analyzer for event probabilities + cashflow attribution even if PV is from QUAD/PDE.

3) Generate report
- Module CLI: `python -m asset.equity.report.autocallable_risk_report --help`
- Bundled runner:
  - `python .codex/skills/autocallable-risk-report/scripts/generate_report.py --help`

4) **Human Review & Interpretation**
- Open the generated `risk_report.md`.
- **Add Analysis:** Based on the Greeks and surfaces, add 1-2 sentences of professional interpretation to the "Risk Interpretation" placeholders.
- **Verify Logic:** Ensure the Scenario Ladder's "Worst Case" aligns with the product's Gamma/Vega profile (e.g., short Gamma Snowballs should lose most in large moves).

## Report Content Checklist

- Executive dashboard with "Traffic Light" status and risk interpretation.
- Terms and schedules table + Market Snapshot.
- Local Greeks (incl. `dividend_rho` and `basis_rho=-dividend_rho`).
- Barrier zoom analysis (Gamma/Vega plots near KI/KO levels).
- Risk surfaces with Skew/Smile and Vanna/Volga analysis.
- Scenario Ladder (Spot × Vol PnL matrix) with "Worst Case" analysis.
- Risk-neutral event stats and conditional cashflow projection.
- Historical replay and parametric shock PnL distributions (using spot/q series).

## Reference
- See `.codex/skills/autocallable-risk-report/references/report-outline.md`

