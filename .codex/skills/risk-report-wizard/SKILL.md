---
name: risk-report-wizard
description: Guide QuantArk users through a structured intake wizard before generating professional risk reports. Use for single-product or comparison risk reports, risk-report specifications, Greeks, sensitivities, scenario ladders, stress tests, VaR/tail risk, event-risk reports, cashflow attribution, hedge diagnostics, and requests for text, charts, tables, raw data, Markdown, HTML, PDF, or DOCX report assets.
---

# Risk Report Wizard

## Purpose

Use this skill as the front door for QuantArk risk reports. First gather a complete report specification, then produce a Risk Report Brief for confirmation, then hand off generation to the appropriate QuantArk reporting skill.

Do not generate report assets until the user confirms the brief.

## Workflow

1. Load `references/report-intake.md` for the question bank and brief template.
2. Read the user's request and infer anything already specified.
3. Ask only for missing high-impact inputs, grouped into small batches.
   - Use `request_user_input` when available.
   - If that tool is unavailable, ask concise plain-text questions.
4. Explicitly fill all required report fields, either from user answers or documented defaults.
5. Produce a Risk Report Brief with assumptions, defaults, handoff skill, expected outputs, and acceptance criteria.
6. After confirmation, delegate report generation:
   - Use `autocallable-risk-report` for Snowball, Phoenix, autocallables, event probabilities, cashflow attribution, dividend/basis risk, or barrier-heavy autocallable surfaces.
   - Use `risk-metric-analyzer` for general QuantArk products, Greeks, sensitivities, scenario ladders, and standard risk metric reports.

## Required Intake

Always collect or explicitly default:

- Purpose: single-product report or comparison report.
- Products: exact product names/classes, asset class, position direction, quantity, notional, or contract multiplier.
- Contract terms: user-provided or auto-decided.
- Pricing engine: user-provided or auto-selected.
- Pricing parameters: user-provided or auto-decided.
- Output assets: text sections, charts, tables, raw data, Markdown, HTML, PDF, or DOCX.

Also capture professional report context:

- Audience and decision use: trader, risk manager, model validation, client, management, or research.
- Valuation date, report horizon, currency, and scaling convention.
- Risk scope: PV, Greeks, DV01/duration, scenario ladders, stress tests, VaR/tail risk, event probabilities, cashflow attribution, and hedge diagnostics.
- Market data source and assumptions: spot, volatility, rates, dividends, basis, curves, and historical series.
- Comparison design: product-vs-product, scenario-vs-scenario, engine-vs-engine, parameter-vs-parameter, and normalization basis.
- Accuracy/performance preferences: analytical, PDE, Monte Carlo, MC paths/seed, PDE grid, and fallback or cross-check engine.

## Defaults

Use these defaults only when the user asks for auto-decided inputs or leaves non-critical fields unspecified:

- Purpose: single-product report.
- Audience: internal risk manager.
- Valuation date: today.
- Currency: USD unless the product or market context implies otherwise.
- Scaling: per-unit plus total position exposure.
- Quantity: 1.
- Contract terms: ATM strike where applicable, 1Y maturity for generic demos, product-class defaults for optional terms.
- Pricing parameters: spot 100, volatility 20%, risk-free rate 5%, dividend yield 2%.
- Engine: auto-select from existing QuantArk product-engine conventions, preferring analytical, then PDE, then Monte Carlo unless the product-specific skill recommends otherwise.
- Output: Markdown report with tables and charts, plus raw CSV data when calculations produce grids.

If an existing product-specific skill defines stronger defaults, prefer that skill's defaults and disclose them in the brief.

## Stop Conditions

Stop after the Risk Report Brief and ask for implementation/product work when:

- The required product class does not exist.
- The requested pricing engine does not exist.
- The requested metric cannot be computed by available QuantArk code.
- User-supplied terms are internally inconsistent.
- Required market data is missing and no reasonable auto default is allowed.

Do not invent pricing behavior inside the wizard.

## Brief Standard

The Risk Report Brief must be specific enough for another agent to generate the report without making new product, engine, data, output, or validation decisions. Use the template in `references/report-intake.md`.
