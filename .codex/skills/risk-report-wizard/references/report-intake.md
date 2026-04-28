# Risk Report Intake

Use this reference to ask the minimum useful questions before creating a QuantArk risk report. Infer answers already present in the request and ask only for missing high-impact information.

## Intake Order

1. Report objective
2. Products and positions
3. Contract terms
4. Pricing engine and model controls
5. Pricing parameters and market data
6. Risk scope and comparison design
7. Output fields and assets
8. Risk Report Brief confirmation

## Question Bank

### 1. Report Objective

Ask:

- Is this a single-product report or a comparison report?
- Who is the audience: trader, risk manager, model validator, client, management, or research?
- What decision should the report support: pricing sign-off, hedging, limit monitoring, scenario review, model comparison, client explanation, or product onboarding?
- What is the valuation date and report horizon?

Default:

- Single-product report for an internal risk manager, valuation date today, horizon to maturity plus near-term scenario views.

### 2. Products and Positions

Ask:

- Which exact QuantArk product class or product type should be reported?
- What is the asset class: equity, bond, rate, OTC structured product, or portfolio?
- Is the position long or short?
- What is the quantity, notional, denominator, or contract multiplier?
- For portfolios, should exposures be reported per line, aggregated, or both?

Default:

- Quantity 1, long position, per-unit plus total exposure.

### 3. Contract Terms

Ask:

- Should contract terms be provided by the user or auto-decided?
- What are the core terms: strike, maturity or exercise date, option type, barriers, coupons, observation schedule, notional, and settlement convention?
- For barriers or autocallables, what are KI/KO levels, observation dates, coupon rules, knock-in observation style, and redemption logic?
- For rates or bonds, what are coupon, day count, schedule, curve reference, payment frequency, and amortization details?

Default:

- ATM strike where applicable, 1Y maturity for generic demos, product-class defaults for optional fields.

### 4. Pricing Engine

Ask:

- Should the pricing engine be user-provided or auto-selected?
- If user-provided, what engine and method should be used?
- Should a second engine be used as a cross-check?
- Are there speed or accuracy constraints?

Default engine order:

- Analytical when available for vanilla or closed-form products.
- PDE for early-exercise, barrier, or grid-sensitive products.
- Monte Carlo for path-dependent or highly structured products.
- For Snowball/Phoenix/autocallables, follow `autocallable-risk-report` guidance.

Ask for numerical controls when relevant:

- MC paths, random seed, quasi-MC preference, antithetic variates.
- PDE time steps, spatial grid, grid refinement near strikes or barriers.
- Finite-difference bump sizes for Greeks.

Default controls:

- Fixed MC seed for repeatable Greeks.
- Product-skill defaults for paths, grids, and bump sizes.

### 5. Pricing Parameters and Market Data

Ask:

- Should pricing parameters be user-provided or auto-decided?
- What are spot, volatility surface or flat volatility, rate curve, dividend yield, basis, credit spread, and FX assumptions?
- What market data source should be cited?
- Are historical series needed for replay, VaR, or stress calibration?
- Should the report include stale/missing data warnings?

Default:

- Spot 100, volatility 20%, risk-free rate 5%, dividend yield 2%, valuation date today.
- State all auto-decided market assumptions in the brief and report.

### 6. Risk Scope

Ask which sections to include:

- PV and decomposition.
- Local Greeks: Delta, Gamma, Vega, Theta, Rho, dividend rho, basis rho, DV01, duration, convexity.
- Higher-order or cross Greeks: Vanna, Volga, Charm, mixed spot-dividend sensitivity.
- Scenario ladders: spot, vol, rate, dividend, basis, curve, and combined shocks.
- Stress tests: named historical scenarios, parametric shocks, barrier proximity shocks, worst-case grid.
- VaR and tail risk: historical, parametric, Monte Carlo, confidence level, horizon.
- Event risk: KO probability, KI probability, survival probability, cashflow attribution.
- Hedge diagnostics: delta hedge, gamma/vega hedge, rebalance sensitivity, hedge cost.
- Model validation: engine comparison, convergence, benchmark checks, data and assumption warnings.

Default:

- Include PV, core Greeks, scenario ladder, risk interpretation, and charts.
- Add event stats and cashflow attribution for autocallables.
- Add DV01/duration/convexity for fixed-income products.

### 7. Comparison Design

For comparison reports, ask:

- What is being compared: product-vs-product, scenario-vs-scenario, engine-vs-engine, parameter-vs-parameter, or date-vs-date?
- What should be held constant across cases?
- What is the normalization basis: same notional, same premium, same delta, same maturity, same underlying, or user-defined?
- Should output show absolute values, differences, ratios, rankings, or all?

Default:

- Same notional normalization, base-case product first, show absolute values and deltas from base.

### 8. Output Fields and Assets

Ask:

- Which text sections are required: executive summary, product terms, market snapshot, methodology, risk interpretation, limitations, appendix?
- Which tables are required: terms, market data, PV, Greeks, scenario matrix, stress results, event probabilities, cashflows, raw assumptions?
- Which charts are required: Greeks vs spot/time, risk surfaces, scenario heatmaps, PnL distribution, barrier zoom, cashflow profile?
- Which output formats are required: Markdown, HTML, PDF, DOCX, CSV, PNG, or ZIP bundle?
- Is the report language, style, or recipient branding constrained?

Default:

- Markdown report with executive summary, terms, market snapshot, PV/Greeks, scenario ladder, charts, methodology, warnings, and raw CSV data when grids are generated.

## Risk Report Brief Template

Use this exact structure before generating report assets:

```markdown
# Risk Report Brief

## Objective
- Purpose:
- Audience:
- Decision use:
- Valuation date:
- Report horizon:

## Products and Positions
- Product(s):
- Asset class:
- Position direction:
- Quantity/notional/multiplier:
- Scaling convention:

## Contract Terms
- Source: user-provided | auto-decided
- Terms:
- Defaults applied:
- Open issues:

## Pricing Setup
- Engine selection: user-provided | auto-selected
- Primary engine:
- Cross-check engine:
- Numerical controls:
- Pricing parameters source: user-provided | auto-decided
- Market assumptions:
- Market data source:

## Risk Scope
- Metrics:
- Scenarios/stresses:
- Event/cashflow analysis:
- Hedge diagnostics:
- Comparison design:

## Outputs
- Text sections:
- Tables:
- Charts:
- Files/assets:
- Output directory:

## Handoff
- Generation skill:
- Reason:
- Expected files:

## Acceptance Criteria
- Required calculations complete:
- Required charts/tables present:
- Assumptions disclosed:
- Warnings included:
- Comparison normalized correctly:
```

After presenting the brief, ask the user to confirm or revise it. Do not proceed to report generation before confirmation.
