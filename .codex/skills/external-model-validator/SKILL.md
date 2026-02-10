---
name: external-model-validator
description: Validate pricing models against external benchmark cases from .docx/.txt/.md inputs. Use when a user provides a case document with product setup, observation rules, and expected outputs (price, Greeks, probabilities) and asks to reproduce results, compare engines, investigate gaps, or generate an external validation report.
---

# External Model Validation

Run a repeatable workflow for validating local model outputs against external case documents.

## Quick Start

1. Extract benchmark targets from external input:
```bash
python .codex/skills/external-model-validator/scripts/extract_external_targets.py \
  --input external/ki_variant_phx/phoenix_example.txt \
  --output external/ki_variant_phx/expected_targets.json
```
2. Run or update the local pricing/validation script to generate actual outputs.
3. Compare expected vs actual by case and engine.
4. Write findings with a clear assumption log and mismatch diagnostics.

## Workflow

### 1) Ingest External Case File

Accept `.docx`, `.txt`, or `.md`.

- Use `scripts/extract_external_targets.py` first for automatic extraction.
- If extraction misses fields, read the source directly and patch assumptions manually.
- Normalize signs and units before any comparison:
  - Holder-side vs issuer-side sign convention.
  - Percent vs decimal barriers/rates.
  - Observation-calendar conventions.

### 2) Build Validation Inputs

Construct a machine-readable case spec:
- Product parameters.
- Monitoring conventions.
- Engine settings (paths, grids, steps, seeds).
- External benchmark values by case and engine.

Keep each assumption explicit and traceable to source lines.

### 3) Run Engines

Run all requested engines under harmonized assumptions:
- Keep `PricingEnvironment` and calendar conventions aligned.
- For stochastic engines, report both price and uncertainty.
- For deterministic engines, report runtime and configuration.

### 4) Compare and Diagnose

Compare by case and engine:
- Absolute difference.
- Relative difference.
- In-sigma diagnostic for MC-family methods.

When gaps are material, isolate causes in this order:
1. Sign convention and payoff perspective.
2. Observation schedule/day-count details.
3. Coupon accrual interpretation.
4. KI/KO state transition semantics.
5. Numerical discretization and boundary settings.

### 5) Produce Validation Report

Use `references/validation_report_template.md` and fill:
- Case summary.
- Assumptions applied.
- Comparison table.
- Root-cause analysis for mismatches.
- Final pass/fail status and residual risks.

## Resource Notes

- Script: `scripts/extract_external_targets.py`
  - Parse case blocks and benchmark targets from `.docx/.txt/.md`.
  - Emit normalized JSON for downstream comparison scripts.
- Template: `references/validation_report_template.md`
  - Keep report structure consistent across validations.
