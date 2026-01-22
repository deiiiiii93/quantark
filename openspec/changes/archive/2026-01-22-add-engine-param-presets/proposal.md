# Proposal: Add Engine Parameter Presets, Factories, and Config Loaders

## Summary
Introduce a batch-friendly configuration layer for QUAD and PDE engines by adding
named parameter presets, factory helpers, and YAML/JSON config loading. This makes
engine setup approachable for new users while preserving full expert control.

## Why
Current `PDEParams` and `QuadParams` expose many numerical knobs. This is flexible
but overwhelming for new or batch users who need a safe, minimal-configuration
workflow. A small set of presets and factory helpers will reduce onboarding time
and improve consistency across batch pricing.

## What Changes
- Add preset profiles for QUAD/PDE params.
- Add factory helpers to build params from profiles + overrides.
- Add YAML/JSON config loaders for batch workflows.

## Scope
In scope:
- Named presets for QUAD and PDE params (`fast`, `balanced`, `accurate`,
  `barrier_sensitive`, `reverse_sensitive`).
- Factory helpers to construct params from presets with optional overrides.
- YAML/JSON config loaders for batch pipelines.
- Documentation and examples showing recommended usage.

Out of scope:
- Changing core numerical algorithms.
- Removing or deprecating existing parameters.
- Breaking backward compatibility.

## Impact
Positive:
- One-line setup for batch pricing.
- Fewer configuration mistakes for new users.
- Standardized pricing quality tiers.

Risks:
- Preset choices may not fit all products; mitigated by overrides.
- Extra API surface; mitigated by keeping helpers optional.

## Rollout
Add the new helpers without changing default behaviors. Existing code paths remain
unchanged unless users opt into presets or config loading.
