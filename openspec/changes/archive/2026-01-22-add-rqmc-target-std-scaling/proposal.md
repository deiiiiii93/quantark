# Proposal: RQMC Target Std Scaling by Notional/Relative Error

## Summary
Add batch-friendly RQMC controls to MCParams, including target standard error
scaling by notional or relative error, to avoid overly strict absolute tolerances
for large notionals and reduce excessive RQMC batching runtimes.

## Why
Current RQMC uses a fixed absolute `target_std` (default 1e-4). For large
notionals, this can require millions of paths and long runtimes, even when a
relative error tolerance would be appropriate. Users need an easy way to scale
`target_std` by notional or relative price targets.

## What Changes
- Add RQMC batching fields to MCParams.
- Add target std scaling modes and helper resolution.
- Update RQMC engines to use resolved target std.

## Scope
In scope:
- Add MCParams fields for RQMC batching (`rqmc_max_batches`, `rqmc_min_batches`)
  and target standard error (`rqmc_target_std`, `rqmc_target_std_mode`, floor).
- Add helper to resolve target std based on notional or relative error.
- Use the helper in RQMC MC engines (Snowball, Phoenix, Barrier, Euro, Asian,
  Digital, American).

Out of scope:
- Changing the default behavior when users do not opt in.
- Rewriting the RQMC batching logic.

## Impact
Positive:
- Predictable runtimes for batch pricing.
- Simple control: absolute vs relative target std.

Risks:
- Misuse of relative tolerances if users misunderstand scale. Mitigated by
  clear docs and safe defaults.

## Rollout
Defaults remain absolute to preserve current behavior. Users can opt in via
`rqmc_target_std_mode`.
