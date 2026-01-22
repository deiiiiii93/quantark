# Proposal: RQMC Total-Paths Mode for Clean MC Interface

## Summary
Add a safe, opt-in mode for RQMC where `MCParams.num_paths` represents the
expected total path count (not per batch). The engine will compute a Sobol-ideal
per-batch path count automatically.

## Why
Batch users expect `num_paths` to represent total work across a run. RQMC currently
interprets `num_paths` as per-batch, which can produce unexpectedly large totals.
An explicit, safe mode keeps the interface clean without breaking existing users.

## What Changes
- Add `rqmc_paths_mode` to MCParams with `per_batch` default.
- Resolve per-batch paths from total using Sobol-ideal sizing.
- Update RQMC engines to use resolved per-batch paths.

## Scope
In scope:
- Add `rqmc_paths_mode` to `MCParams`: `per_batch` (default) or `total`.
- Compute per-batch paths as `next_power_of_two(ceil(num_paths / rqmc_max_batches))`
  when `rqmc_paths_mode=total`.
- Update all RQMC-enabled MC engines to use the resolved per-batch path count.
- Add tests and short documentation.

Out of scope:
- Changing default behavior for existing users.
- Changing Sobol generation semantics.

## Impact
Positive:
- Predictable total path counts for batch pricing.
- Cleaner interface without ambiguity.

Risks:
- Minor user confusion about total vs per-batch semantics. Mitigated via docs.

## Rollout
Default remains `per_batch` for backward compatibility. Users opt in by setting
`rqmc_paths_mode="total"`.
