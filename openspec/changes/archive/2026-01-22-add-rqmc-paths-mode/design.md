# Design: RQMC Paths Mode

## New MCParams Field
- `rqmc_paths_mode`: `per_batch` (default) or `total`.

## Resolution Logic
If `rqmc_paths_mode == "total"`:
- `total = num_paths`
- `per_batch_raw = ceil(total / rqmc_max_batches)`
- `per_batch = next_power_of_two(per_batch_raw)`

If `rqmc_paths_mode == "per_batch"`:
- `per_batch = num_paths`

## Integration
RQMC engines pass the resolved per-batch count to their path generators.

## Compatibility
Default `per_batch` preserves existing behavior. Only opt-in users get the
total-paths semantics.

