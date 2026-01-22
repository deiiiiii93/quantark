## 1. MCParams extensions
- [x] 1.1 Add `rqmc_paths_mode` to MCParams (per_batch|total).
- [x] 1.2 Add helper to resolve per-batch paths from total count.
- [x] 1.3 Add validation for the new mode.

## 2. Engine integration
- [x] 2.1 Update RQMC engines to use resolved per-batch paths.
- [x] 2.2 Ensure Sobol-ideal per-batch sizes are used in total mode.

## 3. Tests and docs
- [x] 3.1 Unit tests for path mode resolution.
- [x] 3.2 Update docs/README with guidance.
