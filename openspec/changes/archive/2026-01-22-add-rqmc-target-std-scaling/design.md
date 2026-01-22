# Design: RQMC Target Std Scaling

## New MCParams Fields
- `rqmc_target_std` (float, default 1e-4): base target standard error.
- `rqmc_target_std_mode` (str): `absolute|relative_notional|relative_price`.
- `rqmc_target_std_floor` (float): minimum absolute std error.
- `rqmc_min_batches` (int, default 4), `rqmc_max_batches` (int, default 32).

## Helper Logic
`MCParams.resolve_rqmc_target_std(product, pricing_env, scale=None)`:
- `absolute`: return `rqmc_target_std`
- `relative_notional`: `rqmc_target_std * notional_scale`
- `relative_price`: `rqmc_target_std * price_scale`
- apply floor via `max(target, rqmc_target_std_floor)`

Scale inference (if not provided):
- Prefer `product.contract_multiplier` or `product.notional`.
- Fall back to 1.0.

## Integration
Use the helper in all RQMC-capable MC engines before calling `run_rqmc`.
Defaults keep current behavior.

