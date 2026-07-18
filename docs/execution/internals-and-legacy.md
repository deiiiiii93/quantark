# QuantArk execution framework — internals and legacy duplicates (spec §17.3)

Phase 6 status of the "duplicate batch, Dask, and preparation scaffolding"
required by the spec to be removed **only after both routes use the same
internal plan and reducer**. There are **no v1 deprecation warnings**; any
future removal of a legacy route requires a separate spec and release cycle
(spec §17.3).

## Unified (single implementation today)

| Concern | Single home | How both routes share it |
|---|---|---|
| RQMC stopping loop | `quantark.montecarlo.qmc_rqmc_driver.run_rqmc_traced` | `run_rqmc` delegates; the session adaptive adapter drives the SAME live loop (Phase 3). `quantark.asset.equity.process.bsm.qmc_rqmc_driver` is a re-export shim, not a copy. |
| Legacy autocallable Dask batch loop | `quantark.asset.equity.engine.mc.autocallable_dask_batch` | Snowball (vanilla + KO-reset) and Phoenix `_price_parallel` all call `run_autocallable_dask_batches`; bitwise-gated by `test/execution/test_legacy_dask_goldens.py`. **Intra-legacy consolidation, not kernel convergence** — the legacy Dask route itself remains. |
| Dupire session preparation | `quantark.execution.prep.dupire.dupire_surface_state` | Equity DCN, equity LV PDE, and FX LV PDE prepared adapters all import the one helper; direct engines keep their own legacy builds by design (§17.1). |
| Fingerprints | `quantark.execution.cache.fingerprint` | Only fingerprint utility in the codebase. |
| Batch reduction arithmetic (DCN) | `_LegAccumulator` / `_finalize_dcn_result` | Legacy `price_detailed` and the Phase 2 batch adapters reduce through the same classes and stderr formulas. |

There is no legacy multiprocessing/joblib helper anywhere in engine code; the
process backend (`quantark.execution.backends.processes`) has no legacy
counterpart to dedupe.

## Kept duplicates (TEMPORARY, with removal preconditions)

Each remaining duplicate exists because spec §17.1 preserves the legacy
surface byte-for-byte, and the frozen downstream deliverable
(quant-mini-project, pinned to wheel 0.2.6 / commit `318009e`, subclassing
`SurfaceAwareLVDCNEngine` via the `_prepare_simulation`/`_resolve_surface`
hooks) depends on the direct route staying untouched.

| Duplicate pair | Why it stays | Removal precondition |
|---|---|---|
| DCN legacy thread-batch loop (`DCNMCEngine.price_detailed`) vs Phase 2 batch adapters (`plan_batches`/`execute_batch`/`reduce_batches`) | `QUANTARK_DCN_MC_WORKERS`, `num_workers`, and the direct `price_detailed` path are preserved surfaces; reduction arithmetic is already shared. | Direct route invokes the kernel's serial batch plan/reducer with legacy validation, warnings, and result unwrapping preserved; exact parity suite; separate deprecation spec. |
| Legacy engine `use_dask` route (shared reducer above) vs session scenario/process/Dask backends | `use_dask` flags and their availability/warning behavior are preserved surfaces; the session Dask backend serves typed plans, not `price()` calls. | Same as above: a direct compatibility facade over kernel internals with byte-preserved behavior, then a separate deprecation spec. |
| Process-global legacy QMC draw cache (`QMCDrawCache`, `QUANTARK_QMC_CACHE_MB`) vs session `DrawRepository` | Two cache SCOPES by design: the session repository charges bytes to the session budget and never double-charges the legacy cache; both wrap the same `SobolNormalGenerator`. | Not a removal target — scope separation is intentional (spec §11). |
| Legacy QMC cache LRU vs `backends/admission.py` budget admission | Different layers (per-cache eviction vs task admission); not duplicates in function. | Not a removal target. |

## Known availability fix (Phase 6)

`PhoenixMCEngine` imported `from dask.compute import compute`, which fails on
modern dask (no such module), silently disabling its parallel path while
dask was installed. Fixed to `from dask import compute, delayed`; a
regression test asserts `DASK_AVAILABLE` is true whenever dask imports.
