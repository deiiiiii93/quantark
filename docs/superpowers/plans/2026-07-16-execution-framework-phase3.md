# Execution Framework Phase 3 (Trimmed): Adaptive RQMC Compatibility + Heston DCN Batch Adapters — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the 12 Snowball/Phoenix autocallable MC engines onto session-planned adaptive RQMC (sequential compatibility mode, bit-identical stopping sequence) and the 3 Heston DCN MC engines onto the Phase-2 fixed-batch path, with deterministic checkpoint tests and declared validation profiles on every adaptive/batch-capable inventory row.

**Architecture:** Three workstreams. (A) The shared `run_rqmc` driver is refactored into `run_rqmc_traced` — one implementation of the Welford batch-means stopping loop that also emits a checkpoint trace — and the Snowball/Phoenix engines expose their RQMC runs as `RQMCRunSpec` seams; a new `AutocallableAdaptiveMCAdapter` builds an immutable `AdaptivePlan`, drives the SAME traced driver on the LIVE engine instance (sequential, no cloning — subclass-safe by construction), and the kernel records the plan fingerprint plus checkpoint trace. (B) The Phase-2 `_DCNBatchMixin` gains plan-metadata hooks and the Heston DCN engines gain a `_uniform_provider` hook mirroring `_draw_provider`; `DrawRepository` gains `uniforms_handle`; three exact-registered Heston batch adapters (Heston, QE, CoupledCoarse) produce bit-identical batch results. (C) The inventory gains `adaptive_state`/`adaptive_rationale`/`validation_profile` fields with CI gates.

**Tech Stack:** Python 3.11+, NumPy/SciPy, pytest (existing `test/execution/` conventions and `matrix_fixtures.py`).

**Evidence-based scope note (spec §21 re-scope):** Parallel-wave adaptive RQMC (spec §8.4) is DEFERRED on Phase 2 benchmark evidence — its speedup channel is thread scaling, which is host-limited (GIL) on the development machine; the deferral is recorded in the inventory rationale. Single-solve equity MC and FX MC engines keep their honest `temporary_legacy` states.

## Global Constraints

- Run all tests from the worktree with `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest` (editable install would otherwise resolve `quantark` to the main repo).
- Canonical `quantark.*` imports only; no new flat imports.
- `quantark/execution/` modules must not statically import asset code (lazy factory imports only, as in `registry.py`).
- Adaptive compatibility mode is SEQUENTIAL by contract (spec §8.4): it evaluates its stopping criterion after the same batches in the same order as `run_rqmc`. Session results must be **bit-identical** to direct calls (price, std_error, batches_used, event probabilities).
- The direct legacy call path must remain bit-identical after every refactor (pure extraction only; existing test modules are the regression net).
- Use `quantark.util.numerical` helpers where float comparison is needed in library code; tests asserting bit-identity use `==` on purpose.
- Existing reviewed decisions stand: PV-only legacy-adapter guarantee, engine-internal parallelism passthrough, `exact=True` for adapters that reconstruct engines from a fixed constructor signature.
- Commits end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. Nothing is pushed to origin.
- Known pre-existing failure `test_snowball_quad_flat_identity_golden` is out of scope.

## File Structure

| File | Responsibility |
|---|---|
| `quantark/montecarlo/qmc_rqmc_driver.py` (modify) | `RQMCCheckpoint`, `RQMCRunSpec`, `run_rqmc_traced`; `run_rqmc` becomes a thin wrapper |
| `quantark/montecarlo/__init__.py` (modify) | export the new names |
| `quantark/asset/equity/engine/mc/snowball_mc_engine.py` (modify) | `_rqmc_spec`, `_ko_reset_rqmc_spec`, `build_rqmc_session_spec`, `_complete_price` seams |
| `quantark/asset/equity/engine/mc/phoenix_mc_engine.py` (modify) | same seams for Phoenix |
| `quantark/execution/contracts.py` (modify) | `AdaptivePlan` |
| `quantark/asset/equity/engine/mc/autocallable_execution_adapters.py` (create) | `AutocallableAdaptiveMCAdapter` |
| `quantark/execution/kernel.py` (modify) | adaptive dispatch branch, trace records |
| `quantark/execution/registry.py` (modify) | snowball/phoenix registrations (non-exact); Heston DCN exact swaps |
| `quantark/execution/cache/draws.py` (modify) | `uniforms_handle` + descriptor `distribution` |
| `quantark/asset/equity/engine/mc/dcn_vol_mc_engines.py` (modify) | `_uniform_provider` hook in `_heston_draws` |
| `quantark/asset/equity/engine/mc/dcn_execution_adapters.py` (modify) | mixin plan hooks; `_UniformDrawProvider`; 3 Heston batch adapters |
| `quantark/execution/inventory.py` (modify) | `adaptive_state`/`adaptive_rationale`/`validation_profile` + row updates |
| `quantark/execution/__init__.py` (modify) | export `AdaptivePlan`; docstring "Phases 0-3" |
| `test/execution/test_rqmc_driver_trace.py` (create) | driver trace/bit-identity unit tests |
| `test/execution/test_adaptive_adapter.py` (create) | adaptive session bit-identity + checkpoint determinism |
| `test/execution/test_adaptive_vol_matrix.py` (create) | 12-engine bitwise matrix |
| `test/execution/test_heston_dcn_batch.py` (create) | Heston/QE/Coupled batch bit-identity |
| `test/execution/test_draw_repository.py` (modify) | uniforms tests |
| `test/execution/test_inventory.py` (modify) | adaptive gates |
| `test/execution/test_registry.py` (modify) | expected registration set |

---

### Task 1: Traced RQMC driver (single-implementation stopping loop)

**Files:**
- Modify: `quantark/montecarlo/qmc_rqmc_driver.py`
- Modify: `quantark/montecarlo/__init__.py`
- Test: `test/execution/test_rqmc_driver_trace.py`

**Interfaces:**
- Produces: `RQMCCheckpoint(batch_index:int, batch_mean:float, running_mean:float, std_error:float|None, stopped:bool)` (frozen dataclass); `RQMCRunSpec(pricer_fn, path_generator, max_batches:int, min_batches:int, target_std:float, paths_per_batch:int, time_steps:int, scheme:str, finalize:Callable[[RQMCResult], object])` (frozen dataclass); `run_rqmc_traced(pricer_fn, path_generator, max_batches, target_std, min_batches=1) -> tuple[RQMCResult, tuple[RQMCCheckpoint, ...]]`. `run_rqmc(...)` keeps its exact signature and returns `run_rqmc_traced(...)[0]`.

- [ ] **Step 1: Write the failing tests**

```python
# test/execution/test_rqmc_driver_trace.py
"""Traced RQMC driver: one implementation of the Welford stopping loop.

Bit-identity is proven against an inline verbatim copy of the PRE-REFACTOR
run_rqmc loop (reference implementation pinned in this test module).
"""
import numpy as np
import pytest

from quantark.montecarlo.qmc_rqmc_driver import (
    RQMCCheckpoint,
    RQMCResult,
    run_rqmc,
    run_rqmc_traced,
)


class _FakeGenerator:
    """Deterministic per-batch payoff seeds keyed by batch_id."""

    def __init__(self, num_paths):
        self.num_paths = num_paths

    def generate_paths(self, seed=None, batch_id=None, return_aux=False):
        rng = np.random.default_rng(1000 + int(batch_id or 0))
        paths = rng.standard_normal((self.num_paths, 3))
        aux = {"batch_id": 0 if batch_id is None else int(batch_id)}
        return paths, aux if return_aux else None


def _pricer(paths, aux):
    return 10.0 + paths[:, -1]


def _reference_run_rqmc(pricer_fn, path_generator, max_batches, target_std,
                        min_batches=1):
    """Verbatim pre-refactor loop (bit-identity oracle)."""
    batch_means = []
    n_paths_per_batch = path_generator.num_paths
    mean = 0.0
    m2 = 0.0
    for batch_id in range(max_batches):
        paths, aux = path_generator.generate_paths(
            batch_id=batch_id, return_aux=True
        )
        payoffs = np.asarray(pricer_fn(paths, aux), dtype=float)
        batch_mean = float(payoffs.mean())
        batch_means.append(batch_mean)
        n = batch_id + 1
        delta = batch_mean - mean
        mean += delta / n
        m2 += delta * (batch_mean - mean)
        if n >= min_batches:
            variance = m2 / (n - 1) if n > 1 else 0.0
            std_error = np.sqrt(variance / n)
            if std_error <= target_std or n == max_batches:
                return RQMCResult(
                    price=mean, std_error=std_error,
                    total_paths=n * n_paths_per_batch, batches_used=n,
                    batch_means=np.array(batch_means, dtype=float),
                )
    raise RuntimeError("unreachable")


@pytest.mark.parametrize("target_std,min_batches,max_batches", [
    (1e-6, 4, 12),   # runs to max
    (1.0, 4, 12),    # stops at min_batches
    (0.02, 2, 32),   # stops mid-run
    (1.0, 1, 1),     # single batch, variance 0.0 branch
])
def test_traced_bitwise_matches_reference(target_std, min_batches, max_batches):
    gen = _FakeGenerator(512)
    ref = _reference_run_rqmc(_pricer, gen, max_batches, target_std, min_batches)
    result, trace = run_rqmc_traced(
        _pricer, gen, max_batches, target_std, min_batches
    )
    assert result.price == ref.price
    assert result.std_error == ref.std_error
    assert result.total_paths == ref.total_paths
    assert result.batches_used == ref.batches_used
    assert np.array_equal(result.batch_means, ref.batch_means)
    # wrapper equivalence
    wrapped = run_rqmc(_pricer, gen, max_batches, target_std, min_batches)
    assert wrapped.price == result.price
    assert wrapped.std_error == result.std_error


def test_trace_shape_and_stop_flags():
    gen = _FakeGenerator(512)
    result, trace = run_rqmc_traced(_pricer, gen, 32, 0.02, 2)
    assert len(trace) == result.batches_used
    assert all(isinstance(c, RQMCCheckpoint) for c in trace)
    assert [c.batch_index for c in trace] == list(range(len(trace)))
    assert all(not c.stopped for c in trace[:-1])
    assert trace[-1].stopped
    # std_error is None strictly before min_batches
    assert all(c.std_error is None for c in trace[: 2 - 1])
    assert all(c.std_error is not None for c in trace[2 - 1:])
    # running mean at the stop checkpoint IS the result price
    assert trace[-1].running_mean == result.price
    assert trace[-1].std_error == result.std_error


def test_trace_deterministic_across_runs():
    gen = _FakeGenerator(512)
    a = run_rqmc_traced(_pricer, gen, 32, 0.02, 2)
    b = run_rqmc_traced(_pricer, gen, 32, 0.02, 2)
    assert a[0] == b[0] or (
        a[0].price == b[0].price and a[0].std_error == b[0].std_error
        and a[0].batches_used == b[0].batches_used
    )
    assert a[1] == b[1]


def test_validation_errors_preserved():
    gen = _FakeGenerator(8)
    with pytest.raises(ValueError):
        run_rqmc_traced(_pricer, gen, 0, 1e-4)
    with pytest.raises(ValueError):
        run_rqmc_traced(_pricer, gen, 4, 1e-4, min_batches=0)
    with pytest.raises(ValueError):
        run_rqmc_traced(_pricer, gen, 4, 1e-4, min_batches=8)
    with pytest.raises(ValueError):
        run_rqmc_traced(_pricer, gen, 4, -1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/execution/test_rqmc_driver_trace.py -v`
Expected: FAIL with `ImportError: cannot import name 'RQMCCheckpoint'`

- [ ] **Step 3: Implement the traced driver**

In `quantark/montecarlo/qmc_rqmc_driver.py`, add after `RQMCResult`:

```python
@dataclass(frozen=True)
class RQMCCheckpoint:
    """Post-batch stopping-criterion evaluation record (spec section 8.4).

    ``std_error`` is None strictly before ``min_batches`` (the criterion is
    not evaluated there); ``stopped`` marks the checkpoint at which the run
    terminated (target reached or ``max_batches`` exhausted).
    """

    batch_index: int
    batch_mean: float
    running_mean: float
    std_error: Optional[float]
    stopped: bool


@dataclass(frozen=True)
class RQMCRunSpec:
    """Engine-provided description of one adaptive RQMC run.

    ``finalize`` assembles the engine-native result object from the
    RQMCResult; it is the SAME callable on the direct and session paths, so
    result assembly (including any extra statistics batch) is shared code.
    """

    pricer_fn: PricerFn
    path_generator: object
    max_batches: int
    min_batches: int
    target_std: float
    paths_per_batch: int
    time_steps: int
    scheme: str
    finalize: object  # Callable[[RQMCResult], engine-native result]
    product: object = None  # the priced product (session postamble needs it)
```

Replace the body of `run_rqmc` and add `run_rqmc_traced` (the loop moves VERBATIM — same statements, same order, same float operations; the only additions are the two `checkpoints.append` lines and the tuple return):

```python
def run_rqmc_traced(
    pricer_fn: PricerFn,
    path_generator: PathGenerator,
    max_batches: int,
    target_std: float,
    min_batches: int = 1,
) -> Tuple[RQMCResult, Tuple[RQMCCheckpoint, ...]]:
    """run_rqmc with a per-batch checkpoint trace (spec section 8.4).

    This is THE stopping-loop implementation; ``run_rqmc`` delegates here,
    so direct and session executions share one arithmetic path by
    construction.
    """
    if max_batches <= 0:
        raise ValueError("max_batches must be positive")
    if min_batches <= 0:
        raise ValueError("min_batches must be positive")
    if min_batches > max_batches:
        raise ValueError("min_batches cannot exceed max_batches")
    if target_std <= 0.0:
        raise ValueError("target_std must be positive")

    batch_means = []
    checkpoints = []
    n_paths_per_batch = path_generator.num_paths

    # Welford's algorithm over batch means
    mean = 0.0
    m2 = 0.0

    for batch_id in range(max_batches):
        paths, aux = path_generator.generate_paths(batch_id=batch_id, return_aux=True)
        payoffs = pricer_fn(paths, aux)
        payoffs = np.asarray(payoffs, dtype=float)
        if payoffs.ndim != 1 or payoffs.shape[0] != n_paths_per_batch:
            raise ValueError(
                "pricer_fn must return a 1D array with one payoff per path "
                f"(expected length {n_paths_per_batch}, got {payoffs.shape})."
            )

        batch_mean = float(payoffs.mean())
        batch_means.append(batch_mean)

        n = batch_id + 1
        delta = batch_mean - mean
        mean += delta / n
        m2 += delta * (batch_mean - mean)

        if n >= min_batches:
            if n > 1:
                variance = m2 / (n - 1)
            else:
                variance = 0.0
            std_error = np.sqrt(variance / n)

            stopped = std_error <= target_std or n == max_batches
            checkpoints.append(RQMCCheckpoint(
                batch_index=batch_id, batch_mean=batch_mean,
                running_mean=mean, std_error=float(std_error),
                stopped=stopped,
            ))
            if stopped:
                result = RQMCResult(
                    price=mean,
                    std_error=std_error,
                    total_paths=n * n_paths_per_batch,
                    batches_used=n,
                    batch_means=np.array(batch_means, dtype=float),
                )
                return result, tuple(checkpoints)
        else:
            checkpoints.append(RQMCCheckpoint(
                batch_index=batch_id, batch_mean=batch_mean,
                running_mean=mean, std_error=None, stopped=False,
            ))

    raise RuntimeError("No batches were run in RQMC driver.")


def run_rqmc(
    pricer_fn: PricerFn,
    path_generator: PathGenerator,
    max_batches: int,
    target_std: float,
    min_batches: int = 1,
) -> RQMCResult:
    """Run randomized QMC (RQMC) in batches with adaptive stopping.

    (docstring unchanged from the original — keep it verbatim)
    """
    return run_rqmc_traced(
        pricer_fn, path_generator, max_batches, target_std, min_batches
    )[0]
```

Note the returned `RQMCResult.std_error` stays the raw `np.sqrt(...)` numpy scalar exactly as before (the checkpoint stores `float(std_error)` separately — do NOT change the result's type). Update `__all__` to add `RQMCCheckpoint`, `RQMCRunSpec`, `run_rqmc_traced`. In `quantark/montecarlo/__init__.py`, extend the existing `qmc_rqmc_driver` import/`__all__` block with the same three names.

- [ ] **Step 4: Run the new tests and the driver's existing consumers**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/execution/test_rqmc_driver_trace.py -v`
Expected: PASS
Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_snowball_mc.py test/test_phoenix_option.py -q` (adjust to the actual snowball/phoenix MC test module names found via `ls test/ | grep -iE "snowball|phoenix"`)
Expected: PASS (direct path unchanged)

- [ ] **Step 5: Commit**

```bash
git add quantark/montecarlo/qmc_rqmc_driver.py quantark/montecarlo/__init__.py test/execution/test_rqmc_driver_trace.py
git commit -m "feat(execution): traced RQMC driver - one Welford stopping-loop implementation"
```

---

### Task 2: Snowball RQMC run-spec seams

**Files:**
- Modify: `quantark/asset/equity/engine/mc/snowball_mc_engine.py`
- Test: `test/execution/test_adaptive_adapter.py` (seam-level tests only in this task)

**Interfaces:**
- Consumes: `RQMCRunSpec`, `run_rqmc` from Task 1.
- Produces: `SnowballMCEngine._rqmc_spec(product, pricing_env, S, T, r, q, sigma) -> RQMCRunSpec`; `_ko_reset_rqmc_spec(...) -> RQMCRunSpec`; `build_rqmc_session_spec(product, pricing_env) -> RQMCRunSpec | None`; `_complete_price(product, result) -> float`. `price()` and `_price_rqmc`/`_price_ko_reset_rqmc` are pure re-expressions over these seams (bit-identical).

- [ ] **Step 1: Write the failing tests**

Create `test/execution/test_adaptive_adapter.py` with a shared fixture module-level helper (later tasks extend this file):

```python
# test/execution/test_adaptive_adapter.py
"""Adaptive RQMC session path: seams, adapter, kernel dispatch.

Bit-identity bar (kickoff decision 2026-07-16): session PRICE must equal the
direct engine call bitwise - price, std_error, batches_used, event probs.
"""
import numpy as np
import pytest

from quantark.montecarlo.qmc_rqmc_driver import RQMCRunSpec, run_rqmc


def _snowball_product():
    from quantark.asset.equity.product.option.snowball_config import BarrierConfig
    from quantark.asset.equity.product.option.snowball_option import SnowballOption

    return SnowballOption(
        initial_price=100.0, strike=100.0,
        barrier_config=BarrierConfig(
            ko_barrier=103.0, ko_rate=0.15,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=75.0, ki_continuous=True,
        ),
        contract_multiplier=1.0, maturity=1.0,
    )


def _env():
    from test.execution.matrix_fixtures import make_flat_env  # reuse if present
    return make_flat_env()


def _rqmc_engine(**overrides):
    from quantark.asset.equity.engine.mc import SnowballMCEngine
    from quantark.asset.equity.param import MCParams
    from quantark.util.enum.engine_enums import MonteCarloMethod

    params = MCParams(
        seed=42, num_paths=2048, time_steps=64,
        rqmc_min_batches=2, rqmc_max_batches=6, rqmc_target_std=1e-9,
        **overrides,
    )
    return SnowballMCEngine(params=params, method=MonteCarloMethod.RANDOMIZED_QUASI)


class TestSnowballRQMCSpecSeam:
    def test_session_spec_none_for_non_rqmc_method(self):
        from quantark.asset.equity.engine.mc import SnowballMCEngine
        from quantark.asset.equity.param import MCParams
        from quantark.util.enum.engine_enums import MonteCarloMethod

        engine = SnowballMCEngine(
            params=MCParams(seed=42, num_paths=1024, time_steps=64),
            method=MonteCarloMethod.QUASI,
        )
        assert engine.build_rqmc_session_spec(_snowball_product(), _env()) is None

    def test_session_spec_shape(self):
        engine = _rqmc_engine()
        spec = engine.build_rqmc_session_spec(_snowball_product(), _env())
        assert isinstance(spec, RQMCRunSpec)
        assert spec.max_batches == 6 and spec.min_batches == 2
        assert spec.paths_per_batch == spec.path_generator.num_paths

    def test_spec_driven_run_equals_direct_price(self):
        product, env = _snowball_product(), _env()
        direct = _rqmc_engine()
        direct_price = direct.price(product, env)
        direct_result = direct.get_last_result()

        session_like = _rqmc_engine()
        spec = session_like.build_rqmc_session_spec(product, env)
        result = spec.finalize(run_rqmc(
            pricer_fn=spec.pricer_fn, path_generator=spec.path_generator,
            max_batches=spec.max_batches, target_std=spec.target_std,
            min_batches=spec.min_batches,
        ))
        price = session_like._complete_price(product, result)

        assert price == direct_price
        assert result.std_error == direct_result.std_error
        assert result.batches_used == direct_result.batches_used
        assert result.ko_probability == direct_result.ko_probability
        assert session_like.get_last_result() is result
```

(If `matrix_fixtures.make_flat_env` does not exist under that name, use the module's actual flat-environment helper — inspect `test/execution/matrix_fixtures.py` and reuse its env builder; do not build a new environment style.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/execution/test_adaptive_adapter.py -v`
Expected: FAIL with `AttributeError: ... no attribute 'build_rqmc_session_spec'`

- [ ] **Step 3: Extract the seams (pure refactor + two new methods)**

In `snowball_mc_engine.py`:

1. Add import: `from quantark.montecarlo.qmc_rqmc_driver import RQMCRunSpec, run_rqmc` (replacing/extending the current `run_rqmc` import).

2. Split `_price_rqmc` into `_rqmc_spec` + a two-line `_price_rqmc`. The extraction moves the existing body VERBATIM; the trailing result assembly (extra `batch_id=0` statistics batch + `SnowballMCResult(...)`) becomes the `finalize` closure:

```python
def _rqmc_spec(self, product, pricing_env, S, T, r, q, sigma) -> RQMCRunSpec:
    """Build the RQMC run description for a SnowballOption.

    Shared by the direct path (_price_rqmc) and the execution-framework
    session path (build_rqmc_session_spec): ONE construction of the grid,
    pricer, generator, controls, and result assembly.
    """
    all_times, dt_array, ko_indices, ki_indices = self._build_time_grid(
        product, pricing_env, T
    )

    def pricer_fn(paths, aux):
        # ... EXACT existing body ...

    params = self.params
    max_batches = getattr(
        params, "rqmc_max_batches", getattr(params, "max_batches", 32)
    )
    # ... EXACT existing min_batches / target_std / per_batch_paths resolution ...

    generator = self._create_path_generator(
        S, r, q, sigma, T, dt_array, num_paths=per_batch_paths
    )

    def finalize(result):
        paths, _ = generator.generate_paths(return_aux=False, batch_id=0)
        _, _, stats = self._compute_payoffs(
            product, pricing_env, paths, all_times, ko_indices, ki_indices,
            r, T, sigma, rng_seed=int(self.params.seed) + 1337,
        )
        return SnowballMCResult(
            price=result.price, std_error=result.std_error,
            num_paths=result.total_paths,
            ko_probability=stats["ko_probability"],
            v0_probability=stats["v0_probability"],
            v1_probability=stats["v1_probability"],
            batches_used=result.batches_used,
        )

    return RQMCRunSpec(
        pricer_fn=pricer_fn, path_generator=generator,
        max_batches=max_batches, min_batches=min_batches,
        target_std=target_std, paths_per_batch=per_batch_paths,
        time_steps=int(dt_array.size),
        scheme=f"{type(self).__qualname__}/rqmc-native/v1",
        finalize=finalize,
        product=product,
    )

def _price_rqmc(self, product, pricing_env, S, T, r, q, sigma) -> SnowballMCResult:
    """Price using Randomized QMC with adaptive batching."""
    spec = self._rqmc_spec(product, pricing_env, S, T, r, q, sigma)
    return spec.finalize(run_rqmc(
        pricer_fn=spec.pricer_fn, path_generator=spec.path_generator,
        max_batches=spec.max_batches, target_std=spec.target_std,
        min_batches=spec.min_batches,
    ))
```

3. Apply the identical split to `_price_ko_reset_rqmc` → `_ko_reset_rqmc_spec` (grid via `_build_time_grid_ko_reset`, payoffs via `_compute_payoffs_ko_reset`, `stats.get(...)` result assembly — all verbatim).

4. Extract the `price()` postamble (the three statements after the dispatch if/else) into `_complete_price`, and call it from `price()`:

```python
def _complete_price(self, product, result) -> float:
    """price() postamble shared with the session adaptive path."""
    self._last_result = result
    # Negative PV is valid for some structures (e.g., principal-excluded
    # notes) where the payoff is effectively "coupon minus embedded option
    # loss".
    if result.price < 0 and product.payoff_config.include_principal:
        raise PricingError(f"Negative price computed: {result.price}")
    return result.price
```

`price()` ends with `return self._complete_price(product, result)`.

5. Add the session seam, mirroring the `price()` preamble exactly:

```python
def build_rqmc_session_spec(self, product, pricing_env):
    """Execution-framework seam: the price() preamble plus the RQMC run
    description, or None when a direct call would not take the adaptive
    RQMC route (non-RQMC method, near-expiry shortcut) - the session then
    falls back to the native legacy call."""
    if self.method != MonteCarloMethod.RANDOMIZED_QUASI:
        return None
    if not isinstance(product, (SnowballOption, KnockOutResetSnowballOption)):
        raise PricingError(
            "SnowballMCEngine only supports SnowballOption or "
            f"KnockOutResetSnowballOption, got {type(product).__name__}"
        )
    S = pricing_env.spot
    T = (
        product.get_max_maturity_time(pricing_env)
        if isinstance(product, KnockOutResetSnowballOption)
        else product.get_maturity(pricing_env)
    )
    r = pricing_env.get_rate(T)
    q = pricing_env.get_div_yield(T)
    sigma = pricing_env.get_vol(product.strike, T)
    self._validate_inputs(S, T, r, q, sigma, product)
    self._term_ctx = (pricing_env, product.strike)
    self._df = make_df_fn(pricing_env)
    if T < 1e-10:
        return None
    if isinstance(product, KnockOutResetSnowballOption):
        return self._ko_reset_rqmc_spec(product, pricing_env, S, T, r, q, sigma)
    return self._rqmc_spec(product, pricing_env, S, T, r, q, sigma)
```

- [ ] **Step 4: Run the seam tests and the full snowball direct-path suite**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/execution/test_adaptive_adapter.py -v`
Expected: PASS
Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -k "snowball and not quad" -q`
Expected: PASS (pure extraction; direct path bit-identical)

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/engine/mc/snowball_mc_engine.py test/execution/test_adaptive_adapter.py
git commit -m "refactor(mc): snowball RQMC run-spec seams for session execution"
```

---

### Task 3: Phoenix RQMC run-spec seams

**Files:**
- Modify: `quantark/asset/equity/engine/mc/phoenix_mc_engine.py`
- Test: `test/execution/test_adaptive_adapter.py` (extend)

**Interfaces:**
- Produces: `PhoenixMCEngine._rqmc_spec(...) -> RQMCRunSpec`, `build_rqmc_session_spec(product, pricing_env) -> RQMCRunSpec | None`, `_complete_price(product, result) -> float` — same contracts as Task 2.

- [ ] **Step 1: Write the failing tests** — add to `test/execution/test_adaptive_adapter.py` a `TestPhoenixRQMCSpecSeam` class mirroring Task 2's three tests, using `create_standard_phoenix` (from `quantark.asset.equity.product.option.phoenix_helpers`) with `initial_price=100.0, strike=100.0, maturity=1.0, ko_barrier=103.0, ki_barrier=75.0, coupon_barrier=85.0, coupon_rate=0.01, num_observations=4`, `PhoenixMCEngine`, and the same `MCParams` RQMC controls. Assert bitwise equality of `price`, `std_error`, `batches_used`, `ko_probability`.

- [ ] **Step 2: Run to verify failure** — `AttributeError: ... 'build_rqmc_session_spec'`.

- [ ] **Step 3: Implement** — mirror Task 2 exactly on `phoenix_mc_engine.py`:
  - `_rqmc_spec` extracts the `_price_rqmc` body (note: the existing method creates `generator` TWICE — once without `num_paths` at the top and again with `num_paths=per_batch_paths`; keep only behavior-relevant construction, i.e. build the throwaway first generator too IF AND ONLY IF removing it changes any consumed state — it does not (construction has no stream side effects; the first object is discarded unused), but bit-identity is the test's verdict: if the phoenix direct suite or the seam test fails, restore the redundant construction verbatim);
  - `finalize` closure runs the `batch_id=0` statistics batch and assembles `PhoenixMCResult` (payoff unpacking `_, _, stats, _, _, _, _, _, _` verbatim);
  - pricer_fn includes the `+ instant_coupon_discounted` term verbatim;
  - `_complete_price` extracts the `price()` postamble (`self._last_result = result`, negative-price check, `return result.price`);
  - `build_rqmc_session_spec` mirrors the phoenix `price()` preamble: `PhoenixOption` isinstance check (raise `PricingError` otherwise), S/T/r/q/sigma extraction, `_validate_inputs`, `_term_ctx`, `_df = make_df_fn(pricing_env)`, `T < 1e-10 -> None`, method gate first.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/execution/test_adaptive_adapter.py -v && PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -k "phoenix" -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/engine/mc/phoenix_mc_engine.py test/execution/test_adaptive_adapter.py
git commit -m "refactor(mc): phoenix RQMC run-spec seams for session execution"
```

---

### Task 4: `AdaptivePlan` contract

**Files:**
- Modify: `quantark/execution/contracts.py`, `quantark/execution/__init__.py`
- Test: `test/execution/test_batch_contracts.py` (extend)

**Interfaces:**
- Produces: frozen `AdaptivePlan(plan_id, engine_class_path, max_batches, min_batches, paths_per_batch, target_std, seed, stream_kind, stream_layout, time_steps, dimension, dtype, scheme, stopping_rule, checkpoint_policy, reduction_order, est_task_peak_bytes, implementation_fingerprint)`.

- [ ] **Step 1: Write the failing test** — in `test/execution/test_batch_contracts.py` add:

```python
def _adaptive_plan(**overrides):
    from quantark.execution.contracts import AdaptivePlan

    kw = dict(
        plan_id="p", engine_class_path="m.C", max_batches=8, min_batches=2,
        paths_per_batch=1024, target_std=1e-4, seed=42,
        stream_kind="sobol-rqmc", stream_layout="batch-shifted-sobol/v1",
        time_steps=64, dimension=64, dtype="float64",
        scheme="SnowballMCEngine/rqmc-native/v1",
        stopping_rule="welford-batch-means/v1",
        checkpoint_policy="after-each-batch/v1",
        reduction_order="batch-order-welford/v1",
        est_task_peak_bytes=1 << 20, implementation_fingerprint="a/1",
    )
    kw.update(overrides)
    return AdaptivePlan(**kw)


class TestAdaptivePlan:
    def test_frozen(self):
        import dataclasses
        plan = _adaptive_plan()
        with pytest.raises(dataclasses.FrozenInstanceError):
            plan.seed = 1

    def test_fingerprintable(self):
        from quantark.execution.cache.fingerprint import try_fingerprint
        assert try_fingerprint(_adaptive_plan()) is not None
        assert try_fingerprint(_adaptive_plan()) == try_fingerprint(_adaptive_plan())
        assert try_fingerprint(_adaptive_plan(seed=7)) != try_fingerprint(_adaptive_plan())
```

- [ ] **Step 2: Verify failure** (`ImportError`), **Step 3: implement** — add the dataclass to `contracts.py` (docstring: "Immutable adaptive RQMC plan (spec section 8.4). Sequential compatibility mode: the stopping criterion is evaluated after the same batches in the same order as the direct path; the checkpoint trace, not executor scheduling, is the reproducibility record."), extend `__all__`, and export `AdaptivePlan` from `quantark/execution/__init__.py`. **Step 4: run** `test_batch_contracts.py`. **Step 5: commit** `feat(execution): AdaptivePlan contract`.

---

### Task 5: `AutocallableAdaptiveMCAdapter`

**Files:**
- Create: `quantark/asset/equity/engine/mc/autocallable_execution_adapters.py`
- Test: `test/execution/test_adaptive_adapter.py` (extend)

**Interfaces:**
- Consumes: `build_rqmc_session_spec` / `_complete_price` (Tasks 2-3), `run_rqmc_traced` (Task 1), `AdaptivePlan` (Task 4), `LegacyPriceAdapter`.
- Produces: adapter with `capabilities()` (adapter_id `"autocallable-adaptive-mc"`, version `"1"`, `supported_backends=frozenset({"serial"})`), `prepare(engine, request, context) -> PreparedState` (payload = spec or None; spec built ONLY for `PricingOperation.PRICE`; acquires the per-engine lock and puts its release handle in `PreparedState.handles`), `plan_adaptive(engine, request, prepared, context) -> AdaptivePlan | None`, `execute_adaptive(engine, plan, prepared, context) -> (value, economics, trace)` (task-scratch lease spans the RQMC run AND finalize/postamble).
- Tests that call `prepare` directly (bypassing the kernel) MUST close `prepared.handles` in a `finally`, mirroring the kernel's cleanup — otherwise the engine lock stays held.

- [ ] **Step 1: Write the failing tests** — extend `test/execution/test_adaptive_adapter.py`:

```python
class TestAutocallableAdaptiveAdapter:
    def _adapter(self):
        from quantark.asset.equity.engine.mc.autocallable_execution_adapters import (
            AutocallableAdaptiveMCAdapter,
        )
        return AutocallableAdaptiveMCAdapter()

    def _context(self):
        from test.execution.test_kernel_prepare import make_context  # reuse the
        # existing context factory used by the batch-adapter tests; if named
        # differently, reuse that module's actual helper.
        return make_context()

    def test_plan_shape(self):
        from quantark.execution.contracts import PricingRequest
        adapter, engine = self._adapter(), _rqmc_engine()
        request = PricingRequest(product=_snowball_product(), pricing_env=_env())
        context = self._context()
        prepared = adapter.prepare(engine, request, context)
        plan = adapter.plan_adaptive(engine, request, prepared, context)
        assert plan is not None
        assert plan.max_batches == 6 and plan.min_batches == 2
        assert plan.paths_per_batch == prepared.payload.paths_per_batch
        assert plan.stopping_rule == "welford-batch-means/v1"
        assert plan.engine_class_path.endswith("SnowballMCEngine")

    def test_plan_none_for_non_rqmc(self):
        from quantark.execution.contracts import PricingRequest
        from quantark.asset.equity.engine.mc import SnowballMCEngine
        from quantark.asset.equity.param import MCParams
        from quantark.util.enum.engine_enums import MonteCarloMethod

        adapter = self._adapter()
        engine = SnowballMCEngine(
            params=MCParams(seed=42, num_paths=1024, time_steps=64),
            method=MonteCarloMethod.QUASI,
        )
        request = PricingRequest(product=_snowball_product(), pricing_env=_env())
        context = self._context()
        prepared = adapter.prepare(engine, request, context)
        assert adapter.plan_adaptive(engine, request, prepared, context) is None

    def test_execute_bitwise_vs_direct(self):
        from quantark.execution.contracts import PricingRequest
        product, env = _snowball_product(), _env()
        direct = _rqmc_engine()
        expected = direct.price(product, env)
        expected_result = direct.get_last_result()

        adapter, engine = self._adapter(), _rqmc_engine()
        request = PricingRequest(product=product, pricing_env=env)
        context = self._context()
        prepared = adapter.prepare(engine, request, context)
        plan = adapter.plan_adaptive(engine, request, prepared, context)
        value, economics, trace = adapter.execute_adaptive(
            engine, plan, prepared, context
        )
        assert value == expected
        econ = dict(economics)
        assert econ["pv"] == expected
        assert econ["std_error"] == float(expected_result.std_error)
        assert len(trace) == expected_result.batches_used
        assert trace[-1].stopped
```

- [ ] **Step 2: Verify failure** (`ModuleNotFoundError`).

- [ ] **Step 3: Implement the adapter**

```python
# quantark/asset/equity/engine/mc/autocallable_execution_adapters.py
"""Adaptive RQMC compatibility adapter (spec section 8.4, Phase 3).

Sequential COMPATIBILITY mode only: execution runs on the LIVE engine
instance on the calling thread, driving the same run_rqmc_traced loop the
direct path uses, so the stopping sequence and every produced number are
bit-identical to a direct call by construction. No cloning: the 12
Snowball/Phoenix engines (BSM/LV/Heston/QE/SLV variants) flow through their
own polymorphic hooks (_create_path_generator, _compute_payoffs), so a
single NON-exact registration on each base class is subclass-safe -
deliberately unlike the DCN batch adapters, whose fixed-signature cloning
forces exact=True registrations.

Instance exclusion (plan-gate finding 2026-07-16): the engine mutates its
request-scoped state (_term_ctx/_df/_last_result) during the run, exactly as
price() does — so two overlapping session dispatches on the SAME engine
instance could overwrite each other's discount function and return
mixed-market PVs. prepare() therefore acquires a per-engine lock whose
release rides in ``PreparedState.handles``: the kernel's ``finally`` closes
every handle after execution, so the lock spans preparation through
execution and is released even on failure. Dispatches on distinct engine
instances are unaffected. (An engine must not re-dispatch itself inside a
session call; that would self-deadlock, exactly as recursive direct pricing
would corrupt its own request state.)

Parallel-wave adaptive RQMC (a distinct opt-in plan) is deferred on Phase 2
benchmark evidence (host-limited thread scaling).
"""
import threading
import weakref

from quantark.execution.contracts import (
    AdaptivePlan,
    EngineCapabilities,
    OutputKind,
    PreparedState,
    PricingOperation,
)
from quantark.execution.errors import CapabilityError
from quantark.execution.legacy_adapter import LegacyPriceAdapter
from quantark.montecarlo.qmc_rqmc_driver import run_rqmc_traced

__all__ = ["AutocallableAdaptiveMCAdapter"]

ADAPTER_ID = "autocallable-adaptive-mc"
ADAPTER_VERSION = "1"

_PRICE_OUTPUTS = frozenset({OutputKind.PV, OutputKind.ERROR_ESTIMATE})

# Per-engine-instance exclusion (plan-gate finding 2026-07-16).
_ENGINE_LOCKS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_LOCKS_GUARD = threading.Lock()


def _engine_lock(engine) -> threading.Lock:
    with _LOCKS_GUARD:
        lock = _ENGINE_LOCKS.get(engine)
        if lock is None:
            lock = threading.Lock()
            _ENGINE_LOCKS[engine] = lock
        return lock


class _LockHandle:
    """Rides in PreparedState.handles; the kernel's finally releases it."""

    def __init__(self, lock):
        self._lock = lock

    def close(self):
        if self._lock is not None:
            lock, self._lock = self._lock, None
            lock.release()


class AutocallableAdaptiveMCAdapter(LegacyPriceAdapter):
    def __init__(self):
        super().__init__(call_shape="product_env")

    def capabilities(self) -> EngineCapabilities:
        base = super().capabilities()
        return EngineCapabilities(
            operations=base.operations,
            output_kinds=_PRICE_OUTPUTS,
            supported_backends=frozenset({"serial"}),
            fixed_planning=False,
            prepared_state_thread_safe=False,
            instance_reentrant=False,
            process_reconstructable=False,
            deterministic_reduction=True,
            peak_memory_estimate="conservative",
            adapter_id=ADAPTER_ID,
            adapter_version=ADAPTER_VERSION,
        )

    def validate(self, engine, request) -> None:
        allowed = (
            _PRICE_OUTPUTS
            if request.operation is PricingOperation.PRICE
            else frozenset({OutputKind.PV})
        )
        extra = request.outputs - allowed
        if extra:
            raise CapabilityError(
                f"outputs {sorted(k.value for k in extra)} unsupported for "
                f"operation {request.operation.value} via {ADAPTER_ID}"
            )
        if request.pricing_env is None:
            raise CapabilityError(
                "pricing_env is required for product_env engines"
            )

    def prepare(self, engine, request, context) -> PreparedState:
        # Per-engine exclusion FIRST: the spec build below mutates the
        # engine's request-scoped state (_term_ctx/_df), and execution reads
        # it. The lock handle rides in PreparedState.handles; the kernel's
        # finally releases it after execution (or on any failure).
        lock = _engine_lock(engine)
        lock.acquire()
        handle = _LockHandle(lock)
        try:
            # The spec build reproduces the price() preamble (validation +
            # request-scoped engine state); only the PRICE operation takes
            # the adaptive route, so other operations skip it entirely.
            spec = None
            if request.operation is PricingOperation.PRICE:
                spec = engine.build_rqmc_session_spec(
                    request.product, request.pricing_env
                )
        except BaseException:
            handle.close()
            raise
        return PreparedState(
            payload=spec, descriptors=(), fingerprint=None,
            byte_estimate=None, handles=(handle,),
        )

    def plan_adaptive(self, engine, request, prepared, context):
        spec = prepared.payload if prepared is not None else None
        if spec is None:
            return None
        cls = type(engine)
        engine_class_path = f"{cls.__module__}.{cls.__qualname__}"
        seed = int(engine.params.seed)
        # paths + payoff work arrays; engine-native draw streams may double
        # the per-step footprint (LV/Heston two-stream variants), and the
        # finalizer's extra statistics batch peaks at one more batch (the
        # RQMC batch memory is freed batch-by-batch, so ONE batch bound
        # covers both phases; the x5 slack absorbs payoff work arrays)
        est = 8 * spec.paths_per_batch * 5 * (spec.time_steps + 1) + (1 << 20)
        return AdaptivePlan(
            plan_id=(
                f"{engine_class_path}:{seed}:{spec.paths_per_batch}"
                f":{spec.min_batches}-{spec.max_batches}:{spec.target_std!r}"
            ),
            engine_class_path=engine_class_path,
            max_batches=int(spec.max_batches),
            min_batches=int(spec.min_batches),
            paths_per_batch=int(spec.paths_per_batch),
            target_std=float(spec.target_std),
            seed=seed,
            stream_kind="sobol-rqmc",
            stream_layout="batch-shifted-sobol/v1",
            time_steps=int(spec.time_steps),
            dimension=int(spec.time_steps),
            dtype="float64",
            scheme=spec.scheme,
            stopping_rule="welford-batch-means/v1",
            checkpoint_policy="after-each-batch/v1",
            reduction_order="batch-order-welford/v1",
            est_task_peak_bytes=est,
            implementation_fingerprint=f"{ADAPTER_ID}/{ADAPTER_VERSION}",
        )

    def execute_adaptive(self, engine, plan, prepared, context):
        spec = prepared.payload
        mgr = context.lease_manager
        est = plan.est_task_peak_bytes or 0
        # The lease spans finalize too (plan-gate finding 2026-07-16): the
        # Snowball/Phoenix finalizers generate one more full statistics
        # batch, which must stay inside admission control.
        if mgr is not None and est:
            mgr.lease_bytes(est, "task_scratch")
        try:
            result, trace = run_rqmc_traced(
                pricer_fn=spec.pricer_fn,
                path_generator=spec.path_generator,
                max_batches=spec.max_batches,
                target_std=spec.target_std,
                min_batches=spec.min_batches,
            )
            native = spec.finalize(result)
            price = engine._complete_price(spec.product, native)
        finally:
            if mgr is not None and est:
                mgr.release_bytes(est, "task_scratch")
        economics = (
            ("pv", float(price)),
            ("std_error", float(native.std_error)),
        )
        return price, economics, trace
```

**Note:** `spec.product` is the `RQMCRunSpec` field defined in Task 1 and populated by the Task 2-3 seams. Verify the `lease_bytes(n, pool)` argument order against `quantark/execution/leases.py:32` before writing the calls.

- [ ] **Step 4: Run** `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/execution/test_adaptive_adapter.py -v` — PASS.

- [ ] **Step 5: Commit** `feat(execution): AutocallableAdaptiveMCAdapter - sequential compatibility plans`.

---

### Task 6: Kernel adaptive dispatch + registry wiring

**Files:**
- Modify: `quantark/execution/kernel.py`, `quantark/execution/registry.py`
- Test: `test/execution/test_adaptive_adapter.py` (extend), `test/execution/test_registry.py` (expected set)

**Interfaces:**
- Kernel: when `hasattr(adapter, "plan_adaptive")` and the plan is non-None → `execute_adaptive`; manifest gets the plan fingerprint; diagnostics records gain `adaptive:batches_used=<n>`, `adaptive:stopped_early=<bool>`, and `adaptive:trace_fingerprint=<fp>` (deterministic fingerprint of the complete `RQMCCheckpoint` tuple — `try_fingerprint(trace)`; if the tuple is not fingerprintable as-is, make `RQMCCheckpoint` fingerprint-friendly rather than dropping the record); `batch_count` timing = batches used. When the plan is None → existing native path (dispatch-wide slot logic unchanged: adaptive adapters have no `plan_batches`, so the slot is held).
- Registry: non-exact registrations `...snowball_mc_engine.SnowballMCEngine` and `...phoenix_mc_engine.PhoenixMCEngine` → lazy `AutocallableAdaptiveMCAdapter` factory (subclasses resolve via MRO — intended: the adapter is subclass-safe).

- [ ] **Step 1: Write the failing tests** — extend `test/execution/test_adaptive_adapter.py`:

```python
class TestKernelAdaptiveDispatch:
    def test_session_price_bitwise_vs_direct_snowball(self):
        product, env = _snowball_product(), _env()
        direct = _rqmc_engine()
        expected = direct.price(product, env)
        expected_result = direct.get_last_result()

        from quantark.execution import PricingSession
        with PricingSession() as session:
            outcome = session.price(_rqmc_engine(), product, env)
        assert outcome.value == expected
        econ = dict(outcome.normalized_economics)
        assert econ["std_error"] == float(expected_result.std_error)
        assert outcome.manifest.adapter_id == "autocallable-adaptive-mc"
        assert outcome.manifest.plan_fingerprint is not None
        records = outcome.diagnostics.records
        assert f"adaptive:batches_used={expected_result.batches_used}" in records

    def test_session_price_bitwise_vs_direct_ko_reset(self):
        # same shape with a KnockOutResetSnowballOption product fixture
        ...

    def test_session_price_bitwise_vs_direct_phoenix(self):
        # same shape with PhoenixMCEngine + phoenix product
        ...

    def test_non_rqmc_method_falls_to_native(self):
        # engine with MonteCarloMethod.QUASI: outcome.value == direct price,
        # manifest.plan_fingerprint is None, adapter_id stays adaptive
        ...

    def test_event_stats_operation_unchanged(self):
        # session.execute with PricingOperation.EVENT_STATS routes native
        # calculate_event_stats and matches the direct call's numbers
        ...
```

Fill the `...` bodies concretely following the first test's pattern (each is a fixture swap; the KO-reset fixture uses `KnockOutResetSnowballOption` — copy the construction from an existing KO-reset test found via `grep -rl KnockOutResetSnowballOption test/ | head -3`). In `test/execution/test_registry.py`, add the two new paths to the expected registration set. Use whatever `PricingSession` price/execute call signature the existing `test/execution/test_session_parity.py` uses — reuse, don't invent.

- [ ] **Step 2: Verify failures** (adapter not registered → `legacy-price` adapter_id; registry expected-set mismatch).

- [ ] **Step 3: Implement**

Kernel — inside `dispatch`, replace the `if batch_capable: ... else: execute_native` block with:

```python
adaptive_records: list = []
executed = False
if hasattr(adapter, "plan_adaptive"):
    adaptive_plan = adapter.plan_adaptive(engine, request, prepared, context)
    if adaptive_plan is not None:
        plan_fingerprint = try_fingerprint(adaptive_plan)
        value, economics, trace = adapter.execute_adaptive(
            engine, adaptive_plan, prepared, context
        )
        batch_count = len(trace)
        adaptive_records.append(f"adaptive:batches_used={len(trace)}")
        adaptive_records.append(
            "adaptive:stopped_early="
            f"{len(trace) < adaptive_plan.max_batches}"
        )
        # Complete-trace evidence (plan-gate finding 2026-07-16): a
        # deterministic fingerprint of every checkpoint value, so two runs
        # with different batch means can never produce identical records.
        adaptive_records.append(
            f"adaptive:trace_fingerprint={try_fingerprint(trace)}"
        )
        executed = True
if not executed and batch_capable:
    # ... existing plan_batches block unchanged ...
elif not executed:
    value, economics = adapter.execute_native(
        engine, request, normalized, context, prepared=prepared
    )
```

and fold `adaptive_records` into the `records` tuple next to the clamp records. Update the kernel module docstring (Phase 3 sentence).

Registry — after the Phase-2 block:

```python
# Phase 3: adaptive RQMC compatibility plans. NON-exact by design: the
# adapter drives the live engine through its own polymorphic hooks (no
# cloning), so every Snowball/Phoenix vol-model subclass inherits it safely.
registry.register(
    "quantark.asset.equity.engine.mc.snowball_mc_engine.SnowballMCEngine",
    _autocallable_adaptive_adapter,
)
registry.register(
    "quantark.asset.equity.engine.mc.phoenix_mc_engine.PhoenixMCEngine",
    _autocallable_adaptive_adapter,
)
```

with the lazy factory at module bottom following the existing pattern.

- [ ] **Step 4: Run** the adaptive tests plus the whole execution suite (`PYTHONPATH=$PWD ... -m pytest test/execution -q`) — the session-parity matrix must still pass (snowball/phoenix engines in non-RQMC configs now resolve to the adaptive adapter but fall through to native — parity tests assert values, and any adapter_id assertions must be extended to accept `autocallable-adaptive-mc`).

- [ ] **Step 5: Commit** `feat(execution): kernel adaptive dispatch + snowball/phoenix registrations`.

---

### Task 7: 12-engine adaptive bitwise matrix + checkpoint determinism

**Files:**
- Create: `test/execution/test_adaptive_vol_matrix.py`
- Test-only task.

- [ ] **Step 1: Write the matrix test.** Parametrize over all 12 engines. Reuse `test/execution/matrix_fixtures.py` builders for LV (GridVolSurface env), Heston (`HestonParams`), and SLV (leverage) configurations — follow how `matrix_fixtures.py` constructs `LocalVolSnowballMCEngine`/`HestonSnowballMCEngine`/etc. today, but force `method=MonteCarloMethod.RANDOMIZED_QUASI` and small controls (`num_paths=1024, time_steps=32, rqmc_min_batches=2, rqmc_max_batches=4, rqmc_target_std=1e-9` so every run goes to max_batches deterministically; plus one BSM case with `rqmc_target_std=1e3` asserting `batches_used == min_batches`). For each engine: direct `price()` + `get_last_result()` vs session `outcome.value`/economics — bitwise equality on price, std_error, batches_used. Include KO-reset under the plain and one vol engine.

Checkpoint determinism (same file):

```python
def test_checkpoint_trace_deterministic_and_fingerprint_stable():
    # two sessions, same engine config: identical adaptive records
    # (including adaptive:trace_fingerprint=...), identical
    # manifest.plan_fingerprint
    ...

def test_trace_fingerprint_changes_when_any_checkpoint_value_changes():
    # unit-level: build a trace tuple of RQMCCheckpoint, fingerprint it,
    # then rebuild with ONE field of ONE checkpoint perturbed (e.g.
    # batch_mean += 1e-12) and assert the fingerprint differs
    ...

def test_stop_boundaries():
    # loose target stops at min_batches; tight target runs to max_batches;
    # batches_used from diagnostics matches get-last-result on the direct
    # engine with identical controls
    ...

def test_concurrent_same_engine_dispatches_are_serialized():
    # plan-gate finding 2026-07-16: ONE engine instance, TWO distinct
    # environments (different rates), two threads released by a
    # threading.Barrier, each running session.price(engine, product, env_i)
    # on a session whose budget allows concurrency. Assert each thread's
    # value equals its own single-threaded direct price computed on a fresh
    # engine with the same config (no mixed-market cross-talk).
    ...
```

(Fill bodies concretely; assertions read `outcome.diagnostics.records` and `outcome.manifest.plan_fingerprint`. For the concurrency test, reuse the session/budget factory from `test/execution/test_backends.py` or `test_session_parity.py`.)

- [ ] **Step 2: Run — engines already wired, so these should PASS; any failure is a real bit-identity defect: STOP and fix the seam (do not loosen to tolerances).**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/execution/test_adaptive_vol_matrix.py -v`

- [ ] **Step 3: Commit** `test(execution): 12-engine adaptive RQMC bitwise matrix + checkpoint determinism`.

---

### Task 8: `DrawRepository.uniforms_handle`

**Files:**
- Modify: `quantark/execution/cache/draws.py`
- Test: `test/execution/test_draw_repository.py` (extend)

**Interfaces:**
- Produces: `DrawRepository.uniforms_handle(*, seed, n_paths, dim, batch_id, writable=False) -> ArtifactHandle`; `descriptor(..., distribution="normal")` gains the distribution axis (uniform blocks: `distribution="uniform"`, `transform_pipeline=()`).

- [ ] **Step 1: Write the failing tests** — extend `test/execution/test_draw_repository.py`:

```python
def test_uniforms_bitwise_match_qmc_uniforms(repo):
    from quantark.asset.equity.engine.mc.qmc_draws import qmc_uniforms
    handle = repo.uniforms_handle(seed=42, n_paths=64, dim=12, batch_id=3)
    expected = qmc_uniforms(42, 64, 12, batch_id=3)
    assert np.array_equal(handle.value, expected)
    assert not handle.value.flags.writeable
    handle.close()

def test_uniform_and_normal_blocks_have_distinct_keys(repo):
    u = repo.uniforms_handle(seed=42, n_paths=32, dim=8, batch_id=None)
    n = repo.normals_handle(seed=42, n_paths=32, dim=8, batch_id=None)
    assert not np.array_equal(u.value, n.value)
    u.close(); n.close()

def test_uniforms_writable_private_copy(repo):
    a = repo.uniforms_handle(seed=1, n_paths=16, dim=4, batch_id=0, writable=True)
    b = repo.uniforms_handle(seed=1, n_paths=16, dim=4, batch_id=0)
    assert a.value.flags.writeable
    a.value[:] = 0.0
    assert not np.array_equal(a.value, b.value)
    a.close(); b.close()

def test_uniforms_cache_hit_on_reuse(repo):
    h1 = repo.uniforms_handle(seed=9, n_paths=32, dim=6, batch_id=1); h1.close()
    before = dict(repo.stats())
    h2 = repo.uniforms_handle(seed=9, n_paths=32, dim=6, batch_id=1); h2.close()
    after = dict(repo.stats())
    assert after["hits"] == before["hits"] + 1
```

(Reuse the file's existing `repo` fixture; adjust the stats key to the actual key names in `PreparedArtifactCache.stats()`.)

- [ ] **Step 2: Verify failure**, **Step 3: implement**:

In `draws.py`, generalize `descriptor()`:

```python
def descriptor(self, *, seed, n_paths, dim, batch_id,
               distribution="normal") -> DrawDescriptor:
    versions = dict(build_versions())
    return DrawDescriptor(
        generator_family="sobol-scrambled",
        implementation_id=_IMPL_ID,
        implementation_version=_IMPL_VERSION,
        distribution=distribution,
        stream_layout=_LAYOUT,
        seed=int(seed),
        batch_id=None if batch_id is None else int(batch_id),
        n_paths=int(n_paths),
        dim=int(dim),
        shape=(int(n_paths), int(dim)),
        memory_order="C",
        dtype="float64",
        antithetic=False,
        transform_pipeline=("ndtri/v1",) if distribution == "normal" else (),
        numpy_version=versions.get("numpy", "unknown"),
        scipy_version=versions.get("scipy", "unknown"),
    )
```

Refactor the shared handle logic:

```python
def _block_handle(self, *, kind, distribution, draw_method, seed, n_paths,
                  dim, batch_id, writable):
    desc = self.descriptor(seed=seed, n_paths=n_paths, dim=dim,
                           batch_id=batch_id, distribution=distribution)
    art = ArtifactDescriptor(
        kind=kind, fingerprint=desc.fingerprint,
        dependency_tags=frozenset({"draws"}), builder_version=_IMPL_VERSION,
    )

    def builder():
        from quantark.montecarlo.qmc_sobol import SobolNormalGenerator

        gen = SobolNormalGenerator(base_seed=int(seed))
        block = np.ascontiguousarray(
            getattr(gen, draw_method)(int(n_paths), int(dim), batch_id=batch_id)
        )
        block.flags.writeable = False
        return block

    handle = self._cache.get_or_build(
        art, builder,
        size_bytes=int(n_paths) * int(dim) * 8,
        measure=lambda block: block.nbytes,
    )
    if not writable:
        return handle
    copy = handle.value.copy()
    handle.close()  # writable scratch is task-owned, not pinned
    return ArtifactHandle(copy, lambda: None)


def normals_handle(self, *, seed, n_paths, dim, batch_id, writable=False):
    return self._block_handle(
        kind="sobol-normal-block", distribution="normal", draw_method="normal",
        seed=seed, n_paths=n_paths, dim=dim, batch_id=batch_id, writable=writable,
    )


def uniforms_handle(self, *, seed, n_paths, dim, batch_id, writable=False):
    return self._block_handle(
        kind="sobol-uniform-block", distribution="uniform", draw_method="uniform",
        seed=seed, n_paths=n_paths, dim=dim, batch_id=batch_id, writable=writable,
    )
```

- [ ] **Step 4: Run** `test_draw_repository.py` (all, not just new) — PASS. **Step 5: Commit** `feat(execution): DrawRepository uniform Sobol blocks`.

---

### Task 9: Heston DCN uniform-provider hook + mixin plan hooks

**Files:**
- Modify: `quantark/asset/equity/engine/mc/dcn_vol_mc_engines.py`, `quantark/asset/equity/engine/mc/dcn_execution_adapters.py`
- Test: `test/execution/test_heston_dcn_batch.py` (created here; adapter tests land in Task 10)

**Interfaces:**
- `HestonDCNMCEngine._uniform_provider = None` class attr; the 3-stream Sobol branch of `_heston_draws` consults it: `block = self._uniform_provider(3 * n_steps, n_paths, batch_id)` must return a WRITABLE private `(n_paths, 3*n_steps)` uniform block bit-identical to `qmc_uniforms(self.seed, n_paths, 3*n_steps, batch_id=batch_id, writable=True)`.
- `_DCNBatchMixin` gains overridable plan hooks: `_plan_scheme(sim) -> str` (default `self._SCHEME`), `_plan_dimension(sim, time_steps) -> int` (default `time_steps`), `_plan_stream_layout(clone) -> tuple[str, str]` (default = current stream_kind/stream_layout branching, moved verbatim), `_plan_est_bytes(sim, clone, batch_size, time_steps) -> tuple[int, int]` (default = current formulas, moved verbatim). `plan_batches` calls the hooks; GBM/LV plans must be UNCHANGED (existing plan-config tests are the regression net).
- `_UniformDrawProvider(repository, seed)` callable in `dcn_execution_adapters.py` returning `repo.uniforms_handle(..., writable=True).value`.

- [ ] **Step 1: Write the failing tests** — create `test/execution/test_heston_dcn_batch.py`:

```python
"""Heston DCN fixed-batch adapters (Phase 3): bit-identity + draw routing."""
import numpy as np
import pytest


def _heston_params():
    from quantark.volmodels.heston import HestonParams
    return HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)


def test_uniform_provider_hook_intercepts_three_stream_draws():
    from quantark.asset.equity.engine.mc.dcn_vol_mc_engines import HestonDCNMCEngine
    from quantark.asset.equity.engine.mc.qmc_draws import qmc_uniforms
    from quantark.util.enum.engine_enums import HestonMCScheme

    engine = HestonDCNMCEngine(
        model_params=_heston_params(), scheme=HestonMCScheme.QUADEXP,
        num_paths=128, seed=7, use_sobol=True, num_batches=1,
    )
    calls = []

    def provider(n_dims, n_paths, batch_id):
        calls.append((n_dims, n_paths, batch_id))
        return qmc_uniforms(engine.seed, n_paths, n_dims,
                            batch_id=batch_id, writable=True)

    baseline = engine._heston_draws(4, 128, None)
    engine._uniform_provider = provider
    hooked = engine._heston_draws(4, 128, None)
    assert calls == [(12, 128, None)]
    for a, b in zip(baseline, hooked):
        assert np.array_equal(a, b)


def test_uniform_provider_ignored_for_two_stream_scheme():
    from quantark.asset.equity.engine.mc.dcn_vol_mc_engines import HestonDCNMCEngine
    from quantark.util.enum.engine_enums import HestonMCScheme

    engine = HestonDCNMCEngine(
        model_params=_heston_params(),
        scheme=HestonMCScheme.FULL_TRUNCATION_EULER,
        num_paths=64, seed=7, use_sobol=True, num_batches=1,
    )
    engine._uniform_provider = lambda *a: pytest.fail("must not be called")
    z_var, z_ind, u_var = engine._heston_draws(4, 64, None)
    assert u_var is None
```

Plus a mixin-hook regression test (in `test/execution/test_dcn_batch_adapter.py`): assert the GBM and LV `plan_batches` outputs (scheme, dimension, stream_kind, stream_layout, est bytes) are byte-for-byte what they were before the hook refactor — if that file already asserts plan fields, extend it to cover all of scheme/dimension/layout/est fields explicitly.

- [ ] **Step 2: Verify failure** (provider attribute has no effect yet).

- [ ] **Step 3: Implement**

`dcn_vol_mc_engines.py` — add `_uniform_provider = None` as a `HestonDCNMCEngine` class attribute (document: "session draw-provider hook, mirroring DCNMCEngine._draw_provider; must return a WRITABLE private block") and change only the 3-stream branch head:

```python
        if self.use_sobol and three_streams:
            from scipy.special import ndtri

            if self._uniform_provider is not None:
                block = self._uniform_provider(3 * n_steps, n_paths, batch_id)
            else:
                # writable=True: this path transforms the block in place, ...
                block = qmc_uniforms(
                    self.seed, n_paths, 3 * n_steps, batch_id=batch_id,
                    writable=True,
                )
            np.clip(block, 1e-12, 1.0 - 1e-12, out=block)
            ...  # unchanged
```

`dcn_execution_adapters.py` — hook refactor of `plan_batches` (move the existing stream branching and est formulas into the four defaults verbatim; call sites replaced by hook calls) and:

```python
class _UniformDrawProvider:
    """Uniform Sobol blocks from the session DrawRepository as WRITABLE
    private copies (the QE draw path clips + ndtri-transforms in place).
    The writable path never pins the master beyond the call, so no
    per-batch scoping is needed."""

    def __init__(self, repository, seed):
        self._repo = repository
        self._seed = seed

    def __call__(self, n_dims, n_paths, batch_id):
        handle = self._repo.uniforms_handle(
            seed=self._seed, n_paths=n_paths, dim=n_dims,
            batch_id=batch_id, writable=True,
        )
        try:
            return handle.value
        finally:
            handle.close()
```

- [ ] **Step 4: Run** the two new tests + the full `test/execution/test_dcn_batch_adapter.py` + `test/execution/test_regression_gates.py` (plan-shape and bitwise gates must be untouched by the hook refactor) — PASS.

- [ ] **Step 5: Commit** `feat(mc): Heston DCN uniform-provider hook + batch-plan metadata hooks`.

---

### Task 10: Heston + QE DCN batch adapters (exact registrations)

**Files:**
- Modify: `quantark/asset/equity/engine/mc/dcn_execution_adapters.py`, `quantark/execution/registry.py`
- Test: `test/execution/test_heston_dcn_batch.py` (extend), `test/execution/test_registry.py`

**Interfaces:**
- Produces: `DCNHestonBatchMCAdapter` (exact for `HestonDCNMCEngine`), `DCNQEBatchMCAdapter` (exact for `QEDCNMCEngine`); both `_DCNBatchMixin` subclasses sharing `_HestonBatchBase`:

```python
class _HestonBatchBase(_DCNBatchMixin, LegacyPriceAdapter):
    def __init__(self):
        super().__init__(call_shape="product_env")

    def _plan_scheme(self, sim):
        clone = sim.engine
        return (
            f"heston-{clone.scheme.name.lower()}"
            f"-sub{clone.substeps_per_interval}/v1"
        )

    def _plan_dimension(self, sim, time_steps):
        clone = sim.engine
        n_fine = time_steps * clone.substeps_per_interval
        return 3 * n_fine if self._three_streams(clone) else 2 * n_fine

    def _plan_stream_layout(self, clone):
        kind, layout = super()._plan_stream_layout(clone)
        streams = "3stream" if self._three_streams(clone) else "2stream"
        return kind, f"{layout.rsplit('/', 1)[0]}-{streams}/v1"

    def _plan_est_bytes(self, sim, clone, batch_size, time_steps):
        n_fine = time_steps * clone.substeps_per_interval
        per_path = 3 * n_fine if self._three_streams(clone) else 2 * n_fine
        # block + writable copy + variance/spot state + contractual nodes
        est_task = 8 * batch_size * (2 * per_path + 4 + 6 * sim.n_obs)
        est_task += 1 << 20
        est_outcome = 8 * (6 * sim.n_obs + 16)
        if sim.stderr_mode == "pathwise_iid":
            est_outcome += 8 * batch_size
        return est_task, est_outcome

    @staticmethod
    def _three_streams(clone):
        from quantark.util.enum.engine_enums import HestonMCScheme
        return clone.scheme in (
            HestonMCScheme.QUADEXP, HestonMCScheme.QUADEXP_M,
        ) or clone.fixed_three_stream_sobol

    def _attach_draw_provider(self, clone, context):
        provider = super()._attach_draw_provider(clone, context)
        if provider is not None:
            clone._uniform_provider = _UniformDrawProvider(
                getattr(context, "draw_repository"), clone.seed
            )
        return provider

    def prepare(self, engine, request, context) -> PreparedState:
        clone = self._clone_engine(engine)
        # Heston market state lives on model_params, not the env vol
        # surface; capture its fingerprint so reduce_batches can verify it.
        provider = self._attach_draw_provider(clone, context)
        sim = self._sim_context(clone, request, context,
                                draw_provider=provider)
        handles = (provider,) if provider is not None else ()
        return PreparedState(
            payload=sim, descriptors=(), fingerprint=None,
            byte_estimate=None, handles=handles,
        )


class DCNHestonBatchMCAdapter(_HestonBatchBase):
    def _clone_engine(self, engine):
        return type(engine)(
            model_params=engine.model_params,
            substeps_per_interval=engine.substeps_per_interval,
            scheme=engine.scheme,
            fixed_three_stream_sobol=engine.fixed_three_stream_sobol,
            num_paths=engine.num_paths, seed=engine.seed,
            use_sobol=engine.use_sobol,
            use_antithetic=engine.use_antithetic,
            num_batches=engine.num_batches,
            num_workers=1,
        )


class DCNQEBatchMCAdapter(_HestonBatchBase):
    def _clone_engine(self, engine):
        return type(engine)(
            model_params=engine.model_params,
            martingale_correction=engine.martingale_correction,
            substeps_per_interval=engine.substeps_per_interval,
            fixed_three_stream_sobol=engine.fixed_three_stream_sobol,
            num_paths=engine.num_paths, seed=engine.seed,
            use_sobol=engine.use_sobol,
            use_antithetic=engine.use_antithetic,
            num_batches=engine.num_batches,
            num_workers=1,
        )
```

  Model-params verification: extend `_sim_context` (mixin) to also fingerprint `getattr(clone, "model_params", None)` into `product_fp`'s companion field `model_fp` on `_DCNSimContext` (new optional field, default None) and check it in `_verify_captured_inputs` (fingerprint changed → `DeterminismViolation`). GBM/LV engines have no `model_params` → field stays None, behavior unchanged.
- Registry: replace the two legacy Heston pins with exact adapter registrations and add the QE one:

```python
    # Phase 3: Heston DCN fixed-batch adapters. exact=True (fixed
    # constructor signatures); unknown stateful subclasses fall through
    # the MRO to the legacy adapter. CoupledCoarse gets its own pair-aware
    # adapter (also exact).
    registry.register(
        "quantark.asset.equity.engine.mc.dcn_vol_mc_engines.HestonDCNMCEngine",
        _dcn_heston_batch_adapter, exact=True,
    )
    registry.register(
        "quantark.asset.equity.engine.mc.dcn_vol_mc_engines.QEDCNMCEngine",
        _dcn_qe_batch_adapter, exact=True,
    )
```

- [ ] **Step 1: Write the failing tests** — extend `test/execution/test_heston_dcn_batch.py`:

```python
def _dcn_product():
    # copy the DCN product fixture from test/execution/test_dcn_batch_adapter.py
    ...

@pytest.mark.parametrize("scheme", ["FULL_TRUNCATION_EULER", "EULERLOG",
                                    "QUADEXP", "QUADEXP_M"])
@pytest.mark.parametrize("mode", ["sobol_multi", "sobol_single", "pseudo_iid"])
def test_heston_session_bitwise_vs_direct(scheme, mode):
    # engine config per mode: sobol_multi = use_sobol/num_batches=4;
    # sobol_single = use_sobol/num_batches=1; pseudo_iid = use_sobol=False.
    # direct: engine.price_detailed(product, env)
    # session: PricingSession().execute(..., PRICE_DETAILED)
    # assert result.pv, result.std_error and every leg/event field bitwise
    # equal (reuse the comparison helper from test_dcn_batch_adapter.py).
    ...

def test_qe_engine_bitwise_and_uniform_draws_from_repo():
    # QEDCNMCEngine sobol num_batches=4: session == direct bitwise AND the
    # session diagnostics draws: records show uniform-block builds == 4 on
    # first dispatch and all hits on an identical second dispatch (CRN).
    ...

def test_heston_threads_backend_bitwise():
    # threads backend, workers=4: bitwise equal to serial session result
    ...

def test_unknown_heston_subclass_falls_to_legacy():
    class _Tweaked(HestonDCNMCEngine):
        pass
    # registry.resolve(_Tweaked instance).capabilities().adapter_id == "legacy-price"
    ...

def test_model_params_mutation_fails_closed():
    # mutate engine.model_params fields between prepare and reduce via a
    # wrapped execute - or simpler: monkeypatch sim.captured verification by
    # replacing env.rate_curve mid-run as in the existing mixed-market test,
    # plus a direct unit call of _verify_captured_inputs with a changed
    # model fingerprint asserting DeterminismViolation.
    ...
```

Fill every `...` with concrete code copied/adapted from `test/execution/test_dcn_batch_adapter.py`'s existing fixtures and helpers (product fixture, env fixture, session factory, bitwise comparison helper) — same conventions, same helper names.

- [ ] **Step 2: Verify failures** (adapter classes missing; registry resolves legacy).
- [ ] **Step 3: Implement** as specified in Interfaces (adapters + `_sim_context`/`_verify_captured_inputs` model_fp extension + registry + lazy factories `_dcn_heston_batch_adapter`/`_dcn_qe_batch_adapter`). Update `__all__` in `dcn_execution_adapters.py` and the module docstring (Heston paragraph replaces the "NOT inherited ... Phase 3" note on `DCNBatchMCAdapter`).
- [ ] **Step 4: Run** `test_heston_dcn_batch.py`, `test_dcn_batch_adapter.py`, `test_registry.py` (update expected set: +2 paths here), `test_regression_gates.py` — PASS.
- [ ] **Step 5: Commit** `feat(execution): Heston/QE DCN fixed-batch adapters with repository-routed draws`.

---

### Task 11: CoupledCoarse Heston DCN batch adapter

**Files:**
- Modify: `quantark/asset/equity/engine/mc/dcn_execution_adapters.py`, `quantark/execution/registry.py`
- Test: `test/execution/test_heston_dcn_batch.py` (extend), `test/execution/test_registry.py`

**Interfaces:**
- Produces: `DCNCoupledCoarseHestonBatchMCAdapter(_HestonBatchBase)` — clones the PAIR:

```python
class DCNCoupledCoarseHestonBatchMCAdapter(_HestonBatchBase):
    """Pair-aware clone: the coarse engine derives every draw from its fine
    partner, so the fine engine is cloned first (it owns the draw streams
    and gets the session providers) and the coarse clone is built around
    it. ``model_params`` is passed as the SAME object to both clones (the
    coupled-pair constructor enforces identity)."""

    def _clone_engine(self, engine):
        fine = engine._fine_engine
        fine_clone = type(fine)(
            model_params=engine.model_params,
            substeps_per_interval=fine.substeps_per_interval,
            scheme=fine.scheme,
            fixed_three_stream_sobol=fine.fixed_three_stream_sobol,
            num_paths=fine.num_paths, seed=fine.seed,
            use_sobol=fine.use_sobol, use_antithetic=fine.use_antithetic,
            num_batches=fine.num_batches, num_workers=1,
        )
        return type(engine)(
            fine_engine=fine_clone,
            model_params=engine.model_params,
            substeps_per_interval=engine.substeps_per_interval,
            scheme=engine.scheme,
            fixed_three_stream_sobol=engine.fixed_three_stream_sobol,
            num_paths=engine.num_paths, seed=engine.seed,
            use_sobol=engine.use_sobol,
            use_antithetic=engine.use_antithetic,
            num_batches=engine.num_batches, num_workers=1,
        )

    def _attach_draw_provider(self, clone, context):
        # draws are generated by the FINE partner; hook the providers there
        provider = _DCNBatchMixin._attach_draw_provider(
            self, clone._fine_engine, context
        )
        if provider is not None:
            clone._fine_engine._uniform_provider = _UniformDrawProvider(
                getattr(context, "draw_repository"), clone.seed
            )
        return provider

    def _plan_dimension(self, sim, time_steps):
        clone = sim.engine
        n_fine_partner = (
            time_steps * clone._fine_engine.substeps_per_interval
        )
        return (
            3 * n_fine_partner
            if self._three_streams(clone) else 2 * n_fine_partner
        )

    def _plan_est_bytes(self, sim, clone, batch_size, time_steps):
        # the fine partner's draw block (2x substeps) dominates
        task, outcome = super()._plan_est_bytes(
            sim, clone._fine_engine, batch_size, time_steps
        )
        return task, outcome
```

- Registry: replace the CoupledCoarse legacy pin with `exact=True` registration to `_dcn_coupled_batch_adapter`.

- [ ] **Step 1: Write failing tests** — extend `test_heston_dcn_batch.py`:

```python
@pytest.mark.parametrize("scheme", ["FULL_TRUNCATION_EULER", "QUADEXP_M"])
def test_coupled_coarse_session_bitwise_vs_direct(scheme):
    # coarse, fine = coupled_heston_ladder_pair(_heston_params(), 1, scheme,
    #                num_paths=512, seed=7, use_sobol=True, num_batches=2)
    # direct coarse.price_detailed vs session on the coarse engine: bitwise.
    ...

def test_coupled_pair_ladder_difference_preserved():
    # session-priced coarse PV minus session-priced fine PV equals the
    # direct pair difference bitwise (the MLMC coupling survives the
    # session path).
    ...
```

- [ ] **Step 2: Verify failure**, **Step 3: implement** (adapter + registry + factory + `__all__`), **Step 4: run** `test_heston_dcn_batch.py` + `test_registry.py` (+1 expected path) — PASS.
- [ ] **Step 5: Commit** `feat(execution): pair-aware CoupledCoarse Heston DCN batch adapter`.

---

### Task 12: Inventory adaptive states + validation profiles + gates

**Files:**
- Modify: `quantark/execution/inventory.py`, `quantark/execution/__init__.py`
- Test: `test/execution/test_inventory.py`

**Interfaces:**
- `InventoryRecord` gains: `adaptive_state: str = "not_applicable"`, `adaptive_rationale: str = _ADAPTIVE_NON_RQMC`, `validation_profile: str = ""`. New module constants:

```python
ADAPTIVE_STATES = ("adaptive_capable", "temporary_legacy", "not_applicable")

_PROFILE_BITWISE = "bitwise-vs-direct/v1"
_ADAPTIVE_NON_RQMC = "engine has no adaptive RQMC stopping mode"
_ADAPTIVE_DEFERRED = (
    "sequential run_rqmc mode exists but session adaptive planning is "
    "deferred with the single-solve batch scope (spec 21 re-scope on "
    "Phase 2 evidence)"
)
```

- Row updates:
  - 12 Snowball/Phoenix rows: `adaptive_state="adaptive_capable"`, `adaptive_rationale=""`, `validation_profile=_PROFILE_BITWISE`, `adoption_state="supported"`; `batch_rationale` updated to note the parallel-wave deferral: `"adaptive RQMC compatibility stopping is sequential by contract (spec 8.4); parallel-wave deferred on Phase 2 benchmark evidence (host-limited thread scaling)"` (update the `_BATCH_ADAPTIVE` constant text once).
  - `DCNMCEngine`/`LocalVolDCNMCEngine` (already batch_capable): `validation_profile=_PROFILE_BITWISE`.
  - `HestonDCNMCEngine`/`QEDCNMCEngine`/`CoupledCoarseHestonDCNMCEngine`: `batch_state="batch_capable"`, `batch_rationale=""`, `adoption_state="supported"`, `validation_profile=_PROFILE_BITWISE`. (`QEDCNMCEngine` has no inventory row today — it is exported; verify with `discover_exported_engine_names` and add the row if the discovery gate demands it, mirroring the Heston row.)
  - The 8 `run_rqmc`-consuming single-solve equity rows (`EuropeanMCEngine`, `AsianOptionMCEngine`, `DigitalOptionMCEngine`, `BarrierOptionMCEngine`, `AmericanOptionMCEngine`, `SingleSharkfinOptionMCEngine`, `DoubleSharkfinOptionMCEngine`, `RangeAccrualMCEngine`): `adaptive_state="temporary_legacy"`, `adaptive_rationale=_ADAPTIVE_DEFERRED`.
  - Everything else: defaults.
- Gates (extend `test_inventory.py`):

```python
def test_adaptive_states_are_valid():
    for rec in ENGINE_INVENTORY:
        assert rec.adaptive_state in ADAPTIVE_STATES

def test_adaptive_capable_rows_declare_profile_and_no_rationale():
    for rec in ENGINE_INVENTORY:
        if rec.adaptive_state == "adaptive_capable":
            assert rec.validation_profile
            assert rec.adaptive_rationale == ""

def test_adaptive_temporary_legacy_requires_rationale():
    for rec in ENGINE_INVENTORY:
        if rec.adaptive_state == "temporary_legacy":
            assert rec.adaptive_rationale

def test_batch_capable_rows_declare_profile():
    for rec in ENGINE_INVENTORY:
        if rec.batch_state == "batch_capable":
            assert rec.validation_profile

def test_autocallable_rows_are_adaptive_capable():
    names = {r.name for r in ENGINE_INVENTORY
             if r.adaptive_state == "adaptive_capable"}
    assert names == {
        "SnowballMCEngine", "LocalVolSnowballMCEngine",
        "HestonSnowballMCEngine", "QESnowballMCEngine",
        "HestonSLVSnowballMCEngine", "HestonSLVQESnowballMCEngine",
        "PhoenixMCEngine", "LocalVolPhoenixMCEngine",
        "HestonPhoenixMCEngine", "QEPhoenixMCEngine",
        "HestonSLVPhoenixMCEngine", "HestonSLVQEPhoenixMCEngine",
    }

def test_heston_dcn_rows_are_batch_capable():
    for name in ("HestonDCNMCEngine", "QEDCNMCEngine",
                 "CoupledCoarseHestonDCNMCEngine"):
        rec = inventory_by_name()[name]
        assert rec.batch_state == "batch_capable"
        assert rec.validation_profile
```

Also update the reachability/adapter-id assertions in `test_inventory.py` to accept `"autocallable-adaptive-mc"` and remove any Heston-pin expectations, and export `ADAPTIVE_STATES` from `inventory.py.__all__`. Update `quantark/execution/__init__.py` docstring to "Phases 0-3".

- [ ] **Step 1: write failing gates → Step 2: verify → Step 3: implement rows → Step 4: run** `test_inventory.py` + `test/execution` — PASS. **Step 5: Commit** `feat(execution): inventory adaptive states + validation profiles (Phase 3 exit gate)`.

---

### Task 13: Full-suite verification

- [ ] **Step 1:** `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -q`
Expected: everything green except the known pre-existing `test_snowball_quad_flat_identity_golden`.
- [ ] **Step 2:** Fix any fallout (session-parity tests asserting `legacy-price` for snowball/phoenix engines must accept the new adapter id; any test constructing engines through the matrix in RQMC configs).
- [ ] **Step 3: Commit** any test-expectation updates: `test(execution): Phase 3 suite alignment`.

## Plan-Gate Findings Applied (Codex, 2026-07-16, 1 iteration)

1. **[high] Live-engine race**: per-engine lock acquired in `prepare`, released through a `PreparedState.handles` entry the kernel closes in `finally`; concurrency regression test added (Task 7).
2. **[high] Scratch lease vs finalize's statistics batch**: the task-scratch lease now spans `run_rqmc_traced` + `finalize` + `_complete_price`; estimate sized for the extra batch.
3. **[medium] Lossy trace evidence**: `adaptive:trace_fingerprint=<try_fingerprint(trace)>` recorded in diagnostics; determinism tests assert fingerprint equality and single-value sensitivity.

## Self-Review Notes

- Spec §21 Phase 3 coverage: sequential compatibility stopping preserved (Tasks 1-7, bit-identical); parallel-wave explicitly DEFERRED with evidence recorded in the inventory rationale (kickoff decision under spec §21 re-scope, documented in Task 12); model families: Heston/QE/Coupled DCN adapted (Tasks 9-11), Snowball/Phoenix families all covered via subclass-safe registration (Task 6), remaining families carry explicit deferral rationales (Task 12). Exit gate: deterministic checkpoint tests (Tasks 1, 7) + declared validation profiles (Task 12).
- Type consistency: `RQMCRunSpec` fields consumed by Task 5 (`pricer_fn`, `path_generator`, controls, `paths_per_batch`, `time_steps`, `scheme`, `finalize`, `product`) all exist in Task 1's definition; Tasks 2-3 populate every field including `product`.
- `lease_bytes(n, pool)` argument order MUST be verified against `leases.py` before Task 5 lands.
