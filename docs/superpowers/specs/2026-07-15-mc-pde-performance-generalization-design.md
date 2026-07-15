# QuantArk Composable Execution Kernel for MC and PDE

**Date:** 2026-07-15

**Status:** Accepted design; ready for implementation planning

**Framework contract:** v1

**Reviewed QuantArk baseline:** `ed0863f`, including performance commit `318009e`

**Supersedes:** the opt-in utility design introduced by `8496c7d`

## 1. Decision

QuantArk will generalize the DCN performance work and the solution-side
scenario runner as a **composable execution kernel with capability adapters**.
This is a framework update, not a set of DCN helpers and not an optional
adoption exercise.

The kernel belongs in a new top-level `quantark.execution` package. Every
exported Monte Carlo or PDE engine across equity, FX, credit, and bond must be
reachable through it. Engines keep
ownership of their numerical algorithms; adapters expose preparation,
batching, reusable artifacts, process reconstruction, result normalization,
and resource estimates to a common execution lifecycle.

Framework v1 removes no API and changes no historical default. Existing
constructors, `price`, `price_detailed`, `calculate_event_stats`, parameter
dataclasses, result types, Dask flags, environment variables, and exception
behavior remain supported.

## 2. What should persist from the reviewed implementation

The reviewed changes demonstrate useful mechanisms, but several of their
current scopes and ownership choices must not become the framework contract.

| Reviewed measure | Decision | Framework form | Do not preserve literally |
|---|---|---|---|
| Byte-budgeted Sobol draw reuse | Persist | Session-aware `DrawRepository` with complete descriptors, immutable masters, single-flight misses, and a resource lease | Incomplete keys or an uncoordinated process-global cache |
| Threaded DCN MC batches | Persist for fixed-batch MC | Immutable `BatchPlan`, bounded executor, compact batch outcomes, canonical incremental reduction | Engine-local `ThreadPoolExecutor` loops or `list(pool.map(...))` |
| Once-per-call Dupire/local-vol construction | Persist and broaden | Immutable `PreparedState` plus value-fingerprinted `PreparedArtifactCache` | Mutable engine fields such as `_active_surface`, or environment identity as a cache key |
| In-place uniform-to-normal conversion | Persist conditionally | A transform step operating only on a leased writable scratch buffer | Mutating a cached master or a caller-owned read-only block |
| Rich DCN/PDE result and event-stat extraction | Persist and broaden | Requested-output bundles that allow one numerical solve to populate PV, error, event stats, cashflows, and reusable surfaces | Calling separate public methods when that repeats the same solve |
| Process-parallel surface shock cells | Persist | Typed `ScenarioSpec`/`ScenarioPlan`, importable `WorkerSpec`, child budgets, complete result normalization | Mutable worker globals, closure-based tasks, or environment variables that bypass policy |
| Process workers plus inner MC threads | Supported only explicitly | Nested execution after parent-level worker and memory leases are reserved | Nested parallelism as a default |
| Exact checkpoint comparison | Persist and strengthen | Comparison of the complete normalized economic payload, with scenario counts and field counts reported separately | Treating five selected fields as the whole result, or labeling 130 field comparisons as 130 cells |

### 2.1 Evidence and limitations

The solution benchmark recorded 26 independent cells in about 271 seconds
versus 1,417 seconds of recorded serial work, and the DCN benchmark recorded
roughly 3.4–3.6x fixed-batch MC speedups at eight threads. These are strong
signals that batch and scenario parallelism are worth productizing.

They are not yet framework acceptance evidence:

- the scenario serial timing predates the engine-level cache, thread, and
  local-vol-hoist changes, so the reported 5.2x is not an apples-to-apples
  backend comparison;
- the MC headline combines cache reuse and thread execution, so isolated
  attribution is required;
- `QUANTARK_QMC_CACHE_MB` currently defaults to 2 GiB per process; six scenario
  workers can therefore admit up to 12 GiB of cache before path, surface, and
  result memory;
- DCN batch execution first materializes every batch cashflow result and only
  then reduces it;
- the solution equality flag covers five fields, while a cell contains more
  economic and numerical output;
- the current local-vol reuse pattern is request-local mutable engine state,
  which is unsafe as a general concurrent-session abstraction.

The implementation must re-benchmark isolated mechanisms and total resource
use before making production-wide speedup claims.

## 3. Goals, non-goals, and invariants

### 3.1 Goals

1. Give every exported QuantArk MC and PDE engine a common, observable,
   resource-bounded execution path.
2. Make preparation reuse, deterministic batch execution, draw reuse, outer
   scenario parallelism, and portfolio grouping reusable capabilities.
3. Preserve exact legacy behavior when the numerical plan is unchanged.
4. Define explicit numerical equivalence when a user selects a different
   arithmetic order, adaptive schedule, grid, or backend.
5. Prevent worker and cache multiplication from exhausting a host.
6. Support serial, threads, local processes, and existing Dask execution from
   the same immutable plans and reducers.
7. Make performance, cache behavior, fallbacks, and reproducibility auditable.

### 3.2 Non-goals for framework v1

- Rewriting product payoff algorithms or model dynamics.
- Parallelizing a PDE backward time march. The march remains engine-owned and
  serial; PDE acceleration comes from preparation reuse and outer requests.
- Replacing `MCParams`, `PDEParams`, or product-specific result classes.
- Introducing a mandatory `BatchedMCEngine` or PDE inheritance hierarchy.
- Caching stateful pseudorandom streams.
- Silently changing an engine's numerical plan to obtain speed.
- Guaranteeing bit identity across different NumPy, SciPy, BLAS, solver, or
  QuantArk implementation versions.
- Removing legacy Dask or DCN settings in v1.

### 3.3 Hard invariants

- `BaseEngine.price(product, pricing_env)` retains its signature.
- Direct legacy calls retain result types, default values, warnings, and
  exception types.
- No mutable active run context is stored globally or in thread-local state.
- Engine instances do not store request-scoped prepared state.
- Cached values are immutable; mutation requires an explicit writable copy.
- Every plan has a stable fingerprint and a canonical output order.
- Explicit new-framework backend requests never silently fall back. A declared
  fallback order is explicit and diagnosed. Direct legacy Dask calls retain
  their historical warning/fallback behavior.
- Nested parallelism is disabled unless explicitly selected and budgeted.
- Invalid or failed partial outcomes never enter aggregates.
- Operational metadata never participates in economic equality.

## 4. Architecture

### 4.1 Package ownership

```text
quantark/execution/
    __init__.py              # stable public exports
    api.py                   # PricingSession and convenience methods
    context.py               # PricingRunContext and child contexts
    contracts.py             # requests, outcomes, plans, capabilities
    policy.py                # execution, resource, determinism policies
    kernel.py                # canonical lifecycle
    registry.py              # adapter/factory/normalizer registries
    errors.py                # framework exceptions
    diagnostics.py           # operational records and sinks
    manifest.py              # reproducibility manifest
    inventory.py             # checked-in exported-engine capability matrix
    cache/
        draws.py             # DrawRepository
        artifacts.py         # PreparedArtifactCache
        fingerprint.py       # canonical value fingerprints
    backends/
        serial.py
        threads.py
        processes.py
        dask.py
    scenario/
        contracts.py         # ScenarioSpec, ScenarioPlan, ScenarioOutcome
        planner.py
        worker.py             # spawn-safe WorkerSpec execution
```

MC- and PDE-specific adapters remain near their engine families or in thin
adapter modules; the kernel may not import equity, FX, credit, bond, or product
code. Invocation adapters normalize legacy call shapes: equity and FX engines
generally accept `(product, pricing_env)`, while some convertible-bond engines
bind the environment at construction and accept `price(product)`. The
framework adapts both without changing either direct API.

### 4.2 Dependency direction

```text
PricingSession
      |
      v
ExecutionKernel --> policies / budgets / caches / diagnostics
      |
      v
capability adapter protocols
      |
      v
product and engine family adapters --> existing numerical engines
```

The kernel owns orchestration. Engines own numerical meaning. Backends own
scheduling only. Caches own bytes only; they do not decide numerical plans.

## 5. Public contracts

The initial public surface is additive:

```python
from quantark.execution import (
    DeterminismPolicy,
    EngineCapabilities,
    ExecutionPolicy,
    PricingFailure,
    PricingOutcome,
    PricingRequest,
    PricingRunContext,
    PricingSession,
    ResourceBudget,
    ScenarioOutcome,
    ScenarioSpec,
)
```

### 5.1 Requests and outcomes

```python
@dataclass(frozen=True)
class PricingRequest:
    product: object
    pricing_env: object
    operation: PricingOperation = PricingOperation.PRICE
    outputs: frozenset[OutputKind] = frozenset({OutputKind.PV})
    operation_options: FrozenMapping[str, NormalizedValue] = EMPTY_MAPPING
    request_id: str | None = None

@dataclass(frozen=True)
class PricingOutcome(Generic[T]):
    value: T
    normalized_economics: Mapping[str, NormalizedValue]
    diagnostics: RunDiagnostics
    manifest: ReproducibilityManifest

@dataclass(frozen=True)
class PricingFailure:
    item_id: str
    error: FrameworkErrorInfo
    diagnostics: RunDiagnostics
    manifest: ReproducibilityManifest | None = None
```

`PricingOperation` includes at least `PRICE`, `PRICE_DETAILED`, and
`EVENT_STATS`. `OutputKind` includes PV, standard error or numerical error,
event statistics, cashflow decomposition, and an engine-owned reusable grid
or surface where supported.

`operation_options` carries validated native options such as event-stream
pruning; an adapter declares its option schema and rejects unknown options.
`request_id` is operational correlation data and is excluded from economic and
plan fingerprints.

`PricingRequest` is a shallow frozen envelope around potentially mutable legacy
objects. Before any cache lookup or parallel submission, the adapter produces
an immutable `NormalizedPricingRequest` snapshot containing every economic and
numerical input consumed by preparation or execution. Raw mutable product and
environment objects are not cache truth and are not submitted across
threads/processes. If an adapter cannot safely snapshot an input, the request
is uncacheable and the adapter verifies its value fingerprint before and after
execution; concurrent mutation raises `DeterminismViolation`.

`value` is the native legacy value. The normalized mapping is an immutable
canonical tree used for comparison, scenario aggregation, and persistence; it
is not a replacement for existing result classes.

A `PricingOutcome` always represents success and contains a valid native value.
Collection APIs use `ExecutionItemOutcome = PricingOutcome[T] | PricingFailure`.
A failure carries identity, typed error data, diagnostics, and any manifest
available before failure, but no economic value or partial normalized result.

### 5.2 Policy objects

`ExecutionPolicy` has separate batch and scenario executor selections. Each
selection records backend (`serial`, `threads`, `processes`, or `dask`), worker
limit, in-flight limit, whether concurrency may shrink under memory pressure,
and any explicit fallback order. The policy also records nested-execution
mode, fail-fast versus collect-errors mode, retry policy, and cancellation
behavior. Defaults are serial, one worker, no nested execution, fail-fast, no
retry, and no fallback.

`DeterminismPolicy` records the required equal-plan behavior, the validation
profile for a changed plan, whether a manifest is mandatory, and whether a
mismatch raises or is returned as a `PricingFailure` in a collection. It never
selects a faster numerical plan by itself.

`ResourceBudget` is the immutable admission limit described in Section 11.
Policies describe intent; the resolved context records the effective settings
after capability and resource validation.

### 5.3 Run context

`PricingRunContext` is an immutable value object containing:

- resolved `ExecutionPolicy`;
- resolved `ResourceBudget`;
- `DeterminismPolicy`;
- stable handles to draw/artifact repositories and the resource lease manager;
- cancellation token and diagnostics sink;
- run ID, parent run ID, and immutable configuration snapshot.

Repositories and lease managers are mutable services behind stable handles;
the context itself never mutates. `context.child(...)` returns a new context
with a child scope and explicit budget lease.

Service ownership is explicit. A session owns and closes repositories,
executors, and lease managers it creates. Externally supplied services are
borrowed unless ownership is explicitly transferred. Shared services require
a common budget domain and reference-counted lifetime.

### 5.4 Base engine compatibility entry point

Add a non-abstract method without changing the abstract `price` contract:

```python
class BaseEngine(ABC):
    def execute(
        self,
        request: PricingRequest,
        context: PricingRunContext,
    ) -> PricingOutcome:
        return ExecutionKernel.dispatch(self, request, context)
```

Existing subclasses need no immediate source change. Specialized adapters are
selected by the registry before the default adapter. The kernel invokes the
selected adapter directly and never recursively calls `engine.execute`.
`LegacyPriceAdapter` is the final fallback and delegates to the existing
`price`/requested legacy method.

`BaseEngine` keeps both its canonical
`quantark.asset.equity.engine.BaseEngine` identity and the supported legacy
`asset.equity.engine.BaseEngine` identity. `execute` is the only new method in
this change and cannot be abstract. Existing defaults such as
`create_bump_context() -> self`, unsupported `calculate_event_stats() -> None`,
and inherited Greek/spot-curve fallbacks remain intact.

Engines that do not inherit equity `BaseEngine` use a registered structural
invocation adapter. They are not forced into the equity class hierarchy.

### 5.5 Session API

```python
with PricingSession(context) as session:
    outcome = session.execute(engine, request)
    pv = session.price(engine, product, pricing_env)
    values = session.price_many(items)
    scenarios = session.run_scenarios(
        base_request, scenario_specs, engine_factory
    )
```

- `execute` returns `PricingOutcome`.
- `price` unwraps the same native value returned by the legacy `price` method.
- `price_many` preserves caller order and native value types.
- `run_scenarios` returns ordered `ScenarioOutcome` values.
- The session owns only services it created and is idempotently closable.
  Shutdown joins executors and releases task/cache handles before closing owned
  repositories; borrowed services remain open.

## 6. Capability adapters

Capabilities are structural protocols, not mandatory base classes:

```python
class PreparedExecutionCapability(Protocol):
    def prepare(
        self, request: PricingRequest, context: PricingRunContext
    ) -> PreparedState: ...

class BatchExecutionCapability(Protocol):
    def plan_batches(
        self, request: PricingRequest, state: PreparedState,
        context: PricingRunContext,
    ) -> BatchPlan: ...

    def execute_batch(
        self, task: BatchTask, state: PreparedState,
        context: PricingRunContext,
    ) -> BatchOutcome: ...

    def reduce_batches(
        self, outcomes: Iterable[BatchOutcome], plan: BatchPlan,
        state: PreparedState, context: PricingRunContext,
    ) -> PricingOutcome: ...

class ReusableArtifactCapability(Protocol):
    def artifact_descriptors(
        self, request: PricingRequest, context: PricingRunContext
    ) -> tuple[ArtifactDescriptor, ...]: ...

class ProcessExecutionCapability(Protocol):
    def to_worker_spec(
        self, request: PricingRequest, context: PricingRunContext
    ) -> WorkerSpec: ...
```

### 6.1 Capability descriptor

`EngineCapabilities` is immutable and records:

- supported operations and output bundles;
- preparation, batch, artifact, and process capabilities;
- supported backends;
- fixed or adaptive numerical planning;
- prepared-state thread safety, engine-instance reentrancy, and process
  reconstruction support;
- deterministic-reduction guarantees;
- whether peak-memory estimates are exact, conservative, or unavailable;
- adapter and normalizer IDs plus implementation versions.

This plural execution descriptor is distinct from the existing singular
`quantark.asset.equity.engine.EngineCapability` model/engine support registry.
That existing type, import path, and behavior remain unchanged.

### 6.2 Adapter resolution

Resolution order is deterministic:

1. an exact engine-class registration;
2. direct structural capability implementation on the concrete class;
3. the nearest unambiguous registered base class;
4. `BaseEngine.execute` through `LegacyPriceAdapter`.

Ambiguous registrations fail at session construction. Registries freeze for
the lifetime of a session. Process-capable adapter and factory IDs must resolve
from importable modules in a fresh spawn interpreter.

Capabilities are validated before preparation. An explicit unsupported
backend or output raises `CapabilityError`; the kernel does not substitute a
different backend, output, or numerical plan.

### 6.3 Prepared state

`PreparedState` is immutable and request-scoped. It contains engine-specific
payloads or immutable artifact handles, their dependency tags, byte estimates,
and a preparation fingerprint. It must not contain a mutable live engine
context that another request can overwrite.

Consequently, adapters replace patterns such as `_active_surface`, `_term_ctx`,
and `_df` with local immutable state passed to batch or solve methods.

## 7. Canonical kernel lifecycle

Every backend follows the same state machine:

1. **Normalize** the request into an immutable snapshot without mutating caller
   objects.
2. **Resolve** policy field by field and snapshot environment/configuration.
3. **Select and validate** the adapter, capabilities, output bundle, and
   backend.
4. **Estimate and reserve** worker, memory, cache, and in-flight leases.
5. **Prepare** immutable state, using safe artifact-cache hits where possible.
6. **Plan** the exact numerical work and fingerprint the plan.
7. **Execute** with bounded submission and child contexts.
8. **Reduce** in canonical order, incrementally.
9. **Normalize and validate** economic output under the determinism policy.
10. **Emit** diagnostics and the reproducibility manifest.
11. **Release** all leases in `finally` blocks.

The numerical plan is completed before backend submission. A backend may
schedule tasks differently but may not reinterpret or repartition the plan.

## 8. Monte Carlo execution contract

### 8.1 Batch plan

An immutable `BatchPlan` includes:

- stable plan and batch IDs;
- requested and effective path counts per batch;
- seed, scramble, antithetic, and stream-layout descriptors;
- time grid, stochastic dimension, dtype, model scheme, and transform plan;
- canonical reduction order;
- adaptive stopping policy, if any;
- estimated peak bytes per task and compact outcome bytes;
- implementation and numerical dependency fingerprints.

The plan, not executor completion order, determines economics.

### 8.2 Batch outcome and reduction

`execute_batch` returns compact sufficient statistics whenever possible:
sums, sum-of-squares, counts, event vectors, and independent scramble means.
It must not return full paths merely to perform a later scalar reduction.

The executor submits at most the leased in-flight count. Completed results are
buffered only until the next canonical batch ID is available, then passed to
`reduce_batches`. Memory is therefore bounded by active tasks plus a bounded
number of compact outcomes.

Reducers preserve existing arithmetic order when the compatibility plan is
selected. Where exact legacy accumulation requires a particular grouping,
that grouping is part of the plan.

### 8.3 Draw repository and transforms

`DrawRepository` is MC-specific. A cacheable `DrawDescriptor` includes:

- generator family, implementation ID, and implementation version;
- distribution and sequence layout;
- base seed, scramble configuration, and batch ID;
- path count, dimension, shape, memory order, and dtype;
- antithetic or stream-composition rules;
- transform pipeline and version;
- generator-specific options and relevant NumPy/SciPy versions.

A descriptor that cannot fully identify generated bytes is not cacheable.
Stateful pseudorandom streams are not cached.

Cached master arrays are read-only. `writable=False` may return the master;
`writable=True` returns an explicit copy charged to the task's memory lease.
In-place inverse-CDF transformation is allowed only on that writable scratch
copy. Draw access uses the same lease-backed pinning contract as prepared
artifacts. Concurrent identical misses are single-flight.

CRN repricings reuse identical draw descriptors only when time grid,
dimension, model scheme, and stream layout remain identical. A changed
descriptor is a new numerical plan, never a nominal cache miss hidden under
the same key.

### 8.4 Fixed and adaptive execution

- Fixed-batch MC may run serially or in threads while retaining the same plan
  and canonical reduction.
- Existing adaptive RQMC compatibility mode remains sequential and evaluates
  its stopping criterion after the same batches in the same order.
- Parallel-wave adaptive RQMC is a distinct opt-in numerical plan. Wave size,
  batch IDs, stop checkpoints, and reduction order are fingerprinted. The
  criterion is evaluated only after a complete wave. It requires paired-error
  qualification and may not replace compatibility mode silently.

### 8.5 Single-solve output bundles

Adapters should request all required outputs before simulation. If a product
can produce PV, standard error, cashflow legs, and event statistics from one
path traversal, it does so once and projects the result into legacy methods.
The kernel must not independently call `price`, `price_detailed`, and
`calculate_event_stats` for one request.

## 9. PDE execution contract

### 9.1 Ownership

The PDE adapter participates fully in normalization, preparation, caching,
resource accounting, diagnostics, result normalization, and outer scenario
execution. The engine continues to own:

- spatial and time discretization semantics;
- boundary and event conditions;
- operator construction;
- the backward march and interpolation;
- engine-specific convergence meaning.

Framework v1 does not parallelize individual time steps.

### 9.2 Cacheable preparation

Subject to complete dependency fingerprints, prepared PDE artifacts may
include:

- spatial grids and event-aligned time grids;
- product schedule/event maps;
- rate, dividend, local-vol, Heston, or SLV term contexts;
- local-vol inversions and leverage-surface snapshots;
- static operator coefficients;
- symbolic or numeric factorization metadata;
- output projection/interpolation maps.

Numeric factorization reuse is legal only when matrix coefficients, time step,
boundary conditions, scheme parameters, and relevant dependency versions are
identical. Time-dependent operators that cannot prove this reuse safe are
rebuilt.

### 9.3 Repeated requests and rich outputs

Outer spot grids, bump Greeks, event-stat requests, and scenario cells should
reuse safe preparation and, where an engine already exposes a full grid or
stacked event surfaces, derive multiple outputs from one solve. This extends
the intent of `calculate_event_stats` and `calculate_spot_greeks_curve`
without changing either public method.

## 10. Cache and fingerprint model

### 10.1 Prepared artifact cache

`PreparedArtifactCache` serves MC and PDE preparation. Keys use canonical
value fingerprints, never Python object identity or `repr` alone. A key
contains, as applicable:

- engine, adapter, model, and implementation versions;
- valuation date, calendars, day counts, and settlement conventions;
- normalized curve nodes and interpolation/extrapolation settings;
- normalized volatility quotes/surfaces and calibration settings;
- model parameters and numerical scheme;
- product economic terms and observation/payment schedules;
- spatial/time grids, tolerances, barriers, and boundary settings;
- dependency versions and explicit dependency tags.

If any required input lacks a safe canonicalizer, the artifact is marked
uncacheable. Hash collisions are defended by retaining canonical metadata for
equality verification on a hit.

Artifacts declare their byte size, dependency tags, and immutable payload.
Miss construction is single-flight. Eviction is LRU by bytes within the
leased cache capacity; eviction never changes the numerical plan.

Cache access returns a lease-backed immutable handle. The handle pins an entry
until its task or `PreparedState` releases it. Pinned bytes remain charged even
if the entry is removed from the lookup index; eviction reclaims capacity only
after the last handle closes. Writable copies hold independent task-memory
leases. Failure, cancellation, and session shutdown release all handles in
`finally` paths.

### 10.2 Invalidation tags

Standard mutation tags include `spot`, `valuation_date`, `rate_curve`,
`dividend_curve`, `vol_surface`, `model_params`, `product_terms`,
`schedule`, `calendar`, `grid`, and `solver_policy`.

Each artifact descriptor declares its dependencies. A scenario declares its
expected mutation footprint. The registered transformer schema declares its
allowed footprint, and the planner derives actual changed components by
comparing normalized before/after snapshots. Actual changes must be a subset
of the declaration; under-declaration raises `ValidationGateError`. If a safe
diff cannot be derived, preparation is conservatively invalidated in full.
Cache reuse never relies only on caller-supplied tags. Scenario names and
Python object identity do not participate.

## 11. Resource budget and scheduling

`ResourceBudget` is an immutable limit set covering:

- parent maximum processes and threads;
- total admitted working memory;
- draw-cache and prepared-artifact sub-budgets;
- maximum in-flight tasks and queued compact outcomes;
- optional per-task and per-worker ceilings;
- whether concurrency may shrink or serial fallback is permitted.

An internal lease manager accounts for workers, task scratch, prepared state,
cache bytes, and child contexts. Outer process workers receive explicit child
budgets; they do not independently re-read environment variables and recreate
the parent cache ceiling.

Before submission or allocation the kernel acquires a lease. Under pressure it
uses this order:

1. evict unpinned cache entries;
2. reduce in-flight work or worker count if policy permits;
3. use the explicitly permitted serial backend;
4. raise `ResourceBudgetExceeded`.

There is no implicit spill to disk in v1. A single task whose conservative
estimate exceeds the total budget fails before execution. Unknown estimates
force serialized admission and emit a diagnostic until an adapter supplies a
conservative estimate.

Admission control is the hard portable guarantee. Parent and worker RSS are
also sampled for diagnostics and overrun detection; v1 does not claim the
instantaneous hard enforcement of an operating-system cgroup on macOS.

### 11.1 Compatibility and explicit-session budgets

Direct legacy calls retain the current `QUANTARK_QMC_CACHE_MB` behavior and do
not gain a new preflight exception. An explicit `PricingSession` enforces its
parent budget. The legacy cache setting remains a requested ceiling but is
clamped by the session's cache and parent memory leases, with the clamp
reported in diagnostics.

`PricingSession()` with no context is serial by default and constructs a safe
auto budget once at session creation. The resolved byte and worker limits are
recorded in the manifest; machine discovery is never repeated inside workers.

## 12. Backends

All backends consume the same task plan and reducer.

### 12.1 Serial

Serial is the compatibility reference and historical default. It creates no
executor and is always required for an engine in the capability inventory.

### 12.2 Threads

Threads are allowed only when the adapter advertises thread-safe prepared
state and batch bodies. Engine-instance mutation during execution is
forbidden. Worker count is limited by tasks, capability, and resource leases.

An engine instance may execute concurrently only when its adapter advertises
instance reentrancy. Otherwise the framework uses a registered factory/clone,
serializes access through a shared weak-reference engine lease domain, or
rejects concurrent ownership with `CapabilityError`. Separate sessions may not
assume independent local locks coordinate the same borrowed instance. The
process-scoped lease registry contains ownership tokens only, never an active
run context or numerical state.

### 12.3 Local processes

Process execution requires an importable, versioned `WorkerSpec`. It contains
registered adapter/factory IDs, normalized constructor payloads, request or
scenario descriptors, child policy, child budget, and expected fingerprints.
It contains no closure, bound live engine, mutable environment, or worker
global dictionary.

The process backend is tested using Python's spawn start method, including on
macOS. Per-worker initialization may populate a child-local artifact cache,
but that cache is bounded by the child lease.

Before preparation, each process or Dask worker verifies worker/schema
versions, adapter and factory IDs, implementation and numerical dependency
fingerprints, dtype, and any solver/BLAS identity required by the determinism
policy. A mismatch fails before numerical execution with `CapabilityError` or
`DeterminismViolation`; it cannot silently downgrade the reproducibility
claim.

### 12.4 Dask

The Dask adapter translates the same `BatchPlan` or `ScenarioPlan` into Dask
tasks and feeds results through the same canonical reducer. It does not keep a
separate numerical implementation.

An explicit new-framework request for unavailable Dask raises
`CapabilityError`. Existing legacy `use_dask=True` behavior, including its
current availability warning/fallback semantics, remains unchanged in v1.
For Snowball/Phoenix, existing RQMC precedence over Dask, exact total-path
partitioning, empty-batch handling, and statistical-parity contract also
remain. The current legacy Dask path is not retroactively labeled
bit-identical. If migration changes its batch partition or reduction, it is a
changed plan and must pass the declared MC gate.

### 12.5 Nested execution

Child contexts are inner-serial by default. Nested processes/threads require
an explicit policy plus reserved parent leases. Worker-side environment
mutation cannot enable unbudgeted inner workers.

## 13. Scenario and portfolio execution

### 13.1 Typed scenario contract

```python
@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    transformer_id: str
    parameters: Mapping[str, NormalizedValue]
    mutation_tags: frozenset[str]
    required_capabilities: frozenset[str] = frozenset()
    validation_policy: ValidationPolicy | None = None
```

Scenario meaning is never parsed from `scenario_id`. The transformer is a
registered, importable pure function that returns a new request or immutable
input snapshot; it does not mutate the base request.

`mutation_tags` is a declared upper bound used for planning, not trusted proof
of what changed. Normalization verifies it against the actual snapshot diff as
specified in Section 10.2.

### 13.2 Planning

Normalization produces an immutable `ScenarioPlan` containing:

- unique stable scenario IDs and caller-order positions;
- base-request and transformation fingerprints;
- invalidation decisions and reusable artifact/draw descriptors;
- engine-factory and worker specifications;
- backend-independent grouping and output ordering;
- per-cell memory estimates and validation gates.

The same plan runs in serial, processes, or Dask. Serial execution is not a
separate reference implementation.

### 13.3 Execution behavior

- Compute each distinct base valuation once.
- Preserve CRN draw descriptors across compatible MC shocks.
- Group compatible requests by adapter and preparation fingerprint.
- Use child contexts with inner serial execution unless nested execution was
  explicitly reserved.
- Bound submitted cells and release their resources incrementally.
- Reassemble results in caller order regardless of grouping or completion.

`price_many` uses the same grouping planner without requiring scenarios.

### 13.4 Outcome and equality

A successful `ScenarioOutcome` contains the stable scenario ID, native value,
complete normalized economics, numerical diagnostics, and manifest
fingerprint. Scenario collections use
`ScenarioItemOutcome = ScenarioOutcome | PricingFailure`. Scheduling metadata
such as time, RSS, PID, host, and queue delay is stored separately.

For exact serial/process comparison, `normalized_cell_payload` means normalized
economics plus plan-dependent numerical diagnostics. It excludes operational
metadata and manifest fields intentionally changed by execution policy or
backend. Serial and process manifests need not be byte-identical; their
normalized-request and numerical-plan fingerprints must match when exact cell
reproduction is claimed.

Exact comparison traverses the complete normalized economic payload: PV,
standard/numerical error, cashflow legs, event distributions, probabilities,
calibration/no-arbitrage status, and any requested output. Schema adapters
must explicitly mark fields as economic, numerical-plan, or operational.

Normalization does not rewrite legacy semantics. In particular, the current
autocallable `ki_probability` field has historically meant `P(KI ever)` in MC
and `P(KI ever and never KO)` in QUAD/PDE. Legacy objects retain those values;
the normalized schema uses explicit `ki_ever_probability` and
`ki_survive_knocked_in_probability` fields and records which are available.
Likewise, `elapsed_seconds`, PIDs, and host data from detailed results are
operational rather than economic.

Reports distinguish:

- scenarios compared and scenarios matching;
- fields compared and fields matching;
- missing/extra fields;
- the first mismatching normalized path.

An aggregate `all_scenarios_match` flag is computed by the validator and is
never accepted as unverified input.

## 14. Determinism and numerical equivalence

### 14.1 Exact reproducibility domain

Bit identity is required when all of the following are unchanged:

- normalized product and market request;
- seed, scramble, random-stream layout, batch IDs, and path partition;
- time/spatial grid, model scheme, dtype, and arithmetic/reduction order;
- adapter, engine, QuantArk, NumPy, SciPy, BLAS, and solver versions;
- backend behavior insofar as it does not alter the plan or arithmetic path.

Thread completion order does not affect reduction. Process scenario execution
must reproduce the complete normalized cell payload exactly when every cell's
inner plan is unchanged.

### 14.2 Changed-plan equivalence

An explicitly selected plan that changes arithmetic, adaptive checkpoints,
grid, solver, or backend does not claim bit identity. It must pass a declared
gate:

- **MC paired gate:** use common draws where valid and compare paired
  differences. The default framework gate is
  `abs(mean(delta)) <= max(product_abs_tolerance, 4 * se(delta))`, with product
  families free to set a stricter reviewed tolerance. The gate records path
  count, effective independent samples, confidence multiplier, and any stderr
  or event-probability criteria.
- **Unpaired MC gate:** allowed only when CRN pairing is impossible; use a
  predeclared two-sample confidence gate and a larger sample budget.
- **PDE convergence gate:** compare at the production grid and at least one
  refined grid. The new path must lie within the product's absolute/relative
  tolerance and must not materially worsen observed convergence order or
  refined-grid error versus the compatibility path.

Tolerance values live in the checked-in validation matrix, not ad hoc test
code. Passing a numerical gate never gets labeled bit-identical.

### 14.3 Reproducibility manifest

Every framework outcome fingerprints:

- normalized request and requested outputs;
- numerical plan and preparation/artifact descriptors;
- resolved policy and resource limits;
- adapter, engine, and factory IDs/versions;
- Python and numerical dependency versions;
- platform, dtype, BLAS/solver identity where discoverable;
- validation policy and result.

Operational timing, cache-hit status, PID, and worker scheduling are recorded
but excluded from the plan fingerprint and economic equality.

## 15. Errors, cancellation, retry, and fallback

Framework APIs use typed errors, all deriving from the existing
`QuantArkException` root:

- `CapabilityError`
- `ResourceBudgetExceeded`
- `PreparationError`
- `TaskExecutionError`
- `DeterminismViolation`
- `ValidationGateError`

Direct legacy methods re-raise their historical exceptions without framework
wrapping.

Framework execution is fail-fast by default. On the first failure it stops
submission, cancels pending work where possible, waits for safe executor
shutdown, and reports completed, running, cancelled, and failed identities.
Single-item `execute` and `price` raise on failure. `collect_errors=True` is an
opt-in for `price_many` and scenario collections; those APIs return
`PricingFailure` items. Aggregators accept only successful `PricingOutcome`
objects, so failures and partial numerical values cannot enter totals.

Retries default to zero. When enabled, only infrastructure failures such as a
lost worker or transport interruption may retry, using an identical serialized
task specification. Numerical, validation, resource, and user-code failures
are not retried.

Backend fallback is disabled by default and requires an explicit ordered
fallback policy. Every fallback is recorded. A fallback that changes the
numerical plan invokes the changed-plan validation contract.

Cancellation is checked before preparation, before submission, between MC
batches, and between scenarios. Framework v1 does not promise interruption in
the middle of one PDE matrix solve/backward march; cancellation takes effect
at the next safe boundary.

## 16. Diagnostics and observability

Each run records:

- requested and resolved policies with source precedence;
- advertised, validated, and used capabilities;
- request, plan, artifact, draw, adapter, and worker fingerprints;
- preparation, queue, execution, reduction, and normalization times;
- worker utilization and bounded in-flight high-water marks;
- cache hits, misses, single-flight waits, evictions, and bytes;
- reserved versus observed memory and parent/child peak RSS;
- retries, cancellations, resource clamps, serializations, and fallbacks;
- validation gates and complete mismatch paths.

Diagnostics are immutable after outcome construction. A default in-memory sink
is sufficient for library use; structured logging/export hooks are additive.
Diagnostics must not alter numerical branches or normalized economics.

## 17. Full backward compatibility

### 17.1 Preserved surfaces

Framework v1 preserves:

- all existing engine constructor signatures and defaults;
- `BaseEngine.price` and its subclass implementations;
- `price_detailed`, `price_with_events`, `calculate_event_stats`,
  `calculate_greeks`, and `calculate_spot_greeks_curve` behavior;
- product-specific result and event-stat classes;
- `MCParams`, `PDEParams`, `EngineParams`, their serialization/equality/copy
  behavior, and all existing defaults;
- `use_dask` flags and existing availability behavior;
- `QUANTARK_DCN_MC_WORKERS` and `QUANTARK_QMC_CACHE_MB`;
- de facto documented subclass hooks until their legacy routes use adapters;
- direct-call warning and exception behavior.

`price_with_events` keeps its frozen `PricingResult`/`EventDistribution`
contract: engines with event stats use them, unsupported engines receive the
existing trivial maturity distribution, and `emit_distribution=False` retains
its current fallback behavior. DCN `price_detailed` continues returning the
frozen native `DCNMCResult`/`DCNPDEResult` objects and their existing `to_dict`
shapes. Session outcomes wrap and normalize these objects; direct calls do not.

Event-stat method signatures remain heterogeneous: PDE Snowball/Phoenix may
accept `npv` and `streams`, while other engine families accept only product
and environment or return `None`. Output-bundle adapters use the available
native seam without widening every direct signature. Equity `PDEEngine` keeps
its current constructor, dynamic scheme assignment, and product-solver cache.

Compatibility includes irregular timing and validation behavior. In
particular:

- `DCNMCEngine(num_workers=None)` resolves `QUANTARK_DCN_MC_WORKERS` at engine
  construction; an explicit constructor value wins, invalid text resolves to
  one, and nonpositive values clamp to one;
- the existing QMC cache reads `QUANTARK_QMC_CACHE_MB` when
  `quantark.montecarlo.qmc_sobol` is imported; invalid text means 2048 MiB and
  negative values clamp to zero;
- Snowball/Phoenix construction with inactive Dask does not eagerly reject a
  nonpositive `num_batches`; validation occurs only when the legacy Dask path
  is entered;
- missing Dask currently emits `UserWarning` and falls back, while missing
  SciPy QMC can raise `ImportError`.

Explicit session APIs may validate their own plans eagerly. They may not make
legacy constructors or direct calls eager. Import-time environment behavior is
covered by isolated subprocess tests and is not silently changed to per-call
resolution.

No mandatory execution fields are added to `MCParams` or `PDEParams`. New
policy belongs in `PricingRunContext`. A product-specific constructor may gain
an optional trailing policy argument only after signature and serialization
compatibility tests prove it safe.

### 17.2 Policy precedence

New-framework resolution is field-by-field, highest precedence first:

1. explicit `PricingRunContext` / `PricingSession` setting;
2. existing or later-approved constructor-level execution setting;
3. legacy engine-specific environment variable or flag;
4. generic QuantArk execution configuration/environment variable;
5. historical default.

Unset fields do not shadow lower levels. Resolution occurs once in the parent;
children receive resolved values and never re-resolve the host environment.
Direct legacy paths retain the historical resolution time described above.

Initial generic environment aliases are:

- `QUANTARK_EXEC_BATCH_BACKEND`
- `QUANTARK_EXEC_BATCH_WORKERS`
- `QUANTARK_EXEC_SCENARIO_BACKEND`
- `QUANTARK_EXEC_SCENARIO_WORKERS`
- `QUANTARK_EXEC_MEMORY_MB`
- `QUANTARK_EXEC_CACHE_MB`
- `QUANTARK_EXEC_MAX_IN_FLIGHT`

Configuration objects remain the preferred interface. Legacy
`QUANTARK_DCN_MC_WORKERS` wins for DCN when no explicit session or constructor
setting exists. Legacy `QUANTARK_QMC_CACHE_MB` remains the requested QMC cache
ceiling subject to an explicit parent session budget.

### 17.3 Migration mechanics

During transition, the session adapts legacy methods without changing the
direct method. Direct constructors and methods do not call `PricingSession` and
are never exposed to framework error wrapping, eager capability validation, or
new policy defaults. Numerical planning/reduction internals may be extracted
and shared only after exact parity tests exist. A later direct compatibility
facade may invoke those serial kernel internals with legacy validation,
warnings, exceptions, and return unwrapping preserved. Duplicate batch, Dask,
and preparation loops are removed only after both routes use the same internal
plan and reducer.

There are no v1 deprecation warnings. Any future removal requires a separate
spec and release cycle.

## 18. Engine inventory and adoption matrix

`quantark.execution.inventory` is the checked-in source of truth. CI discovers
publicly exported classes with MC or PDE engine types and fails if one is
absent.

The reviewed public snapshot is larger than the asset-level convenience
re-exports:

| Public surface | Reviewed count | Important shape |
|---|---:|---|
| Equity MC subpackage | 33 | BSM/LV/Heston/SLV/SABR, fixed and RQMC, vanilla through DCN |
| Equity PDE subpackage | 24 concrete + `BasePDESolver` | 1D BSM/LV and 2D Heston/SLV plus `PDEEngine` dispatcher |
| FX MC subpackage | 8 | Three model-vanilla engines plus five structured-product engines |
| FX PDE subpackage | 3 | LV/Heston/SLV vanilla solvers |
| Credit MC | 1 | Copula `BasketCDSEngine`, separate credit base and constructor scalars |
| Bond PDE | 2 + facade | Environment-bound convertible-bond engines with `price(bond)` call shape |

Discovery must union the explicit public `__all__` surfaces; it cannot inspect
only `quantark.asset.equity.engine` or `quantark.asset.fx.engine`, because those
currently omit public subpackage engines. The inventory also includes public
facades such as equity `PDEEngine` and `ConvertibleBondEngine`, marking whether
they execute, dispatch, or wrap an inventoried solver.

Each inventory record includes:

- public import path and engine type;
- product and asset family;
- BSM, LV, Heston, SLV, or other model family;
- dimension and fixed/adaptive planning;
- supported output bundles;
- adapter and normalizer IDs;
- supported backends and thread/process safety;
- reusable artifact/draw capabilities;
- determinism and validation profiles;
- conformance/performance test IDs;
- adoption state and owner/milestone for any temporary limitation.

Allowed capability states are `supported`, `not_applicable`, and
`temporary_legacy`. `not_applicable` requires a reason. `temporary_legacy`
requires an owner and milestone and cannot mean “not inventoried.” Every
engine, including temporary legacy engines, must be session-reachable through
the serial compatibility adapter.

The initial matrix must cover at least:

- equity, FX, credit, and bond;
- single- and multi-asset;
- vanilla, American, barrier/double-barrier, Asian, sharkfin, accumulator,
  range-accrual, Snowball, Phoenix, KO-reset Snowball, and DCN families;
- BSM/GBM, local volatility, Heston, SLV, SABR, copula, and jump-diffusion
  families where exported;
- all exported 1D and 2D PDE solvers;
- fixed-batch and adaptive RQMC variants;
- existing Snowball/Phoenix Dask routes.

The inventory is a release gate, not documentation generated after the fact.

## 19. Testing strategy

### 19.1 Compatibility tests

For every inventory row, compare direct legacy execution with serial
`PricingSession` execution:

- native result type and field schema;
- warning and exception type/message where contractual;
- PV, error estimates, event stats, cashflow decomposition, and grid outputs;
- seed/batch/grid defaults;
- exact bytes when the numerical plan is identical;
- the declared numerical gate otherwise.

Test constructors and dataclass round-trips so `MCParams`/`PDEParams` equality,
copying, pickling, and dictionary serialization remain unchanged.

Compatibility snapshots also cover `inspect.signature`, `dataclasses.fields`,
canonical/legacy import identity, public result dataclass fields and `to_dict`
shape, the legacy autocallable KI semantic split, and exact constructor
validation timing. Environment-variable tests run in fresh subprocesses so
construction-time and import-time behavior are observable. Missing-Dask
warning/fallback, RQMC-before-Dask precedence, exact legacy path counts, and
statistical Dask parity receive dedicated Snowball and Phoenix tests.
Framework exception tests also preserve the canonical and legacy identities of
`QuantArkException`, `ValidationError`, `NumericalError`, `MarketDataError`,
and `PricingError`, plus contractual message fragments on direct calls.

### 19.2 Capability contract tests

- deliberately permuted completion with canonical reduction;
- bounded in-flight tasks and compact incremental reduction;
- complete draw/artifact keys, hit verification, single-flight misses, LRU
  eviction, writable-copy isolation, and dependency invalidation;
- counting builders proving expected local-vol/grid/operator build counts;
- engine-instance reuse across sequential and concurrent sessions;
- memory admission, cache clamps, child budgets, and oversize-task failure;
- macOS spawn serialization, registry reconstruction, and no worker globals;
- Dask and local backends consuming identical plans/reducers;
- fixed-batch MC serial/thread exactness;
- adaptive RQMC sequential compatibility and parallel-wave qualification;
- PDE event/grid/factorization reuse plus convergence gates;
- base-valuation deduplication and CRN descriptor reuse;
- complete scenario payload comparison and correct scenario/field counts;
- fail-fast, collect-errors, cancellation, infrastructure retry, broken pool,
  and explicit fallback behavior.

### 19.3 Representative numerical matrix

The conformance suite includes BSM/LV/Heston/SLV/SABR; equity/FX/credit/bond;
single/multiasset; vanilla/barrier/Asian/range-accrual/Snowball/Phoenix/
KO-reset/DCN, basket CDS, and convertible bonds; and every exported 1D/2D PDE
family. Product-specific tolerances are reviewed and stored with the validation
matrix.

Existing quarantined failures need a separately tracked owner and rationale;
they cannot satisfy a framework acceptance gate.

## 20. Performance and memory acceptance

Benchmarks use the same build, machine class, request, numerical plan, path
count/grid, warm-up, and dependency environment. Each measured case has at
least five post-warm-up repetitions and reports median, dispersion, cold/warm
cache state, parent RSS, and aggregate child RSS.

Required release gates:

1. Serial compatibility median wall-time regression is no more than 3%.
2. Peak RSS regression with caches disabled is no more than 5%.
3. Fixed-batch, thread-capable MC achieves at least 1.5x at four workers and
   2.5x at eight workers when there are at least two batches per worker.
4. A workload of at least ten compatible CRN repricings is at least 2x faster
   with draw/artifact reuse than uncached serial execution.
5. Standalone PDE framework overhead is no more than 3%.
6. Independent scenario grids with serial wall time above ten seconds achieve
   at least 2.5x on four processes when enough cells exist.
7. Parent worker, in-flight, and memory admission limits are never exceeded.
8. MC reduction does not retain unbounded batch payloads.

Fast deterministic microbenchmarks run in CI to catch gross regressions.
Production-sized gates run on a scheduled controlled host and are required for
release. Cold-cache, warm-cache, threading-only, preparation-only, and combined
results are reported separately; a combined headline cannot substitute for
mechanism attribution.

## 21. Migration plan and gates

### Phase 0 — Contracts and inventory

- Add `quantark.execution` contracts, errors, policy resolution, registry, and
  inventory discovery.
- Add the serial compatibility adapter and manifest skeleton.
- Freeze golden fixtures from the reviewed baseline.

**Exit:** every exported MC/PDE engine is inventoried and session-reachable in
serial; no direct public behavior changes.

### Phase 1 — Serial kernel and preparation

- Implement the lifecycle, leases, diagnostics, canonical fingerprints, and
  session-owned artifact cache.
- Adapt request-local DCN preparation first, then remove mutable active state.

**Exit:** direct versus session parity across the matrix; serial and disabled-
cache regression gates pass.

### Phase 2 — Fixed-batch MC

- Implement `BatchPlan`, bounded thread execution, compact outcomes, and
  incremental canonical reduction.
- Migrate DCN first, then all fixed-batch MC engines.
- Implement `DrawRepository` and guarded in-place transforms.

**Exit:** every fixed-batch exported MC engine either advertises the capability
or has a specific `not_applicable` rationale; exactness, memory, and speed gates
pass.

### Phase 3 — Adaptive RQMC and model families

- Preserve sequential compatibility stopping.
- Add and separately qualify parallel-wave mode.
- Complete GBM/BSM, LV, Heston, SLV, equity, FX, and multiasset adapters.

**Exit:** all adaptive plans have deterministic checkpoint tests and declared
validation profiles.

### Phase 4 — PDE preparation and rich outcomes

- Adapt all 1D/2D PDE engines.
- Move safe grid, event-map, term-context, local-vol, coefficient, and
  factorization reuse behind artifact descriptors.
- Consolidate one-solve PV/event/grid projections where engines support them.

**Exit:** PDE parity, convergence, build-count, memory, and overhead gates pass.

### Phase 5 — Scenario, portfolio, process, and Dask

- Implement typed scenario planning and `WorkerSpec` reconstruction.
- Port the solution surface-risk workflow without its worker globals or name
  parsing.
- Route `price_many`, local processes, and existing Dask paths through the same
  plans/reducers.

**Exit:** spawn, child-budget, complete-payload, fault, and scenario speed gates
pass; nested execution remains off by default.

### Phase 6 — Cleanup and release

- Remove duplicate internal batch, Dask, and preparation scaffolding only after
  legacy entry points use the kernel.
- Publish the capability matrix, policy guide, reproducibility schema, and
  migration examples.

**Exit:** no exported engine is missing; every advertised capability has tests;
all compatibility, numerical, performance, and resource gates pass.

## 22. Versioning

This document defines execution-framework contract v1. The proposed package
release is `0.3.0`, but the version is cut only after Phase 6 gates pass. No
partial phase may claim universal MC/PDE framework support.

Serialized `WorkerSpec`, manifest, normalized-economics, and scenario schemas
carry their own explicit schema versions. Readers reject unknown major schema
versions rather than guessing.

## 23. Rejected alternatives

### 23.1 Mandatory shared engine base class

Rejected because QuantArk has diverse legacy constructors, result types,
adaptive behavior, and non-equity engines. It creates a large inheritance
migration without solving resource, process, or normalization contracts.

### 23.2 Four independent utilities with opt-in adoption

Rejected because it leaves lifecycle, budgets, determinism, errors, scenario
normalization, and engine coverage fragmented. It also makes adoption
opportunistic rather than a framework invariant.

### 23.3 Orchestration-only scenario layer

Rejected because it cannot safely reuse draws/preparation, define batch
reduction, or prevent nested resource multiplication inside engines.

### 23.4 Object-identity preparation memoization

Rejected because the same environment object can be mutated, equal-valued
objects can be safely reusable, and mutable per-engine memo state is not
concurrent-session safe.

## 24. Definition of done

The framework update is complete only when:

- all exported MC/PDE engines are in the checked-in inventory and session-
  reachable;
- direct legacy behavior remains compatible;
- the same immutable plan feeds every advertised backend;
- fixed-batch MC reduces bounded compact outcomes in canonical order;
- adaptive RQMC compatibility mode retains its stopping sequence;
- PDE engines use safe preparation contracts without changing solver meaning;
- scenario/process execution is spawn-safe and parent-budgeted;
- caches use complete value fingerprints, single-flight construction, and
  immutable values;
- exact and changed-plan validation contracts are both exercised;
- diagnostics and manifests explain the resolved execution;
- every numerical, fault, performance, and memory release gate passes.

Only then may legacy duplicate internals be removed or universal framework
support be advertised.

## Appendix A — Reviewed public engine snapshot

This snapshot anchors Phase 0 discovery. The checked-in inventory created by
the implementation is authoritative if public exports change later.

### A.1 Equity MC

- European/model vanilla: `EuropeanMCEngine`, `LocalVolMCEngine`,
  `HestonMCEngine`, `HestonSLVMCEngine`, `SABRMCEngine`.
- American/Asian/digital: `AmericanOptionMCEngine`, `AsianOptionMCEngine`,
  `DigitalOptionMCEngine`.
- Barrier: `BarrierOptionMCEngine`, `LocalVolBarrierMCEngine`,
  `HestonBarrierMCEngine`, `HestonSLVBarrierMCEngine`.
- Other structured products: `SingleSharkfinOptionMCEngine`,
  `DoubleSharkfinOptionMCEngine`, `RangeAccrualMCEngine`,
  `AccumulatorMCEngine`.
- Snowball: `SnowballMCEngine`, `LocalVolSnowballMCEngine`,
  `HestonSnowballMCEngine`, `QESnowballMCEngine`,
  `HestonSLVSnowballMCEngine`, `HestonSLVQESnowballMCEngine`.
- Phoenix: `PhoenixMCEngine`, `LocalVolPhoenixMCEngine`,
  `HestonPhoenixMCEngine`, `QEPhoenixMCEngine`,
  `HestonSLVPhoenixMCEngine`, `HestonSLVQEPhoenixMCEngine`.
- DCN: `DCNMCEngine`, `LocalVolDCNMCEngine`, `HestonDCNMCEngine`,
  `QEDCNMCEngine`, `CoupledCoarseHestonDCNMCEngine`.

### A.2 Equity PDE

- European/model vanilla: `EuropeanPDESolver`, `LocalVolPDESolver`,
  `HestonPDESolver`, `HestonSLVPDESolver`.
- American/barrier/touch: `AmericanPDESolver`, `BarrierPDESolver`,
  `LocalVolBarrierPDESolver`, `HestonBarrierPDESolver`,
  `HestonSLVBarrierPDESolver`, `DoubleBarrierPDESolver`,
  `OneTouchPDESolver`, `DoubleOneTouchPDESolver`.
- Snowball and KO-reset: `SnowballPDESolver`,
  `LocalVolSnowballPDESolver`, `HestonSnowballPDESolver`,
  `HestonSLVSnowballPDESolver`, `KOResetSnowballPDESolver`.
- Phoenix: `PhoenixPDESolver`, `LocalVolPhoenixPDESolver`,
  `HestonPhoenixPDESolver`, `HestonSLVPhoenixPDESolver`.
- DCN: `DCNPDEEngine`, `LocalVolDCNPDEEngine`, `HestonDCNPDESolver`.
- Public abstractions/facades: `BasePDESolver`, `PDEEngine`, `TimeGrid`, and
  `SpatialGrid`; grids are inventoried as supporting types, not engines.

### A.3 FX, credit, and bond

- FX MC: `FxLocalVolMCEngine`, `FxHestonMCEngine`,
  `FxHestonSLVMCEngine`, `FxRangeAccrualMCEngine`, `FxBarrierMCEngine`,
  `FxSharkfinMCEngine`, `FxTarnForwardMCEngine`,
  `FxTargetRedemptionNoteMCEngine`.
- FX PDE: `FxLocalVolPDESolver`, `FxHestonPDESolver`,
  `FxHestonSLVPDESolver`.
- Credit MC: `BasketCDSEngine`.
- Bond PDE/facade: `ConvertibleBondJumpDiffusionEngine`,
  `ConvertibleBondTFEngine`, `ConvertibleBondEngine`.

### A.4 Known abstraction gaps to design for

- `SABRMCEngine` does not follow the common equity base shape.
- FX uses both `FxEngineParams`, `FxMCParams`, and constructor scalars.
- Credit has its own base and no common MC parameter object.
- Convertible-bond engines bind market state in the constructor and use a
  one-argument price call.
- Equity and FX asset-level `engine` re-exports are narrower than their public
  `mc`/`pde` subpackages.

These are adapter cases, not reasons to narrow the framework scope.
