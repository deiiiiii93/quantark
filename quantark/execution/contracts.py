"""Framework request, outcome, capability, and scenario contracts (spec section 5).

``PricingRequest`` is a shallow frozen envelope around potentially mutable
legacy objects. ``NormalizedPricingRequest`` is the immutable snapshot every
post-normalization capability method receives (spec section 6); in Phase 0 the
snapshot is shallow (``snapshot_complete=False``), so requests are uncacheable
and no fingerprint is claimed.

Immutable mappings are represented as sorted tuples of ``(key, value)`` pairs.
"""
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "PricingOperation",
    "OutputKind",
    "DEFAULT_OUTPUTS",
    "PricingRequest",
    "NormalizedPricingRequest",
    "FrameworkErrorInfo",
    "PricingOutcome",
    "PricingFailure",
    "EngineCapabilities",
    "PreparedState",
    "BatchTask",
    "BatchPlan",
    "BatchOutcome",
    "ScenarioSpec",
    "ScenarioOutcome",
    "economics_mapping",
]


class PricingOperation(Enum):
    PRICE = "price"
    PRICE_DETAILED = "price_detailed"
    EVENT_STATS = "event_stats"


class OutputKind(Enum):
    PV = "pv"
    ERROR_ESTIMATE = "error_estimate"
    EVENT_STATS = "event_stats"
    CASHFLOWS = "cashflows"
    GRID = "grid"


DEFAULT_OUTPUTS = frozenset({OutputKind.PV})


@dataclass(frozen=True)
class PricingRequest:
    product: object
    pricing_env: object | None = None
    operation: PricingOperation = PricingOperation.PRICE
    outputs: frozenset = DEFAULT_OUTPUTS
    operation_options: tuple = ()
    request_id: str | None = None


@dataclass(frozen=True)
class NormalizedPricingRequest:
    engine_class_path: str
    operation: PricingOperation
    outputs: frozenset
    operation_options: tuple
    product_ref: object
    pricing_env_ref: object | None
    snapshot_complete: bool
    fingerprint: str | None


@dataclass(frozen=True)
class FrameworkErrorInfo:
    error_type: str
    message: str


@dataclass(frozen=True)
class PricingOutcome:
    value: object
    normalized_economics: tuple
    diagnostics: object
    manifest: object


@dataclass(frozen=True)
class PricingFailure:
    item_id: str
    error: FrameworkErrorInfo
    diagnostics: object
    manifest: object | None = None


@dataclass(frozen=True)
class EngineCapabilities:
    operations: frozenset
    output_kinds: frozenset
    supported_backends: frozenset
    fixed_planning: bool | None
    prepared_state_thread_safe: bool
    instance_reentrant: bool
    process_reconstructable: bool
    deterministic_reduction: bool
    peak_memory_estimate: str  # "exact" | "conservative" | "unavailable"
    adapter_id: str
    adapter_version: str


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    transformer_id: str
    parameters: tuple
    mutation_tags: frozenset
    required_capabilities: frozenset = frozenset()
    validation_policy: object | None = None


@dataclass(frozen=True)
class ScenarioOutcome:
    scenario_id: str
    value: object
    normalized_economics: tuple
    diagnostics: object
    manifest_fingerprint: str | None


@dataclass(frozen=True)
class PreparedState:
    """Immutable request-scoped prepared state (spec section 6.3).

    ``handles`` are cache ArtifactHandles the kernel closes in ``finally``
    after execution, releasing the pinned entries.
    """

    payload: object
    descriptors: tuple
    fingerprint: str | None
    byte_estimate: int | None
    handles: tuple = ()


@dataclass(frozen=True)
class BatchTask:
    """One unit of MC work; ``batch_id`` is the engine stream-shift id
    (None means the engine's single-batch stream)."""

    plan_id: str
    batch_index: int
    batch_id: int | None
    n_paths: int


@dataclass(frozen=True)
class BatchPlan:
    """Immutable fixed-batch MC plan (spec section 8.1). The plan, not
    executor completion order, determines economics."""

    plan_id: str
    engine_class_path: str
    num_batches: int
    paths_per_batch: int
    total_paths: int
    seed: int
    stream_kind: str        # "sobol" | "pseudorandom" | "antithetic"
    stream_layout: str
    time_steps: int
    dimension: int
    dtype: str
    scheme: str
    stderr_mode: str        # "scramble_means" | "pathwise_iid"
    reduction_order: str    # "batch_index/v1"
    tasks: tuple
    est_task_peak_bytes: int | None
    est_outcome_bytes: int | None
    implementation_fingerprint: str | None


@dataclass(frozen=True)
class BatchOutcome:
    """Compact per-batch sufficient statistics (spec section 8.2)."""

    batch_index: int
    n_paths: int
    payload: object


def economics_mapping(outcome) -> dict:
    """Read a normalized-economics tuple back as a dict."""
    return dict(outcome.normalized_economics)
