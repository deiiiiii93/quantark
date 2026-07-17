"""Typed scenario execution contracts (spec sections 12.3, 13.1-13.2).

Every field on these dataclasses is JSON-representable (strings, numbers,
booleans, None, and tuples of pairs) so a ``WorkerSpec`` can cross a spawn
or Dask process boundary as a canonical serialized snapshot — never as a
closure, bound live engine, or raw product/environment reference.

``CallableRef`` is the spawn-reconstruction primitive (plan-gate finding 1,
2026-07-17): registries are process-local dictionaries, so a fresh worker
must IMPORT the registering module (which re-runs its ``register_*`` calls)
before any ID lookup can succeed. The ref records the module and qualified
name so the worker can both import and verify the registered object.
"""
from dataclasses import dataclass

__all__ = [
    "SCENARIO_SCHEMA_VERSION",
    "BaseInputsRef",
    "CallableRef",
    "ScenarioCell",
    "ScenarioPlan",
    "WorkerSpec",
]

SCENARIO_SCHEMA_VERSION = "scenario/v1"


@dataclass(frozen=True)
class CallableRef:
    """Versioned reference to a registered, importable callable."""

    kind: str          # "transformer" | "runner" | "factory"
    ref_id: str
    module: str
    qualname: str
    schema_version: str


@dataclass(frozen=True)
class BaseInputsRef:
    """Registered-factory reference that rebuilds base inputs in any
    process. ``payload`` is a sorted tuple of (key, value) pairs of
    JSON-able primitives."""

    factory_id: str
    payload: tuple = ()


@dataclass(frozen=True)
class ScenarioCell:
    """One normalized scenario cell of an immutable plan."""

    scenario_id: str
    position: int
    transformer_id: str
    runner_id: str
    parameters: tuple            # sorted (key, value) pairs
    mutation_tags: frozenset     # declared upper bound (spec 13.1)
    changed_tags: frozenset      # verified actual changes (spec 10.2)
    invalidate_all: bool         # conservative full invalidation
    cell_fingerprint: str | None
    group_key: tuple
    est_bytes: int | None = None


@dataclass(frozen=True)
class ScenarioPlan:
    """Immutable normalized scenario plan (spec section 13.2). The same
    plan runs on serial, threads, processes, and Dask backends."""

    plan_id: str
    schema_version: str
    base_kind: str               # "request" | "inputs_ref"
    base_fingerprint: str | None
    engine_factory_id: str | None
    cells: tuple
    groups: tuple                # ((group_key, (positions...)), ...)


@dataclass(frozen=True)
class WorkerSpec:
    """Importable, versioned process-worker specification (spec 12.3).

    Contains registered callable references, canonical payloads, explicit
    child policy/budget values, and expected environment fingerprints.
    Never contains a closure, live engine, mutable environment, worker
    global dictionary, or raw product/pricing-environment reference.
    """

    schema_version: str
    base_ref: BaseInputsRef
    callable_refs: tuple         # CallableRef entries the plan needs
    child_policy_values: tuple   # resolved (field, value) pairs
    child_budget_values: tuple   # resolved (field, value) pairs
    expected: tuple              # (name, value) environment fingerprints
