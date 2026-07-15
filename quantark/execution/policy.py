"""Execution, determinism, and resource policies with precedence resolution.

Resolution is field-by-field, highest precedence first (spec section 17.2):
explicit setting > generic ``QUANTARK_EXEC_*`` environment alias > historical
default. Resolution happens once (at session construction); children receive
resolved values and never re-resolve the host environment.

Phase 0 note: resolution accepts any known backend string, but the serial
compatibility adapter only advertises the ``serial`` backend, so a non-serial
resolved backend fails capability validation at dispatch (no silent fallback).
"""
import os
from dataclasses import dataclass

__all__ = [
    "KNOWN_BACKENDS",
    "ExecutorSelection",
    "ExecutionPolicy",
    "DeterminismPolicy",
    "ResourceBudget",
    "resolve_execution_policy",
    "resolve_resource_budget",
    "policy_values",
]

KNOWN_BACKENDS = ("serial", "threads", "processes", "dask")


@dataclass(frozen=True)
class ExecutorSelection:
    backend: str = "serial"
    workers: int = 1
    max_in_flight: int | None = None
    may_shrink: bool = False
    fallback_order: tuple = ()


@dataclass(frozen=True)
class ExecutionPolicy:
    batch: ExecutorSelection = ExecutorSelection()
    scenario: ExecutorSelection = ExecutorSelection()
    nested_execution: bool = False
    fail_fast: bool = True
    retries: int = 0


@dataclass(frozen=True)
class DeterminismPolicy:
    require_manifest: bool = True
    changed_plan_profile: str = "reject"
    mismatch_raises: bool = True


@dataclass(frozen=True)
class ResourceBudget:
    max_processes: int = 1
    max_threads: int = 1
    total_memory_bytes: int | None = None
    draw_cache_bytes: int | None = None
    artifact_cache_bytes: int | None = None
    max_in_flight: int = 1


def _env_int(environ, key, default, field, sources):
    raw = environ.get(key)
    if raw is None:
        sources.append((field, "default"))
        return default
    try:
        value = int(raw)
    except ValueError:
        sources.append((field, "env_invalid_default"))
        return default
    sources.append((field, "env"))
    return value


def _env_backend(environ, key, default, field, sources):
    raw = environ.get(key)
    if raw is None:
        sources.append((field, "default"))
        return default
    if raw not in KNOWN_BACKENDS:
        sources.append((field, "env_invalid_default"))
        return default
    sources.append((field, "env"))
    return raw


def resolve_execution_policy(explicit=None, environ=None):
    """Resolve an ExecutionPolicy once; returns (policy, source map)."""
    if explicit is not None:
        # An explicit policy object is a complete, highest-precedence setting.
        fields = [
            "batch.backend", "batch.workers", "scenario.backend",
            "scenario.workers", "nested_execution", "fail_fast", "retries",
        ]
        return explicit, tuple((f, "explicit") for f in fields)

    environ = os.environ if environ is None else environ
    sources: list = []
    batch = ExecutorSelection(
        backend=_env_backend(environ, "QUANTARK_EXEC_BATCH_BACKEND",
                             "serial", "batch.backend", sources),
        workers=_env_int(environ, "QUANTARK_EXEC_BATCH_WORKERS",
                         1, "batch.workers", sources),
    )
    scenario = ExecutorSelection(
        backend=_env_backend(environ, "QUANTARK_EXEC_SCENARIO_BACKEND",
                             "serial", "scenario.backend", sources),
        workers=_env_int(environ, "QUANTARK_EXEC_SCENARIO_WORKERS",
                         1, "scenario.workers", sources),
    )
    sources.extend(
        [("nested_execution", "default"), ("fail_fast", "default"),
         ("retries", "default")]
    )
    return ExecutionPolicy(batch=batch, scenario=scenario), tuple(sources)


def policy_values(policy, budget, determinism) -> tuple:
    """Canonical value-bearing snapshot of the RESOLVED configuration.

    This is what a reproducibility manifest records (spec section 14.3): the
    effective values, not their sources. Source metadata lives separately in
    diagnostics ``policy_sources``.
    """
    return (
        ("batch.backend", policy.batch.backend),
        ("batch.workers", str(policy.batch.workers)),
        ("batch.max_in_flight", str(policy.batch.max_in_flight)),
        ("scenario.backend", policy.scenario.backend),
        ("scenario.workers", str(policy.scenario.workers)),
        ("nested_execution", str(policy.nested_execution)),
        ("fail_fast", str(policy.fail_fast)),
        ("retries", str(policy.retries)),
        ("budget.max_processes", str(budget.max_processes)),
        ("budget.max_threads", str(budget.max_threads)),
        ("budget.total_memory_bytes", str(budget.total_memory_bytes)),
        ("budget.draw_cache_bytes", str(budget.draw_cache_bytes)),
        ("budget.artifact_cache_bytes", str(budget.artifact_cache_bytes)),
        ("budget.max_in_flight", str(budget.max_in_flight)),
        ("determinism.require_manifest", str(determinism.require_manifest)),
        ("determinism.changed_plan_profile", determinism.changed_plan_profile),
        ("determinism.mismatch_raises", str(determinism.mismatch_raises)),
    )


def resolve_resource_budget(explicit=None, environ=None):
    """Resolve a ResourceBudget once; returns (budget, source map)."""
    if explicit is not None:
        fields = ["total_memory_bytes", "artifact_cache_bytes", "max_in_flight"]
        return explicit, tuple((f, "explicit") for f in fields)

    environ = os.environ if environ is None else environ
    sources: list = []
    memory_mb = _env_int(environ, "QUANTARK_EXEC_MEMORY_MB",
                         None, "total_memory_bytes", sources)
    cache_mb = _env_int(environ, "QUANTARK_EXEC_CACHE_MB",
                        None, "artifact_cache_bytes", sources)
    max_in_flight = _env_int(environ, "QUANTARK_EXEC_MAX_IN_FLIGHT",
                             1, "max_in_flight", sources)
    return (
        ResourceBudget(
            total_memory_bytes=None if memory_mb is None else memory_mb * 2**20,
            artifact_cache_bytes=None if cache_mb is None else cache_mb * 2**20,
            max_in_flight=max_in_flight,
        ),
        tuple(sources),
    )
