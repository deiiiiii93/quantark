"""Spawn-importable scenario fixtures for process/dask backend tests.

Spawn children rebuild everything through ``CallableRef`` imports of THIS
module; nothing here relies on parent-process state.
"""
import dataclasses
import time

from quantark.execution.scenario import registries


@dataclasses.dataclass(frozen=True)
class ToyInputs:
    spot: float
    vol: float


def build_toy_inputs(payload):
    return ToyInputs(spot=payload["spot"], vol=payload["vol"])


def toy_bump(base, parameters):
    return dataclasses.replace(base, spot=base.spot + parameters["ds"])


def toy_engine_factory(parameters):
    return None  # toy runners price without an engine


def toy_runner(cell, resolved, child_context):
    t = resolved.transformed
    pv = t.spot * t.vol  # deterministic, cheap
    return pv, (("pv", pv), ("spot", t.spot), ("numerical.vol", t.vol)), None


def toy_runner_failing(cell, resolved, child_context):
    if resolved.transformed.spot > 100.0:
        raise ValueError("boom")
    return toy_runner(cell, resolved, child_context)


def toy_runner_slow_ok_fast_fail(cell, resolved, child_context):
    """Earlier (non-failing) cells sleep; a late failing cell finishes
    first — the out-of-order fail-fast fixture (plan-gate finding 5)."""
    if resolved.transformed.spot > 100.0:
        raise ValueError("late-boom")
    time.sleep(0.8)
    return toy_runner(cell, resolved, child_context)


def toy_runner_budget_probe(cell, resolved, child_context):
    budget = child_context.resource_budget
    policy = child_context.execution_policy
    economics = (
        ("pv", 1.0),
        ("max_processes", float(budget.max_processes)),
        ("max_threads", float(budget.max_threads)),
        ("nested_execution", bool(policy.nested_execution)),
        ("scenario_backend_is_serial", policy.scenario.backend == "serial"),
        ("batch_backend_is_serial", policy.batch.backend == "serial"),
    )
    return 1.0, economics, None


_COMPONENTS = (
    ("spot", lambda b: b.spot),
    ("vol_surface", lambda b: b.vol),
)

registries.register_factory("toy-inputs/v1", build_toy_inputs)
registries.register_factory("toy-engine/v1", toy_engine_factory)
registries.register_transformer(
    "toy-bump/v1", toy_bump,
    allowed_tags=frozenset({"spot"}), components=_COMPONENTS,
)
registries.register_runner("toy/v1", toy_runner, value_kind="float")
registries.register_runner(
    "toy-failing/v1", toy_runner_failing, value_kind="float"
)
registries.register_runner(
    "toy-slow/v1", toy_runner_slow_ok_fast_fail, value_kind="float"
)
registries.register_runner(
    "toy-budget/v1", toy_runner_budget_probe, value_kind="float"
)
