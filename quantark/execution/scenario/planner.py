"""Scenario normalization and planning (spec sections 10.2, 13.2).

Mutation-footprint verification (hardened by plan-gate finding 2,
2026-07-17): the planner fingerprints every declared component of the base
and transformed snapshots AND the whole snapshots. Enforcement:

(a) changed component tags must be a subset of BOTH the spec's declared
    ``mutation_tags`` and the registration's ``allowed_tags``, else
    ``ValidationGateError``;
(b) a whole-snapshot change that no declared component explains escaped
    the component schema — that is under-declaration and raises
    ``ValidationGateError``;
(c) any uncanonicalizable component or whole snapshot conservatively marks
    the cell ``invalidate_all=True`` (no artifact reuse) instead of
    failing — correctness never depends on cacheability.

Transformer purity (spec 13.1): base-side component fingerprints are
recomputed after every transform; a transformer that mutated the base
raises ``ValidationGateError``.
"""
from quantark.execution.cache.fingerprint import try_fingerprint
from quantark.execution.errors import ValidationGateError
from quantark.execution.scenario import registries
from quantark.execution.scenario.contracts import (
    SCENARIO_SCHEMA_VERSION,
    BaseInputsRef,
    ScenarioCell,
    ScenarioPlan,
)
from quantark.util.exceptions import ValidationError

__all__ = [
    "DEFAULT_RUNNER_ID",
    "plan_price_groups",
    "plan_scenarios",
    "resolve_base",
]

DEFAULT_RUNNER_ID = "request/v1"
_RUNNER_CAPABILITY_PREFIX = "runner:"


def resolve_base(base):
    """Return (base_kind, resolved_base_inputs, base_fingerprint).

    ``inputs_ref`` bases are rebuilt from their registered factory — once
    here in the parent for planning; process/Dask workers call the same
    factory again worker-side. Any other object is a LIVE base
    (``base_kind == "request"``), usable on serial/threads only.
    """
    if isinstance(base, BaseInputsRef):
        registries.check_worker_payload(dict(base.payload))
        factory = registries.get_factory(base.factory_id)
        resolved = factory.fn(dict(base.payload))
        return "inputs_ref", resolved, try_fingerprint(base)
    return "request", base, try_fingerprint(base)


def _runner_id_for(spec) -> str:
    for capability in spec.required_capabilities:
        if isinstance(capability, str) and capability.startswith(
            _RUNNER_CAPABILITY_PREFIX
        ):
            return capability[len(_RUNNER_CAPABILITY_PREFIX):]
    return DEFAULT_RUNNER_ID


def _component_fingerprints(components, snapshot) -> tuple:
    return tuple(
        (tag, try_fingerprint(extractor(snapshot)))
        for tag, extractor in components
    )


def plan_price_groups(items) -> tuple:
    """Group (engine, request) pairs by engine class path (spec 13.3).

    Pure planning: returns ``((group_key, (original_indices, ...)), ...)``
    in first-appearance group order with intra-group caller order, so
    session artifact/draw caches see contiguous compatible work. The
    caller reassembles results by original index.
    """
    groups: dict = {}
    for index, (engine, _request) in enumerate(items):
        cls = type(engine)
        key = f"{cls.__module__}.{cls.__qualname__}"
        groups.setdefault(key, []).append(index)
    return tuple((key, tuple(indices)) for key, indices in groups.items())


def _ensure_builtin_runners() -> None:
    """Built-in runners register at ``runner`` module import; the planner
    imports it lazily so ``request/v1`` resolves without the caller having
    to import execution internals."""
    import quantark.execution.scenario.runner  # noqa: F401


def plan_scenarios(base, scenario_specs, engine_factory) -> ScenarioPlan:
    """Normalize ``ScenarioSpec``s into an immutable ``ScenarioPlan``."""
    _ensure_builtin_runners()
    specs = tuple(scenario_specs)
    seen_ids = set()
    for spec in specs:
        if spec.scenario_id in seen_ids:
            raise ValidationError(
                f"duplicate scenario_id {spec.scenario_id!r}; scenario ids "
                "must be unique within a plan"
            )
        seen_ids.add(spec.scenario_id)

    base_kind, base_inputs, base_fingerprint = resolve_base(base)
    engine_factory_id = engine_factory if isinstance(engine_factory, str) else None
    if engine_factory_id is not None:
        registries.get_factory(engine_factory_id)  # must be registered

    cells = []
    group_positions: dict = {}
    for position, spec in enumerate(specs):
        registration = registries.get_transformer(spec.transformer_id)
        runner_id = _runner_id_for(spec)
        registries.get_runner(runner_id)  # must be registered

        base_component_fps = _component_fingerprints(
            registration.components, base_inputs
        )
        base_whole_fp = try_fingerprint(base_inputs)
        transformed = registration.fn(base_inputs, dict(spec.parameters))

        # Transformer purity: the base must be untouched.
        base_after_fps = _component_fingerprints(
            registration.components, base_inputs
        )
        if base_after_fps != base_component_fps:
            raise ValidationGateError(
                f"transformer {spec.transformer_id!r} mutated the base "
                "inputs in place; transformers must be pure (spec 13.1)"
            )

        transformed_component_fps = _component_fingerprints(
            registration.components, transformed
        )
        transformed_whole_fp = try_fingerprint(transformed)

        changed = set()
        invalidate_all = False
        for (tag, before), (_, after) in zip(
            base_component_fps, transformed_component_fps
        ):
            if before is None or after is None:
                invalidate_all = True
            elif before != after:
                changed.add(tag)

        if base_whole_fp is None or transformed_whole_fp is None:
            invalidate_all = True
        elif base_whole_fp != transformed_whole_fp and not changed:
            raise ValidationGateError(
                f"transformer {spec.transformer_id!r} changed the snapshot "
                "outside its declared component schema; the mutation cannot "
                "be attributed to any tag (under-declared components, "
                "spec 10.2)"
            )

        undeclared = changed - set(spec.mutation_tags)
        if undeclared:
            raise ValidationGateError(
                f"scenario {spec.scenario_id!r} under-declares its mutation "
                f"footprint: transformer changed {sorted(undeclared)} beyond "
                f"declared {sorted(spec.mutation_tags)} (spec 10.2)"
            )
        not_allowed = changed - set(registration.allowed_tags)
        if not_allowed:
            raise ValidationGateError(
                f"transformer {spec.transformer_id!r} changed "
                f"{sorted(not_allowed)} outside its registered allowed_tags "
                f"{sorted(registration.allowed_tags)}"
            )

        cell_fingerprint = try_fingerprint(
            (spec.transformer_id, spec.parameters, base_fingerprint, runner_id)
        )
        group_key = (runner_id, spec.transformer_id)
        cells.append(
            ScenarioCell(
                scenario_id=spec.scenario_id,
                position=position,
                transformer_id=spec.transformer_id,
                runner_id=runner_id,
                parameters=tuple(spec.parameters),
                mutation_tags=frozenset(spec.mutation_tags),
                changed_tags=frozenset(changed),
                invalidate_all=invalidate_all,
                cell_fingerprint=cell_fingerprint,
                group_key=group_key,
            )
        )
        group_positions.setdefault(group_key, []).append(position)

    plan_fingerprint = try_fingerprint(
        (base_fingerprint,
         tuple(c.cell_fingerprint for c in cells),
         tuple(c.scenario_id for c in cells))
    )
    return ScenarioPlan(
        plan_id=plan_fingerprint or "scenario-plan",
        schema_version=SCENARIO_SCHEMA_VERSION,
        base_kind=base_kind,
        base_fingerprint=base_fingerprint,
        engine_factory_id=engine_factory_id,
        cells=tuple(cells),
        groups=tuple(
            (key, tuple(positions))
            for key, positions in group_positions.items()
        ),
    )
