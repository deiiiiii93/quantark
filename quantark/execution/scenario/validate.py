"""Complete-payload scenario comparison (spec sections 2, 13.4).

``normalized_cell_payload`` is THE canonical comparison payload: the
complete normalized economics (economic fields plus ``numerical.``-prefixed
plan-dependent diagnostics) plus the outcome's native value under the
reserved ``value.native`` path — the cross-backend public value contract is
validated, never assumed (plan-gate finding 4). Operational metadata lives
only in diagnostics and never enters the payload.

The report keeps scenario counts and field counts SEPARATE: 130 field
comparisons across 26 cells are never labeled 130 cells (spec section 2).
``all_scenarios_match`` is computed by this validator and is never accepted
as unverified input.
"""
from dataclasses import dataclass

from quantark.execution.cache.fingerprint import try_fingerprint
from quantark.execution.contracts import PricingFailure

__all__ = [
    "ScenarioComparisonReport",
    "compare_scenario_outcomes",
    "normalized_cell_payload",
]


@dataclass(frozen=True)
class ScenarioComparisonReport:
    scenarios_compared: int
    scenarios_matching: int
    fields_compared: int
    fields_matching: int
    missing_fields: tuple    # paths present on the left only
    extra_fields: tuple      # paths present on the right only
    first_mismatch_path: str | None
    all_scenarios_match: bool


def _value_leaf(value):
    """Comparable leaf for the native value: exact for numerics/None,
    best-effort canonical fingerprint (or type name) for native objects."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    fp = try_fingerprint(value)
    if fp is not None:
        return fp
    return f"<uncomparable:{type(value).__module__}.{type(value).__qualname__}>"


def normalized_cell_payload(outcome) -> dict:
    payload = dict(outcome.normalized_economics)
    payload["value.native"] = _value_leaf(outcome.value)
    return payload


def _leaves(node, prefix, out) -> None:
    if isinstance(node, dict):
        items = node.items()
    elif isinstance(node, (list, tuple)):
        pairs = all(
            isinstance(entry, (list, tuple)) and len(entry) == 2
            and isinstance(entry[0], str)
            for entry in node
        )
        if pairs and node:
            items = node
        else:
            items = ((str(i), entry) for i, entry in enumerate(node))
    else:
        out[prefix] = node
        return
    for key, value in items:
        path = f"{prefix}.{key}" if prefix else str(key)
        _leaves(value, path, out)


def compare_scenario_outcomes(left, right) -> ScenarioComparisonReport:
    left_by_id = {_item_id(o): o for o in left}
    right_by_id = {_item_id(o): o for o in right}
    all_ids = list(left_by_id)
    all_ids += [i for i in right_by_id if i not in left_by_id]

    scenarios_matching = 0
    fields_compared = 0
    fields_matching = 0
    missing: list = []
    extra: list = []
    first_mismatch = None

    for scenario_id in all_ids:
        left_item = left_by_id.get(scenario_id)
        right_item = right_by_id.get(scenario_id)
        if left_item is None or right_item is None:
            if first_mismatch is None:
                first_mismatch = f"{scenario_id}:<absent on one side>"
            continue
        if isinstance(left_item, PricingFailure) or isinstance(
            right_item, PricingFailure
        ):
            if first_mismatch is None:
                first_mismatch = f"{scenario_id}:<pricing failure>"
            continue

        left_leaves: dict = {}
        right_leaves: dict = {}
        _leaves(normalized_cell_payload(left_item), "", left_leaves)
        _leaves(normalized_cell_payload(right_item), "", right_leaves)

        cell_matches = True
        for path in left_leaves:
            if path not in right_leaves:
                missing.append(f"{scenario_id}:{path}")
                cell_matches = False
                continue
            fields_compared += 1
            left_value = left_leaves[path]
            right_value = right_leaves[path]
            equal = (
                left_value == right_value
                and type(left_value) is type(right_value)
            ) or (left_value is None and right_value is None)
            if equal:
                fields_matching += 1
            else:
                cell_matches = False
                if first_mismatch is None:
                    first_mismatch = f"{scenario_id}:{path}"
        for path in right_leaves:
            if path not in left_leaves:
                extra.append(f"{scenario_id}:{path}")
                cell_matches = False
        if cell_matches:
            scenarios_matching += 1
        elif first_mismatch is None:
            mismatch_paths = [
                p.split(":", 1)[1] for p in missing + extra
                if p.startswith(f"{scenario_id}:")
            ]
            first_mismatch = f"{scenario_id}:{mismatch_paths[0]}"

    scenarios_compared = len(all_ids)
    return ScenarioComparisonReport(
        scenarios_compared=scenarios_compared,
        scenarios_matching=scenarios_matching,
        fields_compared=fields_compared,
        fields_matching=fields_matching,
        missing_fields=tuple(missing),
        extra_fields=tuple(extra),
        first_mismatch_path=first_mismatch,
        all_scenarios_match=(
            scenarios_matching == scenarios_compared and scenarios_compared > 0
        ),
    )


def _item_id(outcome) -> str:
    if isinstance(outcome, PricingFailure):
        return outcome.item_id
    return outcome.scenario_id
