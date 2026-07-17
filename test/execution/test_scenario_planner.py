"""Scenario planner: normalization, footprint verification, grouping
(spec sections 10.2, 13.2)."""
import dataclasses

import pytest

from quantark.execution.contracts import ScenarioSpec
from quantark.execution.errors import CapabilityError, ValidationGateError
from quantark.execution.scenario import registries
from quantark.execution.scenario.planner import plan_scenarios
from quantark.util.exceptions import ValidationError


@dataclasses.dataclass(frozen=True)
class FakeInputs:
    spot: float
    vol: float
    label: str = "base"


@dataclasses.dataclass(frozen=True)
class OpaqueInputs:
    spot: float
    blob: object = None


class _Mutable:
    def __init__(self):
        self.spot = 100.0


def bump_spot(base, parameters):
    return dataclasses.replace(base, spot=base.spot + parameters["ds"])


def bump_both(base, parameters):
    return dataclasses.replace(
        base, spot=base.spot + parameters["ds"], vol=base.vol + 0.01
    )


def bump_hidden_label(base, parameters):
    # changes a field NOT covered by the component schema
    return dataclasses.replace(base, label="mutated")


def bump_opaque(base, parameters):
    return dataclasses.replace(base, spot=base.spot + parameters["ds"])


def mutate_in_place(base, parameters):
    base.spot += parameters["ds"]
    return base


_COMPONENTS = (
    ("spot", lambda b: b.spot),
    ("vol_surface", lambda b: b.vol),
)

registries.register_transformer(
    "fake-bump-spot/v1", bump_spot,
    allowed_tags=frozenset({"spot"}), components=_COMPONENTS,
)
registries.register_transformer(
    "fake-bump-both/v1", bump_both,
    allowed_tags=frozenset({"spot"}), components=_COMPONENTS,
)
registries.register_transformer(
    "fake-bump-hidden/v1", bump_hidden_label,
    allowed_tags=frozenset({"spot", "vol_surface"}), components=_COMPONENTS,
)
registries.register_transformer(
    "fake-bump-opaque/v1", bump_opaque,
    allowed_tags=frozenset({"spot"}),
    components=(("spot", lambda b: b.spot), ("model_params", lambda b: b.blob)),
)
registries.register_transformer(
    "fake-mutator/v1", mutate_in_place,
    allowed_tags=frozenset({"spot"}),
    components=(("spot", lambda b: b.spot),),
)


def _spec(scenario_id, transformer_id="fake-bump-spot/v1", ds=1.0,
          tags=frozenset({"spot"})):
    return ScenarioSpec(
        scenario_id=scenario_id,
        transformer_id=transformer_id,
        parameters=(("ds", ds),),
        mutation_tags=tags,
    )


BASE = FakeInputs(spot=100.0, vol=0.2)


def test_positions_follow_caller_order():
    plan = plan_scenarios(BASE, [_spec("a", ds=1.0), _spec("b", ds=2.0)], None)
    assert [c.scenario_id for c in plan.cells] == ["a", "b"]
    assert [c.position for c in plan.cells] == [0, 1]
    assert plan.cells[0].changed_tags == frozenset({"spot"})
    assert plan.cells[0].invalidate_all is False


def test_duplicate_scenario_ids_raise():
    with pytest.raises(ValidationError):
        plan_scenarios(BASE, [_spec("a"), _spec("a")], None)


def test_under_declared_mutation_tags_raise():
    # transformer changes spot AND vol, but the spec declares only spot
    with pytest.raises(ValidationGateError):
        plan_scenarios(
            BASE, [_spec("a", transformer_id="fake-bump-both/v1")], None
        )


def test_tag_outside_registration_allowed_tags_raises():
    # spec declares vol_surface, but the registration only allows spot;
    # transformer actually changes both -> vol_surface change is not allowed
    spec = _spec(
        "a", transformer_id="fake-bump-both/v1",
        tags=frozenset({"spot", "vol_surface"}),
    )
    with pytest.raises(ValidationGateError):
        plan_scenarios(BASE, [spec], None)


def test_mutation_escaping_component_schema_raises():
    spec = _spec(
        "a", transformer_id="fake-bump-hidden/v1",
        tags=frozenset({"spot", "vol_surface"}),
    )
    with pytest.raises(ValidationGateError):
        plan_scenarios(BASE, [spec], None)


def test_over_declaration_is_fine():
    plan = plan_scenarios(
        BASE, [_spec("a", tags=frozenset({"spot", "vol_surface"}))], None
    )
    assert plan.cells[0].changed_tags == frozenset({"spot"})


def test_uncanonicalizable_component_marks_invalidate_all():
    base = OpaqueInputs(spot=100.0, blob=object())
    plan = plan_scenarios(
        base, [_spec("a", transformer_id="fake-bump-opaque/v1")], None
    )
    assert plan.cells[0].invalidate_all is True
    assert plan.cells[0].changed_tags == frozenset({"spot"})


def test_in_place_mutation_of_base_raises():
    with pytest.raises(ValidationGateError):
        plan_scenarios(
            _Mutable(), [_spec("a", transformer_id="fake-mutator/v1")], None
        )


def test_unknown_transformer_raises_capability_error():
    with pytest.raises(CapabilityError):
        plan_scenarios(BASE, [_spec("a", transformer_id="nope/v1")], None)


def test_identical_parameters_share_cell_fingerprint():
    plan = plan_scenarios(
        BASE, [_spec("a", ds=1.0), _spec("b", ds=1.0), _spec("c", ds=2.0)],
        None,
    )
    fps = [c.cell_fingerprint for c in plan.cells]
    assert fps[0] == fps[1]
    assert fps[0] != fps[2]


def test_groups_by_runner_and_transformer_first_appearance():
    plan = plan_scenarios(
        BASE,
        [_spec("a"), _spec("b", tags=frozenset({"spot", "vol_surface"})),
         _spec("c", ds=3.0)],
        None,
    )
    assert plan.groups == (
        (("request/v1", "fake-bump-spot/v1"), (0, 1, 2)),
    )
