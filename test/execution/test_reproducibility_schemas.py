"""Reproducibility JSON Schemas validated against LIVE payloads (Phase 6).

The four checked-in schemas under docs/execution/schemas/ must accept every
payload the framework actually emits and reject structural drift in both
directions (missing required fields, unknown fields, version skew).
"""
import dataclasses
import json
import pathlib

import pytest

jsonschema = pytest.importorskip("jsonschema")

from quantark.execution.context import default_context
from quantark.execution.contracts import ScenarioSpec
from quantark.execution.errors import CapabilityError
from quantark.execution.scenario.contracts import BaseInputsRef
from quantark.execution.scenario.planner import plan_scenarios
from quantark.execution.scenario.validate import normalized_cell_payload
from quantark.execution.scenario.worker import (
    _cell_payload,
    build_worker_spec,
    payload_to_worker_spec,
    verify_worker_environment,
    worker_spec_to_payload,
)

import execution.scenario_process_helpers  # noqa: F401 - registers toy fixtures

SCHEMA_DIR = pathlib.Path(__file__).parents[2] / "docs" / "execution" / "schemas"


def _schema(name: str) -> dict:
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


def _validate(payload, schema_name: str) -> None:
    jsonschema.validate(
        instance=payload,
        schema=_schema(schema_name),
        cls=jsonschema.Draft202012Validator,
    )


def _toy_base():
    return BaseInputsRef(
        factory_id="toy-inputs/v1",
        payload=(("spot", 100.0), ("vol", 0.2), ("grid", ((1.0, 2.0), (3.0,)))),
    )


def _toy_plan(n=2):
    specs = [
        ScenarioSpec(
            scenario_id=f"s{i}",
            transformer_id="toy-bump/v1",
            parameters=(("ds", float(i)),),
            mutation_tags=frozenset({"spot"}),
            required_capabilities=frozenset({"runner:toy/v1"}),
        )
        for i in range(n)
    ]
    return plan_scenarios(_toy_base(), specs, "toy-engine/v1")


def _json_roundtrip(payload):
    return json.loads(json.dumps(payload))


# ------------------------------------------------------------ worker-spec


def test_worker_spec_payload_validates():
    spec = build_worker_spec(_toy_plan(), _toy_base(), default_context(), workers=2)
    payload = _json_roundtrip(worker_spec_to_payload(spec))
    _validate(payload, "worker-spec.v1.schema.json")


def test_worker_spec_schema_rejects_missing_and_unknown_fields():
    spec = build_worker_spec(_toy_plan(), _toy_base(), default_context(), workers=2)
    payload = _json_roundtrip(worker_spec_to_payload(spec))

    missing = dict(payload)
    del missing["callable_refs"]
    with pytest.raises(jsonschema.ValidationError):
        _validate(missing, "worker-spec.v1.schema.json")

    unknown = dict(payload)
    unknown["surprise"] = 1
    with pytest.raises(jsonschema.ValidationError):
        _validate(unknown, "worker-spec.v1.schema.json")


def test_worker_spec_version_pin_matches_reader_rejection():
    """Schema `const` and the worker-side reader reject the same skew."""
    spec = build_worker_spec(_toy_plan(), _toy_base(), default_context(), workers=2)
    payload = _json_roundtrip(worker_spec_to_payload(spec))
    payload["schema_version"] = "scenario/v999"

    with pytest.raises(jsonschema.ValidationError):
        _validate(payload, "worker-spec.v1.schema.json")
    with pytest.raises(CapabilityError):
        verify_worker_environment(payload_to_worker_spec(payload))


def test_run_worker_cell_rejects_skew_before_any_import():
    """Entry-point gate (code-gate 2026-07-18): an unknown schema version
    must be rejected on the RAW payload — before parsing, sys.path
    mutation, or payload-selected imports can contaminate a long-lived
    worker."""
    import sys

    from quantark.execution.scenario.worker import run_worker_cell

    spec = build_worker_spec(_toy_plan(), _toy_base(), default_context(), workers=2)
    payload = _json_roundtrip(worker_spec_to_payload(spec))
    payload["schema_version"] = "scenario/v999"
    # Point a callable ref at a module that is definitely not imported yet:
    # if the gate ran after imports, this would land in sys.modules.
    sentinel = "fractions"
    sys.modules.pop(sentinel, None)
    payload["callable_refs"] = [
        ["factory", "toy-inputs/v1", sentinel, "Fraction", "1"]
    ]
    cell = _json_roundtrip(_cell_payload(_toy_plan().cells[0]))

    result = run_worker_cell(payload, cell, None)

    assert result["error"] is not None
    assert result["error"]["type"] == "CapabilityError"
    assert "scenario/v999" in result["error"]["message"]
    assert sentinel not in sys.modules, (
        "worker imported a payload-selected module before the schema gate"
    )


# ------------------------------------------------------------ scenario-cell


def test_cell_payloads_validate():
    plan = _toy_plan(3)
    for cell in plan.cells:
        _validate(_json_roundtrip(_cell_payload(cell)), "scenario-cell.v1.schema.json")


def test_cell_schema_rejects_missing_and_unknown_fields():
    payload = _json_roundtrip(_cell_payload(_toy_plan().cells[0]))

    missing = dict(payload)
    del missing["runner_id"]
    with pytest.raises(jsonschema.ValidationError):
        _validate(missing, "scenario-cell.v1.schema.json")

    unknown = dict(payload)
    unknown["mutation_tags"] = ["spot"]
    with pytest.raises(jsonschema.ValidationError):
        _validate(unknown, "scenario-cell.v1.schema.json")


# ------------------------------------------------------------ manifest


def _live_manifest():
    from datetime import datetime

    from quantark.asset.equity.engine.mc import EuropeanMCEngine
    from quantark.asset.equity.param import MCParams
    from quantark.asset.equity.product.option import EuropeanVanillaOption
    from quantark.execution.api import PricingSession
    from quantark.execution.contracts import PricingRequest
    from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
    from quantark.priceenv import PricingEnvironment
    from quantark.util.enum import OptionType

    env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.05),
        valuation_date=datetime(2024, 1, 1),
    )
    option = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )
    with PricingSession() as session:
        outcome = session.execute(
            EuropeanMCEngine(params=MCParams(num_paths=2_000, seed=7)),
            PricingRequest(product=option, pricing_env=env),
        )
    return outcome.manifest


def test_live_manifest_validates():
    payload = _json_roundtrip(dataclasses.asdict(_live_manifest()))
    _validate(payload, "execution-manifest.v0.schema.json")


def test_manifest_schema_rejects_version_skew_and_unknown_fields():
    payload = _json_roundtrip(dataclasses.asdict(_live_manifest()))

    skewed = dict(payload)
    skewed["schema_version"] = "execution-manifest/999"
    with pytest.raises(jsonschema.ValidationError):
        _validate(skewed, "execution-manifest.v0.schema.json")

    unknown = dict(payload)
    unknown["surprise"] = True
    with pytest.raises(jsonschema.ValidationError):
        _validate(unknown, "execution-manifest.v0.schema.json")


# ------------------------------------------------------ normalized economics


def _toy_outcomes():
    from quantark.execution.api import PricingSession

    with PricingSession() as session:
        return session.run_scenarios(
            _toy_base(),
            [
                ScenarioSpec(
                    scenario_id="s0",
                    transformer_id="toy-bump/v1",
                    parameters=(("ds", 1.0),),
                    mutation_tags=frozenset({"spot"}),
                    required_capabilities=frozenset({"runner:toy/v1"}),
                )
            ],
            "toy-engine/v1",
        )


def test_normalized_economics_float_value_validates():
    outcome = _toy_outcomes()[0]
    payload = _json_roundtrip(normalized_cell_payload(outcome))
    assert isinstance(payload["value.native"], float)
    _validate(payload, "normalized-economics.v1.schema.json")


def test_normalized_economics_full_leaf_union_validates():
    """Non-numeric natives normalize to fingerprint strings; economics
    leaves may be booleans and strings — the schema models the full union."""
    outcome = _toy_outcomes()[0]
    widened = dataclasses.replace(
        outcome,
        value=(1.0, 2.0),  # native object -> canonical fingerprint string
        normalized_economics=outcome.normalized_economics
        + (
            ("numerical.adaptive.stopped_early", True),
            ("engine.label", "toy"),
            ("numerical.missing", None),
        ),
    )
    payload = _json_roundtrip(normalized_cell_payload(widened))
    assert isinstance(payload["value.native"], str)
    _validate(payload, "normalized-economics.v1.schema.json")


def test_normalized_economics_rejects_structured_leaf():
    outcome = _toy_outcomes()[0]
    payload = _json_roundtrip(normalized_cell_payload(outcome))
    payload["not.a.leaf"] = {"nested": 1}
    with pytest.raises(jsonschema.ValidationError):
        _validate(payload, "normalized-economics.v1.schema.json")
