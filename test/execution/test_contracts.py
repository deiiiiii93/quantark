"""Frozen framework value objects (spec sections 5, 14.3, 16)."""
import dataclasses

import pytest


class TestDiagnosticsAndManifest:
    def test_run_diagnostics_is_frozen(self):
        from quantark.execution.diagnostics import RunDiagnostics

        diag = RunDiagnostics(
            adapter_id="legacy-price",
            timings=(("execute_seconds", 0.5),),
            policy_sources=(("batch.backend", "default"),),
            records=(),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            diag.adapter_id = "x"

    def test_in_memory_sink_collects(self):
        from quantark.execution.diagnostics import (
            InMemoryDiagnosticsSink,
            RunDiagnostics,
        )

        sink = InMemoryDiagnosticsSink()
        diag = RunDiagnostics(
            adapter_id="a", timings=(), policy_sources=(), records=()
        )
        sink.emit(diag)
        assert sink.entries == [diag]

    def test_manifest_versions_stamped(self):
        from quantark.execution.manifest import build_versions

        versions = dict(build_versions())
        assert set(versions) == {"python", "quantark", "numpy", "scipy"}
        assert all(isinstance(v, str) and v for v in versions.values())

    def test_manifest_is_frozen_with_schema_version(self):
        from quantark.execution.manifest import (
            MANIFEST_SCHEMA_VERSION,
            ReproducibilityManifest,
            build_versions,
        )

        manifest = ReproducibilityManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            request_fingerprint=None,
            plan_fingerprint=None,
            adapter_id="legacy-price",
            adapter_version="0",
            engine_class_path="x.Y",
            versions=build_versions(),
            platform="test",
            resolved_policy=(),
        )
        assert manifest.schema_version == "execution-manifest/0"
        with pytest.raises(dataclasses.FrozenInstanceError):
            manifest.platform = "other"


class TestRequestAndOutcomeContracts:
    def test_pricing_request_defaults(self):
        from quantark.execution.contracts import (
            DEFAULT_OUTPUTS,
            OutputKind,
            PricingOperation,
            PricingRequest,
        )

        req = PricingRequest(product="P", pricing_env="E")
        assert req.operation is PricingOperation.PRICE
        assert req.outputs == DEFAULT_OUTPUTS == frozenset({OutputKind.PV})
        assert req.operation_options == ()
        assert req.request_id is None

    def test_pricing_request_is_frozen(self):
        from quantark.execution.contracts import PricingRequest

        req = PricingRequest(product="P")
        with pytest.raises(dataclasses.FrozenInstanceError):
            req.product = "Q"

    def test_env_bound_request_allows_missing_env(self):
        from quantark.execution.contracts import PricingRequest

        req = PricingRequest(product="bond")
        assert req.pricing_env is None

    def test_outcome_and_failure_are_frozen(self):
        from quantark.execution.contracts import (
            FrameworkErrorInfo,
            PricingFailure,
            PricingOutcome,
            economics_mapping,
        )
        from quantark.execution.diagnostics import RunDiagnostics

        diag = RunDiagnostics(adapter_id="a")
        outcome = PricingOutcome(
            value=1.25,
            normalized_economics=(("pv", 1.25),),
            diagnostics=diag,
            manifest=None,
        )
        assert economics_mapping(outcome) == {"pv": 1.25}
        with pytest.raises(dataclasses.FrozenInstanceError):
            outcome.value = 2.0

        failure = PricingFailure(
            item_id="0",
            error=FrameworkErrorInfo("ValueError", "bad"),
            diagnostics=diag,
        )
        assert failure.manifest is None

    def test_engine_capabilities_and_scenario_contracts_exist(self):
        from quantark.execution.contracts import (
            EngineCapabilities,
            OutputKind,
            PricingOperation,
            ScenarioOutcome,
            ScenarioSpec,
        )

        caps = EngineCapabilities(
            operations=frozenset({PricingOperation.PRICE}),
            output_kinds=frozenset({OutputKind.PV}),
            supported_backends=frozenset({"serial"}),
            fixed_planning=None,
            prepared_state_thread_safe=False,
            instance_reentrant=False,
            process_reconstructable=False,
            deterministic_reduction=True,
            peak_memory_estimate="unavailable",
            adapter_id="legacy-price",
            adapter_version="0",
        )
        assert "serial" in caps.supported_backends
        spec = ScenarioSpec(
            scenario_id="s1",
            transformer_id="t",
            parameters=(("shift", 0.01),),
            mutation_tags=frozenset({"spot"}),
        )
        assert spec.required_capabilities == frozenset()
        assert ScenarioOutcome is not None
