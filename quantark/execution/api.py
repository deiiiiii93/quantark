"""PricingSession: the explicit framework entry point (spec section 5.5)."""
from quantark.execution.contracts import (
    FrameworkErrorInfo,
    PricingFailure,
    PricingRequest,
)
from quantark.execution.context import PricingRunContext, default_context
from quantark.execution.errors import CapabilityError
from quantark.execution.kernel import ExecutionKernel
from quantark.execution.registry import build_default_registry

__all__ = ["PricingSession"]


class PricingSession:
    """Serial pricing session. Owns only services it creates; idempotently
    closable. ``PricingSession()`` with no context resolves a safe serial
    default exactly once at construction (spec section 11.1)."""

    def __init__(self, context: PricingRunContext | None = None):
        if context is None:
            context = default_context()
        if context.adapter_registry is None:
            import dataclasses

            registry = build_default_registry()
            context = dataclasses.replace(context, adapter_registry=registry)
        # Registries freeze for the lifetime of a session (spec section 6.2)
        # — including caller-supplied ones, so a retained handle cannot alter
        # adapter resolution mid-session. freeze() is idempotent.
        context.adapter_registry.freeze()
        # Cache and lease manager form ONE budget domain: supplied as a
        # validated pair or created as a pair; partial injection is rejected.
        # Supplied (borrowed) services are never closed by the session.
        supplied_cache = context.artifact_cache
        supplied_leases = context.lease_manager
        self._owned_cache = None
        self._owned_leases = None
        self._owned_draw_repo = None
        if (supplied_cache is None) != (supplied_leases is None):
            from quantark.util.exceptions import ValidationError

            raise ValidationError(
                "artifact_cache and lease_manager must be supplied together "
                "(one shared budget domain) or both omitted"
            )
        if supplied_cache is not None:
            if supplied_cache.lease_manager is not supplied_leases:
                from quantark.util.exceptions import ValidationError

                raise ValidationError(
                    "supplied artifact_cache is not backed by the supplied "
                    "lease_manager; budget domains would split"
                )
        else:
            import dataclasses
            import os

            from quantark.execution.cache.artifacts import PreparedArtifactCache
            from quantark.execution.leases import ResourceLeaseManager

            budget = context.resource_budget
            # Safe auto budget (spec section 11.1), applied ONLY to
            # DEFAULT-sourced fields: an operator's env-resolved or
            # explicit limit — including the valid value 1 — is never
            # upgraded (code-gate finding 2026-07-16). The source map is
            # the context's config_snapshot from budget resolution. Batch
            # tasks hold per-task slots (Phase 2), so the resolution
            # default max_in_flight=1 would serialize every threaded plan.
            sources = dict(context.config_snapshot)
            upgrades = {}
            if budget.artifact_cache_bytes is None:
                upgrades["artifact_cache_bytes"] = 512 * 2**20
            if budget.draw_cache_bytes is None:
                upgrades["draw_cache_bytes"] = 512 * 2**20
            cpus = os.cpu_count() or 1
            if budget.max_threads == 1 and sources.get("max_threads") == "default":
                upgrades["max_threads"] = cpus
            if (budget.max_in_flight == 1
                    and sources.get("max_in_flight") == "default"):
                upgrades["max_in_flight"] = cpus
            # Phase 5: the processes scenario backend needs workers; a
            # DEFAULT-sourced max_processes upgrades exactly like
            # max_threads, and an operator's explicit/env value never does.
            if (budget.max_processes == 1
                    and sources.get("max_processes") == "default"):
                upgrades["max_processes"] = cpus
            if upgrades:
                budget = dataclasses.replace(budget, **upgrades)
            leases = ResourceLeaseManager(budget)
            cache = PreparedArtifactCache(leases)
            self._owned_leases = leases
            self._owned_cache = cache
            context = dataclasses.replace(
                context, resource_budget=budget,
                lease_manager=leases, artifact_cache=cache,
                draw_repository=None,  # rebuilt below on the owned leases
            )
        # DrawRepository shares the SAME budget domain: a supplied one must
        # be backed by the (possibly supplied) lease manager; if absent the
        # session creates and owns one on that manager.
        if context.draw_repository is not None:
            if context.draw_repository.lease_manager is not context.lease_manager:
                from quantark.util.exceptions import ValidationError

                raise ValidationError(
                    "supplied draw_repository is not backed by the session "
                    "lease_manager; budget domains would split"
                )
        else:
            import dataclasses

            from quantark.execution.cache.draws import DrawRepository

            repo = DrawRepository(context.lease_manager)
            self._owned_draw_repo = repo
            context = dataclasses.replace(context, draw_repository=repo)
        self._context = context
        self._closed = False

    @property
    def context(self) -> PricingRunContext:
        return self._context

    def execute(self, engine, request: PricingRequest):
        self._ensure_open()
        return ExecutionKernel.dispatch(engine, request, self._context)

    def price(self, engine, product, pricing_env=None):
        outcome = self.execute(
            engine, PricingRequest(product=product, pricing_env=pricing_env)
        )
        return outcome.value

    def price_many(self, items, *, collect_errors: bool = False) -> list:
        """Serial, caller-ordered pricing of (engine, PricingRequest) pairs.

        Fail-fast by default with pure caller-order execution (zero
        behavior change from Phase 0). ``collect_errors=True`` returns a
        ``PricingFailure`` in place of each failed item (spec section 15)
        and executes GROUPED by engine class via the scenario grouping
        planner (spec section 13.3), so session caches see contiguous
        compatible work; results still return in caller order.
        """
        self._ensure_open()
        items = list(items)
        if not collect_errors:
            return [
                self.execute(engine, request).value
                for engine, request in items
            ]
        from quantark.execution.scenario.planner import plan_price_groups

        results: list = [None] * len(items)
        for _group_key, indices in plan_price_groups(items):
            for index in indices:
                engine, request = items[index]
                try:
                    results[index] = self.execute(engine, request).value
                except Exception as exc:  # noqa: BLE001 - typed into the failure record
                    item_id = request.request_id or str(index)
                    from quantark.execution.diagnostics import RunDiagnostics

                    results[index] = PricingFailure(
                        item_id=item_id,
                        error=FrameworkErrorInfo(
                            error_type=type(exc).__name__, message=str(exc)
                        ),
                        diagnostics=RunDiagnostics(adapter_id="unresolved"),
                    )
        return results

    def run_scenarios(self, base_request, scenario_specs, engine_factory,
                      *, collect_errors: bool = False) -> list:
        """Ordered scenario execution (spec section 13).

        ``base_request`` is a live ``PricingRequest`` (serial/threads
        backends) or a ``BaseInputsRef`` naming a registered factory
        (required for the processes/dask backends). ``engine_factory`` is
        a callable ``parameters -> engine`` or a registered factory id.
        Returns caller-ordered ``ScenarioOutcome`` items; with
        ``collect_errors=True`` failed cells become ``PricingFailure``.
        """
        self._ensure_open()
        from quantark.execution.scenario.contracts import BaseInputsRef
        from quantark.execution.scenario.planner import (
            plan_scenarios,
            resolve_base,
        )
        from quantark.execution.scenario.runner import run_plan

        backend = self._context.execution_policy.scenario.backend
        if backend in ("processes", "dask") and not isinstance(
            base_request, BaseInputsRef
        ):
            raise CapabilityError(
                f"the {backend!r} scenario backend requires a BaseInputsRef "
                "base (registered factory); live request objects cannot "
                "cross a process boundary and explicit requests never "
                "silently fall back (spec section 3.3)"
            )
        # The base factory runs exactly ONCE in the parent: planning and
        # serial execution share this instance (code-gate 2026-07-17).
        _, resolved_inputs, _ = resolve_base(base_request)
        plan = plan_scenarios(
            base_request, scenario_specs, engine_factory,
            resolved=resolved_inputs,
        )
        return run_plan(
            plan, base_request, engine_factory, self._context,
            resolved_base=resolved_inputs, collect_errors=collect_errors,
        )

    def run_scenario_plans(self, plan_inputs, engine_factory=None, *,
                           collect_errors: bool = False) -> list:
        """Plan and execute MANY ``(base, specs)`` scenario plans through
        ONE bounded-window pool (spec 2026-07-20: the portfolio × bumps
        shape). Returns per-plan outcome lists aligned with
        ``plan_inputs`` order.

        Per-plan error boundary: with ``collect_errors=True`` a
        base-resolution or planning failure in one plan yields an aligned
        list of ``PricingFailure`` (one per spec, carrying the original
        exception type and message) while every other plan still executes;
        with ``collect_errors=False`` the first failure raises. Serial and
        processes backends only.
        """
        self._ensure_open()
        from quantark.execution.contracts import (
            FrameworkErrorInfo,
            PricingFailure,
        )
        from quantark.execution.diagnostics import RunDiagnostics
        from quantark.execution.scenario.contracts import BaseInputsRef
        from quantark.execution.scenario.planner import (
            plan_scenarios,
            resolve_base,
        )
        from quantark.execution.scenario.runner import run_plan

        backend = self._context.execution_policy.scenario.backend
        if backend not in ("serial", "processes"):
            raise CapabilityError(
                "run_scenario_plans supports the serial and processes "
                f"scenario backends only, got {backend!r}; explicit "
                "requests never silently fall back (spec section 3.3)"
            )
        entries = [(base, list(specs)) for base, specs in plan_inputs]
        if backend == "processes":
            for base_request, _specs in entries:
                if not isinstance(base_request, BaseInputsRef):
                    raise CapabilityError(
                        "the processes scenario backend requires "
                        "BaseInputsRef bases (registered factories); live "
                        "request objects cannot cross a process boundary"
                    )

        results: list = [None] * len(entries)
        planned: list = []  # (entry_index, plan, base_request, resolved)
        for index, (base_request, specs) in enumerate(entries):
            try:
                # The base factory runs exactly ONCE in the parent per
                # plan: planning and serial execution share the instance.
                _, resolved_inputs, _ = resolve_base(base_request)
                plan = plan_scenarios(
                    base_request, specs, engine_factory,
                    resolved=resolved_inputs,
                )
            except Exception as exc:  # noqa: BLE001 - typed into failures
                if not collect_errors:
                    raise
                results[index] = [
                    PricingFailure(
                        item_id=f"plan:{index}:{spec.scenario_id}",
                        error=FrameworkErrorInfo(
                            error_type=type(exc).__name__, message=str(exc)
                        ),
                        diagnostics=RunDiagnostics(
                            adapter_id="scenario-plans"
                        ),
                    )
                    for spec in specs
                ]
                continue
            planned.append((index, plan, base_request, resolved_inputs))

        if backend == "serial":
            for index, plan, base_request, resolved_inputs in planned:
                results[index] = run_plan(
                    plan, base_request, engine_factory, self._context,
                    resolved_base=resolved_inputs,
                    collect_errors=collect_errors,
                )
            return results

        from quantark.execution.scenario import worker as worker_mod

        executed = worker_mod.run_plans_processes(
            [(plan, base_request)
             for _index, plan, base_request, _resolved in planned],
            engine_factory, self._context, collect_errors=collect_errors,
        )
        for (index, _plan, _base, _resolved), plan_results in zip(
            planned, executed
        ):
            results[index] = plan_results
        return results

    def close(self) -> None:
        if not self._closed and self._owned_draw_repo is not None:
            self._owned_draw_repo.close()
        if not self._closed and self._owned_cache is not None:
            self._owned_cache.close()
        if not self._closed and self._owned_leases is not None:
            self._owned_leases.close()
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise CapabilityError("PricingSession is closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
