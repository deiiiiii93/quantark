"""Adaptive RQMC compatibility adapter (spec section 8.4, Phase 3).

Sequential COMPATIBILITY mode only: execution runs on the LIVE engine
instance on the calling thread, driving the same run_rqmc_traced loop the
direct path uses, so the stopping sequence and every produced number are
bit-identical to a direct call by construction. No cloning: the 12
Snowball/Phoenix engines (BSM/LV/Heston/QE/SLV variants) flow through their
own polymorphic hooks (_create_path_generator, _compute_payoffs). Each of
the 12 verified concrete classes is registered exact=True (code-gate
finding 2026-07-16): the adapter drives the session-spec seam rather than
engine.price(), so an unknown subclass overriding price() or preamble
behavior must fall through to the legacy adapter and keep its complete
public price path.

Instance exclusion (plan-gate finding 2026-07-16): the engine mutates its
request-scoped state (_term_ctx/_df/_last_result) during the run, exactly as
price() does - so two overlapping session dispatches on the SAME engine
instance could overwrite each other's discount function and return
mixed-market PVs. prepare() therefore acquires a per-engine lock whose
release rides in ``PreparedState.handles``: the kernel's ``finally`` closes
every handle after execution, so the lock spans preparation through
execution and is released even on failure. Dispatches on distinct engine
instances are unaffected. (An engine must not re-dispatch itself inside a
session call; that would self-deadlock, exactly as recursive direct pricing
would corrupt its own request state.)

Parallel-wave adaptive RQMC (a distinct opt-in plan) is deferred on Phase 2
benchmark evidence (host-limited thread scaling).
"""
import threading
import weakref

from quantark.execution.contracts import (
    AdaptivePlan,
    EngineCapabilities,
    OutputKind,
    PreparedState,
    PricingOperation,
)
from quantark.execution.errors import CapabilityError
from quantark.execution.legacy_adapter import LegacyPriceAdapter
from quantark.montecarlo.qmc_rqmc_driver import run_rqmc_traced

__all__ = ["AutocallableAdaptiveMCAdapter"]

ADAPTER_ID = "autocallable-adaptive-mc"
ADAPTER_VERSION = "1"

_PRICE_OUTPUTS = frozenset({OutputKind.PV, OutputKind.ERROR_ESTIMATE})

# Per-engine-instance exclusion (plan-gate finding 2026-07-16).
_ENGINE_LOCKS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_LOCKS_GUARD = threading.Lock()


def _engine_lock(engine) -> threading.Lock:
    with _LOCKS_GUARD:
        lock = _ENGINE_LOCKS.get(engine)
        if lock is None:
            lock = threading.Lock()
            _ENGINE_LOCKS[engine] = lock
        return lock


class _LockHandle:
    """Rides in PreparedState.handles; the kernel's finally releases it."""

    def __init__(self, lock):
        self._lock = lock

    def close(self):
        if self._lock is not None:
            lock, self._lock = self._lock, None
            lock.release()


class AutocallableAdaptiveMCAdapter(LegacyPriceAdapter):
    def __init__(self):
        super().__init__(call_shape="product_env")

    def capabilities(self) -> EngineCapabilities:
        base = super().capabilities()
        return EngineCapabilities(
            operations=base.operations,
            output_kinds=_PRICE_OUTPUTS,
            supported_backends=frozenset({"serial"}),
            fixed_planning=False,
            prepared_state_thread_safe=False,
            instance_reentrant=False,
            process_reconstructable=False,
            deterministic_reduction=True,
            peak_memory_estimate="conservative",
            adapter_id=ADAPTER_ID,
            adapter_version=ADAPTER_VERSION,
        )

    def validate(self, engine, request) -> None:
        allowed = (
            _PRICE_OUTPUTS
            if request.operation is PricingOperation.PRICE
            else frozenset({OutputKind.PV})
        )
        extra = request.outputs - allowed
        if extra:
            raise CapabilityError(
                f"outputs {sorted(k.value for k in extra)} unsupported for "
                f"operation {request.operation.value} via {ADAPTER_ID}"
            )
        if request.pricing_env is None:
            raise CapabilityError(
                "pricing_env is required for product_env engines"
            )

    def prepare(self, engine, request, context) -> PreparedState:
        # Per-engine exclusion FIRST: the spec build below mutates the
        # engine's request-scoped state (_term_ctx/_df), and execution reads
        # it. The lock handle rides in PreparedState.handles; the kernel's
        # finally releases it after execution (or on any failure).
        lock = _engine_lock(engine)
        lock.acquire()
        handle = _LockHandle(lock)
        try:
            # The spec build reproduces the price() preamble (validation +
            # request-scoped engine state); only the PRICE operation takes
            # the adaptive route, so other operations skip it entirely.
            spec = None
            if request.operation is PricingOperation.PRICE:
                spec = engine.build_rqmc_session_spec(
                    request.product, request.pricing_env
                )
        except BaseException:
            handle.close()
            raise
        return PreparedState(
            payload=spec, descriptors=(), fingerprint=None,
            byte_estimate=None, handles=(handle,),
        )

    def plan_adaptive(self, engine, request, prepared, context):
        spec = prepared.payload if prepared is not None else None
        if spec is None:
            return None
        cls = type(engine)
        engine_class_path = f"{cls.__module__}.{cls.__qualname__}"
        seed = int(engine.params.seed)
        dimension = int(spec.dimension) or int(spec.time_steps)
        # One batch alive at a time (RQMC frees batch memory batch-by-batch
        # and the finalizer's statistics batch peaks at the same bound):
        # draw block + in-place transform copy (2 x dimension) + path-node
        # matrix and payoff work arrays (4 x (steps+1)), plus fixed slack.
        est = 8 * spec.paths_per_batch * (
            2 * dimension + 4 * (spec.time_steps + 1)
        ) + (1 << 20)
        return AdaptivePlan(
            plan_id=(
                f"{engine_class_path}:{spec.scheme}:{seed}"
                f":{spec.paths_per_batch}"
                f":{spec.min_batches}-{spec.max_batches}:{spec.target_std!r}"
            ),
            engine_class_path=engine_class_path,
            max_batches=int(spec.max_batches),
            min_batches=int(spec.min_batches),
            paths_per_batch=int(spec.paths_per_batch),
            target_std=float(spec.target_std),
            seed=seed,
            stream_kind="sobol-rqmc",
            stream_layout="batch-shifted-sobol/v1",
            time_steps=int(spec.time_steps),
            dimension=dimension,
            dtype="float64",
            scheme=spec.scheme,
            stopping_rule="welford-batch-means/v1",
            checkpoint_policy="after-each-batch/v1",
            reduction_order="batch-order-welford/v1",
            est_task_peak_bytes=est,
            implementation_fingerprint=f"{ADAPTER_ID}/{ADAPTER_VERSION}",
        )

    def execute_native(self, engine, request, normalized, context,
                       prepared=None):
        # Fail closed (code-gate finding 2026-07-16): ERROR_ESTIMATE is
        # produced only by the adaptive RQMC route. A request that falls
        # back to the native price() path (non-RQMC method, near-expiry)
        # must not succeed while silently omitting a requested output.
        if (
            request.operation is PricingOperation.PRICE
            and OutputKind.ERROR_ESTIMATE in request.outputs
        ):
            raise CapabilityError(
                "ERROR_ESTIMATE requires the adaptive RQMC route; this "
                "request fell back to the native price() path (non-RQMC "
                "method or near-expiry shortcut)"
            )
        return super().execute_native(
            engine, request, normalized, context, prepared=prepared
        )

    def execute_adaptive(self, engine, plan, prepared, context):
        spec = prepared.payload
        mgr = context.lease_manager
        est = plan.est_task_peak_bytes or 0
        # The lease spans finalize too (plan-gate finding 2026-07-16): the
        # Snowball/Phoenix finalizers generate one more full statistics
        # batch, which must stay inside admission control.
        if mgr is not None and est:
            mgr.lease_bytes(est, "task_scratch")
        try:
            result, trace = run_rqmc_traced(
                pricer_fn=spec.pricer_fn,
                path_generator=spec.path_generator,
                max_batches=spec.max_batches,
                target_std=spec.target_std,
                min_batches=spec.min_batches,
            )
            native = spec.finalize(result)
            price = engine._complete_price(spec.product, native)
        finally:
            if mgr is not None and est:
                mgr.release_bytes(est, "task_scratch")
        economics = (
            ("pv", float(price)),
            ("std_error", float(native.std_error)),
        )
        return price, economics, trace
