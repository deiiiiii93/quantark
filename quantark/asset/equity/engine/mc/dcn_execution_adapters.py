"""Execution-framework adapters for the DCN engines (spec Phases 1-2).

Phase 1 (surfaces): the adapter path serves cache-fetched Dupire surfaces
through PREBUILT-SURFACE FACTORY CLONES: a clone of the target engine is
constructed with ``local_vol_surface=<cached surface>``, so the original
engine's ``_prepare_simulation``/``_resolve_surface`` hooks and
``_active_surface`` state are never touched (spec sections 6.3 + 17.1:
mutable-state removal applies to the adapter path only; the direct path and
its subclass hooks are preserved verbatim).

Preparation inputs are BOUND once, so the cache key, the build, and the
post-build verification all see the same objects; a fingerprint mismatch
after the build raises ``DeterminismViolation`` (spec section 5.1).

Phase 2 (fixed-batch MC): the MC adapters implement the batch capability
(``plan_batches``/``execute_batch``/``reduce_batches``). ``execute_batch``
returns COMPACT per-batch sufficient statistics whose merge in canonical
batch-index order performs exactly the float additions the legacy
per-batch accumulator loop performs, so session results are BIT-IDENTICAL
to the direct path in every mode; per-path totals are retained only when
the pathwise-IID stderr formula requires them (spec section 8.2 "compact
whenever possible"). Sobol draws are served by the session
``DrawRepository`` through a thread-safe pinning provider that rides in
``PreparedState.handles`` (the kernel's ``finally`` releases every pinned
master); non-Sobol modes fall through to the clone's legacy generators.
"""
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from quantark.asset.equity.engine.mc.dcn_mc_engine import (
    _finalize_dcn_result,
    _LegAccumulator,
)
from quantark.asset.equity.engine.mc.dcn_payoff import compute_dcn_cashflows
from quantark.asset.equity.engine.mc.term_inputs import build_mc_term_inputs
from quantark.asset.equity.product.option.dcn_grid import build_dcn_grid_context
from quantark.execution.cache.artifacts import ArtifactDescriptor
from quantark.execution.cache.fingerprint import try_fingerprint
from quantark.execution.contracts import (
    BatchOutcome,
    BatchPlan,
    BatchTask,
    EngineCapabilities,
    OutputKind,
    PreparedState,
    PricingOperation,
)
from quantark.execution.errors import CapabilityError
from quantark.execution.legacy_adapter import LegacyPriceAdapter
from quantark.util.exceptions import ValidationError
from quantark.volmodels.localvol import build_dupire_local_vol

__all__ = [
    "DCNBatchMCAdapter",
    "DCNCoupledCoarseHestonBatchMCAdapter",
    "DCNHestonBatchMCAdapter",
    "DCNLocalVolMCAdapter",
    "DCNLocalVolPDEAdapter",
    "DCNQEBatchMCAdapter",
]

_DUPIRE_TAGS = frozenset({"vol_surface", "spot", "rate_curve", "dividend_curve"})
_BUILDER_VERSION = "1"

_BATCH_ADAPTER_ID = "dcn-batch-mc"
_BATCH_ADAPTER_VERSION = "1"
_BATCH_OUTPUTS = frozenset(
    {OutputKind.PV, OutputKind.ERROR_ESTIMATE, OutputKind.EVENT_STATS}
)
_BATCH_OPS = frozenset(
    {PricingOperation.PRICE, PricingOperation.PRICE_DETAILED}
)


def _estimate_surface_bytes(vol_surface) -> int:
    """Pre-build admission estimate PROPORTIONAL to the source grid.

    The Dupire local-vol grid has the same two-dimensional scale as the
    input IV grid; the factor covers the derived arrays plus interpolants.
    The cache re-measures the built surface (``measure=``) and adjusts the
    charge to actual bytes, so this only needs to be the right order of
    magnitude for reservation.
    """
    iv_nbytes = getattr(getattr(vol_surface, "iv_grid", None), "nbytes", None)
    if isinstance(iv_nbytes, int) and iv_nbytes > 0:
        return 8 * iv_nbytes + (256 << 10)
    return 8 << 20  # unknown source grid: conservative 8 MiB


def _surface_nbytes(surface) -> int:
    total = 0
    for value in vars(surface).values():
        nbytes = getattr(value, "nbytes", None)
        if isinstance(nbytes, int):
            total += nbytes
    return total or (1 << 20)  # conservative floor: 1 MiB


# ---------------------------------------------------------------------------
# Phase 2: fixed-batch MC capability
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DCNBatchStats:
    """Compact per-batch sufficient statistics: exactly the increments the
    legacy ``_LegAccumulator.add`` produces from one batch, so merging them
    in batch order reproduces the direct-path arithmetic bit-for-bit."""

    n: int
    fixed_sum: float
    ko_sum: float
    loss_sum: float
    fixed_by_period: np.ndarray
    ko_by_period: np.ndarray
    coupon_paid: np.ndarray
    ko_timing: np.ndarray
    ki_count: int
    survive_no_ki_count: int
    survive_ki_count: int
    life_sum: float
    batch_mean: float
    totals: Optional[np.ndarray]  # per-path totals; pathwise_iid mode only


@dataclass(frozen=True)
class _DCNSimContext:
    """Immutable request-scoped simulation context (spec section 6.3):
    everything a batch task needs, hoisted once out of the batch loop —
    the same work the direct path performs once per ``price_detailed``.

    ``captured`` binds the market-data fields consumed during execution
    (code-gate finding 2026-07-16): the discount function is built from the
    CAPTURED rate curve, and ``reduce_batches`` verifies the environment
    fields and product fingerprint after execution, so a mid-run field
    replacement fails closed instead of silently mixing market states."""

    engine: object          # adapter-owned clone; num_workers=1 (nested off)
    product: object
    pricing_env: object
    ctx: object
    term: object
    df: object
    spot0: float
    dt_array: np.ndarray
    obs_times: np.ndarray
    n_obs: int
    stderr_mode: str
    t0: float
    captured: tuple = ()
    product_fp: str | None = None
    draw_provider: object | None = None
    # Heston-family engines carry market state on model_params rather than
    # the env vol surface; its fingerprint is verified after execution too.
    model_fp: str | None = None


def _make_captured_df(rate_curve):
    """Vectorized discount factors bound to the CAPTURED rate curve (same
    values as ``make_df_fn(env)``, which delegates to
    ``env.rate_curve.get_discount_factor``, but immune to a mid-run
    ``env.rate_curve`` replacement)."""

    def f(t):
        arr = np.asarray(t, dtype=float)
        if arr.ndim == 0:
            return float(rate_curve.get_discount_factor(float(arr)))
        return np.array(
            [rate_curve.get_discount_factor(float(x)) for x in arr.ravel()]
        ).reshape(arr.shape)

    return f


class _PinnedDrawProvider:
    """Thread-safe draw fetcher with PER-BATCH handle scoping.

    ``execute_batch`` brackets each simulation with ``begin_batch`` /
    ``end_batch`` so a batch's blocks are unpinned (and, for cache-bypassed
    blocks, released to the garbage collector) as soon as the simulation
    has consumed them — retention never grows with the number of batches
    (code-gate finding 2026-07-16). Scoping is thread-local, so concurrent
    batch tasks never close each other's handles. The provider also rides
    in ``PreparedState.handles`` as a backstop: the kernel's ``finally``
    closes anything fetched outside a batch bracket."""

    def __init__(self, repository, seed):
        self._repo = repository
        self._seed = seed
        self._lock = threading.Lock()
        self._local = threading.local()
        self._unscoped = []

    def __call__(self, n_dims, n_paths, batch_id):
        handle = self._repo.normals_handle(
            seed=self._seed, n_paths=n_paths, dim=n_dims, batch_id=batch_id
        )
        scoped = getattr(self._local, "handles", None)
        if scoped is not None:
            scoped.append(handle)
        else:
            with self._lock:
                self._unscoped.append(handle)
        return handle.value

    def begin_batch(self):
        self._local.handles = []

    def end_batch(self):
        handles = getattr(self._local, "handles", None) or ()
        self._local.handles = None
        for handle in handles:
            handle.close()

    def retained_handle_count(self) -> int:
        with self._lock:
            return len(self._unscoped)

    def close(self):
        with self._lock:
            handles, self._unscoped = self._unscoped, []
        for handle in handles:
            handle.close()


class _DCNBatchMixin:
    """Batch capability shared by the GBM and Local-Vol DCN MC adapters."""

    _SCHEME = "gbm-term/v1"

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            operations=_BATCH_OPS,
            output_kinds=_BATCH_OUTPUTS,
            supported_backends=frozenset({"serial", "threads"}),
            fixed_planning=True,
            prepared_state_thread_safe=True,
            instance_reentrant=False,
            process_reconstructable=False,
            deterministic_reduction=True,
            peak_memory_estimate="conservative",
            adapter_id=_BATCH_ADAPTER_ID,
            adapter_version=_BATCH_ADAPTER_VERSION,
        )

    def validate(self, engine, request) -> None:
        if request.operation not in _BATCH_OPS:
            raise CapabilityError(
                f"operation {request.operation} unsupported by "
                f"{_BATCH_ADAPTER_ID}"
            )
        extra = request.outputs - _BATCH_OUTPUTS
        if extra:
            raise CapabilityError(
                f"outputs {sorted(k.value for k in extra)} unsupported by "
                f"{_BATCH_ADAPTER_ID}"
            )
        if request.pricing_env is None:
            raise CapabilityError(
                "pricing_env is required for product_env engines"
            )

    # -- preparation helpers ------------------------------------------------
    def _sim_context(self, clone, request, context,
                     draw_provider=None) -> "_DCNSimContext":
        t0 = time.perf_counter()
        product = request.product
        env = request.pricing_env
        # CAPTURE the market-data fields once; everything a batch consumes
        # is bound to these objects, and reduce_batches verifies them after
        # execution (fail closed on mid-run replacement).
        captured = (env.vol_surface, env.spot_quote, env.rate_curve,
                    env.div_yield)
        ctx = build_dcn_grid_context(product)
        dt_array = np.diff(ctx.times)
        if dt_array.size == 0:
            raise ValidationError(
                "DCN grid has no steps (valuation at maturity?)"
            )
        term = build_mc_term_inputs(
            env,
            ref_strike=product.initial_price,
            maturity=float(ctx.times[-1]),
            time_steps=dt_array.size,
            dt_array=dt_array,
        )
        stderr_mode = (
            "scramble_means"
            if clone.use_sobol and clone.num_batches >= 2
            else "pathwise_iid"
        )
        model_params = getattr(clone, "model_params", None)
        return _DCNSimContext(
            engine=clone,
            product=product,
            pricing_env=env,
            ctx=ctx,
            term=term,
            df=_make_captured_df(captured[2]),
            spot0=float(captured[1].spot),
            dt_array=dt_array,
            obs_times=ctx.times[ctx.obs_cols],
            n_obs=int(ctx.obs_cols.size),
            stderr_mode=stderr_mode,
            t0=t0,
            captured=captured,
            product_fp=try_fingerprint(product),
            draw_provider=draw_provider,
            model_fp=(
                None if model_params is None
                else try_fingerprint(model_params)
            ),
        )

    def _attach_draw_provider(self, clone, context):
        repository = getattr(context, "draw_repository", None)
        if not clone.use_sobol or repository is None:
            return None
        provider = _PinnedDrawProvider(repository, clone.seed)
        clone._draw_provider = provider
        return provider

    # -- plan-metadata hooks (Phase 3): the Heston adapters override these;
    # the defaults preserve the Phase-2 GBM/LV plan fields verbatim -------
    def _plan_scheme(self, sim) -> str:
        return self._SCHEME

    def _plan_dimension(self, sim, time_steps) -> int:
        return time_steps

    def _plan_stream_layout(self, clone):
        if clone.use_sobol:
            return "sobol", "batch-shifted-sobol/v1"
        if clone.use_antithetic:
            return "antithetic", "seed-stride-1000-antithetic/v1"
        return "pseudorandom", "seed-stride-1000/v1"

    def _plan_est_bytes(self, sim, clone, batch_size, time_steps):
        # draws + path nodes + cashflow work arrays, conservative slack
        est_task = 8 * batch_size * (2 * time_steps + 1 + 6 * sim.n_obs)
        est_task += 1 << 20
        est_outcome = 8 * (6 * sim.n_obs + 16)
        if sim.stderr_mode == "pathwise_iid":
            est_outcome += 8 * batch_size
        return est_task, est_outcome

    # -- batch capability ----------------------------------------------------
    def plan_batches(self, engine, request, state, context) -> BatchPlan:
        sim = state.payload
        clone = sim.engine
        num_batches = clone.num_batches
        batch_size = clone.num_paths // num_batches
        time_steps = int(sim.dt_array.size)
        stream_kind, stream_layout = self._plan_stream_layout(clone)
        cls = type(clone)
        engine_class_path = f"{cls.__module__}.{cls.__qualname__}"
        plan_id = (
            f"{engine_class_path}:{clone.seed}:{num_batches}:{clone.num_paths}"
        )
        tasks = tuple(
            BatchTask(
                plan_id=plan_id,
                batch_index=i,
                batch_id=None if num_batches == 1 else i,
                n_paths=batch_size,
            )
            for i in range(num_batches)
        )
        est_task, est_outcome = self._plan_est_bytes(
            sim, clone, batch_size, time_steps
        )
        return BatchPlan(
            plan_id=plan_id,
            engine_class_path=engine_class_path,
            num_batches=num_batches,
            paths_per_batch=batch_size,
            total_paths=clone.num_paths,
            seed=clone.seed,
            stream_kind=stream_kind,
            stream_layout=stream_layout,
            time_steps=time_steps,
            dimension=self._plan_dimension(sim, time_steps),
            dtype="float64",
            scheme=self._plan_scheme(sim),
            stderr_mode=sim.stderr_mode,
            reduction_order="batch_index/v1",
            tasks=tasks,
            est_task_peak_bytes=est_task,
            est_outcome_bytes=est_outcome,
            implementation_fingerprint=(
                f"{_BATCH_ADAPTER_ID}/{_BATCH_ADAPTER_VERSION}"
            ),
        )

    def execute_batch(self, task, state, context) -> BatchOutcome:
        sim = state.payload
        clone = sim.engine
        provider = sim.draw_provider
        if provider is not None:
            provider.begin_batch()
        try:
            paths = clone._simulate(
                sim.spot0, sim.term, sim.dt_array, sim.pricing_env,
                n_paths=task.n_paths, batch_id=task.batch_id,
            )
        finally:
            # unpin (and free, when cache-bypassed) this batch's draw
            # blocks as soon as the simulation has consumed them
            if provider is not None:
                provider.end_batch()
        cf = compute_dcn_cashflows(paths, sim.product, sim.ctx, sim.df)
        # one-batch accumulator: EXACTLY the legacy per-batch arithmetic
        acc = _LegAccumulator(n_obs=sim.n_obs)
        acc.add(cf, sim.obs_times)
        stats = DCNBatchStats(
            n=acc.n,
            fixed_sum=acc.fixed_sum,
            ko_sum=acc.ko_sum,
            loss_sum=acc.loss_sum,
            fixed_by_period=acc.fixed_by_period_sum,
            ko_by_period=acc.ko_by_period_sum,
            coupon_paid=acc.coupon_paid_sum,
            ko_timing=acc.ko_timing_count,
            ki_count=acc.ki_count,
            survive_no_ki_count=acc.survive_no_ki_count,
            survive_ki_count=acc.survive_ki_count,
            life_sum=acc.life_sum,
            batch_mean=float(cf.total_pv.mean()),
            totals=(
                cf.total_pv if sim.stderr_mode == "pathwise_iid" else None
            ),
        )
        return BatchOutcome(
            batch_index=task.batch_index, n_paths=acc.n, payload=stats
        )

    def reduce_batches(self, outcomes, plan, state, context):
        sim = state.payload
        acc = _LegAccumulator(n_obs=sim.n_obs)
        batch_means = []
        for outcome in outcomes:
            s = outcome.payload
            acc.n += s.n
            acc.fixed_sum += s.fixed_sum
            acc.ko_sum += s.ko_sum
            acc.loss_sum += s.loss_sum
            acc.fixed_by_period_sum += s.fixed_by_period
            acc.ko_by_period_sum += s.ko_by_period
            acc.coupon_paid_sum += s.coupon_paid
            acc.ko_timing_count += s.ko_timing
            acc.ki_count += s.ki_count
            acc.survive_no_ki_count += s.survive_no_ki_count
            acc.survive_ki_count += s.survive_ki_count
            acc.life_sum += s.life_sum
            batch_means.append(s.batch_mean)
            if s.totals is not None:
                acc.totals.append(s.totals)
        self._verify_captured_inputs(sim)
        sign = sim.product.direction_sign
        if plan.stderr_mode == "scramble_means":
            means = np.array([float(sign * m) for m in batch_means])
            stderr = float(means.std(ddof=1) / np.sqrt(means.size))
        else:
            totals = sign * np.concatenate(acc.totals)
            stderr = float(totals.std(ddof=1) / np.sqrt(acc.n))
        result = _finalize_dcn_result(
            acc, sim.product, sim.engine.seed, stderr, sim.t0
        )
        economics = (
            ("pv", float(result.pv)),
            ("std_error", float(result.std_error)),
        )
        return result, economics

    def _verify_captured_inputs(self, sim) -> None:
        """Post-execution snapshot verification (code-gate finding
        2026-07-16): every batch consumed the CAPTURED objects, so a
        replaced environment field or a mutated product means the produced
        numbers mix market states — fail closed, never return them."""
        env = sim.pricing_env
        current = (env.vol_surface, env.spot_quote, env.rate_curve,
                   env.div_yield)
        replaced = any(a is not b for a, b in zip(current, sim.captured))
        mutated = (
            sim.product_fp is not None
            and try_fingerprint(sim.product) != sim.product_fp
        )
        model_params = getattr(sim.engine, "model_params", None)
        mutated = mutated or (
            sim.model_fp is not None
            and try_fingerprint(model_params) != sim.model_fp
        )
        if replaced or mutated:
            from quantark.execution.errors import DeterminismViolation

            raise DeterminismViolation(
                "pricing environment or product changed during batch "
                "execution; refusing to return a mixed-market result"
            )

    def project_operation(self, result, operation):
        if operation is PricingOperation.PRICE:
            return result.pv
        return result

    def execute_native(self, engine, request, normalized, context,
                       prepared=None):
        # Non-batch fallback (kernel prefers plan_batches when present):
        # run the prepared clone through the legacy call path.
        if prepared is not None and isinstance(prepared.payload, _DCNSimContext):
            return LegacyPriceAdapter.execute_native(
                self, prepared.payload.engine, request, normalized, context
            )
        return LegacyPriceAdapter.execute_native(
            self, engine, request, normalized, context, prepared=prepared
        )


class _UniformDrawProvider:
    """Uniform Sobol blocks from the session DrawRepository as WRITABLE
    private copies (the QE draw path clips + ndtri-transforms in place).
    The writable path never pins the master beyond the call, so no
    per-batch scoping is needed."""

    def __init__(self, repository, seed):
        self._repo = repository
        self._seed = seed

    def __call__(self, n_dims, n_paths, batch_id):
        handle = self._repo.uniforms_handle(
            seed=self._seed, n_paths=n_paths, dim=n_dims,
            batch_id=batch_id, writable=True,
        )
        try:
            return handle.value
        finally:
            handle.close()


class DCNBatchMCAdapter(_DCNBatchMixin, LegacyPriceAdapter):
    """Batch adapter for the plain GBM ``DCNMCEngine``.

    NOT inherited by the Heston DCN subclasses: the registry gives those
    their own exact-registered pair-aware adapters (Phase 3) so unknown
    stateful subclasses still fall through to the legacy adapter."""

    def __init__(self):
        super().__init__(call_shape="product_env")

    def prepare(self, engine, request, context) -> PreparedState:
        clone = type(engine)(
            num_paths=engine.num_paths, seed=engine.seed,
            use_sobol=engine.use_sobol, use_antithetic=engine.use_antithetic,
            num_batches=engine.num_batches,
            num_workers=1,  # framework owns threading; nested execution off
        )
        provider = self._attach_draw_provider(clone, context)
        sim = self._sim_context(clone, request, context,
                                draw_provider=provider)
        handles = (provider,) if provider is not None else ()
        return PreparedState(
            payload=sim, descriptors=(), fingerprint=None,
            byte_estimate=None, handles=handles,
        )


# ---------------------------------------------------------------------------
# Phase 1: cached Dupire surface preparation (shared by LV MC + PDE)
# ---------------------------------------------------------------------------

class _DCNLocalVolAdapterBase(LegacyPriceAdapter):
    def __init__(self):
        super().__init__(call_shape="product_env")

    def _prepare_surface(self, engine, request, context) -> PreparedState:
        if engine._prebuilt is not None:
            return PreparedState(
                payload=engine._prebuilt, descriptors=(),
                fingerprint=None, byte_estimate=None,
            )
        env = request.pricing_env
        # CAPTURE every dependency once: the builder consumes ONLY the
        # captured objects (including the dividend callable, bound to the
        # captured dividend object rather than the live environment), so a
        # concurrent field REPLACEMENT on the environment cannot mix market
        # states inside one build.
        inputs = (env.vol_surface, env.spot_quote, env.rate_curve, env.div_yield)
        spot = inputs[1].spot
        div_obj = inputs[3]
        if div_obj is None:
            def div_fn(t):
                return 0.0
        else:
            div_fn = div_obj.get_yield
        fp = try_fingerprint(inputs)
        cache = context.artifact_cache

        def builder():
            return build_dupire_local_vol(
                inputs[0], spot=spot, rate_curve=inputs[2], div_yield=div_fn,
            )

        if fp is None or cache is None:
            surface = builder()  # uncacheable: fresh build, still correct
            return PreparedState(
                payload=surface, descriptors=(),
                fingerprint=None, byte_estimate=_surface_nbytes(surface),
            )
        descriptor = ArtifactDescriptor(
            kind="dupire-local-vol", fingerprint=fp,
            dependency_tags=_DUPIRE_TAGS, builder_version=_BUILDER_VERSION,
        )
        handle = cache.get_or_build(
            descriptor, builder,
            size_bytes=_estimate_surface_bytes(inputs[0]),
            measure=_surface_nbytes,
        )
        current = (env.vol_surface, env.spot_quote, env.rate_curve, env.div_yield)
        replaced = any(a is not b for a, b in zip(current, inputs))
        if replaced or try_fingerprint(inputs) != fp:
            handle.close()
            cache.invalidate_tags(_DUPIRE_TAGS)
            from quantark.execution.errors import DeterminismViolation

            raise DeterminismViolation(
                "pricing environment mutated or replaced during preparation; "
                "the cached Dupire surface no longer matches its key"
            )
        return PreparedState(
            payload=handle.value, descriptors=(descriptor,),
            fingerprint=fp, byte_estimate=None, handles=(handle,),
        )

    # Phase-1 behavior (used by the PDE adapter): prepared payload IS the
    # surface; execution clones the engine around it.
    def prepare(self, engine, request, context) -> PreparedState:
        return self._prepare_surface(engine, request, context)

    def execute_native(self, engine, request, normalized, context, prepared=None):
        if prepared is None:
            return super().execute_native(engine, request, normalized, context)
        clone = self._clone_with_surface(engine, prepared.payload)
        return super().execute_native(clone, request, normalized, context)


class DCNLocalVolMCAdapter(_DCNBatchMixin, _DCNLocalVolAdapterBase):
    """Batch-capable LV DCN MC adapter: cached Dupire surface (Phase 1)
    composed with the fixed-batch capability (Phase 2)."""

    _SCHEME = "lv-euler/v1"

    def prepare(self, engine, request, context) -> PreparedState:
        surface_state = self._prepare_surface(engine, request, context)
        clone = self._clone_with_surface(engine, surface_state.payload)
        provider = self._attach_draw_provider(clone, context)
        sim = self._sim_context(clone, request, context,
                                draw_provider=provider)
        handles = surface_state.handles
        if provider is not None:
            handles = handles + (provider,)
        return PreparedState(
            payload=sim,
            descriptors=surface_state.descriptors,
            fingerprint=surface_state.fingerprint,
            byte_estimate=surface_state.byte_estimate,
            handles=handles,
        )

    def _clone_with_surface(self, engine, surface):
        return type(engine)(
            local_vol_surface=surface,
            num_paths=engine.num_paths, seed=engine.seed,
            use_sobol=engine.use_sobol, use_antithetic=engine.use_antithetic,
            num_batches=engine.num_batches,
            num_workers=1,  # framework owns threading; nested execution off
        )


class DCNLocalVolPDEAdapter(_DCNLocalVolAdapterBase):
    def _clone_with_surface(self, engine, surface):
        return type(engine)(
            local_vol_surface=surface,
            num_space_nodes=engine.n,
            s_min_mult=engine.s_min_mult, s_max_mult=engine.s_max_mult,
            rannacher_steps=engine.rannacher_steps,
            concentration=engine.concentration,
        )


# ---------------------------------------------------------------------------
# Phase 3: Heston DCN fixed-batch adapters
# ---------------------------------------------------------------------------

class _HestonBatchBase(_DCNBatchMixin, LegacyPriceAdapter):
    """Shared Heston DCN batch machinery: plan metadata reflecting the
    scheme/substeps/stream layout, repository-routed normal AND uniform
    draws, and fixed-signature cloning (exact registrations only)."""

    def __init__(self):
        super().__init__(call_shape="product_env")

    @staticmethod
    def _three_streams(clone) -> bool:
        from quantark.util.enum.engine_enums import HestonMCScheme

        return clone.scheme in (
            HestonMCScheme.QUADEXP, HestonMCScheme.QUADEXP_M,
        ) or clone.fixed_three_stream_sobol

    def _plan_scheme(self, sim) -> str:
        clone = sim.engine
        return (
            f"heston-{clone.scheme.name.lower()}"
            f"-sub{clone.substeps_per_interval}/v1"
        )

    def _plan_dimension(self, sim, time_steps) -> int:
        clone = sim.engine
        n_fine = time_steps * clone.substeps_per_interval
        return 3 * n_fine if self._three_streams(clone) else 2 * n_fine

    def _plan_stream_layout(self, clone):
        kind, layout = super()._plan_stream_layout(clone)
        if kind == "sobol":
            streams = "3stream" if self._three_streams(clone) else "2stream"
            layout = f"batch-shifted-sobol-{streams}/v1"
        return kind, layout

    def _plan_est_bytes(self, sim, clone, batch_size, time_steps):
        n_fine = time_steps * clone.substeps_per_interval
        per_path = 3 * n_fine if self._three_streams(clone) else 2 * n_fine
        # draw block + writable copy + variance/spot state + contractual
        # nodes + cashflow work arrays, conservative slack
        est_task = 8 * batch_size * (2 * per_path + 4 + 6 * sim.n_obs)
        est_task += 1 << 20
        est_outcome = 8 * (6 * sim.n_obs + 16)
        if sim.stderr_mode == "pathwise_iid":
            est_outcome += 8 * batch_size
        return est_task, est_outcome

    def _attach_draw_provider(self, clone, context):
        provider = super()._attach_draw_provider(clone, context)
        if provider is not None:
            clone._uniform_provider = _UniformDrawProvider(
                context.draw_repository, clone.seed
            )
        return provider

    def prepare(self, engine, request, context) -> PreparedState:
        clone = self._clone_engine(engine)
        provider = self._attach_draw_provider(clone, context)
        sim = self._sim_context(clone, request, context,
                                draw_provider=provider)
        handles = (provider,) if provider is not None else ()
        return PreparedState(
            payload=sim, descriptors=(), fingerprint=None,
            byte_estimate=None, handles=handles,
        )


class DCNHestonBatchMCAdapter(_HestonBatchBase):
    def _clone_engine(self, engine):
        return type(engine)(
            model_params=engine.model_params,
            substeps_per_interval=engine.substeps_per_interval,
            scheme=engine.scheme,
            fixed_three_stream_sobol=engine.fixed_three_stream_sobol,
            num_paths=engine.num_paths, seed=engine.seed,
            use_sobol=engine.use_sobol,
            use_antithetic=engine.use_antithetic,
            num_batches=engine.num_batches,
            num_workers=1,  # framework owns threading; nested execution off
        )


class DCNQEBatchMCAdapter(_HestonBatchBase):
    def _clone_engine(self, engine):
        return type(engine)(
            model_params=engine.model_params,
            martingale_correction=engine.martingale_correction,
            substeps_per_interval=engine.substeps_per_interval,
            fixed_three_stream_sobol=engine.fixed_three_stream_sobol,
            num_paths=engine.num_paths, seed=engine.seed,
            use_sobol=engine.use_sobol,
            use_antithetic=engine.use_antithetic,
            num_batches=engine.num_batches,
            num_workers=1,  # framework owns threading; nested execution off
        )


class DCNCoupledCoarseHestonBatchMCAdapter(_HestonBatchBase):
    """Pair-aware clone: the coarse engine derives every draw from its fine
    partner, so the fine engine is cloned first (it owns the draw streams
    and gets the session providers) and the coarse clone is built around
    it. ``model_params`` is passed as the SAME object to both clones (the
    coupled-pair constructor enforces identity)."""

    def _clone_engine(self, engine):
        fine = engine._fine_engine
        fine_clone = type(fine)(
            model_params=engine.model_params,
            substeps_per_interval=fine.substeps_per_interval,
            scheme=fine.scheme,
            fixed_three_stream_sobol=fine.fixed_three_stream_sobol,
            num_paths=fine.num_paths, seed=fine.seed,
            use_sobol=fine.use_sobol, use_antithetic=fine.use_antithetic,
            num_batches=fine.num_batches,
            num_workers=1,  # framework owns threading; nested execution off
        )
        return type(engine)(
            fine_engine=fine_clone,
            model_params=engine.model_params,
            substeps_per_interval=engine.substeps_per_interval,
            scheme=engine.scheme,
            fixed_three_stream_sobol=engine.fixed_three_stream_sobol,
            num_paths=engine.num_paths, seed=engine.seed,
            use_sobol=engine.use_sobol,
            use_antithetic=engine.use_antithetic,
            num_batches=engine.num_batches,
            num_workers=1,  # framework owns threading; nested execution off
        )

    def _attach_draw_provider(self, clone, context):
        # Draws are generated by the FINE partner; hook the providers there.
        provider = _DCNBatchMixin._attach_draw_provider(
            self, clone._fine_engine, context
        )
        if provider is not None:
            clone._fine_engine._uniform_provider = _UniformDrawProvider(
                context.draw_repository, clone.seed
            )
        return provider

    def _plan_dimension(self, sim, time_steps):
        clone = sim.engine
        n_fine_partner = (
            time_steps * clone._fine_engine.substeps_per_interval
        )
        return (
            3 * n_fine_partner
            if self._three_streams(clone) else 2 * n_fine_partner
        )

    def _plan_est_bytes(self, sim, clone, batch_size, time_steps):
        # the fine partner's draw block (2x substeps) dominates the peak
        return super()._plan_est_bytes(
            sim, clone._fine_engine, batch_size, time_steps
        )
