"""Execution-framework adapters for the equity PDE solvers (spec Phase 4).

Prepared-clone pattern (the reviewed Phase 1/3 contract): ``prepare`` clones
the target engine and injects session artifacts (grids, step coefficients,
factorization packs — built through ``pde_session_prep`` behind
``PreparedArtifactCache`` descriptors), so the LIVE engine is never mutated
and concurrent dispatches cannot mix market state. Every registration is
``exact=True``: an unknown subclass overriding ``price()`` falls to the
legacy adapter instead of being silently driven through the clone.

Rich outputs (spec sections 8/9.3): a PRICE request for
``{PV, EVENT_STATS, GRID}`` runs ONE value solve through the engines'
``_session_outputs`` seam — the same implementation behind
``price_with_events`` — plus the engines' designed event-indicator sweep;
the session never adds backward marches over the direct path. When outputs
beyond ``{PV}`` are requested the outcome ``value`` is a
``PDESessionValue`` (there is no legacy value shape to preserve there — the
legacy adapter rejected such requests). A GRID request that short-circuits
(expired / knocked-out product) fails closed with ``CapabilityError``
rather than silently returning pv-only economics.

Session dispatch does not populate live-engine profile stats: the clone is
constructed with default profiling.
"""
from dataclasses import dataclass

from quantark.asset.equity.engine.pde.pde_session_prep import (
    factorization_state,
    grid_state,
    step_coefficients_state,
)
from quantark.execution.cache.fingerprint import try_fingerprint
from quantark.execution.contracts import (
    EngineCapabilities,
    OutputKind,
    PreparedState,
    PricingOperation,
)
from quantark.execution.errors import CapabilityError
from quantark.execution.legacy_adapter import LegacyPriceAdapter
from quantark.execution.prep.dupire import dupire_surface_state
from quantark.execution.prep.verify import capture_market, verify_market

__all__ = [
    "PDESessionValue",
    "EquityPDE1DSessionAdapter",
    "EquityPDEAutocallableSessionAdapter",
    "EquityLVPDESessionAdapter",
    "EquityLVAutocallableSessionAdapter",
    "Heston2DAutocallableSessionAdapter",
]

ADAPTER_ID = "equity-pde-prepared"
ADAPTER_VERSION = "1"


@dataclass(frozen=True)
class PDESessionValue:
    """Outcome value for PRICE requests with outputs beyond {PV}."""

    pv: float
    event_stats: object
    event_distribution: object
    grid: tuple


@dataclass(frozen=True)
class _PreparedPDE:
    clone: object
    capture: object = None  # MarketCapture; verified after execution


def _option(request, key):
    for pair in request.operation_options:
        if isinstance(pair, tuple) and len(pair) == 2 and pair[0] == key:
            return pair[1]
    return None


def _surface_nbytes(surface) -> int:
    total = 0
    for value in vars(surface).values():
        nbytes = getattr(value, "nbytes", None)
        if isinstance(nbytes, int):
            total += nbytes
    return total or (1 << 20)


def _estimate_surface_bytes(vol_surface) -> int:
    iv_nbytes = getattr(getattr(vol_surface, "iv_grid", None), "nbytes", None)
    if isinstance(iv_nbytes, int) and iv_nbytes > 0:
        return 8 * iv_nbytes + (256 << 10)
    return 8 << 20


class _EquityPDESessionBase(LegacyPriceAdapter):
    _GRID_CAPABLE = True
    _EVENTS_CAPABLE = False

    def __init__(self):
        super().__init__(call_shape="product_env")

    # -- capability surface -------------------------------------------------
    def _output_kinds(self) -> frozenset:
        kinds = {OutputKind.PV}
        if self._GRID_CAPABLE:
            kinds.add(OutputKind.GRID)
        if self._EVENTS_CAPABLE:
            kinds.add(OutputKind.EVENT_STATS)
        return frozenset(kinds)

    def _operations(self) -> frozenset:
        ops = {PricingOperation.PRICE}
        if self._EVENTS_CAPABLE:
            ops.add(PricingOperation.EVENT_STATS)
        return frozenset(ops)

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            operations=self._operations(),
            output_kinds=self._output_kinds(),
            supported_backends=frozenset({"serial"}),
            fixed_planning=None,
            prepared_state_thread_safe=False,
            instance_reentrant=False,
            process_reconstructable=False,
            deterministic_reduction=True,
            peak_memory_estimate="conservative",
            adapter_id=ADAPTER_ID,
            adapter_version=ADAPTER_VERSION,
        )

    def validate(self, engine, request) -> None:
        if request.operation not in self._operations():
            raise CapabilityError(
                f"operation {request.operation} unsupported by {ADAPTER_ID} "
                f"for {type(engine).__qualname__}"
            )
        if request.operation is PricingOperation.PRICE:
            allowed = self._output_kinds()
        else:
            # EVENT_STATS operation keeps the legacy PV-only guarantee.
            allowed = frozenset({OutputKind.PV})
        extra = request.outputs - allowed
        if extra:
            raise CapabilityError(
                f"outputs {sorted(k.value for k in extra)} unsupported for "
                f"operation {request.operation.value} via {ADAPTER_ID}"
            )
        if request.pricing_env is None:
            raise CapabilityError("pricing_env is required for product_env engines")

    # -- preparation ---------------------------------------------------------
    def _clone_engine(self, engine):
        return type(engine)(params=engine.params)

    def _market_fields(self, env) -> tuple:
        return (env.rate_curve, env.div_yield, env.vol_surface, env.spot_quote)

    def _capture(self, request):
        return capture_market(
            self._market_fields(request.pricing_env), request.product
        )

    def _verify_capture(self, prepared, request) -> None:
        state = prepared.payload if prepared is not None else None
        if state is None or state.capture is None:
            return
        verify_market(
            state.capture,
            self._market_fields(request.pricing_env),
            request.product,
        )

    def prepare(self, engine, request, context) -> PreparedState:
        clone = self._clone_engine(engine)
        product, env = request.product, request.pricing_env
        handles: list = []
        descriptors: list = []
        fps: list = []
        try:
            try:
                tau = float(product.get_maturity(env))
            except Exception:
                tau = 0.0  # let the execute path surface the native error
            if tau > 0.0:
                self._prepare_artifacts(
                    clone, product, env, context, handles, descriptors, fps
                )
        except BaseException:
            for handle in handles:
                handle.close()
            raise
        fingerprint = (
            try_fingerprint(tuple(fps)) if fps and all(fps) else None
        )
        return PreparedState(
            payload=_PreparedPDE(clone=clone, capture=self._capture(request)),
            descriptors=tuple(descriptors),
            fingerprint=fingerprint,
            byte_estimate=None,
            handles=tuple(handles),
        )

    @staticmethod
    def _track(state, handles, descriptors, fps):
        if state.handle is not None:
            handles.append(state.handle)
        if state.descriptor is not None:
            descriptors.append(state.descriptor)
        fps.append(state.fingerprint)

    def _prepare_artifacts(
        self, clone, product, env, context, handles, descriptors, fps
    ) -> None:
        # Grid-key evaluation needs the same pre-grid state a direct solve
        # sets (KI regime, valuation flags) — the shared _solve preamble.
        clone._prepare_solve_state(product, env)
        curves_fp = try_fingerprint((env.rate_curve, env.div_yield, env.vol_surface))
        grid = grid_state(clone, product, env, context)
        self._track(grid, handles, descriptors, fps)
        clone._session_grids = grid.value
        coeff = step_coefficients_state(
            clone, product, env, grid.value, grid.fingerprint, curves_fp, context
        )
        self._track(coeff, handles, descriptors, fps)
        # Inject coefficients BEFORE the factorization build (plan-gate
        # ordering finding): _session_factorization_packs consumes them via
        # _step_coefficients_for_solve.
        clone._session_step_coefficients = coeff.value
        fact = factorization_state(
            clone, product, env, grid.value, coeff.fingerprint, context
        )
        self._track(fact, handles, descriptors, fps)
        matrix_pack, banded_pack = fact.value
        clone._session_matrix_pack = matrix_pack
        if hasattr(clone, "_session_banded_pack"):
            clone._session_banded_pack = banded_pack

    # -- execution -----------------------------------------------------------
    def execute_native(self, engine, request, normalized, context, prepared=None):
        result = self._execute(engine, request, normalized, context, prepared)
        # End-to-end mutation guard (code-gate finding 2026-07-16): a market
        # mutation between prepare and execute would price a MIXED state
        # (stale injected artifacts + live boundary reads) — verify the
        # captured market after execution and fail closed.
        self._verify_capture(prepared, request)
        return result

    def _execute(self, engine, request, normalized, context, prepared=None):
        clone = prepared.payload.clone if prepared is not None else engine
        if request.operation is not PricingOperation.PRICE:
            return super().execute_native(clone, request, normalized, context)
        product, env = request.product, request.pricing_env
        if request.outputs == frozenset({OutputKind.PV}):
            value = clone.price(product, env)
            return value, (("pv", float(value)),)

        want_events = OutputKind.EVENT_STATS in request.outputs
        want_grid = OutputKind.GRID in request.outputs
        streams_opt = _option(request, "event_streams")
        streams = frozenset(streams_opt) if streams_opt is not None else None
        out = clone._session_outputs(
            product, env,
            want_events=want_events, want_grid=want_grid, streams=streams,
        )
        grid_rows: tuple = ()
        if want_grid:
            if out.solution is None:
                raise CapabilityError(
                    "grid projection unavailable for an expired or "
                    "knocked-out product; requested GRID output cannot be "
                    "served (fail closed)"
                )
            levels = _option(request, "grid_spot_levels")
            grid_rows = tuple(
                clone._grid_projection_from_solution(
                    out.solution, None if levels is None else list(levels)
                )
            )
        dist = out.event_distribution
        if want_events and dist is None:
            # Mirror price_with_events' degenerate-distribution semantics
            # for short-circuits and event-less product types.
            from quantark.cashleg.event_distribution import EventDistribution

            tau = product.get_maturity(env)
            dist = EventDistribution.trivial(max(float(tau), 0.0))
        value = PDESessionValue(
            pv=float(out.npv),
            event_stats=out.event_stats if want_events else None,
            event_distribution=dist if want_events else None,
            grid=grid_rows,
        )
        return value, (("pv", float(out.npv)),)


class EquityPDE1DSessionAdapter(_EquityPDESessionBase):
    """European/American/Barrier/DoubleBarrier/OneTouch/DoubleOneTouch."""


class EquityPDEAutocallableSessionAdapter(_EquityPDESessionBase):
    """Snowball/KOResetSnowball/Phoenix: adds one-solve event outputs."""

    _EVENTS_CAPABLE = True


class EquityLVPDESessionAdapter(_EquityPDESessionBase):
    """LocalVolPDESolver / LocalVolBarrierPDESolver: cached Dupire surface,
    prebuilt-surface clone; these engines discretize inside the volmodels
    kernel, so there are no grid/coefficient artifacts to inject."""

    _GRID_CAPABLE = False

    def _clone_engine(self, engine):
        raise NotImplementedError("LV adapters clone inside prepare()")

    def _surface_state(self, engine, env, context) -> PreparedState:
        from quantark.volmodels.localvol import build_dupire_local_vol

        inputs = (env.vol_surface, env.spot_quote, env.rate_curve, env.div_yield)
        spot = inputs[1].spot
        div_obj = inputs[3]
        if div_obj is None:
            def div_fn(t):
                return 0.0
        else:
            div_fn = div_obj.get_yield

        def builder():
            return build_dupire_local_vol(
                inputs[0], spot=spot, rate_curve=inputs[2], div_yield=div_fn,
            )

        return dupire_surface_state(
            prebuilt=engine._prebuilt,
            inputs=inputs,
            recapture=lambda: (
                env.vol_surface, env.spot_quote, env.rate_curve, env.div_yield
            ),
            builder=builder,
            context=context,
            estimate_bytes=_estimate_surface_bytes(inputs[0]),
            measure=_surface_nbytes,
        )

    def _clone_with_surface(self, engine, surface):
        return type(engine)(params=engine.params, local_vol_surface=surface)

    def prepare(self, engine, request, context) -> PreparedState:
        surface_state = self._surface_state(engine, request.pricing_env, context)
        clone = self._clone_with_surface(engine, surface_state.payload)
        return PreparedState(
            payload=_PreparedPDE(clone=clone, capture=self._capture(request)),
            descriptors=surface_state.descriptors,
            fingerprint=surface_state.fingerprint,
            byte_estimate=surface_state.byte_estimate,
            handles=surface_state.handles,
        )


class EquityLVAutocallableSessionAdapter(EquityLVPDESessionAdapter):
    """LocalVolSnowball/LocalVolPhoenix: Dupire surface + the full 1D
    artifact chain (grids, per-step LV coefficients, factorization packs)."""

    _GRID_CAPABLE = True
    _EVENTS_CAPABLE = True

    def prepare(self, engine, request, context) -> PreparedState:
        surface_state = self._surface_state(engine, request.pricing_env, context)
        clone = self._clone_with_surface(engine, surface_state.payload)
        product, env = request.product, request.pricing_env
        handles = list(surface_state.handles)
        descriptors = list(surface_state.descriptors)
        fps = [surface_state.fingerprint]
        try:
            try:
                tau = float(product.get_maturity(env))
            except Exception:
                tau = 0.0
            if tau > 0.0:
                self._prepare_lv_artifacts(
                    clone, product, env, context,
                    surface_state.payload, surface_state.fingerprint,
                    handles, descriptors, fps,
                )
        except BaseException:
            for handle in handles:
                handle.close()
            raise
        fingerprint = try_fingerprint(tuple(fps)) if all(fps) else None
        return PreparedState(
            payload=_PreparedPDE(clone=clone, capture=self._capture(request)),
            descriptors=tuple(descriptors),
            fingerprint=fingerprint,
            byte_estimate=None,
            handles=tuple(handles),
        )

    def _prepare_lv_artifacts(
        self, clone, product, env, context, surface, surface_fp,
        handles, descriptors, fps,
    ) -> None:
        clone._prepare_solve_state(product, env)
        grid = grid_state(clone, product, env, context)
        self._track(grid, handles, descriptors, fps)
        # The LV coefficient/factorization builders need the surface context
        # active through BOTH builds (plan-gate ordering finding: cold-cache
        # preparation raises PricingError otherwise), and the surface
        # fingerprint takes the market-identity slot in the coefficient key.
        clone._active_lv_surface = surface
        clone._active_s_vec = grid.value[1]
        try:
            coeff = step_coefficients_state(
                clone, product, env, grid.value, grid.fingerprint,
                surface_fp, context,
            )
            self._track(coeff, handles, descriptors, fps)
            clone._session_grids = grid.value
            clone._session_step_coefficients = coeff.value
            fact = factorization_state(
                clone, product, env, grid.value, coeff.fingerprint, context
            )
            self._track(fact, handles, descriptors, fps)
            matrix_pack, banded_pack = fact.value
            clone._session_matrix_pack = matrix_pack
            clone._session_banded_pack = banded_pack
        finally:
            clone._active_lv_surface = None  # _with_surface re-arms at execute
            clone._active_s_vec = None


class Heston2DAutocallableSessionAdapter(_EquityPDESessionBase):
    """Heston/SLV Snowball + Phoenix: clone + one-solve rich outputs.

    No preparation artifacts: the time-dependent ADI operators cannot prove
    factorization reuse safe and are rebuilt (spec section 9.2 legality),
    and per-solve term sampling is negligible next to the 2D march.
    """

    _GRID_CAPABLE = False
    _EVENTS_CAPABLE = True

    def _clone_engine(self, engine):
        kwargs = dict(
            model_params=engine.model_params,
            params=engine.params,
            n_x=engine.n_x, n_v=engine.n_v, n_t=engine.n_t,
            scheme=engine.scheme,
            grid_style=engine.grid_style,
            grid_focus=engine.grid_focus,
            pin_critical_spots=engine.pin_critical_spots,
        )
        if hasattr(engine, "leverage_surface"):
            kwargs.update(
                leverage_surface=engine.leverage_surface, eta=engine.eta
            )
        return type(engine)(**kwargs)

    def prepare(self, engine, request, context) -> PreparedState:
        return PreparedState(
            payload=_PreparedPDE(
                clone=self._clone_engine(engine),
                capture=self._capture(request),
            ),
            descriptors=(),
            fingerprint=None,
            byte_estimate=None,
        )
