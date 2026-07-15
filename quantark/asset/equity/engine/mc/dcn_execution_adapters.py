"""Execution-framework adapters for the DCN local-vol engines (spec Phase 1).

The adapter path serves cache-fetched Dupire surfaces through PREBUILT-SURFACE
FACTORY CLONES: a clone of the target engine is constructed with
``local_vol_surface=<cached surface>``, so the original engine's
``_prepare_simulation``/``_resolve_surface`` hooks and ``_active_surface``
state are never touched (spec sections 6.3 + 17.1: mutable-state removal
applies to the adapter path only; the direct path and its subclass hooks are
preserved verbatim).

Preparation inputs are BOUND once, so the cache key, the build, and the
post-build verification all see the same objects; a fingerprint mismatch
after the build raises ``DeterminismViolation`` (spec section 5.1).
"""
from quantark.execution.cache.artifacts import ArtifactDescriptor
from quantark.execution.cache.fingerprint import try_fingerprint
from quantark.execution.contracts import PreparedState
from quantark.execution.legacy_adapter import LegacyPriceAdapter
from quantark.volmodels.localvol import build_dupire_local_vol

__all__ = ["DCNLocalVolMCAdapter", "DCNLocalVolPDEAdapter"]

_DUPIRE_TAGS = frozenset({"vol_surface", "spot", "rate_curve", "dividend_curve"})
_BUILDER_VERSION = "1"

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


class _DCNLocalVolAdapterBase(LegacyPriceAdapter):
    def __init__(self):
        super().__init__(call_shape="product_env")

    def prepare(self, engine, request, context) -> PreparedState:
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

    def execute_native(self, engine, request, normalized, context, prepared=None):
        if prepared is None:
            return super().execute_native(engine, request, normalized, context)
        clone = self._clone_with_surface(engine, prepared.payload)
        return super().execute_native(clone, request, normalized, context)


class DCNLocalVolMCAdapter(_DCNLocalVolAdapterBase):
    def _clone_with_surface(self, engine, surface):
        return type(engine)(
            local_vol_surface=surface,
            num_paths=engine.num_paths, seed=engine.seed,
            use_sobol=engine.use_sobol, use_antithetic=engine.use_antithetic,
            num_batches=engine.num_batches, num_workers=engine.num_workers,
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
