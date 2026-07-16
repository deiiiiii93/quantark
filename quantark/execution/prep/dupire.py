"""Shared Dupire local-vol surface artifact state (spec sections 9.2/10.1).

Extracted verbatim from the Phase 1 DCN adapter so the equity and FX PDE/MC
adapter families share ONE capture-once / single-flight / post-build-reverify
implementation. The caller supplies the captured input tuple (the cache key
material AND the builder's only data source) plus a re-capture callable for
the post-build identity check.
"""
from quantark.execution.cache.artifacts import ArtifactDescriptor
from quantark.execution.cache.fingerprint import try_fingerprint
from quantark.execution.contracts import PreparedState

__all__ = ["DUPIRE_TAGS", "dupire_surface_state"]

DUPIRE_TAGS = frozenset({"vol_surface", "spot", "rate_curve", "dividend_curve"})
_BUILDER_VERSION = "1"


def dupire_surface_state(
    prebuilt,
    inputs: tuple,
    recapture,
    builder,
    context,
    estimate_bytes: int,
    measure,
) -> PreparedState:
    """Serve a Dupire surface through the session artifact cache.

    ``prebuilt``: an engine-held surface (returned as-is, uncached).
    ``inputs``: the captured dependency tuple — key material and the ONLY
    objects the ``builder`` closure may consume.
    ``recapture``: zero-arg callable returning the LIVE dependency tuple for
    the post-build identity check.
    ``builder``: zero-arg closure building the surface from the captures.
    ``measure``: actual-bytes measurer for cache admission adjustment.
    """
    if prebuilt is not None:
        return PreparedState(
            payload=prebuilt, descriptors=(),
            fingerprint=None, byte_estimate=None,
        )
    fp = try_fingerprint(inputs)
    cache = getattr(context, "artifact_cache", None)
    if fp is None or cache is None:
        surface = builder()  # uncacheable: fresh build, still correct
        return PreparedState(
            payload=surface, descriptors=(),
            fingerprint=None, byte_estimate=measure(surface),
        )
    descriptor = ArtifactDescriptor(
        kind="dupire-local-vol", fingerprint=fp,
        dependency_tags=DUPIRE_TAGS, builder_version=_BUILDER_VERSION,
    )
    handle = cache.get_or_build(
        descriptor, builder, size_bytes=estimate_bytes, measure=measure,
    )
    current = recapture()
    replaced = any(a is not b for a, b in zip(current, inputs))
    if replaced or try_fingerprint(inputs) != fp:
        handle.close()
        cache.invalidate_tags(DUPIRE_TAGS)
        from quantark.execution.errors import DeterminismViolation

        raise DeterminismViolation(
            "pricing environment mutated or replaced during preparation; "
            "the cached Dupire surface no longer matches its key"
        )
    return PreparedState(
        payload=handle.value, descriptors=(descriptor,),
        fingerprint=fp, byte_estimate=None, handles=(handle,),
    )
