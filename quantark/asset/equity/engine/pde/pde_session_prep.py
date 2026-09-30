"""Session preparation artifact states for the 1D equity PDE families.

Spec section 9.2, spec 2 §5.10 (pde-prep/v2): the Layout a march binds, the
per-step coefficient sets on it (pre-flat-exact), and the warmed factorizations
of the march go behind ``PreparedArtifactCache`` descriptors. All three are
built through the one preparation path (``core.prepare``): the grid key is the
declarative ``GridRequest`` plus the market snapshot, the coefficient key the
coefficients' identity (``coefficient_key``), the factorization key the
coefficient key plus the stepper, the groups' backends and their θ schedules.
Anything uncanonicalizable builds fresh (spec section 10.1: correctness never
depends on cacheability).

Capture-and-reverify (plan-gate finding 2026-07-16): every state captures
its environment dependencies ONCE, builds, then re-verifies both object
identity and the recomputed fingerprint before returning; a mismatch closes
the handle, invalidates the state's tags, and raises DeterminismViolation —
the same contract as the Dupire surface state.
"""
import types
from typing import NamedTuple, Optional

import numpy as np

from quantark.asset.equity.engine.pde.core.components import CentralDifference
from quantark.asset.equity.engine.pde.core.groups import Backend
from quantark.asset.equity.engine.pde.core.identity import (
    coefficient_key,
    component_fingerprint,
    group_schedule,
    market_fingerprints,
)
from quantark.asset.equity.engine.pde.core.prepare import (
    CompositeProblem,
    Operation,
    OutputRequest,
    SessionArtefacts,
    bind_layout,
    prepare,
    request_geometry,
    session_step_sets,
)
from quantark.execution.cache.artifacts import ArtifactDescriptor
from quantark.execution.cache.fingerprint import try_fingerprint
from quantark.execution.errors import DeterminismViolation

__all__ = [
    "ArtifactState",
    "grid_state",
    "step_coefficients_state",
    "factorization_state",
]

_BUILDER_VERSION = "pde-prep/v2"
_GRID_TAGS = frozenset(
    {
        "spot",
        "product_terms",
        "grid",
        "vol_surface",
        "rate_curve",
        "dividend_curve",
        "valuation_date",
    }
)
_COEFF_TAGS = _GRID_TAGS
_FACT_TAGS = _GRID_TAGS | {"solver_policy"}
#: The request a session prepares for: its artefacts serve PRICE and the rich
#: session outputs alike (the same layout, coefficients and value groups).
_PRICE = OutputRequest(Operation.PRICE)


class ArtifactState(NamedTuple):
    """A prepared value plus its (optional) cache identity."""

    value: object
    descriptor: Optional[ArtifactDescriptor]
    handle: object  # Optional[ArtifactHandle]
    fingerprint: Optional[str]


def _class_path(engine) -> str:
    cls = type(engine)
    return f"{cls.__module__}.{cls.__qualname__}"


def _env_capture(pricing_env):
    return (
        pricing_env.rate_curve,
        pricing_env.div_yield,
        pricing_env.vol_surface,
        pricing_env.spot_quote,
    )


def _reverify(pricing_env, capture, recompute_fp, expected_fp, handle, cache, tags):
    current = _env_capture(pricing_env)
    replaced = any(a is not b for a, b in zip(current, capture))
    if replaced or recompute_fp() != expected_fp:
        handle.close()
        if cache is not None:
            cache.invalidate_tags(tags)
        raise DeterminismViolation(
            "pricing environment mutated or replaced during PDE preparation; "
            "the cached artifact no longer matches its key"
        )


def _layout_nbytes(layout) -> int:
    arrays = (layout.spatial.x, layout.spatial.s, layout.spatial.dx, layout.time.t, layout.time.dt)
    return int(sum(np.asarray(arr).nbytes for arr in arrays))


def _coeffs_nbytes(sc) -> int:
    total = np.asarray(sc.set_index).nbytes
    for lcu in sc.lcu_sets:
        total += sum(np.asarray(arr).nbytes for arr in lcu)
    return int(total)


def _packs_nbytes(value) -> int:
    """Measured banded bytes + estimated splu bytes (SuperLU is opaque; use
    3.5x the M1 sparse size as a conservative per-entry stand-in)."""
    total = 0
    for m1, _lu in value[Backend.SPARSE_LU].values():
        m1_bytes = m1.data.nbytes + m1.indices.nbytes + m1.indptr.nbytes
        total += int(3.5 * m1_bytes)
    for entry in value[Backend.BANDED].values():
        total += sum(np.asarray(arr).nbytes for arr in entry)
    return total + 1024


def grid_state(engine, product, pricing_env, context) -> ArtifactState:
    """The Layout a march binds (``bind_layout``) behind a descriptor; the value
    is ``None`` for a request with a terminal outcome (nothing to prepare)."""

    def build():
        return bind_layout(engine, product, pricing_env, _PRICE)

    cache = getattr(context, "artifact_cache", None)
    if not engine._is_cache_enabled() or cache is None:
        return ArtifactState(build(), None, None, None)

    capture = _env_capture(pricing_env)

    def key_fp():
        # the grid request after the solve state it depends on (a knocked-in
        # product drops the discrete-KI interior nodes): prepare steps 0-4
        geometry = request_geometry(engine, product, pricing_env, _PRICE)
        if geometry is None:
            return None
        grid_request, market = geometry
        return try_fingerprint(
            ("pde-grid", _class_path(engine), grid_request, market, engine.grid_binder.config.key)
        )

    fp = key_fp()
    if fp is None:
        return ArtifactState(build(), None, None, None)
    descriptor = ArtifactDescriptor(
        kind="pde-grid",
        fingerprint=fp,
        dependency_tags=_GRID_TAGS,
        builder_version=_BUILDER_VERSION,
    )
    handle = cache.get_or_build(
        descriptor, build, size_bytes=256 * 1024, measure=_layout_nbytes
    )
    _reverify(pricing_env, capture, key_fp, fp, handle, cache, _GRID_TAGS)
    return ArtifactState(handle.value, descriptor, handle, fp)


def step_coefficients_state(
    engine, product, pricing_env, layout, context, lv_surface_fp=None
) -> ArtifactState:
    """Per-step operator coefficient sets on ``layout`` (PRE-flat-exact; the
    adapter's ``coefficients`` hook applies the flat-exact replacement on the
    injected and the built path alike).

    Keyed by the coefficients' identity (decision 18): the layout, every
    sampler input, the operator and -- local vol -- the surface. No key (an
    unfingerprintable market) builds fresh and stores nothing.
    """

    def build():
        return session_step_sets(engine, product, pricing_env, layout)

    cache = getattr(context, "artifact_cache", None)
    if not engine._is_cache_enabled() or cache is None:
        return ArtifactState(build(), None, None, None)

    capture = _env_capture(pricing_env)
    strike = engine.adapter._strike(product, pricing_env)
    operator_fp = component_fingerprint(CentralDifference())

    def key_fp():
        key = coefficient_key(layout.fingerprint, market_fingerprints(pricing_env), strike,
                              operator_fp, lv_surface_fp)
        if key is None:
            return None
        return try_fingerprint(("pde-step-coeffs", _class_path(engine), *key))

    fp = key_fp()
    if fp is None:
        return ArtifactState(build(), None, None, None)
    descriptor = ArtifactDescriptor(
        kind="pde-step-coefficients",
        fingerprint=fp,
        dependency_tags=_COEFF_TAGS,
        builder_version=_BUILDER_VERSION,
    )
    handle = cache.get_or_build(
        descriptor, build, size_bytes=256 * 1024, measure=_coeffs_nbytes
    )
    _reverify(pricing_env, capture, key_fp, fp, handle, cache, _COEFF_TAGS)
    return ArtifactState(handle.value, descriptor, handle, fp)


# Warmed packs hold at most this many factorization entries; further keys
# fall through to the march's own cache, bitwise identically (code-gate
# finding 2026-07-16: LV coefficients create one key per step, so an uncapped
# eager build could allocate far beyond the reserved estimate).
_PACK_MAX_ENTRIES = 64

_EMPTY_PACKS = types.MappingProxyType(
    {Backend.SPARSE_LU: types.MappingProxyType({}), Backend.BANDED: types.MappingProxyType({})}
)


def _marched_problem(problem, engine):
    """The prepared problem ``engine`` marches: a composite's own part (a knock-in's
    knock-out leg); ``None`` when nothing is marched (a terminal outcome)."""
    if isinstance(problem, CompositeProblem):
        own = [p for p in problem.parts if p.solver is engine]
        problem = own[0] if own else None
    if problem is None or problem.terminal is not None:
        return None
    return problem


def factorization_state(
    engine, product, pricing_env, layout, coefficients, coeff_fp, context
) -> ArtifactState:
    """Warmed, size-bounded factorizations ``{backend: mapping}`` behind a descriptor.

    Legality (spec section 9.2): the key is the coefficient fingerprint (grid +
    market + operator identity), the stepper, the groups' backends and their θ
    schedules, so a hit proves identical coefficients, time steps, θ and scheme.
    The packs are ``march.warm_factorizations`` over the problem the session's
    march prepares -- the march's own entries, in march order.

    Budget honesty (code-gate finding 2026-07-16): the build is bounded to
    ``_PACK_MAX_ENTRIES`` entries -- the bound the admission estimate charges --
    and when even that estimate exceeds the configured artifact budget the pack
    is skipped BEFORE any allocation (empty packs; the march builds its
    factorizations itself, bitwise identically).
    """
    from quantark.asset.equity.engine.pde.core.march import warm_factorizations

    n_steps = max(1, len(layout.time.t) - 1)
    estimate = min(n_steps, _PACK_MAX_ENTRIES) * len(layout.spatial.x) * 8 * 12 + (1 << 20)
    budget = getattr(
        getattr(context, "resource_budget", None), "artifact_cache_bytes", None
    )
    if budget is not None and estimate > budget:
        return ArtifactState(_EMPTY_PACKS, None, None, None)

    prepared = _marched_problem(
        prepare(engine, product, pricing_env, _PRICE,
                session=SessionArtefacts(layout=layout, coefficients=coefficients)),
        engine,
    )
    if prepared is None:
        return ArtifactState(_EMPTY_PACKS, None, None, None)

    def build():
        return types.MappingProxyType(
            warm_factorizations(prepared, max_entries=_PACK_MAX_ENTRIES)
        )

    cache = getattr(context, "artifact_cache", None)
    if (
        not engine._is_cache_enabled()
        or cache is None
        or coeff_fp is None
    ):
        return ArtifactState(build(), None, None, None)

    capture = _env_capture(pricing_env)
    # The post-build verification recomputes a LIVE market fingerprint, so an
    # in-place mutation during the build is still caught (code-gate finding
    # 2026-07-16).
    market_fp = try_fingerprint(
        (pricing_env.rate_curve, pricing_env.div_yield, pricing_env.vol_surface)
    )
    if market_fp is None:
        return ArtifactState(build(), None, None, None)
    fp = try_fingerprint(
        (
            "pde-fact",
            _class_path(engine),
            coeff_fp,
            component_fingerprint(prepared.stepper),
            tuple(g.backend.value for g in prepared.groups),
            try_fingerprint(tuple(group_schedule(prepared, g) for g in prepared.groups)),
        )
    )
    if fp is None:
        return ArtifactState(build(), None, None, None)

    def live_fp():
        live_market = try_fingerprint(
            (pricing_env.rate_curve, pricing_env.div_yield,
             pricing_env.vol_surface)
        )
        return fp if live_market == market_fp else None

    descriptor = ArtifactDescriptor(
        kind="pde-factorization-pack",
        fingerprint=fp,
        dependency_tags=_FACT_TAGS,
        builder_version=_BUILDER_VERSION,
    )
    handle = cache.get_or_build(
        descriptor, build, size_bytes=estimate, measure=_packs_nbytes
    )
    _reverify(pricing_env, capture, live_fp, fp, handle, cache, _FACT_TAGS)
    return ArtifactState(handle.value, descriptor, handle, fp)
