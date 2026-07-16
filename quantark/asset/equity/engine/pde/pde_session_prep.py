"""Session preparation artifact states for the 1D equity PDE families.

Spec section 9.2: grids (with event-aligned time grids), term-structure step
coefficients (the materialized term context), and eager factorization packs
go behind ``PreparedArtifactCache`` descriptors. Keys derive from the
engine's own sanctioned reuse contract — the polymorphic ``_grid_cache_key``
tuple — plus curve/surface value fingerprints; anything uncanonicalizable
builds fresh (spec section 10.1: correctness never depends on cacheability).

Capture-and-reverify (plan-gate finding 2026-07-16): every state captures
its environment dependencies ONCE, builds, then re-verifies both object
identity and the recomputed fingerprint before returning; a mismatch closes
the handle, invalidates the state's tags, and raises DeterminismViolation —
the same contract as the Dupire surface state.
"""
import types
from typing import NamedTuple, Optional

import numpy as np

from quantark.execution.cache.artifacts import ArtifactDescriptor
from quantark.execution.cache.fingerprint import try_fingerprint
from quantark.execution.errors import DeterminismViolation

__all__ = [
    "ArtifactState",
    "market_scalars",
    "grid_state",
    "step_coefficients_state",
    "factorization_state",
]

_BUILDER_VERSION = "pde-prep/v1"
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


class ArtifactState(NamedTuple):
    """A prepared value plus its (optional) cache identity."""

    value: object
    descriptor: Optional[ArtifactDescriptor]
    handle: object  # Optional[ArtifactHandle]
    fingerprint: Optional[str]


def market_scalars(product, pricing_env):
    """The _solve preamble scalars: (spot, strike, tau, r, q, sigma)."""
    spot = pricing_env.spot
    tau = product.get_maturity(pricing_env)
    strike = getattr(product, "strike", spot)
    r = pricing_env.get_rate(tau)
    q = pricing_env.get_div_yield(tau)
    sigma = pricing_env.get_vol(strike, tau)
    return spot, strike, tau, r, q, sigma


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


def _grids_nbytes(grids) -> int:
    return int(sum(np.asarray(arr).nbytes for arr in grids))


def _coeffs_nbytes(sc) -> int:
    total = np.asarray(sc.set_index).nbytes
    for lcu in sc.lcu_sets:
        total += sum(np.asarray(arr).nbytes for arr in lcu)
    return int(total)


def _packs_nbytes(value) -> int:
    """Measured banded bytes + estimated splu bytes (SuperLU is opaque; use
    3.5x the M1 sparse size as a conservative per-entry stand-in)."""
    matrix_pack, banded_pack = value
    total = 0
    for m1, _lu in matrix_pack.values():
        m1_bytes = m1.data.nbytes + m1.indices.nbytes + m1.indptr.nbytes
        total += int(3.5 * m1_bytes)
    for entry in banded_pack.values():
        total += sum(np.asarray(arr).nbytes for arr in entry)
    return total + 1024


def grid_state(engine, product, pricing_env, context) -> ArtifactState:
    """Spatial + event-aligned time grids behind a descriptor."""
    spot, strike, tau, r, q, sigma = market_scalars(product, pricing_env)

    def build():
        return engine._build_grids(product, pricing_env, spot, sigma, tau, r, q)

    cache = getattr(context, "artifact_cache", None)
    if engine._resolve_cache_strategy() == "disable" or cache is None:
        return ArtifactState(build(), None, None, None)

    capture = _env_capture(pricing_env)

    def key_fp():
        return try_fingerprint(
            (
                "pde-grid",
                _class_path(engine),
                engine._grid_cache_key(product, pricing_env, spot, sigma, tau, r, q),
            )
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
        descriptor, build, size_bytes=256 * 1024, measure=_grids_nbytes
    )
    _reverify(pricing_env, capture, key_fp, fp, handle, cache, _GRID_TAGS)
    return ArtifactState(handle.value, descriptor, handle, fp)


def step_coefficients_state(
    engine, product, pricing_env, grids, grid_fp, extra_fp, context
) -> ArtifactState:
    """Per-step operator coefficient sets (the materialized term context).

    ``extra_fp`` carries the market identity beyond the grid key: the curve
    fingerprint for the base family, the Dupire-surface fingerprint (plus the
    curve fingerprint) for the LV family. The stored value is PRE-flat-exact;
    the solve applies ``_flat_exact_step_coefficients`` identically on both
    the injected and the built path.
    """
    x_vec, _s_vec, dx_vec, t_vec, _dt_vec = grids
    _spot, strike, _tau, _r, _q, _sigma = market_scalars(product, pricing_env)
    num_x = len(x_vec)

    def build():
        return engine._build_step_coefficients(
            pricing_env, strike, t_vec, dx_vec, num_x
        )

    cache = getattr(context, "artifact_cache", None)
    if (
        engine._resolve_cache_strategy() == "disable"
        or cache is None
        or grid_fp is None
        or extra_fp is None
    ):
        return ArtifactState(build(), None, None, None)

    capture = _env_capture(pricing_env)
    curves_fp = try_fingerprint(
        (pricing_env.rate_curve, pricing_env.div_yield, pricing_env.vol_surface)
    )
    if curves_fp is None:
        return ArtifactState(build(), None, None, None)

    def key_fp():
        live_curves = try_fingerprint(
            (pricing_env.rate_curve, pricing_env.div_yield, pricing_env.vol_surface)
        )
        return try_fingerprint(
            ("pde-step-coeffs", _class_path(engine), grid_fp, extra_fp, live_curves)
        )

    fp = try_fingerprint(
        ("pde-step-coeffs", _class_path(engine), grid_fp, extra_fp, curves_fp)
    )
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


def factorization_state(
    engine, product, pricing_env, grids, coeff_fp, context
) -> ArtifactState:
    """Eager (matrix_pack, banded_pack) maps behind a descriptor.

    Legality (spec section 9.2): the key derives from the coefficient
    fingerprint (which encodes grid + market + params identity) plus the
    banded-solver policy bit, so a hit proves identical coefficients, time
    steps, theta schedule, and scheme. The caller must have injected the
    coefficient artifact on ``engine`` BEFORE calling this (plan-gate
    ordering finding) so the builder consumes it via
    ``_step_coefficients_for_solve``.
    """
    x_vec = grids[0]

    def build():
        matrix_pack, banded_pack = engine._session_factorization_packs(
            product, pricing_env, grids
        )
        return (
            types.MappingProxyType(matrix_pack),
            types.MappingProxyType(banded_pack),
        )

    cache = getattr(context, "artifact_cache", None)
    if (
        engine._resolve_cache_strategy() == "disable"
        or cache is None
        or coeff_fp is None
    ):
        return ArtifactState(build(), None, None, None)

    capture = _env_capture(pricing_env)
    banded_policy = bool(getattr(engine.params, "use_banded_solver", False))

    def key_fp():
        return try_fingerprint(
            ("pde-fact", _class_path(engine), coeff_fp, banded_policy)
        )

    fp = key_fp()
    if fp is None:
        return ArtifactState(build(), None, None, None)
    descriptor = ArtifactDescriptor(
        kind="pde-factorization-pack",
        fingerprint=fp,
        dependency_tags=_FACT_TAGS,
        builder_version=_BUILDER_VERSION,
    )
    n_steps = max(1, len(grids[3]) - 1)
    estimate = min(n_steps, 64) * len(x_vec) * 8 * 12 + (1 << 20)
    handle = cache.get_or_build(
        descriptor, build, size_bytes=estimate, measure=_packs_nbytes
    )
    _reverify(pricing_env, capture, key_fp, fp, handle, cache, _FACT_TAGS)
    return ArtifactState(handle.value, descriptor, handle, fp)
