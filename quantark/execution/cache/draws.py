"""Session-owned Sobol draw repository (spec section 8.3).

Reuses the hardened ``PreparedArtifactCache`` single-flight/lease machinery
on the dedicated ``draw_cache`` pool. Masters are published READ-ONLY;
``writable=True`` hands out a private copy (the only surface on which
in-place transforms are permitted). Only fully-descriptor-identified
scrambled-Sobol blocks are cacheable; stateful pseudorandom streams never
reach this repository (engines keep their legacy ``_draws`` for those).

The builder invokes ``SobolNormalGenerator`` directly rather than the
process-global ``QMCDrawCache`` path, so a block's bytes are never charged
to both the session budget and the legacy cache.
"""
from dataclasses import dataclass

import numpy as np

from quantark.execution.cache.artifacts import (
    ArtifactDescriptor,
    ArtifactHandle,
    PreparedArtifactCache,
)
from quantark.execution.cache.fingerprint import fingerprint
from quantark.execution.manifest import build_versions

__all__ = ["DrawDescriptor", "DrawRepository"]

_IMPL_ID = "quantark.montecarlo.qmc_sobol.SobolNormalGenerator"
_IMPL_VERSION = "1"
_LAYOUT = "batch-shifted-sobol/v1"


@dataclass(frozen=True)
class DrawDescriptor:
    """Complete identification of a generated draw block (spec section 8.3).
    A block not fully identified by its descriptor is not cacheable."""

    generator_family: str
    implementation_id: str
    implementation_version: str
    distribution: str
    stream_layout: str
    seed: int
    batch_id: int | None
    n_paths: int
    dim: int
    shape: tuple
    memory_order: str
    dtype: str
    antithetic: bool
    transform_pipeline: tuple
    numpy_version: str
    scipy_version: str

    @property
    def fingerprint(self) -> str:
        return fingerprint(self)


class DrawRepository:
    def __init__(self, lease_manager):
        self._cache = PreparedArtifactCache(lease_manager, pool="draw_cache")
        self.lease_manager = lease_manager  # public: pairing identity check

    def descriptor(self, *, seed, n_paths, dim, batch_id,
                   distribution="normal") -> DrawDescriptor:
        versions = dict(build_versions())
        return DrawDescriptor(
            generator_family="sobol-scrambled",
            implementation_id=_IMPL_ID,
            implementation_version=_IMPL_VERSION,
            distribution=distribution,
            stream_layout=_LAYOUT,
            seed=int(seed),
            batch_id=None if batch_id is None else int(batch_id),
            n_paths=int(n_paths),
            dim=int(dim),
            shape=(int(n_paths), int(dim)),
            memory_order="C",
            dtype="float64",
            antithetic=False,
            transform_pipeline=(
                ("ndtri/v1",) if distribution == "normal" else ()
            ),
            numpy_version=versions.get("numpy", "unknown"),
            scipy_version=versions.get("scipy", "unknown"),
        )

    def _block_handle(self, *, kind, distribution, draw_method, seed,
                      n_paths, dim, batch_id, writable) -> ArtifactHandle:
        desc = self.descriptor(
            seed=seed, n_paths=n_paths, dim=dim, batch_id=batch_id,
            distribution=distribution,
        )
        art = ArtifactDescriptor(
            kind=kind,
            fingerprint=desc.fingerprint,
            dependency_tags=frozenset({"draws"}),
            builder_version=_IMPL_VERSION,
        )

        def builder():
            from quantark.montecarlo.qmc_sobol import SobolNormalGenerator

            gen = SobolNormalGenerator(base_seed=int(seed))
            block = np.ascontiguousarray(
                getattr(gen, draw_method)(
                    int(n_paths), int(dim), batch_id=batch_id
                )
            )
            block.flags.writeable = False
            return block

        handle = self._cache.get_or_build(
            art, builder,
            size_bytes=int(n_paths) * int(dim) * 8,
            measure=lambda block: block.nbytes,
        )
        if not writable:
            return handle
        copy = handle.value.copy()
        handle.close()  # writable scratch is task-owned, not pinned
        return ArtifactHandle(copy, lambda: None)

    def normals_handle(
        self, *, seed, n_paths, dim, batch_id, writable=False
    ) -> ArtifactHandle:
        return self._block_handle(
            kind="sobol-normal-block", distribution="normal",
            draw_method="normal", seed=seed, n_paths=n_paths, dim=dim,
            batch_id=batch_id, writable=writable,
        )

    def uniforms_handle(
        self, *, seed, n_paths, dim, batch_id, writable=False
    ) -> ArtifactHandle:
        return self._block_handle(
            kind="sobol-uniform-block", distribution="uniform",
            draw_method="uniform", seed=seed, n_paths=n_paths, dim=dim,
            batch_id=batch_id, writable=writable,
        )

    def stats(self) -> dict:
        return self._cache.stats()

    def close(self) -> None:
        self._cache.close()
