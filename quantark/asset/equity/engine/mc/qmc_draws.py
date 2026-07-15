"""Public Sobol uniform and standard-normal draws for MC engines.

Thin adapter over the shared :class:`quantark.montecarlo.qmc_sobol.SobolNormalGenerator`
(the same generator behind the snowball/phoenix vol engines' module-local
``_qmc_normals`` helpers), exposed publicly for the DCN engines.

Blocks are memoized in the process-wide byte-budgeted LRU
:class:`~quantark.montecarlo.qmc_sobol.QMCDrawCache`: scrambled-Sobol draws
are pure functions of (seed, batch_id, n_paths, dim), and CRN Greek/scenario
reports deliberately reprice with the same seed, so without the cache the
identical blocks are regenerated on every bumped repricing. Cached blocks are
returned as READ-ONLY views; pass ``writable=True`` to receive a private copy
(needed only by callers that transform draws in place).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from quantark.montecarlo.qmc_sobol import (
    SobolNormalGenerator,
    get_qmc_draw_cache,
)


def _cached_block(
    kind: str, seed: int, n_paths: int, dim: int,
    batch_id: Optional[int], writable: bool,
) -> np.ndarray:
    cache = get_qmc_draw_cache()
    key = (kind, int(seed), None if batch_id is None else int(batch_id),
           int(n_paths), int(dim))
    block = cache.get(key)
    if block is None:
        generator = SobolNormalGenerator(base_seed=int(seed))
        draw = generator.normal if kind == "normal" else generator.uniform
        block = np.ascontiguousarray(
            draw(n_paths, dim, batch_id=batch_id)
        )
        block = cache.put(key, block)
    return block.copy() if writable else block


def qmc_normals(
    seed: int, n_paths: int, dim: int, batch_id: Optional[int] = None,
    writable: bool = False,
) -> np.ndarray:
    """Sobol N(0,1) draws of shape (n_paths, dim); batch_id shifts the stream.

    Returns a read-only (possibly cached) block unless ``writable=True``.
    """
    return _cached_block("normal", seed, n_paths, dim, batch_id, writable)


def qmc_uniforms(
    seed: int, n_paths: int, dim: int, batch_id: Optional[int] = None,
    writable: bool = False,
) -> np.ndarray:
    """Scrambled Sobol U(0,1) draws of shape ``(n_paths, dim)``.

    Returns a read-only (possibly cached) block unless ``writable=True``.
    """
    return _cached_block("uniform", seed, n_paths, dim, batch_id, writable)
