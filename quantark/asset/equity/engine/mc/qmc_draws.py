"""Public Sobol uniform and standard-normal draws for MC engines.

Thin adapter over the shared :class:`quantark.montecarlo.qmc_sobol.SobolNormalGenerator`
(the same generator behind the snowball/phoenix vol engines' module-local
``_qmc_normals`` helpers), exposed publicly for the DCN engines.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from quantark.montecarlo.qmc_sobol import SobolNormalGenerator


def qmc_normals(
    seed: int, n_paths: int, dim: int, batch_id: Optional[int] = None
) -> np.ndarray:
    """Sobol N(0,1) draws of shape (n_paths, dim); batch_id shifts the stream."""
    return SobolNormalGenerator(base_seed=int(seed)).normal(
        n_paths, dim, batch_id=batch_id
    )


def qmc_uniforms(
    seed: int, n_paths: int, dim: int, batch_id: Optional[int] = None
) -> np.ndarray:
    """Scrambled Sobol U(0,1) draws of shape ``(n_paths, dim)``."""
    return SobolNormalGenerator(base_seed=int(seed)).uniform(
        n_paths, dim, batch_id=batch_id
    )
