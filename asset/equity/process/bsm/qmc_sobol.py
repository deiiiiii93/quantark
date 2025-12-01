"""
Created on Mon Nov 17 2025

@author: yaofuxin
@description: Sobol-based and pseudorandom normal generators for MC/QMC path
               construction, with support for RQMC batching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

try:
    from scipy import special
    from scipy.stats import qmc

    HAS_SCIPY_QMC = True
except ImportError:  # pragma: no cover - fallback path without SciPy
    HAS_SCIPY_QMC = False
    special = None  # type: ignore
    qmc = None  # type: ignore


class RandomStream(Protocol):
    """
    Minimal interface for random streams used by path generators.

    Implementations should return standard normal random numbers with shape
    (n_paths, dim). The optional batch_id is used to generate independent
    randomized QMC batches.
    """

    def normal(
        self, n_paths: int, dim: int, batch_id: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate standard normal random numbers.

        Parameters
        ----------
        n_paths : int
            Number of Monte Carlo paths.
        dim : int
            Dimension of each path (typically number of time steps).
        batch_id : int, optional
            Identifier for the RQMC batch. Implementations can use this to
            produce independent randomized batches.

        Returns
        -------
        np.ndarray
            Array of shape (n_paths, dim) with N(0, 1) samples.
        """
        ...


@dataclass
class PseudoRandomNormalGenerator:
    """
    Pseudorandom standard normal generator based on NumPy.

    This implements the RandomStream protocol and is suitable for classical
    Monte Carlo simulations.
    """

    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def normal(
        self, n_paths: int, dim: int, batch_id: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate standard normal samples using NumPy's Generator.

        The batch_id argument is accepted for API compatibility but ignored,
        since independent batches are handled via the RNG state.
        """
        return self._rng.standard_normal(size=(n_paths, dim))


def _next_power_of_two(n: int) -> int:
    """Return the smallest power of two >= n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


@dataclass
class SobolNormalGenerator:
    """
    Sobol-based standard normal generator with optional RQMC batching.

    This class wraps scipy.stats.qmc.Sobol and uses scipy.special.ndtri to
    transform low-discrepancy uniform samples into standard normals. To
    preserve Sobol structure, the number of generated samples is always a
    power of two (2**m). If the requested path count is not a power of two,
    the generator will either round up and truncate or raise an error,
    depending on the configuration.
    """

    base_seed: int = 1234
    strict_power_of_two: bool = False

    def _check_scipy(self) -> None:
        if not HAS_SCIPY_QMC:
            raise ImportError(
                "SobolNormalGenerator requires scipy.stats.qmc and scipy.special.ndtri. "
                "Please install SciPy or use PseudoRandomNormalGenerator instead."
            )

    def _make_engine(self, dim: int, batch_id: Optional[int]) -> "qmc.Sobol":
        """
        Create a new Sobol engine for the given dimension and batch.

        A different seed is used for each batch_id to obtain independent
        scrambled Sobol sequences (RQMC).
        """
        self._check_scipy()
        # Use different seeds for different batches to obtain independent scrambles
        if batch_id is None:
            seed = self.base_seed
        else:
            seed = self.base_seed + int(batch_id)
        return qmc.Sobol(d=dim, scramble=True, seed=seed)  # type: ignore[call-arg]

    def normal(
        self, n_paths: int, dim: int, batch_id: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate standard normal samples using a scrambled Sobol sequence.

        Parameters
        ----------
        n_paths : int
            Requested number of paths.
        dim : int
            Dimension of each path (typically number of time steps).
        batch_id : int, optional
            Batch identifier for RQMC. Different batch_ids produce independent
            randomized Sobol sequences.

        Returns
        -------
        np.ndarray
            Array of shape (n_paths, dim) containing N(0, 1) samples.
        """
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")
        if dim <= 0:
            raise ValueError("dim must be positive")

        self._check_scipy()

        # Ensure we use exactly 2**m Sobol points to preserve balance properties
        n_total = _next_power_of_two(n_paths)
        if self.strict_power_of_two and n_total != n_paths:
            raise ValueError(
                f"SobolNormalGenerator with strict_power_of_two=True requires "
                f"n_paths to be a power of two, got {n_paths}."
            )

        m = int(np.log2(n_total))
        engine = self._make_engine(dim=dim, batch_id=batch_id)

        # Use random_base2 to get exactly 2**m points
        u = engine.random_base2(m)  # shape: (n_total, dim)

        # Transform uniforms to standard normals using ndtri
        if special is None:
            # Fallback, should not happen if _check_scipy passed
            from scipy.stats import norm  # type: ignore

            z = norm.ppf(u)
        else:
            z = special.ndtri(u)

        if n_paths != n_total:
            z = z[:n_paths]

        return np.asarray(z, dtype=float)


__all__ = [
    "RandomStream",
    "PseudoRandomNormalGenerator",
    "SobolNormalGenerator",
]
