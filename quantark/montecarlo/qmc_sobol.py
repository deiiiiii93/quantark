"""
Created on Mon Nov 17 2025

@author: yaofuxin
@description: Sobol-based and pseudorandom normal generators for MC/QMC path
               construction, with support for RQMC batching.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Hashable, Optional, Protocol

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

    def uniform(
        self, n_paths: int, dim: int, batch_id: Optional[int] = None
    ) -> np.ndarray:
        """Uniform (0,1) samples, the dual of ``normal`` for inverse-CDF draws.

        ``batch_id`` is accepted for API symmetry but ignored (independent batches
        come from the RNG state).
        """
        return self._rng.random(size=(n_paths, dim))


class QMCDrawCache:
    """Byte-budgeted, thread-safe LRU cache of deterministic draw blocks.

    Scrambled-Sobol blocks are pure functions of (kind, seed, batch_id,
    n_paths, dim), and CRN risk reports deliberately reprice with the same
    seed, so identical blocks are otherwise regenerated on every bump.
    Cached arrays are marked read-only: callers that need to mutate must
    request a writable copy (see ``qmc_draws``). Eviction is least-recently
    -used by total bytes; a block larger than the whole budget is simply
    not cached.
    """

    def __init__(self, max_bytes: int):
        self._store: "OrderedDict[Hashable, np.ndarray]" = OrderedDict()
        self._lock = threading.Lock()
        self._bytes = 0
        self.max_bytes = int(max_bytes)
        self.hits = 0
        self.misses = 0

    def get(self, key: Hashable) -> Optional[np.ndarray]:
        with self._lock:
            block = self._store.get(key)
            if block is None:
                self.misses += 1
                return None
            self._store.move_to_end(key)
            self.hits += 1
            return block

    def put(self, key: Hashable, block: np.ndarray) -> np.ndarray:
        block.flags.writeable = False
        if block.nbytes > self.max_bytes:
            return block
        with self._lock:
            existing = self._store.pop(key, None)
            if existing is not None:
                self._bytes -= existing.nbytes
            self._store[key] = block
            self._bytes += block.nbytes
            while self._bytes > self.max_bytes and self._store:
                _, evicted = self._store.popitem(last=False)
                self._bytes -= evicted.nbytes
        return block

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._bytes = 0

    @property
    def current_bytes(self) -> int:
        return self._bytes


def _default_cache_bytes() -> int:
    """Budget from QUANTARK_QMC_CACHE_MB (default 2048 MB; 0 disables)."""
    try:
        megabytes = float(os.environ.get("QUANTARK_QMC_CACHE_MB", "2048"))
    except ValueError:
        megabytes = 2048.0
    return max(int(megabytes * 1024 * 1024), 0)


_DRAW_CACHE = QMCDrawCache(_default_cache_bytes())


def get_qmc_draw_cache() -> QMCDrawCache:
    """The process-wide draw cache shared by the QMC engine adapters."""
    return _DRAW_CACHE


def set_qmc_cache_budget_bytes(max_bytes: int) -> None:
    """Resize (and prune) the process-wide draw cache budget."""
    _DRAW_CACHE.max_bytes = max(int(max_bytes), 0)
    if _DRAW_CACHE.current_bytes > _DRAW_CACHE.max_bytes:
        _DRAW_CACHE.clear()


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
        # Guard against 0/1 values that map to +/-inf under ndtri; both the
        # clip and the inverse-CDF run in place — bit-identical values, no
        # intermediate full-block copies.
        eps = 1e-12
        np.clip(u, eps, 1.0 - eps, out=u)

        if special is None:
            # Fallback, should not happen if _check_scipy passed
            from scipy.stats import norm  # type: ignore

            z = np.asarray(norm.ppf(u), dtype=float)
        else:
            special.ndtri(u, out=u)
            z = u

        if n_paths != n_total:
            z = z[:n_paths]

        return z

    def uniform(
        self, n_paths: int, dim: int, batch_id: Optional[int] = None
    ) -> np.ndarray:
        """Scrambled Sobol uniforms in (0,1) (pre-ndtri), the dual of ``normal``.

        Uses exactly 2**m points (m = ceil(log2 n_paths)) to preserve balance, then
        truncates to ``n_paths``; clipped off {0,1} so downstream ndtri stays finite.
        """
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._check_scipy()
        n_total = _next_power_of_two(n_paths)
        if self.strict_power_of_two and n_total != n_paths:
            raise ValueError(
                f"SobolNormalGenerator with strict_power_of_two=True requires "
                f"n_paths to be a power of two, got {n_paths}."
            )
        m = int(np.log2(n_total))
        engine = self._make_engine(dim=dim, batch_id=batch_id)
        u = engine.random_base2(m)
        eps = 1e-12
        np.clip(u, eps, 1.0 - eps, out=u)
        if n_paths != n_total:
            u = u[:n_paths]
        return u


__all__ = [
    "RandomStream",
    "PseudoRandomNormalGenerator",
    "SobolNormalGenerator",
    "QMCDrawCache",
    "get_qmc_draw_cache",
    "set_qmc_cache_budget_bytes",
]
