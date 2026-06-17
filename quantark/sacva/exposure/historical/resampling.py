"""Multivariate historical resampling schemes. All schemes resample whole
same-date return/residual **vectors** by a common time index, so cross-factor
co-movement is empirical (no Cholesky recolouring).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from quantark.util.exceptions import ValidationError


class ResamplingScheme(Enum):
    IID_RAW = "iid_raw"
    BLOCK_FHS = "block_fhs"
    STATIONARY_BLOCK = "stationary_block"


@dataclass
class Resampler:
    scheme: ResamplingScheme
    block_length: Optional[int] = None
    expected_block_length: Optional[float] = None
    overlap: Optional[bool] = None        # fixed-block only; must be None for stationary
    seed: int = 0

    def __post_init__(self):
        if self.scheme is ResamplingScheme.STATIONARY_BLOCK and self.overlap is not None:
            raise ValidationError("overlap is not applicable to STATIONARY_BLOCK")

    def sample(self, vectors, n_paths, n_steps, min_raw_obs=0):
        """vectors: (n_obs, n_factors). Returns (n_paths, n_steps, n_factors)."""
        V = np.asarray(vectors, float)
        if V.ndim != 2 or V.shape[1] == 0:
            raise ValidationError("vectors must be (n_obs, n_factors>0)")
        if not np.all(np.isfinite(V)):
            raise ValidationError("vectors contain non-finite values")
        if not isinstance(n_paths, int) or not isinstance(n_steps, int) \
                or n_paths <= 0 or n_steps <= 0:
            raise ValidationError("n_paths and n_steps must be positive ints")
        n_obs = V.shape[0]
        if n_obs < min_raw_obs:
            raise ValidationError(f"insufficient obs {n_obs} < {min_raw_obs}")
        rng = np.random.default_rng(self.seed)
        idx = self._index_matrix(rng, n_obs, n_paths, n_steps)
        return V[idx]

    def _index_matrix(self, rng, n_obs, n_paths, n_steps):
        if self.scheme is ResamplingScheme.IID_RAW:
            return rng.integers(0, n_obs, size=(n_paths, n_steps))
        if self.scheme is ResamplingScheme.BLOCK_FHS:
            if self.block_length is None or self.block_length < 1:
                raise ValidationError("BLOCK_FHS requires explicit block_length >= 1")
            if self.block_length > n_obs:
                raise ValidationError("block_length exceeds history")
            return self._fixed_block_idx(
                rng, n_obs, n_paths, n_steps, self.block_length,
                overlap=(True if self.overlap is None else self.overlap))
        if self.scheme is ResamplingScheme.STATIONARY_BLOCK:
            if self.expected_block_length is None or self.expected_block_length < 1:
                raise ValidationError("STATIONARY_BLOCK requires expected_block_length >= 1")
            return self._stationary_idx(rng, n_obs, n_paths, n_steps)
        raise ValidationError(f"unknown scheme {self.scheme}")

    def _fixed_block_idx(self, rng, n_obs, n_paths, n_steps, L, overlap):
        if overlap:
            starts_pool = np.arange(0, n_obs - L + 1)
        else:
            starts_pool = np.arange(0, n_obs - L + 1, L)
        out = np.empty((n_paths, n_steps), dtype=int)
        for p in range(n_paths):
            pos = 0
            while pos < n_steps:
                start = int(rng.choice(starts_pool))
                take = min(L, n_steps - pos)
                out[p, pos:pos + take] = np.arange(start, start + take)
                pos += take
        return out

    def _stationary_idx(self, rng, n_obs, n_paths, n_steps):
        # Politis-Romano: geometric block length, restart prob p = 1/expected_len,
        # SAME index process applied to all factors (via V[idx]); circular indexing.
        p = 1.0 / float(self.expected_block_length)
        out = np.empty((n_paths, n_steps), dtype=int)
        for r in range(n_paths):
            cur = int(rng.integers(0, n_obs))
            for t in range(n_steps):
                if t > 0 and rng.random() < p:
                    cur = int(rng.integers(0, n_obs))    # restart new block
                out[r, t] = cur
                cur = (cur + 1) % n_obs                   # circular advance
        return out
