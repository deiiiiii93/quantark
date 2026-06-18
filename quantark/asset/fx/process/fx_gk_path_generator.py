"""Forward-consistent Garman-Kohlhagen path generator for FX Monte Carlo.

S(t_i) = F(0, t_i) * exp(-0.5 sigma^2 t_i + sigma W(t_i)), built with per-step
log drift  drift_k = [log F(0,t_k) - log F(0,t_{k-1})] - 0.5 sigma^2 dt_k. For
flat curves this collapses to the constant (r_d - r_f - 0.5 sigma^2) dt; for a
term structure it keeps E[S(t_i)] = F(0, t_i) exact at every grid time.

The continuous-time model is piecewise-log-forward-linear between grid points,
so the Brownian-bridge crossing formula consumed downstream is exact for that
piecewise model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from quantark.montecarlo.qmc_brownian_bridge import apply_brownian_bridge
from quantark.montecarlo.qmc_sobol import (
    PseudoRandomNormalGenerator,
    RandomStream,
    SobolNormalGenerator,
)
from quantark.montecarlo.qmc_variance_reduction import (
    VarianceReductionConfig,
    apply_variance_reduction_to_normals,
)
from quantark.util.exceptions import ValidationError


@dataclass
class FxGKPathGenerator:
    """GK path generator driven by a RandomStream and a forward curve."""

    spot: float
    sigma: float
    forward_fn: Callable[[float], float]
    times: np.ndarray
    num_paths: int
    random_stream: Optional[RandomStream] = None
    use_brownian_bridge: bool = False
    vr_config: Optional[VarianceReductionConfig] = None
    is_qmc: bool = False

    def __post_init__(self) -> None:
        if self.spot <= 0.0:
            raise ValidationError(f"spot must be positive, got {self.spot}")
        if self.sigma < 0.0:
            raise ValidationError(f"sigma must be non-negative, got {self.sigma}")
        if self.num_paths <= 0:
            raise ValidationError(f"num_paths must be positive, got {self.num_paths}")

        self.times = np.asarray(self.times, dtype=float)
        if self.times.ndim != 1 or self.times.size == 0:
            raise ValidationError("times must be a non-empty 1-D array")
        grid = np.concatenate([[0.0], self.times])
        self.dt = np.diff(grid)
        if np.any(self.dt <= 0.0):
            raise ValidationError("times must be strictly increasing and positive")

        if self.random_stream is None:
            self.random_stream = PseudoRandomNormalGenerator()
            self.is_qmc = False

        self.n_steps = int(self.times.size)

        forwards = np.array(
            [float(self.forward_fn(float(t))) for t in self.times], dtype=float
        )
        if np.any(forwards <= 0.0):
            raise ValidationError("forward curve must be strictly positive")
        log_fwd = np.log(forwards)
        prev_log_fwd = np.concatenate([[np.log(self.spot)], log_fwd[:-1]])
        self._drift_term = (
            (log_fwd - prev_log_fwd) - 0.5 * self.sigma * self.sigma * self.dt
        )

    def _base_normals(self, batch_id: Optional[int]) -> np.ndarray:
        antithetic = (
            self.vr_config is not None
            and self.vr_config.antithetic
            and not self.is_qmc
        )
        n_base = (self.num_paths + 1) // 2 if antithetic else self.num_paths
        assert self.random_stream is not None
        return self.random_stream.normal(
            n_paths=n_base, dim=self.n_steps, batch_id=batch_id
        )

    def _increments(self, z: np.ndarray) -> np.ndarray:
        if self.use_brownian_bridge:
            return apply_brownian_bridge(z, self.times)
        return np.sqrt(self.dt)[None, :] * z

    def generate_paths(
        self,
        seed: Optional[int] = None,
        batch_id: Optional[int] = None,
        return_aux: bool = False,
    ) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        """Simulate FX paths of shape (num_paths, n_steps + 1) with S_0 first."""
        if seed is not None and isinstance(
            self.random_stream, PseudoRandomNormalGenerator
        ):
            self.random_stream = PseudoRandomNormalGenerator(seed=seed)

        base = self._base_normals(batch_id)
        z, weights, _ = apply_variance_reduction_to_normals(
            n_paths=self.num_paths,
            dim=self.n_steps,
            base_normals=base,
            vr_config=self.vr_config,
            is_qmc=self.is_qmc,
        )
        dW = self._increments(z)
        log_increments = self._drift_term[None, :] + self.sigma * dW

        paths = np.empty((self.num_paths, self.n_steps + 1), dtype=float)
        paths[:, 0] = self.spot
        paths[:, 1:] = self.spot * np.exp(np.cumsum(log_increments, axis=1))

        aux: Optional[Dict[str, np.ndarray]] = None
        if return_aux:
            aux = {"weights": weights} if weights is not None else {}
        return paths, aux


@dataclass
class FxGKPathGeneratorQMC(FxGKPathGenerator):
    """Convenience subclass defaulting to a scrambled Sobol stream."""

    def __post_init__(self) -> None:
        if self.random_stream is None:
            self.random_stream = SobolNormalGenerator()
        self.is_qmc = True
        super().__post_init__()


__all__ = ["FxGKPathGenerator", "FxGKPathGeneratorQMC"]
