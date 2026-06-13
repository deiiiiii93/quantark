"""
Garman-Kohlhagen stochastic process for FX rates.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
from scipy.stats import norm, qmc

from quantark.util.exceptions import ValidationError


@dataclass
class GarmanKohlhagenProcess:
    """
    Garman-Kohlhagen process: geometric Brownian motion for an FX rate under
    the domestic risk-neutral measure,

        dS/S = (r_d - r_f) dt + sigma dW

    The foreign rate plays the role of a continuous dividend yield in the
    equivalent Black-Scholes-Merton dynamics.

    Attributes:
        spot: Current FX rate (quote per base)
        volatility: Annualized volatility
        domestic_rate: Domestic (quote currency) risk-free rate
        foreign_rate: Foreign (base currency) risk-free rate
    """

    spot: float
    volatility: float
    domestic_rate: float
    foreign_rate: float = 0.0

    def __post_init__(self):
        """Validate process parameters."""
        if self.spot <= 0:
            raise ValidationError(f"Spot must be positive, got {self.spot}")
        if self.volatility <= 0:
            raise ValidationError(
                f"Volatility must be positive, got {self.volatility}"
            )
        if self.volatility > 5.0:
            raise ValidationError(
                f"Volatility seems unreasonably high: {self.volatility}"
            )

    @property
    def drift(self) -> float:
        """Risk-neutral drift under the domestic measure: r_d - r_f."""
        return self.domestic_rate - self.foreign_rate

    def get_forward_rate(self, time_to_maturity: float) -> float:
        """
        Outright forward rate: F(T) = S * exp((r_d - r_f) * T).

        Raises:
            ValidationError: If time to maturity is negative
        """
        if time_to_maturity < 0:
            raise ValidationError(
                f"Time to maturity must be non-negative, got {time_to_maturity}"
            )
        return self.spot * float(np.exp(self.drift * time_to_maturity))

    def generate_paths(
        self,
        time_steps: int,
        num_paths: int,
        maturity: float,
        seed: int = 1234,
        antithetic: bool = True,
        use_qmc: bool = False,
        return_time_grid: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Simulate FX rate paths with exact log-Euler discretization:

            S_{i+1} = S_i * exp((r_d - r_f - sigma^2/2) dt + sigma sqrt(dt) Z)

        Args:
            time_steps: Number of time steps
            num_paths: Number of simulated paths
            maturity: Path horizon in years
            seed: Random seed
            antithetic: Use antithetic variates (ignored with QMC to preserve
                the low-discrepancy structure)
            use_qmc: Use a scrambled Sobol sequence instead of pseudo-random
                normals
            return_time_grid: Also return the time grid

        Returns:
            Array of shape (num_paths, time_steps + 1) with S_0 in the first
            column; optionally also the time grid of length time_steps + 1.

        Raises:
            ValidationError: If simulation parameters are invalid
        """
        if time_steps <= 0:
            raise ValidationError(f"time_steps must be positive, got {time_steps}")
        if num_paths <= 0:
            raise ValidationError(f"num_paths must be positive, got {num_paths}")
        if maturity <= 0:
            raise ValidationError(f"Maturity must be positive, got {maturity}")

        dt = maturity / time_steps
        normals = self._generate_normals(
            time_steps, num_paths, seed, antithetic, use_qmc
        )

        log_increments = (
            self.drift - 0.5 * self.volatility**2
        ) * dt + self.volatility * np.sqrt(dt) * normals
        log_paths = np.cumsum(log_increments, axis=1)

        paths = np.empty((num_paths, time_steps + 1))
        paths[:, 0] = self.spot
        paths[:, 1:] = self.spot * np.exp(log_paths)

        if return_time_grid:
            return paths, np.arange(time_steps + 1) * dt
        return paths

    @staticmethod
    def _generate_normals(
        time_steps: int,
        num_paths: int,
        seed: Optional[int],
        antithetic: bool,
        use_qmc: bool,
    ) -> np.ndarray:
        """Standard normal increments, optionally antithetic or Sobol-based."""
        if use_qmc:
            sobol = qmc.Sobol(d=time_steps, scramble=True, seed=seed)
            uniforms = sobol.random(num_paths)
            return norm.ppf(uniforms)

        rng = np.random.default_rng(seed)
        if not antithetic:
            return rng.standard_normal((num_paths, time_steps))

        half = (num_paths + 1) // 2
        base = rng.standard_normal((half, time_steps))
        if num_paths % 2 == 0:
            return np.concatenate((base, -base), axis=0)
        return np.concatenate((base, -base[:-1]), axis=0)

    def __repr__(self):
        return (
            f"GarmanKohlhagenProcess(S={self.spot:.6f}, σ={self.volatility:.2%}, "
            f"r_d={self.domestic_rate:.2%}, r_f={self.foreign_rate:.2%})"
        )
