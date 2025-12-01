"""
Created on Mon Nov 17 2025

@author: yaofuxin
@description: MC/QMC GBM/BSM path generators built on top of generic random
               streams, Brownian bridge utilities and variance reduction
               helpers. The generators are NumPy-first and do not depend on
               Dask, making them suitable for use in production pricing
               engines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .qmc_brownian_bridge import apply_brownian_bridge
from .qmc_sobol import PseudoRandomNormalGenerator, RandomStream, SobolNormalGenerator
from .qmc_variance_reduction import (
    VarianceReductionConfig,
    apply_variance_reduction_to_normals,
)


def _build_time_grid(
    maturity: float,
    time_steps: int,
    dt_array: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a time grid and associated dt array.

    Parameters
    ----------
    maturity : float
        Option maturity.
    time_steps : int
        Number of time steps.
    dt_array : np.ndarray, optional
        Custom time step sizes. If provided, it must have length equal to
        time_steps and sum to maturity.

    Returns
    -------
    times : np.ndarray
        Time grid of shape (time_steps,) with strictly increasing values.
    dt : np.ndarray
        Time increments of shape (time_steps,).
    """
    if time_steps <= 0:
        raise ValueError("time_steps must be positive")
    if maturity <= 0.0:
        raise ValueError("maturity must be positive")

    if dt_array is not None:
        dt = np.asarray(dt_array, dtype=float)
        if dt.ndim != 1 or dt.shape[0] != time_steps:
            raise ValueError(
                f"dt_array must be one-dimensional with length {time_steps}, "
                f"got shape {dt.shape}."
            )
        if np.any(dt <= 0.0):
            raise ValueError("dt_array must have strictly positive entries")
        if not np.isclose(dt.sum(), maturity):
            raise ValueError(
                f"Sum of dt_array ({dt.sum()}) must equal maturity ({maturity})."
            )
        times = np.cumsum(dt)
    else:
        dt_value = maturity / float(time_steps)
        dt = np.full(time_steps, dt_value, dtype=float)
        times = np.cumsum(dt)

    return times, dt


@dataclass
class GBMPathGenerator:
    """
    General-purpose GBM/BSM path generator driven by a RandomStream.

    This class supports both classical MC (with PseudoRandomNormalGenerator)
    and QMC (with SobolNormalGenerator), with optional Brownian bridge and
    variance reduction techniques.
    """

    initial_value: float = 100.0
    vol: float = 0.2
    rrf: float = 0.0
    div: float = 0.0
    maturity: float = 1.0
    time_steps: int = 244
    num_paths: int = 10000
    model: str = "bsm"
    random_stream: Optional[RandomStream] = None
    use_brownian_bridge: bool = False
    vr_config: Optional[VarianceReductionConfig] = None
    is_qmc: bool = False
    dt_array: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.initial_value <= 0.0:
            raise ValueError("initial_value must be positive")
        if self.vol < 0.0:
            raise ValueError("vol must be non-negative")
        if self.num_paths <= 0:
            raise ValueError("num_paths must be positive")

        valid_models = ["bsm", "black", "gbm"]
        if self.model not in valid_models:
            raise ValueError(
                f"Invalid model '{self.model}', choose from {valid_models}"
            )

        # Default to pseudorandom generator if none is provided
        if self.random_stream is None:
            self.random_stream = PseudoRandomNormalGenerator()
            self.is_qmc = False

        # Precompute drift based on model type
        self._set_drift()

        # Build time grid
        self.times, self.dt_vector = _build_time_grid(
            maturity=self.maturity,
            time_steps=self.time_steps,
            dt_array=self.dt_array,
        )

    def _set_drift(self) -> None:
        if self.model == "black":
            self.drift = self.rrf
        elif self.model == "bsm":
            self.drift = self.rrf - self.div
        elif self.model == "gbm":
            self.drift = 0.0
        else:
            raise ValueError(f"Invalid model '{self.model}'")

    def _generate_base_normals(self, batch_id: Optional[int]) -> np.ndarray:
        """
        Generate base standard normals before variance reduction is applied.

        For antithetic variates in MC mode, we generate ceil(num_paths / 2)
        base paths so that the expanded matrix after pairing reaches at least
        num_paths paths.
        """
        dim = self.time_steps

        if self.vr_config is not None and self.vr_config.antithetic and not self.is_qmc:
            n_base = (self.num_paths + 1) // 2
        else:
            n_base = self.num_paths

        if self.random_stream is None:
            raise RuntimeError("Random stream is not configured")

        return self.random_stream.normal(n_paths=n_base, dim=dim, batch_id=batch_id)

    def _build_brownian_increments(self, z: np.ndarray) -> np.ndarray:
        """
        Convert standard normals to Brownian increments dW_t.

        If use_brownian_bridge is True, a Brownian bridge is used to map the
        normals into Brownian motion and then to increments. Otherwise, dW_t
        = sqrt(dt_t) * Z_t is used directly.
        """
        if self.use_brownian_bridge:
            return apply_brownian_bridge(z, self.times)

        # Direct construction without Brownian bridge
        sqrt_dt = np.sqrt(self.dt_vector).reshape(1, -1)
        return sqrt_dt * z

    def generate_paths(
        self,
        seed: Optional[int] = None,
        batch_id: Optional[int] = None,
        return_aux: bool = False,
    ) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        """
        Generate GBM/BSM paths.

        Parameters
        ----------
        seed : int, optional
            Seed used for pseudorandom streams. If the underlying random
            stream supports seeding via constructor only (e.g., Sobol),
            this argument is ignored.
        batch_id : int, optional
            Batch identifier for RQMC usage. Different batch_ids produce
            independent randomized Sobol batches.
        return_aux : bool
            If True, also return auxiliary data such as importance sampling
            weights and control variates.

        Returns
        -------
        paths : np.ndarray
            Simulated paths of shape (num_paths, time_steps + 1).
        aux : dict or None
            Dictionary containing auxiliary arrays if return_aux is True.
            Keys:
                - "weights": importance sampling weights (if enabled)
                - "control_variate": GBM control variate (if enabled)
        """
        # Optional reseeding for pseudorandom generator
        if seed is not None and isinstance(
            self.random_stream, PseudoRandomNormalGenerator
        ):
            self.random_stream = PseudoRandomNormalGenerator(seed=seed)

        base_normals = self._generate_base_normals(batch_id=batch_id)

        # Variance reduction applied at the level of standard normals
        z_processed, weights, control_variate = apply_variance_reduction_to_normals(
            n_paths=self.num_paths,
            dim=self.time_steps,
            base_normals=base_normals,
            vr_config=self.vr_config,
            is_qmc=self.is_qmc,
        )

        # Convert to Brownian increments
        dW = self._build_brownian_increments(z_processed)

        # Build GBM/BSM paths
        paths = np.zeros((self.num_paths, self.time_steps + 1), dtype=float)
        paths[:, 0] = self.initial_value

        drift_term = (self.drift - 0.5 * self.vol * self.vol) * self.dt_vector
        drift_term = drift_term.reshape(1, -1)

        diffusion_term = self.vol * dW
        exp_term = np.exp(drift_term + diffusion_term)

        paths[:, 1:] = self.initial_value * np.cumprod(exp_term, axis=1)

        aux: Optional[Dict[str, np.ndarray]] = None
        if return_aux:
            aux = {}
            if weights is not None:
                aux["weights"] = weights
            if control_variate is not None:
                aux["control_variate"] = control_variate

        return paths, aux

    def generate_terminal_values_qmc(
        self,
        batch_id: Optional[int] = None,
        seed: Optional[int] = None,
        dim: int = 1,
    ) -> np.ndarray:
        """
        Generate terminal asset values S_T using low-dimensional QMC.

        This method is intended for path-independent payoffs, where using a
        low-dimensional Sobol sequence (e.g., 1D) provides excellent
        effectiveness and avoids the high-dimensional degradation that occurs
        when full paths are simulated.

        Parameters
        ----------
        batch_id : int, optional
            Batch identifier for RQMC usage.
        seed : int, optional
            Seed for pseudorandom streams; ignored for Sobol-based streams.
        dim : int
            Sobol dimension to use (typically 1 or 2). Only the first column
            is used to construct the terminal value.

        Returns
        -------
        np.ndarray
            Array of terminal values S_T of shape (num_paths,).
        """
        if seed is not None and isinstance(
            self.random_stream, PseudoRandomNormalGenerator
        ):
            self.random_stream = PseudoRandomNormalGenerator(seed=seed)

        # Use low-dimensional normals for the terminal value
        z = self.random_stream.normal(self.num_paths, dim, batch_id=batch_id)
        z_1d = z[:, 0]

        T = self.maturity
        drift_term = (self.drift - 0.5 * self.vol * self.vol) * T
        diffusion_term = self.vol * np.sqrt(T) * z_1d

        return self.initial_value * np.exp(drift_term + diffusion_term)


@dataclass
class GBMPathGeneratorQMC(GBMPathGenerator):
    """
    Convenience subclass for QMC-based GBM path generation.

    This simply defaults to using SobolNormalGenerator and sets is_qmc=True.
    """

    def __post_init__(self) -> None:
        if self.random_stream is None:
            self.random_stream = SobolNormalGenerator()
        self.is_qmc = True
        super().__post_init__()


@dataclass
class MultiAssetGBMPathGenerator:
    """
    Multi-asset GBM/BSM path generator with correlation support.

    This class generates correlated paths for multiple assets using Cholesky
    decomposition to transform independent standard normals into correlated
    normals. It supports both classical MC (with PseudoRandomNormalGenerator)
    and QMC (with SobolNormalGenerator), with optional Brownian bridge per
    asset for improved QMC effectiveness.

    The dimension layout for QMC is: dimensions 0 to T-1 for asset 0,
    dimensions T to 2T-1 for asset 1, etc. This grouping allows per-asset
    Brownian bridge construction.

    Parameters
    ----------
    initial_values : np.ndarray
        Initial spot prices for each asset, shape (n_assets,).
    vols : np.ndarray
        Volatilities for each asset, shape (n_assets,).
    rrfs : np.ndarray
        Risk-free rates for each asset, shape (n_assets,).
    divs : np.ndarray
        Dividend yields for each asset, shape (n_assets,).
    correlation_matrix : np.ndarray
        Correlation matrix of shape (n_assets, n_assets). Must be symmetric
        and positive semi-definite.
    maturity : float
        Time to maturity in years.
    time_steps : int
        Number of time steps for path discretization.
    num_paths : int
        Number of Monte Carlo paths to generate.
    model : str
        Model type: "bsm" (Black-Scholes-Merton), "black", or "gbm".
    random_stream : RandomStream, optional
        Random stream for generating normals. Defaults to pseudorandom.
    use_brownian_bridge : bool
        Whether to use Brownian bridge construction for each asset.
    vr_config : VarianceReductionConfig, optional
        Variance reduction configuration (antithetic not supported for QMC).
    is_qmc : bool
        Whether this is a QMC-based generator.
    dt_array : np.ndarray, optional
        Custom time step sizes.
    """

    initial_values: np.ndarray = None
    vols: np.ndarray = None
    rrfs: np.ndarray = None
    divs: np.ndarray = None
    correlation_matrix: np.ndarray = None
    maturity: float = 1.0
    time_steps: int = 244
    num_paths: int = 10000
    model: str = "bsm"
    random_stream: Optional[RandomStream] = None
    use_brownian_bridge: bool = False
    vr_config: Optional[VarianceReductionConfig] = None
    is_qmc: bool = False
    dt_array: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        # Convert inputs to numpy arrays
        self.initial_values = np.asarray(self.initial_values, dtype=float)
        self.vols = np.asarray(self.vols, dtype=float)
        self.rrfs = np.asarray(self.rrfs, dtype=float)
        self.divs = np.asarray(self.divs, dtype=float)
        self.correlation_matrix = np.asarray(self.correlation_matrix, dtype=float)

        # Determine number of assets
        self.n_assets = self.initial_values.shape[0]

        # Validate shapes
        if self.vols.shape != (self.n_assets,):
            raise ValueError(
                f"vols must have shape ({self.n_assets},), got {self.vols.shape}"
            )
        if self.rrfs.shape != (self.n_assets,):
            raise ValueError(
                f"rrfs must have shape ({self.n_assets},), got {self.rrfs.shape}"
            )
        if self.divs.shape != (self.n_assets,):
            raise ValueError(
                f"divs must have shape ({self.n_assets},), got {self.divs.shape}"
            )
        if self.correlation_matrix.shape != (self.n_assets, self.n_assets):
            raise ValueError(
                f"correlation_matrix must have shape ({self.n_assets}, {self.n_assets}), "
                f"got {self.correlation_matrix.shape}"
            )

        # Validate values
        if np.any(self.initial_values <= 0.0):
            raise ValueError("All initial_values must be positive")
        if np.any(self.vols < 0.0):
            raise ValueError("All vols must be non-negative")
        if self.num_paths <= 0:
            raise ValueError("num_paths must be positive")

        # Validate correlation matrix
        if not np.allclose(self.correlation_matrix, self.correlation_matrix.T):
            raise ValueError("correlation_matrix must be symmetric")
        if not np.allclose(np.diag(self.correlation_matrix), 1.0):
            raise ValueError("correlation_matrix diagonal must be all ones")

        valid_models = ["bsm", "black", "gbm"]
        if self.model not in valid_models:
            raise ValueError(
                f"Invalid model '{self.model}', choose from {valid_models}"
            )

        # Default to pseudorandom generator if none is provided
        if self.random_stream is None:
            self.random_stream = PseudoRandomNormalGenerator()
            self.is_qmc = False

        # Compute Cholesky decomposition for correlation
        try:
            self.cholesky_L = np.linalg.cholesky(self.correlation_matrix)
        except np.linalg.LinAlgError:
            raise ValueError(
                "correlation_matrix must be positive semi-definite. "
                "Cholesky decomposition failed."
            )

        # Precompute drifts based on model type
        self._set_drifts()

        # Build time grid
        self.times, self.dt_vector = _build_time_grid(
            maturity=self.maturity,
            time_steps=self.time_steps,
            dt_array=self.dt_array,
        )

    def _set_drifts(self) -> None:
        """Set drift for each asset based on model type."""
        if self.model == "black":
            self.drifts = self.rrfs.copy()
        elif self.model == "bsm":
            self.drifts = self.rrfs - self.divs
        elif self.model == "gbm":
            self.drifts = np.zeros(self.n_assets, dtype=float)
        else:
            raise ValueError(f"Invalid model '{self.model}'")

    def _generate_base_normals(self, batch_id: Optional[int]) -> np.ndarray:
        """
        Generate base independent standard normals for all assets and time steps.

        The dimension is n_assets * time_steps, laid out so that dimensions
        0 to time_steps-1 correspond to asset 0, etc.

        Returns
        -------
        np.ndarray
            Array of shape (n_paths, n_assets * time_steps).
        """
        total_dim = self.n_assets * self.time_steps

        if self.vr_config is not None and self.vr_config.antithetic and not self.is_qmc:
            n_base = (self.num_paths + 1) // 2
        else:
            n_base = self.num_paths

        if self.random_stream is None:
            raise RuntimeError("Random stream is not configured")

        return self.random_stream.normal(
            n_paths=n_base, dim=total_dim, batch_id=batch_id
        )

    def _apply_correlation(self, z_independent: np.ndarray) -> np.ndarray:
        """
        Apply Cholesky correlation to independent normals.

        Parameters
        ----------
        z_independent : np.ndarray
            Independent standard normals of shape (n_paths, n_assets, time_steps).

        Returns
        -------
        np.ndarray
            Correlated standard normals of shape (n_paths, n_assets, time_steps).
        """
        n_paths, n_assets, time_steps = z_independent.shape

        # Apply Cholesky: for each time step, correlate across assets
        # z_correlated[i, :, t] = L @ z_independent[i, :, t]
        # Using einsum for efficient batch matrix multiplication
        z_correlated = np.einsum("ij,kjt->kit", self.cholesky_L, z_independent)

        return z_correlated

    def _build_brownian_increments_per_asset(
        self, z: np.ndarray, asset_idx: int
    ) -> np.ndarray:
        """
        Convert standard normals to Brownian increments for a single asset.

        Parameters
        ----------
        z : np.ndarray
            Standard normals for one asset, shape (n_paths, time_steps).
        asset_idx : int
            Asset index (for debugging/logging purposes).

        Returns
        -------
        np.ndarray
            Brownian increments dW of shape (n_paths, time_steps).
        """
        if self.use_brownian_bridge:
            return apply_brownian_bridge(z, self.times)

        # Direct construction without Brownian bridge
        sqrt_dt = np.sqrt(self.dt_vector).reshape(1, -1)
        return sqrt_dt * z

    def generate_paths(
        self,
        seed: Optional[int] = None,
        batch_id: Optional[int] = None,
        return_aux: bool = False,
    ) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        """
        Generate correlated GBM/BSM paths for all assets.

        Parameters
        ----------
        seed : int, optional
            Seed used for pseudorandom streams. Ignored for Sobol-based streams.
        batch_id : int, optional
            Batch identifier for RQMC usage. Different batch_ids produce
            independent randomized Sobol batches.
        return_aux : bool
            If True, also return auxiliary data such as importance sampling
            weights.

        Returns
        -------
        paths : np.ndarray
            Simulated paths of shape (n_assets, num_paths, time_steps + 1).
        aux : dict or None
            Dictionary containing auxiliary arrays if return_aux is True.
            Keys:
                - "weights": importance sampling weights (if enabled)
        """
        # Optional reseeding for pseudorandom generator
        if seed is not None and isinstance(
            self.random_stream, PseudoRandomNormalGenerator
        ):
            self.random_stream = PseudoRandomNormalGenerator(seed=seed)

        # Generate base normals: shape (n_paths or n_base, n_assets * time_steps)
        base_normals = self._generate_base_normals(batch_id=batch_id)

        # Apply variance reduction if configured
        total_dim = self.n_assets * self.time_steps
        z_processed, weights, _ = apply_variance_reduction_to_normals(
            n_paths=self.num_paths,
            dim=total_dim,
            base_normals=base_normals,
            vr_config=self.vr_config,
            is_qmc=self.is_qmc,
        )

        # Reshape to (n_paths, n_assets, time_steps)
        # Layout: first time_steps columns are asset 0, next are asset 1, etc.
        z_reshaped = z_processed.reshape(self.num_paths, self.n_assets, self.time_steps)

        # Build Brownian increments per asset independently (with optional Brownian bridge)
        # This preserves QMC effectiveness for each asset's marginal distribution
        dW_independent = np.zeros((self.n_assets, self.num_paths, self.time_steps))
        for i in range(self.n_assets):
            dW_independent[i] = self._build_brownian_increments_per_asset(
                z_reshaped[:, i, :], asset_idx=i
            )

        # Apply correlation to the Brownian increments
        # Reshape for correlation: (n_paths, n_assets, time_steps)
        dW_independent_transposed = np.transpose(dW_independent, (1, 0, 2))
        dW_correlated = self._apply_correlation(dW_independent_transposed)
        # Transpose back to (n_assets, n_paths, time_steps)
        dW = np.transpose(dW_correlated, (1, 0, 2))

        # Build GBM/BSM paths for each asset
        paths = np.zeros(
            (self.n_assets, self.num_paths, self.time_steps + 1), dtype=float
        )

        for i in range(self.n_assets):
            paths[i, :, 0] = self.initial_values[i]

            drift_term = (self.drifts[i] - 0.5 * self.vols[i] ** 2) * self.dt_vector
            drift_term = drift_term.reshape(1, -1)

            diffusion_term = self.vols[i] * dW[i]
            exp_term = np.exp(drift_term + diffusion_term)

            paths[i, :, 1:] = self.initial_values[i] * np.cumprod(exp_term, axis=1)

        aux: Optional[Dict[str, np.ndarray]] = None
        if return_aux:
            aux = {}
            if weights is not None:
                aux["weights"] = weights

        return paths, aux

    def generate_terminal_values(
        self,
        batch_id: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate terminal asset values S_T for all assets using low-dimensional QMC.

        This method is intended for path-independent payoffs on multiple assets,
        where using a low-dimensional Sobol sequence (n_assets dimensions) provides
        excellent effectiveness.

        Parameters
        ----------
        batch_id : int, optional
            Batch identifier for RQMC usage.
        seed : int, optional
            Seed for pseudorandom streams; ignored for Sobol-based streams.

        Returns
        -------
        np.ndarray
            Array of terminal values of shape (n_assets, num_paths).
        """
        if seed is not None and isinstance(
            self.random_stream, PseudoRandomNormalGenerator
        ):
            self.random_stream = PseudoRandomNormalGenerator(seed=seed)

        # Generate low-dimensional normals: one dimension per asset
        z = self.random_stream.normal(self.num_paths, self.n_assets, batch_id=batch_id)

        # Apply correlation
        z_correlated = (self.cholesky_L @ z.T).T  # shape (num_paths, n_assets)

        T = self.maturity
        terminal_values = np.zeros((self.n_assets, self.num_paths), dtype=float)

        for i in range(self.n_assets):
            drift_term = (self.drifts[i] - 0.5 * self.vols[i] ** 2) * T
            diffusion_term = self.vols[i] * np.sqrt(T) * z_correlated[:, i]
            terminal_values[i] = self.initial_values[i] * np.exp(
                drift_term + diffusion_term
            )

        return terminal_values


@dataclass
class MultiAssetGBMPathGeneratorQMC(MultiAssetGBMPathGenerator):
    """
    Convenience subclass for QMC-based multi-asset GBM path generation.

    This simply defaults to using SobolNormalGenerator and sets is_qmc=True.
    """

    def __post_init__(self) -> None:
        if self.random_stream is None:
            self.random_stream = SobolNormalGenerator()
        self.is_qmc = True
        super().__post_init__()


__all__ = [
    "GBMPathGenerator",
    "GBMPathGeneratorQMC",
    "MultiAssetGBMPathGenerator",
    "MultiAssetGBMPathGeneratorQMC",
]
