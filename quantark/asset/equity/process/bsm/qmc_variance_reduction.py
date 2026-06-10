"""
Created on Mon Nov 17 2025

@author: yaofuxin
@description: Variance reduction utilities (antithetic variates, control
               variates, importance sampling) designed to work with both
               classical MC and QMC without destroying Sobol structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class VarianceReductionConfig:
    """
    Configuration for variance reduction techniques.

    Parameters
    ----------
    antithetic : bool
        Whether to use antithetic variates. This is intended for classical
        MC (pseudorandom) and is disabled for QMC by default.
    control_variate : bool
        Whether to construct a GBM-based control variate.
    importance_sampling : bool
        Whether to apply a mean shift to the driving Brownian increments.
    importance_shift : float
        Shift parameter used in importance sampling. The same shift is applied
        to all normal dimensions.
    control_variate_mu : float, optional
        Drift parameter for the GBM control variate.
    control_variate_sigma : float, optional
        Volatility for the GBM control variate.
    control_variate_T : float, optional
        Maturity of the GBM control variate.
    """

    antithetic: bool = False
    control_variate: bool = False
    importance_sampling: bool = False
    importance_shift: float = 0.0
    control_variate_mu: Optional[float] = None
    control_variate_sigma: Optional[float] = None
    control_variate_T: Optional[float] = None


def build_antithetic_pairs(z: np.ndarray) -> np.ndarray:
    """
    Construct antithetic variates from a base matrix of standard normals.

    Parameters
    ----------
    z : np.ndarray
        Base standard normals of shape (n_base_paths, dim).

    Returns
    -------
    np.ndarray
        Array of shape (2 * n_base_paths, dim) containing [z, -z].
    """
    z = np.asarray(z, dtype=float)
    return np.concatenate([z, -z], axis=0)


def apply_importance_sampling_shift(z: np.ndarray, shift: float) -> np.ndarray:
    """
    Apply a constant mean shift to standard normal samples.

    This function is safe for QMC usage because it preserves the low-discrepancy
    structure of the underlying sequence while changing the sampling measure.

    Parameters
    ----------
    z : np.ndarray
        Standard normal samples of shape (n_paths, dim).
    shift : float
        Constant shift applied to all dimensions.

    Returns
    -------
    np.ndarray
        Shifted standard normals of the same shape as z.
    """
    z = np.asarray(z, dtype=float)
    if shift == 0.0:
        return z
    return z + shift


def importance_sampling_weights(z_shifted: np.ndarray, shift: float) -> np.ndarray:
    """
    Compute likelihood ratio weights for Gaussian importance sampling.

    If Z' ~ N(shift, 1), the Radon–Nikodym derivative dP/dQ at Z' is

        w = exp(shift * Z' - 0.5 * shift^2)

    For a d-dimensional vector of independent components, the total weight is

        w = exp(shift * sum(Z'_j) - 0.5 * d * shift^2).

    Parameters
    ----------
    z_shifted : np.ndarray
        Shifted standard normal samples of shape (n_paths, dim), i.e., samples
        drawn from N(shift, 1) along each dimension.
    shift : float
        Shift parameter used in the importance sampling scheme.

    Returns
    -------
    np.ndarray
        Weights of shape (n_paths,) to reweight payoffs back to the original
        N(0, 1) measure.
    """
    z_shifted = np.asarray(z_shifted, dtype=float)
    if z_shifted.ndim != 2:
        raise ValueError("z_shifted must be a 2D array of shape (n_paths, dim)")
    if shift == 0.0:
        return np.ones(z_shifted.shape[0], dtype=float)

    n_paths, dim = z_shifted.shape
    sum_z = np.sum(z_shifted, axis=1)
    exponent = shift * sum_z - 0.5 * (shift ** 2) * dim
    return np.exp(exponent)


def gbm_control_variate(
    z: np.ndarray,
    mu: float,
    sigma: float,
    T: float,
) -> np.ndarray:
    """
    Construct a GBM-based control variate using the same driving normals.

    This control variate follows the exact GBM solution:

        S_t = exp((mu - 0.5 * sigma^2) * t + sigma * sqrt(t) * Z)

    using the column-wise time points.

    Parameters
    ----------
    z : np.ndarray
        Standard normal samples of shape (n_paths, n_steps).
    mu : float
        Drift parameter.
    sigma : float
        Volatility parameter.
    T : float
        Maturity.

    Returns
    -------
    np.ndarray
        Control variate array of shape (n_paths, n_steps), with one control
        variate value per time step along each path.
    """
    z = np.asarray(z, dtype=float)
    if z.ndim != 2:
        raise ValueError("z must be a 2D array of shape (n_paths, n_steps)")

    n_steps = z.shape[1]
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")

    dt = T / n_steps
    t = np.arange(1, n_steps + 1, dtype=float) * dt

    drift = (mu - 0.5 * sigma * sigma) * t
    diffusion = sigma * np.sqrt(t)

    drift = drift.reshape(1, -1)
    diffusion = diffusion.reshape(1, -1)

    return np.exp(drift + diffusion * z)


def apply_variance_reduction_to_normals(
    n_paths: int,
    dim: int,
    base_normals: np.ndarray,
    vr_config: Optional[VarianceReductionConfig],
    is_qmc: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Apply configured variance reduction techniques to a matrix of normals.

    Parameters
    ----------
    n_paths : int
        Desired final number of paths.
    dim : int
        Dimension of each path.
    base_normals : np.ndarray
        Base N(0, 1) samples of shape (n_paths, dim) or smaller if antithetic
        variates will expand the path count.
    vr_config : VarianceReductionConfig, optional
        Variance reduction configuration.
    is_qmc : bool
        Whether the underlying sampling is QMC (Sobol). Antithetic variates are
        disabled by default in QMC mode to avoid destroying low-discrepancy
        structure.

    Returns
    -------
    z_processed : np.ndarray
        Processed normals of shape (n_paths, dim) after applying antithetic
        variates and/or importance sampling.
    weights : np.ndarray or None
        Importance sampling weights if enabled, otherwise None.
    control_variate : np.ndarray or None
        GBM-based control variate if enabled, otherwise None.
    """
    z = np.asarray(base_normals, dtype=float)
    if z.ndim != 2:
        raise ValueError("base_normals must be a 2D array")

    weights: Optional[np.ndarray] = None
    control_variate: Optional[np.ndarray] = None

    if vr_config is None:
        if z.shape != (n_paths, dim):
            raise ValueError(
                f"Expected base_normals to have shape ({n_paths}, {dim}) without "
                f"variance reduction, got {z.shape}."
            )
        return z, weights, control_variate

    # Antithetic variates: only for classical MC by default.
    if vr_config.antithetic:
        if is_qmc:
            raise ValueError(
                "Antithetic variates are not enabled for QMC mode to avoid "
                "breaking Sobol low-discrepancy properties."
            )
        # base_normals is assumed to have shape (ceil(n_paths / 2), dim)
        z = build_antithetic_pairs(z)
        if z.shape[0] < n_paths:
            raise ValueError(
                "Antithetic base_normals must be large enough to reach n_paths."
            )
        z = z[:n_paths, :]

    # Importance sampling (safe for both MC and QMC)
    if vr_config.importance_sampling and vr_config.importance_shift != 0.0:
        z = apply_importance_sampling_shift(z, vr_config.importance_shift)
        weights = importance_sampling_weights(z, vr_config.importance_shift)

    # Control variate based on GBM analytical solution
    if vr_config.control_variate:
        if (
            vr_config.control_variate_mu is None
            or vr_config.control_variate_sigma is None
            or vr_config.control_variate_T is None
        ):
            raise ValueError(
                "control_variate_mu, control_variate_sigma and control_variate_T "
                "must be provided in VarianceReductionConfig when control_variate "
                "is enabled."
            )
        control_variate = gbm_control_variate(
            z,
            mu=vr_config.control_variate_mu,
            sigma=vr_config.control_variate_sigma,
            T=vr_config.control_variate_T,
        )

    if z.shape != (n_paths, dim):
        raise ValueError(
            f"Processed normals have shape {z.shape}, expected ({n_paths}, {dim})."
        )

    return z, weights, control_variate


__all__ = [
    "VarianceReductionConfig",
    "build_antithetic_pairs",
    "apply_importance_sampling_shift",
    "importance_sampling_weights",
    "gbm_control_variate",
    "apply_variance_reduction_to_normals",
]


