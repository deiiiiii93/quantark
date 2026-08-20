"""
Created on Mon Nov 17 2025

@author: yaofuxin
@description: Brownian bridge utilities for constructing Brownian motion
               paths from standard normal draws, plus barrier crossing
               helpers for barrier-style options.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantark.montecarlo.bridge_kernels import bridge_transform
from quantark.util.numerical import safe_log


@dataclass
class BrownianBridge:
    """
    Brownian bridge constructor for a given time grid.

    The bridge is defined on times t_1 < ... < t_n with t_0 = 0.0. Given
    independent standard normal samples Z_1, ..., Z_n, the bridge produces
    Brownian motion values W(t_k) with the correct covariance structure but
    in a dimension ordering that is better suited for QMC (low effective
    dimension).
    """

    times: np.ndarray
    indices: np.ndarray
    left: np.ndarray
    right: np.ndarray
    variances: np.ndarray

    @classmethod
    def from_time_grid(cls, times: np.ndarray) -> "BrownianBridge":
        """
        Precompute the Brownian bridge structure for a strictly increasing
        time grid.

        Parameters
        ----------
        times : np.ndarray
            One-dimensional array of shape (n_steps,) with t_k > 0 and
            strictly increasing.
        """
        times = np.asarray(times, dtype=float)
        if times.ndim != 1:
            raise ValueError("times must be a one-dimensional array")
        if not np.all(np.isfinite(times)):
            raise ValueError("times must be finite")
        if np.any(times <= 0.0):
            raise ValueError("times must be strictly positive")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("times must be strictly increasing")

        n = times.shape[0]
        indices = np.empty(n, dtype=int)
        left = np.full(n, -1, dtype=int)
        right = np.full(n, -1, dtype=int)
        variances = np.zeros(n, dtype=float)

        # First dimension corresponds to terminal time T
        indices[0] = n - 1
        left[0] = -1
        right[0] = -1
        variances[0] = times[-1]

        # Recursively fill midpoints
        counter = 1

        def build(l_idx: int, r_idx: int, cnt: int) -> int:
            # Build bridge between times t_l and t_r, where l_idx/r_idx are
            # indices in [0, n-1], with l_idx == -1 representing time 0.
            if r_idx - l_idx <= 1:
                return cnt
            m_idx = (l_idx + r_idx) // 2
            indices[cnt] = m_idx
            left[cnt] = l_idx
            right[cnt] = r_idx

            t_l = 0.0 if l_idx == -1 else times[l_idx]
            t_r = times[r_idx]
            t_m = times[m_idx]

            variance = (t_m - t_l) * (t_r - t_m) / (t_r - t_l)
            variances[cnt] = max(variance, 0.0)

            cnt = build(l_idx, m_idx, cnt + 1)
            cnt = build(m_idx, r_idx, cnt)
            return cnt

        counter = build(-1, n - 1, counter)
        if counter != n:
            raise RuntimeError(
                "BrownianBridge construction failed to fill all dimensions"
            )

        return cls(
            times=times, indices=indices, left=left, right=right, variances=variances
        )

    def transform(self, z: np.ndarray) -> np.ndarray:
        """
        Map independent normals z to Brownian increments via Brownian bridge.

        Parameters
        ----------
        z : np.ndarray
            Array of shape (n_paths, n_steps) with independent N(0, 1)
            samples along the second axis.

        Returns
        -------
        np.ndarray
            Array of Brownian increments dW of shape (n_paths, n_steps),
            ordered chronologically in time.
        """
        z = np.asarray(z, dtype=float)
        if z.ndim != 2:
            raise ValueError("z must be a 2D array of shape (n_paths, n_steps)")

        n_steps = z.shape[1]
        if n_steps != self.times.shape[0]:
            raise ValueError(
                f"z has {n_steps} time steps but BrownianBridge is configured "
                f"for {self.times.shape[0]} steps."
            )

        # Delegated to the shared kernel (Numba-accelerated when
        # quantark[accel] is installed, NumPy reference otherwise -- bitwise
        # either way, asserted in test_bridge_transform_kernel.py).
        return bridge_transform(
            z, self.times, self.indices, self.left, self.right, self.variances
        )


def apply_brownian_bridge(z: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    Convenience function: apply Brownian bridge to standard normals.

    Parameters
    ----------
    z : np.ndarray
        Standard normal samples of shape (n_paths, n_steps).
    times : np.ndarray
        Time grid with shape (n_steps,) and strictly increasing values.

    Returns
    -------
    np.ndarray
        Brownian increments dW with shape (n_paths, n_steps).
    """
    bridge = BrownianBridge.from_time_grid(times)
    return bridge.transform(z)


def apply_brownian_bridge_multi_asset(z: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    Apply Brownian bridge construction to multi-asset standard normals.

    This function applies the Brownian bridge transformation independently
    to each asset's time series. This is the recommended approach for
    multi-asset QMC simulations as it preserves the low-discrepancy
    properties of the underlying sequence for each asset's marginal
    distribution.

    Parameters
    ----------
    z : np.ndarray
        Standard normal samples of shape (n_assets, n_paths, n_steps).
        Each asset's normals are in z[i, :, :].
    times : np.ndarray
        Time grid with shape (n_steps,) and strictly increasing values.
        The same time grid is used for all assets.

    Returns
    -------
    np.ndarray
        Brownian increments dW with shape (n_assets, n_paths, n_steps),
        where dW[i, :, :] contains the increments for asset i.

    Notes
    -----
    The Brownian bridge is applied per-asset because:
    1. It preserves QMC effectiveness for path-dependent payoffs
    2. Correlation between assets is applied separately after building
       the increments, using Cholesky decomposition
    3. Each asset's marginal distribution benefits from the low effective
       dimension property of Brownian bridge construction

    Example
    -------
    >>> import numpy as np
    >>> from asset.equity.process.qmc_brownian_bridge import (
    ...     apply_brownian_bridge_multi_asset
    ... )
    >>> n_assets, n_paths, n_steps = 3, 1000, 252
    >>> z = np.random.standard_normal((n_assets, n_paths, n_steps))
    >>> times = np.linspace(1/252, 1.0, n_steps)
    >>> dW = apply_brownian_bridge_multi_asset(z, times)
    >>> dW.shape
    (3, 1000, 252)
    """
    z = np.asarray(z, dtype=float)
    times = np.asarray(times, dtype=float)

    if z.ndim != 3:
        raise ValueError(
            f"z must be a 3D array of shape (n_assets, n_paths, n_steps), "
            f"got ndim={z.ndim}"
        )

    n_assets, n_paths, n_steps = z.shape

    if times.ndim != 1:
        raise ValueError("times must be a 1D array")
    if times.shape[0] != n_steps:
        raise ValueError(
            f"times has {times.shape[0]} elements but z has {n_steps} time steps"
        )

    # Create bridge once and reuse for all assets
    bridge = BrownianBridge.from_time_grid(times)

    # Apply bridge to each asset
    dW = np.zeros_like(z)
    for i in range(n_assets):
        dW[i] = bridge.transform(z[i])

    return dW


def compute_step_crossing_probabilities(
    paths: np.ndarray,
    barrier_level: float,
    sigma: float,
    times: np.ndarray,
) -> np.ndarray:
    """
    Compute step-wise barrier crossing probabilities using a Brownian bridge.

    For each simulated path and each time interval, it returns the probability
    that the path has crossed the barrier between the endpoints.

    Parameters
    ----------
    paths : np.ndarray
        Simulated spot paths of shape (n_paths, n_steps + 1).
    barrier_level : float
        Barrier level in spot space.
    sigma : float
        Volatility used in the GBM dynamics.
    times : np.ndarray
        Time grid of shape (n_steps,) corresponding to the path intervals
        (excluding t=0).

    Returns
    -------
    np.ndarray
        Array of shape (n_paths, n_steps) containing the crossing probability
        per step.
    """
    paths = np.asarray(paths, dtype=float)
    times = np.asarray(times, dtype=float)

    if paths.ndim != 2:
        raise ValueError("paths must be a 2D array of shape (n_paths, n_steps + 1)")
    if times.ndim != 1:
        raise ValueError("times must be a 1D array of shape (n_steps,)")
    if barrier_level <= 0.0:
        raise ValueError("barrier_level must be positive")
    sigma_vec = np.asarray(sigma, dtype=float)
    if np.any(sigma_vec <= 0.0):
        raise ValueError("sigma must be positive")
    if sigma_vec.ndim not in (0, 1):
        raise ValueError("sigma must be a scalar or 1D per-step array")
    if paths.shape[1] != times.shape[0] + 1:
        raise ValueError(
            "paths second dimension must be n_steps + 1 where n_steps = len(times)"
        )
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be strictly increasing")

    n_paths, n_cols = paths.shape
    n_steps = n_cols - 1

    S0 = paths[:, :-1]
    S1 = paths[:, 1:]
    dt = np.empty(n_steps, dtype=float)
    dt[0] = times[0]
    if n_steps > 1:
        dt[1:] = np.diff(times)

    # Broadcast dt to match shape of path slices
    dt_matrix = dt.reshape(1, -1)
    if sigma_vec.ndim == 1:
        if sigma_vec.shape[0] != n_steps:
            raise ValueError(
                f"per-step sigma must have length {n_steps}, got {sigma_vec.shape[0]}"
            )
        sig2 = (sigma_vec * sigma_vec).reshape(1, -1)
    else:
        sig2 = float(sigma_vec) * float(sigma_vec)
    h2 = sig2 * dt_matrix

    # Initialize probabilities per step
    prob = np.zeros_like(S0, dtype=float)

    # Determine where paths are on different sides of the barrier
    crossed_mask = ((S0 < barrier_level) & (S1 > barrier_level)) | (
        (S0 > barrier_level) & (S1 < barrier_level)
    )
    touched_mask = (S0 == barrier_level) | (S1 == barrier_level)

    # Opposite-side or touching endpoints imply a hit with probability 1
    prob[crossed_mask | touched_mask] = 1.0

    # Same-side endpoints: Brownian-bridge crossing probability
    same_side = ~(crossed_mask | touched_mask)
    if np.any(same_side):
        log_term = safe_log(S0 / barrier_level) * safe_log(S1 / barrier_level)
        bridge_prob = np.exp(-2.0 * log_term / h2)
        bridge_prob = np.clip(bridge_prob, 0.0, 1.0)
        prob[same_side] = bridge_prob[same_side]

    return prob


def compute_barrier_crossing_probabilities(
    paths: np.ndarray,
    barrier_level: float,
    sigma: float,
    times: np.ndarray,
) -> np.ndarray:
    """
    Approximate barrier crossing probabilities using a Brownian bridge.

    This utility is designed for geometric Brownian motion paths and can be
    used by pricing engines for barrier-style options. For each simulated
    path, it returns the probability that the path has crossed the barrier
    between any pair of consecutive observation times.

    Parameters
    ----------
    paths : np.ndarray
        Simulated spot paths of shape (n_paths, n_steps + 1). The second axis
        is assumed to be ordered chronologically.
    barrier_level : float
        Barrier level in spot space.
    sigma : float
        Volatility used in the GBM dynamics.
    times : np.ndarray
        Time grid of shape (n_steps,) corresponding to the path intervals
        (excluding t=0).

    Returns
    -------
    np.ndarray
        Array of shape (n_paths,) containing the approximate probability that
        each path crosses the barrier at least once.
    """
    prob = compute_step_crossing_probabilities(paths, barrier_level, sigma, times)

    # Combine step-wise probabilities into overall crossing probability
    # Assuming conditional independence given endpoints (standard Brownian bridge approximation)
    no_hit_prob = np.prod(1.0 - prob, axis=1)
    return 1.0 - no_hit_prob


__all__ = [
    "BrownianBridge",
    "apply_brownian_bridge",
    "apply_brownian_bridge_multi_asset",
    "compute_step_crossing_probabilities",
    "compute_barrier_crossing_probabilities",
]
