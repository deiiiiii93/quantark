"""Common-random-number coupling for QE/QE-M substep refinement."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np
from scipy.special import ndtr, ndtri

from quantark.montecarlo.qmc_brownian_bridge import BrownianBridge
from quantark.montecarlo.qmc_sobol import SobolNormalGenerator


def invert_brownian_bridge(d_w: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Recover bridge-ordered independent normals from Brownian increments."""

    increments = np.asarray(d_w, dtype=float)
    bridge = BrownianBridge.from_time_grid(np.asarray(times, dtype=float))
    if increments.ndim != 2 or increments.shape[1] != bridge.times.size:
        raise ValueError("d_w must have shape (n_paths, len(times))")
    w = np.cumsum(increments, axis=1)
    z = np.empty_like(w)
    terminal_index = int(bridge.indices[0])
    z[:, 0] = w[:, terminal_index] / np.sqrt(bridge.variances[0])
    for j in range(1, bridge.times.size):
        k = int(bridge.indices[j])
        left = int(bridge.left[j])
        right = int(bridge.right[j])
        t_left = 0.0 if left == -1 else float(bridge.times[left])
        t_right = float(bridge.times[right])
        t_mid = float(bridge.times[k])
        w_left = 0.0 if left == -1 else w[:, left]
        conditional_mean = (
            (t_right - t_mid) * w_left
            + (t_mid - t_left) * w[:, right]
        ) / (t_right - t_left)
        variance = float(bridge.variances[j])
        if variance <= 0.0:
            raise ValueError("Brownian bridge has non-positive conditional variance")
        z[:, j] = (w[:, k] - conditional_mean) / np.sqrt(variance)
    return z


@dataclass(frozen=True)
class CoupledQESubstepDrawProvider:
    """Provide target/fine QE draws from one finest scrambled-Sobol point set.

    The variance-normal and latent QE-branch streams are generated in Brownian
    bridge order before conversion to chronological fine-grid innovations, so
    the lowest Sobol coordinates drive their largest time-scale variation.
    Target innovations are normalized sums of those disjoint fine increments.
    The independent spot stream is likewise aggregated in chronological
    Brownian-increment space and transformed back to each grid's bridge order,
    preserving the same terminal spot factor. Every role has the exact required
    marginal law while target-minus-fine differences retain strong
    common-random-number covariance.
    """

    seed: int
    n_paths: int
    target_dt: np.ndarray
    fine_dt: np.ndarray
    role: str
    reuse_count: int = 1
    _draw_cache: dict = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _draw_cache_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    def __post_init__(self):
        target = np.asarray(self.target_dt, dtype=float)
        fine = np.asarray(self.fine_dt, dtype=float)
        if self.role not in {"target", "fine"}:
            raise ValueError("role must be 'target' or 'fine'")
        if isinstance(self.reuse_count, bool) or int(self.reuse_count) < 1:
            raise ValueError("reuse_count must be a positive integer")
        if self.n_paths <= 0 or target.ndim != 1 or fine.ndim != 1:
            raise ValueError("invalid coupled QE draw dimensions")
        if np.any(target <= 0.0) or np.any(fine <= 0.0):
            raise ValueError("coupled QE time steps must be positive")
        if fine.size % target.size != 0:
            raise ValueError("fine QE grid must be an integer refinement of target")
        ratio = fine.size // target.size
        if ratio < 2:
            raise ValueError("fine QE grid must refine target by at least two")
        if not np.allclose(
            fine.reshape(target.size, ratio).sum(axis=1),
            target,
            rtol=0.0,
            atol=2e-14,
        ):
            raise ValueError("target and fine QE time grids do not align")
        object.__setattr__(self, "target_dt", target)
        object.__setattr__(self, "fine_dt", fine)

    @property
    def ratio(self) -> int:
        return int(self.fine_dt.size // self.target_dt.size)

    @property
    def dimension(self) -> int:
        return int(3 * self.fine_dt.size)

    @property
    def label(self) -> str:
        return f"coupled-qe-bridge-streams-{self.role}-r{self.ratio}"

    @property
    def randomization_key(self) -> tuple:
        return (
            "coupled_qe_bridge_streams_v2",
            int(self.seed),
            int(self.n_paths),
            int(self.target_dt.size),
            int(self.fine_dt.size),
            self.role,
        )

    def draws(
        self,
        *,
        n_paths: int,
        dt_array: np.ndarray,
        batch_id: int | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if int(n_paths) != int(self.n_paths):
            raise ValueError("coupled QE provider path count mismatch")
        requested_dt = np.asarray(dt_array, dtype=float)
        expected_dt = self.target_dt if self.role == "target" else self.fine_dt
        if requested_dt.shape != expected_dt.shape or not np.allclose(
            requested_dt, expected_dt, rtol=0.0, atol=2e-14
        ):
            raise ValueError("coupled QE provider time grid mismatch")

        cache_key = 0 if batch_id is None else int(batch_id)
        if self.reuse_count > 1:
            with self._draw_cache_lock:
                cached = self._draw_cache.get(cache_key)
                if cached is not None:
                    values, remaining = cached
                    if remaining <= 1:
                        self._draw_cache.pop(cache_key, None)
                    else:
                        self._draw_cache[cache_key] = (values, remaining - 1)
                    return values

        m_fine = int(self.fine_dt.size)
        # SobolNormalGenerator.uniform already clips away 0/1. The three
        # blocks are bridge-ordered variance normals, bridge-ordered spot
        # normals, and bridge-ordered latent normals for the QE branch uniform.
        uniforms = SobolNormalGenerator(base_seed=int(self.seed)).uniform(
            int(self.n_paths), 3 * m_fine, batch_id=batch_id
        )
        z_var_bridge_fine = uniforms[:, :m_fine]
        z_spot_bridge_fine = uniforms[:, m_fine : 2 * m_fine]
        z_u_bridge_fine = uniforms[:, 2 * m_fine :]
        ndtri(z_var_bridge_fine, out=z_var_bridge_fine)
        ndtri(z_spot_bridge_fine, out=z_spot_bridge_fine)
        ndtri(z_u_bridge_fine, out=z_u_bridge_fine)

        fine_bridge = BrownianBridge.from_time_grid(np.cumsum(self.fine_dt))
        sqrt_fine_dt = np.sqrt(self.fine_dt)[None, :]
        d_w_var_fine = fine_bridge.transform(z_var_bridge_fine)
        z_var_fine = d_w_var_fine / sqrt_fine_dt
        d_w_u_fine = fine_bridge.transform(z_u_bridge_fine)
        z_u_fine = d_w_u_fine / sqrt_fine_dt
        u_fine = np.clip(ndtr(z_u_fine), 1e-12, 1.0 - 1e-12)
        if self.role == "fine":
            result = z_var_fine, z_spot_bridge_fine, u_fine
            for values in result:
                values.flags.writeable = False
            if self.reuse_count > 1:
                with self._draw_cache_lock:
                    self._draw_cache[cache_key] = (
                        result,
                        int(self.reuse_count) - 1,
                    )
            return result

        ratio = self.ratio
        m_target = int(self.target_dt.size)
        d_w_var_target = d_w_var_fine.reshape(
            self.n_paths, m_target, ratio
        ).sum(axis=2)
        z_var_target = d_w_var_target / np.sqrt(self.target_dt)[None, :]

        d_w_fine = fine_bridge.transform(z_spot_bridge_fine)
        d_w_target = d_w_fine.reshape(
            self.n_paths, m_target, ratio
        ).sum(axis=2)
        z_spot_bridge_target = invert_brownian_bridge(
            d_w_target, np.cumsum(self.target_dt)
        )

        d_w_u_target = d_w_u_fine.reshape(
            self.n_paths, m_target, ratio
        ).sum(axis=2)
        z_u_target = d_w_u_target / np.sqrt(self.target_dt)[None, :]
        u_target = np.clip(ndtr(z_u_target), 1e-12, 1.0 - 1e-12)
        result = z_var_target, z_spot_bridge_target, u_target
        for values in result:
            values.flags.writeable = False
        if self.reuse_count > 1:
            with self._draw_cache_lock:
                self._draw_cache[cache_key] = (
                    result,
                    int(self.reuse_count) - 1,
                )
        return result


__all__ = ["CoupledQESubstepDrawProvider", "invert_brownian_bridge"]
