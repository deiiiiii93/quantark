"""Surface construction and finite-difference utilities for reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from quantark.util.exceptions import ValidationError


@dataclass(frozen=True)
class GridSpec:
    spot_min_frac: float = 0.60
    spot_max_frac: float = 1.20
    spot_nodes: int = 31

    q_bump_bp: float = 500.0
    q_nodes: int = 21

    vol_rel_width: float = 0.05
    vol_nodes: int = 11

    q_bump_for_rho: float = 1e-4  # 1bp absolute bump
    time_bump_years: float = 1.0 / 252.0


def build_spot_grid(spot: float, spec: GridSpec) -> np.ndarray:
    if spot <= 0:
        raise ValidationError(f"spot must be positive, got {spot}")
    if spec.spot_nodes < 3:
        raise ValidationError("spot_nodes must be >= 3")
    return np.linspace(spec.spot_min_frac * spot, spec.spot_max_frac * spot, spec.spot_nodes)


def build_q_grid(q0: float, spec: GridSpec) -> np.ndarray:
    if spec.q_nodes < 3:
        raise ValidationError("q_nodes must be >= 3")
    width = float(spec.q_bump_bp) / 1e4
    q_min = max(0.0, q0 - width)
    q_max = q0 + width
    if q_max < q_min:
        raise ValidationError(f"Invalid q grid bounds: [{q_min}, {q_max}]")
    return np.linspace(q_min, q_max, spec.q_nodes)


def build_vol_grid(vol0: float, spec: GridSpec) -> np.ndarray:
    if vol0 <= 0:
        raise ValidationError(f"vol must be positive, got {vol0}")
    if spec.vol_nodes < 3:
        raise ValidationError("vol_nodes must be >= 3")
    width = float(spec.vol_rel_width)
    return vol0 * np.linspace(1.0 - width, 1.0 + width, spec.vol_nodes)


def derivative_1d(values: np.ndarray, x: np.ndarray, axis: int) -> np.ndarray:
    if values.shape[axis] != x.size:
        raise ValidationError(
            f"axis length mismatch: values.shape[{axis}]={values.shape[axis]} vs x.size={x.size}"
        )
    return np.gradient(values, x, axis=axis, edge_order=2)


def derivative_2d_mixed(
    values: np.ndarray, x0: np.ndarray, x1: np.ndarray, axis0: int = 0, axis1: int = 1
) -> np.ndarray:
    d_dx0 = derivative_1d(values, x0, axis=axis0)
    return derivative_1d(d_dx0, x1, axis=axis1)


@dataclass(frozen=True)
class SurfaceSet:
    spot_grid: np.ndarray
    q_grid: np.ndarray
    vol_grid: np.ndarray

    pv_sq: np.ndarray  # shape (n_spot, n_q)
    delta_sq: np.ndarray  # dV/dS, shape (n_spot, n_q)
    rhoq_sq: np.ndarray  # per 1% q change, shape (n_spot, n_q)
    v_sq: np.ndarray  # d^2V/(dS dq), shape (n_spot, n_q)

    pv_sv: np.ndarray  # shape (n_spot, n_vol)
    rhoq_sv: np.ndarray  # per 1% q change, shape (n_spot, n_vol)


def compute_surfaces_from_pv(
    *,
    spot_grid: np.ndarray,
    q_grid: np.ndarray,
    vol_grid: np.ndarray,
    pv_sq: np.ndarray,
    pv_sv: np.ndarray,
    pv_sv_q_up: Optional[np.ndarray],
    q_bump_for_rho: float,
) -> SurfaceSet:
    if pv_sq.shape != (spot_grid.size, q_grid.size):
        raise ValidationError("pv_sq shape mismatch")
    if pv_sv.shape != (spot_grid.size, vol_grid.size):
        raise ValidationError("pv_sv shape mismatch")

    delta_sq = derivative_1d(pv_sq, spot_grid, axis=0)
    dv_dq = derivative_1d(pv_sq, q_grid, axis=1)
    rhoq_sq = dv_dq * 0.01
    v_sq = derivative_2d_mixed(pv_sq, spot_grid, q_grid, axis0=0, axis1=1)

    if pv_sv_q_up is None:
        raise ValidationError("pv_sv_q_up is required to compute rhoq_sv")
    if pv_sv_q_up.shape != pv_sv.shape:
        raise ValidationError("pv_sv_q_up shape mismatch")
    rhoq_sv = (pv_sv_q_up - pv_sv) * (0.01 / q_bump_for_rho)

    return SurfaceSet(
        spot_grid=spot_grid,
        q_grid=q_grid,
        vol_grid=vol_grid,
        pv_sq=pv_sq,
        delta_sq=delta_sq,
        rhoq_sq=rhoq_sq,
        v_sq=v_sq,
        pv_sv=pv_sv,
        rhoq_sv=rhoq_sv,
    )
