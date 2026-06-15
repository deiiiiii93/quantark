"""Immutable sticky-strike surface and Heston-parameter scenarios."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from quantark.param import GridVolSurface
from quantark.util.exceptions import ValidationError
from quantark.volmodels.heston.params import HestonParams
from quantark.volmodels.localvol.surface import LocalVolSurface
from quantark.volmodels.slv.leverage import LeverageSurface

from .contracts import SurfaceBucketKind, SurfaceBump


def _grid_shape(surface) -> tuple[int, int]:
    if isinstance(surface, GridVolSurface):
        return len(surface.maturities), len(surface.strikes)
    if isinstance(surface, (LocalVolSurface, LeverageSurface)):
        return int(surface.time_grid.size), int(surface.strike_grid.size)
    raise ValidationError(f"unsupported surface type: {type(surface).__name__}")


def all_surface_bumps(surface, bump_size: float = 0.01) -> tuple[SurfaceBump, ...]:
    """Generate parallel, row, column, and node sticky-strike buckets.

    This expands quadratically with grid size. Using the result for fully recalibrated
    SLV risk performs two leverage calibrations per returned bucket.
    """
    n_t, n_k = _grid_shape(surface)
    bumps = [SurfaceBump.parallel(bump_size)]
    bumps.extend(SurfaceBump.maturity_row(i, bump_size) for i in range(n_t))
    bumps.extend(SurfaceBump.strike_column(j, bump_size) for j in range(n_k))
    bumps.extend(SurfaceBump.node(i, j, bump_size) for i in range(n_t) for j in range(n_k))
    return tuple(bumps)


def _bumped_grid(values: np.ndarray, bump: SurfaceBump, direction: int, relative: bool) -> np.ndarray:
    if direction not in (-1, 1):
        raise ValidationError("direction must be -1 or 1")
    out = np.array(values, dtype=float, copy=True)
    n_t, n_k = out.shape
    if bump.maturity_index is not None and bump.maturity_index >= n_t:
        raise ValidationError(f"maturity_index {bump.maturity_index} outside surface")
    if bump.strike_index is not None and bump.strike_index >= n_k:
        raise ValidationError(f"strike_index {bump.strike_index} outside surface")
    if bump.kind == SurfaceBucketKind.PARALLEL:
        mask = np.ones_like(out, dtype=bool)
    elif bump.kind == SurfaceBucketKind.MATURITY_ROW:
        mask = np.zeros_like(out, dtype=bool)
        mask[bump.maturity_index, :] = True
    elif bump.kind == SurfaceBucketKind.STRIKE_COLUMN:
        mask = np.zeros_like(out, dtype=bool)
        mask[:, bump.strike_index] = True
    else:
        mask = np.zeros_like(out, dtype=bool)
        mask[bump.maturity_index, bump.strike_index] = True
    if relative:
        out[mask] *= 1.0 + direction * bump.bump_size
    else:
        out[mask] += direction * bump.bump_size
    return out


def bump_grid_vol_surface(
    surface: GridVolSurface, bump: SurfaceBump, direction: int
) -> GridVolSurface:
    """Apply an absolute sticky-strike IV quote bump."""
    return GridVolSurface(
        strikes=list(surface.strikes),
        maturities=list(surface.maturities),
        iv_grid=_bumped_grid(surface.iv_grid, bump, direction, relative=False),
        strike_interpolation=surface.strike_interpolation,
    )


def bump_local_vol_surface(
    surface: LocalVolSurface, bump: SurfaceBump, direction: int
) -> LocalVolSurface:
    """Apply an absolute bump to a derived local-volatility surface."""
    return LocalVolSurface(
        strike_grid=np.array(surface.strike_grid, copy=True),
        time_grid=np.array(surface.time_grid, copy=True),
        lv_grid=_bumped_grid(surface.lv_grid, bump, direction, relative=False),
    )


def bump_leverage_surface(
    surface: LeverageSurface, bump: SurfaceBump, direction: int
) -> LeverageSurface:
    """Apply a relative multiplicative bump to an SLV leverage surface."""
    return LeverageSurface(
        strike_grid=np.array(surface.strike_grid, copy=True),
        time_grid=np.array(surface.time_grid, copy=True),
        leverage_grid=_bumped_grid(surface.leverage_grid, bump, direction, relative=True),
    )


def heston_parameter_bump_size(
    params: HestonParams,
    name: str,
    relative_bump: float = 0.01,
    variance_floor: float = 1e-4,
    positive_floor: float = 1e-3,
    rho_bump: float = 1e-3,
) -> float:
    """Return the configured central-difference bump for a Heston parameter."""
    if name not in ("v0", "kappa", "theta", "sigma", "rho"):
        raise ValidationError(f"unknown Heston parameter: {name}")
    for bump_name, value in (
        ("relative_bump", relative_bump),
        ("variance_floor", variance_floor),
        ("positive_floor", positive_floor),
        ("rho_bump", rho_bump),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValidationError(f"{bump_name} must be finite and positive")
    if name == "rho":
        return float(rho_bump)
    floor = variance_floor if name in ("v0", "theta") else positive_floor
    return float(max(abs(getattr(params, name)) * relative_bump, floor))


def bump_heston_parameter(
    params: HestonParams,
    name: str,
    direction: int,
    relative_bump: float = 0.01,
    variance_floor: float = 1e-4,
    positive_floor: float = 1e-3,
    rho_bump: float = 1e-3,
) -> tuple[HestonParams, float]:
    """Return an immutable Heston parameter bump without clipping boundaries."""
    if direction not in (-1, 1):
        raise ValidationError("direction must be -1 or 1")
    bump = heston_parameter_bump_size(
        params, name, relative_bump, variance_floor, positive_floor, rho_bump
    )
    return replace(params, **{name: getattr(params, name) + direction * bump}), bump
