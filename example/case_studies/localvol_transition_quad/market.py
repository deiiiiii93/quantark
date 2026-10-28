"""Market inputs: flat r/q and a ``LocalVolSurface`` (spec D6, D9)."""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from quantark.param import FlatRateCurve, FlatVolSurface
from quantark.param.vol.surface_history import IvSurfaceArtifact
from quantark.volmodels.localvol import LocalVolSurface, build_dupire_local_vol
from quantark.volmodels.risk.contracts import SurfaceBump
from quantark.volmodels.risk.scenarios import bump_local_vol_surface

FIXTURE = Path(__file__).resolve().parents[3] / "test/golden/modelvalidation/data/iv_surface_20240208.json"
FIXTURE_RATE = 0.02            # the committed snowball_localvol_1d.yaml study rate
VEGA_BUMP = 0.01               # absolute parallel shift of lv_grid
RATE_BUMP = 1e-4               # 1bp on flat r and on flat q


@dataclass(frozen=True)
class Market:
    spot: float
    r: float
    q: float
    surface: LocalVolSurface
    iv_surface: object = None   # only used to give production environments a vol quote

    def bumped(self, kind: str, sign: int) -> "Market":
        """Central-difference scenario: 'vega' (LV grid), 'rho' (r), 'carry' (q). The surface is never rebuilt."""
        if sign not in (-1, 1):
            raise ValueError("sign must be -1 or 1")
        if kind == "vega":
            return replace(self, surface=bump_local_vol_surface(self.surface, SurfaceBump.parallel(VEGA_BUMP), sign))
        if kind == "rho":
            return replace(self, r=self.r + sign * RATE_BUMP)
        if kind == "carry":
            return replace(self, q=self.q + sign * RATE_BUMP)
        raise ValueError(f"unknown bump kind {kind!r}")

    def vol_quote(self):
        return self.iv_surface if self.iv_surface is not None else FlatVolSurface(
            volatility=float(np.median(self.surface.lv_grid)))


def constant_surface(sigma: float) -> LocalVolSurface:
    """Time- and spot-independent local vol: the Black-Scholes control."""
    return LocalVolSurface(strike_grid=np.array([1e-8, 1e12]), time_grid=np.array([0.0]),
                           lv_grid=np.full((1, 2), float(sigma)))


@lru_cache(maxsize=4)
def load_fixture(r: float = FIXTURE_RATE, q: float | None = None) -> Market:
    """The 2024-02-08 CSI1000 artifact with Dupire built ONCE under flat r and flat q.

    ``q`` defaults to the parity-implied yield at the artifact's longest pillar.
    A refused Dupire build propagates: no floor, no smoothing (spec D6).
    """
    artifact = IvSurfaceArtifact.from_file(FIXTURE)
    if q is None:
        _, yields = artifact.implied_q_pillars(r)
        q = float(yields[-1])
    grid = artifact.grid_vol_surface()
    surface = build_dupire_local_vol(grid, spot=float(artifact.s0), rate_curve=FlatRateCurve(r),
                                     div_yield=lambda t: q)
    return Market(float(artifact.s0), float(r), float(q), surface, iv_surface=grid)
