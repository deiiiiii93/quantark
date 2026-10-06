"""The vol-of-vol -> 0 limit of Heston and Heston-SLV as a local-volatility surface.

With sigma = 0 the variance follows its mean-reversion ODE, v(t) = theta + (v0 - theta) e^(-kappa t), so the spot
is a one-factor diffusion with sigma_loc(S, t) = L(S, t) sqrt(v(t)) (L = 1 under Heston). The 2D ADI barrier solvers
price this limit as a 1D local-vol problem on the shared PDE core, where their own grid is ill-conditioned.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from .params import HestonParams

if TYPE_CHECKING:  # the SLV package imports Heston; the leverage is only called, never constructed, here
    from quantark.volmodels.slv.leverage import LeverageSurface


@dataclass(frozen=True)
class DeterministicVarianceLocalVol:
    """``local_vol(S, t) = L(S, t) * sqrt(v(t))`` on the deterministic variance path of ``params``."""

    params: HestonParams
    leverage: Optional["LeverageSurface"] = None

    def variance(self, t):
        """v(t) = theta + (v0 - theta) e^(-kappa t): exactly v0 without mean reversion."""
        p = self.params
        return p.theta + (p.v0 - p.theta) * np.exp(-p.kappa * np.asarray(t, dtype=float))

    def local_vol(self, spot, t):
        s = np.asarray(spot, dtype=float)
        vol = np.sqrt(self.variance(t))
        if self.leverage is not None:
            return self.leverage.leverage(s, t) * vol
        return np.broadcast_to(vol, np.broadcast(s, np.asarray(t, dtype=float)).shape).copy()
