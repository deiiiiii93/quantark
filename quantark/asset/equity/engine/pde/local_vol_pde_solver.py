"""Equity local-volatility PDE solver for European vanillas, on the shared core (spec 3C §5.3)."""
from __future__ import annotations

from typing import Dict, Optional

from quantark.asset.equity.engine.localvol_greeks import local_vol_rho_theta
from quantark.asset.equity.engine.settlement_support import terminal_lifecycle_pv
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.param.pde_removed_names import reject_removed_kwargs
from quantark.volmodels.localvol import LocalVolSurface

from .core.adapters.local_vol import LocalVolEuropeanAdapter
from .core.prepare import Operation, OutputRequest, bind_layout
from .european_pde_solver import EuropeanPDESolver
from .local_vol_surface import LocalVolSurfaceMixin


class LocalVolPDESolver(LocalVolSurfaceMixin, EuropeanPDESolver):
    """European vanillas under a Dupire local-volatility surface, marched on the core.

    The shared presets, projection, damping row, boundary rows and readout; one local-vol row per step at the
    interval midpoint on the solve's S grid. Greeks: price, delta and gamma from the grid; rho and theta re-solved
    on a frozen layout with the surface held fixed (spec 3C decision 7). Deterministic: never invokes MC.
    """

    _adapter_class = LocalVolEuropeanAdapter

    def __init__(self, params: Optional[PDEParams] = None, local_vol_surface: Optional[LocalVolSurface] = None,
                 **removed):
        reject_removed_kwargs(type(self).__name__, removed)
        super().__init__(params)
        self._init_local_vol(local_vol_surface)

    def price(self, product, pricing_env, *, lifecycle_state=None) -> float:
        return self._with_surface(pricing_env, lambda: EuropeanPDESolver.price(
            self, product, pricing_env, lifecycle_state=lifecycle_state))

    def calculate_greeks(self, product, pricing_env, *, lifecycle_state=None) -> Dict[str, float]:
        def run():
            grid = EuropeanPDESolver.calculate_greeks(self, product, pricing_env, lifecycle_state=lifecycle_state)
            if terminal_lifecycle_pv(lifecycle_state, pricing_env) is not None or product.get_maturity(pricing_env) <= 0.0:
                return grid                     # a terminal state or maturity: the three keys (spec 3C §5.3, P9)
            ctx = self._fixed_surface_context(product, pricing_env)
            bump = self.params.get_effective_bump_config()
            grid.update(local_vol_rho_theta(
                lambda p, e, surface: ctx.price(p, e, lifecycle_state=lifecycle_state), product, pricing_env,
                self._active_lv_surface, rate_bump=bump.rate_bump, theta_days=bump.time_bump_days,
                base=grid["price"]))
            return grid
        return self._with_surface(pricing_env, run)

    def _price_with_surface(self, product, pricing_env, lv_surface) -> float:
        """The price with ``lv_surface`` held fixed: the structured vol-model risk contract
        (``quantark.asset.vol_model_risk``) recognises a local-vol engine by this method."""
        return type(self)(params=self.params, local_vol_surface=lv_surface).price(product, pricing_env)

    def _fixed_surface_context(self, product, pricing_env):
        """A bump context holding this solve's surface and layout: rate shifts re-solve on the same grid, a theta
        roll rebinds time only (spec 3C §4.5: 1.1e-5 rho, 1.2e-6 theta against 7.2e-5 / 8.7e-5 on fresh grids)."""
        clone = type(self)(params=self.params, local_vol_surface=self._active_lv_surface)
        clone._frozen_base_layout = bind_layout(clone, product, pricing_env, OutputRequest(Operation.PRICE))
        return clone

    def _session_outputs(self, product, pricing_env, **kwargs):
        return self._with_surface(pricing_env, lambda: EuropeanPDESolver._session_outputs(
            self, product, pricing_env, **kwargs))

    def calculate_spot_greeks_curve(self, product, pricing_env, spot_levels):
        return self._with_surface(pricing_env, lambda: EuropeanPDESolver.calculate_spot_greeks_curve(
            self, product, pricing_env, spot_levels))

    def __repr__(self):
        return "LocalVolPDESolver()"
