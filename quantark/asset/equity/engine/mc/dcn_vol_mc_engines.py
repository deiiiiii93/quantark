"""LV / Heston DCN MC engines (spec WP1.5): path generation overrides only.

Payoff, discounting, leg decomposition, and event stats are inherited from
DCNMCEngine; these classes replace ``_simulate`` with the same per-step loops
used by phoenix_vol_mc_engines.py (LV: sigma = lv.local_vol(S, t); Heston:
full-truncation Euler, as in the phoenix Euler precedent).

``HestonSLVDCNMCEngine`` is an explicit stretch item (spec §4) and is NOT
implemented here.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from quantark.asset.equity.engine.mc.dcn_mc_engine import DCNMCEngine
from quantark.param import GridVolSurface
from quantark.util.exceptions import PricingError
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.localvol import LocalVolSurface, build_dupire_local_vol


class LocalVolDCNMCEngine(DCNMCEngine):
    """DCN MC under Dupire local vol built from ``env.vol_surface``.

    Needs a :class:`GridVolSurface` on the environment (or a prebuilt
    :class:`LocalVolSurface`), mirroring the phoenix LV engine.
    """

    def __init__(
        self, local_vol_surface: Optional[LocalVolSurface] = None, **kwargs
    ):
        super().__init__(**kwargs)
        self._prebuilt = local_vol_surface

    def _build_surface(self, env) -> LocalVolSurface:
        if self._prebuilt is not None:
            return self._prebuilt
        if not isinstance(env.vol_surface, GridVolSurface):
            raise PricingError(
                "LocalVolDCNMCEngine needs a GridVolSurface or a prebuilt "
                "LocalVolSurface"
            )
        return build_dupire_local_vol(
            env.vol_surface,
            spot=env.spot,
            rate_curve=env.rate_curve,
            div_yield=env.get_div_yield,
        )

    def _simulate(self, spot0, term, dt_array, pricing_env,
                  n_paths, batch_id) -> np.ndarray:
        lv = self._build_surface(pricing_env)
        n_steps = dt_array.size
        z_all = self._draws(n_steps, n_paths, batch_id)
        nodes = np.empty((z_all.shape[0], n_steps + 1))
        spot = np.full(z_all.shape[0], float(spot0))
        nodes[:, 0] = spot
        t = 0.0
        sqrt_dt = np.sqrt(dt_array)
        for i in range(n_steps):
            vol = np.asarray(lv.local_vol(spot, t), dtype=float)
            drift = float(term.rrf[i] - term.div[i])
            spot = spot * np.exp(
                (drift - 0.5 * vol * vol) * dt_array[i]
                + vol * sqrt_dt[i] * z_all[:, i]
            )
            nodes[:, i + 1] = spot
            t += float(dt_array[i])
        return nodes


class HestonDCNMCEngine(DCNMCEngine):
    """DCN MC under Heston (full-truncation Euler, phoenix precedent)."""

    def __init__(self, model_params: HestonParams, **kwargs):
        super().__init__(**kwargs)
        self.model_params = model_params

    def _simulate(self, spot0, term, dt_array, pricing_env,
                  n_paths, batch_id) -> np.ndarray:
        p = self.model_params
        n_steps = dt_array.size
        z_all = self._draws(2 * n_steps, n_paths, batch_id)
        z_all = z_all.reshape(-1, 2, n_steps)
        n = z_all.shape[0]
        nodes = np.empty((n, n_steps + 1))
        log_s = np.full(n, np.log(float(spot0)))
        var = np.full(n, max(float(p.v0), 0.0))
        nodes[:, 0] = np.exp(log_s)
        rho = float(np.clip(p.rho, -0.999, 0.999))
        rho_bar = float(np.sqrt(max(1.0 - rho * rho, 0.0)))
        for i in range(n_steps):
            dt = float(dt_array[i])
            sqrt_dt = np.sqrt(dt)
            v_plus = np.maximum(var, 0.0)
            sqrt_v = np.sqrt(v_plus)
            d_w_v = z_all[:, 0, i] * sqrt_dt
            d_w_s = (rho * z_all[:, 0, i] + rho_bar * z_all[:, 1, i]) * sqrt_dt
            drift = float(term.rrf[i] - term.div[i])
            log_s = log_s + (drift - 0.5 * v_plus) * dt + sqrt_v * d_w_s
            var = var + p.kappa * (p.theta - v_plus) * dt \
                + p.sigma * sqrt_v * d_w_v
            nodes[:, i + 1] = np.exp(log_s)
        return nodes
