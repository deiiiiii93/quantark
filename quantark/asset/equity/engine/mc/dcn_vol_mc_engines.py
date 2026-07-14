"""LV / Heston DCN MC engines (spec WP1.5): path generation overrides only.

Payoff, discounting, leg decomposition, and event stats are inherited from
DCNMCEngine; these classes replace ``_simulate`` with the same per-step loops
used by phoenix_vol_mc_engines.py. Heston exposes the historical DCN plain
full-truncation Euler update as an explicit scheme and also supports the shared
canonical log-Euler, QE, and QE-M path kernel.

``HestonSLVDCNMCEngine`` is an explicit stretch item (spec §4) and is NOT
implemented here.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from quantark.asset.equity.engine.mc.dcn_mc_engine import (
    DCNMCEngine,
    _BATCH_SEED_STRIDE,
)
from quantark.asset.equity.engine.mc.qmc_draws import qmc_uniforms
from quantark.param import GridVolSurface
from quantark.util.enum.engine_enums import HestonMCScheme
from quantark.util.exceptions import PricingError, ValidationError
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.heston.mc_kernel import simulate_heston_spot_nodes
from quantark.volmodels.localvol import LocalVolSurface, build_dupire_local_vol


_DCN_HESTON_SCHEMES = (
    HestonMCScheme.FULL_TRUNCATION_EULER,
    HestonMCScheme.EULERLOG,
    HestonMCScheme.QUADEXP,
    HestonMCScheme.QUADEXP_M,
)


def _resolve_dcn_heston_scheme(
    scheme: HestonMCScheme | str,
) -> HestonMCScheme:
    if isinstance(scheme, str):
        try:
            scheme = HestonMCScheme[scheme.upper()]
        except KeyError:
            valid = [item.name for item in _DCN_HESTON_SCHEMES]
            raise ValidationError(
                f"Invalid DCN Heston MC scheme '{scheme}'. Valid schemes: {valid}"
            )
    if not isinstance(scheme, HestonMCScheme):
        raise ValidationError("scheme must be a HestonMCScheme or string")
    if scheme not in _DCN_HESTON_SCHEMES:
        valid = [item.name for item in _DCN_HESTON_SCHEMES]
        raise ValidationError(
            f"HestonDCNMCEngine does not support {scheme.name}; "
            f"supported schemes: {valid}"
        )
    return scheme


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
    """DCN MC under selectable Heston path discretizations.

    ``substeps_per_interval`` refines variance/spot evolution between two
    contractual SSE observation dates while recording only the contractual
    nodes consumed by the payoff kernel. The default
    ``FULL_TRUNCATION_EULER`` preserves the historical DCN update and its
    two-stream seeded layout.

    Set ``fixed_three_stream_sobol=True`` on both an Euler and a QE engine to
    compare schemes with the same scrambled Sobol block
    ``[z_var | z_ind | u_var]``. Euler ignores the final uniform block.
    """

    def __init__(
        self,
        model_params: HestonParams,
        substeps_per_interval: int = 1,
        scheme: HestonMCScheme | str = HestonMCScheme.FULL_TRUNCATION_EULER,
        fixed_three_stream_sobol: bool = False,
        **kwargs,
    ):
        if not isinstance(model_params, HestonParams):
            raise ValidationError("model_params must be a HestonParams")
        if (
            isinstance(substeps_per_interval, bool)
            or not isinstance(substeps_per_interval, (int, np.integer))
            or int(substeps_per_interval) < 1
        ):
            raise ValidationError(
                "substeps_per_interval must be a positive integer"
            )
        if not isinstance(fixed_three_stream_sobol, (bool, np.bool_)):
            raise ValidationError("fixed_three_stream_sobol must be a bool")
        resolved_scheme = _resolve_dcn_heston_scheme(scheme)
        super().__init__(**kwargs)
        if fixed_three_stream_sobol and not self.use_sobol:
            raise ValidationError(
                "fixed_three_stream_sobol requires use_sobol=True"
            )
        self.model_params = model_params
        self.substeps_per_interval = int(substeps_per_interval)
        self.scheme = resolved_scheme
        self.fixed_three_stream_sobol = bool(fixed_three_stream_sobol)

    def _heston_draws(self, n_steps, n_paths, batch_id):
        """Return ``(z_var, z_ind, u_var)`` without extra full-block copies."""
        three_streams = self.scheme in (
            HestonMCScheme.QUADEXP,
            HestonMCScheme.QUADEXP_M,
        ) or self.fixed_three_stream_sobol

        if self.use_sobol and three_streams:
            from scipy.special import ndtri

            block = qmc_uniforms(
                self.seed, n_paths, 3 * n_steps, batch_id=batch_id
            )
            np.clip(block, 1e-12, 1.0 - 1e-12, out=block)
            # In-place inverse-CDF transform keeps peak batch memory bounded to
            # one 3-stream block plus simulation state and contractual nodes.
            ndtri(block[:, : 2 * n_steps], out=block[:, : 2 * n_steps])
            return (
                block[:, :n_steps],
                block[:, n_steps : 2 * n_steps],
                block[:, 2 * n_steps :],
            )

        if self.use_sobol or not three_streams:
            # The ordinary FTE/EULERLOG route deliberately retains the exact
            # historical 2-stream draw call and reshaping convention.
            z_all = self._draws(2 * n_steps, n_paths, batch_id)
            z_all = z_all.reshape(-1, 2, n_steps)
            return z_all[:, 0, :], z_all[:, 1, :], None

        seed = self.seed + (
            0 if batch_id is None else int(batch_id) * _BATCH_SEED_STRIDE
        )
        rng = np.random.default_rng(seed)
        if self.use_antithetic:
            half = (n_paths + 1) // 2
            z_half = rng.standard_normal((half, 2 * n_steps))
            z_all = np.vstack([z_half, -z_half])[:n_paths]
            u_half = rng.random((half, n_steps))
            u_var = np.vstack([u_half, 1.0 - u_half])[:n_paths]
        else:
            z_all = rng.standard_normal((n_paths, 2 * n_steps))
            u_var = rng.random((n_paths, n_steps))
        z_all = z_all.reshape(-1, 2, n_steps)
        return z_all[:, 0, :], z_all[:, 1, :], u_var

    def _simulate(self, spot0, term, dt_array, pricing_env,
                  n_paths, batch_id) -> np.ndarray:
        n_steps = dt_array.size
        substeps = self.substeps_per_interval
        n_fine = n_steps * substeps
        fine_dt = np.repeat(
            np.asarray(dt_array, dtype=float) / substeps, substeps
        )
        fine_r = np.repeat(np.asarray(term.rrf, dtype=float), substeps)
        fine_carry = np.repeat(np.asarray(term.div, dtype=float), substeps)
        z_var, z_ind, u_var = self._heston_draws(
            n_fine, n_paths, batch_id
        )
        record_steps = np.arange(0, n_fine + 1, substeps, dtype=int)
        return simulate_heston_spot_nodes(
            float(spot0),
            self.model_params,
            self.scheme,
            fine_dt,
            fine_r,
            fine_carry,
            z_var,
            z_ind,
            u_var,
            record_steps=record_steps,
        )


class QEDCNMCEngine(HestonDCNMCEngine):
    """Convenience DCN engine fixed to Heston QE or QE-M paths."""

    def __init__(
        self,
        model_params: HestonParams,
        martingale_correction: bool = False,
        **kwargs,
    ):
        if "scheme" in kwargs:
            raise ValidationError(
                "QEDCNMCEngine fixes scheme to QUADEXP/QUADEXP_M; "
                "use HestonDCNMCEngine for explicit scheme selection"
            )
        if not isinstance(martingale_correction, (bool, np.bool_)):
            raise ValidationError("martingale_correction must be a bool")
        scheme = (
            HestonMCScheme.QUADEXP_M
            if martingale_correction
            else HestonMCScheme.QUADEXP
        )
        super().__init__(model_params=model_params, scheme=scheme, **kwargs)
        self.martingale_correction = bool(martingale_correction)
