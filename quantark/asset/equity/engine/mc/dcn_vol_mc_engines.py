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
        self._active_surface: Optional[LocalVolSurface] = None
        self._active_surface_env = None

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

    def _prepare_simulation(self, product, pricing_env) -> None:
        """Invert Dupire ONCE per price call, not once per batch.

        The environment is constant across the batch loop, so the rebuild is
        env-invariant per call; bumped repricings still re-invert because
        every ``price_detailed`` prepares afresh. The env identity check
        below makes a stale surface impossible for direct ``_simulate``
        callers.
        """
        self._active_surface = self._build_surface(pricing_env)
        self._active_surface_env = pricing_env

    def _resolve_surface(self, pricing_env) -> LocalVolSurface:
        if (
            self._active_surface is not None
            and self._active_surface_env is pricing_env
        ):
            return self._active_surface
        return self._build_surface(pricing_env)

    def _simulate(self, spot0, term, dt_array, pricing_env,
                  n_paths, batch_id) -> np.ndarray:
        lv = self._resolve_surface(pricing_env)
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

            # writable=True: this path transforms the block in place, so it
            # needs a private copy rather than the shared read-only cache
            # entry (the Sobol generation itself still caches).
            block = qmc_uniforms(
                self.seed, n_paths, 3 * n_steps, batch_id=batch_id,
                writable=True,
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


class CoupledCoarseHestonDCNMCEngine(HestonDCNMCEngine):
    """Coarse half of a coupled Heston timestep-ladder pair.

    Instead of drawing its own randomness, this engine derives every draw
    from the paired fine engine's (deterministic, seed/batch-keyed) block:

    - standardized Gaussian streams are pair-aggregated,
      ``(z[2i] + z[2i+1]) / sqrt(2)`` — the exact Brownian-increment sum, so
      the coarse asset/variance noise is the fine Brownian path observed on
      the coarse grid;
    - QE branch uniforms take the first draw of each fine pair, which keeps
      an exact uniform marginal and a strongly correlated branch selection.

    Marginally the coarse simulation therefore remains a valid randomized
    estimator (every derived draw has the exact target law under Sobol
    scrambling or pseudorandom seeding); jointly it is highly correlated
    with the fine simulation, so coarse-minus-fine differences computed
    batch-by-batch on the same seed form a low-variance paired ladder —
    the multilevel Monte Carlo coupling — instead of the independent-
    scramble design whose difference variance is the sum of two marginal
    variances. The achieved reduction is an empirical output, not an
    assumption — and it inverts naive intuition in Feller-violating
    regimes: FTE's exact Brownian aggregation is undermined by v=0
    truncation events flipping between resolutions (measured ~1.4x variance
    reduction on the DCN fixtures), while QE-M's near-exact transition
    sampling keeps resolutions aligned (~4x). Gaussian-aggregating the
    branch uniforms was measured strictly worse than first-of-pair and is
    deliberately not used.

    Build pairs with :func:`coupled_heston_ladder_pair` so seed, scheme and
    stream-layout consistency are enforced.
    """

    def __init__(self, fine_engine: HestonDCNMCEngine, **kwargs):
        if not isinstance(fine_engine, HestonDCNMCEngine):
            raise ValidationError(
                "fine_engine must be a HestonDCNMCEngine"
            )
        super().__init__(**kwargs)
        if fine_engine.substeps_per_interval != 2 * self.substeps_per_interval:
            raise ValidationError(
                "fine engine must use exactly 2x the coarse substeps "
                f"(coarse={self.substeps_per_interval}, "
                f"fine={fine_engine.substeps_per_interval})"
            )
        for attr in (
            "seed", "scheme", "use_sobol", "use_antithetic",
            "fixed_three_stream_sobol",
        ):
            if getattr(fine_engine, attr) != getattr(self, attr):
                raise ValidationError(
                    f"coupled pair must agree on {attr}: "
                    f"fine={getattr(fine_engine, attr)!r} vs "
                    f"coarse={getattr(self, attr)!r}"
                )
        if fine_engine.model_params is not self.model_params:
            raise ValidationError(
                "coupled pair must share the same HestonParams instance"
            )
        self._fine_engine = fine_engine

    def _heston_draws(self, n_steps, n_paths, batch_id):
        z_var_fine, z_ind_fine, u_var_fine = self._fine_engine._heston_draws(
            2 * n_steps, n_paths, batch_id
        )
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        z_var = (z_var_fine[:, 0::2] + z_var_fine[:, 1::2]) * inv_sqrt2
        z_ind = (z_ind_fine[:, 0::2] + z_ind_fine[:, 1::2]) * inv_sqrt2
        u_var = (
            None if u_var_fine is None
            else np.ascontiguousarray(u_var_fine[:, 0::2])
        )
        return z_var, z_ind, u_var


def coupled_heston_ladder_pair(
    model_params: HestonParams,
    coarse_substeps_per_interval: int,
    scheme: HestonMCScheme | str,
    **engine_kwargs,
) -> tuple[CoupledCoarseHestonDCNMCEngine, HestonDCNMCEngine]:
    """Build a (coarse, fine) coupled timestep-ladder engine pair.

    Both engines share seed, scheme and every other engine keyword; the fine
    engine uses ``2 * coarse_substeps_per_interval`` substeps and draws its
    own randomness, while the coarse engine derives its draws from the fine
    block for the same seed/batch. Price both with the same product and
    environment (one seed per RQMC batch, ``num_batches=1``) and difference
    the batch metrics to obtain the low-variance coupled ladder.
    """
    fine = HestonDCNMCEngine(
        model_params=model_params,
        substeps_per_interval=2 * int(coarse_substeps_per_interval),
        scheme=scheme,
        **engine_kwargs,
    )
    coarse = CoupledCoarseHestonDCNMCEngine(
        fine_engine=fine,
        model_params=model_params,
        substeps_per_interval=int(coarse_substeps_per_interval),
        scheme=scheme,
        **engine_kwargs,
    )
    return coarse, fine


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
