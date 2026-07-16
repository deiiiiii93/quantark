"""Phoenix Monte-Carlo engines under Local Vol / Heston / Heston-SLV processes."""

from __future__ import annotations

from typing import Optional

import numpy as np

from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
from quantark.asset.equity.engine.mc.snowball_vol_mc_engines import (
    _ArrayPathGenerator,
    _QE_KMIN,
    _QMC_METHODS,
    _VAR_FLOOR,
    _effective_path_count,
    _normal_draws,
    _qmc_normals,
    _qmc_uniforms,
    _resolve_heston_scheme,
)
from quantark.asset.equity.engine.mc.term_inputs import build_mc_term_inputs
from quantark.asset.equity.param import MCParams
from quantark.param import GridVolSurface
from quantark.priceenv import PricingEnvironment
from quantark.util.enum.engine_enums import (
    EngineType,
    HestonMCScheme,
    MonteCarloMethod,
)
from quantark.util.exceptions import NumericalError, PricingError, ValidationError
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.localvol import LocalVolSurface, build_dupire_local_vol
from quantark.volmodels.slv import BinMethod
from quantark.volmodels.slv.leverage import (
    DEFAULT_LEVERAGE_CLIP,
    LeverageSurface,
    bin_conditional,
    eval_binned,
)


class _VolModelPhoenixMCBase(PhoenixMCEngine):
    engine_type = EngineType.MONTE_CARLO

    def __init__(
        self,
        params: Optional[MCParams] = None,
        method: MonteCarloMethod | str | tuple | None = None,
        use_dask: bool = False,
        num_batches: int = 4,
    ):
        super().__init__(
            params=params,
            method=method,
            use_dask=use_dask,
            num_batches=num_batches,
        )

    def _uses_qmc(self) -> bool:
        return self.method in _QMC_METHODS

    def _validate_rng_controls(self, use_antithetic: bool) -> None:
        if self._uses_qmc() and use_antithetic:
            raise ValidationError(
                "QMC/RQMC and use_antithetic are mutually exclusive"
            )

    def _batch_seed(self, batch_id: Optional[int], seed: Optional[int] = None) -> int:
        if seed is not None:
            return int(seed)
        return int(self.params.seed) + (
            0 if batch_id is None else int(batch_id) * 1000
        )

    def _term_inputs(self, T: float, dt_array: np.ndarray):
        term_ctx = getattr(self, "_term_ctx", None)
        if term_ctx is None:
            raise PricingError("Pricing environment context was not initialized")
        env, ref_strike = term_ctx
        return build_mc_term_inputs(
            env,
            ref_strike=ref_strike,
            maturity=T,
            time_steps=len(dt_array),
            dt_array=dt_array,
        )


class LocalVolPhoenixMCEngine(_VolModelPhoenixMCBase):
    """Phoenix MC under a Dupire local-volatility surface."""

    def __init__(
        self,
        params: Optional[MCParams] = None,
        local_vol_surface: Optional[LocalVolSurface] = None,
        **kwargs,
    ):
        super().__init__(params=params, **kwargs)
        self._prebuilt = local_vol_surface

    def _build_surface(self, env: PricingEnvironment) -> LocalVolSurface:
        if self._prebuilt is not None:
            return self._prebuilt
        if not isinstance(env.vol_surface, GridVolSurface):
            raise PricingError(
                "LocalVolPhoenixMCEngine needs a GridVolSurface or a prebuilt LocalVolSurface"
            )
        return build_dupire_local_vol(
            env.vol_surface,
            spot=env.spot,
            rate_curve=env.rate_curve,
            div_yield=env.get_div_yield,
        )

    def _create_path_generator(
        self,
        S: float,
        r: float,
        q: float,
        sigma: float,
        T: float,
        dt_array: np.ndarray,
        batch_id: Optional[int] = None,
        num_paths: Optional[int] = None,
    ):
        term = self._term_inputs(T, dt_array)
        env, _ = self._term_ctx
        lv = self._build_surface(env)
        n_paths = int(self.params.num_paths if num_paths is None else num_paths)
        use_antithetic = bool(getattr(self.params, "use_antithetic", False))
        self._validate_rng_controls(use_antithetic)
        n_eff = (
            n_paths
            if self._uses_qmc()
            else _effective_path_count(n_paths, use_antithetic)
        )

        def simulate(batch_id=None, seed=None):
            rng = np.random.default_rng(self._batch_seed(batch_id, seed))
            z_all = (
                _qmc_normals(int(self.params.seed), n_eff, len(dt_array), batch_id)
                if self._uses_qmc()
                else None
            )
            nodes = np.empty((n_eff, len(dt_array) + 1), dtype=float)
            spot = np.full(n_eff, float(S), dtype=float)
            nodes[:, 0] = spot
            t = 0.0
            sqrt_dt = np.sqrt(np.asarray(dt_array, dtype=float))
            for i, dt in enumerate(dt_array):
                vol = np.asarray(lv.local_vol(spot, t), dtype=float)
                z = (
                    z_all[:, i]
                    if z_all is not None
                    else _normal_draws(rng, n_eff, use_antithetic)
                )
                drift = float(term.rrf[i] - term.div[i])
                spot = spot * np.exp(
                    (drift - 0.5 * vol * vol) * dt + vol * sqrt_dt[i] * z
                )
                nodes[:, i + 1] = spot
                t += float(dt)
            return nodes

        return _ArrayPathGenerator(simulate, n_eff, batch_id=batch_id)


class HestonPhoenixMCEngine(_VolModelPhoenixMCBase):
    """Phoenix MC under the Heston stochastic-volatility model."""

    def _rqmc_streams_per_step(self) -> int:
        from quantark.util.enum.engine_enums import HestonMCScheme

        if self.scheme in (HestonMCScheme.QUADEXP, HestonMCScheme.QUADEXP_M):
            return 3  # [z_var | z_ind | u_var] uniform block
        return 2      # [z_var | z_ind]

    def __init__(
        self,
        model_params: HestonParams,
        params: Optional[MCParams] = None,
        scheme: HestonMCScheme | str = HestonMCScheme.QUADEXP,
        **kwargs,
    ):
        if not isinstance(model_params, HestonParams):
            raise ValidationError("model_params must be a HestonParams")
        super().__init__(params=params, **kwargs)
        self.model_params = model_params
        self.scheme = _resolve_heston_scheme(scheme)

    def _create_path_generator(
        self,
        S: float,
        r: float,
        q: float,
        sigma: float,
        T: float,
        dt_array: np.ndarray,
        batch_id: Optional[int] = None,
        num_paths: Optional[int] = None,
    ):
        term = self._term_inputs(T, dt_array)
        n_paths = int(self.params.num_paths if num_paths is None else num_paths)
        use_antithetic = bool(getattr(self.params, "use_antithetic", False))
        self._validate_rng_controls(use_antithetic)
        n_eff = (
            n_paths
            if self._uses_qmc()
            else _effective_path_count(n_paths, use_antithetic)
        )
        p = self.model_params
        scheme = self.scheme
        m_steps = len(dt_array)

        def _draws(batch_id=None, seed=None):
            if self._uses_qmc():
                from scipy.special import ndtri

                if scheme in (HestonMCScheme.QUADEXP, HestonMCScheme.QUADEXP_M):
                    block = np.clip(
                        _qmc_uniforms(
                            int(self.params.seed), n_eff, 3 * m_steps, batch_id
                        ),
                        1e-12,
                        1.0 - 1e-12,
                    )
                    return (
                        ndtri(block[:, 0:m_steps]),
                        ndtri(block[:, m_steps : 2 * m_steps]),
                        block[:, 2 * m_steps : 3 * m_steps],
                    )

                block = np.clip(
                    _qmc_uniforms(
                        int(self.params.seed), n_eff, 2 * m_steps, batch_id
                    ),
                    1e-12,
                    1.0 - 1e-12,
                )
                return (
                    ndtri(block[:, 0:m_steps]),
                    ndtri(block[:, m_steps : 2 * m_steps]),
                    None,
                )

            rng = np.random.default_rng(self._batch_seed(batch_id, seed))
            if use_antithetic:
                half = (n_paths + 1) // 2
                z_var_h = rng.standard_normal((half, m_steps))
                z_ind_h = rng.standard_normal((half, m_steps))
                z_var = np.concatenate([z_var_h, -z_var_h], axis=0)
                z_ind = np.concatenate([z_ind_h, -z_ind_h], axis=0)
                if scheme in (HestonMCScheme.QUADEXP, HestonMCScheme.QUADEXP_M):
                    u_var_h = rng.random((half, m_steps))
                    u_var = np.concatenate([u_var_h, 1.0 - u_var_h], axis=0)
                else:
                    u_var = None
            else:
                z_var = rng.standard_normal((n_eff, m_steps))
                z_ind = rng.standard_normal((n_eff, m_steps))
                u_var = (
                    rng.random((n_eff, m_steps))
                    if scheme in (HestonMCScheme.QUADEXP, HestonMCScheme.QUADEXP_M)
                    else None
                )
            return z_var, z_ind, u_var

        def simulate(batch_id=None, seed=None):
            z_var, z_ind, u_var = _draws(batch_id=batch_id, seed=seed)
            nodes = np.empty((n_eff, len(dt_array) + 1), dtype=float)
            log_s = np.full(n_eff, np.log(max(float(S), 1e-12)), dtype=float)
            var = np.full(n_eff, max(float(p.v0), 0.0), dtype=float)
            nodes[:, 0] = np.exp(log_s)
            rho = float(np.clip(p.rho, -0.999, 0.999))
            rho_bar = float(np.sqrt(max(1.0 - rho * rho, 0.0)))
            sigma2 = float(p.sigma * p.sigma)
            if scheme not in (
                HestonMCScheme.EULERLOG,
                HestonMCScheme.QUADEXP,
                HestonMCScheme.QUADEXP_M,
            ):
                raise ValidationError(
                    "HestonPhoenixMCEngine supports EULERLOG, QUADEXP, and QUADEXP_M"
                )
            if scheme == HestonMCScheme.EULERLOG:
                for i, dt in enumerate(dt_array):
                    sqrt_dt = float(np.sqrt(dt))
                    z1 = z_var[:, i] * sqrt_dt
                    z2 = z_ind[:, i] * sqrt_dt
                    z_s = rho * z1 + rho_bar * z2
                    v_plus = np.maximum(var, 0.0)
                    sqrt_v = np.sqrt(v_plus)
                    drift = float(term.rrf[i] - term.div[i])
                    log_s = log_s + (drift - 0.5 * v_plus) * dt + sqrt_v * z_s
                    var = (
                        var
                        + p.kappa * (p.theta - v_plus) * dt
                        + p.sigma * sqrt_v * z1
                        + 0.25 * sigma2 * (z1 * z1 - dt)
                    )
                    nodes[:, i + 1] = np.exp(log_s)
                return nodes

            martingale = scheme == HestonMCScheme.QUADEXP_M
            psi_c = 1.5
            deterministic_vol = p.sigma <= 1e-8
            diff_coef = 1.0 if deterministic_vol else rho_bar
            for i, dt in enumerate(dt_array):
                sqrt_dt = float(np.sqrt(dt))
                drift = float(term.rrf[i] - term.div[i])
                exp_kdt = np.exp(-p.kappa * dt)
                omexp = -np.expm1(-p.kappa * dt)
                m = p.theta + (var - p.theta) * exp_kdt
                if p.kappa > _QE_KMIN:
                    inv_k = 1.0 / p.kappa
                    s2 = (
                        var * sigma2 * exp_kdt * (omexp * inv_k)
                        + p.theta * sigma2 * (omexp * omexp * inv_k) / 2.0
                    )
                else:
                    s2 = var * sigma2 * dt
                with np.errstate(divide="ignore", invalid="ignore"):
                    psi = np.where(m <= 1e-12, 0.0, s2 / (m * m))
                psi = np.maximum(psi, 0.0)

                phi = 2.0 / np.maximum(psi, 1e-16)
                rad = np.maximum(phi * (phi - 1.0), 0.0)
                b = np.sqrt(np.maximum(phi - 1.0 + np.sqrt(rad), 0.0))
                a = m / (1.0 + b * b)
                v_a = a * (b + z_var[:, i]) * (b + z_var[:, i])

                prob_zero = np.clip((psi - 1.0) / (psi + 1.0), 0.0, 0.999999)
                beta = np.maximum(
                    (1.0 - prob_zero) / np.maximum(m, _QE_KMIN), _QE_KMIN
                )
                u_clip = np.clip(u_var[:, i], 1e-12, 1.0 - 1e-12)
                with np.errstate(divide="ignore", invalid="ignore"):
                    v_b = np.where(
                        u_clip <= prob_zero,
                        0.0,
                        np.log((1.0 - prob_zero) / (1.0 - u_clip)) / beta,
                    )

                v_np = np.maximum(np.where(psi <= psi_c, v_a, v_b), 0.0)
                v_bar = np.maximum(0.5 * (v_np + np.maximum(var, 0.0)), 0.0)
                if deterministic_vol:
                    corr = 0.0
                else:
                    corr = (rho / p.sigma) * (
                        v_np - var - p.kappa * (p.theta - v_bar) * dt
                    )
                if martingale and not deterministic_vol:
                    ros = rho / p.sigma
                    k3 = 0.5 * (1.0 - rho * rho) * dt
                    k1 = 0.5 * dt * (p.kappa * ros - 0.5) - ros
                    k2 = 0.5 * dt * (p.kappa * ros - 0.5) + ros
                    a_mgf = k2 + 0.5 * k3
                    quad_mask = psi <= psi_c
                    denom_q = 1.0 - 2.0 * a_mgf * a
                    denom_e = beta - a_mgf
                    bad = (quad_mask & (denom_q <= 0.0)) | (
                        ~quad_mask & (denom_e <= 0.0)
                    )
                    if np.any(bad):
                        raise NumericalError(
                            "QE-M martingale MGF is undefined at these parameters "
                            "(A outside the CIR-transition MGF domain); tighten dt or use QUADEXP"
                        )
                    safe_q = np.where(denom_q > 0.0, denom_q, 1.0)
                    safe_e = np.where(denom_e > 0.0, denom_e, 1.0)
                    m_quad = np.exp(a_mgf * a * b * b / safe_q) / np.sqrt(safe_q)
                    m_exp = prob_zero + (1.0 - prob_zero) * beta / safe_e
                    ln_m = np.log(np.where(quad_mask, m_quad, m_exp))
                    k0 = -ros * p.kappa * p.theta * dt
                    k0_star = -ln_m - (k1 + 0.5 * k3) * var
                    log_s = (
                        log_s
                        + (drift - 0.5 * v_bar) * dt
                        + corr
                        - k0
                        + k0_star
                        + np.sqrt(v_bar) * sqrt_dt * diff_coef * z_ind[:, i]
                    )
                else:
                    log_s = (
                        log_s
                        + (drift - 0.5 * v_bar) * dt
                        + corr
                        + np.sqrt(v_bar) * sqrt_dt * diff_coef * z_ind[:, i]
                    )
                var = v_np
                nodes[:, i + 1] = np.exp(log_s)
            return nodes

        return _ArrayPathGenerator(simulate, n_eff, batch_id=batch_id)


class QEPhoenixMCEngine(HestonPhoenixMCEngine):
    """Standalone Phoenix MC engine using the Heston QE variance scheme."""

    def __init__(
        self,
        model_params: HestonParams,
        params: Optional[MCParams] = None,
        martingale_correction: bool = False,
        **kwargs,
    ):
        if "scheme" in kwargs:
            raise ValidationError(
                "QEPhoenixMCEngine fixes the scheme to QUADEXP/QUADEXP_M; "
                "use HestonPhoenixMCEngine for explicit scheme selection"
            )
        scheme = (
            HestonMCScheme.QUADEXP_M
            if bool(martingale_correction)
            else HestonMCScheme.QUADEXP
        )
        super().__init__(
            model_params=model_params,
            params=params,
            scheme=scheme,
            **kwargs,
        )
        self.martingale_correction = bool(martingale_correction)


class HestonSLVPhoenixMCEngine(_VolModelPhoenixMCBase):
    """Phoenix MC under Heston-SLV using a precomputed or on-the-fly leverage surface."""

    def _rqmc_streams_per_step(self) -> int:
        return 2  # correlated spot/variance normal pair

    def __init__(
        self,
        model_params: HestonParams,
        params: Optional[MCParams] = None,
        local_vol_surface: Optional[LocalVolSurface] = None,
        leverage_surface: Optional[LeverageSurface] = None,
        eta: float = 1.0,
        num_bins: int = 20,
        bin_method: BinMethod = BinMethod.EQUAL_WEIGHTED,
        leverage_clip=DEFAULT_LEVERAGE_CLIP,
        **kwargs,
    ):
        if not isinstance(model_params, HestonParams):
            raise ValidationError("model_params must be a HestonParams")
        if leverage_surface is not None and not isinstance(
            leverage_surface, LeverageSurface
        ):
            raise ValidationError("leverage_surface must be a LeverageSurface")
        if eta < 0:
            raise ValidationError("eta must be non-negative")
        super().__init__(params=params, **kwargs)
        self.model_params = model_params
        self._prebuilt = local_vol_surface
        self.leverage_surface = leverage_surface
        self.eta = float(eta)
        self.num_bins = int(num_bins)
        self.bin_method = bin_method
        self.leverage_clip = leverage_clip

    def _build_surface(self, env: PricingEnvironment) -> LocalVolSurface:
        if self._prebuilt is not None:
            return self._prebuilt
        if not isinstance(env.vol_surface, GridVolSurface):
            raise PricingError(
                "HestonSLVPhoenixMCEngine needs a GridVolSurface or a prebuilt LocalVolSurface"
            )
        return build_dupire_local_vol(
            env.vol_surface,
            spot=env.spot,
            rate_curve=env.rate_curve,
            div_yield=env.get_div_yield,
        )

    def _create_path_generator(
        self,
        S: float,
        r: float,
        q: float,
        sigma: float,
        T: float,
        dt_array: np.ndarray,
        batch_id: Optional[int] = None,
        num_paths: Optional[int] = None,
    ):
        term = self._term_inputs(T, dt_array)
        env, _ = self._term_ctx
        lv = self._build_surface(env)
        n_paths = int(self.params.num_paths if num_paths is None else num_paths)
        use_antithetic = bool(getattr(self.params, "use_antithetic", False))
        self._validate_rng_controls(use_antithetic)
        n_eff = (
            n_paths
            if self._uses_qmc()
            else _effective_path_count(n_paths, use_antithetic)
        )
        p = self.model_params
        lo, hi = self.leverage_clip

        def simulate(batch_id=None, seed=None):
            rng = np.random.default_rng(self._batch_seed(batch_id, seed))
            qmc_z = None
            if self._uses_qmc():
                qmc_z = _qmc_normals(
                    int(self.params.seed), n_eff, 2 * len(dt_array), batch_id
                )
                qmc_z = qmc_z.reshape(n_eff, 2, len(dt_array))
            nodes = np.empty((n_eff, len(dt_array) + 1), dtype=float)
            log_s = np.full(n_eff, np.log(max(float(S), 1e-12)), dtype=float)
            var = np.full(n_eff, max(float(p.v0), 0.0), dtype=float)
            nodes[:, 0] = np.exp(log_s)
            rho = float(np.clip(p.rho, -0.999, 0.999))
            rho_bar = float(np.sqrt(max(1.0 - rho * rho, 0.0)))
            sigma_eff = float(self.eta * p.sigma)
            t = 0.0
            for i, dt in enumerate(dt_array):
                spot = np.exp(log_s)
                if self.leverage_surface is None:
                    boundaries, means = bin_conditional(
                        spot, var, self.num_bins, self.bin_method
                    )
                    econd = np.maximum(eval_binned(spot, boundaries, means), _VAR_FLOOR)
                    sigma_lv = np.asarray(lv.local_vol(spot, t), dtype=float)
                    leverage = np.clip(sigma_lv / np.sqrt(econd), lo, hi)
                else:
                    leverage = np.asarray(
                        self.leverage_surface.leverage(spot, t), dtype=float
                    )
                    if not np.all(np.isfinite(leverage)):
                        raise ValidationError(
                            "precomputed leverage returned non-finite values"
                        )
                v_plus = np.maximum(var, 0.0)
                sqrt_v = np.sqrt(v_plus)
                eff_vol = leverage * sqrt_v
                sqrt_dt = float(np.sqrt(dt))
                if qmc_z is not None:
                    d_w_v = qmc_z[:, 0, i] * sqrt_dt
                    d_w_i = qmc_z[:, 1, i] * sqrt_dt
                else:
                    d_w_v = _normal_draws(rng, n_eff, use_antithetic) * sqrt_dt
                    d_w_i = _normal_draws(rng, n_eff, use_antithetic) * sqrt_dt
                d_w_s = rho * d_w_v + rho_bar * d_w_i
                drift = float(term.rrf[i] - term.div[i])
                log_s = np.maximum(
                    log_s + (drift - 0.5 * eff_vol * eff_vol) * dt + eff_vol * d_w_s,
                    np.log(1e-12),
                )
                var = (
                    var
                    + p.kappa * (p.theta - v_plus) * dt
                    + sigma_eff * sqrt_v * d_w_v
                )
                nodes[:, i + 1] = np.exp(log_s)
                t += float(dt)
            return nodes

        return _ArrayPathGenerator(simulate, n_eff, batch_id=batch_id)


class HestonSLVQEPhoenixMCEngine(HestonSLVPhoenixMCEngine):
    """Standalone Phoenix MC under Heston-SLV with frozen-leverage QE variance."""

    def _rqmc_streams_per_step(self) -> int:
        return 3  # [z_var | z_ind | u_var] uniform block

    def __init__(
        self,
        model_params: HestonParams,
        params: Optional[MCParams] = None,
        martingale_correction: bool = False,
        **kwargs,
    ):
        super().__init__(model_params=model_params, params=params, **kwargs)
        self.martingale_correction = bool(martingale_correction)

    def _create_path_generator(
        self,
        S: float,
        r: float,
        q: float,
        sigma: float,
        T: float,
        dt_array: np.ndarray,
        batch_id: Optional[int] = None,
        num_paths: Optional[int] = None,
    ):
        term = self._term_inputs(T, dt_array)
        env, _ = self._term_ctx
        lv = self._build_surface(env)
        n_paths = int(self.params.num_paths if num_paths is None else num_paths)
        use_antithetic = bool(getattr(self.params, "use_antithetic", False))
        self._validate_rng_controls(use_antithetic)
        n_eff = (
            n_paths
            if self._uses_qmc()
            else _effective_path_count(n_paths, use_antithetic)
        )
        p = self.model_params
        lo, hi = self.leverage_clip
        m_steps = len(dt_array)

        def _draws(batch_id=None, seed=None):
            if self._uses_qmc():
                from scipy.special import ndtri

                block = np.clip(
                    _qmc_uniforms(int(self.params.seed), n_eff, 3 * m_steps, batch_id),
                    1e-12,
                    1.0 - 1e-12,
                )
                return (
                    ndtri(block[:, 0:m_steps]),
                    ndtri(block[:, m_steps : 2 * m_steps]),
                    block[:, 2 * m_steps : 3 * m_steps],
                )

            rng = np.random.default_rng(self._batch_seed(batch_id, seed))
            if use_antithetic:
                half = (n_paths + 1) // 2
                z_var_h = rng.standard_normal((half, m_steps))
                z_ind_h = rng.standard_normal((half, m_steps))
                u_var_h = rng.random((half, m_steps))
                return (
                    np.concatenate([z_var_h, -z_var_h], axis=0),
                    np.concatenate([z_ind_h, -z_ind_h], axis=0),
                    np.concatenate([u_var_h, 1.0 - u_var_h], axis=0),
                )
            return (
                rng.standard_normal((n_eff, m_steps)),
                rng.standard_normal((n_eff, m_steps)),
                rng.random((n_eff, m_steps)),
            )

        def simulate(batch_id=None, seed=None):
            z_var, z_ind, u_var = _draws(batch_id=batch_id, seed=seed)
            nodes = np.empty((n_eff, len(dt_array) + 1), dtype=float)
            log_s = np.full(n_eff, np.log(max(float(S), 1e-12)), dtype=float)
            var = np.full(n_eff, max(float(p.v0), 0.0), dtype=float)
            nodes[:, 0] = np.exp(log_s)
            rho = float(np.clip(p.rho, -0.999, 0.999))
            rho_bar = float(np.sqrt(max(1.0 - rho * rho, 0.0)))
            sigma_eff = float(self.eta * p.sigma)
            sigma_eff2 = sigma_eff * sigma_eff
            deterministic_vol = sigma_eff <= 1e-8
            diff_coef = 1.0 if deterministic_vol else rho_bar
            martingale = self.martingale_correction and not deterministic_vol
            psi_c = 1.5
            t = 0.0

            for i, dt in enumerate(dt_array):
                spot = np.exp(log_s)
                if self.leverage_surface is None:
                    boundaries, means = bin_conditional(
                        spot, var, self.num_bins, self.bin_method
                    )
                    econd = np.maximum(eval_binned(spot, boundaries, means), _VAR_FLOOR)
                    sigma_lv = np.asarray(lv.local_vol(spot, t), dtype=float)
                    leverage = np.clip(sigma_lv / np.sqrt(econd), lo, hi)
                else:
                    leverage = np.asarray(
                        self.leverage_surface.leverage(spot, t), dtype=float
                    )
                    if not np.all(np.isfinite(leverage)):
                        raise ValidationError(
                            "precomputed leverage returned non-finite values"
                        )

                drift = float(term.rrf[i] - term.div[i])
                sqrt_dt = float(np.sqrt(dt))
                exp_kdt = np.exp(-p.kappa * dt)
                omexp = -np.expm1(-p.kappa * dt)
                m = p.theta + (var - p.theta) * exp_kdt
                if p.kappa > _QE_KMIN:
                    inv_k = 1.0 / p.kappa
                    s2 = (
                        var * sigma_eff2 * exp_kdt * (omexp * inv_k)
                        + p.theta * sigma_eff2 * (omexp * omexp * inv_k) / 2.0
                    )
                else:
                    s2 = var * sigma_eff2 * dt
                with np.errstate(divide="ignore", invalid="ignore"):
                    psi = np.where(m <= 1e-12, 0.0, s2 / (m * m))
                psi = np.maximum(psi, 0.0)

                phi = 2.0 / np.maximum(psi, 1e-16)
                rad = np.maximum(phi * (phi - 1.0), 0.0)
                b = np.sqrt(np.maximum(phi - 1.0 + np.sqrt(rad), 0.0))
                a = m / (1.0 + b * b)
                v_a = a * (b + z_var[:, i]) * (b + z_var[:, i])

                prob_zero = np.clip((psi - 1.0) / (psi + 1.0), 0.0, 0.999999)
                beta = np.maximum(
                    (1.0 - prob_zero) / np.maximum(m, _QE_KMIN), _QE_KMIN
                )
                u_clip = np.clip(u_var[:, i], 1e-12, 1.0 - 1e-12)
                with np.errstate(divide="ignore", invalid="ignore"):
                    v_b = np.where(
                        u_clip <= prob_zero,
                        0.0,
                        np.log((1.0 - prob_zero) / (1.0 - u_clip)) / beta,
                    )

                v_np = np.maximum(np.where(psi <= psi_c, v_a, v_b), 0.0)
                v_bar = np.maximum(0.5 * (v_np + np.maximum(var, 0.0)), 0.0)
                leverage2 = leverage * leverage
                if deterministic_vol:
                    corr = 0.0
                else:
                    corr = leverage * (rho / sigma_eff) * (
                        v_np - var - p.kappa * (p.theta - v_bar) * dt
                    )

                if martingale:
                    ros = leverage * rho / sigma_eff
                    k3 = 0.5 * leverage2 * (1.0 - rho * rho) * dt
                    k1 = 0.5 * dt * (p.kappa * ros - 0.5 * leverage2) - ros
                    k2 = 0.5 * dt * (p.kappa * ros - 0.5 * leverage2) + ros
                    a_mgf = k2 + 0.5 * k3
                    quad_mask = psi <= psi_c
                    denom_q = 1.0 - 2.0 * a_mgf * a
                    denom_e = beta - a_mgf
                    bad = (quad_mask & (denom_q <= 0.0)) | (
                        ~quad_mask & (denom_e <= 0.0)
                    )
                    if np.any(bad):
                        raise NumericalError(
                            "SLV QE-M martingale MGF is undefined at these parameters "
                            "(A outside the QE transition MGF domain); tighten dt or disable "
                            "martingale_correction"
                        )
                    safe_q = np.where(denom_q > 0.0, denom_q, 1.0)
                    safe_e = np.where(denom_e > 0.0, denom_e, 1.0)
                    m_quad = np.exp(a_mgf * a * b * b / safe_q) / np.sqrt(safe_q)
                    m_exp = prob_zero + (1.0 - prob_zero) * beta / safe_e
                    ln_m = np.log(np.where(quad_mask, m_quad, m_exp))
                    k0 = -ros * p.kappa * p.theta * dt
                    k0_star = -ln_m - (k1 + 0.5 * k3) * var
                    log_s = (
                        log_s
                        + (drift - 0.5 * leverage2 * v_bar) * dt
                        + corr
                        - k0
                        + k0_star
                        + leverage
                        * np.sqrt(v_bar)
                        * sqrt_dt
                        * diff_coef
                        * z_ind[:, i]
                    )
                else:
                    log_s = (
                        log_s
                        + (drift - 0.5 * leverage2 * v_bar) * dt
                        + corr
                        + leverage
                        * np.sqrt(v_bar)
                        * sqrt_dt
                        * diff_coef
                        * z_ind[:, i]
                    )
                var = v_np
                nodes[:, i + 1] = np.exp(log_s)
                t += float(dt)
            return nodes

        return _ArrayPathGenerator(simulate, n_eff, batch_id=batch_id)
