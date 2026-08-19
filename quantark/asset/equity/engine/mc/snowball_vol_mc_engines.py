"""Snowball Monte-Carlo engines under Local Vol / Heston / Heston-SLV processes."""

from __future__ import annotations

from typing import Optional

import numpy as np

from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.mc.term_inputs import build_mc_term_inputs
from quantark.asset.equity.param import MCParams
from quantark.param import GridVolSurface
from quantark.priceenv import PricingEnvironment
from quantark.montecarlo.qmc_sobol import SobolNormalGenerator
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

_VAR_FLOOR = 1e-8
_QE_KMIN = 1e-12
_QMC_METHODS = (MonteCarloMethod.QUASI, MonteCarloMethod.RANDOMIZED_QUASI)


def _validate_substeps_per_interval(substeps) -> int:
    """Validate the sub-observation refinement factor (mirrors HestonDCNMCEngine)."""
    if (
        isinstance(substeps, bool)
        or not isinstance(substeps, (int, np.integer))
        or int(substeps) < 1
    ):
        raise ValidationError(
            "substeps_per_interval must be a positive integer"
        )
    return int(substeps)


def _effective_path_count(num_paths: int, use_antithetic: bool) -> int:
    if num_paths <= 0:
        raise ValidationError(f"num_paths must be positive, got {num_paths}")
    if not use_antithetic:
        return int(num_paths)
    return 2 * ((int(num_paths) + 1) // 2)


def _normal_draws(rng, n_paths: int, use_antithetic: bool) -> np.ndarray:
    if not use_antithetic:
        return rng.standard_normal(n_paths)
    half = (n_paths + 1) // 2
    z = rng.standard_normal(half)
    return np.concatenate([z, -z])[:n_paths]


def _resolve_heston_scheme(scheme: HestonMCScheme | str) -> HestonMCScheme:
    if isinstance(scheme, HestonMCScheme):
        return scheme
    if isinstance(scheme, str):
        try:
            return HestonMCScheme[scheme.upper()]
        except KeyError:
            valid = [s.name for s in HestonMCScheme]
            raise ValidationError(
                f"Invalid Heston MC scheme '{scheme}'. Valid schemes: {valid}"
            )
    raise ValidationError(
        f"scheme must be HestonMCScheme or str, got {type(scheme).__name__}"
    )


def _qmc_normals(seed: int, n_paths: int, dim: int, batch_id: Optional[int]):
    return SobolNormalGenerator(base_seed=int(seed)).normal(
        n_paths, dim, batch_id=batch_id
    )


def _qmc_uniforms(seed: int, n_paths: int, dim: int, batch_id: Optional[int]):
    return SobolNormalGenerator(base_seed=int(seed)).uniform(
        n_paths, dim, batch_id=batch_id
    )


class _ArrayPathGenerator:
    """Small adapter matching ``GBMPathGenerator.generate_paths`` for SnowballMCEngine."""

    def __init__(self, simulator, num_paths: int, batch_id: Optional[int] = None):
        self._simulator = simulator
        self.num_paths = int(num_paths)
        self._batch_id = batch_id

    def generate_paths(self, seed=None, batch_id=None, return_aux: bool = False):
        effective_batch = self._batch_id if batch_id is None else batch_id
        paths = self._simulator(batch_id=effective_batch, seed=seed)
        aux = {"batch_id": 0 if effective_batch is None else int(effective_batch)}
        return paths, aux if return_aux else None


class _SubstepRefinementMixin:
    """Opt-in sub-observation SDE refinement for schedule-based vol MC engines.

    ``substeps_per_interval=n`` refines every contractual interval into n
    equal SDE steps while the recorded path nodes (and therefore payoff
    kernels, event stats, and RQMC batching) stay on the contractual grid —
    the same contract as ``HestonDCNMCEngine.substeps_per_interval``. The
    default factor 1 is bitwise-identical to unrefined stepping.
    """

    substeps_per_interval: int = 1

    def _refined_dt_array(self, dt_array: np.ndarray) -> np.ndarray:
        """Refine every contractual interval into ``substeps_per_interval``
        equal SDE steps (identity, bitwise, at the default factor 1)."""
        n = self.substeps_per_interval
        if n <= 1:
            return dt_array
        return np.repeat(np.asarray(dt_array, dtype=float) / n, n)

    def _make_path_generator(self, simulate, n_eff: int, batch_id):
        """Wrap ``simulate`` so recorded nodes stay contractual: with
        refinement active, every n-th fine column is a contractual node."""
        n = self.substeps_per_interval
        if n <= 1:
            return _ArrayPathGenerator(simulate, n_eff, batch_id=batch_id)

        def simulate_contractual(batch_id=None, seed=None):
            return simulate(batch_id=batch_id, seed=seed)[:, ::n]

        return _ArrayPathGenerator(simulate_contractual, n_eff, batch_id=batch_id)

    def _rqmc_substep_factor(self) -> int:
        return self.substeps_per_interval

    # --- continuous-KI bridge variance -------------------------------------
    #
    # The bridge asks how much log-variance a path accumulated between two
    # RECORDED nodes. These engines know it exactly -- it is the quantity each
    # scheme already multiplies into its own log-spot increment -- so they
    # record it per fine SDE step and fold it onto the contractual grid, the
    # same grid `_make_path_generator` records the nodes on.

    def _new_step_log_variance(self, n_paths: int, n_fine_steps: int):
        """Buffer for the per-fine-step log-variance, or None when no
        continuous-KI bridge will run (it costs a second paths-sized array)."""
        if not getattr(self, "_ki_bridge_wanted", False):
            return None
        return np.empty((int(n_paths), int(n_fine_steps)), dtype=float)

    def _record_step_log_variance(self, fine) -> None:
        """Fold the fine SDE steps onto the contractual grid and keep it.

        Variance is additive over sub-intervals, so a contractual interval's
        log-variance is the SUM of its substeps'.
        """
        if fine is None:
            self._step_log_variance = None
            return
        n = self.substeps_per_interval
        if n > 1:
            fine = fine.reshape(fine.shape[0], -1, n).sum(axis=2)
        self._step_log_variance = fine

    def _ki_bridge_step_log_variance(self, paths):
        recorded = getattr(self, "_step_log_variance", None)
        if recorded is None:
            raise PricingError(
                f"{type(self).__name__} ran the continuous-KI bridge without "
                "recording the variance its own paths accumulated"
            )
        expected = (paths.shape[0], paths.shape[1] - 1)
        if recorded.shape != expected:
            raise PricingError(
                f"recorded step variance {recorded.shape} does not describe "
                f"these paths {expected}"
            )
        return recorded


class _VolModelSnowballMCBase(_SubstepRefinementMixin, SnowballMCEngine):
    engine_type = EngineType.MONTE_CARLO

    def __init__(
        self,
        params: Optional[MCParams] = None,
        method: MonteCarloMethod | str | tuple | None = None,
        use_dask: bool = False,
        num_batches: int = 4,
        substeps_per_interval: int = 1,
    ):
        self.substeps_per_interval = _validate_substeps_per_interval(
            substeps_per_interval
        )
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


class LocalVolSnowballMCEngine(_VolModelSnowballMCBase):
    """Snowball MC under a Dupire local-volatility surface."""

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
                "LocalVolSnowballMCEngine needs a GridVolSurface or a prebuilt LocalVolSurface"
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
        dt_array = self._refined_dt_array(dt_array)
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
            h2 = self._new_step_log_variance(n_eff, len(dt_array))
            spot = np.full(n_eff, float(S), dtype=float)
            nodes[:, 0] = spot
            t = 0.0
            sqrt_dt = np.sqrt(np.asarray(dt_array, dtype=float))
            for i, dt in enumerate(dt_array):
                vol = np.asarray(lv.local_vol(spot, t), dtype=float)
                if h2 is not None:
                    h2[:, i] = vol * vol * dt
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
            self._record_step_log_variance(h2)
            return nodes

        return self._make_path_generator(simulate, n_eff, batch_id)


class HestonSnowballMCEngine(_VolModelSnowballMCBase):
    """Snowball MC under the Heston stochastic-volatility model."""

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
        dt_array = self._refined_dt_array(dt_array)
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
        M = len(dt_array)

        def _draws(batch_id=None, seed=None):
            if self._uses_qmc():
                if scheme in (HestonMCScheme.QUADEXP, HestonMCScheme.QUADEXP_M):
                    from scipy.special import ndtri

                    block = np.clip(
                        _qmc_uniforms(int(self.params.seed), n_eff, 3 * M, batch_id),
                        1e-12,
                        1.0 - 1e-12,
                    )
                    return (
                        ndtri(block[:, 0:M]),
                        ndtri(block[:, M:2 * M]),
                        block[:, 2 * M:3 * M],
                    )
                from scipy.special import ndtri

                block = np.clip(
                    _qmc_uniforms(int(self.params.seed), n_eff, 2 * M, batch_id),
                    1e-12,
                    1.0 - 1e-12,
                )
                return (
                    ndtri(block[:, 0:M]),
                    ndtri(block[:, M:2 * M]),
                    None,
                )

            rng = np.random.default_rng(self._batch_seed(batch_id, seed))
            if use_antithetic:
                half = (n_paths + 1) // 2
                z_var_h = rng.standard_normal((half, M))
                z_ind_h = rng.standard_normal((half, M))
                z_var = np.concatenate([z_var_h, -z_var_h], axis=0)
                z_ind = np.concatenate([z_ind_h, -z_ind_h], axis=0)
                if scheme in (HestonMCScheme.QUADEXP, HestonMCScheme.QUADEXP_M):
                    u_var_h = rng.random((half, M))
                    u_var = np.concatenate([u_var_h, 1.0 - u_var_h], axis=0)
                else:
                    u_var = None
            else:
                z_var = rng.standard_normal((n_eff, M))
                z_ind = rng.standard_normal((n_eff, M))
                u_var = (
                    rng.random((n_eff, M))
                    if scheme in (HestonMCScheme.QUADEXP, HestonMCScheme.QUADEXP_M)
                    else None
                )
            return z_var, z_ind, u_var

        def simulate(batch_id=None, seed=None):
            z_var, z_ind, u_var = _draws(batch_id=batch_id, seed=seed)
            nodes = np.empty((n_eff, len(dt_array) + 1), dtype=float)
            h2 = self._new_step_log_variance(n_eff, len(dt_array))
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
                    "HestonSnowballMCEngine supports EULERLOG, QUADEXP, and QUADEXP_M"
                )
            if scheme == HestonMCScheme.EULERLOG:
                for i, dt in enumerate(dt_array):
                    sqrt_dt = float(np.sqrt(dt))
                    z1 = z_var[:, i] * sqrt_dt
                    z2 = z_ind[:, i] * sqrt_dt
                    z_s = rho * z1 + rho_bar * z2
                    v_plus = np.maximum(var, 0.0)
                    sqrt_v = np.sqrt(v_plus)
                    if h2 is not None:
                        h2[:, i] = v_plus * dt
                    drift = float(term.rrf[i] - term.div[i])
                    log_s = log_s + (drift - 0.5 * v_plus) * dt + sqrt_v * z_s
                    var = (
                        var
                        + p.kappa * (p.theta - v_plus) * dt
                        + p.sigma * sqrt_v * z1
                        + 0.25 * sigma2 * (z1 * z1 - dt)
                    )
                    nodes[:, i + 1] = np.exp(log_s)
                self._record_step_log_variance(h2)
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
                B = np.maximum(phi - 1.0 + np.sqrt(rad), 0.0)
                b = np.sqrt(B)
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

                v_np = np.where(psi <= psi_c, v_a, v_b)
                v_np = np.maximum(v_np, 0.0)
                v_bar = np.maximum(0.5 * (v_np + np.maximum(var, 0.0)), 0.0)
                if h2 is not None:
                    h2[:, i] = v_bar * dt
                if deterministic_vol:
                    corr = 0.0
                else:
                    corr = (rho / p.sigma) * (
                        v_np - var - p.kappa * (p.theta - v_bar) * dt
                    )
                if martingale and not deterministic_vol:
                    ros = rho / p.sigma
                    K3 = 0.5 * (1.0 - rho * rho) * dt
                    K1 = 0.5 * dt * (p.kappa * ros - 0.5) - ros
                    K2 = 0.5 * dt * (p.kappa * ros - 0.5) + ros
                    A = K2 + 0.5 * K3
                    quad_mask = psi <= psi_c
                    denom_q = 1.0 - 2.0 * A * a
                    denom_e = beta - A
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
                    m_quad = np.exp(A * a * b * b / safe_q) / np.sqrt(safe_q)
                    m_exp = prob_zero + (1.0 - prob_zero) * beta / safe_e
                    ln_M = np.log(np.where(quad_mask, m_quad, m_exp))
                    K0 = -ros * p.kappa * p.theta * dt
                    K0_star = -ln_M - (K1 + 0.5 * K3) * var
                    log_s = (
                        log_s
                        + (drift - 0.5 * v_bar) * dt
                        + corr
                        - K0
                        + K0_star
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
            self._record_step_log_variance(h2)
            return nodes

        return self._make_path_generator(simulate, n_eff, batch_id)


class QESnowballMCEngine(HestonSnowballMCEngine):
    """Standalone Snowball MC engine using the Heston QE variance scheme."""

    def __init__(
        self,
        model_params: HestonParams,
        params: Optional[MCParams] = None,
        martingale_correction: bool = False,
        **kwargs,
    ):
        if "scheme" in kwargs:
            raise ValidationError(
                "QESnowballMCEngine fixes the scheme to QUADEXP/QUADEXP_M; "
                "use HestonSnowballMCEngine for explicit scheme selection"
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


class HestonSLVSnowballMCEngine(_VolModelSnowballMCBase):
    """Snowball MC under Heston-SLV using a precomputed or on-the-fly leverage surface."""

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
                "HestonSLVSnowballMCEngine needs a GridVolSurface or a prebuilt LocalVolSurface"
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
        dt_array = self._refined_dt_array(dt_array)
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
            h2 = self._new_step_log_variance(n_eff, len(dt_array))
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
                if h2 is not None:
                    h2[:, i] = eff_vol * eff_vol * dt
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
            self._record_step_log_variance(h2)
            return nodes

        return self._make_path_generator(simulate, n_eff, batch_id)


class HestonSLVQESnowballMCEngine(HestonSLVSnowballMCEngine):
    """Standalone Snowball MC under Heston-SLV with frozen-leverage QE variance."""

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
        dt_array = self._refined_dt_array(dt_array)
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
        M = len(dt_array)

        def _draws(batch_id=None, seed=None):
            if self._uses_qmc():
                from scipy.special import ndtri

                block = np.clip(
                    _qmc_uniforms(int(self.params.seed), n_eff, 3 * M, batch_id),
                    1e-12,
                    1.0 - 1e-12,
                )
                return (
                    ndtri(block[:, 0:M]),
                    ndtri(block[:, M:2 * M]),
                    block[:, 2 * M:3 * M],
                )

            rng = np.random.default_rng(self._batch_seed(batch_id, seed))
            if use_antithetic:
                half = (n_paths + 1) // 2
                z_var_h = rng.standard_normal((half, M))
                z_ind_h = rng.standard_normal((half, M))
                u_var_h = rng.random((half, M))
                return (
                    np.concatenate([z_var_h, -z_var_h], axis=0),
                    np.concatenate([z_ind_h, -z_ind_h], axis=0),
                    np.concatenate([u_var_h, 1.0 - u_var_h], axis=0),
                )
            return (
                rng.standard_normal((n_eff, M)),
                rng.standard_normal((n_eff, M)),
                rng.random((n_eff, M)),
            )

        def simulate(batch_id=None, seed=None):
            z_var, z_ind, u_var = _draws(batch_id=batch_id, seed=seed)
            nodes = np.empty((n_eff, len(dt_array) + 1), dtype=float)
            h2 = self._new_step_log_variance(n_eff, len(dt_array))
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
                if h2 is not None:
                    h2[:, i] = leverage2 * v_bar * dt
                if deterministic_vol:
                    corr = 0.0
                else:
                    corr = leverage * (rho / sigma_eff) * (
                        v_np - var - p.kappa * (p.theta - v_bar) * dt
                    )

                if martingale:
                    ros = leverage * rho / sigma_eff
                    K3 = 0.5 * leverage2 * (1.0 - rho * rho) * dt
                    K1 = 0.5 * dt * (p.kappa * ros - 0.5 * leverage2) - ros
                    K2 = 0.5 * dt * (p.kappa * ros - 0.5 * leverage2) + ros
                    A = K2 + 0.5 * K3
                    quad_mask = psi <= psi_c
                    denom_q = 1.0 - 2.0 * A * a
                    denom_e = beta - A
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
                    m_quad = np.exp(A * a * b * b / safe_q) / np.sqrt(safe_q)
                    m_exp = prob_zero + (1.0 - prob_zero) * beta / safe_e
                    ln_M = np.log(np.where(quad_mask, m_quad, m_exp))
                    K0 = -ros * p.kappa * p.theta * dt
                    K0_star = -ln_M - (K1 + 0.5 * K3) * var
                    log_s = (
                        log_s
                        + (drift - 0.5 * leverage2 * v_bar) * dt
                        + corr
                        - K0
                        + K0_star
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
            self._record_step_log_variance(h2)
            return nodes

        return self._make_path_generator(simulate, n_eff, batch_id)
