"""Snowball Monte-Carlo engines under Local Vol / Heston / Heston-SLV processes."""

from __future__ import annotations

import math
from typing import Optional

import threading

import numpy as np
from scipy.special import ndtr, ndtri

from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.mc.term_inputs import build_mc_term_inputs
from quantark.asset.equity.param import MCParams
from quantark.param import GridVolSurface
from quantark.priceenv import PricingEnvironment
from quantark.montecarlo.qe_kernels import qe_variance_step
from quantark.montecarlo.qmc_brownian_bridge import apply_brownian_bridge
from quantark.montecarlo.qmc_sobol import SobolNormalGenerator
from quantark.util.enum import CouponPayType, ObservationType
from quantark.util.enum.engine_enums import (
    EngineType,
    HestonMCScheme,
    MonteCarloMethod,
)
from quantark.util.exceptions import NumericalError, PricingError, ValidationError
from quantark.util.numerical import safe_log
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


def _advance_conditional_affine_spot(
    log_base: np.ndarray,
    loading: np.ndarray,
    leverage: np.ndarray,
    *,
    var: np.ndarray,
    v_np: np.ndarray,
    v_bar: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    beta: np.ndarray,
    prob_zero: np.ndarray,
    quad_mask: np.ndarray,
    drift: float,
    dt: float,
    rho: float,
    sigma_eff: float,
    kappa: float,
    theta: float,
    sqrt_dt: float,
    diff_coef: float,
    residual_z: np.ndarray,
    spot_loading: float,
    martingale: bool,
    deterministic_vol: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance one exact-affine conditional spot proxy on shared QE draws."""
    leverage = np.asarray(leverage, dtype=float)
    leverage2 = leverage * leverage
    if deterministic_vol:
        corr = 0.0
    else:
        corr = leverage * (rho / sigma_eff) * (
            v_np - var - kappa * (theta - v_bar) * dt
        )[:, None]
    if martingale:
        ros = leverage * rho / sigma_eff
        k3 = 0.5 * leverage2 * (1.0 - rho * rho) * dt
        k1 = 0.5 * dt * (kappa * ros - 0.5 * leverage2) - ros
        k2 = 0.5 * dt * (kappa * ros - 0.5 * leverage2) + ros
        mgf_argument = k2 + 0.5 * k3
        denominator_quadratic = 1.0 - 2.0 * mgf_argument * a[:, None]
        denominator_exponential = beta[:, None] - mgf_argument
        invalid = (
            quad_mask[:, None] & (denominator_quadratic <= 0.0)
        ) | (~quad_mask[:, None] & (denominator_exponential <= 0.0))
        if np.any(invalid):
            raise NumericalError(
                "Heston conditional-control MGF is undefined at these "
                "parameters; tighten dt"
            )
        safe_quadratic = np.where(
            denominator_quadratic > 0.0, denominator_quadratic, 1.0
        )
        safe_exponential = np.where(
            denominator_exponential > 0.0, denominator_exponential, 1.0
        )
        mgf_quadratic = (
            np.exp(
                mgf_argument
                * a[:, None]
                * b[:, None]
                * b[:, None]
                / safe_quadratic
            )
            / np.sqrt(safe_quadratic)
        )
        mgf_exponential = (
            prob_zero[:, None]
            + (1.0 - prob_zero[:, None])
            * beta[:, None]
            / safe_exponential
        )
        log_mgf = np.log(
            np.where(quad_mask[:, None], mgf_quadratic, mgf_exponential)
        )
        k0 = -ros * kappa * theta * dt
        k0_star = -log_mgf - (k1 + 0.5 * k3) * var[:, None]
        base_increment = (
            (drift - 0.5 * leverage2 * v_bar[:, None]) * dt
            + corr
            - k0
            + k0_star
        )
    else:
        base_increment = (
            (drift - 0.5 * leverage2 * v_bar[:, None]) * dt + corr
        )
    diffusion = leverage * np.sqrt(v_bar[:, None]) * sqrt_dt * diff_coef
    return (
        log_base + base_increment + diffusion * residual_z,
        loading + diffusion * spot_loading,
    )


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
        generated = self._simulator(batch_id=effective_batch, seed=seed)
        if isinstance(generated, tuple):
            paths, simulator_aux = generated
            aux = dict(simulator_aux or {})
        else:
            paths = generated
            aux = {}
        aux["batch_id"] = 0 if effective_batch is None else int(effective_batch)
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
            generated = simulate(batch_id=batch_id, seed=seed)
            if not isinstance(generated, tuple):
                return generated[:, ::n]
            paths, aux = generated
            aux = dict(aux or {})
            if aux.pop("_paths_are_contractual", False):
                return paths, aux
            control = aux.get("control_paths")
            if (
                isinstance(control, np.ndarray)
                and control.ndim == 2
                and control.shape[1] == paths.shape[1]
            ):
                aux["control_paths"] = control[:, ::n]
            return paths[:, ::n], aux

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

    def _bridge_variance_store(self) -> dict:
        """Per-thread slots for the recorded variance.

        The recording happens inside ``simulate`` and is consumed by the bridge
        during ``_compute_payoffs``. Those are one call apart in serial use, but
        the legacy dask path runs each batch through ``delayed``/``compute`` on
        the THREADED scheduler, so several batches sit between their own
        generate and consume at once on this same engine object. A single
        instance attribute is a race there -- batches of 6667 and 6666 paths
        overwriting each other -- so each thread gets its own slot. Every batch
        generates and consumes within one ``batch_fn`` call, hence one thread.
        """
        store = self.__dict__.get("_bridge_variance_slots")
        if store is None:
            store = self.__dict__.setdefault("_bridge_variance_slots", {})
        return store

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
        if fine is not None:
            n = self.substeps_per_interval
            if n > 1:
                fine = fine.reshape(fine.shape[0], -1, n).sum(axis=2)
        self._bridge_variance_store()[threading.get_ident()] = fine

    @property
    def _step_log_variance(self):
        """The variance recorded by THIS thread's most recent simulate."""
        return self._bridge_variance_store().get(threading.get_ident())

    def _ki_bridge_step_log_variance(self, paths):
        recorded = self._bridge_variance_store().get(threading.get_ident())
        if recorded is None:
            raise PricingError(
                f"{type(self).__name__} ran the continuous-KI bridge without "
                "recording the variance its own paths accumulated"
            )
        # Shape, not identity: with substeps_per_interval > 1 the recorded
        # nodes are handed on as a strided view, so the array the payoffs see
        # is not the one simulate built. Shape plus per-thread scoping is what
        # is actually checkable, and the shape check is what caught the dask
        # race described in _bridge_variance_store.
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
        rqmc_affine_spot_factor: bool = False,
        rqmc_heston_conditional_control: bool = False,
        rqmc_frozen_leverage_conditional_control: bool = False,
        rqmc_conditional_control_only: bool = False,
        rqmc_qe_draw_provider=None,
        rqmc_spot_strata: int = 1,
        rqmc_spot_antithetic: bool = False,
        rqmc_spot_bridge_strata: int = 1,
        rqmc_spot_bridge_dimensions: int = 1,
    ):
        if not isinstance(rqmc_affine_spot_factor, bool):
            raise ValidationError("rqmc_affine_spot_factor must be bool")
        if not isinstance(rqmc_heston_conditional_control, bool):
            raise ValidationError("rqmc_heston_conditional_control must be bool")
        if not isinstance(rqmc_frozen_leverage_conditional_control, bool):
            raise ValidationError(
                "rqmc_frozen_leverage_conditional_control must be bool"
            )
        if not isinstance(rqmc_conditional_control_only, bool):
            raise ValidationError("rqmc_conditional_control_only must be bool")
        if rqmc_affine_spot_factor and rqmc_heston_conditional_control:
            raise ValidationError(
                "exact affine conditioning and the Heston conditional control "
                "are mutually exclusive"
            )
        if (
            isinstance(rqmc_spot_strata, bool)
            or not isinstance(rqmc_spot_strata, (int, np.integer))
            or int(rqmc_spot_strata) < 1
        ):
            raise ValidationError("rqmc_spot_strata must be a positive integer")
        if not isinstance(rqmc_spot_antithetic, bool):
            raise ValidationError("rqmc_spot_antithetic must be bool")
        if (
            isinstance(rqmc_spot_bridge_strata, bool)
            or not isinstance(rqmc_spot_bridge_strata, (int, np.integer))
            or int(rqmc_spot_bridge_strata) < 1
        ):
            raise ValidationError(
                "rqmc_spot_bridge_strata must be a positive integer"
            )
        if (
            isinstance(rqmc_spot_bridge_dimensions, bool)
            or not isinstance(rqmc_spot_bridge_dimensions, (int, np.integer))
            or int(rqmc_spot_bridge_dimensions) < 1
        ):
            raise ValidationError(
                "rqmc_spot_bridge_dimensions must be a positive integer"
            )
        if int(rqmc_spot_strata) > 1 and not rqmc_heston_conditional_control:
            raise ValidationError(
                "rqmc_spot_strata > 1 requires rqmc_heston_conditional_control"
            )
        if rqmc_spot_antithetic and not rqmc_heston_conditional_control:
            raise ValidationError(
                "rqmc_spot_antithetic requires rqmc_heston_conditional_control"
            )
        if (
            int(rqmc_spot_bridge_strata) > 1
            and not (
                rqmc_heston_conditional_control or rqmc_affine_spot_factor
            )
        ):
            raise ValidationError(
                "rqmc_spot_bridge_strata > 1 requires "
                "exact affine conditioning or rqmc_heston_conditional_control"
            )
        if (
            int(rqmc_spot_bridge_dimensions) > 1
            and int(rqmc_spot_bridge_strata) == 1
        ):
            raise ValidationError(
                "rqmc_spot_bridge_dimensions > 1 requires "
                "rqmc_spot_bridge_strata > 1"
            )
        if (
            rqmc_frozen_leverage_conditional_control
            and not rqmc_heston_conditional_control
        ):
            raise ValidationError(
                "rqmc_frozen_leverage_conditional_control requires "
                "rqmc_heston_conditional_control"
            )
        if rqmc_conditional_control_only and not (
            rqmc_heston_conditional_control
            and rqmc_frozen_leverage_conditional_control
        ):
            raise ValidationError(
                "rqmc_conditional_control_only requires "
                "the frozen-leverage Heston conditional control"
            )
        if rqmc_conditional_control_only and (
            int(rqmc_spot_strata) != 1 or rqmc_spot_antithetic
        ):
            raise ValidationError(
                "conditional-control-only RQMC requires one spot stratum "
                "without spot antithetic sampling"
            )
        self.substeps_per_interval = _validate_substeps_per_interval(
            substeps_per_interval
        )
        self.rqmc_affine_spot_factor = rqmc_affine_spot_factor
        self.rqmc_heston_conditional_control = rqmc_heston_conditional_control
        self.rqmc_frozen_leverage_conditional_control = (
            rqmc_frozen_leverage_conditional_control
        )
        self.rqmc_conditional_control_only = rqmc_conditional_control_only
        self.rqmc_spot_strata = int(rqmc_spot_strata)
        self.rqmc_spot_antithetic = rqmc_spot_antithetic
        self.rqmc_spot_bridge_strata = int(rqmc_spot_bridge_strata)
        self.rqmc_spot_bridge_dimensions = int(rqmc_spot_bridge_dimensions)
        if rqmc_qe_draw_provider is not None and not all(
            hasattr(rqmc_qe_draw_provider, name)
            for name in ("draws", "dimension", "label", "randomization_key")
        ):
            raise ValidationError("invalid rqmc_qe_draw_provider")
        self.rqmc_qe_draw_provider = rqmc_qe_draw_provider
        super().__init__(
            params=params,
            method=method,
            use_dask=use_dask,
            num_batches=num_batches,
        )

    def _uses_qmc(self) -> bool:
        return self.method in _QMC_METHODS

    def _rqmc_scheme_label(self) -> str:
        label = super()._rqmc_scheme_label()
        if self.rqmc_affine_spot_factor:
            label += "#affine-spot-factor"
        if self.rqmc_heston_conditional_control:
            label += "#heston-conditional-control"
        if self.rqmc_frozen_leverage_conditional_control:
            label += "#frozen-leverage-proxy"
        if self.rqmc_conditional_control_only:
            label += "#conditional-control-only"
        if self.rqmc_spot_strata > 1:
            label += f"#spot-strata-{self.rqmc_spot_strata}"
        if self.rqmc_spot_antithetic:
            label += "#spot-antithetic"
        if self.rqmc_spot_bridge_strata > 1:
            label += f"#spot-bridge-strata-{self.rqmc_spot_bridge_strata}"
            label += f"-dimensions-{self.rqmc_spot_bridge_dimensions}"
        if self.rqmc_qe_draw_provider is not None:
            label += f"#{self.rqmc_qe_draw_provider.label}"
        return label

    def _rqmc_dimension(self, contractual_time_steps: int) -> int:
        if self.rqmc_qe_draw_provider is not None:
            return int(self.rqmc_qe_draw_provider.dimension)
        return super()._rqmc_dimension(contractual_time_steps)

    def _rqmc_randomization_key(self, contractual_time_steps: int):
        if self.rqmc_qe_draw_provider is not None:
            return self.rqmc_qe_draw_provider.randomization_key
        return super()._rqmc_randomization_key(contractual_time_steps)

    def _conditioned_spot_normals(
        self,
        z_ind: np.ndarray,
        dt_array: np.ndarray,
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Remove the terminal Brownian factor and expose its affine loading.

        The first Brownian-bridge coordinate is ``W_T / sqrt(T)``.  Setting it
        to zero leaves a residual chronological normal matrix. The caller
        records the cumulative log-spot loading ``B_t`` corresponding to
        ``sqrt(dt / T)`` and an exact conditional payoff integrator then
        evaluates ``E[payoff | variance, residual spot factors]`` analytically.
        """
        if not (
            self.rqmc_affine_spot_factor
            or self.rqmc_heston_conditional_control
        ):
            return None
        if not self._uses_qmc():
            raise ValidationError(
                "spot-factor conditioning requires QMC or randomized QMC"
            )
        dt = np.asarray(dt_array, dtype=float)
        if np.any(dt <= 0.0):
            raise ValidationError("spot-factor conditioning requires positive dt")
        raw = np.array(z_ind, dtype=float, copy=True)
        raw[:, 0] = 0.0
        residual_dw = apply_brownian_bridge(raw, np.cumsum(dt))
        residual_z = residual_dw / np.sqrt(dt)[None, :]
        loadings = np.sqrt(dt / float(np.sum(dt)))
        return residual_z, loadings

    def _bridge_stratified_spot_normals(
        self,
        z_ind: np.ndarray,
        dt_array: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Condition the terminal factor and stratify leading bridge factors.

        The first residual bridge coordinate uses ordinary one-dimensional
        stratification. Additional coordinates use independently randomized
        rank-one rotations driven by their original scrambled-Sobol uniforms.
        Each inner point is therefore marginally uniform and the average is
        unbiased, while leading Brownian-bridge main effects are integrated at
        no additional outer-path count.
        """
        bridge_ordered_source = np.asarray(z_ind, dtype=float)
        if bridge_ordered_source.ndim != 2:
            raise ValidationError("spot bridge normals must be a 2D array")
        n_outer, steps = bridge_ordered_source.shape
        strata = int(self.rqmc_spot_bridge_strata)
        dimensions = int(self.rqmc_spot_bridge_dimensions)
        if steps <= dimensions:
            raise ValidationError(
                "spot bridge stratification dimensions must be smaller than "
                "the number of time steps"
            )
        dt = np.asarray(dt_array, dtype=float)
        if dt.shape != (steps,) or np.any(dt <= 0.0):
            raise ValidationError(
                "spot bridge stratification requires positive aligned dt"
            )

        bridge_ordered = np.repeat(
            bridge_ordered_source[:, None, :], strata, axis=1
        )
        bridge_ordered[:, :, 0] = 0.0
        inner_index = np.arange(strata, dtype=float)[None, :]
        for local_index, bridge_index in enumerate(range(1, dimensions + 1)):
            base_uniform = ndtr(bridge_ordered_source[:, bridge_index])[:, None]
            if local_index == 0:
                uniforms = (base_uniform + inner_index) / float(strata)
            else:
                stride = 2 * local_index + 1
                while math.gcd(stride, strata) != 1:
                    stride += 2
                uniforms = np.mod(
                    base_uniform + inner_index * (stride / float(strata)),
                    1.0,
                )
            bridge_ordered[:, :, bridge_index] = ndtri(
                np.clip(uniforms, 1e-12, 1.0 - 1e-12)
            )

        residual_dw = apply_brownian_bridge(
            bridge_ordered.reshape(n_outer * strata, steps),
            np.cumsum(dt),
        )
        residual_z = (
            residual_dw / np.sqrt(dt)[None, :]
        ).reshape(n_outer, strata, steps)
        loadings = np.sqrt(dt / float(np.sum(dt)))
        return residual_z, loadings

    def _validate_rng_controls(self, use_antithetic: bool) -> None:
        if self.rqmc_qe_draw_provider is not None and not self._uses_qmc():
            raise ValidationError(
                "rqmc_qe_draw_provider requires QMC or randomized QMC"
            )
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
    """Snowball MC under a Dupire local-volatility surface.

    ``lv_time_sampling`` selects where each SDE step samples the surface's time
    axis: ``"left"`` (default, bitwise-identical to the historical scheme),
    ``"mid"`` (t + dt/2), or ``"integrated"`` (exact per-step time-averaged
    variance; exact for time-only surfaces). The recorded step log-variance --
    and therefore the continuous-KI bridge -- always uses the same effective vol
    the step was drawn with.
    """

    _TIME_SAMPLING_MODES = ("left", "mid", "integrated")
    _ESTIMATORS = ("plain", "one_step_survival")

    def __init__(
        self,
        params: Optional[MCParams] = None,
        local_vol_surface: Optional[LocalVolSurface] = None,
        lv_time_sampling: str = "left",
        estimator: str = "plain",
        **kwargs,
    ):
        super().__init__(params=params, **kwargs)
        self._prebuilt = local_vol_surface
        if lv_time_sampling not in self._TIME_SAMPLING_MODES:
            raise ValidationError(
                "lv_time_sampling must be one of "
                f"{self._TIME_SAMPLING_MODES}, got {lv_time_sampling!r}"
            )
        self.lv_time_sampling = lv_time_sampling
        if estimator not in self._ESTIMATORS:
            raise ValidationError(
                f"estimator must be one of {self._ESTIMATORS}, got {estimator!r}"
            )
        self.estimator = estimator
        if estimator == "one_step_survival":
            if self.method == MonteCarloMethod.RANDOMIZED_QUASI:
                raise ValidationError(
                    "estimator='one_step_survival' does not support "
                    "RANDOMIZED_QUASI (the RQMC control-variate machinery "
                    "assumes the plain estimator); use PSEUDO or QUASI"
                )
            if bool(getattr(self.params, "use_antithetic", False)):
                raise ValidationError(
                    "estimator='one_step_survival' does not support antithetic "
                    "sampling (survival conditioning is not sign-symmetric)"
                )

    def _rqmc_scheme_label(self) -> str:
        label = super()._rqmc_scheme_label()
        if self.lv_time_sampling != "left":
            label = f"{label}+ts:{self.lv_time_sampling}"
        if self.estimator != "plain":
            label = f"{label}+est:{self.estimator}"
        return label

    # --- one-step survival (Glasserman-Staum / Alm et al.) ------------------
    #
    # The estimator factorizes the snowball across KO observation dates:
    #     E[payoff] = sum_k E[W_{k-1} q_k] * KO_value_k
    #                 + E[W_N * maturity_leg(steered path)]
    # where q_k is the one-step KO probability conditional on the state one SDE
    # step before date k, W_k = prod_{j<=k}(1 - q_j), and the simulated path is
    # steered into the surviving region at each KO-date step. The maturity leg
    # (V0/V1 classification, continuous-KI bridge, protection/airbag/rebate
    # variants) is evaluated by the CERTIFIED base `_compute_payoffs` on the
    # steered never-KO paths; only the branch weights are new. Unbiasedness is
    # the iterated-conditional-expectation identity; the estimator is Lipschitz
    # in spot, which is what makes small-bump FD gamma stable.

    def _validate_inputs(self, S, T, r, q, sigma, product) -> None:
        super()._validate_inputs(S, T, r, q, sigma, product)
        if self.estimator == "one_step_survival":
            self._validate_oss_product(product)
            self._oss_product = product

    def _validate_oss_product(self, product) -> None:
        from quantark.asset.equity.product.option.ko_reset_snowball_option import (
            KnockOutResetSnowballOption,
        )

        if isinstance(product, KnockOutResetSnowballOption):
            raise ValidationError(
                "estimator='one_step_survival' does not support KO-reset snowballs"
            )
        if product.barrier_config.ko_observation_type == ObservationType.CONTINUOUS:
            raise ValidationError(
                "estimator='one_step_survival' requires DISCRETE KO observation "
                "(the conditioning acts at KO observation dates)"
            )
        if product.barrier_config.disable_ko_after_ki:
            raise ValidationError(
                "estimator='one_step_survival' does not support "
                "disable_ko_after_ki (the KO survival factorization assumes KO "
                "is always live)"
            )
        if bool(getattr(product, "_otc_lifecycle_knocked_in", False)):
            raise ValidationError(
                "estimator='one_step_survival' does not support the "
                "already-knocked-in lifecycle state"
            )

    def _oss_store(self) -> dict:
        """Per-thread slots for the recorded per-date KO probabilities
        (same concurrency contract as ``_bridge_variance_store``)."""
        store = self.__dict__.get("_oss_prob_slots")
        if store is None:
            store = self.__dict__.setdefault("_oss_prob_slots", {})
        return store

    def calculate_event_stats(self, *args, **kwargs):
        if self.estimator == "one_step_survival":
            raise ValidationError(
                "calculate_event_stats is not supported with "
                "estimator='one_step_survival'; use the plain estimator"
            )
        return super().calculate_event_stats(*args, **kwargs)

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
        if self.rqmc_qe_draw_provider is not None:
            raise ValidationError(
                "rqmc_qe_draw_provider is implemented only for QE/QE-M engines"
            )
        n_contract = len(dt_array)
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

        if self.estimator == "one_step_survival":
            return self._create_oss_path_generator(
                S, T, dt_array, n_contract, term, lv, n_eff, batch_id
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
            time_sampling = self.lv_time_sampling
            for i, dt in enumerate(dt_array):
                if time_sampling == "left":
                    vol = np.asarray(lv.local_vol(spot, t), dtype=float)
                elif time_sampling == "mid":
                    vol = np.asarray(
                        lv.local_vol(spot, t + 0.5 * float(dt)), dtype=float
                    )
                else:  # "integrated"
                    vol = np.sqrt(
                        np.asarray(
                            lv.time_avg_var(spot, t, t + float(dt)), dtype=float
                        )
                    )
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

    def _create_oss_path_generator(
        self, S, T, dt_array, n_contract, term, lv, n_eff, batch_id
    ):
        """Steered path generator for the one-step-survival estimator.

        At every refined SDE step ending on a KO observation node the normal
        draw is replaced by its conditional-on-survival counterpart and the
        per-path KO probability is recorded (thread-keyed, consumed by
        ``_compute_payoffs``). All other steps are the plain scheme driven by
        ndtri-transformed uniforms.
        """
        product = getattr(self, "_oss_product", None)
        if product is None:
            raise PricingError(
                "one_step_survival pricing ran without product context "
                "(_validate_inputs was bypassed)"
            )
        env, _ = self._term_ctx
        all_times, dt_contract, ko_indices, _ = self._build_time_grid(
            product, env, T
        )
        if len(dt_contract) != n_contract:
            raise PricingError(
                "one_step_survival grid mismatch: generator received "
                f"{n_contract} contractual steps, _build_time_grid produced "
                f"{len(dt_contract)}"
            )
        ko_profile = product.get_ko_observation_profile(env)
        ko_barriers = np.broadcast_to(
            np.asarray(ko_profile["barriers"], dtype=float), (len(ko_indices),)
        ).copy()
        is_reverse = bool(product.is_reverse)
        n_sub = max(int(self.substeps_per_interval), 1)
        n_fine = len(dt_array)
        # Refined step i ends on fine node i+1; contractual node col c = (i+1)/n_sub.
        # KO observation k sits at contractual node col ko_indices[k] + 1.
        ko_slot_by_step = np.full(n_fine, -1, dtype=int)
        for k, col in enumerate(np.asarray(ko_indices, dtype=int) + 1):
            ko_slot_by_step[int(col) * n_sub - 1] = k
        n_ko = len(ko_indices)
        tiny = 1e-15

        def simulate(batch_id=None, seed=None):
            rng = np.random.default_rng(self._batch_seed(batch_id, seed))
            u_all = (
                _qmc_uniforms(int(self.params.seed), n_eff, n_fine, batch_id)
                if self._uses_qmc()
                else None
            )
            nodes = np.empty((n_eff, n_fine + 1), dtype=float)
            h2 = self._new_step_log_variance(n_eff, n_fine)
            q_ko = np.zeros((n_eff, n_ko), dtype=float)
            spot = np.full(n_eff, float(S), dtype=float)
            nodes[:, 0] = spot
            t = 0.0
            sqrt_dt = np.sqrt(np.asarray(dt_array, dtype=float))
            time_sampling = self.lv_time_sampling
            for i, dt in enumerate(dt_array):
                if time_sampling == "left":
                    vol = np.asarray(lv.local_vol(spot, t), dtype=float)
                elif time_sampling == "mid":
                    vol = np.asarray(
                        lv.local_vol(spot, t + 0.5 * float(dt)), dtype=float
                    )
                else:  # "integrated"
                    vol = np.sqrt(
                        np.asarray(
                            lv.time_avg_var(spot, t, t + float(dt)), dtype=float
                        )
                    )
                if h2 is not None:
                    h2[:, i] = vol * vol * dt
                u = np.clip(
                    u_all[:, i] if u_all is not None else rng.random(n_eff),
                    tiny,
                    1.0 - tiny,
                )
                drift = float(term.rrf[i] - term.div[i])
                k = int(ko_slot_by_step[i])
                if k >= 0:
                    sd = np.maximum(vol * sqrt_dt[i], 1e-300)
                    z_star = (
                        np.log(ko_barriers[k] / spot)
                        - (drift - 0.5 * vol * vol) * dt
                    ) / sd
                    if is_reverse:
                        # DOWN-KO: survive means end strictly ABOVE the barrier
                        p_hit = np.clip(ndtr(z_star), 0.0, 1.0 - 1e-300)
                        z = ndtri(np.clip(p_hit + u * (1.0 - p_hit), tiny, 1.0 - tiny))
                    else:
                        # UP-KO: survive means end strictly BELOW the barrier
                        p_surv = np.clip(ndtr(z_star), 1e-300, 1.0)
                        z = ndtri(np.clip(u * p_surv, tiny, 1.0 - tiny))
                        p_hit = 1.0 - p_surv
                    q_ko[:, k] = p_hit
                else:
                    z = ndtri(u)
                spot = spot * np.exp(
                    (drift - 0.5 * vol * vol) * dt + vol * sqrt_dt[i] * z
                )
                if k >= 0:
                    # Enforce the conditioned branch against float rounding: a
                    # steered endpoint must sit strictly on the surviving side,
                    # or the downstream hard KO check re-introduces the event
                    # the weights already account for.
                    if is_reverse:
                        spot = np.maximum(spot, ko_barriers[k] * (1.0 + 1e-12))
                    else:
                        spot = np.minimum(spot, ko_barriers[k] * (1.0 - 1e-12))
                nodes[:, i + 1] = spot
                t += float(dt)
            self._record_step_log_variance(h2)
            self._oss_store()[threading.get_ident()] = q_ko
            return nodes

        return self._make_path_generator(simulate, n_eff, batch_id)

    def _oss_ki_survival_weight(
        self, product, pricing_env, paths, ki_indices
    ) -> np.ndarray:
        """Per-path probability of NOT knocking in, as a smooth weight.

        For continuous KI this is the Brownian-bridge no-cross product (the
        conditional expectation of the sampled indicator the plain estimator
        draws -- same expectation, Rao-Blackwellized, and continuous in spot,
        which is what keeps small-bump FD gamma stable). The bridge uses the
        per-step log-variance the scheme recorded, and a step with a breached
        endpoint contributes survival 0 -- continuously, since the no-cross
        probability already tends to 0 as an endpoint approaches the barrier.
        For discrete KI the hard indicator is deterministic given the path and
        is kept as a 0/1 weight.
        """
        n_paths = paths.shape[0]
        if not product.has_ki_barrier:
            return np.ones(n_paths)
        ki_profile = product.get_ki_observation_profile(pricing_env)
        ki_barriers_val = np.array(ki_profile["barriers"])
        ki_continuous = (
            product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
            or product.barrier_config.ki_continuous
        )
        if not ki_continuous:
            ki_triggered, _ = self._check_ki_barriers(
                paths, ki_indices, ki_barriers_val, product.is_reverse
            )
            return np.where(ki_triggered, 0.0, 1.0)
        if ki_barriers_val.shape not in ((), (1,)):
            raise ValidationError(
                "Continuous KI monitoring requires a scalar ki_barrier."
            )
        barrier = float(ki_barriers_val.reshape(-1)[0])
        h2 = self._ki_bridge_step_log_variance(paths)
        s_i, s_j = paths[:, :-1], paths[:, 1:]
        if product.is_reverse:
            breached = (s_i >= barrier) | (s_j >= barrier)
        else:
            breached = (s_i <= barrier) | (s_j <= barrier)
        log_term = safe_log(s_i / barrier) * safe_log(s_j / barrier)
        with np.errstate(divide="ignore", invalid="ignore"):
            exponent = -2.0 * log_term / h2
        exponent = np.clip(exponent, -745.0, 0.0)
        step_surv = np.where(breached, 0.0, 1.0 - np.exp(exponent))
        return np.prod(step_surv, axis=1)

    def _compute_payoffs(
        self,
        product,
        pricing_env,
        paths,
        all_times,
        ko_indices,
        ki_indices,
        r,
        T,
        sigma,
        rng_seed,
    ):
        if self.estimator != "one_step_survival":
            return super()._compute_payoffs(
                product, pricing_env, paths, all_times, ko_indices, ki_indices,
                r, T, sigma, rng_seed,
            )
        n_paths = paths.shape[0]
        n_ko = len(ko_indices)
        # pop, not get: the slot holds a paths-sized buffer per thread, and a
        # consumed-but-retained array could both leak and satisfy the shape
        # guard for a LATER pricing whose own recording was skipped.
        q_ko = self._oss_store().pop(threading.get_ident(), None)
        if q_ko is None or q_ko.shape != (n_paths, n_ko):
            raise PricingError(
                "one_step_survival KO probabilities were not recorded for "
                "these paths"
            )
        ko_profile = product.get_ko_observation_profile(pricing_env)
        ko_triggered, _ = self._check_ko_barriers(
            paths, np.asarray(ko_indices, dtype=int),
            np.array(ko_profile["barriers"]), product.is_reverse,
        )
        if bool(ko_triggered.any()):
            raise NumericalError(
                "steered one_step_survival paths triggered the hard KO check; "
                "the survival conditioning is inconsistent with the barrier "
                "comparison"
            )
        surv = 1.0 - q_ko
        # W_prev[:, k] = prod_{j<k}(1 - q_j); W_final = prod over all KO dates
        W_prev = np.cumprod(
            np.concatenate([np.ones((n_paths, 1)), surv[:, :-1]], axis=1), axis=1
        )
        W_final = W_prev[:, -1] * surv[:, -1] if n_ko else np.ones(n_paths)

        # KO branch: deterministic per-date payoffs weighted by the recorded
        # one-step probabilities (same schedule/timings the plain path uses).
        ko_pay = np.asarray(ko_profile["payoffs"], dtype=float)
        ko_times = np.asarray(ko_profile["observation_times"], dtype=float)
        timings = self._payment_timings
        if product.accrual_config.coupon_pay_type == CouponPayType.INSTANT:
            ko_settle = np.asarray(timings.observation_payment_times, dtype=float)
        else:
            ko_settle = np.full(n_ko, float(timings.terminal.payment_time))
        ko_df = np.asarray(self._df(ko_settle), dtype=float)
        ko_w = W_prev * q_ko  # per-path branch weight of KO at date k
        branch_pv = ko_w @ (ko_pay * ko_df)

        # Maturity leg on the steered (never-KO) paths, with the KI indicator
        # replaced by its smooth survival weight. Payoff variants stay in the
        # product's own certified methods.
        w_ki = self._oss_ki_survival_weight(product, pricing_env, paths, ki_indices)
        terminal_spots = paths[:, -1]
        v0_payoffs = np.zeros(n_paths)
        v1_payoffs = np.zeros(n_paths)
        need_v0 = w_ki > 1e-14
        need_v1 = w_ki < 1.0 - 1e-14
        if need_v0.any():
            v0_payoffs[need_v0] = np.array(
                [
                    product.get_maturity_payoff_v0(spot, pricing_env)
                    for spot in terminal_spots[need_v0]
                ]
            )
        if need_v1.any():
            v1_payoffs[need_v1] = np.array(
                [
                    product.get_maturity_payoff_v1(spot, pricing_env)
                    for spot in terminal_spots[need_v1]
                ]
            )
        terminal_df = float(self._df(float(timings.terminal.payment_time)))
        maturity_pv = (
            W_final * (w_ki * v0_payoffs + (1.0 - w_ki) * v1_payoffs) * terminal_df
        )
        pv = branch_pv + maturity_pv

        # Weighted event stats (probabilities exact; counts rounded, reporting-only).
        ko_prob_path = ko_w.sum(axis=1)
        ko_probability = float(ko_prob_path.mean())
        v0_probability = float((W_final * w_ki).mean())
        v1_probability = float((W_final * (1.0 - w_ki)).mean())
        ko_time_sum = float((ko_w @ ko_times).sum()) if n_ko else 0.0
        ko_weight_total = float(ko_prob_path.sum())
        stats_oss = {
            "ko_probability": ko_probability,
            "v0_probability": v0_probability,
            "v1_probability": v1_probability,
            "ko_count": int(round(ko_weight_total)),
            "v0_count": int(round(v0_probability * n_paths)),
            "v1_count": int(round(v1_probability * n_paths)),
            "avg_ko_time": (
                ko_time_sum / ko_weight_total if ko_weight_total > 0.0 else None
            ),
            "ko_time_sum": ko_time_sum,
            "ko_time_count": int(round(ko_weight_total)),
        }
        # PVs are already discounted; hand back zero settlement times so the
        # caller's df(0) = 1 discounting is a no-op.
        return pv, np.zeros(n_paths, dtype=float), stats_oss


class HestonSnowballMCEngine(_VolModelSnowballMCBase):
    """Snowball MC under the Heston stochastic-volatility model."""

    rqmc_homogeneous_spot_scaling = True

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
        if self.rqmc_heston_conditional_control:
            raise ValidationError(
                "rqmc_heston_conditional_control is for Heston-SLV QE only"
            )
        self.model_params = model_params
        self.scheme = _resolve_heston_scheme(scheme)
        if self.rqmc_qe_draw_provider is not None and self.scheme not in (
            HestonMCScheme.QUADEXP,
            HestonMCScheme.QUADEXP_M,
        ):
            raise ValidationError(
                "rqmc_qe_draw_provider requires the QUADEXP or QUADEXP_M scheme"
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
                if self.rqmc_qe_draw_provider is not None:
                    return self.rqmc_qe_draw_provider.draws(
                        n_paths=n_eff,
                        dt_array=dt_array,
                        batch_id=batch_id,
                    )
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
            conditioning = self._conditioned_spot_normals(z_ind, dt_array)
            if conditioning is None and self.rqmc_qe_draw_provider is not None:
                # CoupledQESubstepDrawProvider exposes the independent spot
                # stream in Brownian-bridge order. Unconditioned consumers
                # must transform it to chronological normalized increments;
                # conditioned consumers do this inside _conditioned_spot_normals.
                z_ind = apply_brownian_bridge(
                    z_ind, np.cumsum(dt_array)
                ) / np.sqrt(dt_array)[None, :]
            if conditioning is None:
                log_s = np.full(
                    n_eff, np.log(max(float(S), 1e-12)), dtype=float
                )
                nodes = np.empty((n_eff, len(dt_array) + 1), dtype=float)
                h2 = self._new_step_log_variance(n_eff, len(dt_array))
            else:
                # The conditional-control path records nodes on the CONTRACTUAL
                # grid with one row per (path, bridge stratum), which the
                # bridge's (paths, steps) variance buffer cannot describe.  Leave
                # it unrecorded: _ki_bridge_step_log_variance then raises rather
                # than pricing a continuous KI off a variance these paths never
                # accumulated.  No certified configuration reaches both at once.
                h2 = None
                residual_z, spot_loadings = conditioning
                bridge_strata = int(self.rqmc_spot_bridge_strata)
                if bridge_strata > 1:
                    residual_z, spot_loadings = (
                        self._bridge_stratified_spot_normals(z_ind, dt_array)
                    )
                else:
                    residual_z = residual_z[:, None, :]
                log_s = np.full(
                    (n_eff, bridge_strata),
                    np.log(max(float(S), 1e-12)),
                    dtype=float,
                )
                contractual_steps = len(dt_array) // self.substeps_per_interval
                nodes = np.empty(
                    (n_eff * bridge_strata, contractual_steps + 1),
                    dtype=float,
                )
                log_spot_factor_loadings = np.zeros(
                    (n_eff * bridge_strata, contractual_steps + 1), dtype=float
                )
                factor_loading = np.zeros(
                    (n_eff, bridge_strata), dtype=float
                )
            var = np.full(n_eff, max(float(p.v0), 0.0), dtype=float)
            nodes[:, 0] = np.exp(log_s).reshape(-1)
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
                if conditioning is not None:
                    raise ValidationError(
                        "spot-factor conditioning is implemented only for Heston QE/QE-M"
                    )
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
                    base_increment = (
                        (drift - 0.5 * v_bar) * dt + corr - K0 + K0_star
                    )
                else:
                    base_increment = (drift - 0.5 * v_bar) * dt + corr
                if conditioning is None:
                    log_s = (
                        log_s
                        + base_increment
                        + np.sqrt(v_bar) * sqrt_dt * diff_coef * z_ind[:, i]
                    )
                else:
                    log_s = (
                        log_s
                        + base_increment[:, None]
                        + np.sqrt(v_bar)[:, None]
                        * sqrt_dt
                        * diff_coef
                        * residual_z[:, :, i]
                    )
                    factor_loading += (
                        np.sqrt(v_bar)[:, None]
                        * sqrt_dt
                        * diff_coef
                        * spot_loadings[i]
                    )
                var = v_np
                if conditioning is None:
                    nodes[:, i + 1] = np.exp(log_s)
                elif (i + 1) % self.substeps_per_interval == 0:
                    contractual_index = (i + 1) // self.substeps_per_interval
                    nodes[:, contractual_index] = np.exp(log_s).reshape(-1)
                    log_spot_factor_loadings[:, contractual_index] = (
                        factor_loading.reshape(-1)
                    )
            self._record_step_log_variance(h2)
            if conditioning is None:
                return nodes
            aux = {
                "affine_spot_factor": "standard_normal",
                "log_spot_factor_loadings": log_spot_factor_loadings,
                "_paths_are_contractual": True,
            }
            if bridge_strata > 1:
                aux["conditional_outer_group_size"] = bridge_strata
            return nodes, aux

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
        if (
            (
                self.rqmc_frozen_leverage_conditional_control
                or self.rqmc_spot_bridge_strata > 1
            )
            and leverage_surface is None
        ):
            raise ValidationError(
                "frozen-leverage control and bridge stratification require a "
                "precomputed LeverageSurface"
            )
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
        if (
            self.rqmc_affine_spot_factor
            or self.rqmc_heston_conditional_control
            or self.rqmc_qe_draw_provider is not None
        ):
            raise ValidationError(
                "spot-factor conditioning is implemented only for QE/QE-M engines"
            )
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
        if self.rqmc_affine_spot_factor:
            raise ValidationError(
                "exact affine spot conditioning is not valid for state-dependent "
                "SLV leverage; use rqmc_heston_conditional_control"
            )
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
                if self.rqmc_qe_draw_provider is not None:
                    return self.rqmc_qe_draw_provider.draws(
                        n_paths=n_eff,
                        dt_array=dt_array,
                        batch_id=batch_id,
                    )
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
            conditioning = None
            conditional_spot_factors = None
            control_only = bool(self.rqmc_conditional_control_only)
            # The conditional-control paths below are recorded on the CONTRACTUAL
            # grid with one row per (path, stratum), a shape the continuous-KI
            # bridge's (paths, steps) variance buffer cannot describe.  Only the
            # plain path records it; the conditional one leaves it None so
            # _ki_bridge_step_log_variance raises rather than falling back to a
            # variance these paths never accumulated.
            h2 = None
            if self.rqmc_heston_conditional_control:
                if self.rqmc_spot_strata > 1 and self.leverage_surface is None:
                    raise ValidationError(
                        "rqmc_spot_strata > 1 requires a precomputed "
                        "LeverageSurface"
                    )
                bridge_strata = int(self.rqmc_spot_bridge_strata)
                if bridge_strata > 1:
                    residual_z, spot_loadings = (
                        self._bridge_stratified_spot_normals(z_ind, dt_array)
                    )
                else:
                    conditioning = self._conditioned_spot_normals(z_ind, dt_array)
                    if conditioning is None:  # defensive: flag and helper must agree
                        raise PricingError(
                            "missing Heston conditional-control factors"
                        )
                    residual_base, spot_loadings = conditioning
                    residual_z = residual_base[:, None, :]
                conditioning = residual_z, spot_loadings
                conditional_outer_group_size = bridge_strata
                contractual_steps = len(dt_array) // self.substeps_per_interval
                control_base_paths = np.empty(
                    (
                        n_eff * conditional_outer_group_size,
                        contractual_steps + 1,
                    ),
                    dtype=float,
                )
                control_loadings = np.zeros_like(control_base_paths)
                control_log_base = np.full(
                    (n_eff, conditional_outer_group_size),
                    np.log(max(float(S), 1e-12)),
                    dtype=float,
                )
                control_loading = np.zeros_like(control_log_base)
                if control_only:
                    nodes = control_base_paths
                    heston_base_paths = np.empty_like(control_base_paths)
                    heston_loadings = np.zeros_like(control_base_paths)
                    heston_log_base = np.full_like(
                        control_log_base,
                        np.log(max(float(S), 1e-12)),
                    )
                    heston_loading = np.zeros_like(control_log_base)
                    heston_unit_leverage = np.ones_like(control_log_base)
                    log_s = None
                else:
                    terminal_uniform = ndtr(
                        np.asarray(z_ind[:, 0], dtype=float)
                    )
                    strata = int(self.rqmc_spot_strata)
                    primary_uniforms = (
                        terminal_uniform[:, None]
                        + np.arange(strata, dtype=float)[None, :]
                    ) / float(strata)
                    if self.rqmc_spot_antithetic:
                        antithetic_uniforms = (
                            (1.0 - terminal_uniform[:, None])
                            + np.arange(strata, dtype=float)[None, :]
                        ) / float(strata)
                        stratified_uniforms = np.concatenate(
                            (primary_uniforms, antithetic_uniforms), axis=1
                        )
                    else:
                        stratified_uniforms = primary_uniforms
                    conditional_spot_factors = ndtri(
                        np.clip(stratified_uniforms, 1e-12, 1.0 - 1e-12)
                    )
                    conditional_group_size = int(
                        conditional_spot_factors.shape[1]
                    )
                    nodes = np.empty(
                        (
                            n_eff
                            * conditional_outer_group_size
                            * conditional_group_size,
                            contractual_steps + 1,
                        ),
                        dtype=float,
                    )
                    control_paths = np.empty_like(nodes)
                    log_s = np.full(
                        (
                            n_eff,
                            conditional_outer_group_size,
                            conditional_group_size,
                        ),
                        np.log(max(float(S), 1e-12)),
                        dtype=float,
                    )
            else:
                if self.rqmc_qe_draw_provider is not None:
                    z_ind = apply_brownian_bridge(
                        z_ind, np.cumsum(dt_array)
                    ) / np.sqrt(dt_array)[None, :]
                nodes = np.empty((n_eff, len(dt_array) + 1), dtype=float)
                h2 = self._new_step_log_variance(n_eff, len(dt_array))
                log_s = np.full(
                    n_eff, np.log(max(float(S), 1e-12)), dtype=float
                )
            var = np.full(n_eff, max(float(p.v0), 0.0), dtype=float)
            if control_only:
                nodes[:, 0] = float(S)
            else:
                nodes[:, 0] = np.exp(log_s).reshape(-1)
            if conditioning is not None:
                control_base_paths[:, 0] = float(S)
                if control_only:
                    heston_base_paths[:, 0] = float(S)
                else:
                    control_paths[:, 0] = float(S)
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
                if not control_only:
                    spot = np.exp(log_s)
                    if self.leverage_surface is None:
                        boundaries, means = bin_conditional(
                            spot, var, self.num_bins, self.bin_method
                        )
                        econd = np.maximum(
                            eval_binned(spot, boundaries, means), _VAR_FLOOR
                        )
                        sigma_lv = np.asarray(
                            lv.local_vol(spot, t), dtype=float
                        )
                        leverage = np.clip(
                            sigma_lv / np.sqrt(econd), lo, hi
                        )
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
                # Shared QE variance step: identical arithmetic to the block it
                # replaced, routed through the Numba accelerator when installed
                # (bit-identical either way -- see quantark/montecarlo/qe_kernels.py).
                qe_step = qe_variance_step(
                    var,
                    z_var[:, i],
                    u_var[:, i],
                    kappa=p.kappa,
                    theta=p.theta,
                    sigma2=sigma_eff2,
                    dt=dt,
                    psi_c=psi_c,
                    kmin=_QE_KMIN,
                )
                m = qe_step.m
                a = qe_step.a
                b = qe_step.b
                beta = qe_step.beta
                prob_zero = qe_step.prob_zero
                v_np = qe_step.v_np
                v_bar = qe_step.v_bar
                quad_mask = qe_step.quad_mask
                if h2 is not None:
                    # Quadratic variation of log S over this fine step. The SLV
                    # diffusion is L(S,t)*sqrt(v), so the interval accumulates
                    # L^2 * v_bar * dt with the scheme's own trapezoidal v_bar.
                    h2[:, i] = leverage * leverage * v_bar * dt
                if not control_only:
                    if conditioning is None:
                        v_bar_p = v_bar
                        var_p = var
                        a_p = a
                        b_p = b
                        beta_p = beta
                        prob_zero_p = prob_zero
                        quad_mask_p = quad_mask
                        variance_residual_p = (
                            v_np
                            - var
                            - p.kappa * (p.theta - v_bar) * dt
                        )
                        z_spot = z_ind[:, i]
                    else:
                        v_bar_p = v_bar[:, None, None]
                        var_p = var[:, None, None]
                        a_p = a[:, None, None]
                        b_p = b[:, None, None]
                        beta_p = beta[:, None, None]
                        prob_zero_p = prob_zero[:, None, None]
                        quad_mask_p = quad_mask[:, None, None]
                        variance_residual_p = (
                            v_np
                            - var
                            - p.kappa * (p.theta - v_bar) * dt
                        )[:, None, None]
                        z_spot = (
                            residual_z[:, :, i, None]
                            + spot_loadings[i]
                            * conditional_spot_factors[:, None, :]
                        )
                    leverage2 = leverage * leverage
                    if deterministic_vol:
                        corr = 0.0
                    else:
                        corr = (
                            leverage
                            * (rho / sigma_eff)
                            * variance_residual_p
                        )

                    if martingale:
                        ros = leverage * rho / sigma_eff
                        K3 = 0.5 * leverage2 * (1.0 - rho * rho) * dt
                        K1 = (
                            0.5
                            * dt
                            * (p.kappa * ros - 0.5 * leverage2)
                            - ros
                        )
                        K2 = (
                            0.5
                            * dt
                            * (p.kappa * ros - 0.5 * leverage2)
                            + ros
                        )
                        A = K2 + 0.5 * K3
                        denom_q = 1.0 - 2.0 * A * a_p
                        denom_e = beta_p - A
                        bad = (quad_mask_p & (denom_q <= 0.0)) | (
                            ~quad_mask_p & (denom_e <= 0.0)
                        )
                        if np.any(bad):
                            raise NumericalError(
                                "SLV QE-M martingale MGF is undefined at these "
                                "parameters (A outside the QE transition MGF "
                                "domain); tighten dt or disable "
                                "martingale_correction"
                            )
                        safe_q = np.where(denom_q > 0.0, denom_q, 1.0)
                        safe_e = np.where(denom_e > 0.0, denom_e, 1.0)
                        m_quad = (
                            np.exp(A * a_p * b_p * b_p / safe_q)
                            / np.sqrt(safe_q)
                        )
                        m_exp = (
                            prob_zero_p
                            + (1.0 - prob_zero_p) * beta_p / safe_e
                        )
                        ln_M = np.log(np.where(quad_mask_p, m_quad, m_exp))
                        K0 = -ros * p.kappa * p.theta * dt
                        K0_star = -ln_M - (K1 + 0.5 * K3) * var_p
                        log_s = (
                            log_s
                            + (drift - 0.5 * leverage2 * v_bar_p) * dt
                            + corr
                            - K0
                            + K0_star
                            + leverage
                            * np.sqrt(v_bar_p)
                            * sqrt_dt
                            * diff_coef
                            * z_spot
                        )
                    else:
                        log_s = (
                            log_s
                            + (drift - 0.5 * leverage2 * v_bar_p) * dt
                            + corr
                            + leverage
                            * np.sqrt(v_bar_p)
                            * sqrt_dt
                            * diff_coef
                            * z_spot
                        )

                if conditioning is not None:
                    if self.rqmc_frozen_leverage_conditional_control:
                        control_leverage = np.asarray(
                            self.leverage_surface.leverage(
                                np.exp(control_log_base), t
                            ),
                            dtype=float,
                        )
                        if (
                            control_leverage.shape
                            != (n_eff, conditional_outer_group_size)
                            or not np.all(np.isfinite(control_leverage))
                        ):
                            raise ValidationError(
                                "frozen-leverage control returned invalid values"
                            )
                    else:
                        control_leverage = np.ones_like(control_log_base)
                    advance_inputs = {
                        "var": var,
                        "v_np": v_np,
                        "v_bar": v_bar,
                        "a": a,
                        "b": b,
                        "beta": beta,
                        "prob_zero": prob_zero,
                        "quad_mask": quad_mask,
                        "drift": drift,
                        "dt": float(dt),
                        "rho": rho,
                        "sigma_eff": sigma_eff,
                        "kappa": float(p.kappa),
                        "theta": float(p.theta),
                        "sqrt_dt": sqrt_dt,
                        "diff_coef": diff_coef,
                        "residual_z": residual_z[:, :, i],
                        "spot_loading": float(spot_loadings[i]),
                        "martingale": martingale,
                        "deterministic_vol": deterministic_vol,
                    }
                    control_log_base, control_loading = (
                        _advance_conditional_affine_spot(
                            control_log_base,
                            control_loading,
                            control_leverage,
                            **advance_inputs,
                        )
                    )
                    if control_only:
                        heston_log_base, heston_loading = (
                            _advance_conditional_affine_spot(
                                heston_log_base,
                                heston_loading,
                                heston_unit_leverage,
                                **advance_inputs,
                            )
                        )
                var = v_np
                if conditioning is None:
                    nodes[:, i + 1] = np.exp(log_s)
                elif (i + 1) % self.substeps_per_interval == 0:
                    contractual_index = (i + 1) // self.substeps_per_interval
                    control_base_paths[:, contractual_index] = np.exp(
                        control_log_base
                    ).reshape(-1)
                    control_loadings[:, contractual_index] = (
                        control_loading.reshape(-1)
                    )
                    if control_only:
                        heston_base_paths[:, contractual_index] = np.exp(
                            heston_log_base
                        ).reshape(-1)
                        heston_loadings[:, contractual_index] = (
                            heston_loading.reshape(-1)
                        )
                    else:
                        nodes[:, contractual_index] = np.exp(log_s).reshape(-1)
                        control_paths[:, contractual_index] = np.exp(
                            control_log_base[:, :, None]
                            + control_loading[:, :, None]
                            * conditional_spot_factors[:, None, :]
                        ).reshape(-1)
                t += float(dt)
            self._record_step_log_variance(h2)
            if conditioning is None:
                return nodes
            if control_only:
                return nodes, {
                    "affine_spot_factor": "standard_normal",
                    "log_spot_factor_loadings": control_loadings,
                    "heston_conditional_control": True,
                    "control_base_paths": heston_base_paths,
                    "control_log_spot_factor_loadings": heston_loadings,
                    "conditional_outer_group_size": (
                        conditional_outer_group_size
                    ),
                    "_paths_are_contractual": True,
                }
            return nodes, {
                "heston_conditional_control": True,
                "control_paths": control_paths,
                "control_base_paths": control_base_paths,
                "control_log_spot_factor_loadings": control_loadings,
                "conditional_group_size": conditional_group_size,
                "conditional_outer_group_size": conditional_outer_group_size,
                "_paths_are_contractual": True,
            }

        return self._make_path_generator(simulate, n_eff, batch_id)
