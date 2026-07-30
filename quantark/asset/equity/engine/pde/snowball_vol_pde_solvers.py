"""Snowball PDE solvers under Local Vol / Heston / Heston-SLV processes."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Optional

import numpy as np

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.engine.pde.base_pde_solver import StepCoefficients
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.pde.vol_continuous_ki import (
    Heston2DBarrierCrossingMixin,
    LocalVolBarrierCrossingMixin,
    SLVBarrierLeverageMixin,
)
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import GridVolSurface
from quantark.priceenv import PricingEnvironment, TermMarketContext
from quantark.priceenv.term_sampling import TermCoefficients
from quantark.util.enum import ObservationType
from quantark.util.enum.engine_enums import ADIScheme, EngineType
from quantark.util.exceptions import PricingError, ValidationError
from quantark.util.numerical import is_close, is_zero
from quantark.volmodels.adi_core import HestonSLVADICore
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.localvol import LocalVolSurface, build_dupire_local_vol
from quantark.volmodels.slv.leverage import LeverageSurface


def event_damped_step_keys(params, event_maps, n_t: int):
    """Integer step keys that run as damped Douglas ADI restarts.

    After each discrete event key the next ``params.event_rannacher_steps``
    steps run at ``params.event_theta`` — the 2D counterpart of the 1D
    ``BackwardOperator.theta_by_step`` event schedule, mirroring its gating
    exactly: ``use_rannacher=False`` is the master off-switch, and events at
    the maturity node (tau key 0) are excluded because the terminal Rannacher
    start-up owns the payoff discontinuity (1D: ``0 < idx < num_t - 1``; the
    valuation-date key ``n_t`` is excluded by the ``<= n_t`` clip below).
    Continuous KI has no discrete keys, so it never triggers damping
    (continuous-barrier treatment). Returns ``None`` when damping is disabled.
    """
    if not bool(getattr(params, "use_rannacher", True)):
        return None
    if not bool(getattr(params, "rannacher_at_events", True)):
        return None
    ers = int(getattr(params, "event_rannacher_steps", 0) or 0)
    if ers <= 0:
        return None
    event_keys: set[int] = set()
    for stream in ("ko", "ki", "coupon"):
        stream_map = event_maps.get(stream)
        if stream_map:
            event_keys |= set(stream_map.keys())
    keys: set[int] = set()
    for k in event_keys:
        if k == 0:
            continue
        for step in range(1, ers + 1):
            if k + step <= n_t:
                keys.add(k + step)
    return keys or None


class LocalVolSnowballPDESolver(LocalVolBarrierCrossingMixin, SnowballPDESolver):

    """Two-surface Snowball PDE with Dupire local volatility on the S grid."""

    engine_type = EngineType.PDE
    _solver_name = "LocalVolSnowballPDESolver"

    def __init__(
        self,
        params: Optional[PDEParams] = None,
        local_vol_surface: Optional[LocalVolSurface] = None,
        enable_profiling: bool = False,
    ):
        super().__init__(params=params, enable_profiling=enable_profiling)
        self._prebuilt = local_vol_surface
        self._active_lv_surface: Optional[LocalVolSurface] = None
        self._active_s_vec: Optional[np.ndarray] = None

    def _build_surface(self, env: PricingEnvironment) -> LocalVolSurface:
        if self._prebuilt is not None:
            return self._prebuilt
        if not isinstance(env.vol_surface, GridVolSurface):
            raise PricingError(
                "LocalVolSnowballPDESolver needs a GridVolSurface or a prebuilt LocalVolSurface"
            )
        return build_dupire_local_vol(
            env.vol_surface,
            spot=env.spot,
            rate_curve=env.rate_curve,
            div_yield=env.get_div_yield,
        )

    def _with_surface(self, env: PricingEnvironment, fn):
        previous = self._active_lv_surface
        self._active_lv_surface = self._build_surface(env)
        try:
            return fn()
        finally:
            self._active_lv_surface = previous

    def price(self, product: BaseEquityProduct, pricing_env: PricingEnvironment) -> float:
        return self._with_surface(
            pricing_env,
            lambda: SnowballPDESolver.price(self, product, pricing_env),
        )

    def calculate_greeks(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> Dict[str, float]:
        return self._with_surface(
            pricing_env,
            lambda: SnowballPDESolver.calculate_greeks(self, product, pricing_env),
        )

    def calculate_event_stats(self, product, pricing_env, **kwargs):
        return self._with_surface(
            pricing_env,
            lambda: SnowballPDESolver.calculate_event_stats(
                self, product, pricing_env, **kwargs
            ),
        )

    def _session_outputs(self, product, pricing_env, **kwargs):
        return self._with_surface(
            pricing_env,
            lambda: SnowballPDESolver._session_outputs(
                self, product, pricing_env, **kwargs
            ),
        )

    def _build_grids(self, product, pricing_env, spot, sigma, tau, r, q):
        result = super()._build_grids(product, pricing_env, spot, sigma, tau, r, q)
        self._active_s_vec = result[1]
        return result

    def _build_step_coefficients(
        self,
        pricing_env: PricingEnvironment,
        ref_strike: float,
        t_vec: np.ndarray,
        dx_vec: np.ndarray,
        num_x: int,
    ) -> StepCoefficients:
        if self._active_lv_surface is None or self._active_s_vec is None:
            raise PricingError("Local-vol surface is not initialized for this solve")

        tc = TermCoefficients.from_env(
            pricing_env, np.asarray(t_vec, dtype=float), ref_strike=float(ref_strike)
        )
        s_vec = np.asarray(self._active_s_vec, dtype=float)
        dt = np.diff(np.asarray(t_vec, dtype=float))
        lcu_sets = []
        for j, dt_j in enumerate(dt):
            t_mid = float(t_vec[j] + 0.5 * dt_j)
            sigma_vec = np.asarray(self._active_lv_surface.local_vol(s_vec, t_mid), dtype=float)
            lcu_sets.append(
                self._local_vol_coefficients(
                    float(tc.fwd_rates[j]),
                    float(tc.fwd_carry[j]),
                    sigma_vec,
                    dx_vec,
                    num_x,
                )
            )
        return StepCoefficients(
            lcu_sets=lcu_sets,
            set_index=np.arange(len(lcu_sets), dtype=int),
            n_unique=len(lcu_sets),
        )

    def _flat_exact_step_coefficients(self, sc, r, q, sigma, dx_vec, num_x):
        return sc

    @staticmethod
    def _local_vol_coefficients(r, q, sigma_vec, dx_vec, num_x):
        sigma_vec = np.asarray(sigma_vec, dtype=float)
        if sigma_vec.shape[0] != num_x:
            raise ValidationError("local-vol vector length must match PDE spatial grid")
        if not np.all(np.isfinite(sigma_vec)) or np.any(sigma_vec <= 0.0):
            raise ValidationError("local-vol surface returned non-positive or non-finite vols")

        diff = 0.5 * sigma_vec * sigma_vec
        drift = float(r) - float(q) - diff
        dx_vec = np.asarray(dx_vec, dtype=float)

        if is_close(float(np.max(dx_vec)), float(np.min(dx_vec))):
            dx = float(dx_vec[0])
            l = diff / (dx * dx) - drift / (2.0 * dx)
            c = -2.0 * diff / (dx * dx) - float(r)
            u = diff / (dx * dx) + drift / (2.0 * dx)
            return l, c, u

        h_m, h_p = dx_vec[:-1], dx_vec[1:]
        h_sum, h_prod = h_m + h_p, h_m * h_p
        d_i = diff[1:-1]
        mu_i = drift[1:-1]
        l = np.zeros(num_x)
        c = np.zeros(num_x)
        u = np.zeros(num_x)
        l[1:-1] = 2.0 * d_i / (h_m * h_sum) - mu_i * h_p / (h_m * h_sum)
        c[1:-1] = -2.0 * d_i / h_prod + mu_i * (h_p - h_m) / h_prod - float(r)
        u[1:-1] = 2.0 * d_i / (h_p * h_sum) + mu_i * h_m / (h_p * h_sum)
        l[0], c[0], u[0] = l[1], c[1], u[1]
        l[-1], c[-1], u[-1] = l[-2], c[-2], u[-2]
        return l, c, u



class _Heston2DSnowballPDEBase(Heston2DBarrierCrossingMixin, SnowballPDESolver):

    """Shared 2-D (log-spot, variance) ADI machinery for the Snowball solvers.

    ``v0_boundary`` defaults to ``"degenerate_pde"``, diverging from the ADI
    core's own ``"neumann"`` default.  Concentrated grids use a regime-aware
    variance axis: power grading near ``v=0`` in ordinary/low-Feller regimes,
    and a path-focused grid with exact theta/v0 nodes when vol-of-vol collapses.
    ``v_grid_power=0`` remains an explicit legacy cross-check; uniform grids
    resolve to that ungraded setting because grading requires non-uniform
    stencils.  The variance drift defaults to an adaptive M-matrix-preserving
    stencil shared by both halves of the ADI split.
    """

    engine_type = EngineType.PDE
    _solver_name = "Heston2DSnowballPDESolver"
    DEFAULT_V_GRID_POWER = 2.5
    DEFAULT_BARRIER_GREEK_STEPS_PER_TICK = 8
    PRODUCTION_BARRIER_GREEK_STEPS_PER_TICK = 16
    PRODUCTION_GREEK_MIN_N_X = 300
    PRODUCTION_GREEK_MIN_N_V = 135
    PRODUCTION_GREEK_MIN_STEPS_PER_YEAR = 1600
    PRODUCTION_BARRIER_GREEK_MIN_N_X = 600
    DENSE_KI_EVENTS_PER_YEAR = 120.0

    def __init__(
        self,
        model_params: HestonParams,
        params: Optional[PDEParams] = None,
        n_x: int = 200,
        n_v: int = 100,
        n_t: int = 100,
        scheme: ADIScheme | str = ADIScheme.CRAIG_SNEYD,
        grid_style: str = "concentrated",
        grid_focus: str = "auto",
        pin_critical_spots: bool = False,
        v0_boundary: str = "degenerate_pde",
        v_grid_power: Optional[float] = None,
        variance_grid_mode: str = "auto",
        v_drift_scheme: str = "adaptive_upwind",
        barrier_greek_steps_per_tick: int = DEFAULT_BARRIER_GREEK_STEPS_PER_TICK,
        greek_min_n_x: int = 0,
        greek_min_n_v: int = 0,
        greek_min_steps_per_year: int = 0,
        barrier_greek_min_n_x: int = 0,
    ):
        if not isinstance(model_params, HestonParams):
            raise ValidationError("model_params must be a HestonParams")
        super().__init__(params=params or PDEParams())
        try:
            scheme = ADIScheme[scheme.upper()] if isinstance(scheme, str) else scheme
        except KeyError:
            raise ValidationError(f"unknown ADI scheme: {scheme}")
        if scheme == ADIScheme.MCS:
            raise ValidationError("MCS is not implemented for the Heston/SLV Snowball PDE")
        if grid_focus not in {"auto", "ko", "ki", "strike", "spot"}:
            raise ValidationError(
                "grid_focus must be one of: auto, ko, ki, strike, spot"
            )
        if grid_style not in {"uniform", "concentrated"}:
            raise ValidationError(
                "grid_style must be 'uniform' or 'concentrated'"
            )
        if v0_boundary not in ("neumann", "degenerate_pde"):
            raise ValidationError(
                "v0_boundary must be 'neumann' or 'degenerate_pde'"
            )
        if variance_grid_mode not in {"legacy", "power", "path_focused", "auto"}:
            raise ValidationError(
                "variance_grid_mode must be one of: legacy, power, "
                "path_focused, auto"
            )
        if v_drift_scheme not in {
            "centered", "adaptive_upwind", "semi_lagrangian", "auto"
        }:
            raise ValidationError(
                "v_drift_scheme must be 'centered', 'adaptive_upwind', "
                "'semi_lagrangian', or 'auto'"
            )
        if (
            isinstance(barrier_greek_steps_per_tick, bool)
            or not isinstance(barrier_greek_steps_per_tick, (int, np.integer))
            or int(barrier_greek_steps_per_tick) < 0
        ):
            raise ValidationError(
                "barrier_greek_steps_per_tick must be a non-negative integer"
            )
        for name, value in (
            ("greek_min_n_x", greek_min_n_x),
            ("greek_min_n_v", greek_min_n_v),
            ("greek_min_steps_per_year", greek_min_steps_per_year),
            ("barrier_greek_min_n_x", barrier_greek_min_n_x),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or int(value) < 0
            ):
                raise ValidationError(f"{name} must be a non-negative integer")
        explicit_v_grid_power = v_grid_power is not None
        if explicit_v_grid_power:
            try:
                resolved_v_grid_power = float(v_grid_power)
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    "v_grid_power must be a finite value >= 1, or 0 to disable"
                ) from exc
        else:
            resolved_v_grid_power = 0.0
        if (
            not np.isfinite(resolved_v_grid_power)
            or resolved_v_grid_power < 0.0
            or 0.0 < resolved_v_grid_power < 1.0
        ):
            raise ValidationError(
                "v_grid_power must be a finite value >= 1, or 0 to disable"
            )
        if grid_style == "uniform" and resolved_v_grid_power > 0.0:
            raise ValidationError(
                "v_grid_power requires grid_style='concentrated'"
            )
        if grid_style == "uniform" and variance_grid_mode in {
            "power",
            "path_focused",
        }:
            raise ValidationError(
                "variance_grid_mode power/path_focused requires "
                "grid_style='concentrated'"
            )

        resolved_variance_grid_mode = str(variance_grid_mode)
        if grid_style == "uniform":
            resolved_variance_grid_mode = "legacy"
            resolved_v_grid_power = 0.0
        elif explicit_v_grid_power and resolved_v_grid_power > 0.0:
            # Preserve the long-standing explicit power control.
            resolved_variance_grid_mode = "power"
        elif (
            explicit_v_grid_power
            and resolved_v_grid_power == 0.0
            and resolved_variance_grid_mode == "auto"
        ):
            # Preserve ``v_grid_power=0`` as an explicit legacy opt-out.
            resolved_variance_grid_mode = "legacy"
        elif resolved_variance_grid_mode == "auto":
            sigma_eff = float(getattr(self, "eta", 1.0)) * float(model_params.sigma)
            if sigma_eff == 0.0:
                feller_ratio = float("inf")
            else:
                feller_ratio = (
                    2.0
                    * float(model_params.kappa)
                    * float(model_params.theta)
                    / (sigma_eff * sigma_eff)
                )
            state_separation = abs(
                float(model_params.v0) - float(model_params.theta)
            ) / max(float(model_params.v0), float(model_params.theta), 1e-12)
            if feller_ratio >= 25.0 and (
                state_separation >= 0.10 or abs(sigma_eff) <= 0.01
            ):
                resolved_variance_grid_mode = "path_focused"
            else:
                resolved_variance_grid_mode = "power"
                resolved_v_grid_power = self.DEFAULT_V_GRID_POWER
        elif resolved_variance_grid_mode == "power" and resolved_v_grid_power == 0.0:
            resolved_v_grid_power = self.DEFAULT_V_GRID_POWER
        self.model_params = model_params
        self.n_x = int(n_x)
        self.n_v = int(n_v)
        self.n_t = int(n_t)
        self.scheme = scheme
        self.grid_style = grid_style
        self.grid_focus = grid_focus
        self.pin_critical_spots = bool(pin_critical_spots)
        self.v0_boundary = str(v0_boundary)
        self.v_grid_power = float(resolved_v_grid_power)
        self.variance_grid_mode = resolved_variance_grid_mode
        self.v_drift_scheme = str(v_drift_scheme)
        self.barrier_greek_steps_per_tick = int(
            barrier_greek_steps_per_tick
        )
        self.greek_min_n_x = int(greek_min_n_x)
        self.greek_min_n_v = int(greek_min_n_v)
        self.greek_min_steps_per_year = int(greek_min_steps_per_year)
        self.barrier_greek_min_n_x = int(barrier_greek_min_n_x)

    def representative_vol(self, product, pricing_env) -> float:
        # sqrt(var_eff) with var_eff ported VERBATIM from the adi_core x-width
        # computation (max of theta, v0, 0.25*sig_eff^2, 0.04; eta=1 here).
        p = self.model_params
        var_eff = max(p.theta, p.v0, 0.25 * (p.sigma * p.sigma), 0.04)
        return float(np.sqrt(var_eff))

    # 2D S-axis binder + frozen layout, SEPARATE from the base 1D slots: the
    # x-axis is bound at (points=n_x, num_std=8) — the ADI solve
    # configuration — while inherited 1D sub-paths (event-stats machinery)
    # keep the base-config _frozen_base_layout.
    _layer_binder_2d = None
    _frozen_x_layout = None

    def _layer_binder(self):
        binder = self._layer_binder_2d
        if binder is None:
            from quantark.asset.equity.engine.pde.grid import GridBinder, GridConfig

            binder = GridBinder(
                "standard",
                GridConfig(points=int(self.n_x), num_std=8.0),
                cache_enabled=self._is_cache_enabled(),
            )
            self._layer_binder_2d = binder
        return binder

    def _layer_x_nodes(self, product: SnowballOption, env: PricingEnvironment, T: float):
        """S-axis nodes from the declarative grid layer (spec 4.6, Phase 3).

        One spatial builder for 1D and 2D: spot/strike/barrier concentration,
        bump-stable, shareable. num_std=8 preserves the certified adi_core
        domain width (8*sqrt(var_eff*T)); the time axis stays the solver's
        uniform ADI grid (event-aligned 2D time is a recorded deferral).
        Bump clones route through ``resolve_bound_layout`` so the frozen
        S-axis is reused by identity (and misuse fails closed) exactly like
        the 1D path.
        """
        from quantark.asset.equity.engine.pde.grid import resolve_bound_layout

        market = self.market_snapshot(product, env)
        request = self.grid_request(product, market, float(T))
        layout = resolve_bound_layout(
            self._layer_binder(), self._frozen_x_layout, request, market
        )
        return layout.spatial.x

    def _bump_transient_attrs(self) -> tuple:
        return super()._bump_transient_attrs() + (
            "_layer_binder_2d",
            "_frozen_x_layout",
        )

    def create_bump_context(self, product, pricing_env):
        """Freeze BOTH grid slots: the base 1D layout (inherited machinery)
        and the 2D S-axis layout at the ADI solve configuration."""
        clone = super().create_bump_context(product, pricing_env)
        if clone is self:
            return clone
        tau = product.get_maturity(pricing_env)
        market = clone.market_snapshot(product, pricing_env)
        request = clone.grid_request(product, market, tau)
        clone._frozen_x_layout = clone._layer_binder().bind(request, market)
        return clone

    def _make_core(self, product: SnowballOption, env: PricingEnvironment, T: float):
        t_grid = np.linspace(0.0, float(T), self.n_t + 1)
        market = TermMarketContext.from_env(
            env,
            t_grid,
            ref_strike=None,
        )
        x_nodes = self._layer_x_nodes(product, env, T)
        return HestonSLVADICore(
            float(env.spot),
            float(product.strike),
            T,
            float(env.get_rate(T)),
            float(env.get_div_yield(T)),
            self.model_params,
            self.n_x,
            self.n_v,
            self.n_t,
            market_context=market,
            leverage=None,
            eta=1.0,
            grid_style=self.grid_style,
            v0_boundary=self.v0_boundary,
            v_grid_power=self.v_grid_power,
            variance_grid_mode=self.variance_grid_mode,
            v_drift_scheme=self.v_drift_scheme,
            barrier_concentrate=self._grid_concentration_spot(product, env),
            critical_spots=(
                self._grid_critical_spots(product, env)
                if self.pin_critical_spots
                else None
            ),
            x_nodes=x_nodes,
        )

    def _primary_barrier(self, product: SnowballOption) -> float:
        ko_barriers = self._positive_levels(product.barrier_config.ko_barrier)
        if not ko_barriers:
            return float(product.strike)
        return float(min(ko_barriers) if product.is_reverse else max(ko_barriers))

    @staticmethod
    def _positive_levels(value) -> list[float]:
        if value is None:
            return []
        if isinstance(value, np.ndarray):
            raw_values = value.ravel()
        elif isinstance(value, (list, tuple)):
            raw_values = value
        else:
            raw_values = [value]
        levels = []
        for raw in raw_values:
            try:
                level = float(raw)
            except (TypeError, ValueError):
                continue
            if np.isfinite(level) and level > 0.0:
                levels.append(level)
        return levels

    @staticmethod
    def _dedupe_levels(levels: list[float], tol: float = 1e-10) -> list[float]:
        out: list[float] = []
        for level in sorted(levels):
            if not out or abs(level - out[-1]) > tol:
                out.append(float(level))
        return out

    def _primary_ki_barrier(self, product: SnowballOption) -> Optional[float]:
        ki_barriers = self._positive_levels(product.barrier_config.ki_barrier)
        if not ki_barriers:
            return None
        return float(max(ki_barriers) if product.is_reverse else min(ki_barriers))

    def _auto_grid_focus(self, product: SnowballOption) -> str:
        # The KO level is still pinned as a critical point. The concentration
        # center is set around the KI transition when it exists because the
        # two-surface Snowball value is most grid-sensitive around that state
        # switch and the terminal downside kink.
        if product.has_ki_barrier:
            return "ki"
        return "strike"

    def _grid_concentration_spot(
        self, product: SnowballOption, env: PricingEnvironment
    ) -> float:
        focus = self._auto_grid_focus(product) if self.grid_focus == "auto" else self.grid_focus
        if focus == "ko":
            return self._primary_barrier(product)
        if focus == "ki":
            ki_barrier = self._primary_ki_barrier(product)
            return float(ki_barrier if ki_barrier is not None else product.strike)
        if focus == "spot":
            return float(env.spot)
        return float(product.strike)

    def _grid_critical_spots(
        self, product: SnowballOption, env: PricingEnvironment
    ) -> list[float]:
        levels = [
            float(env.spot),
            float(product.initial_price),
            float(product.strike),
        ]
        levels.extend(self._positive_levels(product.barrier_config.ko_barrier))
        levels.extend(self._positive_levels(product.barrier_config.ki_barrier))

        call_strike = getattr(product.payoff_config, "call_strike", None)
        if call_strike is not None:
            levels.extend(self._positive_levels(call_strike))
        airbag_barrier = getattr(product.airbag_config, "airbag_barrier", None)
        if airbag_barrier is not None:
            levels.extend(self._positive_levels(airbag_barrier))
        airbag_strike = getattr(product.airbag_config, "airbag_strike", None)
        if airbag_strike is not None:
            levels.extend(self._positive_levels(airbag_strike))
        return self._dedupe_levels(levels)

    def calculate_event_stats(
        self,
        product,
        pricing_env,
        *,
        npv: Optional[float] = None,
        streams: Optional[frozenset] = None,
    ):
        if not isinstance(product, SnowballOption):
            return None
        if pricing_env is None:
            return None
        pde_pv = float(npv) if npv is not None else float(self.price(product, pricing_env))
        return SnowballPDESolver.calculate_event_stats(
            self,
            product,
            pricing_env,
            npv=pde_pv,
            streams=streams,
        )

    def _session_outputs(
        self,
        product,
        pricing_env,
        want_events: bool = False,
        want_grid: bool = False,
        streams: Optional[frozenset] = None,
    ):
        """One 2D value solve; never exposes a 1D grid solution.

        Behavior note (deliberate, 2026-07-16): for an EXPIRED product the
        shared ``price_with_events`` wrapper now clamps the trivial
        distribution to maturity 0.0 (matching 1D semantics) where the old
        2D override passed the negative tau through.
        """
        from quantark.cashleg.event_distribution import EventDistribution

        from quantark.asset.equity.engine.pde.base_pde_solver import (
            PDESessionOutputs,
        )

        npv = float(self.price(product, pricing_env))
        stats = None
        dist = None
        if want_events:
            stats = self.calculate_event_stats(
                product, pricing_env, npv=npv, streams=streams
            )
            if stats is not None:
                dist = EventDistribution.from_autocallable_stats(stats)
        return PDESessionOutputs(
            npv=npv, solution=None, event_stats=stats, event_distribution=dist
        )

    def calculate_greeks(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> Dict[str, float]:
        self._check_product_type(product)
        if pricing_env is None:
            raise ValidationError(
                f"PricingEnvironment is required for {self._solver_name}"
            )
        self._validate_product(product)

        policy = self.greek_time_grid_policy(product, pricing_env)
        risk_engine = self
        if any(
            int(policy[f"resolved_{axis}"]) != int(getattr(self, axis))
            for axis in ("n_x", "n_v", "n_t")
        ):
            risk_engine = self._bump_clone()
            risk_engine.n_x = int(policy["resolved_n_x"])
            risk_engine.n_v = int(policy["resolved_n_v"])
            risk_engine.n_t = int(policy["resolved_n_t"])

        bump_config = risk_engine.params.get_effective_bump_config()
        delta_bump = float(bump_config.spot_bump)
        gamma_bump = float(
            bump_config.gamma_spot_bump
            if bump_config.gamma_spot_bump is not None
            else delta_bump
        )
        spot = float(pricing_env.spot)
        relative_bumps = sorted({delta_bump, gamma_bump})
        shifted_envs = {0.0: pricing_env}
        for bump in relative_bumps:
            for sign in (-1.0, 1.0):
                env = deepcopy(pricing_env)
                env.spot_quote.spot = spot * (1.0 + sign * bump)
                shifted_envs[sign * bump] = env

        # A frozen Heston/SLV operator and payoff surface are independent of
        # the readout spot.  Reuse one V0/V1 march only when every stencil
        # point has the same deterministic valuation-date state.  A t=0 KO,
        # terminal payoff, or continuous/discrete t=0 KI crossing falls back
        # to the general bump-and-reprice implementation.
        signatures = {
            shift: risk_engine._valuation_state_signature(product, env)
            for shift, env in shifted_envs.items()
        }
        if len(set(signatures.values())) != 1:
            return BaseEngine.calculate_greeks(
                risk_engine, product, pricing_env
            )
        signature = next(iter(signatures.values()))
        if signature[0] != "live":
            return BaseEngine.calculate_greeks(
                risk_engine, product, pricing_env
            )

        bump_engine = risk_engine.create_bump_context(product, pricing_env)
        if bump_engine is None:
            bump_engine = risk_engine
        T = float(product.get_maturity(pricing_env))
        core, surface = bump_engine._solve_live_surface(
            product,
            pricing_env,
            T,
            knocked_in=bool(signature[1]),
        )
        for env in shifted_envs.values():
            shifted_spot = float(env.spot)
            if not (core.S_grid[0] <= shifted_spot <= core.S_grid[-1]):
                raise ValidationError(
                    "spot Greek stencil falls outside the frozen Heston/SLV "
                    "Snowball PDE grid"
                )

        prices = {
            shift: float(
                core.interpolate(
                    surface,
                    np.log(float(env.spot)),
                    bump_engine.model_params.v0,
                )
            )
            for shift, env in shifted_envs.items()
        }
        base_price = prices[0.0]
        delta_h = spot * delta_bump
        gamma_h = spot * gamma_bump
        result = {
            "price": base_price,
            "delta": (
                prices[delta_bump] - prices[-delta_bump]
            ) / (2.0 * delta_h),
            "gamma": (
                prices[gamma_bump]
                - 2.0 * base_price
                + prices[-gamma_bump]
            ) / (gamma_h * gamma_h),
        }
        if not all(np.isfinite(value) for value in result.values()):
            raise PricingError("Heston/SLV Snowball Greek readout is non-finite")
        return result

    def _valuation_state_signature(
        self,
        product: SnowballOption,
        pricing_env: PricingEnvironment,
    ) -> tuple[str, bool]:
        spot = float(pricing_env.spot)
        T = float(product.get_maturity(pricing_env))
        if T <= 0.0 or is_zero(T):
            return ("terminal", False)
        if self._is_knocked_out_at_valuation(product, spot, pricing_env):
            return ("immediate_ko", False)
        ki_continuous = (
            product.barrier_config.ki_continuous
            or product.barrier_config.ki_observation_type
            == ObservationType.CONTINUOUS
        )
        return (
            "live",
            bool(
                self._is_knocked_in_at_valuation(
                    product,
                    spot,
                    pricing_env,
                    ki_continuous=ki_continuous,
                )
            ),
        )

    @staticmethod
    def _clock_basis(
        event_times: list[float],
        maturity: float,
        preferred_basis: int,
    ) -> Optional[int]:
        values = np.asarray([float(maturity), *event_times], dtype=float)
        candidates = tuple(
            dict.fromkeys((int(preferred_basis), 365, 252, 360, 366))
        )
        for basis in candidates:
            if basis <= 0:
                continue
            ticks = values * float(basis)
            if np.max(np.abs(ticks - np.rint(ticks))) <= 1e-8:
                return int(basis)
        return None

    def greek_time_grid_policy(
        self,
        product: SnowballOption,
        pricing_env: PricingEnvironment,
    ) -> dict:
        """Resolve the deterministic production grid used by spot Greeks.

        Generic S/V/time floors let a production engine retain its PV-certified
        medium grid while its one-surface delta/gamma solve uses the stronger
        certification candidate. In addition, a dense discrete KI schedule
        can impose a stricter exactly aligned time mesh when the finite-bump
        stencil straddles the barrier.

        Dense discrete KI schedules repeatedly inject a state-switch kink.
        When the declared finite-bump stencil straddles that barrier, use at
        least eight ADI steps per underlying schedule tick and choose a total
        step count that exactly aligns the common ACT/365 or 252-day clock.
        Other states retain the price-certified grid unchanged.
        """
        T = (
            float(product.get_maturity(pricing_env))
            if isinstance(product, SnowballOption) and pricing_env is not None
            else 0.0
        )
        resolved_n_x = max(int(self.n_x), int(self.greek_min_n_x))
        resolved_n_v = max(int(self.n_v), int(self.greek_min_n_v))
        resolved_n_t = int(self.n_t)
        if T > 0.0 and self.greek_min_steps_per_year > 0:
            resolved_n_t = max(
                resolved_n_t,
                int(np.ceil(self.greek_min_steps_per_year * T)),
            )
        reasons = []
        if resolved_n_x > self.n_x or resolved_n_v > self.n_v:
            reasons.append("production Greek spatial floor")
        if resolved_n_t > self.n_t:
            reasons.append("production Greek time-density floor")
        base = {
            "configured_n_x": int(self.n_x),
            "configured_n_v": int(self.n_v),
            "configured_n_t": int(self.n_t),
            "resolved_n_x": resolved_n_x,
            "resolved_n_v": resolved_n_v,
            "resolved_n_t": resolved_n_t,
            "refined": bool(reasons),
            "reason": "; ".join(reasons) if reasons else (
                "production stencil does not require Greek-grid refinement"
            ),
            "clock_basis": None,
            "steps_per_tick": int(self.barrier_greek_steps_per_tick),
            "minimum_steps_per_year": int(self.greek_min_steps_per_year),
            "barrier_minimum_n_x": int(self.barrier_greek_min_n_x),
        }
        if self.barrier_greek_steps_per_tick <= 0:
            if not reasons:
                base["reason"] = (
                    "barrier-adjacent Greek time refinement is disabled"
                )
            return base
        if not isinstance(product, SnowballOption) or pricing_env is None:
            return base
        bc = product.barrier_config
        if (
            bc.ki_continuous
            or bc.ki_observation_type != ObservationType.DISCRETE
            or not product.has_ki_barrier
        ):
            return base
        if T <= 0.0 or is_zero(T):
            return base

        ki_records = [
            rec
            for rec in product.resolve_ki_observations(pricing_env)
            if rec.observation_time is not None
            and -1e-12 <= float(rec.observation_time) <= T + 1e-12
        ]
        ki_times = sorted(
            {
                round(float(rec.observation_time), 12)
                for rec in ki_records
            }
        )
        if len(ki_times) / T < self.DENSE_KI_EVENTS_PER_YEAR:
            return base

        bump_config = self.params.get_effective_bump_config()
        relative_bump = max(
            float(bump_config.spot_bump),
            float(
                bump_config.gamma_spot_bump
                if bump_config.gamma_spot_bump is not None
                else bump_config.spot_bump
            ),
        )
        spot = float(pricing_env.spot)
        lo, hi = spot * (1.0 - relative_bump), spot * (1.0 + relative_bump)
        barriers = [
            float(rec.barrier)
            for rec in ki_records
            if rec.barrier is not None and np.isfinite(float(rec.barrier))
        ]
        if not any(lo <= barrier <= hi for barrier in barriers):
            return base

        event_times = list(ki_times)
        for rec in product.resolve_ko_observations(pricing_env):
            if (
                rec.observation_time is not None
                and -1e-12 <= float(rec.observation_time) <= T + 1e-12
            ):
                event_times.append(round(float(rec.observation_time), 12))
        basis = self._clock_basis(
            event_times,
            T,
            int(getattr(self.params, "bus_days_in_year", 252)),
        )
        if basis is not None:
            ticks = int(round(T * basis))
        else:
            # Unknown clocks still receive the same density floor; without a
            # provable rational clock the report exposes ``clock_basis=None``.
            ticks = max(len(ki_times), int(np.ceil(365.0 * T)))
        resolved = max(
            int(base["resolved_n_t"]),
            int(self.barrier_greek_steps_per_tick) * max(ticks, 1),
        )
        resolved_n_x = max(
            int(base["resolved_n_x"]),
            int(self.barrier_greek_min_n_x),
        )
        if resolved_n_x > int(base["resolved_n_x"]):
            reasons.append("dense-KI finite-bump spatial floor")
        if resolved > int(base["resolved_n_t"]):
            reasons.append(
                "finite-bump stencil straddles a dense discrete KI; "
                "time grid aligned and refined"
            )
        base.update(
            {
                "resolved_n_t": resolved,
                "resolved_n_x": resolved_n_x,
                "refined": bool(reasons),
                "reason": "; ".join(reasons),
                "clock_basis": basis,
            }
        )
        return base

    def price(self, product: BaseEquityProduct, pricing_env: PricingEnvironment) -> float:
        self._check_product_type(product)
        if pricing_env is None:
            raise ValidationError(f"PricingEnvironment is required for {self._solver_name}")
        self._validate_product(product)

        spot = float(pricing_env.spot)
        T = float(product.get_maturity(pricing_env))
        if T <= 0.0 or is_zero(T):
            return self._calculate_terminal_value(product, spot, pricing_env)
        if self._is_knocked_out_at_valuation(product, spot, pricing_env):
            return self._get_immediate_ko_payoff(product, pricing_env)
        # Valuation-date readout state: events at t=0 are deterministic at
        # the known spot, so the readout uses the smooth 0+ surface captured
        # by the hooks instead of interpolating across the nodal t=0 jump.
        self._t0_pre_U = None

        ki_continuous = (
            product.barrier_config.ki_continuous
            or product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
        )
        knocked_in = self._is_knocked_in_at_valuation(
            product, spot, pricing_env, ki_continuous=ki_continuous
        )
        core, read_surface = self._solve_live_surface(
            product, pricing_env, T, knocked_in=knocked_in
        )
        return float(
            core.interpolate(read_surface, np.log(spot), self.model_params.v0)
        )

    def _solve_live_surface(
        self,
        product: SnowballOption,
        pricing_env: PricingEnvironment,
        T: float,
        *,
        knocked_in: bool,
    ):
        """Solve the live V0/V1 system and return its reusable readout surface."""
        spot = float(pricing_env.spot)
        # Valuation-date readout state: events at t=0 are deterministic at
        # the known spot, so the readout uses the smooth 0+ surface captured
        # by the hooks instead of interpolating across the nodal t=0 jump.
        self._t0_pre_U = None
        ki_continuous = (
            product.barrier_config.ki_continuous
            or product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
        )
        self._prepare_state(product, pricing_env, T, ki_continuous)

        core = self._make_core(product, pricing_env, T)
        if not (core.S_grid[0] <= spot <= core.S_grid[-1]):
            raise ValidationError("spot falls outside the Heston/SLV Snowball PDE grid")
        self._prepare_2d_continuous_ki_correction(core, product)

        v1_snapshots: dict[float, np.ndarray] = {}
        event_maps = self._build_event_maps(product, pricing_env, T, core.dt)
        damped_keys = event_damped_step_keys(self.params, event_maps, core.N_T)

        terminal_v1 = self._terminal_surface(core, product, pricing_env, knocked_in=True)
        U1 = core.solve(
            is_call=True,
            scheme=self.scheme,
            theta=float(self.params.theta),
            rannacher=bool(self.params.use_rannacher),
            terminal_override=terminal_v1,
            boundary_hook=self._boundary_hook(core, product, pricing_env, knocked_in=True),
            step_hook=self._v1_hook(core, product, pricing_env, T, event_maps, v1_snapshots),
            damped_step_keys=damped_keys,
            damped_step_theta=float(self.params.event_theta),
        )

        if knocked_in:
            surface = U1
        else:
            terminal_v0 = self._terminal_surface(core, product, pricing_env, knocked_in=False)
            surface = core.solve(
                is_call=True,
                scheme=self.scheme,
                theta=float(self.params.theta),
                rannacher=bool(self.params.use_rannacher),
                terminal_override=terminal_v0,
                boundary_hook=self._boundary_hook(core, product, pricing_env, knocked_in=False),
                step_hook=self._v0_hook(
                    core, product, pricing_env, T, event_maps, v1_snapshots
                ),
                damped_step_keys=damped_keys,
                damped_step_theta=float(self.params.event_theta),
            )

        read_surface = self._t0_pre_U if self._t0_pre_U is not None else surface
        return core, read_surface

    def _capture_t0_pre_event_surface(self, U, tau, T, event_maps) -> None:
        """Capture the smooth 0+ surface before valuation-date events land.

        Continuous KI is excluded: its value function is continuous at the
        barrier, so there is no t=0 jump to shield the readout from."""
        if not self._use_cell_average_events():
            return
        # tau vs T, not current_time vs 0: tau accumulates FP step
        # increments, so a relative is_close against zero can never fire.
        if not is_close(float(tau), float(T)):
            return
        k = self._hook_tau_key(tau, event_maps["dt"])
        if k is None:
            return
        has_events = bool(event_maps["ko"].get(k))
        if not has_events and not self._ki_continuous:
            has_events = event_maps["ki"].get(k) is not None
        if has_events:
            self._t0_pre_U = np.array(U, copy=True)

    def _prepare_state(self, product, pricing_env, T: float, ki_continuous: bool) -> None:
        self._total_tau = float(T)
        self._structured_terminal_delay_df = self._terminal_delay_df(
            product, pricing_env
        )
        self._is_reverse = product.is_reverse
        self._ki_continuous = bool(ki_continuous)
        self._bgk_active = False
        self._ki_barrier = 0.0
        self._ki_barrier_by_tidx.clear()
        self._ki_fp = None  # rebuilt once the ADI core (and its V grid) exists
        if product.has_ki_barrier:
            ki_barrier = product.barrier_config.ki_barrier
            self._ki_barrier = float(ki_barrier[0] if isinstance(ki_barrier, list) else ki_barrier)

    def _terminal_surface(self, core, product, env, knocked_in: bool) -> np.ndarray:
        if knocked_in:
            values = [product.get_maturity_payoff_v1(float(s), env) for s in core.S_grid]
        else:
            values = [product.get_maturity_payoff_v0(float(s), env) for s in core.S_grid]
        values = np.asarray(values, dtype=float) * self._structured_terminal_delay_df
        return np.repeat(values[:, None], core.V_grid.size, axis=1)

    def _boundary_hook(self, core, product, env, knocked_in: bool):
        def hook(U, tau):
            grid = np.zeros((core.S_grid.size, 1), dtype=float)
            if knocked_in:
                self._set_boundary_conditions_v1(
                    grid, core.X_grid, core.S_grid, 0, float(tau), product, env
                )
            else:
                self._set_boundary_conditions_v0(
                    grid, core.X_grid, core.S_grid, 0, float(tau), product, env
                )
            U[0, :] = grid[0, 0]
            U[-1, :] = grid[-1, 0]
            return U

        return hook

    def _build_event_maps(self, product, env, T: float, dt: float):
        ko_by_key: dict[int, list] = {}
        for rec in self._get_cached_ko_records(env, product):
            if rec.observation_time is None:
                continue
            obs_time = float(rec.observation_time)
            if -1e-12 <= obs_time <= T + 1e-12:
                key = self._integer_tau_key(T - obs_time, dt)
                ko_by_key.setdefault(key, []).append(rec)

        ki_by_key: dict[int, float] = {}
        if product.has_ki_barrier and not self._ki_continuous:
            profile = self._get_cached_ki_profile(env, product)
            times = profile.get("observation_times") or []
            barriers = profile.get("barriers") or []
            for obs_time, barrier in zip(times, barriers):
                if obs_time is None or barrier is None:
                    continue
                obs_time = float(obs_time)
                if -1e-12 <= obs_time <= T + 1e-12:
                    ki_by_key[self._integer_tau_key(T - obs_time, dt)] = float(barrier)

        return {"ko": ko_by_key, "ki": ki_by_key, "dt": float(dt)}

    @staticmethod
    def _integer_tau_key(tau: float, dt: float) -> int:
        if dt <= 0:
            raise ValidationError("ADI time step must be positive")
        return int(round(max(float(tau), 0.0) / float(dt)))

    @staticmethod
    def _hook_tau_key(tau: float, dt: float) -> Optional[int]:
        if dt <= 0:
            return None
        k_float = float(tau) / float(dt)
        k = int(round(k_float))
        if abs(k_float - k) > 1e-8:
            return None
        return k

    @staticmethod
    def _snapshot_key(tau: float) -> float:
        return round(float(tau), 12)

    def _apply_ko(self, U, core, product, env, T: float, tau: float, event_maps):
        k = self._hook_tau_key(tau, event_maps["dt"])
        if k is None:
            return U
        for rec in event_maps["ko"].get(k, []):
            if rec.barrier is None:
                continue
            cashflow = rec.payoff if rec.payoff is not None else 0.0
            current_time = max(T - float(tau), 0.0)
            value = self._cashflow_value_at_time(
                pricing_env=env,
                cashflow=cashflow,
                current_time=current_time,
                settlement_time=rec.settlement_time,
            )
            # An observation at the valuation date (current_time == 0) is
            # deterministic — apply the exact inclusive trigger, not a cell
            # average [2026-07-23 review, finding 2]. Compare tau against T
            # (two O(1) numbers): tau accumulates FP step increments, so
            # current_time lands at ~1e-16 rather than 0.0, and a relative
            # is_close against zero can never fire.
            at_valuation = is_close(float(tau), float(T))
            if self._use_cell_average_events() and not at_valuation:
                U = self._project_event_values(
                    core.S_grid, float(rec.barrier), product.is_reverse, True,
                    U, float(value),
                )
                continue
            mask = self._event_nodal_mask(
                core.S_grid, float(rec.barrier), product.is_reverse, True,
                at_valuation=at_valuation,
            )
            U[mask, :] = float(value)
        return U

    def _should_apply_ki(self, tau: float, event_maps) -> tuple[bool, Optional[float]]:
        if self._ki_continuous:
            return True, float(self._ki_barrier)
        k = self._hook_tau_key(tau, event_maps["dt"])
        if k is None:
            return False, None
        barrier = event_maps["ki"].get(k)
        if barrier is None:
            return False, None
        return True, float(barrier)

    def _v1_hook(self, core, product, env, T, event_maps, snapshots):
        def hook(U, tau):
            self._capture_t0_pre_event_surface(U, tau, T, event_maps)
            U = self._apply_ko(U, core, product, env, T, tau, event_maps)
            snapshots[self._snapshot_key(tau)] = np.array(U, copy=True)
            return U

        return hook

    def _v0_hook(self, core, product, env, T, event_maps, snapshots):
        def hook(U, tau):
            self._capture_t0_pre_event_surface(U, tau, T, event_maps)
            U = self._apply_ko(U, core, product, env, T, tau, event_maps)
            if product.has_ki_barrier:
                should_apply, barrier = self._should_apply_ki(tau, event_maps)
                if should_apply and barrier is not None:
                    key = self._snapshot_key(tau)
                    v1 = snapshots.get(key)
                    if v1 is None:
                        raise PricingError("missing V1 snapshot for Heston/SLV Snowball KI jump")
                    # Continuous KI stays a nodal mask (continuous-barrier
                    # treatment); only discretely observed KI events project —
                    # and a valuation-date observation is deterministic, so it
                    # is applied with the exact inclusive trigger.
                    ki_discrete = not (self._ki_continuous or self._bgk_active)
                    # tau vs T, not current_time vs 0: see _apply_ko.
                    at_valuation = is_close(float(tau), float(T))
                    if (
                        self._use_cell_average_events()
                        and ki_discrete
                        and not at_valuation
                    ):
                        U = self._project_event_values(
                            core.S_grid, barrier, product.is_reverse, False, U, v1
                        )
                    else:
                        mask = self._event_nodal_mask(
                            core.S_grid, barrier, product.is_reverse, False,
                            at_valuation=(at_valuation and ki_discrete),
                        )
                        U[mask, :] = v1[mask, :]
                        U = self._apply_continuous_ki_correction(
                            U, core, tau, barrier, v1
                        )
            return U

        return hook


class HestonSnowballPDESolver(_Heston2DSnowballPDEBase):
    """Two-surface Snowball PDE under the Heston stochastic-volatility model."""

    _solver_name = "HestonSnowballPDESolver"


class HestonSLVSnowballPDESolver(SLVBarrierLeverageMixin, _Heston2DSnowballPDEBase):
    """Two-surface Snowball PDE under Heston-SLV using a calibrated leverage surface."""

    _solver_name = "HestonSLVSnowballPDESolver"

    def __init__(
        self,
        model_params: HestonParams,
        leverage_surface: LeverageSurface,
        eta: float = 1.0,
        **kwargs,
    ):
        if not isinstance(leverage_surface, LeverageSurface):
            raise ValidationError("leverage_surface must be a calibrated LeverageSurface")
        if eta < 0:
            raise ValidationError("eta must be non-negative")
        # Set before the base constructor so its automatic variance-grid
        # classifier uses the effective SLV vol-of-vol eta*sigma.
        self.eta = float(eta)
        super().__init__(model_params=model_params, **kwargs)
        self.leverage_surface = leverage_surface

    def _make_core(self, product: SnowballOption, env: PricingEnvironment, T: float):
        t_grid = np.linspace(0.0, float(T), self.n_t + 1)
        market = TermMarketContext.from_env(
            env,
            t_grid,
            ref_strike=None,
        )
        x_nodes = self._layer_x_nodes(product, env, T)
        return HestonSLVADICore(
            float(env.spot),
            float(product.strike),
            T,
            float(env.get_rate(T)),
            float(env.get_div_yield(T)),
            self.model_params,
            self.n_x,
            self.n_v,
            self.n_t,
            market_context=market,
            leverage=self.leverage_surface,
            eta=self.eta,
            grid_style=self.grid_style,
            v0_boundary=self.v0_boundary,
            v_grid_power=self.v_grid_power,
            variance_grid_mode=self.variance_grid_mode,
            v_drift_scheme=self.v_drift_scheme,
            barrier_concentrate=self._grid_concentration_spot(product, env),
            critical_spots=(
                self._grid_critical_spots(product, env)
                if self.pin_critical_spots
                else None
            ),
            x_nodes=x_nodes,
        )
