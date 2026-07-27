"""Phoenix PDE solvers under Local Vol / Heston / Heston-SLV processes."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.engine.pde.base_pde_solver import StepCoefficients
from quantark.asset.equity.engine.pde.grid.events import (
    project_piecewise_event,
)
from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.param import GridVolSurface
from quantark.priceenv import PricingEnvironment, TermMarketContext
from quantark.priceenv.term_sampling import TermCoefficients
from quantark.util.enum import CouponPayType, ObservationType
from quantark.util.enum.engine_enums import ADIScheme, EngineType
from quantark.util.exceptions import PricingError, ValidationError
from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import (
    event_damped_step_keys,
)
from quantark.util.numerical import is_close, is_zero
from quantark.volmodels.adi_core import HestonSLVADICore
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.localvol import LocalVolSurface, build_dupire_local_vol
from quantark.volmodels.slv.leverage import LeverageSurface


class LocalVolPhoenixPDESolver(PhoenixPDESolver):

    """Two-surface Phoenix PDE with Dupire local volatility on the S grid."""

    engine_type = EngineType.PDE
    _solver_name = "LocalVolPhoenixPDESolver"

    def __init__(
        self,
        params: Optional[PDEParams] = None,
        local_vol_surface: Optional[LocalVolSurface] = None,
    ):
        super().__init__(params=params)
        self._prebuilt = local_vol_surface
        self._active_lv_surface: Optional[LocalVolSurface] = None
        self._active_s_vec: Optional[np.ndarray] = None

    def _build_surface(self, env: PricingEnvironment) -> LocalVolSurface:
        if self._prebuilt is not None:
            return self._prebuilt
        if not isinstance(env.vol_surface, GridVolSurface):
            raise PricingError(
                "LocalVolPhoenixPDESolver needs a GridVolSurface or a prebuilt LocalVolSurface"
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

    def price(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        return self._with_surface(
            pricing_env,
            lambda: PhoenixPDESolver.price(self, product, pricing_env),
        )

    def calculate_greeks(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> Dict[str, float]:
        return self._with_surface(
            pricing_env,
            lambda: PhoenixPDESolver.calculate_greeks(self, product, pricing_env),
        )

    def calculate_event_stats(self, product, pricing_env, **kwargs):
        return self._with_surface(
            pricing_env,
            lambda: PhoenixPDESolver.calculate_event_stats(
                self, product, pricing_env, **kwargs
            ),
        )

    def _session_outputs(self, product, pricing_env, **kwargs):
        return self._with_surface(
            pricing_env,
            lambda: PhoenixPDESolver._session_outputs(
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
            sigma_vec = np.asarray(
                self._active_lv_surface.local_vol(s_vec, t_mid), dtype=float
            )
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
            raise ValidationError(
                "local-vol surface returned non-positive or non-finite vols"
            )

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


class _Heston2DPhoenixPDEBase(PhoenixPDESolver):

    engine_type = EngineType.PDE
    _solver_name = "Heston2DPhoenixPDESolver"

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
    ):
        if not isinstance(model_params, HestonParams):
            raise ValidationError("model_params must be a HestonParams")
        super().__init__(params=params or PDEParams())
        try:
            scheme = ADIScheme[scheme.upper()] if isinstance(scheme, str) else scheme
        except KeyError:
            raise ValidationError(f"unknown ADI scheme: {scheme}")
        if scheme == ADIScheme.MCS:
            raise ValidationError("MCS is not implemented for the Heston/SLV Phoenix PDE")
        if grid_focus not in {"auto", "ko", "ki", "coupon", "strike", "spot"}:
            raise ValidationError(
                "grid_focus must be one of: auto, ko, ki, coupon, strike, spot"
            )
        self.model_params = model_params
        self.n_x = int(n_x)
        self.n_v = int(n_v)
        self.n_t = int(n_t)
        self.scheme = scheme
        self.grid_style = grid_style
        self.grid_focus = grid_focus
        self.pin_critical_spots = bool(pin_critical_spots)

    def representative_vol(self, product, pricing_env) -> float:
        # sqrt(var_eff) ported VERBATIM from the adi_core x-width computation.
        p = self.model_params
        var_eff = max(p.theta, p.v0, 0.25 * (p.sigma * p.sigma), 0.04)
        return float(np.sqrt(var_eff))

    def _layer_x_nodes(self, product: PhoenixOption, env: PricingEnvironment, T: float):
        """S-axis nodes from the declarative grid layer (spec §4.6, Phase 3);
        num_std=8 preserves the certified adi_core domain width."""
        from quantark.asset.equity.engine.pde.grid import GridBinder, GridConfig

        market = self.market_snapshot(product, env)
        request = self.grid_request(product, market, float(T))
        binder = GridBinder(
            "standard",
            GridConfig(points=int(self.n_x), num_std=8.0),
            cache_enabled=self._is_cache_enabled(),
        )
        return binder.bind(request, market).spatial.x

    def _make_core(self, product: PhoenixOption, env: PricingEnvironment, T: float):
        t_grid = np.linspace(0.0, float(T), self.n_t + 1)
        market = TermMarketContext.from_env(env, t_grid, ref_strike=None)
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
            barrier_concentrate=self._grid_concentration_spot(product, env),
            critical_spots=(
                self._grid_critical_spots(product, env)
                if self.pin_critical_spots
                else None
            ),
            x_nodes=x_nodes,
        )

    def _primary_barrier(self, product: PhoenixOption) -> float:
        ko_barriers = self._positive_levels(product.barrier_config.ko_barrier)
        if not ko_barriers:
            return float(product.strike)
        return float(min(ko_barriers) if product.is_reverse else max(ko_barriers))

    def _primary_ki_barrier(self, product: PhoenixOption) -> Optional[float]:
        ki_barriers = self._positive_levels(product.barrier_config.ki_barrier)
        if not ki_barriers:
            return None
        return float(max(ki_barriers) if product.is_reverse else min(ki_barriers))

    def _primary_coupon_barrier(self, product: PhoenixOption) -> Optional[float]:
        coupon_barriers = self._positive_levels(product.coupon_config.coupon_barrier)
        if not coupon_barriers:
            return None
        return float(min(coupon_barriers) if product.is_reverse else max(coupon_barriers))

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

    def _auto_grid_focus(self, product: PhoenixOption) -> str:
        if product.has_ki_barrier:
            return "ki"
        if self._primary_coupon_barrier(product) is not None:
            return "coupon"
        return "strike"

    def _grid_concentration_spot(
        self, product: PhoenixOption, env: PricingEnvironment
    ) -> float:
        focus = self._auto_grid_focus(product) if self.grid_focus == "auto" else self.grid_focus
        if focus == "ko":
            return self._primary_barrier(product)
        if focus == "ki":
            ki_barrier = self._primary_ki_barrier(product)
            return float(ki_barrier if ki_barrier is not None else product.strike)
        if focus == "coupon":
            coupon_barrier = self._primary_coupon_barrier(product)
            return float(coupon_barrier if coupon_barrier is not None else product.strike)
        if focus == "spot":
            return float(env.spot)
        return float(product.strike)

    def _grid_critical_spots(
        self, product: PhoenixOption, env: PricingEnvironment
    ) -> list[float]:
        levels = [
            float(env.spot),
            float(product.initial_price),
            float(product.strike),
        ]
        levels.extend(self._positive_levels(product.barrier_config.ko_barrier))
        levels.extend(self._positive_levels(product.barrier_config.ki_barrier))
        levels.extend(self._positive_levels(product.coupon_config.coupon_barrier))

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
        if not isinstance(product, PhoenixOption):
            return None
        if pricing_env is None:
            return None
        if product.has_memory_coupon:
            raise ValidationError(
                f"{self._solver_name} supports non-memory Phoenix coupons only; "
                "use PhoenixPDESolver or a Phoenix MC engine for memory coupons."
            )
        pde_pv = float(npv) if npv is not None else float(self.price(product, pricing_env))
        return PhoenixPDESolver.calculate_event_stats(
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
        return BaseEngine.calculate_greeks(self, product, pricing_env)

    def price(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        self._check_product_type(product)
        if pricing_env is None:
            raise ValidationError(f"PricingEnvironment is required for {self._solver_name}")
        self._validate_product(product)
        if product.has_memory_coupon:
            raise ValidationError(
                f"{self._solver_name} supports non-memory Phoenix coupons only; "
                "use PhoenixPDESolver or a Phoenix MC engine for memory coupons."
            )

        spot = float(pricing_env.spot)
        T = float(product.get_maturity(pricing_env))
        if T <= 0.0 or is_zero(T):
            return self._calculate_terminal_value(product, spot, pricing_env)
        if self._is_knocked_out_at_valuation(product, spot, pricing_env):
            return self._get_immediate_ko_payoff(product, pricing_env)
        # Valuation-date readout state: events at t=0 are deterministic at the
        # known spot, so the readout uses the smooth 0+ surface captured by
        # the hooks (plus a pointwise coupon transition) instead of
        # interpolating across the nodal t=0 jump.
        self._t0_pre_U = None
        self._t0_readout_override = None

        ki_continuous = (
            product.barrier_config.ki_continuous
            or product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
        )
        knocked_in = self._is_knocked_in_at_valuation(
            product, spot, pricing_env, ki_continuous=ki_continuous
        )
        self._prepare_state(product, pricing_env, T, ki_continuous)

        core = self._make_core(product, pricing_env, T)
        if not (core.S_grid[0] <= spot <= core.S_grid[-1]):
            raise ValidationError("spot falls outside the Heston/SLV Phoenix PDE grid")

        event_maps = self._build_event_maps(product, pricing_env, T, core.dt)
        damped_keys = event_damped_step_keys(self.params, event_maps, core.N_T)
        v1_snapshots: dict[float, np.ndarray] = {}

        terminal_v1 = self._terminal_surface(core, product, pricing_env, knocked_in=True)
        U1 = core.solve(
            is_call=True,
            scheme=self.scheme,
            theta=float(self.params.theta),
            rannacher=bool(self.params.use_rannacher),
            terminal_override=terminal_v1,
            boundary_hook=self._boundary_hook(
                core, product, pricing_env, knocked_in=True
            ),
            step_hook=self._v1_hook(
                core, product, pricing_env, T, event_maps, v1_snapshots
            ),
            damped_step_keys=damped_keys,
            damped_step_theta=float(self.params.event_theta),
        )

        if knocked_in:
            surface = U1
        else:
            terminal_v0 = self._terminal_surface(
                core, product, pricing_env, knocked_in=False
            )
            surface = core.solve(
                is_call=True,
                scheme=self.scheme,
                theta=float(self.params.theta),
                rannacher=bool(self.params.use_rannacher),
                terminal_override=terminal_v0,
                boundary_hook=self._boundary_hook(
                    core, product, pricing_env, knocked_in=False
                ),
                step_hook=self._v0_hook(
                    core, product, pricing_env, T, event_maps, v1_snapshots
                ),
                damped_step_keys=damped_keys,
                damped_step_theta=float(self.params.event_theta),
            )

        if self._t0_readout_override is not None:
            return float(self._t0_readout_override)
        read_surface = self._t0_pre_U if self._t0_pre_U is not None else surface
        return float(core.interpolate(read_surface, np.log(spot), self.model_params.v0))

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
        has_events = bool(event_maps["ko"].get(k)) or (
            event_maps.get("coupon", {}).get(k) is not None
        )
        if not has_events and not self._ki_continuous:
            has_events = event_maps["ki"].get(k) is not None
        if has_events:
            self._t0_pre_U = np.array(U, copy=True)

    def _prepare_state(
        self, product: PhoenixOption, pricing_env, T: float, ki_continuous: bool
    ) -> None:
        self._total_tau = float(T)
        self._is_reverse = product.is_reverse
        self._ki_continuous = bool(ki_continuous)
        self._bgk_active = False
        self._ki_barrier = 0.0
        self._ki_barrier_by_tidx.clear()
        self._ko_observation_indices.clear()
        self._ki_observation_indices.clear()
        self._coupon_observation_indices.clear()
        if product.has_ki_barrier:
            ki_barrier = product.barrier_config.ki_barrier
            self._ki_barrier = float(
                ki_barrier[0] if isinstance(ki_barrier, list) else ki_barrier
            )

    def _terminal_surface(self, core, product, env, knocked_in: bool) -> np.ndarray:
        if knocked_in:
            values = [product.get_maturity_payoff_v1(float(s), env) for s in core.S_grid]
        else:
            values = [
                product.get_maturity_payoff_v0(
                    float(s), accumulated_coupons=0.0, pricing_env=env
                )
                for s in core.S_grid
            ]
        return np.repeat(np.asarray(values, dtype=float)[:, None], core.V_grid.size, axis=1)

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

    def _build_event_maps(self, product: PhoenixOption, env, T: float, dt: float):
        ko_records = self._get_cached_ko_records(env, product)
        self._setup_coupon_schedule(product, env, ko_records)

        ko_by_key: dict[int, list] = {}
        coupon_by_key: dict[int, int] = {}
        for obs_idx, rec in enumerate(ko_records):
            if rec.observation_time is None:
                continue
            obs_time = float(rec.observation_time)
            if -1e-12 <= obs_time <= T + 1e-12:
                key = self._integer_tau_key(T - obs_time, dt)
                ko_by_key.setdefault(key, []).append(rec)
                coupon_by_key[key] = obs_idx

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

        return {
            "ko": ko_by_key,
            "ki": ki_by_key,
            "coupon": coupon_by_key,
            "dt": float(dt),
        }

    def _setup_coupon_schedule(self, product, env, ko_records) -> None:
        self._coupon_observation_indices.clear()
        num_obs = len(ko_records)
        if num_obs == 0:
            self._coupon_barriers = np.array([], dtype=float)
            self._coupon_amounts = np.array([], dtype=float)
            self._coupon_cumulative = np.array([0.0], dtype=float)
            return

        coupon_barrier = product.coupon_config.coupon_barrier
        if isinstance(coupon_barrier, list):
            if len(coupon_barrier) != num_obs:
                raise ValidationError(
                    "Coupon barrier schedule length does not match KO observations."
                )
            self._coupon_barriers = np.array(coupon_barrier, dtype=float)
        else:
            self._coupon_barriers = np.full(num_obs, float(coupon_barrier))

        ko_times = [rec.observation_time for rec in ko_records]
        period_year_fractions = np.array(
            product.get_coupon_period_year_fractions(ko_times),
            dtype=float,
        )
        self._coupon_amounts = np.array(
            [
                product.get_coupon_payoff(i, year_fraction=period_year_fractions[i])
                for i in range(num_obs)
            ],
            dtype=float,
        )
        self._coupon_cumulative = np.concatenate(
            ([0.0], np.cumsum(self._coupon_amounts))
        )

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

    def _apply_coupon(self, U, core, product, env, T: float, tau: float, obs_idx: int):
        if obs_idx < 0 or obs_idx >= self._coupon_barriers.shape[0]:
            return U
        current_time = max(T - float(tau), 0.0)
        settlement_time = (
            T
            if product.coupon_config.coupon_pay_type == CouponPayType.EXPIRY
            else current_time
        )
        coupon_discount = self._df_between_times(env, current_time, settlement_time)
        coupon_value = float(self._coupon_amounts[obs_idx]) * coupon_discount
        # A coupon observed at the valuation date is deterministic — apply
        # the exact inclusive trigger [2026-07-23 review, finding 2].
        # tau vs T, not current_time vs 0: tau accumulates FP step
        # increments, so a relative is_close against zero can never fire.
        at_valuation = is_close(float(tau), float(T))
        if self._use_cell_average_events() and not at_valuation:
            return self._project_event_values(
                core.S_grid,
                float(self._coupon_barriers[obs_idx]),
                product.is_reverse,
                True,
                U,
                U + coupon_value,
            )
        if at_valuation and self._use_cell_average_events():
            # Pointwise-exact readout: resolve today's trigger at the known
            # spot and add the coupon to the smooth pre-coupon surface value
            # instead of letting price() interpolate across the t=0 jump
            # [2026-07-24 review, finding 1].
            spot = float(env.spot)
            pay = bool(
                self._event_nodal_mask(
                    np.asarray([spot], dtype=float),
                    float(self._coupon_barriers[obs_idx]),
                    product.is_reverse,
                    True,
                    at_valuation=True,
                )[0]
            )
            base_val = float(
                core.interpolate(U, np.log(spot), self.model_params.v0)
            )
            self._t0_readout_override = base_val + (coupon_value if pay else 0.0)
        pay_mask = self._event_nodal_mask(
            core.S_grid,
            float(self._coupon_barriers[obs_idx]),
            product.is_reverse,
            True,
            at_valuation=at_valuation,
        )
        if np.any(pay_mask):
            U[pay_mask, :] += coupon_value
        return U

    def _apply_ko(
        self,
        U,
        core,
        product,
        env,
        T: float,
        tau: float,
        ko_record,
        obs_idx: Optional[int],
    ):
        if ko_record.barrier is None:
            return U
        current_time = max(T - float(tau), 0.0)

        base_payoff = float(ko_record.payoff or 0.0)
        df = 1.0
        if ko_record.settlement_time is not None and ko_record.settlement_time > current_time:
            df = self._df_between_times(env, current_time, ko_record.settlement_time)

        # A KO observed at the valuation date is deterministic — apply the
        # exact inclusive trigger [2026-07-23 review, finding 2].
        # tau vs T, not current_time vs 0: see _apply_coupon.
        at_valuation = is_close(float(tau), float(T))
        if self._use_cell_average_events() and not at_valuation:
            if self._joint_coupon_ko_active_2d(ko_record, obs_idx):
                # Coincident coupon + KO: ONE piecewise cell average of the
                # complete contractual event (the hook skipped the standalone
                # coupon application) [2026-07-24 review, finding 3].
                return self._apply_joint_coupon_ko_projection_2d(
                    U, core, product, env, T, tau, ko_record, obs_idx,
                    base_payoff, df,
                )
            # Degenerate fallback (non-positive barriers): project the inner
            # coupon jump into the payoff profile, then the KO transition.
            if obs_idx is not None and 0 <= obs_idx < self._coupon_amounts.shape[0]:
                total = self._project_event_values(
                    core.S_grid,
                    float(self._coupon_barriers[obs_idx]),
                    product.is_reverse,
                    True,
                    base_payoff * df,
                    (base_payoff + float(self._coupon_amounts[obs_idx])) * df,
                )
            else:
                total = np.full(core.S_grid.shape, base_payoff * df, dtype=float)
            return self._project_event_values(
                core.S_grid, float(ko_record.barrier), product.is_reverse, True,
                U, total[:, None],
            )

        ko_mask = self._event_nodal_mask(
            core.S_grid, float(ko_record.barrier), product.is_reverse, True,
            at_valuation=at_valuation,
        )
        if not np.any(ko_mask):
            return U

        total = np.full(core.S_grid.shape, base_payoff * df, dtype=float)

        if obs_idx is not None and 0 <= obs_idx < self._coupon_amounts.shape[0]:
            pay_mask = self._event_nodal_mask(
                core.S_grid,
                float(self._coupon_barriers[obs_idx]),
                product.is_reverse,
                True,
                at_valuation=at_valuation,
            )
            total = np.where(
                pay_mask,
                total + float(self._coupon_amounts[obs_idx]) * df,
                total,
            )

        U[ko_mask, :] = total[ko_mask, None]
        return U

    def _joint_coupon_ko_active_2d(self, ko_record, obs_idx) -> bool:
        """Coincident coupon + KO handled as ONE piecewise projection."""
        if obs_idx is None or ko_record is None or ko_record.barrier is None:
            return False
        if not (0 <= obs_idx < self._coupon_barriers.shape[0]):
            return False
        return (
            float(ko_record.barrier) > 0.0
            and float(self._coupon_barriers[obs_idx]) > 0.0
        )

    def _apply_joint_coupon_ko_projection_2d(
        self, U, core, product, env, T, tau, ko_record, obs_idx, base_payoff, df
    ):
        """One-pass cell average of the coincident coupon + KO event,
        slice-wise over the variance dimension [2026-07-24 review, finding 3].

        Regions of the contractual post-event value:
            KO:             (base + coupon_amt * 1_pay) * df
            survive & pay:  U + coupon (coupon settlement discount)
            survive & miss: U
        """
        current_time = max(T - float(tau), 0.0)
        settlement_time = (
            T
            if product.coupon_config.coupon_pay_type == CouponPayType.EXPIRY
            else current_time
        )
        coupon_discount = self._df_between_times(env, current_time, settlement_time)
        coupon_amt = float(self._coupon_amounts[obs_idx])
        coupon_value = coupon_amt * coupon_discount
        n = core.S_grid.shape[0]
        x_vec = np.log(np.asarray(core.S_grid, dtype=float))
        trig_up = not bool(product.is_reverse)
        b_c_x = np.log(float(self._coupon_barriers[obs_idx]))
        b_ko_x = np.log(float(ko_record.barrier))
        breaks = sorted((b_c_x, b_ko_x))
        lows = [-np.inf] + breaks
        his = breaks + [np.inf]
        branches = []
        for j in range(len(breaks) + 1):
            if trig_up:
                m_ko = b_ko_x <= lows[j]
                m_pay = b_c_x <= lows[j]
            else:
                m_ko = his[j] <= b_ko_x
                m_pay = his[j] <= b_c_x
            if m_ko:
                branches.append(
                    np.full(
                        (n, 1),
                        (base_payoff + (coupon_amt if m_pay else 0.0)) * df,
                    )
                )
            elif m_pay:
                branches.append(U + coupon_value)
            else:
                branches.append(U)
        return project_piecewise_event(x_vec, breaks, branches)

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

    def _apply_ki(self, U, core, product, barrier: float, v1, at_valuation=False):
        # Continuous KI stays a nodal mask (continuous-barrier treatment);
        # only discretely observed KI events project — and a valuation-date
        # observation is deterministic, so it uses the exact inclusive trigger.
        ki_discrete = not (self._ki_continuous or self._bgk_active)
        if self._use_cell_average_events() and ki_discrete and not at_valuation:
            return self._project_event_values(
                core.S_grid, barrier, product.is_reverse, False, U, v1
            )
        mask = self._event_nodal_mask(
            core.S_grid, barrier, product.is_reverse, False,
            at_valuation=(at_valuation and ki_discrete),
        )
        if np.any(mask):
            U[mask, :] = v1[mask, :]
        return U

    def _hook_coupon_is_joint(self, tau, T, event_maps, k, obs_idx) -> bool:
        """True when the KO site will apply the coincident coupon fan-in as
        part of ONE piecewise projection (skip the standalone coupon)."""
        if obs_idx is None or k is None:
            return False
        if not self._use_cell_average_events():
            return False
        if is_close(float(tau), float(T)):
            return False
        recs = event_maps["ko"].get(k, [])
        if len(recs) != 1:
            return False
        return self._joint_coupon_ko_active_2d(recs[0], obs_idx)

    def _v1_hook(self, core, product, env, T, event_maps, snapshots):
        def hook(U, tau):
            self._capture_t0_pre_event_surface(U, tau, T, event_maps)
            k = self._hook_tau_key(tau, event_maps["dt"])
            obs_idx = None if k is None else event_maps["coupon"].get(k)
            if obs_idx is not None and not self._hook_coupon_is_joint(
                tau, T, event_maps, k, obs_idx
            ):
                U = self._apply_coupon(U, core, product, env, T, tau, obs_idx)
            if k is not None:
                for rec in event_maps["ko"].get(k, []):
                    U = self._apply_ko(U, core, product, env, T, tau, rec, obs_idx)
            snapshots[self._snapshot_key(tau)] = np.array(U, copy=True)
            return U

        return hook

    def _v0_hook(self, core, product, env, T, event_maps, snapshots):
        def hook(U, tau):
            self._capture_t0_pre_event_surface(U, tau, T, event_maps)
            k = self._hook_tau_key(tau, event_maps["dt"])
            obs_idx = None if k is None else event_maps["coupon"].get(k)
            if obs_idx is not None and not self._hook_coupon_is_joint(
                tau, T, event_maps, k, obs_idx
            ):
                U = self._apply_coupon(U, core, product, env, T, tau, obs_idx)
            if product.has_ki_barrier:
                should_apply, barrier = self._should_apply_ki(tau, event_maps)
                if should_apply and barrier is not None:
                    key = self._snapshot_key(tau)
                    v1 = snapshots.get(key)
                    if v1 is None:
                        raise PricingError("missing V1 snapshot for Heston/SLV Phoenix KI jump")
                    U = self._apply_ki(
                        U, core, product, barrier, v1,
                        at_valuation=is_close(float(tau), float(T)),
                    )
            if k is not None:
                for rec in event_maps["ko"].get(k, []):
                    U = self._apply_ko(U, core, product, env, T, tau, rec, obs_idx)
            return U

        return hook


class HestonPhoenixPDESolver(_Heston2DPhoenixPDEBase):
    """Two-surface Phoenix PDE under the Heston stochastic-volatility model."""

    _solver_name = "HestonPhoenixPDESolver"


class HestonSLVPhoenixPDESolver(_Heston2DPhoenixPDEBase):
    """Two-surface Phoenix PDE under Heston-SLV using a calibrated leverage surface."""

    _solver_name = "HestonSLVPhoenixPDESolver"

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
        super().__init__(model_params=model_params, **kwargs)
        self.leverage_surface = leverage_surface
        self.eta = float(eta)

    def _make_core(self, product: PhoenixOption, env: PricingEnvironment, T: float):
        t_grid = np.linspace(0.0, float(T), self.n_t + 1)
        market = TermMarketContext.from_env(env, t_grid, ref_strike=None)
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
            barrier_concentrate=self._grid_concentration_spot(product, env),
            critical_spots=(
                self._grid_critical_spots(product, env)
                if self.pin_critical_spots
                else None
            ),
        )
