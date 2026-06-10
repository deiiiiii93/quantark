"""
Base PDE solver providing common infrastructure for finite difference pricing.

Implements the Crank-Nicolson scheme for solving the Black-Scholes PDE
backward in time, with support for Rannacher smoothing.
"""

from abc import abstractmethod
from collections import OrderedDict
import math
import threading
from typing import Dict, Optional, Tuple, List, NamedTuple
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.param import PDEParams
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import PricingError, NumericalError
from quantark.util.numerical import is_close
from quantark.util.enum.option_enums import ExerciseType, ObservationType
from quantark.util.enum.engine_enums import EngineType

from .time_grid import TimeGrid
from .spatial_grid import SpatialGrid


class PDESolutionResult(NamedTuple):
    """
    Result from PDE solving containing solution and grid data.

    This type allows both price() and calculate_greeks() to share the
    common solving logic via _solve(), eliminating code duplication.

    Attributes:
        solution_vec: Solution values at t=0 (present time)
        x_vec: Log-price grid points
        s_vec: Price grid points (for convenience)
        spot_log: Log of spot price for interpolation
    """

    solution_vec: np.ndarray
    x_vec: np.ndarray
    s_vec: np.ndarray
    spot_log: float


class BasePDESolver(BaseEngine):
    """
    Abstract base class for PDE-based option pricing.

    Solves the Black-Scholes PDE using finite difference methods in log-price space (x = ln(S)):
        dV/dt + (r-q-0.5*sigma^2)*dV/dx + 0.5*sigma^2*d2V/dx^2 - r*V = 0

    The PDE is solved backward in time from maturity to valuation date.
    """

    engine_type = EngineType.PDE

    _shared_grid_cache: "OrderedDict[Tuple, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]" = OrderedDict()
    _grid_cache_max_entries: int = 128
    _global_cache_enabled: bool = True
    _global_cache_strategy: Optional[str] = None
    _cache_lock = threading.Lock()

    def __init__(self, params: Optional[PDEParams] = None):
        """Initialize the PDE solver with configuration parameters."""
        super().__init__(params if params is not None else PDEParams())
        self._matrix_cache: Dict[Tuple[float, float], Tuple] = {}
        self._grid_cache = self.__class__._shared_grid_cache
        self._cache_enabled = bool(getattr(self.params, "cache_enabled", True))
        self._cache_strategy = getattr(self.params, "cache_strategy", "standard")
        cache_size = getattr(self.params, "grid_cache_max_entries", None)
        if cache_size is not None:
            self.set_grid_cache_max_entries(cache_size)
        self._critical_points_cache: "OrderedDict[Tuple, Tuple[float, ...]]" = (
            OrderedDict()
        )

    @classmethod
    def clear_grid_cache(cls) -> None:
        """Clear the shared grid cache for this solver class."""
        with cls._cache_lock:
            cls._shared_grid_cache.clear()

    @classmethod
    def set_cache_enabled(cls, enabled: bool, clear: bool = False) -> None:
        """Enable or disable cache usage for this solver class."""
        cls._global_cache_enabled = bool(enabled)
        if clear:
            cls.clear_grid_cache()

    @classmethod
    def set_cache_strategy(cls, strategy: str, clear: bool = False) -> None:
        """Set the cache strategy for this solver class."""
        if strategy not in ("disable", "strict", "standard", "aggressive"):
            raise PricingError(
                "cache_strategy must be one of disable, strict, standard, aggressive, got "
                f"{strategy}"
            )
        cls._global_cache_strategy = strategy
        if clear:
            cls.clear_grid_cache()

    @classmethod
    def set_grid_cache_max_entries(cls, max_entries: int) -> None:
        """Set maximum number of grid entries to keep in cache."""
        if max_entries <= 0:
            raise PricingError(
                f"Grid cache size must be positive, got {max_entries}"
            )
        with cls._cache_lock:
            cls._grid_cache_max_entries = max_entries
            while len(cls._shared_grid_cache) > cls._grid_cache_max_entries:
                cls._shared_grid_cache.popitem(last=False)

    def _is_cache_enabled(self) -> bool:
        return self._resolve_cache_strategy() != "disable"

    def _resolve_cache_strategy(self) -> str:
        if not self.__class__._global_cache_enabled or not self._cache_enabled:
            return "disable"
        strategy = self.__class__._global_cache_strategy
        if strategy is None:
            strategy = self._cache_strategy
        return strategy

    def _freeze_cache_value(self, value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple)):
            return tuple(self._freeze_cache_value(v) for v in value)
        if isinstance(value, dict):
            return tuple(
                sorted(
                    (self._freeze_cache_value(k), self._freeze_cache_value(v))
                    for k, v in value.items()
                )
            )
        return repr(value)

    def _cache_value_is_simple(
        self, value, depth: int, max_depth: int, max_len: int
    ) -> bool:
        if value is None or isinstance(value, (str, int, float, bool)):
            return True
        if depth >= max_depth:
            return False
        if isinstance(value, (list, tuple)):
            if len(value) > max_len:
                return False
            return all(
                self._cache_value_is_simple(v, depth + 1, max_depth, max_len)
                for v in value
            )
        if isinstance(value, dict):
            if len(value) > max_len:
                return False
            return all(
                self._cache_value_is_simple(k, depth + 1, max_depth, max_len)
                and self._cache_value_is_simple(v, depth + 1, max_depth, max_len)
                for k, v in value.items()
            )
        return False

    def _cache_dict_is_reasonable(
        self, attrs: Dict, max_depth: int = 2, max_len: int = 64
    ) -> bool:
        if len(attrs) > max_len:
            return False
        for value in attrs.values():
            if not self._cache_value_is_simple(value, 0, max_depth, max_len):
                return False
        return True

    def _product_cache_token(self, product: BaseEquityProduct, strategy: str) -> Tuple[str, object]:
        if strategy == "strict":
            return ("id", id(product))
        key_fn = getattr(product, "cache_key", None)
        if callable(key_fn):
            return ("key", self._freeze_cache_value(key_fn()))
        if strategy in ("standard", "aggressive"):
            attrs = getattr(product, "__dict__", None)
            if attrs is not None and self._cache_dict_is_reasonable(attrs):
                return ("dict", self._freeze_cache_value(attrs))
            if strategy == "aggressive":
                return ("repr", repr(product))
        return ("id", id(product))

    def _params_cache_key(self) -> Tuple:
        params: PDEParams = self.params
        return (
            params.grid_size,
            params.time_steps,
            params.adaptive_grid,
            params.grid_cache_max_entries,
            params.auto_grid,
            params.s_min,
            params.s_max,
            params.time_grid_type,
            params.grade_exponent,
            params.bus_days_in_year,
            params.event_steps_per_day,
            params.event_min_steps_per_interval,
            params.max_time_steps,
            params.log_dx_target,
            params.max_grid_size,
            params.include_spot_in_critical_points,
            params.rannacher_at_events,
            params.event_theta,
            params.event_rannacher_steps,
            params.barrier_refine_log_width,
            params.barrier_refine_levels,
            params.barrier_domain_expand,
            params.boundary_mode,
            params.theta,
            params.use_rannacher,
            params.rannacher_steps,
        )

    def _grid_cache_key(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        spot: float,
        sigma: float,
        tau: float,
        r: float,
        q: float,
    ) -> Tuple:
        barriers = tuple(
            sorted(
                [
                    round(b, 12)
                    for b in self._get_barriers(product)
                    if b is not None
                ]
            )
        )
        event_times = tuple(
            sorted(
                [round(t, 12) for t in (self._get_event_times(product, tau) or [])]
            )
        )
        critical_points = self._get_cached_critical_points(product, pricing_env, spot)
        return (
            f"{product.__class__.__module__}.{product.__class__.__qualname__}",
            round(spot, 12),
            round(sigma, 12),
            round(tau, 12),
            round(r, 12),
            round(q, 12),
            barriers,
            event_times,
            critical_points,
            self._params_cache_key(),
        )

    @abstractmethod
    def set_terminal_condition(
        self,
        grid: np.ndarray,
        x_vec: np.ndarray,
        s_vec: np.ndarray,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
    ) -> None:
        """Set the terminal condition (payoff at maturity)."""
        pass

    @abstractmethod
    def set_boundary_conditions(
        self,
        grid: np.ndarray,
        x_vec: np.ndarray,
        s_vec: np.ndarray,
        t_idx: int,
        tau: float,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
    ) -> None:
        """Set boundary conditions at the spatial edges for a given time step."""
        pass

    def get_critical_points(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> List[float]:
        """Return a list of critical prices (e.g., strikes) for grid concentration."""
        points = []
        if hasattr(product, "strike") and product.strike > 0:
            points.append(product.strike)
        return points

    def _solve(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> PDESolutionResult:
        """
        Core PDE solving logic shared by price() and calculate_greeks().

        This method contains all the common grid building, terminal condition setup,
        and time stepping logic. Subclasses can override this method to implement
        custom solving strategies (e.g., two-surface approach for Snowball).

        Args:
            product: The equity product to price
            pricing_env: Pricing environment with market data

        Returns:
            PDESolutionResult containing solution vector and grid data

        Note:
            This method assumes tau > 0 (not expired). Callers should handle
            the expired case before calling _solve().
        """
        spot = pricing_env.spot
        tau = product.get_maturity(pricing_env)

        strike = getattr(product, "strike", spot)
        r = pricing_env.get_rate(tau)
        q = pricing_env.get_div_yield(tau)
        sigma = pricing_env.get_vol(strike, tau)

        # 1. Discretization
        x_vec, s_vec, dx_vec, t_vec, dt_vec = self._build_grids(
            product, pricing_env, spot, sigma, tau, r, q
        )

        # 2. Setup System
        grid = np.zeros((len(x_vec), len(t_vec)))
        self.set_terminal_condition(grid, x_vec, s_vec, product, pricing_env)

        l, c, u = self._calculate_coefficients(r, q, sigma, dx_vec, len(x_vec))
        A = self._build_operator_matrix(l, c, u, len(x_vec))

        # 3. Solve
        self._time_stepping(
            grid,
            A,
            l,
            u,
            x_vec,
            s_vec,
            t_vec,
            dt_vec,
            product,
            pricing_env,
            r,
            q,
            sigma,
            tau,
        )

        return PDESolutionResult(
            solution_vec=grid[:, 0],
            x_vec=x_vec,
            s_vec=s_vec,
            spot_log=np.log(spot),
        )

    def price(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        """Price the option using the PDE finite difference method."""
        spot = pricing_env.spot
        tau = product.get_maturity(pricing_env)

        if tau <= 0:
            return self._calculate_intrinsic(product, spot)

        result = self._solve(product, pricing_env)
        return self._interpolate_price(result.solution_vec, result.x_vec, result.spot_log)

    def calculate_greeks(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> Dict[str, float]:
        """Calculate Delta and Gamma directly from the PDE solution surface."""
        spot = pricing_env.spot
        tau = product.get_maturity(pricing_env)

        if tau <= 0:
            return {
                "price": self._calculate_intrinsic(product, spot),
                "delta": self._intrinsic_delta(product, spot),
                "gamma": 0.0,
            }

        result = self._solve(product, pricing_env)
        price = self._interpolate_price(result.solution_vec, result.x_vec, result.spot_log)
        delta, gamma = self._calculate_delta_gamma(
            result.solution_vec, result.x_vec, result.spot_log, spot
        )

        return {"price": price, "delta": delta, "gamma": gamma}

    def _build_grids(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        spot: float,
        sigma: float,
        tau: float,
        r: float,
        q: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Construct the spatial and temporal grids for solving the PDE."""
        if not self._is_cache_enabled():
            return self._build_grids_uncached(
                product, pricing_env, spot, sigma, tau, r, q
            )

        cache_key = self._grid_cache_key(product, pricing_env, spot, sigma, tau, r, q)
        with self.__class__._cache_lock:
            cached = self._grid_cache.get(cache_key)
            if cached is not None:
                self._grid_cache.move_to_end(cache_key)
                return cached

        barriers = self._get_barriers(product)
        s_min, s_max = self._resolve_spatial_bounds(
            product, spot, sigma, tau, r, q, barriers
        )
        critical_points = self._resolve_critical_points(
            product, pricing_env, spot, barriers
        )

        grid_size, use_adaptive = self._resolve_spatial_config(
            barriers, critical_points, s_min, s_max
        )

        x_vec, s_vec, dx_vec = SpatialGrid.build(
            s_min,
            s_max,
            grid_size,
            critical_points=critical_points,
            use_adaptive=use_adaptive,
        )

        t_vec, dt_vec = self._resolve_time_grid(product, tau, barriers)

        result = (x_vec, s_vec, dx_vec, t_vec, dt_vec)
        with self.__class__._cache_lock:
            self._grid_cache[cache_key] = result
            self._grid_cache.move_to_end(cache_key)
            if len(self._grid_cache) > self._grid_cache_max_entries:
                self._grid_cache.popitem(last=False)
        return result

    def _build_grids_uncached(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        spot: float,
        sigma: float,
        tau: float,
        r: float,
        q: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        barriers = self._get_barriers(product)
        s_min, s_max = self._resolve_spatial_bounds(
            product, spot, sigma, tau, r, q, barriers
        )
        critical_points = self._resolve_critical_points(
            product, pricing_env, spot, barriers
        )
        grid_size, use_adaptive = self._resolve_spatial_config(
            barriers, critical_points, s_min, s_max
        )
        x_vec, s_vec, dx_vec = SpatialGrid.build(
            s_min,
            s_max,
            grid_size,
            critical_points=critical_points,
            use_adaptive=use_adaptive,
        )
        t_vec, dt_vec = self._resolve_time_grid(product, tau, barriers)
        return x_vec, s_vec, dx_vec, t_vec, dt_vec

    def _resolve_spatial_bounds(
        self, product, spot, sigma, tau, r, q, barriers
    ) -> Tuple[float, float]:
        """Determine domain boundaries, optionally clamping to knock-out barriers."""
        params: PDEParams = self.params

        # Base bounds calculation
        if params.s_min > 0 and params.s_max > 0:
            s_min, s_max = params.s_min, params.s_max
        else:
            strike = getattr(product, "strike", spot)
            s_min, s_max = SpatialGrid.calculate_auto_bounds(
                spot, sigma, tau, r, q, strike=strike, barriers=barriers
            )
            s_min = params.s_min if params.s_min > 0 else s_min
            s_max = params.s_max if params.s_max > 0 else s_max

        # Continuous knock-out barriers define the survival domain boundaries
        obs_type = getattr(product, "observation_type", None)
        is_ko = getattr(product, "is_knock_out", False)

        if obs_type == ObservationType.CONTINUOUS and is_ko:
            if hasattr(product, "lower_barrier") and hasattr(product, "upper_barrier"):
                lb, ub = getattr(product, "lower_barrier", 0), getattr(
                    product, "upper_barrier", 0
                )
                if lb > 0 and params.s_min <= 0:
                    s_min = float(lb)
                if ub > 0 and params.s_max <= 0:
                    s_max = float(ub)
            elif hasattr(product, "barrier"):
                b = getattr(product, "barrier", 0)
                if b > 0:
                    if getattr(product, "is_up_barrier", False):
                        if params.s_max <= 0:
                            s_max = float(b)
                    elif params.s_min <= 0:
                        s_min = float(b)

        if s_min >= s_max:
            raise PricingError(f"Invalid spatial bounds: [{s_min}, {s_max}]")

        if params.barrier_domain_expand > 0.0 and barriers:
            expand = params.barrier_domain_expand
            for barrier in barriers:
                if barrier is None or barrier <= 0:
                    continue
                if barrier >= spot:
                    s_max = max(s_max, barrier * (1.0 + expand))
                else:
                    s_min = min(s_min, barrier * max(1.0 - expand, 1e-6))
            if s_min >= s_max:
                raise PricingError(
                    f"Invalid spatial bounds after expansion: [{s_min}, {s_max}]"
                )

        return s_min, s_max

    def _resolve_critical_points(
        self, product, pricing_env, spot, barriers
    ) -> List[float]:
        """Aggregate all key price levels for grid concentration."""
        raw_points = self._get_cached_critical_points(product, pricing_env, spot)
        return self._merge_critical_points(raw_points, spot, barriers)

    def _merge_critical_points(
        self, raw_points: Tuple[float, ...], spot: float, barriers: List[float]
    ) -> List[float]:
        """Merge raw critical points with dynamic points (spot/barriers)."""
        params: PDEParams = self.params
        points = list(raw_points)

        if params.auto_grid:
            if params.include_spot_in_critical_points and spot > 0:
                points.append(spot)
            points.extend([b for b in barriers if b is not None and b > 0])
            if params.barrier_refine_log_width > 0.0 and params.barrier_refine_levels > 0:
                for barrier in [b for b in barriers if b is not None and b > 0]:
                    for level in range(1, params.barrier_refine_levels + 1):
                        width = params.barrier_refine_log_width * level
                        points.append(barrier * math.exp(width))
                        points.append(barrier * math.exp(-width))

        return sorted(list(set([p for p in points if p is not None and p > 0])))

    def _resolve_spatial_config(
        self, barriers, critical_points, s_min, s_max
    ) -> Tuple[int, bool]:
        """Determine required grid size and whether to use adaptive spacing."""
        params: PDEParams = self.params
        size = params.grid_size
        adaptive = params.adaptive_grid

        if not params.auto_grid:
            return size, adaptive

        if len(barriers) > 0 or len(critical_points) > 1:
            adaptive = True

        if adaptive and critical_points:
            active = [p for p in critical_points if s_min < p < s_max]
            if active:
                x_points = np.log(np.array([s_min] + sorted(active) + [s_max]))
                total_range = x_points[-1] - x_points[0]
                # Heuristic: ensure we have at least capacity to resolve sensitive regions
                suggested = int(total_range / (params.log_dx_target * 2.0))
                size = min(max(size, suggested), params.max_grid_size)

        return size, adaptive

    def _resolve_time_grid(
        self, product, tau, barriers
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Determine time grid type and number of steps."""
        params: PDEParams = self.params
        obs_type = getattr(product, "observation_type", None)
        has_barriers = len(barriers) > 0
        event_times = self._get_event_times(product, tau)

        if not params.auto_grid:
            return TimeGrid.build(
                tau,
                params.time_steps,
                method=params.time_grid_type,
                event_times=event_times,
                grade_exponent=params.grade_exponent,
            )

        # Logic for suggested time steps
        days = max(1, int(round(tau * float(params.bus_days_in_year))))
        suggested = days
        if has_barriers:
            if obs_type == ObservationType.CONTINUOUS:
                suggested = int(round(params.event_steps_per_day * float(days)))
            else:
                suggested = int(round(2.0 * float(days)))
        elif getattr(product, "exercise_type", None) == ExerciseType.AMERICAN:
            suggested = int(round(1.5 * float(days)))

        steps = min(max(params.time_steps, suggested), params.max_time_steps)

        # Decide method
        method = "uniform"
        if event_times and (has_barriers or obs_type == ObservationType.DISCRETE):
            method = "event_aligned"
        elif params.time_grid_type != "uniform":
            method = params.time_grid_type

        return TimeGrid.build(
            tau,
            steps,
            method=method,
            event_times=event_times,
            grade_exponent=params.grade_exponent,
            steps_per_day=params.event_steps_per_day,
            day_count=params.bus_days_in_year,
            min_steps_per_interval=params.event_min_steps_per_interval,
            max_steps_total=params.max_time_steps,
        )

    def _calculate_coefficients(
        self, r: float, q: float, sigma: float, dx_vec: np.ndarray, num_x: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate coefficients (l, c, u) for the tridiagonal FD system."""
        mu = r - q - 0.5 * sigma * sigma
        D = 0.5 * sigma * sigma

        if (np.max(dx_vec) - np.min(dx_vec)) < 1e-10:
            dx = dx_vec[0]
            l = np.full(num_x, D / (dx * dx) - mu / (2.0 * dx))
            c = np.full(num_x, -2.0 * D / (dx * dx) - r)
            u = np.full(num_x, D / (dx * dx) + mu / (2.0 * dx))
        else:
            h_m, h_p = dx_vec[:-1], dx_vec[1:]
            h_sum, h_prod = h_m + h_p, h_m * h_p

            l_diff, c_diff, u_diff = (
                2.0 * D / (h_m * h_sum),
                -2.0 * D / h_prod,
                2.0 * D / (h_p * h_sum),
            )
            l_drift, c_drift, u_drift = (
                -mu * h_p / (h_m * h_sum),
                mu * (h_p - h_m) / h_prod,
                mu * h_m / (h_p * h_sum),
            )

            l, c, u = np.zeros(num_x), np.zeros(num_x), np.zeros(num_x)
            l[1:-1], c[1:-1], u[1:-1] = (
                l_diff + l_drift,
                c_diff + c_drift - r,
                u_diff + u_drift,
            )
            l[0], c[0], u[0] = l[1], c[1], u[1]
            l[-1], c[-1], u[-1] = l[-2], c[-2], u[-2]

        return l, c, u

    def _build_operator_matrix(
        self, l: np.ndarray, c: np.ndarray, u: np.ndarray, num_x: int
    ) -> sp.csc_matrix:
        """Construct the sparse spatial operator matrix A."""
        diagonals = [l[2:-1], c[1:-1], u[1:-2]]
        return sp.diags(
            diagonals, offsets=[-1, 0, 1], shape=(num_x - 2, num_x - 2), format="csc"
        )

    def _time_stepping(
        self,
        grid: np.ndarray,
        A: sp.csc_matrix,
        l: np.ndarray,
        u: np.ndarray,
        x_vec: np.ndarray,
        s_vec: np.ndarray,
        t_vec: np.ndarray,
        dt_vec: np.ndarray,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        r: float,
        q: float,
        sigma: float,
        tau: float,
    ) -> None:
        """Backward time stepping using the Crank-Nicolson scheme."""
        params: PDEParams = self.params
        num_t, num_x = len(t_vec), len(x_vec)
        I_int = sp.eye(num_x - 2, format="csc")
        self._matrix_cache.clear()

        # Rannacher smoothing at events
        smooth_js = set()
        if params.use_rannacher and params.auto_grid and params.rannacher_at_events:
            for et in self._get_event_times(product, tau) or []:
                idx = int(np.argmin(np.abs(t_vec - et)))
                if 0 < idx < num_t - 1 and is_close(float(t_vec[idx]), float(et)):
                    smooth_js.update(
                        [
                            idx - 1 - k
                            for k in range(params.rannacher_steps)
                            if idx - 1 - k >= 0
                        ]
                    )

        for j in range(num_t - 2, -1, -1):
            dt = dt_vec[j]
            steps_from_end = num_t - 1 - j

            # Smoothing uses theta=1.0 (Backward Euler)
            theta = (
                1.0
                if params.use_rannacher
                and (steps_from_end < params.rannacher_steps or j in smooth_js)
                else params.theta
            )

            M1, M2_lu = self._get_matrices(I_int, A, dt, theta)
            self.set_boundary_conditions(
                grid, x_vec, s_vec, j, tau - t_vec[j], product, pricing_env
            )

            # Solve system for interior points
            rhs = M1 @ grid[1:-1, j + 1]
            self._inject_boundary_contributions(rhs, grid, l, u, j, dt, theta)
            grid[1:-1, j] = M2_lu.solve(rhs)

            self._apply_step_modifications(
                grid, x_vec, s_vec, j, tau - t_vec[j], product, pricing_env
            )

    def _inject_boundary_contributions(self, rhs, grid, l, u, j, dt, theta) -> None:
        """Add Dirichlet boundary terms to the RHS of the reduced interior system."""
        if len(grid) > 2:
            # Terms from V[0]
            rhs[0] += dt * (
                (1.0 - theta) * l[1] * grid[0, j + 1] + theta * l[1] * grid[0, j]
            )
            # Terms from V[-1]
            rhs[-1] += dt * (
                (1.0 - theta) * u[-2] * grid[-1, j + 1] + theta * u[-2] * grid[-1, j]
            )

    def _get_matrices(
        self, I: sp.csc_matrix, A: sp.csc_matrix, dt: float, theta: float
    ) -> Tuple[sp.csc_matrix, spla.SuperLU]:
        """
        Get or compute matrices for time stepping.

        Caches LU factorizations for efficiency when dt and theta repeat.

        Args:
            I: Identity matrix
            A: Spatial operator matrix
            dt: Time step size
            theta: Scheme parameter (0.5 = CN, 1.0 = BE)

        Returns:
            Tuple of (M1, M2_lu) where:
                M1 = I + (1-theta)*dt*A (for RHS)
                M2_lu = LU factorization of I - theta*dt*A (for LHS)
        """
        # Round dt to avoid floating point comparison issues
        key = (round(dt, 12), round(theta, 6))

        if self._is_cache_enabled() and key in self._matrix_cache:
            return self._matrix_cache[key]

        # Build matrices
        M1 = I + (1.0 - theta) * dt * A
        M2 = I - theta * dt * A

        # LU factorization of M2
        try:
            M2_lu = spla.splu(M2)
        except Exception as e:
            raise NumericalError(f"Failed to factorize matrix: {e}")

        if self._is_cache_enabled():
            self._matrix_cache[key] = (M1, M2_lu)
        return M1, M2_lu

    def _apply_step_modifications(
        self,
        grid: np.ndarray,
        x_vec: np.ndarray,
        s_vec: np.ndarray,
        t_idx: int,
        tau: float,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
    ) -> None:
        """
        Apply product-specific modifications after each time step.

        Override this method for American options (early exercise)
        or barrier options (barrier checks).

        Args:
            grid: Solution grid
            x_vec: Log-price grid points
            s_vec: Price grid points
            t_idx: Current time index
            tau: Time remaining to maturity
            product: The option product
            pricing_env: Pricing environment
        """
        pass  # Default: no modifications

    def _interpolate_price(
        self, v_vec: np.ndarray, x_vec: np.ndarray, x_target: float
    ) -> float:
        """
        Interpolate option value at target log-price.

        Uses linear interpolation between nearest grid points.

        Args:
            v_vec: Option values at grid points
            x_vec: Log-price grid points
            x_target: Target log-price

        Returns:
            Interpolated option value
        """
        return float(np.interp(x_target, x_vec, v_vec))

    def _calculate_delta_gamma(
        self, v_vec: np.ndarray, x_vec: np.ndarray, x_target: float, spot: float
    ) -> Tuple[float, float]:
        """
        Calculate delta and gamma from the solution vector.

        In log-space:
            dV/dS = (1/S) * dV/dx
            d2V/dS2 = (1/S^2) * (d2V/dx^2 - dV/dx)

        Args:
            v_vec: Option values at grid points
            x_vec: Log-price grid points
            x_target: Target log-price (ln(spot))
            spot: Current spot price

        Returns:
            Tuple of (delta, gamma)
        """
        # Find nearest grid points
        idx = np.searchsorted(x_vec, x_target)
        idx = max(1, min(idx, len(x_vec) - 2))

        # Local grid spacing
        dx_left = x_vec[idx] - x_vec[idx - 1]
        dx_right = x_vec[idx + 1] - x_vec[idx]
        dx_avg = (dx_left + dx_right) / 2.0

        # Central differences for derivatives in log-space
        dv_dx = (v_vec[idx + 1] - v_vec[idx - 1]) / (dx_left + dx_right)
        d2v_dx2 = (v_vec[idx + 1] - 2 * v_vec[idx] + v_vec[idx - 1]) / (dx_avg**2)

        # Convert to price-space derivatives
        delta = dv_dx / spot
        gamma = (d2v_dx2 - dv_dx) / (spot**2)

        return delta, gamma

    def _calculate_intrinsic(self, product: BaseEquityProduct, spot: float) -> float:
        """
        Calculate intrinsic value of the option.

        Args:
            product: The option product
            spot: Current spot price

        Returns:
            Intrinsic value
        """
        if hasattr(product, "get_payoff"):
            return product.get_payoff(spot)
        return 0.0

    def _intrinsic_delta(self, product: BaseEquityProduct, spot: float) -> float:
        """
        Calculate delta of intrinsic value.

        Args:
            product: The option product
            spot: Current spot price

        Returns:
            Intrinsic delta
        """
        multiplier = getattr(product, "contract_multiplier", 1.0)
        if hasattr(product, "is_call") and hasattr(product, "strike"):
            if product.is_call():
                delta = 1.0 if spot > product.strike else 0.0
            else:
                delta = -1.0 if spot < product.strike else 0.0
            return delta * multiplier
        return 0.0 * multiplier

    def _get_barriers(self, product: BaseEquityProduct) -> List[float]:
        """Helper to collect barrier levels from known product attributes."""
        barriers = []
        for attr in ("barrier", "upper_barrier", "lower_barrier"):
            if hasattr(product, attr):
                val = getattr(product, attr)
                if val is not None and val > 0:
                    barriers.append(val)
        return barriers

    def _get_event_times(
        self, product: BaseEquityProduct, tau: float
    ) -> Optional[List[float]]:
        """Helper to collect observation/event times."""
        schedule = getattr(product, "observation_schedule", None)
        if schedule is not None and getattr(schedule, "times", None):
            return [t for t in schedule.times if 0 < t < tau]
        for attr in ("observation_dates", "obs_times", "event_times"):
            if hasattr(product, attr):
                times = getattr(product, attr)
                if times:
                    return [t for t in times if 0 < t < tau]
        return None
    def _critical_points_cache_key(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        spot: float,
    ) -> Tuple:
        strategy = self._resolve_cache_strategy()
        return (
            strategy,
            f"{product.__class__.__module__}.{product.__class__.__qualname__}",
            self._product_cache_token(product, strategy),
            round(spot, 12),
            pricing_env.valuation_date,
            pricing_env.day_count_convention,
            pricing_env.bus_days_in_year,
        )

    def _get_cached_critical_points(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        spot: float,
    ) -> Tuple[float, ...]:
        if not self._is_cache_enabled():
            return tuple(
                sorted(
                    [
                        round(p, 12)
                        for p in self.get_critical_points(product, pricing_env)
                        if p is not None
                    ]
                )
            )
        key = self._critical_points_cache_key(product, pricing_env, spot)
        cached = self._critical_points_cache.get(key)
        if cached is not None:
            self._critical_points_cache.move_to_end(key)
            return cached
        points = tuple(
            sorted(
                [
                    round(p, 12)
                    for p in self.get_critical_points(product, pricing_env)
                    if p is not None
                ]
            )
        )
        self._critical_points_cache[key] = points
        self._critical_points_cache.move_to_end(key)
        max_entries = max(1, self._grid_cache_max_entries)
        if len(self._critical_points_cache) > max_entries:
            self._critical_points_cache.popitem(last=False)
        return points
