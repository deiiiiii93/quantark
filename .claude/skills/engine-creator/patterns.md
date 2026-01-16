# Engine Implementation Patterns

Detailed patterns extracted from the QuantArk codebase for implementing pricing engines.

## Base Engine Interface

```python
from abc import ABC, abstractmethod
from typing import Dict, Optional
from asset.equity.product.base_equity_product import BaseEquityProduct
from priceenv import PricingEnvironment
from asset.equity.param import EngineParams


class BaseEngine(ABC):
    """Abstract base class for all pricing engines."""

    def __init__(self, params: Optional[EngineParams] = None):
        self.params = params if params is not None else EngineParams()

    @abstractmethod
    def price(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        """Calculate the price of the product."""
        pass

    def calculate_greeks(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> Dict[str, float]:
        """Calculate Greeks using finite difference (default)."""
        # Default bump-and-reprice implementation
        pass
```

## Analytical Engine Pattern

### Simple Closed-Form Engine

```python
from typing import Dict, Optional
import math
from scipy.stats import norm

from asset.equity.engine.base_engine import BaseEngine
from asset.equity.product.option import EuropeanVanillaOption
from priceenv import PricingEnvironment
from util.exceptions import PricingError, ValidationError
from util.numerical import is_zero, safe_log, safe_sqrt, safe_exp


class MyAnalyticalEngine(BaseEngine):
    """
    Analytical pricing engine for [Product Name].

    Implements [Formula Name] closed-form solution.

    References:
        - [Author, Year, Paper Title]
    """

    def price(
        self, product: EuropeanVanillaOption, pricing_env: PricingEnvironment
    ) -> float:
        """
        Calculate price using closed-form formula.

        Args:
            product: The option to price
            pricing_env: Market data environment

        Returns:
            Option price

        Raises:
            PricingError: If product type not supported
            ValidationError: If parameters invalid
        """
        # Step 1: Validate product type
        if not isinstance(product, EuropeanVanillaOption):
            raise PricingError(
                f"MyAnalyticalEngine only supports EuropeanVanillaOption, "
                f"got {type(product).__name__}"
            )

        # Step 2: Extract market data
        S = pricing_env.spot
        K = product.strike
        T = product.get_maturity(pricing_env)
        r = pricing_env.rate
        q = pricing_env.dividend_yield
        sigma = pricing_env.volatility

        # Step 3: Handle edge cases
        if is_zero(T):
            return product.get_payoff(S)

        if is_zero(sigma):
            # Zero vol: deterministic forward
            forward = S * safe_exp((r - q) * T)
            df = safe_exp(-r * T)
            return df * product.get_payoff(forward)

        # Step 4: Calculate price using formula
        sigma_sqrt_t = sigma * safe_sqrt(T)
        d1 = (safe_log(S / K) + (r - q + 0.5 * sigma**2) * T) / sigma_sqrt_t
        d2 = d1 - sigma_sqrt_t

        if product.is_call():
            price = (
                S * safe_exp(-q * T) * norm.cdf(d1)
                - K * safe_exp(-r * T) * norm.cdf(d2)
            )
        else:
            price = (
                K * safe_exp(-r * T) * norm.cdf(-d2)
                - S * safe_exp(-q * T) * norm.cdf(-d1)
            )

        # Step 5: Validate result
        intrinsic = product.intrinsic_value(S)
        if price < intrinsic - 1e-10:
            raise PricingError(f"Price {price} below intrinsic {intrinsic}")

        return price

    def calculate_greeks(
        self, product: EuropeanVanillaOption, pricing_env: PricingEnvironment
    ) -> Dict[str, float]:
        """Calculate analytical Greeks."""
        # ... analytical formulas for delta, gamma, vega, theta, rho
        pass
```

### Multi-Method Analytical Engine

```python
from typing import Optional, Union
from util.enum.engine_enums import EngineType, MyMethodEnum


class MultiMethodAnalyticalEngine(BaseEngine):
    """
    Analytical engine supporting multiple pricing methods.

    Methods:
        - METHOD_A: [Description]
        - METHOD_B: [Description]
        - METHOD_C: [Description]
    """

    def __init__(
        self,
        method: Optional[Union[tuple, MyMethodEnum, str]] = None,
        params: Optional[EngineParams] = None,
    ):
        super().__init__(params)
        self.method = self._resolve_method(method)

    def _resolve_method(
        self, method: Optional[Union[tuple, MyMethodEnum, str]]
    ) -> MyMethodEnum:
        """Resolve method from various input formats."""
        if method is None:
            return MyMethodEnum.DEFAULT

        # Two-level enum: EngineType.ANALYTICAL(MyMethodEnum.XYZ)
        if isinstance(method, tuple):
            engine_type, specific_method = method
            if not isinstance(specific_method, MyMethodEnum):
                raise ValidationError(f"Invalid method: {specific_method}")
            return specific_method

        # Direct enum
        if isinstance(method, MyMethodEnum):
            return method

        # String (backward compatible)
        if isinstance(method, str):
            try:
                return MyMethodEnum(method)
            except ValueError:
                raise ValidationError(
                    f"Unknown method '{method}'. "
                    f"Valid: {[m.value for m in MyMethodEnum]}"
                )

        raise ValidationError(f"Invalid method type: {type(method)}")

    def price(self, product, pricing_env) -> float:
        """Price using selected method."""
        if self.method == MyMethodEnum.METHOD_A:
            return self._price_method_a(product, pricing_env)
        elif self.method == MyMethodEnum.METHOD_B:
            return self._price_method_b(product, pricing_env)
        elif self.method == MyMethodEnum.METHOD_C:
            return self._price_method_c(product, pricing_env)
        else:
            raise PricingError(f"Unknown method: {self.method}")

    def _price_method_a(self, product, pricing_env) -> float:
        """Implementation of Method A."""
        pass

    def _price_method_b(self, product, pricing_env) -> float:
        """Implementation of Method B."""
        pass

    def _price_method_c(self, product, pricing_env) -> float:
        """Implementation of Method C."""
        pass
```

## Monte Carlo Engine Pattern

```python
from dataclasses import dataclass
from typing import Optional, Union
import numpy as np

from asset.equity.engine.base_engine import BaseEngine
from asset.equity.param import MCParams
from asset.equity.process.bsm import GBMPathGenerator
from util.enum.engine_enums import EngineType, MonteCarloMethod


@dataclass
class MCResult:
    """Container for Monte Carlo results."""
    price: float
    std_error: float
    confidence_interval: tuple
    paths_used: int
    convergence_achieved: bool


class MyMCEngine(BaseEngine):
    """
    Monte Carlo pricing engine for [Product Name].

    Methods:
        - PSEUDO: Standard pseudorandom simulation
        - QUASI: Quasi-Monte Carlo with Sobol sequences
        - RANDOMIZED_QUASI: Randomized QMC with batching
    """

    def __init__(
        self,
        method: Optional[Union[tuple, MonteCarloMethod, str]] = None,
        params: Optional[MCParams] = None,
        use_dask: bool = False,
        num_batches: int = 4,
    ):
        super().__init__(params or MCParams())
        self.method = self._resolve_method(method)
        self.use_dask = use_dask
        self.num_batches = num_batches
        self._last_result: Optional[MCResult] = None

    def _resolve_method(self, method) -> MonteCarloMethod:
        """Resolve MC method from various input formats."""
        if method is None:
            return MonteCarloMethod.PSEUDO

        if isinstance(method, tuple):
            _, specific_method = method
            return specific_method

        if isinstance(method, MonteCarloMethod):
            return method

        if isinstance(method, str):
            return MonteCarloMethod(method)

        raise ValidationError(f"Invalid method: {method}")

    def price(self, product, pricing_env) -> float:
        """
        Price using Monte Carlo simulation.

        Returns:
            Expected discounted payoff
        """
        # Validate product type
        if not isinstance(product, SupportedProductType):
            raise PricingError(f"Unsupported product: {type(product)}")

        # Extract parameters
        S0 = pricing_env.spot
        r = pricing_env.rate
        q = pricing_env.dividend_yield
        sigma = pricing_env.volatility
        T = product.get_maturity(pricing_env)

        # Handle edge case
        if is_zero(T):
            return product.get_payoff(S0)

        # Generate paths
        paths = self._generate_paths(S0, r, q, sigma, T)

        # Calculate payoffs
        payoffs = self._calculate_payoffs(paths, product, pricing_env)

        # Discount and average
        df = np.exp(-r * T)
        price = df * np.mean(payoffs)
        std_error = df * np.std(payoffs) / np.sqrt(len(payoffs))

        # Store result
        self._last_result = MCResult(
            price=price,
            std_error=std_error,
            confidence_interval=(price - 1.96 * std_error, price + 1.96 * std_error),
            paths_used=len(payoffs),
            convergence_achieved=True,
        )

        return price

    def _generate_paths(self, S0, r, q, sigma, T) -> np.ndarray:
        """Generate price paths using selected method."""
        generator = GBMPathGenerator(
            num_paths=self.params.num_paths,
            num_steps=self.params.time_steps,
            use_sobol=(self.method != MonteCarloMethod.PSEUDO),
            randomize=(self.method == MonteCarloMethod.RANDOMIZED_QUASI),
            seed=self.params.seed,
        )
        return generator.generate_paths(S0, r, q, sigma, T)

    def _calculate_payoffs(self, paths, product, pricing_env) -> np.ndarray:
        """Calculate payoffs for all paths."""
        # Vectorized payoff calculation
        final_prices = paths[:, -1]
        payoffs = np.array([product.get_payoff(s) for s in final_prices])
        return payoffs

    def get_last_result(self) -> Optional[MCResult]:
        """Get detailed result from last pricing."""
        return self._last_result
```

## PDE Solver Pattern

### Base PDE Solver

```python
from abc import abstractmethod
from collections import OrderedDict
import threading
from typing import Dict, List, Optional, Tuple, NamedTuple
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from asset.equity.engine.base_engine import BaseEngine
from asset.equity.product.base_equity_product import BaseEquityProduct
from asset.equity.param import PDEParams
from priceenv import PricingEnvironment
from util.exceptions import PricingError, NumericalError
from util.enum.engine_enums import EngineType

from .time_grid import TimeGrid
from .spatial_grid import SpatialGrid


class PDESolutionResult(NamedTuple):
    """Result from PDE solving containing solution and grid data."""
    solution_vec: np.ndarray
    x_vec: np.ndarray
    s_vec: np.ndarray
    spot_log: float


class BasePDESolver(BaseEngine):
    """
    Abstract base class for PDE-based pricing with high-performance caching.

    Implements Crank-Nicolson finite difference scheme with:
    - Log-price transformation (x = ln(S))
    - Tavella-Randall grid concentration
    - Rannacher smoothing for initial steps
    - Class-level shared grid cache for cross-instance performance
    - Thread-safe LRU caching with configurable strategies
    """

    engine_type = EngineType.PDE

    # Class-level shared cache for grid reuse across instances
    _shared_grid_cache: "OrderedDict[Tuple, Tuple]" = OrderedDict()
    _grid_cache_max_entries: int = 128
    _global_cache_enabled: bool = True
    _global_cache_strategy: Optional[str] = None
    _cache_lock = threading.Lock()

    def __init__(self, params: Optional[PDEParams] = None):
        super().__init__(params or PDEParams())
        self._matrix_cache: Dict[Tuple[float, float], Tuple] = {}
        self._grid_cache = self.__class__._shared_grid_cache
        self._cache_enabled = bool(getattr(self.params, "cache_enabled", True))
        self._cache_strategy = getattr(self.params, "cache_strategy", "standard")
        cache_size = getattr(self.params, "grid_cache_max_entries", None)
        if cache_size is not None:
            self.set_grid_cache_max_entries(cache_size)

    # === Cache Management Methods ===

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
        """
        Set the cache strategy for this solver class.

        Strategies:
        - "disable": No caching
        - "strict": Cache by object identity only
        - "standard": Use cache_key() if available, else dict hash
        - "aggressive": Fallback to repr() if dict hash fails
        """
        if strategy not in ("disable", "strict", "standard", "aggressive"):
            raise PricingError(f"Invalid cache_strategy: {strategy}")
        cls._global_cache_strategy = strategy
        if clear:
            cls.clear_grid_cache()

    @classmethod
    def set_grid_cache_max_entries(cls, max_entries: int) -> None:
        """Set maximum number of grid entries to keep in cache."""
        if max_entries <= 0:
            raise PricingError(f"Grid cache size must be positive")
        with cls._cache_lock:
            cls._grid_cache_max_entries = max_entries
            while len(cls._shared_grid_cache) > cls._grid_cache_max_entries:
                cls._shared_grid_cache.popitem(last=False)

    # === Core PDE Solving ===

    def _solve(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> PDESolutionResult:
        """
        Core PDE solving logic shared by price() and calculate_greeks().

        This method handles grid building (with caching), terminal condition
        setup, and backward time stepping.
        """
        spot = pricing_env.spot
        tau = product.get_maturity(pricing_env)

        strike = getattr(product, "strike", spot)
        r = pricing_env.get_rate(tau)
        q = pricing_env.get_div_yield(tau)
        sigma = pricing_env.get_vol(strike, tau)

        # Build grids (uses cache when enabled)
        x_vec, s_vec, dx_vec, t_vec, dt_vec = self._build_grids(
            product, pricing_env, spot, sigma, tau, r, q
        )

        # Setup solution grid
        grid = np.zeros((len(x_vec), len(t_vec)))
        self.set_terminal_condition(grid, x_vec, s_vec, product, pricing_env)

        # Build operator matrix
        l, c, u = self._calculate_coefficients(r, q, sigma, dx_vec, len(x_vec))
        A = self._build_operator_matrix(l, c, u, len(x_vec))

        # Backward time stepping
        self._time_stepping(
            grid, A, l, u, x_vec, s_vec, t_vec, dt_vec,
            product, pricing_env, r, q, sigma, tau,
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

    # === Grid Building with Caching ===

    def _build_grids(
        self, product, pricing_env, spot, sigma, tau, r, q
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Construct spatial and temporal grids with LRU caching.

        Cache key includes: product class, market data, barriers, event times.
        """
        if not self._is_cache_enabled():
            return self._build_grids_uncached(product, pricing_env, spot, sigma, tau, r, q)

        cache_key = self._grid_cache_key(product, pricing_env, spot, sigma, tau, r, q)
        with self.__class__._cache_lock:
            cached = self._grid_cache.get(cache_key)
            if cached is not None:
                self._grid_cache.move_to_end(cache_key)  # LRU update
                return cached

        # Build grids and cache
        result = self._build_grids_uncached(product, pricing_env, spot, sigma, tau, r, q)
        with self.__class__._cache_lock:
            self._grid_cache[cache_key] = result
            self._grid_cache.move_to_end(cache_key)
            if len(self._grid_cache) > self._grid_cache_max_entries:
                self._grid_cache.popitem(last=False)  # Evict oldest
        return result

    def _is_cache_enabled(self) -> bool:
        return self._resolve_cache_strategy() != "disable"

    def _resolve_cache_strategy(self) -> str:
        if not self.__class__._global_cache_enabled or not self._cache_enabled:
            return "disable"
        strategy = self.__class__._global_cache_strategy
        if strategy is None:
            strategy = self._cache_strategy
        return strategy

    def _grid_cache_key(
        self, product, pricing_env, spot, sigma, tau, r, q
    ) -> Tuple:
        """Generate cache key for grid lookup."""
        barriers = tuple(sorted([b for b in self._get_barriers(product) if b]))
        event_times = tuple(sorted([t for t in (self._get_event_times(product, tau) or [])]))
        critical_points = tuple(sorted(self.get_critical_points(product, pricing_env)))

        # Use product.cache_key() if available for content-based caching
        product_token = self._product_cache_token(product, self._resolve_cache_strategy())

        return (
            f"{product.__class__.__module__}.{product.__class__.__qualname__}",
            product_token,
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

    def _product_cache_token(self, product, strategy: str) -> Tuple[str, object]:
        """Generate cache token for product based on strategy."""
        if strategy == "strict":
            return ("id", id(product))
        key_fn = getattr(product, "cache_key", None)
        if callable(key_fn):
            return ("key", key_fn())
        if strategy in ("standard", "aggressive"):
            attrs = getattr(product, "__dict__", None)
            if attrs is not None:
                return ("dict", tuple(sorted(attrs.items())))
        return ("id", id(product))

    # === Abstract Methods ===

    @abstractmethod
    def set_terminal_condition(
        self, grid: np.ndarray, x_vec: np.ndarray, s_vec: np.ndarray,
        product: BaseEquityProduct, pricing_env: PricingEnvironment,
    ) -> None:
        """Set the terminal condition (payoff at maturity)."""
        pass

    @abstractmethod
    def set_boundary_conditions(
        self, grid: np.ndarray, x_vec: np.ndarray, s_vec: np.ndarray,
        t_idx: int, tau: float,
        product: BaseEquityProduct, pricing_env: PricingEnvironment,
    ) -> None:
        """Set boundary conditions at the spatial edges for a given time step."""
        pass

    def get_critical_points(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> List[float]:
        """Return critical prices for grid concentration (override in subclass)."""
        points = []
        if hasattr(product, "strike") and product.strike > 0:
            points.append(product.strike)
        return points

    def _get_barriers(self, product: BaseEquityProduct) -> List[float]:
        """Collect barrier levels from product attributes."""
        barriers = []
        for attr in ("barrier", "upper_barrier", "lower_barrier"):
            if hasattr(product, attr):
                val = getattr(product, attr)
                if val is not None and val > 0:
                    barriers.append(val)
        return barriers

    def _get_event_times(self, product, tau: float) -> Optional[List[float]]:
        """Collect observation/event times."""
        schedule = getattr(product, "observation_schedule", None)
        if schedule is not None and getattr(schedule, "times", None):
            return [t for t in schedule.times if 0 < t < tau]
        return None

    # Additional helper methods (_time_stepping, _interpolate_price, etc.)
    # omitted for brevity - see base_pde_solver.py for full implementation
```

### Concrete PDE Solver

```python
class EuropeanPDESolver(BasePDESolver):
    """PDE solver for European vanilla options."""

    def _validate_product(self, product) -> None:
        if not isinstance(product, EuropeanVanillaOption):
            raise PricingError(
                f"EuropeanPDESolver requires EuropeanVanillaOption"
            )

    def _set_terminal_condition(
        self, spatial_grid: SpatialGrid, product, pricing_env
    ) -> np.ndarray:
        """Set payoff at maturity."""
        S_vec = spatial_grid.s_vec
        return np.array([product.get_payoff(S) for S in S_vec])

    def _set_boundary_conditions(
        self, V: np.ndarray, spatial_grid: SpatialGrid,
        product, pricing_env, t: float
    ) -> np.ndarray:
        """Apply Dirichlet boundary conditions."""
        S_min, S_max = spatial_grid.s_vec[0], spatial_grid.s_vec[-1]
        T = product.get_maturity(pricing_env)
        r = pricing_env.rate
        q = pricing_env.dividend_yield
        K = product.strike

        if product.is_call():
            V[0] = 0.0  # Call worthless at S=0
            V[-1] = S_max * np.exp(-q * (T - t)) - K * np.exp(-r * (T - t))
        else:
            V[0] = K * np.exp(-r * (T - t))  # Put worth K at S=0
            V[-1] = 0.0  # Put worthless at S=inf

        return V
```

### American PDE Solver (with Early Exercise)

```python
class AmericanPDESolver(BasePDESolver):
    """PDE solver for American options with early exercise."""

    def _apply_early_exercise(
        self, V: np.ndarray, spatial_grid: SpatialGrid,
        product, pricing_env, t: float
    ) -> np.ndarray:
        """Apply early exercise constraint using projected SOR."""
        intrinsic = np.array([product.get_payoff(S) for S in spatial_grid.s_vec])
        return np.maximum(V, intrinsic)
```

## Product Caching Pattern

Products can implement a `cache_key()` method to enable content-based caching
in PDE and Quadrature engines. This allows cache reuse across multiple product
instances with identical parameters.

```python
from dataclasses import dataclass, fields


@dataclass
class MyProduct(BaseEquityProduct):
    """Product with cache_key() support for PDE caching."""
    strike: float
    maturity: float
    barrier: float
    # ... other fields

    def cache_key(self) -> tuple:
        """
        Generate a hashable key for caching based on product state.

        Uses dataclasses.fields() to automatically capture all field values,
        ensuring the cache key stays synchronized with the dataclass definition.
        """
        return tuple(
            (field.name, getattr(self, field.name))
            for field in fields(self)
        )
```

**Cache Strategy Behavior:**

| Strategy | `cache_key()` exists | No `cache_key()` |
|----------|---------------------|------------------|
| `strict` | Uses `id(product)` | Uses `id(product)` |
| `standard` | Uses `cache_key()` | Uses `__dict__` hash |
| `aggressive` | Uses `cache_key()` | Falls back to `repr()` |
| `disable` | No caching | No caching |

**Best Practices:**
- Implement `cache_key()` for products used in batch pricing scenarios
- Include all fields that affect pricing (strike, barriers, observation times)
- Exclude volatile fields (e.g., cached intermediate results)
- Return a hashable tuple of (field_name, value) pairs

## Quadrature Engine Pattern

Quadrature engines use FFT-based convolution for efficient pricing of
discretely-monitored path-dependent options.

### QuadParams

```python
@dataclass
class QuadParams(EngineParams):
    """
    Quadrature engine configuration.

    Attributes:
        grid_points: Number of integration points (default: 1001).
                     Must be odd for Simpson's rule.
        num_std_devs: Number of standard deviations for integration bounds (default: 10).
                      Larger values capture more of the distribution tail.
    """

    grid_points: int = 1001  # Odd number for Simpson's rule
    num_std_devs: float = 10.0

    def __post_init__(self):
        """Validate quadrature parameters."""
        super().__post_init__()
        if self.grid_points <= 0:
            raise ValidationError(f"grid_points must be positive")
        if self.grid_points < 100:
            raise ValidationError(f"grid_points should be at least 100 for accuracy")
        # Ensure odd number for Simpson's rule
        if self.grid_points % 2 == 0:
            self.grid_points += 1
        if self.num_std_devs < 3:
            raise ValidationError(f"num_std_devs should be at least 3 for accuracy")
```

### Quadrature Engine Template

```python
from typing import Optional, Sequence
import math
import numpy as np
from scipy.special import erfc

from asset.equity.engine.base_engine import BaseEngine
from asset.equity.engine.quad.quad_math import QuadratureMath
from asset.equity.param import QuadParams
from asset.equity.product.base_equity_product import BaseEquityProduct
from priceenv import PricingEnvironment
from util.enum.engine_enums import EngineType
from util.exceptions import PricingError, ValidationError
from util.numerical import Tolerance, is_zero, safe_log, validate_positive


class MyQuadEngine(BaseEngine):
    """
    Quadrature pricing engine for [Product Name].

    Uses FFT-based convolution for efficient backward recursion
    on a log-price grid with discrete observation handling.

    Key Components:
    - Log-price grid spanning multiple standard deviations
    - Simpson's rule integration weights
    - FFT convolution for diffusion operator
    - Two-surface approach for knock-in products (v_in, v_out)
    """

    engine_type = EngineType.QUADRATURE

    def __init__(self, params: Optional[QuadParams] = None) -> None:
        if params is None:
            params = QuadParams()
        if not isinstance(params, QuadParams):
            raise ValidationError(
                f"params must be QuadParams instance, got {type(params).__name__}"
            )
        super().__init__(params)

    def price(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        """
        Price using backward quadrature recursion.

        Returns:
            Option price at current spot
        """
        # Step 1: Validate product type
        if not isinstance(product, SupportedProductType):
            raise PricingError(
                f"MyQuadEngine only supports SupportedProductType, "
                f"got {type(product).__name__}"
            )

        # Step 2: Extract and validate market data
        spot = pricing_env.spot
        maturity = product.get_maturity(pricing_env)
        validate_positive(spot, "spot")
        if is_zero(maturity, tol=Tolerance.ZERO):
            return product.get_payoff(spot, pricing_env)

        rate = pricing_env.get_rate(maturity)
        div = pricing_env.get_div_yield(maturity)
        vol = pricing_env.get_vol(product.strike, maturity)
        validate_positive(vol, "volatility")

        # Step 3: Setup log-price grid using QuadratureMath
        math_utils = QuadratureMath(
            grid_x=self.params.grid_points,
            spot=spot,
            maturity=maturity,
            vol_max=vol,
        )
        grid = math_utils.grid  # Log-price grid centered at 0
        spot_grid = spot * np.exp(grid)  # Price grid

        # Step 4: Build observation time grid
        observation_times = self._get_observation_times(product, pricing_env)
        dt = self._build_dt(observation_times)

        # Step 5: Compute diffusion parameters
        tau = 0.5 * vol * vol * dt  # Variance per step
        alpha = (rate - div - 0.5 * vol * vol) / (vol * vol)
        beta = ((rate - div - 0.5 * vol * vol) ** 2 / (vol ** 4)
                + 2.0 * rate / (vol * vol))

        # Step 6: Initialize terminal values
        # For knock-in products, use two surfaces:
        v_in = np.array([product.get_payoff_knocked_in(s) for s in spot_grid])
        v_out = np.array([product.get_payoff_not_knocked_in(s) for s in spot_grid])

        # Step 7: Backward recursion through observation times
        full_p_lr, full_p_ur, full_p0 = 0, len(grid) - 1, (len(grid) - 1) % 2
        omega_grid = math_utils.z_grid

        for step_index in range(len(observation_times), 0, -1):
            obs_time = observation_times[step_index - 1]

            # Apply barrier logic at observation
            self._apply_observation_logic(
                v_in, v_out, spot_grid, product, obs_time, pricing_env
            )

            # Diffusion step (FFT convolution)
            tau_step = float(tau[step_index])
            prefactor = math.exp(-beta * tau_step) / math.sqrt(math.pi * tau_step) / 2.0
            omega_array = np.exp(
                -(omega_grid ** 2) / (4.0 * tau_step) - alpha * omega_grid
            )

            v_in = self._diffuse_fft(
                v_in, math_utils, omega_array, prefactor,
                full_p_lr, full_p_ur, full_p0, alpha, beta, tau_step
            )
            v_out = self._diffuse_fft(
                v_out, math_utils, omega_array, prefactor,
                full_p_lr, full_p_ur, full_p0, alpha, beta, tau_step
            )

        # Step 8: Interpolate to spot (x=0 in log-space)
        return math_utils.interpolate(v_out, x=0.0)

    def _diffuse_fft(
        self,
        values: np.ndarray,
        math_utils: QuadratureMath,
        omega_array: np.ndarray,
        prefactor: float,
        p_lr: int,
        p_ur: int,
        p0: int,
        alpha: float,
        beta: float,
        tau_step: float,
    ) -> np.ndarray:
        """
        Apply diffusion operator using FFT convolution.

        Uses Simpson's rule weights for integration accuracy
        and erfc-based tail corrections.
        """
        # Apply Simpson weights
        u_array = math_utils.simpson_weights(values, p_lr, p_ur, p0)

        # FFT convolution
        conv = math_utils.convolution_fft(omega_array, u_array)
        base = prefactor * conv

        # Add tail correction for truncated domain
        return base + self._tail_correction(values, math_utils, alpha, beta, tau_step)

    def _tail_correction(
        self,
        values: np.ndarray,
        math_utils: QuadratureMath,
        alpha: float,
        beta: float,
        tau_step: float,
    ) -> np.ndarray:
        """
        Compute tail integral contribution using erfc approximation.

        Accounts for probability mass outside the truncated grid domain.
        """
        grid = math_utils.grid
        x_min, x_max = grid[0], grid[-1]
        sqrt_tau = math.sqrt(tau_step)

        u_left = (grid - x_min + 2.0 * tau_step * alpha) / (2.0 * sqrt_tau)
        u_right = (grid - x_max + 2.0 * tau_step * alpha) / (2.0 * sqrt_tau)
        tail_scale = 0.5 * math.exp(tau_step * (alpha * alpha - beta))

        return (
            values[0] * tail_scale * erfc(u_left)
            + values[-1] * tail_scale * erfc(-u_right)
        )

    def _apply_observation_logic(
        self,
        v_in: np.ndarray,
        v_out: np.ndarray,
        spot_grid: np.ndarray,
        product,
        obs_time: float,
        pricing_env: PricingEnvironment,
    ) -> None:
        """
        Apply barrier/observation logic at discrete observation time.

        Override this method for product-specific KO/KI handling.
        """
        pass  # Product-specific implementation

    def _get_observation_times(self, product, pricing_env) -> list:
        """Extract observation times from product."""
        # Product-specific extraction
        pass

    def _build_dt(self, times: Sequence[float]) -> np.ndarray:
        """Build time step array from observation times."""
        times_full = np.concatenate(([0.0], np.asarray(times, dtype=float)))
        dt = np.diff(times_full)
        if np.any(dt <= Tolerance.ZERO):
            raise ValidationError("observation_times must be strictly increasing.")
        return np.concatenate(([0.0], dt))
```

**Key Quadrature Concepts:**

1. **Log-price Grid**: Work in x = ln(S/S0) space for numerical stability
2. **Simpson's Rule**: Use odd grid points for accurate integration
3. **FFT Convolution**: O(n log n) diffusion operator application
4. **Two-Surface Approach**: Track v_in/v_out for knock-in products
5. **Tail Corrections**: Use erfc() to account for truncated domain

**Reference Implementation**: See `asset/equity/engine/quad/snowball_quad_engine.py`

## Facade Engine Pattern

```python
from typing import Dict, Type

from asset.equity.engine.base_engine import BaseEngine
from asset.equity.product.option import (
    EuropeanVanillaOption,
    AmericanOption,
    BarrierOption,
)
from .pde import (
    EuropeanPDESolver,
    AmericanPDESolver,
    BarrierPDESolver,
)


class PDEEngine(BaseEngine):
    """
    Unified PDE engine with automatic product-to-solver dispatch.

    This facade provides a single interface for PDE pricing regardless
    of the specific product type.
    """

    PRODUCT_SOLVER_MAP: Dict[Type, Type[BasePDESolver]] = {
        EuropeanVanillaOption: EuropeanPDESolver,
        AmericanOption: AmericanPDESolver,
        BarrierOption: BarrierPDESolver,
    }

    def price(self, product, pricing_env) -> float:
        """
        Price product using appropriate PDE solver.

        Automatically dispatches to correct solver based on product type.
        """
        solver_class = self._get_solver_class(product)
        solver = solver_class(params=self.params)
        return solver.price(product, pricing_env)

    def _get_solver_class(self, product) -> Type[BasePDESolver]:
        """Get appropriate solver class for product."""
        product_type = type(product)

        if product_type in self.PRODUCT_SOLVER_MAP:
            return self.PRODUCT_SOLVER_MAP[product_type]

        # Check inheritance
        for supported_type, solver_class in self.PRODUCT_SOLVER_MAP.items():
            if isinstance(product, supported_type):
                return solver_class

        raise PricingError(
            f"No PDE solver for {product_type.__name__}. "
            f"Supported: {list(self.PRODUCT_SOLVER_MAP.keys())}"
        )

    def calculate_greeks(self, product, pricing_env) -> Dict[str, float]:
        """Calculate Greeks using bump-and-reprice."""
        solver_class = self._get_solver_class(product)
        solver = solver_class(params=self.params)
        return solver.calculate_greeks(product, pricing_env)
```

## Engine Parameters Pattern

```python
from dataclasses import dataclass
from util.exceptions import ValidationError


@dataclass
class EngineParams:
    """Base engine parameters."""
    bump_size: float = 1e-4
    bus_days_in_year: int = 252

    def __post_init__(self):
        if self.bump_size <= 0 or self.bump_size > 0.01:
            raise ValidationError(
                f"bump_size must be in (0, 0.01], got {self.bump_size}"
            )


@dataclass
class MCParams(EngineParams):
    """Monte Carlo engine parameters."""
    seed: int = 42
    num_paths: int = 10000
    time_steps: int = 100
    use_antithetic: bool = True
    use_control_variate: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.num_paths < 100:
            raise ValidationError(f"num_paths must be >= 100")
        if self.time_steps < 1:
            raise ValidationError(f"time_steps must be >= 1")


@dataclass
class PDEParams(EngineParams):
    """PDE solver parameters with high-performance caching support."""
    grid_size: int = 400
    time_steps: int = 200
    adaptive_grid: bool = False

    # Caching configuration
    cache_enabled: bool = True
    cache_strategy: str = "standard"  # disable, strict, standard, aggressive
    grid_cache_max_entries: int = 128
    use_banded_solver: bool = True
    banded_cache_max_entries: int = 512

    # Feature-aware default grids
    auto_grid: bool = True

    # Spatial grid configuration
    s_min: float = 0.0  # 0 = auto-calculate
    s_max: float = 0.0  # 0 = auto-calculate

    # Time grid configuration
    time_grid_type: str = "uniform"  # uniform, graded, event_clustered, event_aligned
    grade_exponent: float = 2.0

    # Auto-grid tuning parameters
    event_steps_per_day: int = 4
    event_min_steps_per_interval: int = 10
    max_time_steps: int = 5000
    log_dx_target: float = 0.003
    max_grid_size: int = 2000
    include_spot_in_critical_points: bool = True
    rannacher_at_events: bool = True

    # Numerical scheme configuration
    theta: float = 0.5  # 0.5=Crank-Nicolson, 1.0=Backward Euler
    use_rannacher: bool = True
    rannacher_steps: int = 1

    def __post_init__(self):
        super().__post_init__()
        if self.grid_size < 50:
            raise ValidationError(f"grid_size must be >= 50")
        if not 0 <= self.theta <= 1:
            raise ValidationError(f"theta must be in [0, 1]")
        if self.cache_strategy not in ("disable", "strict", "standard", "aggressive"):
            raise ValidationError(
                f"cache_strategy must be one of disable, strict, standard, aggressive"
            )
```

## Two-Level Enum Pattern

```python
# util/enum/engine_enums.py

from enum import Enum, auto


class EngineType(Enum):
    """Top-level engine type classification."""
    ANALYTICAL = auto()
    MONTE_CARLO = auto()
    PDE = auto()
    QUADRATURE = auto()
    TREE = auto()

    def __call__(self, method=None):
        """Enable two-level syntax: EngineType.ANALYTICAL(Method.XYZ)"""
        if method is not None:
            return (self, method)
        return self


class AmericanAnalyticalMethod(Enum):
    """Methods for American option analytical approximation."""
    BS93 = "BS93"      # Bjerksund-Stensland 1993
    BS02 = "BS02"      # Bjerksund-Stensland 2002
    BAW = "BAW"        # Barone-Adesi-Whaley


class MonteCarloMethod(Enum):
    """Monte Carlo simulation methods."""
    PSEUDO = "pseudo"
    QUASI = "quasi"
    RANDOMIZED_QUASI = "randomized_quasi"


class PDEMethod(Enum):
    """PDE discretization schemes."""
    CRANK_NICOLSON = "crank_nicolson"
    EXPLICIT_EULER = "explicit_euler"
    IMPLICIT_EULER = "implicit_euler"


class TreeMethod(Enum):
    """Tree-based pricing methods."""
    BINOMIAL_CRR = "binomial_crr"      # Cox-Ross-Rubinstein
    BINOMIAL_JR = "binomial_jr"        # Jarrow-Rudd
    TRINOMIAL = "trinomial"
```

## Module Export Pattern

```python
# asset/equity/engine/analytical/__init__.py

from .black_scholes_engine import BlackScholesEngine
from .american_option_engine import AmericanOptionAnalyticalEngine
from .barrier_analytical_engine import BarrierAnalyticalEngine
from .digital_option_engine import DigitalOptionAnalyticalEngine
from .deltaone_engine import DeltaOneEngine
from .one_touch_analytical_engine import OneTouchAnalyticalEngine

__all__ = [
    "BlackScholesEngine",
    "AmericanOptionAnalyticalEngine",
    "BarrierAnalyticalEngine",
    "DigitalOptionAnalyticalEngine",
    "DeltaOneEngine",
    "OneTouchAnalyticalEngine",
]
```

## Numerical Stability Patterns

```python
from util.numerical import is_zero, safe_log, safe_sqrt, safe_exp, safe_divide

# Check for expired options
if is_zero(T):
    return product.get_payoff(S)

# Safe logarithm (prevents log(0))
log_moneyness = safe_log(S / K)

# Safe square root (prevents sqrt(negative))
sigma_sqrt_t = safe_sqrt(sigma**2 * T)

# Safe exponential (prevents overflow)
discount = safe_exp(-r * T)

# Safe division (prevents div by zero)
d1 = safe_divide(
    log_moneyness + (r - q + 0.5 * sigma**2) * T,
    sigma_sqrt_t
)
```

## Edge Case Handling

```python
def price(self, product, pricing_env) -> float:
    S = pricing_env.spot
    K = product.strike
    T = product.get_maturity(pricing_env)
    sigma = pricing_env.volatility
    r = pricing_env.rate

    # Edge case 1: Expired option
    if is_zero(T):
        return product.get_payoff(S)

    # Edge case 2: Zero volatility
    if is_zero(sigma):
        forward = S * safe_exp((r - pricing_env.dividend_yield) * T)
        df = safe_exp(-r * T)
        return df * product.get_payoff(forward)

    # Edge case 3: Deep ITM/OTM
    moneyness = S / K
    if moneyness > 10:  # Deep ITM call
        # Use asymptotic approximation
        pass
    elif moneyness < 0.1:  # Deep OTM call
        # Use asymptotic approximation
        pass

    # Normal pricing
    return self._calculate_price(S, K, T, r, sigma)
```

## Greeks Calculation Patterns

**IMPORTANT**: Greeks calculation is NOT replicated in engine scripts. Instead:

1. **Engines**: Focus on `price()` method only (no `calculate_greeks()` override by default)
2. **Analytical Greeks**: Only override `calculate_greeks()` in engine when closed-form formulas exist
3. **Numerical Greeks**: Use centralized `GreeksCalculator` in `asset/<type>/riskmeasures/greeks_calculator.py`

### Correct Engine Structure (DO NOT Override Greeks)

```python
class MyEngine(BaseEngine):
    """Engine that ONLY implements price() - typical case."""

    def price(self, product, pricing_env) -> float:
        """Calculate price - this is the engine's ONLY responsibility."""
        # ... pricing logic
        return price

    # DO NOT override calculate_greeks() unless analytical formulas available
    # Use GreeksCalculator from riskmeasures/ for numerical Greeks
```

### Using GreeksCalculator (Recommended Approach)

Each asset type has a `GreeksCalculator` in `riskmeasures/`:
- `asset/equity/riskmeasures/greeks_calculator.py`
- `asset/bond/riskmeasures/bond_greeks_calculator.py`

```python
from asset.equity.riskmeasures import GreeksCalculator
from asset.equity.param import EngineParams

# Create calculator
calculator = GreeksCalculator(params=EngineParams())

# For European vanilla - use analytical Greeks
greeks = calculator.calculate_analytical_greeks(product, pricing_env)

# For ANY product/engine combo - use numerical Greeks
greeks = calculator.calculate_numerical_greeks(product, pricing_env, engine)

# Individual Greeks (for efficiency when only one needed)
delta = calculator.calculate_numerical_delta(product, pricing_env, engine)
gamma = calculator.calculate_numerical_gamma(product, pricing_env, engine)
vega = calculator.calculate_numerical_vega(product, pricing_env, engine)
theta = calculator.calculate_numerical_theta(product, pricing_env, engine)
rho = calculator.calculate_numerical_rho(product, pricing_env, engine)

# Compare analytical vs numerical
comparison = calculator.compare_greeks(analytical_greeks, numerical_greeks)
```

### GreeksCalculator Features

- **Analytical Greeks**: Black-Scholes formulas (EuropeanVanillaOption only)
- **Numerical Greeks**: Central difference bump-and-reprice for ANY product/engine
- **Edge cases**: Expiry handling, delta-one products detection
- **Theta**: Proper observation schedule bumping for path-dependent products
- **Validation**: Compare analytical vs numerical for testing

### When to Override `calculate_greeks()` in Engine (Rare)

Only override when closed-form analytical formulas exist:

```python
class BlackScholesEngine(BaseEngine):
    """Black-Scholes engine - HAS analytical Greeks formulas."""

    def price(self, product, pricing_env) -> float:
        # ... Black-Scholes pricing
        pass

    def calculate_greeks(self, product, pricing_env) -> Dict[str, float]:
        """Override ONLY because analytical formulas exist."""
        # ... closed-form delta, gamma, vega, theta, rho
        # See GreeksCalculator.calculate_analytical_greeks() for formulas
        pass
```

**Examples of engines that SHOULD override:**
- `BlackScholesEngine` - Has closed-form BS Greeks

**Examples of engines that should NOT override:**
- `AmericanOptionAnalyticalEngine` - Use numerical Greeks
- `BarrierPDESolver` - Use numerical Greeks
- `SnowballMCEngine` - Use numerical Greeks
- Most other engines - Use `GreeksCalculator`
