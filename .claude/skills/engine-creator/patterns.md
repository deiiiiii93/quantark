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
from typing import Optional, Tuple
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu

from asset.equity.engine.base_engine import BaseEngine
from asset.equity.param import PDEParams
from .time_grid import TimeGrid
from .spatial_grid import SpatialGrid


class BasePDESolver(BaseEngine):
    """
    Abstract base class for PDE-based pricing.

    Implements Crank-Nicolson finite difference scheme with:
    - Log-price transformation (x = ln(S))
    - Tavella-Randall grid concentration
    - Rannacher smoothing for initial steps
    - LU factorization caching
    """

    def __init__(self, params: Optional[PDEParams] = None):
        super().__init__(params or PDEParams())
        self._lu_cache = {}

    def price(self, product, pricing_env) -> float:
        """
        Solve PDE using finite difference method.

        Returns:
            Option price at current spot
        """
        # Validate product
        self._validate_product(product)

        # Build grids
        T = product.get_maturity(pricing_env)
        time_grid = self._build_time_grid(T, product, pricing_env)
        spatial_grid = self._build_spatial_grid(product, pricing_env)

        # Initialize solution
        V = self._set_terminal_condition(spatial_grid, product, pricing_env)

        # Time stepping (backward from T to 0)
        for n in range(len(time_grid.times) - 1, 0, -1):
            dt = time_grid.times[n] - time_grid.times[n - 1]
            t = time_grid.times[n - 1]

            # Build tridiagonal matrix
            A = self._build_matrix(spatial_grid, pricing_env, dt, t)

            # Solve linear system
            V = self._solve_step(A, V, dt)

            # Apply boundary conditions
            V = self._set_boundary_conditions(V, spatial_grid, product, pricing_env, t)

            # Apply early exercise (for American)
            V = self._apply_early_exercise(V, spatial_grid, product, pricing_env, t)

        # Interpolate to current spot
        return self._interpolate(V, spatial_grid, pricing_env.spot)

    @abstractmethod
    def _validate_product(self, product) -> None:
        """Validate product type."""
        pass

    @abstractmethod
    def _set_terminal_condition(
        self, spatial_grid: SpatialGrid, product, pricing_env
    ) -> np.ndarray:
        """Set terminal (payoff) condition at T."""
        pass

    @abstractmethod
    def _set_boundary_conditions(
        self, V: np.ndarray, spatial_grid: SpatialGrid,
        product, pricing_env, t: float
    ) -> np.ndarray:
        """Apply boundary conditions at each time step."""
        pass

    def _apply_early_exercise(
        self, V: np.ndarray, spatial_grid: SpatialGrid,
        product, pricing_env, t: float
    ) -> np.ndarray:
        """Apply early exercise constraint (override for American)."""
        return V  # No early exercise by default

    def _build_time_grid(self, T, product, pricing_env) -> TimeGrid:
        """Build temporal discretization."""
        return TimeGrid.build(
            T=T,
            num_steps=self.params.time_steps,
            method=self.params.time_grid_type,
            grade_exponent=self.params.grade_exponent,
        )

    def _build_spatial_grid(self, product, pricing_env) -> SpatialGrid:
        """Build spatial discretization in log-price space."""
        critical_points = self._get_critical_points(product, pricing_env)
        return SpatialGrid.build(
            spot=pricing_env.spot,
            sigma=pricing_env.volatility,
            T=product.get_maturity(pricing_env),
            num_points=self.params.grid_size,
            critical_points=critical_points,
            s_min=self.params.s_min,
            s_max=self.params.s_max,
        )

    def _get_critical_points(self, product, pricing_env) -> list:
        """Get points requiring grid concentration (override in subclass)."""
        return [product.strike]  # Default: concentrate around strike

    def _build_matrix(self, spatial_grid, pricing_env, dt, t) -> np.ndarray:
        """Build tridiagonal matrix for Crank-Nicolson."""
        # ... implementation
        pass

    def _solve_step(self, A, V, dt) -> np.ndarray:
        """Solve one time step using cached LU factorization."""
        cache_key = (dt, self.params.theta)
        if cache_key not in self._lu_cache:
            self._lu_cache[cache_key] = splu(A)
        return self._lu_cache[cache_key].solve(V)

    def _interpolate(self, V, spatial_grid, spot) -> float:
        """Interpolate solution to current spot price."""
        x = np.log(spot)
        return np.interp(x, spatial_grid.x_vec, V)
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
    """PDE solver parameters."""
    grid_size: int = 400
    time_steps: int = 200
    theta: float = 0.5  # 0.5=Crank-Nicolson, 1.0=Implicit Euler
    use_rannacher: bool = True
    rannacher_steps: int = 2
    time_grid_type: str = "uniform"  # uniform, graded, event_clustered
    grade_exponent: float = 2.0
    s_min: float = 0.0  # 0 = auto
    s_max: float = 0.0  # 0 = auto

    def __post_init__(self):
        super().__post_init__()
        if self.grid_size < 50:
            raise ValidationError(f"grid_size must be >= 50")
        if not 0 <= self.theta <= 1:
            raise ValidationError(f"theta must be in [0, 1]")
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
