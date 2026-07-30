"""
Quadrature pricing engine for European vanilla options.

Uses numerical integration of the risk-neutral expectation to price options.
"""

import math
from typing import Optional, Union

import numpy as np
from scipy.integrate import simpson, fixed_quad

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.engine.capabilities import SettlementSupport
from quantark.asset.equity.engine.settlement_support import (
    pending_receivable_pv,
    resolve_terminal_timing,
    terminal_lifecycle_pv,
    validate_settlement_capability,
)
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.param import QuadParams
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import ValidationError, NumericalError, PricingError
from quantark.util.enum.engine_enums import EngineType, QuadratureMethod
from quantark.util.numerical import safe_exp


class EuropeanQuadEngine(BaseEngine):
    """
    Quadrature pricing engine for European vanilla options.

    Uses numerical integration of the risk-neutral expectation:
        V = e^(-rT) * E[payoff(S_T)]
          = e^(-rT) * ∫ payoff(S) * p(S) dS

    where p(S) is the log-normal density under the risk-neutral measure.

    This engine provides:
    - A foundation for more complex products (barriers, autocallables)
    - Validation against analytical Black-Scholes solutions
    - A template for future quadrature engines

    Attributes:
        engine_type: EngineType.QUADRATURE
        method: The quadrature method (SIMPSON or GAUSS_LEGENDRE)
        params: QuadParams configuration
    """

    engine_type = EngineType.QUADRATURE
    settlement_support = SettlementSupport.TERMINAL_ONLY
    supports_lifecycle_state = True
    DEFAULT_METHOD = QuadratureMethod.SIMPSON

    def __init__(
        self,
        params: Optional[QuadParams] = None,
        method: Union[str, QuadratureMethod, tuple, None] = None,
    ):
        """
        Initialize the quadrature engine.

        Args:
            params: Quadrature configuration parameters
            method: Quadrature method. Can be:
                - QuadratureMethod enum value
                - String ("simpson", "gauss_legendre")
                - Two-level enum: EngineType.QUADRATURE(QuadratureMethod.SIMPSON)
                - None (uses DEFAULT_METHOD)
        """
        if params is None:
            params = QuadParams()
        super().__init__(params)

        # Parse method from various input formats
        self.method = self._parse_method(method)

    def _parse_method(
        self, method: Union[str, QuadratureMethod, tuple, None]
    ) -> QuadratureMethod:
        """Parse method from various input formats."""
        if method is None:
            return self.DEFAULT_METHOD

        # Handle two-level enum pattern
        if isinstance(method, tuple) and len(method) == 2 and method[0] == EngineType.QUADRATURE:
            return method[1]

        # Handle QuadratureMethod enum
        if isinstance(method, QuadratureMethod):
            return method

        # Handle string
        if isinstance(method, str):
            for m in QuadratureMethod:
                if m.value == method.lower():
                    return m
            valid = [m.value for m in QuadratureMethod]
            raise ValidationError(
                f"Unknown quadrature method: {method}. Valid: {valid}"
            )

        raise ValidationError(f"Invalid method type: {type(method)}")

    def price(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        *,
        lifecycle_state=None,
    ) -> float:
        """
        Price a European vanilla option using numerical quadrature.

        Args:
            product: European vanilla option
            pricing_env: Pricing environment with market data

        Returns:
            Option price

        Raises:
            PricingError: If product is not a European vanilla option
            NumericalError: If numerical issues occur during calculation
        """
        if not isinstance(product, EuropeanVanillaOption):
            raise PricingError(
                f"EuropeanQuadEngine only supports EuropeanVanillaOption, "
                f"got {type(product).__name__}"
            )

        validate_settlement_capability(self, product, lifecycle_state)
        fixed_pv = terminal_lifecycle_pv(lifecycle_state, pricing_env)
        if fixed_pv is not None:
            return fixed_pv
        timing = resolve_terminal_timing(product, pricing_env)

        # Extract parameters
        S = pricing_env.spot
        K = product.strike
        T = product.get_maturity(pricing_env)
        r = pricing_env.get_rate(T)
        q = pricing_env.get_div_yield(T)
        sigma = pricing_env.get_vol(K, T)

        # Validate inputs
        self._validate_inputs(S, K, T, r, q, sigma)

        # Handle edge cases
        if T < 1e-10:  # Option has expired
            return (
                product.get_payoff(S) * timing.delay_df
                + pending_receivable_pv(lifecycle_state, pricing_env)
            )

        # Price using quadrature
        if self.method == QuadratureMethod.SIMPSON:
            price = self._price_simpson(
                S, K, T, r, q, sigma, product.is_call(),
                df=timing.payment_df,
            )
        elif self.method == QuadratureMethod.GAUSS_LEGENDRE:
            price = self._price_gauss_legendre(
                S, K, T, r, q, sigma, product.is_call(),
                df=timing.payment_df,
            )
        else:
            raise PricingError(f"Unsupported quadrature method: {self.method}")

        price *= product.contract_multiplier

        # Sanity checks
        if price < 0:
            raise NumericalError(f"Negative price computed: {price}")

        lower_bound = self._european_lower_bound(
            product,
            S,
            K,
            T,
            q,
            timing.payment_df,
            timing.delay_df,
        )
        if price < lower_bound - 1e-4:  # Small tolerance for numerical errors
            raise NumericalError(
                f"Price ({price:.6f}) below discounted European lower bound "
                f"({lower_bound:.6f})"
            )

        return price + pending_receivable_pv(lifecycle_state, pricing_env)

    def _european_lower_bound(
        self,
        product: EuropeanVanillaOption,
        S: float,
        K: float,
        T: float,
        q: float,
        payment_df: float,
        delay_df: float,
    ) -> float:
        """Calculate the discounted no-arbitrage lower bound for a European option."""
        spot_pv = S * safe_exp(-q * T) * delay_df
        strike_pv = K * payment_df
        if product.is_call():
            lower_bound = max(spot_pv - strike_pv, 0.0)
        else:
            lower_bound = max(strike_pv - spot_pv, 0.0)
        return lower_bound * product.contract_multiplier

    def _validate_inputs(
        self, S: float, K: float, T: float, r: float, q: float, sigma: float
    ) -> None:
        """Validate input parameters for numerical stability."""
        if S <= 0:
            raise ValidationError(f"Spot price must be positive, got {S}")
        if K <= 0:
            raise ValidationError(f"Strike price must be positive, got {K}")
        if T < 0:
            raise ValidationError(f"Time to maturity must be non-negative, got {T}")
        if sigma <= 0:
            raise ValidationError(f"Volatility must be positive, got {sigma}")
        if sigma > 5.0:
            raise ValidationError(
                f"Volatility too high for numerical stability: {sigma}"
            )
        if abs(r) > 1.0:
            raise ValidationError(f"Risk-free rate outside reasonable bounds: {r}")
        if T > 100:
            raise ValidationError(
                f"Maturity too long for numerical stability: {T} years"
            )

    def _price_simpson(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        is_call: bool,
        df: float | None = None,
    ) -> float:
        """
        Price option using Simpson's rule integration.

        Integrates the discounted expected payoff:
            V = e^(-rT) * ∫ payoff(S_T) * p(S_T) dS_T

        where p(S_T) is the log-normal density.
        """
        params = self.params
        n_points = params.grid_points
        n_std = params.num_std_devs

        # Set up integration in log-space for better numerical properties
        # Under risk-neutral measure: ln(S_T) ~ N(ln(S) + (r-q-σ²/2)T, σ²T)
        drift = (r - q - 0.5 * sigma * sigma) * T
        vol_sqrt_t = sigma * math.sqrt(T)

        # Integration bounds in log-space
        log_S = math.log(S)
        x_min = log_S + drift - n_std * vol_sqrt_t
        x_max = log_S + drift + n_std * vol_sqrt_t

        # Create grid in log-space
        x_grid = np.linspace(x_min, x_max, n_points)
        dx = (x_max - x_min) / (n_points - 1)

        # Convert to price space
        S_grid = np.exp(x_grid)

        # Calculate payoffs
        if is_call:
            payoffs = np.maximum(S_grid - K, 0.0)
        else:
            payoffs = np.maximum(K - S_grid, 0.0)

        # Log-normal density in log-space (Gaussian)
        # p(x) = (1 / (σ√(2πT))) * exp(-(x - μ)² / (2σ²T))
        # where μ = ln(S) + (r-q-σ²/2)T
        mean = log_S + drift
        variance = sigma * sigma * T

        # Gaussian density
        z = (x_grid - mean) / math.sqrt(variance)
        density = np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi * variance)

        # Integrand: payoff * density
        integrand = payoffs * density

        # Simpson's rule integration
        integral = simpson(integrand, dx=dx)

        # Discount to present value
        # Curve-exact DF; the terminal-density integral uses cumulative-to-T
        # inputs, which are term-structure exact for Europeans (spec:
        # documented convention, same as the analytical engines).
        discount = df if df is not None else math.exp(-r * T)
        price = discount * integral

        return price

    def _price_gauss_legendre(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        is_call: bool,
        df: float | None = None,
    ) -> float:
        """
        Price option using Gauss-Legendre quadrature.

        Uses scipy's fixed_quad for Gauss-Legendre integration.
        """

        params = self.params
        n_points = params.grid_points
        n_std = params.num_std_devs

        # Set up integration bounds
        drift = (r - q - 0.5 * sigma * sigma) * T
        vol_sqrt_t = sigma * math.sqrt(T)
        log_S = math.log(S)

        x_min = log_S + drift - n_std * vol_sqrt_t
        x_max = log_S + drift + n_std * vol_sqrt_t

        mean = log_S + drift
        variance = sigma * sigma * T

        def integrand(x):
            """Integrand: payoff * log-normal density."""
            S_T = np.exp(x)
            if is_call:
                payoff = np.maximum(S_T - K, 0.0)
            else:
                payoff = np.maximum(K - S_T, 0.0)

            # Gaussian density in log-space
            z = (x - mean) / math.sqrt(variance)
            density = np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi * variance)

            return payoff * density

        # Gauss-Legendre quadrature
        # Note: fixed_quad uses n points, not n+1 like Simpson
        n_quad = min(n_points // 10, 100)  # Gauss-Legendre converges faster
        integral, _ = fixed_quad(integrand, x_min, x_max, n=n_quad)

        # Discount to present value
        # Curve-exact DF; the terminal-density integral uses cumulative-to-T
        # inputs, which are term-structure exact for Europeans (spec:
        # documented convention, same as the analytical engines).
        discount = df if df is not None else math.exp(-r * T)
        price = discount * integral

        return price

    def __repr__(self):
        return f"EuropeanQuadEngine(method={self.method.value})"
