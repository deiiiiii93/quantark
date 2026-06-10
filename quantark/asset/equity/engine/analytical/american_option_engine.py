"""
Analytical pricing engine for American vanilla options.

This module implements three approximation methods:
- BS93: Bjerksund-Stensland (1993) single-barrier approximation
- BS02: Bjerksund-Stensland (2002) two-barrier approximation
- BAW: Barone-Adesi & Whaley (1987) quadratic approximation

References:
    [1] Bjerksund, P., and Stensland, G., 1993. Closed-form approximation of American options.
    [2] Bjerksund, P., and Stensland, G., 2002. Closed-form approximation of American options.
        Scandinavian Journal of Management, 18(4), 487-507.
    [3] Barone-Adesi, G., and Whaley, R. E., 1987. Efficient analytic approximation of American
        option values. Journal of Finance, 42(2), 301-320.
"""

import numpy as np
from typing import Optional, Union

from scipy.stats import norm, multivariate_normal
from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.product.option import AmericanOption
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.param import EngineParams
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import ValidationError, NumericalError, PricingError
from quantark.util.enum.engine_enums import AmericanAnalyticalMethod, EngineType


class AmericanOptionAnalyticalEngine(BaseEngine):
    """
    Analytical pricing engine for American vanilla options.

    Supports three approximation methods:
    - BS93 (default): Fast, single-barrier approximation
    - BS02: More accurate two-barrier approximation
    - BAW: Quadratic approximation with iterative critical price search

    For American puts, BS93/BS02 use put-call transformation while BAW uses direct put pricing.
    """

    engine_type = EngineType.ANALYTICAL

    DEFAULT_METHOD = AmericanAnalyticalMethod.BS93

    MIN_VOL = 0.001
    MAX_VOL = 5.0
    MIN_MATURITY = 1e-6
    MAX_MATURITY = 30.0

    def __init__(self, params: Optional[EngineParams] = None, method: Union[str, AmericanAnalyticalMethod, tuple] = None):
        """
        Initialize American option analytical engine.

        Args:
            params: Engine configuration parameters
            method: Pricing method, can be:
                - AmericanAnalyticalMethod enum (e.g., AmericanAnalyticalMethod.BS93)
                - String "BS93"/"BS02"/"BAW" (backward compatibility)
                - Tuple from EngineType.ANALYTICAL(AmericanAnalyticalMethod.BS93)
                - None (defaults to BS93)

        Raises:
            ValidationError: If invalid method is specified
        """
        super().__init__(params)
        
        if method is None:
            self.method = self.DEFAULT_METHOD
        elif isinstance(method, tuple):
            engine_type, analytical_method = method
            if engine_type != EngineType.ANALYTICAL:
                raise ValidationError(
                    f"Expected EngineType.ANALYTICAL, got {engine_type}"
                )
            if not isinstance(analytical_method, AmericanAnalyticalMethod):
                raise ValidationError(
                    f"Expected AmericanAnalyticalMethod, got {type(analytical_method).__name__}"
                )
            self.method = analytical_method
        elif isinstance(method, AmericanAnalyticalMethod):
            self.method = method
        elif isinstance(method, str):
            try:
                self.method = AmericanAnalyticalMethod[method.upper()]
            except KeyError:
                valid_methods = ', '.join([m.name for m in AmericanAnalyticalMethod])
                raise ValidationError(
                    f"Invalid method '{method}'. "
                    f"Valid methods are: {valid_methods}"
                )
        else:
            raise ValidationError(
                f"Method must be AmericanAnalyticalMethod enum, string, or EngineType tuple, got {type(method).__name__}"
            )

    def price(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        """
        Price an American vanilla option using the selected approximation method.

        Args:
            product: American vanilla option
            pricing_env: Pricing environment with market data

        Returns:
            Option price

        Raises:
            PricingError: If product is not an American option
            ValidationError: If input parameters are invalid
            NumericalError: If numerical computation fails
        """
        if not isinstance(product, AmericanOption):
            raise PricingError(
                f"AmericanOptionAnalyticalEngine only supports AmericanOption, "
                f"got {type(product).__name__}"
            )

        S = pricing_env.spot
        K = product.strike
        T = product.get_maturity(pricing_env)
        r = pricing_env.get_rate(T)
        q = pricing_env.get_div_yield(T)
        sigma = pricing_env.get_vol(K, T)
        multiplier = product.contract_multiplier

        self._validate_inputs(S, K, T, r, q, sigma)

        if T < self.MIN_MATURITY:
            return product.get_payoff(S)

        sigma = np.clip(sigma, self.MIN_VOL, self.MAX_VOL)
        T = np.clip(T, self.MIN_MATURITY, self.MAX_MATURITY)

        b = r - q

        is_call = product.is_call()

        if is_call and b >= r and r >= 0:
            return self._european_call_bsm(S, K, T, r, b, sigma) * multiplier

        if is_call and b >= r and r < 0 and q <= r:
            return self._european_call_bsm(S, K, T, r, b, sigma) * multiplier

        try:
            if self.method == AmericanAnalyticalMethod.BS93:
                price = self._price_bs93(S, K, T, r, b, sigma, is_call)
            elif self.method == AmericanAnalyticalMethod.BS02:
                price = self._price_bs02(S, K, T, r, b, sigma, is_call)
            else:
                price = self._price_baw(S, K, T, r, b, sigma, is_call)

            if np.isnan(price) or np.isinf(price):
                raise NumericalError("NaN or Inf result detected")

            price *= multiplier
            intrinsic = product.intrinsic_value(S)
            if price < intrinsic - 1e-6:
                raise NumericalError(
                    f"Price ({price:.6f}) below intrinsic value ({intrinsic:.6f})"
                )

            return max(price, intrinsic)

        except Exception as e:
            import warnings

            warnings.warn(
                f"American option pricing failed ({e}), using European fallback"
            )
            if is_call:
                return self._european_call_bsm(S, K, T, r, b, sigma) * multiplier
            else:
                return self._european_put_bsm(S, K, T, r, b, sigma) * multiplier

    def _validate_inputs(
        self, S: float, K: float, T: float, r: float, q: float, sigma: float
    ):
        """Validate input parameters."""
        if S <= 0:
            raise ValidationError(f"Spot price must be positive, got {S}")
        if K <= 0:
            raise ValidationError(f"Strike price must be positive, got {K}")
        if T < 0:
            raise ValidationError(f"Time to maturity must be non-negative, got {T}")
        if sigma <= 0:
            raise ValidationError(f"Volatility must be positive, got {sigma}")

    def _price_bs93(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        b: float,
        sigma: float,
        is_call: bool,
    ) -> float:
        """Price using Bjerksund-Stensland 1993 approximation."""
        if is_call:
            return self._price_american_call_bs93(S, K, T, r, b, sigma)
        else:
            if r <= 0 and r <= b:
                return self._european_put_bsm(S, K, T, r, b, sigma)
            return self._price_american_put_bs93(S, K, T, r, b, sigma)

    def _price_bs02(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        b: float,
        sigma: float,
        is_call: bool,
    ) -> float:
        """Price using Bjerksund-Stensland 2002 approximation."""
        if is_call:
            return self._price_american_call_bs02(S, K, T, r, b, sigma)
        else:
            if r <= 0 and r <= b:
                return self._european_put_bsm(S, K, T, r, b, sigma)
            return self._price_american_put_bs02(S, K, T, r, b, sigma)

    def _price_baw(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        b: float,
        sigma: float,
        is_call: bool,
    ) -> float:
        """Price using Barone-Adesi-Whaley approximation."""
        if is_call:
            return self._price_american_call_baw(S, K, T, r, b, sigma)
        else:
            return self._price_american_put_baw(S, K, T, r, b, sigma)

    def _price_american_call_bs93(
        self, S: float, K: float, T: float, r: float, b: float, sigma: float
    ) -> float:
        """
        Bjerksund-Stensland 1993 approximation for American call.

        Single-barrier approximation using optimal exercise boundary.
        """
        beta = (0.5 - b / sigma**2) + np.sqrt(
            (0.5 - b / sigma**2) ** 2 + 2 * r / sigma**2
        )

        B_infinity = beta * K / (beta - 1.0)
        B_0 = max(K, r * K / (r - b))

        h_T = -1.0 * (b * T + 2.0 * sigma * np.sqrt(T)) * B_0 / (B_infinity - B_0)

        I = B_0 + (B_infinity - B_0) * (1.0 - self._safe_exp(h_T))

        if S >= I:
            return S - K

        alpha = (I - K) * I ** (-beta)

        value = (
            alpha * S**beta
            - alpha * self._phi_bs93(S, T, beta, I, I, r, b, sigma)
            + self._phi_bs93(S, T, 1, I, I, r, b, sigma)
            - self._phi_bs93(S, T, 1, K, I, r, b, sigma)
            - K * self._phi_bs93(S, T, 0, I, I, r, b, sigma)
            + K * self._phi_bs93(S, T, 0, K, I, r, b, sigma)
        )

        return value

    def _price_american_put_bs93(
        self, S: float, K: float, T: float, r: float, b: float, sigma: float
    ) -> float:
        """American put via put-call transformation for BS93."""
        # Transform parameters for put-call analogue
        # Put(S,K,r,q) = Call(K,S,q,r) with transformed b
        S_new = K
        K_new = S
        r_new = r - b  # This is q (dividend yield)
        b_new = -b

        call_value = self._price_american_call_bs93(
            S_new, K_new, T, r_new, b_new, sigma
        )

        return call_value

    def _phi_bs93(
        self,
        S: float,
        T: float,
        gamma: float,
        H: float,
        I: float,
        r: float,
        b: float,
        sigma: float,
    ) -> float:
        """
        Auxiliary function φ for BS93.

        φ(S, T, γ, H, I) = e^(λT) S^γ [N(d) - (I/S)^κ N(d₂)]

        Note: This follows the original BS93 formula structure.
        """
        lamda = T * (-r + b * gamma + 0.5 * gamma * (gamma - 1) * sigma**2)

        d = (
            -1.0
            * (self._safe_log(S / H) + (b + (gamma - 0.5) * sigma**2) * T)
            / (sigma * np.sqrt(T))
        )

        kappa = 2 * b / sigma**2 + 2 * gamma - 1

        d2 = d - 2 * self._safe_log(I / S) / (sigma * np.sqrt(T))

        return (
            self._safe_exp(lamda)
            * S**gamma
            * (norm.cdf(d) - (I / S) ** kappa * norm.cdf(d2))
        )

    def _price_american_call_bs02(
        self, S: float, K: float, T: float, r: float, b: float, sigma: float
    ) -> float:
        """
        Bjerksund-Stensland 2002 approximation for American call.

        More accurate two-barrier approximation.
        """
        beta = (0.5 - b / sigma**2) + np.sqrt(
            (0.5 - b / sigma**2) ** 2 + 2 * r / sigma**2
        )

        B_infinity = beta * K / (beta - 1.0)
        B_0 = max(K, r * K / (r - b))

        t1 = 0.5 * (np.sqrt(5) - 1) * T

        h1 = -(b * t1 + 2 * sigma * np.sqrt(t1)) * K**2 / ((B_infinity - B_0) * B_0)
        h2 = -(b * T + 2 * sigma * np.sqrt(T)) * K**2 / ((B_infinity - B_0) * B_0)

        I1 = B_0 + (B_infinity - B_0) * (1 - self._safe_exp(h1))
        I2 = B_0 + (B_infinity - B_0) * (1 - self._safe_exp(h2))

        if I1 > K:
            log_alpha1 = self._safe_log(I1 - K) - beta * self._safe_log(I1)
            alpha1 = self._safe_exp(log_alpha1) if log_alpha1 > -700 else 0.0
        else:
            alpha1 = 0.0

        if I2 > K:
            log_alpha2 = self._safe_log(I2 - K) - beta * self._safe_log(I2)
            alpha2 = self._safe_exp(log_alpha2) if log_alpha2 > -700 else 0.0
        else:
            alpha2 = 0.0

        if S >= I2:
            return S - K

        value = (
            alpha2 * S**beta
            - alpha2 * self._phi_bs02(S, t1, beta, I2, I2, r, b, sigma)
            + self._phi_bs02(S, t1, 1, I2, I2, r, b, sigma)
            - self._phi_bs02(S, t1, 1, I1, I2, r, b, sigma)
            - K * self._phi_bs02(S, t1, 0, I2, I2, r, b, sigma)
            + K * self._phi_bs02(S, t1, 0, I1, I2, r, b, sigma)
            + alpha1 * self._phi_bs02(S, t1, beta, I1, I2, r, b, sigma)
            - alpha1 * self._psi_bs02(S, T, beta, I1, I2, I1, t1, r, b, sigma)
            + self._psi_bs02(S, T, 1, I1, I2, I1, t1, r, b, sigma)
            - self._psi_bs02(S, T, 1, K, I2, I1, t1, r, b, sigma)
            - K * self._psi_bs02(S, T, 0, I1, I2, I1, t1, r, b, sigma)
            + K * self._psi_bs02(S, T, 0, K, I2, I1, t1, r, b, sigma)
        )

        return value

    def _price_american_put_bs02(
        self, S: float, K: float, T: float, r: float, b: float, sigma: float
    ) -> float:
        """American put via put-call transformation for BS02."""
        S_orig, K_orig, r_orig, b_orig = S, K, r, b

        S = K_orig
        K = S_orig
        r = r_orig - b_orig
        b = -b_orig

        call_value = self._price_american_call_bs02(S, K, T, r, b, sigma)

        return call_value

    def _phi_bs02(
        self,
        S: float,
        T: float,
        gamma: float,
        H: float,
        I: float,
        r: float,
        b: float,
        sigma: float,
    ) -> float:
        """
        Auxiliary function φ for BS02.

        φ(S, T, γ, H, I) = e^(λT) S^γ [N(-d) - (I/S)^κ N(-d₂)]
        """
        if S <= 0 or T <= 0 or H <= 0 or I <= 0:
            return 0.0

        lambda_val = -r + gamma * b + 0.5 * gamma * (gamma - 1) * sigma**2
        kappa = 2 * b / sigma**2 + 2 * gamma - 1

        d = (self._safe_log(S / H) + (b + (gamma - 0.5) * sigma**2) * T) / (
            sigma * np.sqrt(T)
        )
        d2 = (
            self._safe_log(I**2 / (S * H)) + (b + (gamma - 0.5) * sigma**2) * T
        ) / (sigma * np.sqrt(T))

        term1 = norm.cdf(-d)
        term2 = (I / S) ** kappa * norm.cdf(-d2)

        return self._safe_exp(lambda_val * T) * S**gamma * (term1 - term2)

    def _psi_bs02(
        self,
        S: float,
        T: float,
        gamma: float,
        H: float,
        I2: float,
        I1: float,
        t1: float,
        r: float,
        b: float,
        sigma: float,
    ) -> float:
        """
        Auxiliary function Ψ for BS02.

        Uses bivariate normal CDF for improved accuracy.
        """
        if S <= 0 or T <= 0 or t1 <= 0 or H <= 0 or I1 <= 0 or I2 <= 0:
            return 0.0

        lambda_val = -r + gamma * b + 0.5 * gamma * (gamma - 1) * sigma**2
        kappa = 2 * b / sigma**2 + 2 * gamma - 1

        rho = np.sqrt(t1 / T)

        # Calculate e parameters (for t1 dimension)
        e1 = (self._safe_log(S / I1) + (b + (gamma - 0.5) * sigma**2) * t1) / (
            sigma * np.sqrt(t1)
        )
        e2 = (
            self._safe_log(I2**2 / (S * I1)) + (b + (gamma - 0.5) * sigma**2) * t1
        ) / (sigma * np.sqrt(t1))
        e3 = (self._safe_log(S / I1) - (b + (gamma - 0.5) * sigma**2) * t1) / (
            sigma * np.sqrt(t1)
        )
        e4 = (
            self._safe_log(I2**2 / (S * I1)) - (b + (gamma - 0.5) * sigma**2) * t1
        ) / (sigma * np.sqrt(t1))

        # Calculate f parameters (for T dimension)
        f1 = (self._safe_log(S / H) + (b + (gamma - 0.5) * sigma**2) * T) / (
            sigma * np.sqrt(T)
        )
        f2 = (
            self._safe_log(I2**2 / (S * H)) + (b + (gamma - 0.5) * sigma**2) * T
        ) / (sigma * np.sqrt(T))
        f3 = (
            self._safe_log(I2**2 / (S * H)) + (b + (gamma - 0.5) * sigma**2) * T
        ) / (sigma * np.sqrt(T))
        f4 = (
            self._safe_log(S * I1**2 / (H * I2**2)) + (b + (gamma - 0.5) * sigma**2) * T
        ) / (sigma * np.sqrt(T))

        M1 = self._bivariate_normal_cdf(-e1, -f1, rho)
        M2 = self._bivariate_normal_cdf(-e2, -f2, rho)
        M3 = self._bivariate_normal_cdf(-e3, -f3, -rho)
        M4 = self._bivariate_normal_cdf(-e4, -f4, -rho)

        term1 = M1
        term2 = (I2 / S) ** kappa * M2
        term3 = (I1 / S) ** kappa * M3
        term4 = (I1 / I2) ** kappa * M4

        return (
            self._safe_exp(lambda_val * T) * S**gamma * (term1 - term2 - term3 + term4)
        )

    def _bivariate_normal_cdf(self, x: float, y: float, rho: float) -> float:
        """
        Bivariate normal CDF M(x, y, ρ).

        Uses scipy's multivariate_normal for accurate computation.
        """
        if abs(rho) < 1e-10:
            return norm.cdf(x) * norm.cdf(y)

        if abs(rho) >= 1.0:
            if rho > 0:
                return min(norm.cdf(x), norm.cdf(y))
            else:
                return max(norm.cdf(x) + norm.cdf(y) - 1, 0.0)

        mean = [0, 0]
        cov = [[1, rho], [rho, 1]]
        return multivariate_normal.cdf([x, y], mean, cov)

    def _price_american_call_baw(
        self, S: float, K: float, T: float, r: float, b: float, sigma: float
    ) -> float:
        """
        Barone-Adesi-Whaley approximation for American call.

        Quadratic approximation with iterative critical price search.
        """
        if b >= r:
            return self._european_call_bsm(S, K, T, r, b, sigma)

        M = 2 * r / sigma**2
        N = 2 * b / sigma**2
        K_param = 1 - self._safe_exp(-r * T)

        discriminant = (N - 1) ** 2 + 4 * M / K_param
        if discriminant < 0:
            return self._european_call_bsm(S, K, T, r, b, sigma)

        q2 = (-(N - 1) + np.sqrt(discriminant)) / 2

        S_star = self._find_critical_call_price(K, T, r, b, sigma, q2)

        if S >= S_star:
            return S - K

        d1_s_star = self._d1_baw(S_star, K, T, b, sigma)
        A2 = (S_star / q2) * (1 - self._safe_exp((b - r) * T) * norm.cdf(d1_s_star))

        c_bsm = self._european_call_bsm(S, K, T, r, b, sigma)

        return c_bsm + A2 * (S / S_star) ** q2

    def _price_american_put_baw(
        self, S: float, K: float, T: float, r: float, b: float, sigma: float
    ) -> float:
        """
        Barone-Adesi-Whaley approximation for American put.

        Direct put pricing (not transformation).
        """
        if r <= 0 and r <= b:
            return self._european_put_bsm(S, K, T, r, b, sigma)

        M = 2 * r / sigma**2
        N = 2 * b / sigma**2
        K_param = 1 - self._safe_exp(-r * T)

        discriminant = (N - 1) ** 2 + 4 * M / K_param
        if discriminant < 0:
            return self._european_put_bsm(S, K, T, r, b, sigma)

        q1 = (-(N - 1) - np.sqrt(discriminant)) / 2

        S_star_star = self._find_critical_put_price(K, T, r, b, sigma, q1)

        if S <= S_star_star:
            return K - S

        d1_s_star_star = self._d1_baw(S_star_star, K, T, b, sigma)
        A1 = -(S_star_star / q1) * (
            1 - self._safe_exp((b - r) * T) * norm.cdf(-d1_s_star_star)
        )

        p_bsm = self._european_put_bsm(S, K, T, r, b, sigma)

        return p_bsm + A1 * (S / S_star_star) ** q1

    def _find_critical_call_price(
        self, K: float, T: float, r: float, b: float, sigma: float, q2: float
    ) -> float:
        """Find critical stock price S* for American call using optimization."""
        from scipy.optimize import fmin

        def objective(S_star):
            if S_star <= 0:
                return float("inf")
            d1_s = self._d1_baw(S_star, K, T, b, sigma)
            c_bsm = self._european_call_bsm(S_star, K, T, r, b, sigma)
            lhs = S_star - K
            rhs = (
                c_bsm + (1 - self._safe_exp((b - r) * T) * norm.cdf(d1_s)) * S_star / q2
            )
            return abs(lhs - rhs)

        S0 = max(K, K * 1.1)
        try:
            result = fmin(objective, S0, disp=False, full_output=True)
            return result[0][0] if result[4] == 0 else K * 1.5
        except:
            return K * 1.5

    def _find_critical_put_price(
        self, K: float, T: float, r: float, b: float, sigma: float, q1: float
    ) -> float:
        """Find critical stock price S** for American put using optimization."""
        from scipy.optimize import fmin

        def objective(S_star_star):
            if S_star_star <= 0:
                return float("inf")
            d1_s = self._d1_baw(S_star_star, K, T, b, sigma)
            p_bsm = self._european_put_bsm(S_star_star, K, T, r, b, sigma)
            lhs = K - S_star_star
            rhs = (
                p_bsm
                - (1 - self._safe_exp((b - r) * T) * norm.cdf(-d1_s)) * S_star_star / q1
            )
            return abs(lhs - rhs)

        S0 = min(K, K * 0.9)
        try:
            result = fmin(objective, S0, disp=False, full_output=True)
            return result[0][0] if result[4] == 0 else K * 0.5
        except:
            return K * 0.5

    def _d1_baw(self, S: float, K: float, T: float, b: float, sigma: float) -> float:
        """Calculate d1 for BAW method."""
        return (self._safe_log(S / K) + (b + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    def _european_call_bsm(
        self, S: float, K: float, T: float, r: float, b: float, sigma: float
    ) -> float:
        """European call price using Black-Scholes-Merton formula."""
        d1 = (self._safe_log(S / K) + (b + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        return S * self._safe_exp((b - r) * T) * norm.cdf(d1) - K * self._safe_exp(
            -r * T
        ) * norm.cdf(d2)

    def _european_put_bsm(
        self, S: float, K: float, T: float, r: float, b: float, sigma: float
    ) -> float:
        """European put price using Black-Scholes-Merton formula."""
        d1 = (self._safe_log(S / K) + (b + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        return K * self._safe_exp(-r * T) * norm.cdf(-d2) - S * self._safe_exp(
            (b - r) * T
        ) * norm.cdf(-d1)

    def _safe_log(self, x: float) -> float:
        """Safe logarithm to avoid log(0) or log(negative)."""
        return np.log(max(x, 1e-16))

    def _safe_sqrt(self, x: float) -> float:
        """Safe square root to avoid sqrt(negative)."""
        return np.sqrt(max(x, 0.0))

    def _safe_exp(self, x: float) -> float:
        """Safe exponential to avoid overflow/underflow."""
        if x > 700:
            return np.exp(700)
        if x < -700:
            return 0.0
        return np.exp(x)

    def __repr__(self):
        return f"AmericanOptionAnalyticalEngine(method='{self.method}')"
