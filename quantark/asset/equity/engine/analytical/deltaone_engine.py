"""
Analytical pricing engine for delta one products.
"""

import math
from typing import Dict, Optional
from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.deltaone import BaseDeltaOneProduct, SpotInstrument, Futures
from quantark.asset.equity.param import EngineParams
from quantark.priceenv import PricingEnvironment
from quantark.util.enum.deltaone_enums import DeltaOneType
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import ValidationError, PricingError


class DeltaOneEngine(BaseEngine):
    """
    Analytical pricing engine for delta one products.
    
    This engine prices stocks, indices, ETFs, and futures contracts using
    forward pricing formulas with cost-of-carry. For futures, it supports
    both theoretical pricing (with basis) and mark-to-market pricing based
    on observed market prices.
    
    Pricing Methods:
        - SpotInstrument: Forward pricing F(t,T) = S(t) * exp((r - q) * T)
        - Futures: Forward with basis, or mark-to-market if available
    
    Greeks:
        - Delta: ~1.0 (or exp(-q*T) for forward positions)
        - Gamma: 0 (linear payoff)
        - Vega: 0 (no volatility exposure)
        - Theta: carry costs
        - Rho: value * time
    
    Attributes:
        engine_type: EngineType.ANALYTICAL
        use_market_price: If True, use mark-to-market for futures when available
    """
    
    engine_type = EngineType.ANALYTICAL

    def __init__(self, params: Optional[EngineParams] = None, use_market_price: bool = False):
        """
        Initialize delta one engine.
        
        Args:
            params: Engine configuration parameters
            use_market_price: If True, use mark-to-market prices for futures when available
        """
        super().__init__(params)
        self.use_market_price = use_market_price
    
    def price(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment
    ) -> float:
        """
        Calculate the price of a delta one product.
        
        For spot instruments, returns current spot value.
        For futures, returns either theoretical price or mark-to-market based on settings.
        
        Args:
            product: Delta one product to price
            pricing_env: Pricing environment with market data
        
        Returns:
            Product price
        
        Raises:
            PricingError: If product is not a delta one product
        """
        if not isinstance(product, BaseDeltaOneProduct):
            raise PricingError(
                f"DeltaOneEngine only supports BaseDeltaOneProduct, "
                f"got {type(product).__name__}"
            )
        
        # Handle SpotInstrument
        if isinstance(product, SpotInstrument):
            return self._price_spot_instrument(product, pricing_env)
        
        # Handle Futures
        elif isinstance(product, Futures):
            return self._price_futures(product, pricing_env)
        
        else:
            raise PricingError(f"Unknown delta one product type: {type(product).__name__}")
    
    def _price_spot_instrument(
        self,
        product: SpotInstrument,
        pricing_env: PricingEnvironment
    ) -> float:
        """
        Price a spot instrument (stock, index, ETF).
        
        For immediate valuation, returns current spot price.
        For forward valuation, uses cost-of-carry model.
        
        Args:
            product: Spot instrument
            pricing_env: Pricing environment
        
        Returns:
            Spot instrument price
        """
        # For spot instruments, price equals current spot
        return pricing_env.spot
    
    def _price_futures(
        self,
        product: Futures,
        pricing_env: PricingEnvironment
    ) -> float:
        """
        Price a futures contract.
        
        If use_market_price is True and market price is available, returns
        mark-to-market value. Otherwise returns theoretical price with basis.
        
        Args:
            product: Futures contract
            pricing_env: Pricing environment
        
        Returns:
            Futures price (per unit, not including multiplier effect on P&L)
        """
        # Check for mark-to-market pricing
        if self.use_market_price and product.market_price is not None:
            return product.market_price
        
        # Calculate theoretical price with basis
        T = product.get_maturity(pricing_env)
        S = pricing_env.spot
        r = pricing_env.get_rate(T)
        q = pricing_env.get_div_yield(T)
        
        # Use product's forward pricing method
        forward_price = product.get_forward_price(S, r, q, T)
        
        return forward_price
    
    def calculate_greeks(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment
    ) -> Dict[str, float]:
        """
        Calculate Greeks for delta one products.
        
        Delta one products have:
        - Delta: approximately 1.0 (or exp(-q*T) for forwards)
        - Gamma: 0 (linear payoff)
        - Vega: 0 (no volatility exposure)
        - Theta: carry costs
        - Rho: value sensitivity to rates
        
        Args:
            product: Delta one product
            pricing_env: Pricing environment
        
        Returns:
            Dictionary of Greeks
        """
        if not isinstance(product, BaseDeltaOneProduct):
            raise PricingError(
                f"DeltaOneEngine only supports BaseDeltaOneProduct, "
                f"got {type(product).__name__}"
            )
        
        base_price = self.price(product, pricing_env)
        greeks = {"price": base_price}
        
        # Get parameters
        S = pricing_env.spot
        T = product.get_maturity(pricing_env)
        r = pricing_env.get_rate(T)
        q = pricing_env.get_div_yield(T)
        
        # Calculate analytical Greeks for delta one products
        if isinstance(product, SpotInstrument):
            greeks.update(self._calculate_spot_greeks(product, pricing_env, S, T, r, q))
        elif isinstance(product, Futures):
            greeks.update(self._calculate_futures_greeks(product, pricing_env, S, T, r, q))
        
        return greeks
    
    def _calculate_spot_greeks(
        self,
        product: SpotInstrument,
        pricing_env: PricingEnvironment,
        S: float,
        T: float,
        r: float,
        q: float
    ) -> Dict[str, float]:
        """
        Calculate analytical Greeks for spot instruments.
        
        Args:
            product: Spot instrument
            pricing_env: Pricing environment
            S: Spot price
            T: Time to maturity (very large for perpetual)
            r: Risk-free rate
            q: Dividend yield
        
        Returns:
            Dictionary of Greeks
        """
        greeks = {}
        
        # Delta: 1.0 for spot holdings
        greeks["delta"] = 1.0
        
        # Gamma: 0 for linear payoff
        greeks["gamma"] = 0.0
        
        # Vega: 0 (no volatility exposure)
        greeks["vega"] = 0.0
        
        # Theta: carry costs (r - q) * S
        # For spot holdings, theta represents the drift from carry
        greeks["theta"] = (r - q) * S
        
        # Rho: S (value increases linearly with rate for forward positions)
        # For spot, rho is approximately S * T, but T is very large
        # Use a normalized value
        greeks["rho"] = S * min(T, 1.0)
        
        return greeks
    
    def _calculate_futures_greeks(
        self,
        product: Futures,
        pricing_env: PricingEnvironment,
        S: float,
        T: float,
        r: float,
        q: float
    ) -> Dict[str, float]:
        """
        Calculate analytical Greeks for futures contracts.
        
        Args:
            product: Futures contract
            pricing_env: Pricing environment
            S: Spot price
            T: Time to maturity
            r: Risk-free rate
            q: Dividend yield
        
        Returns:
            Dictionary of Greeks
        """
        greeks = {}
        
        # If using market price, Greeks are simplified
        if self.use_market_price and product.market_price is not None:
            # Delta: 1.0 (futures track spot 1-to-1)
            greeks["delta"] = 1.0
            greeks["gamma"] = 0.0
            greeks["vega"] = 0.0
            greeks["theta"] = 0.0  # Market price is fixed
            greeks["rho"] = 0.0  # Market price independent of model rate
            greeks["dividend_rho"] = 0.0  # market price independent of model carry
            return greeks
        
        # Theoretical Greeks
        carry_cost = (r - q) * T
        discount_factor = math.exp(-q * T)
        
        # Delta: exp(-q*T) for forward/futures
        # This is the delta with respect to spot changes
        greeks["delta"] = discount_factor * math.exp(carry_cost) / S * S
        # Simplifies to exp(r*T), but more accurately:
        greeks["delta"] = math.exp(r * T)
        
        # For futures, delta is often considered 1.0 since futures track spot
        # Using simplified delta = 1.0 for practical hedging
        greeks["delta"] = 1.0
        
        # Gamma: 0 (linear payoff)
        greeks["gamma"] = 0.0
        
        # Vega: 0 (no volatility exposure)
        greeks["vega"] = 0.0
        
        # Theta: derivative of forward price with respect to time
        # F = S*exp((r-q)*T) + basis*exp(-λ*T)
        # dF/dT = S*(r-q)*exp((r-q)*T) - basis*λ*exp(-λ*T)
        forward_theta = S * (r - q) * math.exp(carry_cost)
        basis_theta = -product.basis * product.basis_decay_rate * math.exp(-product.basis_decay_rate * T)
        greeks["theta"] = forward_theta + basis_theta
        
        # Rho: derivative with respect to rate
        # dF/dr = S*T*exp((r-q)*T)
        greeks["rho"] = S * T * math.exp(carry_cost)

        # Dividend rho: dF/dq = -S*T*exp((r-q)*T); basis term is q-independent.
        # Per 1% q change; negative for long futures (higher carry lowers F).
        greeks["dividend_rho"] = -S * T * math.exp(carry_cost) * 0.01

        return greeks
    
    def get_forward_price(
        self,
        product: BaseDeltaOneProduct,
        pricing_env: PricingEnvironment,
        forward_time: float
    ) -> float:
        """
        Calculate forward price at a specific forward time.
        
        This is useful for pricing at different forward dates or
        for term structure analysis.
        
        Args:
            product: Delta one product
            pricing_env: Pricing environment
            forward_time: Time to forward date in years
        
        Returns:
            Forward price
        
        Raises:
            ValidationError: If forward_time is negative
        """
        if forward_time < 0:
            raise ValidationError(f"Forward time must be non-negative, got {forward_time}")
        
        S = pricing_env.spot
        r = pricing_env.get_rate(forward_time)
        q = pricing_env.get_div_yield(forward_time)
        
        return product.get_forward_price(S, r, q, forward_time)
    
    def __repr__(self):
        mtm_str = ", MTM" if self.use_market_price else ""
        return f"DeltaOneEngine({mtm_str})"

