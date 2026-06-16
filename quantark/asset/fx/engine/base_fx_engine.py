"""
Base class for FX pricing engines.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Optional

from quantark.asset.fx.product.base_fx_product import BaseFxProduct
from quantark.param import FlatVolSurface, TermStructureVolSurface
from quantark.param.vol.vannavolga import VannaVolgaVolSurface, SmileQuotes
from quantark.param.rrf import ParallelShiftRateCurve
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import PricingError
from quantark.util.numerical import is_zero, safe_exp


@dataclass
class FxEngineParams:
    """
    Bump sizes for finite-difference FX Greeks.

    The conventions mirror the legacy FX risk measurement factors:
        - spot_bump: relative spot bump for delta/gamma
        - vol_bump: relative vol bump; vega is reported per 1% vol change
        - rate_bump: absolute rate bump; rho is reported as dV/dr / 100
        - theta_days: forward time step in days; theta is daily decay
    """

    spot_bump: float = 0.001
    vol_bump: float = 0.01
    rate_bump: float = 0.0001
    theta_days: float = 1.0


class BaseFxEngine(ABC):
    """
    Abstract base class for FX pricing engines.

    Engines compute prices and Greeks from a product specification and an
    FxPricingEnvironment. The default calculate_greeks implementation uses
    bump-and-reprice; analytical engines override it with closed forms.
    """

    engine_type: EngineType = EngineType.ANALYTICAL

    def __init__(self, params: Optional[FxEngineParams] = None):
        self.params = params if params is not None else FxEngineParams()

    @abstractmethod
    def price(self, product: BaseFxProduct, fx_env: FxPricingEnvironment) -> float:
        """
        Calculate the product price in domestic (quote) currency.

        Args:
            product: The FX product to price
            fx_env: FX pricing environment with market data

        Returns:
            Product price in domestic currency
        """

    def calculate_greeks(
        self, product: BaseFxProduct, fx_env: FxPricingEnvironment
    ) -> Dict[str, float]:
        """
        Calculate FX Greeks by central finite differences.

        Returns a dictionary with the legacy FX conventions:
            - price: base price
            - delta, gamma: spot sensitivities (absolute)
            - delta_percentage, gamma_percentage: scaled by spot (spot^2) / value
            - fwd_delta: delta grown to delivery, delta * exp(r_f * t_del)
            - delta_premium / fwd_delta_premium: premium-adjusted deltas when
              the product carries a foreign-currency premium
            - vega: per 1% vol change
            - theta: daily time decay V(t+1d) - V(t)
            - rho_dom, rho_for: dV/dr / 100
        """
        base_price = self.price(product, fx_env)
        greeks: Dict[str, float] = {"price": base_price}

        delta, gamma = self._fdm_delta_gamma(product, fx_env, base_price)
        greeks["delta"] = delta
        greeks["gamma"] = gamma
        greeks["delta_percentage"] = (
            delta * fx_env.spot / base_price if not is_zero(base_price) else 0.0
        )
        greeks["gamma_percentage"] = (
            gamma * fx_env.spot**2 / base_price if not is_zero(base_price) else 0.0
        )

        t_delivery = product.get_delivery(fx_env)
        r_for = fx_env.get_foreign_rate(t_delivery)
        fwd_delta = delta * float(safe_exp(r_for * t_delivery))
        greeks["fwd_delta"] = fwd_delta
        greeks["delta_premium"], greeks["fwd_delta_premium"] = self._premium_deltas(
            product, fx_env, delta, fwd_delta
        )

        greeks["vega"] = self._fdm_vega(product, fx_env)
        greeks["theta"] = self._fdm_theta(product, fx_env, base_price)
        greeks["rho_dom"] = self._fdm_rho(product, fx_env, curve="domestic")
        greeks["rho_for"] = self._fdm_rho(product, fx_env, curve="foreign")

        return greeks

    # ------------------------------------------------------------------
    # Finite-difference building blocks
    # ------------------------------------------------------------------

    def _fdm_delta_gamma(
        self,
        product: BaseFxProduct,
        fx_env: FxPricingEnvironment,
        base_price: float,
    ) -> tuple:
        h = self.params.spot_bump
        spot = fx_env.spot

        env_up = deepcopy(fx_env)
        env_up.spot_quote.spot = spot * (1 + h)
        env_down = deepcopy(fx_env)
        env_down.spot_quote.spot = spot * (1 - h)

        price_up = self.price(product, env_up)
        price_down = self.price(product, env_down)

        delta = (price_up - price_down) / (2 * spot * h)
        gamma = (price_up - 2 * base_price + price_down) / (spot * h) ** 2
        return delta, gamma

    def _fdm_vega(
        self, product: BaseFxProduct, fx_env: FxPricingEnvironment
    ) -> float:
        if fx_env.vol_surface is None:
            return 0.0
        h = self.params.vol_bump
        t = product.get_maturity(fx_env)
        strike = getattr(product, "strike", fx_env.spot)
        current_vol = fx_env.get_vol(strike, t)

        env_up = self._vol_scaled_env(fx_env, 1 + h)
        env_down = self._vol_scaled_env(fx_env, 1 - h)
        price_up = self.price(product, env_up)
        price_down = self.price(product, env_down)

        # Central difference, reported per 1% vol change
        return (price_up - price_down) / (2 * current_vol * h) * 0.01

    def _vol_scaled_env(
        self, fx_env: FxPricingEnvironment, factor: float
    ) -> FxPricingEnvironment:
        env = deepcopy(fx_env)
        surface = fx_env.vol_surface
        if isinstance(surface, FlatVolSurface):
            env.vol_surface = FlatVolSurface(volatility=surface.volatility * factor)
        elif isinstance(surface, TermStructureVolSurface):
            env.vol_surface = TermStructureVolSurface(
                times=list(surface.times),
                vols=[float(v) * factor for v in surface.vols],
            )
        elif isinstance(surface, VannaVolgaVolSurface):
            # Full-quote vega bump: scale ATM/RR/BF together so the whole smile
            # shifts (sticky-delta). RR/BF can be zero or negative, so scaling
            # by `factor` is the right multiplicative bump for all three.
            q = surface.quotes
            env.vol_surface = surface.with_quotes(
                SmileQuotes(
                    sigma_atm=q.sigma_atm * factor,
                    rr25=q.rr25 * factor,
                    bf25_2vol=q.bf25_2vol * factor,
                )
            )
        else:
            raise PricingError(
                f"Vol bumping not supported for surface type "
                f"{type(surface).__name__}"
            )
        return env

    def _fdm_theta(
        self,
        product: BaseFxProduct,
        fx_env: FxPricingEnvironment,
        base_price: float,
    ) -> float:
        days = self.params.theta_days
        dt = days / 365.0

        if product.maturity is not None:
            shifted = deepcopy(product)
            shifted.time_shift(dt)
            future_price = self.price(shifted, fx_env)
        else:
            from datetime import timedelta

            env_future = deepcopy(fx_env)
            env_future.valuation_date = fx_env.valuation_date + timedelta(days=days)
            future_price = self.price(product, env_future)

        # Forward difference: daily decay
        return (future_price - base_price) / days

    def _fdm_rho(
        self,
        product: BaseFxProduct,
        fx_env: FxPricingEnvironment,
        curve: str,
    ) -> float:
        h = self.params.rate_bump

        env_up = deepcopy(fx_env)
        env_down = deepcopy(fx_env)
        if curve == "domestic":
            env_up.domestic_curve = ParallelShiftRateCurve(fx_env.domestic_curve, h)
            env_down.domestic_curve = ParallelShiftRateCurve(fx_env.domestic_curve, -h)
        else:
            env_up.foreign_curve = ParallelShiftRateCurve(fx_env.foreign_curve, h)
            env_down.foreign_curve = ParallelShiftRateCurve(fx_env.foreign_curve, -h)

        price_up = self.price(product, env_up)
        price_down = self.price(product, env_down)

        # Legacy convention: dV/dr scaled by 1/100
        return (price_up - price_down) / (2 * h) / 100.0

    def _premium_deltas(
        self,
        product: BaseFxProduct,
        fx_env: FxPricingEnvironment,
        delta: float,
        fwd_delta: float,
    ) -> tuple:
        """
        Premium-adjusted deltas for options whose premium is paid in the
        foreign currency. Products without premium fields pass through.
        """
        premium_currency = getattr(product, "premium_currency", None)
        premium_amount = getattr(product, "premium_amount", None)
        if (
            premium_currency is None
            or premium_amount is None
            or str(premium_currency).upper() not in ("FOR", "FOREIGN")
        ):
            return delta, fwd_delta

        spot = fx_env.effective_spot()
        forward = fx_env.get_forward(product.get_delivery(fx_env))
        delta_premium = delta + (premium_amount / spot if not is_zero(spot) else 0.0)
        fwd_delta_premium = fwd_delta + (
            premium_amount / forward if not is_zero(forward) else 0.0
        )
        return delta_premium, fwd_delta_premium

    def __repr__(self):
        return f"{self.__class__.__name__}()"
