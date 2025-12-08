"""
Greeks calculation for equity derivatives.
"""

import math
from datetime import timedelta
from typing import Dict, Optional, Tuple
from scipy import stats
from copy import deepcopy
from asset.equity.product.option import EuropeanVanillaOption
from asset.equity.product.option.observation_schedule import ObservationSchedule
from asset.equity.product.base_equity_product import BaseEquityProduct
from asset.equity.engine.base_engine import BaseEngine
from asset.equity.param import EngineParams
from priceenv import PricingEnvironment
from util.exceptions import ValidationError
from util.calendar import calculate_year_fraction


class GreeksCalculator:
    """
    Calculator for option Greeks using both analytical and numerical methods.

    Supports:
    - Analytical Greeks: Using closed-form Black-Scholes formulas
    - Numerical Greeks: Using finite difference method (FDM)
    """

    def __init__(self, params: Optional[EngineParams] = None):
        """
        Initialize Greeks calculator.

        Args:
            params: Engine parameters (for bump size in FDM)
        """
        self.params = params if params is not None else EngineParams()

    def calculate_analytical_greeks(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        price: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Calculate Greeks using analytical Black-Scholes formulas.

        Only works for European vanilla options under Black-Scholes model.

        Args:
            product: European vanilla option
            pricing_env: Pricing environment
            price: Pre-calculated price (optional, will calculate if not provided)

        Returns:
            Dictionary of Greeks: delta, gamma, vega, theta, rho

        Raises:
            ValidationError: If product is not a European vanilla option
        """
        if not isinstance(product, EuropeanVanillaOption):
            raise ValidationError(
                f"Analytical Greeks only support EuropeanVanillaOption, "
                f"got {type(product).__name__}"
            )

        # Extract parameters
        S = pricing_env.spot
        K = product.strike
        T = product.get_maturity(pricing_env)
        r = pricing_env.get_rate(T)
        q = pricing_env.get_div_yield(T)
        sigma = pricing_env.get_vol(K, T)

        # Handle edge case: option at expiry
        if T < 1e-10:
            return self._greeks_at_expiry(product, S)

        # Calculate d1 and d2
        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        # Calculate discount factors
        discount_div = math.exp(-q * T)
        discount_rf = math.exp(-r * T)

        # Standard normal PDF and CDF
        n_d1 = stats.norm.pdf(d1)  # phi(d1)
        N_d1 = stats.norm.cdf(d1)  # Phi(d1)
        N_d2 = stats.norm.cdf(d2)  # Phi(d2)

        greeks = {}

        # Calculate price if not provided
        if price is None:
            if product.is_call():
                price = S * discount_div * N_d1 - K * discount_rf * N_d2
            else:
                price = K * discount_rf * stats.norm.cdf(
                    -d2
                ) - S * discount_div * stats.norm.cdf(-d1)
        greeks["price"] = price

        # Delta: ∂V/∂S
        if product.is_call():
            delta = discount_div * N_d1
        else:
            delta = -discount_div * stats.norm.cdf(-d1)
        greeks["delta"] = delta

        # Gamma: ∂²V/∂S²
        gamma = discount_div * n_d1 / (S * sigma * sqrt_T)
        greeks["gamma"] = gamma

        # Vega: ∂V/∂σ (divided by 100 for 1% change)
        vega = S * discount_div * n_d1 * sqrt_T / 100
        greeks["vega"] = vega

        # Theta: ∂V/∂t (per day, divided by 365)
        term1 = -S * discount_div * n_d1 * sigma / (2 * sqrt_T)
        if product.is_call():
            term2 = -r * K * discount_rf * N_d2
            term3 = q * S * discount_div * N_d1
            theta = (term1 + term2 + term3) / 365
        else:
            term2 = r * K * discount_rf * stats.norm.cdf(-d2)
            term3 = -q * S * discount_div * stats.norm.cdf(-d1)
            theta = (term1 + term2 + term3) / 365
        greeks["theta"] = theta

        # Rho: ∂V/∂r (divided by 100 for 1% change)
        if product.is_call():
            rho = K * T * discount_rf * N_d2 / 100
        else:
            rho = -K * T * discount_rf * stats.norm.cdf(-d2) / 100
        greeks["rho"] = rho

        return greeks

    def calculate_numerical_greeks(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        base_price: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Calculate Greeks using finite difference method (FDM).

        Uses central differences for better accuracy.
        Works for any product and engine combination.

        Args:
            product: The derivative product
            pricing_env: Pricing environment
            engine: Pricing engine to use
            base_price: Pre-calculated base price (optional)

        Returns:
            Dictionary of Greeks: delta, gamma, vega, theta, rho
        """
        # Calculate base price if not provided
        if base_price is None:
            base_price = engine.price(product, pricing_env)

        # Check if this is a delta-one product (SpotInstrument, Futures)
        if self._is_deltaone_product(product):
            return self._greeks_for_deltaone(product, base_price)

        greeks = {"price": base_price}
        bump = self.params.bump_size

        # Precompute spot bumps once for delta/gamma.
        spot_prices = self._spot_bumped_prices(
            product, pricing_env, engine, bump, base_price=base_price
        )[1:]

        greeks["delta"] = self.calculate_numerical_delta(
            product,
            pricing_env,
            engine,
            base_price=base_price,
            spot_prices=spot_prices,
            bump=bump,
        )
        greeks["gamma"] = self.calculate_numerical_gamma(
            product,
            pricing_env,
            engine,
            base_price=base_price,
            spot_prices=spot_prices,
            bump=bump,
        )
        greeks["vega"] = self.calculate_numerical_vega(
            product, pricing_env, engine, base_price=base_price
        )
        greeks["theta"] = self.calculate_numerical_theta(
            product, pricing_env, engine, base_price=base_price
        )
        greeks["rho"] = self.calculate_numerical_rho(
            product, pricing_env, engine, base_price=base_price
        )

        return greeks

    def calculate_numerical_delta(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        base_price: Optional[float] = None,
        spot_prices: Optional[Tuple[float, float]] = None,
        bump: Optional[float] = None,
    ) -> float:
        """Numerical delta using central spot bump."""
        bump = bump if bump is not None else self.params.bump_size
        _, price_up_spot, price_down_spot = self._spot_bumped_prices(
            product,
            pricing_env,
            engine,
            bump,
            base_price=base_price,
            reuse=spot_prices,
        )
        return (price_up_spot - price_down_spot) / (2 * pricing_env.spot * bump)

    def calculate_numerical_gamma(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        base_price: Optional[float] = None,
        spot_prices: Optional[Tuple[float, float]] = None,
        bump: Optional[float] = None,
    ) -> float:
        """Numerical gamma using central spot bump."""
        bump = bump if bump is not None else self.params.bump_size
        base_price = (
            base_price if base_price is not None else engine.price(product, pricing_env)
        )
        _, price_up_spot, price_down_spot = self._spot_bumped_prices(
            product,
            pricing_env,
            engine,
            bump,
            base_price=base_price,
            reuse=spot_prices,
        )
        return (price_up_spot - 2 * base_price + price_down_spot) / (
            pricing_env.spot * bump
        ) ** 2

    def calculate_numerical_vega(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        base_price: Optional[float] = None,
    ) -> float:
        """Numerical vega from a 1% absolute vol bump."""
        base_price = (
            base_price if base_price is not None else engine.price(product, pricing_env)
        )
        env_up_vol = deepcopy(pricing_env)
        T = product.get_maturity(pricing_env)
        strike = getattr(product, "strike", pricing_env.spot)
        current_vol = pricing_env.get_vol(strike, T)
        from param.vol import FlatVolSurface

        env_up_vol.vol_surface = FlatVolSurface(current_vol + 0.01)
        price_up_vol = engine.price(product, env_up_vol)
        return price_up_vol - base_price  # Per 1% change

    def calculate_numerical_theta(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        base_price: Optional[float] = None,
    ) -> float:
        """Numerical theta via time bump with observation schedule handling."""
        base_price = (
            base_price if base_price is not None else engine.price(product, pricing_env)
        )
        product_theta = deepcopy(product)
        env_theta = deepcopy(pricing_env)
        current_maturity = product.get_maturity(pricing_env)
        
        bumped_date = pricing_env.valuation_date + timedelta(days=1)
        time_bump = calculate_year_fraction(
            pricing_env.valuation_date,
            bumped_date,
            pricing_env.day_count_convention,
            pricing_env.bus_days_in_year
        )

        if current_maturity <= time_bump:
            return 0.0

        valuation_bumped = False

        if getattr(product_theta, "exercise_date", None) is None:
            product_theta.maturity -= time_bump
        else:
            env_theta.valuation_date = bumped_date
            valuation_bumped = True

        dropped_all_observations = False
        schedule_ref = getattr(product_theta, "observation_schedule", None)
        schedule_is_observation = isinstance(schedule_ref, ObservationSchedule)
        bumped_schedule = None

        if schedule_is_observation:
            uses_dates = any(
                getattr(rec, "observation_date", None) is not None
                for rec in schedule_ref.records
            )
            if uses_dates and not valuation_bumped:
                env_theta.valuation_date = bumped_date
                valuation_bumped = True

            bumped_schedule = self._bump_observation_schedule_for_theta(
                schedule_ref,
                time_bump=time_bump,
                bumped_valuation_date=(
                    env_theta.valuation_date if uses_dates else None
                ),
            )

            if bumped_schedule is None:
                dropped_all_observations = True
            else:
                product_theta.observation_schedule = bumped_schedule
                self._sync_legacy_observation_dates(
                    product_theta, bumped_schedule, env_theta
                )

        if dropped_all_observations:
            return 0.0

        price_theta = engine.price(product_theta, env_theta)
        return price_theta - base_price

    def calculate_numerical_rho(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        base_price: Optional[float] = None,
    ) -> float:
        """Numerical rho from a 1% rate bump."""
        base_price = (
            base_price if base_price is not None else engine.price(product, pricing_env)
        )
        env_up_rate = deepcopy(pricing_env)
        from param.rrf import FlatRateCurve

        T = product.get_maturity(pricing_env)
        current_rate = pricing_env.get_rate(T)
        env_up_rate.rate_curve = FlatRateCurve(current_rate + 0.01)
        price_up_rate = engine.price(product, env_up_rate)
        return price_up_rate - base_price  # Per 1% change

    def _spot_bumped_prices(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        bump: float,
        base_price: Optional[float] = None,
        reuse: Optional[Tuple[float, float]] = None,
    ) -> Tuple[float, float, float]:
        """
        Compute base, up, and down spot bump prices, optionally reusing bumps.
        """
        base_price = (
            base_price if base_price is not None else engine.price(product, pricing_env)
        )
        if reuse is not None:
            price_up_spot, price_down_spot = reuse
        else:
            env_up = deepcopy(pricing_env)
            env_up.spot_quote.spot *= 1 + bump
            price_up_spot = engine.price(product, env_up)

            env_down = deepcopy(pricing_env)
            env_down.spot_quote.spot *= 1 - bump
            price_down_spot = engine.price(product, env_down)

        return base_price, price_up_spot, price_down_spot

    def _is_deltaone_product(self, product: BaseEquityProduct) -> bool:
        """
        Check if product is a delta-one product (no optionality).

        Delta-one products include SpotInstrument and Futures.

        Args:
            product: Product to check

        Returns:
            True if delta-one product
        """
        # Check by class name to avoid circular imports
        product_class_name = product.__class__.__name__
        return product_class_name in ["SpotInstrument", "Futures"]

    def _greeks_for_deltaone(
        self, product: BaseEquityProduct, price: float
    ) -> Dict[str, float]:
        """
        Calculate Greeks for delta-one products.

        Delta-one products have trivial Greeks:
        - Delta = 1.0 (always)
        - Gamma, Vega, Theta, Rho = 0.0 (no optionality)

        Args:
            product: Delta-one product
            price: Current price

        Returns:
            Dictionary of Greeks
        """
        return {
            "price": price,
            "delta": 1.0,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "rho": 0.0,
        }

    def _greeks_at_expiry(
        self, product: EuropeanVanillaOption, spot: float
    ) -> Dict[str, float]:
        """
        Calculate Greeks at expiry.

        At expiry:
        - Price = intrinsic value
        - Delta = 1 (ITM call), -1 (ITM put), 0 (OTM)
        - Gamma, Vega, Theta, Rho = 0

        Args:
            product: European vanilla option
            spot: Current spot price

        Returns:
            Dictionary of Greeks
        """
        price = product.get_payoff(spot)

        # Delta at expiry
        if product.is_call():
            delta = 1.0 if spot > product.strike else 0.0
        else:
            delta = -1.0 if spot < product.strike else 0.0

        return {
            "price": price,
            "delta": delta,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "rho": 0.0,
        }

    def compare_greeks(
        self, analytical: Dict[str, float], numerical: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare analytical and numerical Greeks.

        Args:
            analytical: Analytical Greeks
            numerical: Numerical Greeks

        Returns:
            Dictionary with 'analytical', 'numerical', and 'difference' sub-dictionaries
        """
        difference = {}
        for key in analytical:
            if key in numerical:
                diff = analytical[key] - numerical[key]
                rel_diff = diff / analytical[key] if abs(analytical[key]) > 1e-10 else 0
                difference[key] = {"absolute": diff, "relative": rel_diff}

        return {
            "analytical": analytical,
            "numerical": numerical,
            "difference": difference,
        }

    def _bump_observation_schedule_for_theta(
        self,
        schedule: ObservationSchedule,
        time_bump: float,
        bumped_valuation_date=None,
    ) -> Optional[ObservationSchedule]:
        """
        Rebase an ObservationSchedule for theta bump:
        - Drop records that are in the past at the bumped valuation.
        - Reduce observation_time entries by the time bump without shifting dates forward.
        """
        if schedule is None or not getattr(schedule, "records", None):
            return schedule

        updated_records = []
        for rec in schedule.records:
            rec_copy = deepcopy(rec)

            if getattr(rec_copy, "observation_date", None) is not None:
                if (
                    bumped_valuation_date is not None
                    and rec_copy.observation_date <= bumped_valuation_date
                ):
                    continue  # already observed

            if getattr(rec_copy, "observation_time", None) is not None:
                adjusted_time = rec_copy.observation_time - time_bump
                if adjusted_time <= 0:
                    continue  # already observed
                rec_copy.observation_time = adjusted_time

            updated_records.append(rec_copy)

        if not updated_records:
            return None

        return ObservationSchedule(
            records=updated_records,
            aggregation_mode=schedule.aggregation_mode,
            frequency=schedule.frequency,
        )

    def _sync_legacy_observation_dates(
        self,
        product,
        bumped_schedule: ObservationSchedule,
        pricing_env: PricingEnvironment,
    ) -> None:
        """
        Synchronize legacy observation_dates with bumped ObservationSchedule.
        
        Only update legacy observation_dates when the schedule uses observation_time values
        to avoid overwriting legacy dates for date-based schedules.
        """
        if not hasattr(product, "observation_dates"):
            return
        
        if bumped_schedule.times:
            product.observation_dates = bumped_schedule.times