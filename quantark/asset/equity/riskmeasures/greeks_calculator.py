"""
Greeks calculation for equity derivatives.
"""

import math
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Dict, Optional, Sequence, Tuple

from scipy import stats

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.param import EngineParams
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import DayCountConvention, calculate_year_fraction
from quantark.util.enum import CommonGreek, EquityGreek
from quantark.util.enum.engine_enums import EngineType, GreeksCalculationMode
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_zero


class GreeksCalculator:
    """
    Calculator for option Greeks using both analytical and numerical methods.

    Supports:
    - Analytical Greeks: Using closed-form Black-Scholes formulas
    - Numerical Greeks: Using finite difference method (FDM)

    The greeks_mode parameter controls delta/gamma calculation for engines that
    implement their own calculate_greeks() method (e.g., PDE engines):
    - GreeksCalculationMode.BUMP: Always use finite difference bump method
    - GreeksCalculationMode.ENGINE: Use engine.calculate_greeks() when overridden
    - GreeksCalculationMode.AUTO: Use engine method for PDE engines, bump otherwise
    """

    def __init__(
        self,
        params: Optional[EngineParams] = None,
        greeks_mode: GreeksCalculationMode = GreeksCalculationMode.BUMP,
    ):
        """
        Initialize Greeks calculator.

        Args:
            params: Engine parameters (for bump sizes in FDM)
            greeks_mode: Mode for delta/gamma calculation when engine has
                        its own calculate_greeks() method (e.g., PDE engines)
        """
        self.params = params if params is not None else EngineParams()
        self._bump_config = self.params.get_effective_bump_config()
        self.greeks_mode = greeks_mode

    def calculate(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        method: str = "auto",
        greeks: Optional[Sequence[object]] = None,
    ) -> Dict[str, float]:
        """Unified entry point for Greeks calculation."""
        method = method.lower()
        if method not in ("auto", "analytical", "numerical"):
            raise ValidationError(f"Unknown greeks method: {method}")

        requested = self._normalize_greeks(greeks)
        analytical_supported = {
            "price",
            "delta",
            "gamma",
            "vega",
            "theta",
            "rho",
        }

        if method in ("auto", "analytical") and isinstance(
            product, EuropeanVanillaOption
        ):
            if requested is None or requested.issubset(analytical_supported):
                greeks_out = self.calculate_analytical_greeks(product, pricing_env)
                if requested is None:
                    return greeks_out
                return {key: greeks_out[key] for key in greeks_out if key in requested}
            if method == "analytical":
                raise ValidationError(
                    "Analytical greeks do not support requested greeks: "
                    f"{sorted(requested - analytical_supported)}"
                )

        return self.calculate_numerical_greeks(
            product, pricing_env, engine, greeks=greeks
        )

    def _normalize_greeks(
        self, greeks: Optional[Sequence[object]]
    ) -> Optional[set[str]]:
        if greeks is None:
            return None
        if len(greeks) == 0:
            return set()
        aliases = {
            "deltaq": "delta_q",
            "deltadq": "delta_q",
            "d_delta_d_q": "delta_q",
            "d_delta_dq": "delta_q",
            "rhoq": "dividend_rho",
            "div_rho": "dividend_rho",
            "dividendrho": "dividend_rho",
        }
        allowed = {
            "price",
            "delta",
            "gamma",
            "vega",
            "theta",
            "rho",
            "dividend_rho",
            "vanna",
            "volga",
            "delta_q",
            "charm",
            "color",
            "convexity_theta",
            "r_theta",
            "q_theta",
        }
        normalized: set[str] = set()
        for greek in greeks:
            if isinstance(greek, (CommonGreek, EquityGreek)):
                name = greek.value
            elif isinstance(greek, str):
                name = greek.strip().lower()
            else:
                raise ValidationError(
                    f"Unsupported greek identifier type: {type(greek).__name__}"
                )
            name = aliases.get(name, name)
            if name not in allowed:
                raise ValidationError(f"Unknown greek name: {name}")
            normalized.add(name)
        return normalized

    def _has_custom_greeks(self, engine: BaseEngine) -> bool:
        """Return True if engine overrides calculate_greeks()."""
        engine_calculate_greeks = getattr(engine.__class__, "calculate_greeks", None)
        return (
            engine_calculate_greeks is not None
            and engine_calculate_greeks is not BaseEngine.calculate_greeks
        )

    def _should_use_engine_greeks(self, engine: BaseEngine) -> bool:
        """
        Check if engine's calculate_greeks() should be used for delta/gamma.

        Args:
            engine: The pricing engine

        Returns:
            True if engine.calculate_greeks() should be used
        """
        if not self._has_custom_greeks(engine):
            return False

        if self.greeks_mode == GreeksCalculationMode.BUMP:
            return False
        if self.greeks_mode == GreeksCalculationMode.ENGINE:
            return True
        # AUTO mode: use for PDE engines
        return getattr(engine, "engine_type", None) == EngineType.PDE

    def _ensure_base_price(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        base_price: Optional[float],
    ) -> float:
        """Return base price, computing it if needed."""
        return (
            base_price if base_price is not None else engine.price(product, pricing_env)
        )

    def _resolve_bump_engine(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
    ) -> BaseEngine:
        """Return the engine context used for numerical bump repricing."""
        create_context = getattr(engine, "create_bump_context", None)
        if not callable(create_context):
            return engine
        bump_engine = create_context(product, pricing_env)
        return bump_engine if bump_engine is not None else engine

    def _calculate_sensitivity(
        self,
        base_price: float,
        price_up: float,
        price_down: Optional[float] = None,
        bump: float = 1.0,
        scale: float = 1.0,
        mode: str = "central",
    ) -> float:
        """Generic finite-difference sensitivity helper."""
        if mode == "central":
            if price_down is None:
                raise ValidationError("central mode requires price_down")
            return (price_up - price_down) / (2.0 * scale * bump)
        if mode == "second_order":
            if price_down is None:
                raise ValidationError("second_order mode requires price_down")
            return (price_up - 2.0 * base_price + price_down) / (
                scale * bump
            ) ** 2
        if mode == "one_sided":
            return price_up - base_price
        raise ValidationError(f"Unknown sensitivity mode: {mode}")

    def _get_delta_gamma(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        base_price: Optional[float],
    ) -> Tuple[float, float, float]:
        """Get base price, delta, and gamma via engine or bump method."""
        if self._should_use_engine_greeks(engine):
            engine_greeks = engine.calculate_greeks(product, pricing_env)
            if base_price is None:
                base_price = engine_greeks["price"]
            return base_price, engine_greeks["delta"], engine_greeks["gamma"]

        base_price = self._ensure_base_price(product, pricing_env, engine, base_price)
        spot_prices = self._spot_bumped_prices(
            product, pricing_env, engine, self._bump_config.spot_bump, base_price=base_price
        )[1:]

        delta = self.calculate_numerical_delta(
            product,
            pricing_env,
            engine,
            base_price=base_price,
            spot_prices=spot_prices,
            bump=self._bump_config.spot_bump,
        )
        gamma = self.calculate_numerical_gamma(
            product,
            pricing_env,
            engine,
            base_price=base_price,
            spot_prices=spot_prices,
            bump=self._bump_config.spot_bump,
        )

        return base_price, delta, gamma

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
        if is_zero(T):
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

        multiplier = product.contract_multiplier

        # Calculate price if not provided (per-unit)
        if price is None:
            if product.is_call():
                price = S * discount_div * N_d1 - K * discount_rf * N_d2
            else:
                price = K * discount_rf * stats.norm.cdf(
                    -d2
                ) - S * discount_div * stats.norm.cdf(-d1)
        else:
            price = price / multiplier
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
        # Decomposed into three components:
        #   convexity_theta: time decay from gamma/convexity erosion (always negative)
        #   r_theta: time decay from interest rate cost of carry
        #   q_theta: time decay from dividend yield
        term1 = -S * discount_div * n_d1 * sigma / (2 * sqrt_T)
        if product.is_call():
            term2 = -r * K * discount_rf * N_d2
            term3 = q * S * discount_div * N_d1
        else:
            term2 = r * K * discount_rf * stats.norm.cdf(-d2)
            term3 = -q * S * discount_div * stats.norm.cdf(-d1)

        # Store decomposed components (per day)
        convexity_theta = term1 / 365
        r_theta = term2 / 365
        q_theta = term3 / 365
        theta = convexity_theta + r_theta + q_theta

        greeks["theta"] = theta
        greeks["convexity_theta"] = convexity_theta
        greeks["r_theta"] = r_theta
        greeks["q_theta"] = q_theta

        # Rho: ∂V/∂r (divided by 100 for 1% change)
        if product.is_call():
            rho = K * T * discount_rf * N_d2 / 100
        else:
            rho = -K * T * discount_rf * stats.norm.cdf(-d2) / 100
        greeks["rho"] = rho

        # Dividend Rho: ∂V/∂q (divided by 100 for 1% change)
        if product.is_call():
            dividend_rho = -S * T * discount_div * N_d1 / 100
        else:
            dividend_rho = S * T * discount_div * stats.norm.cdf(-d1) / 100
        greeks["dividend_rho"] = dividend_rho

        for key, value in greeks.items():
            greeks[key] = value * multiplier

        return greeks

    def calculate_numerical_greeks(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        base_price: Optional[float] = None,
        greeks: Optional[Sequence[object]] = None,
    ) -> Dict[str, float]:
        """
        Calculate Greeks using finite difference method (FDM).

        Uses central differences for better accuracy.
        Works for any product and engine combination.

        Bump sizes are configured via BumpConfig in EngineParams:
            - Delta/Gamma: Relative spot bump (default: 1%)
            - Vega: Absolute vol bump (default: 1 vol point)
            - Theta: Time bump in days (default: 1 day)
            - Rho: Absolute rate bump (default: 1bp), scaled to per 1% change
            - Dividend Rho: Absolute div yield bump (default: 1bp), scaled to per 1% change

        For delta and gamma, if greeks_mode is ENGINE or AUTO (with PDE engine),
        the engine's own calculate_greeks() method is used instead of bumping.

        Args:
            product: The derivative product
            pricing_env: Pricing environment
            engine: Pricing engine to use
            base_price: Pre-calculated base price (optional)

        Returns:
            Dictionary of Greeks for the requested set (or defaults if None).
        """
        requested = self._normalize_greeks(greeks)
        if requested is None:
            requested = {
                "price",
                "delta",
                "gamma",
                "vega",
                "theta",
                "rho",
                "dividend_rho",
                "convexity_theta",
                "r_theta",
                "q_theta",
            }

        if product.is_linear:
            base_price = self._ensure_base_price(product, pricing_env, engine, base_price)
            greeks_out = self._greeks_for_linear(product, base_price)
            for extra in requested:
                greeks_out.setdefault(extra, 0.0)
            return {key: greeks_out[key] for key in greeks_out if key in requested}

        bump_engine = self._resolve_bump_engine(product, pricing_env, engine)
        greeks_out: Dict[str, float] = {}
        if "price" in requested and base_price is not None:
            greeks_out["price"] = base_price

        delta = None
        gamma = None
        if {"delta", "gamma", "delta_q", "vanna"} & requested:
            base_price, delta, gamma = self._get_delta_gamma(
                product, pricing_env, bump_engine, base_price
            )
        if delta is not None and "delta" in requested:
            greeks_out["delta"] = delta
        if gamma is not None and "gamma" in requested:
            greeks_out["gamma"] = gamma

        if base_price is None:
            base_price = self._ensure_base_price(
                product, pricing_env, bump_engine, base_price
            )
        if "price" in requested and "price" not in greeks_out:
            greeks_out["price"] = base_price

        # Other Greeks always use bump method
        if "vega" in requested:
            greeks_out["vega"] = self.calculate_numerical_vega(
                product,
                pricing_env,
                bump_engine,
                base_price=base_price,
                vol_bump=self._bump_config.vol_bump,
            )
        if "volga" in requested:
            greeks_out["volga"] = self.calculate_numerical_volga(
                product,
                pricing_env,
                bump_engine,
                base_price=base_price,
                vol_bump=self._bump_config.vol_bump,
            )
        if "vanna" in requested:
            greeks_out["vanna"] = self.calculate_numerical_vanna(
                product,
                pricing_env,
                bump_engine,
                base_price=base_price,
                vol_bump=self._bump_config.vol_bump,
            )
        if "delta_q" in requested:
            greeks_out["delta_q"] = self.calculate_numerical_delta_q(
                product,
                pricing_env,
                bump_engine,
                base_price=base_price,
                div_bump=self._bump_config.div_bump,
                base_delta=delta,
            )
        if "theta" in requested:
            greeks_out["theta"] = self.calculate_numerical_theta(
                product,
                pricing_env,
                bump_engine,
                base_price=base_price,
                time_bump_days=self._bump_config.time_bump_days,
            )
        if "rho" in requested:
            greeks_out["rho"] = self.calculate_numerical_rho(
                product,
                pricing_env,
                bump_engine,
                base_price=base_price,
                rate_bump=self._bump_config.rate_bump,
            )
        if "dividend_rho" in requested:
            greeks_out["dividend_rho"] = self.calculate_numerical_dividend_rho(
                product,
                pricing_env,
                bump_engine,
                base_price=base_price,
                div_bump=self._bump_config.div_bump,
            )

        # Estimate theta components using fast approximation from existing Greeks
        if {"convexity_theta", "r_theta", "q_theta"} & requested:
            if "theta" not in greeks_out:
                greeks_out["theta"] = self.calculate_numerical_theta(
                    product,
                    pricing_env,
                    bump_engine,
                    base_price=base_price,
                    time_bump_days=self._bump_config.time_bump_days,
                )
            if "rho" not in greeks_out:
                greeks_out["rho"] = self.calculate_numerical_rho(
                    product,
                    pricing_env,
                    bump_engine,
                    base_price=base_price,
                    rate_bump=self._bump_config.rate_bump,
                )
            if "dividend_rho" not in greeks_out:
                greeks_out["dividend_rho"] = self.calculate_numerical_dividend_rho(
                    product,
                    pricing_env,
                    bump_engine,
                    base_price=base_price,
                    div_bump=self._bump_config.div_bump,
                )
            T = product.get_maturity(pricing_env)
            r = pricing_env.get_rate(T)
            q = pricing_env.get_div_yield(T)
            theta_components = self.estimate_theta_components(
                theta=greeks_out["theta"],
                rho=greeks_out["rho"],
                dividend_rho=greeks_out["dividend_rho"],
                r=r,
                q=q,
                T=T,
            )
            for key, value in theta_components.items():
                if key in requested:
                    greeks_out[key] = value

        return {key: greeks_out[key] for key in greeks_out if key in requested}

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
        base_price, price_up_spot, price_down_spot = self._spot_bumped_prices(
            product,
            pricing_env,
            engine,
            bump,
            base_price=base_price,
            reuse=spot_prices,
        )
        return self._calculate_sensitivity(
            base_price,
            price_up_spot,
            price_down_spot,
            bump=bump,
            scale=pricing_env.spot,
            mode="central",
        )

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
        base_price, price_up_spot, price_down_spot = self._spot_bumped_prices(
            product,
            pricing_env,
            engine,
            bump,
            base_price=base_price,
            reuse=spot_prices,
        )
        return self._calculate_sensitivity(
            base_price,
            price_up_spot,
            price_down_spot,
            bump=bump,
            scale=pricing_env.spot,
            mode="second_order",
        )

    def calculate_numerical_vega(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        base_price: Optional[float] = None,
        vol_bump: Optional[float] = None,
    ) -> float:
        """Numerical vega from a vol bump."""
        engine = self._resolve_bump_engine(product, pricing_env, engine)
        vol_bump = vol_bump if vol_bump is not None else self._bump_config.vol_bump
        base_price = self._ensure_base_price(product, pricing_env, engine, base_price)
        T = product.get_maturity(pricing_env)
        strike = getattr(product, "strike", pricing_env.spot)
        current_vol = pricing_env.get_vol(strike, T)
        env_up_vol = self._build_vol_bumped_env(
            pricing_env, product, current_vol, vol_bump, direction=1.0
        )
        price_up_vol = engine.price(product, env_up_vol)
        return self._calculate_sensitivity(
            base_price, price_up_vol, bump=vol_bump, mode="one_sided"
        )

    def calculate_numerical_volga(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        base_price: Optional[float] = None,
        vol_bump: Optional[float] = None,
    ) -> float:
        """Numerical volga (second derivative wrt vol) using vol bumps."""
        engine = self._resolve_bump_engine(product, pricing_env, engine)
        vol_bump = vol_bump if vol_bump is not None else self._bump_config.vol_bump
        base_price = self._ensure_base_price(product, pricing_env, engine, base_price)
        T = product.get_maturity(pricing_env)
        strike = getattr(product, "strike", pricing_env.spot)
        current_vol = pricing_env.get_vol(strike, T)

        if current_vol - vol_bump <= 0:
            env_up = self._build_vol_bumped_env(
                pricing_env, product, current_vol, vol_bump, direction=1.0
            )
            vega_base = self.calculate_numerical_vega(
                product, pricing_env, engine, base_price=base_price, vol_bump=vol_bump
            )
            vega_up = self.calculate_numerical_vega(
                product, env_up, engine, base_price=None, vol_bump=vol_bump
            )
            return (vega_up - vega_base) / vol_bump

        env_up = self._build_vol_bumped_env(
            pricing_env, product, current_vol, vol_bump, direction=1.0
        )
        env_down = self._build_vol_bumped_env(
            pricing_env, product, current_vol, vol_bump, direction=-1.0
        )
        price_up = engine.price(product, env_up)
        price_down = engine.price(product, env_down)
        return self._calculate_sensitivity(
            base_price, price_up, price_down, bump=vol_bump, mode="second_order"
        )

    def calculate_numerical_vanna(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        base_price: Optional[float] = None,
        vol_bump: Optional[float] = None,
    ) -> float:
        """Numerical vanna (cross derivative wrt spot and vol)."""
        engine = self._resolve_bump_engine(product, pricing_env, engine)
        vol_bump = vol_bump if vol_bump is not None else self._bump_config.vol_bump
        base_price = self._ensure_base_price(product, pricing_env, engine, base_price)
        T = product.get_maturity(pricing_env)
        strike = getattr(product, "strike", pricing_env.spot)
        current_vol = pricing_env.get_vol(strike, T)

        env_up = self._build_vol_bumped_env(
            pricing_env, product, current_vol, vol_bump, direction=1.0
        )
        env_down = self._build_vol_bumped_env(
            pricing_env, product, current_vol, vol_bump, direction=-1.0
        )

        if current_vol - vol_bump <= 0:
            base_delta = self.calculate_numerical_delta(
                product,
                pricing_env,
                engine,
                base_price=base_price,
                bump=self._bump_config.spot_bump,
            )
            delta_up = self.calculate_numerical_delta(
                product,
                env_up,
                engine,
                base_price=base_price,
                bump=self._bump_config.spot_bump,
            )
            return (delta_up - base_delta) / vol_bump

        delta_up = self.calculate_numerical_delta(
            product,
            env_up,
            engine,
            base_price=base_price,
            bump=self._bump_config.spot_bump,
        )
        delta_down = self.calculate_numerical_delta(
            product,
            env_down,
            engine,
            base_price=base_price,
            bump=self._bump_config.spot_bump,
        )
        return (delta_up - delta_down) / (2.0 * vol_bump)

    def calculate_numerical_theta(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        base_price: Optional[float] = None,
        time_bump_days: Optional[int] = None,
        time_bump_mode: Optional[str] = None,
    ) -> float:
        """
        Numerical theta via time bump with observation schedule handling.

        Theta date advancement is controlled by BumpConfig.time_bump_mode:
        "calendar_days" preserves legacy calendar-date bumps, "business_days"
        advances by valid pricing-calendar business days, and "auto" uses
        business days for BUSINESS_DAYS pricing environments with a calendar.
        """
        engine = self._resolve_bump_engine(product, pricing_env, engine)
        time_bump_days = (
            time_bump_days
            if time_bump_days is not None
            else self._bump_config.time_bump_days
        )
        time_bump_mode = (
            time_bump_mode
            if time_bump_mode is not None
            else getattr(self._bump_config, "time_bump_mode", "auto")
        )
        base_price = self._ensure_base_price(product, pricing_env, engine, base_price)
        product_theta = deepcopy(product)
        env_theta = deepcopy(pricing_env)
        current_maturity = product.get_maturity(pricing_env)

        bumped_date, time_bump, resolved_mode = self._advance_theta_bump(
            pricing_env, time_bump_days, time_bump_mode
        )

        if time_bump <= 0.0:
            if current_maturity <= 0.0:
                return 0.0
            if resolved_mode == "business_days":
                raise ValidationError(
                    "Business-day theta bump did not advance time: "
                    f"valuation_date={pricing_env.valuation_date}, "
                    f"bumped_date={bumped_date}, time_bump_days={time_bump_days}"
                )
            return 0.0
        if current_maturity <= time_bump:
            return 0.0

        env_theta.valuation_date = bumped_date
        dropped_all_observations = product_theta.time_shift(
            time_bump, bumped_date, env_theta
        )

        if dropped_all_observations:
            return 0.0

        price_theta = engine.price(product_theta, env_theta)
        return price_theta - base_price

    def _advance_theta_bump(
        self,
        pricing_env: PricingEnvironment,
        time_bump_days: int,
        time_bump_mode: str,
    ) -> Tuple[datetime, float, str]:
        """Advance the theta valuation date and return date, year fraction, mode."""
        mode = self._resolve_theta_bump_mode(pricing_env, time_bump_mode)
        if mode == "calendar_days":
            bumped_date = pricing_env.valuation_date + timedelta(days=time_bump_days)
        else:
            calendar = getattr(pricing_env, "calendar", None)
            if calendar is None or not hasattr(calendar, "add_business_days"):
                raise ValidationError(
                    "time_bump_mode='business_days' requires pricing_env.calendar "
                    "with add_business_days()"
                )
            bumped_date = calendar.add_business_days(
                pricing_env.valuation_date, time_bump_days
            )

        time_bump = calculate_year_fraction(
            pricing_env.valuation_date,
            bumped_date,
            pricing_env.day_count_convention,
            pricing_env.bus_days_in_year,
            calendar=getattr(pricing_env, "calendar", None),
        )
        return bumped_date, time_bump, mode

    @staticmethod
    def _resolve_theta_bump_mode(
        pricing_env: PricingEnvironment, time_bump_mode: str
    ) -> str:
        """Resolve auto theta mode against the pricing environment."""
        mode = time_bump_mode.lower()
        if mode not in {"auto", "calendar_days", "business_days"}:
            raise ValidationError(
                "time_bump_mode must be one of 'auto', 'calendar_days', "
                f"or 'business_days', got {time_bump_mode!r}"
            )
        if mode != "auto":
            return mode

        if (
            pricing_env.day_count_convention == DayCountConvention.BUSINESS_DAYS
            and getattr(pricing_env, "calendar", None) is not None
        ):
            return "business_days"
        return "calendar_days"

    def calculate_numerical_rho(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        base_price: Optional[float] = None,
        rate_bump: Optional[float] = None,
    ) -> float:
        """Numerical rho from a rate bump (per 1% rate change)."""
        engine = self._resolve_bump_engine(product, pricing_env, engine)
        rate_bump = rate_bump if rate_bump is not None else self._bump_config.rate_bump
        base_price = self._ensure_base_price(product, pricing_env, engine, base_price)
        env_up_rate = deepcopy(pricing_env)
        from quantark.param.rrf import FlatRateCurve

        T = product.get_maturity(pricing_env)
        current_rate = pricing_env.get_rate(T)
        env_up_rate.rate_curve = FlatRateCurve(current_rate + rate_bump)
        price_up_rate = engine.price(product, env_up_rate)
        raw = self._calculate_sensitivity(
            base_price, price_up_rate, bump=rate_bump, mode="one_sided"
        )
        return raw * (0.01 / rate_bump)

    def calculate_numerical_dividend_rho(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        base_price: Optional[float] = None,
        div_bump: Optional[float] = None,
    ) -> float:
        """
        Numerical dividend_rho (psi) from dividend yield bump.

        Measures price sensitivity to dividend yield changes:
            dividend_rho = dV/dq

        Args:
            product: The derivative product
            pricing_env: Pricing environment
            engine: Pricing engine
            base_price: Pre-calculated base price
            div_bump: Absolute dividend yield bump (uses config if None)

        Returns:
            Dividend rho value (price change per 1% div_yield change).
            Negative for call options (higher div = lower call price).
            Positive for put options (higher div = higher put price).
        """
        engine = self._resolve_bump_engine(product, pricing_env, engine)
        div_bump = div_bump if div_bump is not None else self._bump_config.div_bump
        base_price = self._ensure_base_price(product, pricing_env, engine, base_price)
        T = product.get_maturity(pricing_env)
        current_div = pricing_env.get_div_yield(T)
        env_up_div = self._build_div_bumped_env(
            pricing_env, product, current_div, div_bump, direction=1.0
        )
        price_up_div = engine.price(product, env_up_div)
        raw = self._calculate_sensitivity(
            base_price, price_up_div, bump=div_bump, mode="one_sided"
        )
        return raw * (0.01 / div_bump)

    def calculate_numerical_delta_q(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        base_price: Optional[float] = None,
        div_bump: Optional[float] = None,
        base_delta: Optional[float] = None,
    ) -> float:
        """Numerical dDelta/dq via dividend yield bumps."""
        engine = self._resolve_bump_engine(product, pricing_env, engine)
        div_bump = div_bump if div_bump is not None else self._bump_config.div_bump
        base_price = self._ensure_base_price(product, pricing_env, engine, base_price)
        T = product.get_maturity(pricing_env)
        current_div = pricing_env.get_div_yield(T)

        if base_delta is None:
            base_delta = self.calculate_numerical_delta(
                product,
                pricing_env,
                engine,
                base_price=base_price,
                bump=self._bump_config.spot_bump,
            )

        if current_div - div_bump < 0:
            env_up = self._build_div_bumped_env(
                pricing_env, product, current_div, div_bump, direction=1.0
            )
            delta_up = self.calculate_numerical_delta(
                product,
                env_up,
                engine,
                base_price=base_price,
                bump=self._bump_config.spot_bump,
            )
            return (delta_up - base_delta) / div_bump

        env_up = self._build_div_bumped_env(
            pricing_env, product, current_div, div_bump, direction=1.0
        )
        env_down = self._build_div_bumped_env(
            pricing_env, product, current_div, div_bump, direction=-1.0
        )
        delta_up = self.calculate_numerical_delta(
            product,
            env_up,
            engine,
            base_price=base_price,
            bump=self._bump_config.spot_bump,
        )
        delta_down = self.calculate_numerical_delta(
            product,
            env_down,
            engine,
            base_price=base_price,
            bump=self._bump_config.spot_bump,
        )
        return (delta_up - delta_down) / (2.0 * div_bump)

    def estimate_theta_components(
        self,
        theta: float,
        rho: float,
        dividend_rho: float,
        r: float,
        q: float,
        T: float,
        rate_bump: float = 0.01,
        dividend_bump: float = 0.01,
    ) -> Dict[str, float]:
        """
        Fast estimation of theta components from existing Greeks.

        Uses the relationships between theta components and other Greeks:
            r_theta ≈ -r/T * rho / rate_bump (corrected for scale and daily conversion)
            q_theta ≈ -q/T * dividend_rho / dividend_bump (corrected for scale and daily conversion)
            convexity_theta ≈ theta - r_theta - q_theta

        This is an approximation that avoids repricing. For exact decomposition,
        use _calculate_numerical_theta_components() instead.

        Args:
            theta: Total theta (per day)
            rho: Rho (sensitivity to rate, per 1% change)
            dividend_rho: Dividend rho (sensitivity to dividend yield, per 1% change)
            r: Interest rate (annual)
            q: Dividend yield (annual)
            T: Time to maturity in years
            rate_bump: Rate scale of the rho input (default: 1% = 0.01)
            dividend_bump: Dividend scale of the dividend_rho input (default: 1% = 0.01)

        Returns:
            Dictionary with convexity_theta, r_theta, q_theta (all per day)
        """
        if is_zero(T):
            return {
                "convexity_theta": 0.0,
                "r_theta": 0.0,
                "q_theta": 0.0,
            }

        # Rho/Dividend Rho are per rate_bump/dividend_bump size, so divide by scale.
        # Divide by 365 to convert annual rate decay to daily theta equivalent.
        # Divide by T to cancel out the T term in Rho (Rho = dV/dr = T * dV/d(rT) approx).
        r_theta = -r / T * (rho / rate_bump) / 365
        q_theta = -q / T * (dividend_rho / dividend_bump) / 365
        convexity_theta = theta - r_theta - q_theta

        return {
            "convexity_theta": convexity_theta,
            "r_theta": r_theta,
            "q_theta": q_theta,
        }

    def _calculate_numerical_theta_components(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        base_price: Optional[float] = None,
        time_bump_days: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Exact numerical theta decomposition via repricing with zeroed r/q.

        This method computes exact theta components by repricing with different
        rate and dividend yield combinations:
            1. theta_no_rq = theta with r=0, q=0 → convexity_theta
            2. theta_no_q = theta with q=0 → r_theta = theta_no_q - convexity_theta
            3. theta_no_r = theta with r=0 → q_theta = theta_no_r - convexity_theta

        Note: This is computationally expensive (3 extra pricings) and should
        be treated as a slow path. For fast estimation, use
        estimate_theta_components() instead.

        Args:
            product: The derivative product
            pricing_env: Pricing environment
            engine: Pricing engine
            base_price: Pre-calculated base price
            time_bump_days: Time bump in days

        Returns:
            Dictionary with convexity_theta, r_theta, q_theta (all per day)
        """
        from quantark.param.div import ContinuousDividendYield
        from quantark.param.rrf import FlatRateCurve

        time_bump_days = (
            time_bump_days
            if time_bump_days is not None
            else self._bump_config.time_bump_days
        )

        T = product.get_maturity(pricing_env)
        if is_zero(T):
            return {
                "convexity_theta": 0.0,
                "r_theta": 0.0,
                "q_theta": 0.0,
            }

        # Create environments with zeroed r and/or q
        env_no_r = deepcopy(pricing_env)
        env_no_r.rate_curve = FlatRateCurve(0.0)

        env_no_q = deepcopy(pricing_env)
        env_no_q.div_yield = ContinuousDividendYield(0.0)

        env_no_rq = deepcopy(pricing_env)
        env_no_rq.rate_curve = FlatRateCurve(0.0)
        env_no_rq.div_yield = ContinuousDividendYield(0.0)

        # Calculate theta in each environment
        theta_no_rq = self.calculate_numerical_theta(
            product, env_no_rq, engine, time_bump_days=time_bump_days
        )
        theta_no_q = self.calculate_numerical_theta(
            product, env_no_q, engine, time_bump_days=time_bump_days
        )
        theta_no_r = self.calculate_numerical_theta(
            product, env_no_r, engine, time_bump_days=time_bump_days
        )

        # Decompose
        convexity_theta = theta_no_rq
        r_theta = theta_no_q - convexity_theta
        q_theta = theta_no_r - convexity_theta

        return {
            "convexity_theta": convexity_theta,
            "r_theta": r_theta,
            "q_theta": q_theta,
        }

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
        base_price = self._ensure_base_price(product, pricing_env, engine, base_price)
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

    def _build_vol_bumped_env(
        self,
        pricing_env: PricingEnvironment,
        product: BaseEquityProduct,
        current_vol: float,
        vol_bump: float,
        *,
        direction: float,
    ) -> PricingEnvironment:
        from quantark.param.vol import FlatVolSurface, TermStructureVolSurface

        new_vol = current_vol + direction * vol_bump
        if new_vol <= 0:
            raise ValidationError(
                f"Stressed volatility must be positive, got {new_vol}"
            )

        env = deepcopy(pricing_env)
        if isinstance(pricing_env.vol_surface, TermStructureVolSurface):
            new_vols = [float(v) + direction * vol_bump for v in pricing_env.vol_surface.vols]
            if any(v <= 0 for v in new_vols):
                raise ValidationError("Stressed term-structure vol must be positive.")
            env.vol_surface = TermStructureVolSurface(
                times=list(pricing_env.vol_surface.times), vols=new_vols
            )
        else:
            env.vol_surface = FlatVolSurface(new_vol)
        return env

    def _build_div_bumped_env(
        self,
        pricing_env: PricingEnvironment,
        product: BaseEquityProduct,
        current_div: float,
        div_bump: float,
        *,
        direction: float,
    ) -> PricingEnvironment:
        from quantark.param.div import ContinuousDividendYield, TermStructureDividendYield

        new_div = current_div + direction * div_bump
        if new_div < 0:
            raise ValidationError(
                f"Stressed dividend yield cannot be negative, got {new_div}"
            )

        env = deepcopy(pricing_env)
        if isinstance(pricing_env.div_yield, TermStructureDividendYield):
            new_yields = [
                float(y) + direction * div_bump for y in pricing_env.div_yield.yields
            ]
            if any(y < 0 for y in new_yields):
                raise ValidationError(
                    "Stressed term-structure dividend yield cannot be negative."
                )
            env.div_yield = TermStructureDividendYield(
                times=list(pricing_env.div_yield.times), yields=new_yields
            )
        else:
            env.div_yield = ContinuousDividendYield(new_div)
        return env

    def _greeks_for_linear(
        self, product: BaseEquityProduct, price: float
    ) -> Dict[str, float]:
        """
        Calculate Greeks for linear (delta-one) products.

        Delta-one products have trivial Greeks:
        - Delta = 1.0 (always)
        - Gamma, Vega, Theta, Rho, Dividend Rho = 0.0 (no optionality)

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
            "convexity_theta": 0.0,
            "r_theta": 0.0,
            "q_theta": 0.0,
            "rho": 0.0,
            "dividend_rho": 0.0,
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
        multiplier = product.contract_multiplier
        price = product.get_payoff(spot) / multiplier

        # Delta at expiry
        if product.is_call():
            delta = 1.0 if spot > product.strike else 0.0
        else:
            delta = -1.0 if spot < product.strike else 0.0

        return {
            "price": price * multiplier,
            "delta": delta * multiplier,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "convexity_theta": 0.0,
            "r_theta": 0.0,
            "q_theta": 0.0,
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
