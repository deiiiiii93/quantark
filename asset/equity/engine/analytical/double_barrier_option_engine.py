"""
Analytical pricing engine for double-barrier options.

Implements the Ikeda & Kuintomo (1992) infinite-series formula for double
knock-out options with continuous monitoring. Supports:
- Continuous observation (direct closed-form)
- Discrete observation (barrier-shift approximation)
- Expiry-only observation (truncated-domain vanilla payoff)

Knock-in options are priced via parity: knock-in = vanilla - knock-out.
"""

import math
from typing import Optional

from scipy import stats

from asset.equity.engine.base_engine import BaseEngine
from asset.equity.product.base_equity_product import BaseEquityProduct
from asset.equity.product.option import DoubleBarrierOption, EuropeanVanillaOption
from asset.equity.param import EngineParams
from priceenv import PricingEnvironment
from util.barrier_shift import apply_barrier_shift
from util.enum import ObservationType
from util.enum.engine_enums import EngineType
from util.exceptions import ValidationError, PricingError
from util.numerical import (
    is_zero,
    safe_log,
    safe_exp,
    safe_sqrt,
    safe_power,
    validate_positive,
    validate_non_negative,
)

from .black_scholes_engine import BlackScholesEngine


class DoubleBarrierOptionAnalyticalEngine(BaseEngine):
    """
    Closed-form pricing engine for double-barrier options.

    Uses the Ikeda & Kuintomo (1992) infinite-series solution for
    continuous monitoring. Discrete monitoring is approximated via
    Broadie-Glasserman-Kou barrier shift. Expiry observation prices
    the truncated-domain vanilla payoff.

    Edge Cases Handled:
    -------------------
    1. T = 0: Returns intrinsic payoff or rebate immediately.
    2. sigma = 0: Returns discounted deterministic payoff.
    3. Spot outside barriers (KO): Returns rebate immediately.
    4. Spot outside barriers (KI): Returns vanilla price (already knocked in).
    5. Strike outside [L, U]: Raises ValidationError (Ikeda-Kuintomo limitation).
    """

    engine_type = EngineType.ANALYTICAL

    MIN_VOL = 1e-12
    MAX_VOL = 5.0
    MAX_MATURITY = 100.0
    DEFAULT_MAX_TERMS = 10
    SERIES_TOLERANCE = 1e-15

    def __init__(self, params: Optional[EngineParams] = None):
        """
        Initialize the double-barrier analytical engine.

        Args:
            params: Engine configuration parameters.
        """
        super().__init__(params)
        self._bs_engine = BlackScholesEngine(params)

    def price(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        """
        Price a double-barrier option.

        Args:
            product: DoubleBarrierOption to price.
            pricing_env: Pricing environment with market data.

        Returns:
            Option price.

        Raises:
            PricingError: If product is not a DoubleBarrierOption.
            ValidationError: If inputs are invalid.
            NumericalError: If numerical issues occur during calculation.
        """
        if not isinstance(product, DoubleBarrierOption):
            raise PricingError(
                f"DoubleBarrierOptionAnalyticalEngine only supports DoubleBarrierOption, "
                f"got {type(product).__name__}"
            )

        S = pricing_env.spot
        K = product.strike
        T = product.get_maturity(pricing_env)
        r = pricing_env.get_rate(T)
        q = pricing_env.get_div_yield(T)
        sigma = pricing_env.get_vol(K, T)
        L = product.lower_barrier
        U = product.upper_barrier
        multiplier = product.contract_multiplier

        self._validate_inputs(S, K, T, r, q, sigma, L, U)

        # Edge case: zero maturity
        if is_zero(T):
            return self._price_zero_maturity(product, S, multiplier)

        # Edge case: zero volatility
        if is_zero(sigma):
            return self._price_zero_vol(product, S, T, r, q, multiplier)

        # Spot already outside barriers
        if product.is_barrier_hit(S):
            if product.is_knock_out:
                return product.rebate * multiplier
            # Knock-in: already activated, price as vanilla
            vanilla = self._create_vanilla(product)
            return self._bs_engine.price(vanilla, pricing_env) * multiplier

        obs_type = product.observation_type

        if obs_type == ObservationType.EXPIRY:
            return self._price_expiry(product, pricing_env, S, K, T, r, q, sigma, L, U, multiplier)

        if obs_type == ObservationType.DISCRETE:
            return self._price_discrete(product, pricing_env, S, K, T, r, q, sigma, L, U, multiplier)

        if obs_type == ObservationType.CONTINUOUS:
            return self._price_continuous(product, pricing_env, S, K, T, r, q, sigma, L, U, multiplier)

        raise PricingError(f"Unsupported observation type: {obs_type}")

    def _price_zero_maturity(
        self, product: DoubleBarrierOption, spot: float, multiplier: float
    ) -> float:
        """Return price at maturity."""
        if product.is_knock_out:
            if product.is_barrier_hit(spot):
                return product.rebate * multiplier
            return product.get_payoff(spot)
        # Knock-in
        if product.is_barrier_hit(spot):
            return product.get_payoff(spot)
        return product.rebate * multiplier

    def _price_zero_vol(
        self,
        product: DoubleBarrierOption,
        spot: float,
        T: float,
        r: float,
        q: float,
        multiplier: float,
    ) -> float:
        """Price when volatility is zero (deterministic path)."""
        forward = float(spot * safe_exp((r - q) * T))
        is_outside = product.is_barrier_hit(forward)
        discount = float(safe_exp(-r * T))

        if product.is_knock_out:
            if is_outside:
                return product.rebate * multiplier
            payoff = product.get_payoff(forward) / multiplier  # get_payoff already applies multiplier
            return payoff * discount

        # Knock-in
        if is_outside:
            payoff = product.get_payoff(forward) / multiplier
            return payoff * discount
        return product.rebate * multiplier

    def _price_continuous(
        self,
        product: DoubleBarrierOption,
        pricing_env: PricingEnvironment,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        L: float,
        U: float,
        multiplier: float,
        delta1: float = 0.0,
        delta2: float = 0.0,
    ) -> float:
        """Price using Ikeda & Kuintomo continuous formula."""
        b = r - q
        ko_price = self._price_knock_out_ikeda_kuintomo(
            S=S,
            K=K,
            L=L,
            U=U,
            T=T,
            r=r,
            b=b,
            sigma=sigma,
            is_call=product.is_call(),
            delta1=delta1,
            delta2=delta2,
        )

        if product.is_knock_out:
            price = max(ko_price, 0.0) + product.rebate * float(safe_exp(-r * T))
            return price * multiplier

        # Knock-in via parity
        vanilla = self._create_vanilla(product)
        vanilla_price = self._bs_engine.price(vanilla, pricing_env)
        ki_price = max(vanilla_price - max(ko_price, 0.0), 0.0)
        prob_inside = float(self._prob_inside_at_expiry(S, T, r, q, sigma, L, U))
        price = ki_price + product.rebate * float(safe_exp(-r * T)) * prob_inside
        return float(price * multiplier)

    def _price_discrete(
        self,
        product: DoubleBarrierOption,
        pricing_env: PricingEnvironment,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        L: float,
        U: float,
        multiplier: float,
    ) -> float:
        """Price discrete observation using barrier shift."""
        schedule = product.observation_schedule
        if schedule is None or not schedule.records:
            raise PricingError(
                "Discrete barrier monitoring requires ObservationSchedule."
            )
        schedule.assert_analytical_ready(default_payoff=product.rebate)
        freq = schedule.ensure_regular_frequency(schedule.times)

        lower_shifted = apply_barrier_shift(
            barrier=L,
            is_up_barrier=False,
            volatility=sigma,
            observation_interval=freq,
        )
        upper_shifted = apply_barrier_shift(
            barrier=U,
            is_up_barrier=True,
            volatility=sigma,
            observation_interval=freq,
        )

        return self._price_continuous(
            product=product,
            pricing_env=pricing_env,
            S=S,
            K=K,
            T=T,
            r=r,
            q=q,
            sigma=sigma,
            L=lower_shifted,
            U=upper_shifted,
            multiplier=multiplier,
        )

    def _price_expiry(
        self,
        product: DoubleBarrierOption,
        pricing_env: PricingEnvironment,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        L: float,
        U: float,
        multiplier: float,
    ) -> float:
        """Price expiry-observed double barrier using truncated-domain vanilla."""
        vanilla = self._create_vanilla(product)
        vanilla_price = self._bs_engine.price(vanilla, pricing_env)

        sqrt_t = safe_sqrt(T)
        drift_adj = (r - q - 0.5 * sigma * sigma) * T
        d2_u = (safe_log(S / U) + drift_adj) / (sigma * sqrt_t)
        d2_l = (safe_log(S / L) + drift_adj) / (sigma * sqrt_t)

        discount = float(safe_exp(-r * T))
        prob_outside = float(stats.norm.cdf(float(d2_u)) + stats.norm.cdf(-float(d2_l)))
        prob_inside = float(stats.norm.cdf(float(d2_l)) - stats.norm.cdf(float(d2_u)))

        if product.is_call():
            # Truncated call payoff = Call(K) - Call(U) - (U-K) * digital(U)
            d1_u = float(d2_u) + sigma * float(sqrt_t)
            call_u = self._bs_engine._price_call(S, U, T, r, q, d1_u, float(d2_u))
            truncated = max(vanilla_price - call_u - (U - K) * discount * stats.norm.cdf(float(d2_u)), 0.0)
        else:
            # Truncated put payoff = Put(K) - Put(L) - (K-L) * digital(L)
            d1_l = float(d2_l) + sigma * float(sqrt_t)
            put_l = self._bs_engine._price_put(S, L, T, r, q, d1_l, float(d2_l))
            truncated = max(vanilla_price - put_l - (K - L) * discount * stats.norm.cdf(-float(d2_l)), 0.0)

        if product.is_knock_out:
            price = truncated + product.rebate * discount * prob_outside
        else:
            price = max(vanilla_price - truncated, 0.0) + product.rebate * discount * prob_inside

        return float(price * multiplier)

    def _price_knock_out_ikeda_kuintomo(
        self,
        S: float,
        K: float,
        L: float,
        U: float,
        T: float,
        r: float,
        b: float,
        sigma: float,
        is_call: bool,
        delta1: float = 0.0,
        delta2: float = 0.0,
        max_terms: int = DEFAULT_MAX_TERMS,
    ) -> float:
        """
        Ikeda & Kuintomo (1992) infinite-series formula for double knock-out.

        Returns the unadjusted theoretical price (per unit, no multiplier).
        """
        F = float(U * float(safe_exp(delta1 * T)))
        E = float(L * float(safe_exp(delta2 * T)))

        if is_call:
            arg1 = K  # for d1, d3
            arg2 = F  # for d2, d4
        else:
            arg1 = E  # for y1, y3
            arg2 = K  # for y2, y4

        sqrt_t = float(safe_sqrt(T))
        denom = sigma * sqrt_t
        drift = float((b + 0.5 * sigma * sigma) * T)
        sig2 = sigma * sigma

        asset_sum = 0.0
        strike_sum = 0.0

        for n in range(-max_terms, max_terms + 1):
            U_pow = math.pow(U, n)
            L_pow = math.pow(L, n)
            U2n = U_pow * U_pow
            L2n = L_pow * L_pow
            L2n2 = L2n * L * L

            mu1 = 2.0 * (b - delta2 - n * (delta1 - delta2)) / sig2 + 1.0
            mu2 = 2.0 * n * (delta1 - delta2) / sig2
            mu3 = 2.0 * (b - delta2 + n * (delta1 - delta2)) / sig2 + 1.0

            w1 = float(safe_power((U_pow / L_pow), mu1) * safe_power((L / S), mu2))
            w2 = float(safe_power((math.pow(L, n + 1) / (U_pow * S)), mu3))

            d_a = float((float(safe_log(S * U2n / (arg1 * L2n))) + drift) / denom)
            d_b = float((float(safe_log(S * U2n / (arg2 * L2n))) + drift) / denom)
            d_c = float((float(safe_log(L2n2 / (arg1 * S * U2n))) + drift) / denom)
            d_d = float((float(safe_log(L2n2 / (arg2 * S * U2n))) + drift) / denom)

            cdf_ab = stats.norm.cdf(d_a) - stats.norm.cdf(d_b)
            cdf_cd = stats.norm.cdf(d_c) - stats.norm.cdf(d_d)

            term_asset = 0.0
            if math.isfinite(w1) and math.isfinite(w2):
                term_asset = w1 * cdf_ab - w2 * cdf_cd
            elif math.isfinite(w1):
                term_asset = w1 * cdf_ab
            elif math.isfinite(w2):
                term_asset = -w2 * cdf_cd
            asset_sum += term_asset

            w1_strike = float(safe_power((U_pow / L_pow), mu1 - 2.0) * safe_power((L / S), mu2))
            w2_strike = float(safe_power((math.pow(L, n + 1) / (U_pow * S)), mu3 - 2.0))

            cdf_ab_strike = stats.norm.cdf(d_a - denom) - stats.norm.cdf(d_b - denom)
            cdf_cd_strike = stats.norm.cdf(d_c - denom) - stats.norm.cdf(d_d - denom)

            term_strike = 0.0
            if math.isfinite(w1_strike) and math.isfinite(w2_strike):
                term_strike = w1_strike * cdf_ab_strike - w2_strike * cdf_cd_strike
            elif math.isfinite(w1_strike):
                term_strike = w1_strike * cdf_ab_strike
            elif math.isfinite(w2_strike):
                term_strike = -w2_strike * cdf_cd_strike
            strike_sum += term_strike

        df_carry = float(safe_exp((b - r) * T))
        df_riskfree = float(safe_exp(-r * T))

        if is_call:
            price = S * df_carry * asset_sum - K * df_riskfree * strike_sum
        else:
            price = K * df_riskfree * strike_sum - S * df_carry * asset_sum

        return float(price)

    def _prob_inside_at_expiry(
        self, S: float, T: float, r: float, q: float, sigma: float, L: float, U: float
    ) -> float:
        """Risk-neutral probability that spot is inside [L, U] at expiry."""
        sqrt_t = float(safe_sqrt(T))
        drift_adj = float((r - q - 0.5 * sigma * sigma) * T)
        d2_u = float((float(safe_log(S / U)) + drift_adj) / (sigma * sqrt_t))
        d2_l = float((float(safe_log(S / L)) + drift_adj) / (sigma * sqrt_t))
        return float(stats.norm.cdf(d2_l) - stats.norm.cdf(d2_u))

    def _create_vanilla(self, product: DoubleBarrierOption) -> EuropeanVanillaOption:
        """Create a vanilla option with same terms as the double barrier."""
        return EuropeanVanillaOption(
            strike=product.strike,
            option_type=product.option_type,
            maturity=product.maturity,
            exercise_date=product.exercise_date,
            settlement_date=product.settlement_date,
        )

    def _validate_inputs(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        L: float,
        U: float,
    ) -> None:
        """Comprehensive input validation."""
        validate_positive(S, "spot")
        validate_positive(K, "strike")
        validate_non_negative(T, "maturity")
        validate_positive(sigma, "volatility")
        validate_positive(L, "lower_barrier")
        validate_positive(U, "upper_barrier")

        if L >= U:
            raise ValidationError(
                f"Lower barrier ({L}) must be less than upper barrier ({U})"
            )

        if not (L < K < U):
            raise ValidationError(
                f"Strike ({K}) must be strictly between lower ({L}) and upper ({U}) "
                f"barriers for the Ikeda-Kuintomo formula."
            )

        if sigma < self.MIN_VOL or sigma > self.MAX_VOL:
            raise ValidationError(
                f"Volatility {sigma} outside supported range "
                f"[{self.MIN_VOL}, {self.MAX_VOL}]"
            )

        if T > self.MAX_MATURITY:
            raise ValidationError(
                f"Maturity {T} years exceeds maximum {self.MAX_MATURITY}"
            )

        if abs(r) > 1.0:
            raise ValidationError(f"Risk-free rate outside reasonable bounds: {r}")

        if q < 0:
            raise ValidationError(f"Dividend yield must be non-negative, got {q}")

    def __repr__(self):
        return "DoubleBarrierOptionAnalyticalEngine()"
