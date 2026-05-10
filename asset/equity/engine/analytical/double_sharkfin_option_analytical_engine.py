"""
Analytical pricing engine for double sharkfin options.

The payoff is decomposed into a no-rebate double knock-out option, a
double-touch cash leg for the knock-out rebate, and a double-no-touch cash leg
for the no-hit rebate.
"""

from typing import Optional

import numpy as np
from scipy import stats

from asset.equity.engine.base_engine import BaseEngine
from asset.equity.product.base_equity_product import BaseEquityProduct
from asset.equity.product.option import DoubleBarrierOption, DoubleSharkfinOption
from asset.equity.param import EngineParams
from priceenv import PricingEnvironment
from util.barrier_shift import apply_barrier_shift
from util.enum import DoubleBarrierType, ObservationType
from util.enum.engine_enums import EngineType
from util.exceptions import PricingError, ValidationError
from util.numerical import (
    is_zero,
    safe_exp,
    safe_log,
    safe_sqrt,
    validate_non_negative,
    validate_positive,
)

from .double_barrier_option_engine import DoubleBarrierOptionAnalyticalEngine


class DoubleSharkfinOptionAnalyticalEngine(BaseEngine):
    """
    Semi-closed-form analytical engine for DoubleSharkfinOption.

    Expiry-only monitoring is priced from truncated lognormal probabilities.
    Continuous monitoring uses the Ikeda-Kunitomo double-barrier option engine
    for the participation leg and a killed log-price density series for cash
    survival legs. Discrete monitoring applies a BGK barrier shift.
    """

    engine_type = EngineType.ANALYTICAL

    MIN_MATURITY = 1e-10
    MIN_VOL = 0.001
    MAX_VOL = 5.0
    MAX_MATURITY = 50.0
    DEFAULT_MAX_TERMS = 80
    DEFAULT_QUAD_POINTS = 48

    def __init__(
        self,
        params: Optional[EngineParams] = None,
        max_terms: int = DEFAULT_MAX_TERMS,
        quad_points: int = DEFAULT_QUAD_POINTS,
    ):
        """
        Initialize double sharkfin analytical engine.

        Args:
            params: Engine configuration parameters.
            max_terms: Terms in the double-barrier survival series.
            quad_points: Gauss-Legendre nodes for pay-at-hit integration.

        Raises:
            ValidationError: If engine controls are invalid.
        """
        super().__init__(params)
        if max_terms <= 0:
            raise ValidationError(f"max_terms must be positive, got {max_terms}")
        if quad_points <= 0:
            raise ValidationError(f"quad_points must be positive, got {quad_points}")

        self.max_terms = int(max_terms)
        self.quad_points = int(quad_points)
        self._double_barrier_engine = DoubleBarrierOptionAnalyticalEngine(params)

    def price(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        """
        Price a double sharkfin option analytically.

        Args:
            product: DoubleSharkfinOption instance.
            pricing_env: Market data environment.

        Returns:
            Present value scaled by product.contract_multiplier.
        """
        if not isinstance(product, DoubleSharkfinOption):
            raise PricingError(
                "DoubleSharkfinOptionAnalyticalEngine only supports "
                f"DoubleSharkfinOption, got {type(product).__name__}"
            )

        spot = pricing_env.spot
        maturity = product.get_maturity(pricing_env)
        rate = pricing_env.get_rate(maturity)
        div = pricing_env.get_div_yield(maturity)
        vol = pricing_env.get_vol(product.strike, maturity)

        self._validate_inputs(
            spot=spot,
            strike=product.strike,
            lower_barrier=product.lower_barrier,
            upper_barrier=product.upper_barrier,
            maturity=maturity,
            rate=rate,
            div=div,
            vol=vol,
            participation_rate=product.participation_rate,
            knock_out_rebate=product.knock_out_rebate,
            no_hit_rebate=product.no_hit_rebate,
            contract_multiplier=product.contract_multiplier,
        )

        if maturity < self.MIN_MATURITY:
            return product.get_payoff(spot)

        if product.observation_type not in (
            ObservationType.EXPIRY,
            ObservationType.CONTINUOUS,
            ObservationType.DISCRETE,
        ):
            raise PricingError(
                f"Unsupported observation type: {product.observation_type}"
            )

        if product.observation_type != ObservationType.EXPIRY and product.is_barrier_hit(spot):
            return self._price_already_hit(product, rate, maturity)

        option_leg = self._price_no_rebate_knock_out(product, pricing_env)
        survival_prob = self._survival_probability(
            product, maturity, rate, div, vol, pricing_env
        )
        knock_out_cash = self._price_knock_out_cash_leg(
            product=product,
            maturity=maturity,
            rate=rate,
            div=div,
            vol=vol,
            pricing_env=pricing_env,
            survival_prob=survival_prob,
        )
        no_hit_cash = product.no_hit_rebate * safe_exp(-rate * maturity) * survival_prob

        value = product.participation_rate * option_leg + knock_out_cash + no_hit_cash
        return max(float(value), 0.0) * product.contract_multiplier

    def _price_already_hit(
        self, product: DoubleSharkfinOption, rate: float, maturity: float
    ) -> float:
        """Return the value when a monitored product is already hit."""
        if product.pay_at_hit:
            return product.knock_out_rebate * product.contract_multiplier
        return (
            product.knock_out_rebate
            * safe_exp(-rate * maturity)
            * product.contract_multiplier
        )

    def _price_no_rebate_knock_out(
        self, product: DoubleSharkfinOption, pricing_env: PricingEnvironment
    ) -> float:
        """Value the capped participation leg as a double knock-out option."""
        double_barrier = DoubleBarrierOption(
            strike=product.strike,
            option_type=product.option_type,
            upper_barrier=product.upper_barrier,
            lower_barrier=product.lower_barrier,
            barrier_type=DoubleBarrierType.KNOCK_OUT,
            maturity=product.maturity,
            exercise_date=product.exercise_date,
            settlement_date=product.settlement_date,
            rebate=0.0,
            observation_type=product.observation_type,
            observation_dates=product.observation_dates,
            observation_schedule=product.observation_schedule,
            contract_multiplier=1.0,
        )
        return self._double_barrier_engine.price(double_barrier, pricing_env)

    def _survival_probability(
        self,
        product: DoubleSharkfinOption,
        maturity: float,
        rate: float,
        div: float,
        vol: float,
        pricing_env: PricingEnvironment,
    ) -> float:
        """Probability that neither double sharkfin barrier is hit."""
        if product.observation_type == ObservationType.EXPIRY:
            prob = self._prob_inside_at_expiry(
                spot=pricing_env.spot,
                maturity=maturity,
                rate=rate,
                div=div,
                vol=vol,
                lower_barrier=product.lower_barrier,
                upper_barrier=product.upper_barrier,
            )
        elif product.observation_type == ObservationType.CONTINUOUS:
            prob = self._survival_probability_continuous(
                spot=pricing_env.spot,
                maturity=maturity,
                rate=rate,
                div=div,
                vol=vol,
                lower_barrier=product.lower_barrier,
                upper_barrier=product.upper_barrier,
            )
        elif product.observation_type == ObservationType.DISCRETE:
            lower_barrier, upper_barrier = self._shift_discrete_barriers(
                product, vol
            )
            prob = self._survival_probability_continuous(
                spot=pricing_env.spot,
                maturity=maturity,
                rate=rate,
                div=div,
                vol=vol,
                lower_barrier=lower_barrier,
                upper_barrier=upper_barrier,
            )
        else:
            raise PricingError(
                f"Unsupported observation type: {product.observation_type}"
            )

        return min(max(float(prob), 0.0), 1.0)

    def _price_knock_out_cash_leg(
        self,
        product: DoubleSharkfinOption,
        maturity: float,
        rate: float,
        div: float,
        vol: float,
        pricing_env: PricingEnvironment,
        survival_prob: float,
    ) -> float:
        """Value the fixed cash leg paid when either barrier is hit."""
        if product.knock_out_rebate <= 0.0:
            return 0.0

        if product.observation_type == ObservationType.EXPIRY or not product.pay_at_hit:
            return product.knock_out_rebate * safe_exp(-rate * maturity) * (
                1.0 - survival_prob
            )

        if product.observation_type == ObservationType.DISCRETE:
            hit_discount_factor = self._discrete_hit_discount_factor(
                product=product,
                maturity=maturity,
                rate=rate,
                div=div,
                vol=vol,
                spot=pricing_env.spot,
            )
        else:
            hit_discount_factor = self._continuous_hit_discount_factor(
                spot=pricing_env.spot,
                maturity=maturity,
                rate=rate,
                div=div,
                vol=vol,
                lower_barrier=product.lower_barrier,
                upper_barrier=product.upper_barrier,
                survival_at_maturity=survival_prob,
            )

        return product.knock_out_rebate * max(float(hit_discount_factor), 0.0)

    def _discrete_hit_discount_factor(
        self,
        product: DoubleSharkfinOption,
        maturity: float,
        rate: float,
        div: float,
        vol: float,
        spot: float,
    ) -> float:
        """Approximate discounted first-hit probability on a discrete grid."""
        lower_barrier, upper_barrier = self._shift_discrete_barriers(product, vol)
        times = product.get_observation_times()
        if not times:
            raise PricingError(
                "Discrete double sharkfin monitoring requires observation times."
            )

        previous_survival = 1.0
        hit_discount_factor = 0.0
        for observation_time in times:
            current_time = min(float(observation_time), maturity)
            current_survival = self._survival_probability_continuous(
                spot=spot,
                maturity=current_time,
                rate=rate,
                div=div,
                vol=vol,
                lower_barrier=lower_barrier,
                upper_barrier=upper_barrier,
            )
            first_hit_prob = max(previous_survival - current_survival, 0.0)
            hit_discount_factor += first_hit_prob * safe_exp(-rate * current_time)
            previous_survival = min(max(current_survival, 0.0), previous_survival)

        return hit_discount_factor

    def _continuous_hit_discount_factor(
        self,
        spot: float,
        maturity: float,
        rate: float,
        div: float,
        vol: float,
        lower_barrier: float,
        upper_barrier: float,
        survival_at_maturity: float,
    ) -> float:
        """
        Compute E[exp(-r tau) 1_{tau<=T}] from the survival curve.
        """
        if is_zero(rate):
            return 1.0 - survival_at_maturity

        nodes, weights = np.polynomial.legendre.leggauss(self.quad_points)
        times = 0.5 * maturity * (nodes + 1.0)
        integral = 0.0
        for time_value, weight in zip(times, weights):
            survival = self._survival_probability_continuous(
                spot=spot,
                maturity=float(time_value),
                rate=rate,
                div=div,
                vol=vol,
                lower_barrier=lower_barrier,
                upper_barrier=upper_barrier,
            )
            integral += float(weight) * safe_exp(-rate * float(time_value)) * survival
        integral *= 0.5 * maturity

        hit_factor = (
            1.0
            - safe_exp(-rate * maturity) * survival_at_maturity
            - rate * integral
        )
        return max(float(hit_factor), 0.0)

    def _shift_discrete_barriers(
        self, product: DoubleSharkfinOption, vol: float
    ) -> tuple[float, float]:
        """Apply BGK shifts to discrete lower and upper barriers."""
        schedule = product.observation_schedule
        if schedule is None or not schedule.records:
            raise PricingError(
                "Discrete double sharkfin monitoring requires ObservationSchedule."
            )
        schedule.assert_analytical_ready(
            default_payoff=product.knock_out_rebate,
            business_days_in_year=product.business_days_in_year,
        )
        frequency = schedule.ensure_regular_frequency(
            schedule.times,
            business_days_in_year=product.business_days_in_year,
        )
        lower_barrier = apply_barrier_shift(
            barrier=product.lower_barrier,
            is_up_barrier=False,
            volatility=vol,
            observation_interval=frequency,
        )
        upper_barrier = apply_barrier_shift(
            barrier=product.upper_barrier,
            is_up_barrier=True,
            volatility=vol,
            observation_interval=frequency,
        )
        if not (lower_barrier < product.strike < upper_barrier):
            raise PricingError(
                "Discrete barrier shift produced barriers that do not contain "
                f"strike: lower={lower_barrier}, strike={product.strike}, "
                f"upper={upper_barrier}"
            )
        return lower_barrier, upper_barrier

    def _prob_inside_at_expiry(
        self,
        spot: float,
        maturity: float,
        rate: float,
        div: float,
        vol: float,
        lower_barrier: float,
        upper_barrier: float,
    ) -> float:
        """Risk-neutral probability that terminal spot is inside (L, U)."""
        sqrt_t = safe_sqrt(maturity)
        drift_adj = (rate - div - 0.5 * vol * vol) * maturity
        d2_upper = (safe_log(spot / upper_barrier) + drift_adj) / (vol * sqrt_t)
        d2_lower = (safe_log(spot / lower_barrier) + drift_adj) / (vol * sqrt_t)
        return float(stats.norm.cdf(float(d2_lower)) - stats.norm.cdf(float(d2_upper)))

    def _survival_probability_continuous(
        self,
        spot: float,
        maturity: float,
        rate: float,
        div: float,
        vol: float,
        lower_barrier: float,
        upper_barrier: float,
    ) -> float:
        """Double-barrier no-touch probability from killed log-price density."""
        if maturity <= self.MIN_MATURITY:
            return 0.0 if spot <= lower_barrier or spot >= upper_barrier else 1.0
        if spot <= lower_barrier or spot >= upper_barrier:
            return 0.0

        x0 = safe_log(spot)
        lower = safe_log(lower_barrier)
        upper = safe_log(upper_barrier)
        width = upper - lower
        drift = rate - div - 0.5 * vol * vol
        alpha = drift / (vol * vol)
        alpha_shift = alpha * (lower - x0)
        decay_base = -drift * drift * maturity / (2.0 * vol * vol)
        z0 = x0 - lower

        total = 0.0
        for n in range(1, self.max_terms + 1):
            lambda_n = n * np.pi / width
            eigen_decay = safe_exp(-0.5 * vol * vol * lambda_n * lambda_n * maturity)
            integral = self._weighted_sine_integral(alpha, lambda_n, width)
            total += float(np.sin(lambda_n * z0) * eigen_decay * integral)

        survival = safe_exp(decay_base) * (2.0 / width) * safe_exp(alpha_shift) * total
        return min(max(float(survival), 0.0), 1.0)

    def _weighted_sine_integral(
        self, alpha: float, lambda_n: float, width: float
    ) -> float:
        """Return int_0^width exp(alpha z) sin(lambda_n z) dz."""
        denominator = alpha * alpha + lambda_n * lambda_n
        return float(
            lambda_n
            * (1.0 - safe_exp(alpha * width) * np.cos(lambda_n * width))
            / denominator
        )

    def _validate_inputs(
        self,
        spot: float,
        strike: float,
        lower_barrier: float,
        upper_barrier: float,
        maturity: float,
        rate: float,
        div: float,
        vol: float,
        participation_rate: float,
        knock_out_rebate: float,
        no_hit_rebate: float,
        contract_multiplier: float,
    ) -> None:
        """Validate market and product inputs for analytical pricing."""
        validate_positive(spot, "spot")
        validate_positive(strike, "strike")
        validate_positive(lower_barrier, "lower_barrier")
        validate_positive(upper_barrier, "upper_barrier")
        validate_non_negative(maturity, "maturity")
        validate_positive(vol, "volatility")
        validate_non_negative(participation_rate, "participation_rate")
        validate_non_negative(knock_out_rebate, "knock_out_rebate")
        validate_non_negative(no_hit_rebate, "no_hit_rebate")
        validate_positive(contract_multiplier, "contract_multiplier")

        if lower_barrier >= upper_barrier:
            raise ValidationError(
                f"Lower barrier ({lower_barrier}) must be less than upper barrier "
                f"({upper_barrier})"
            )
        if not (lower_barrier < strike < upper_barrier):
            raise ValidationError(
                f"Strike ({strike}) must be strictly between lower "
                f"({lower_barrier}) and upper ({upper_barrier}) barriers."
            )
        if vol < self.MIN_VOL or vol > self.MAX_VOL:
            raise ValidationError(
                f"Volatility {vol} outside supported range "
                f"[{self.MIN_VOL}, {self.MAX_VOL}]"
            )
        if maturity > self.MAX_MATURITY:
            raise ValidationError(
                f"Maturity too long for analytical double sharkfin pricing: {maturity}"
            )
        if div < 0:
            raise ValidationError(f"Dividend yield must be non-negative, got {div}")
        if abs(rate) > 1.0:
            raise ValidationError(f"Risk-free rate outside reasonable bounds: {rate}")

    def __repr__(self):
        return "DoubleSharkfinOptionAnalyticalEngine()"
