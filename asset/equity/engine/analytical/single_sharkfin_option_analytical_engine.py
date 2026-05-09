"""
Analytical pricing engine for single sharkfin options.

The payoff is decomposed into:
- a no-rebate knock-out vanilla option,
- a one-touch cash leg for the knock-out rebate, and
- a no-touch cash leg for the no-hit rebate.

Discrete monitoring is handled by the underlying barrier and one-touch
analytical engines using the Broadie-Glasserman-Kou barrier shift.
"""

from typing import Optional

from asset.equity.engine.base_engine import BaseEngine
from asset.equity.product.base_equity_product import BaseEquityProduct
from asset.equity.product.option import (
    BarrierOption,
    OneTouchOption,
    SingleSharkfinOption,
)
from asset.equity.param import EngineParams
from priceenv import PricingEnvironment
from util.enum import BarrierDirection, BarrierType, ObservationType, TouchType
from util.enum.engine_enums import EngineType
from util.exceptions import PricingError, ValidationError
from util.numerical import validate_non_negative, validate_positive

from .barrier_analytical_engine import BarrierAnalyticalEngine
from .one_touch_analytical_engine import OneTouchAnalyticalEngine


class SingleSharkfinOptionAnalyticalEngine(BaseEngine):
    """
    Closed-form analytical engine for SingleSharkfinOption.

    Supports expiry-only and continuous monitoring exactly under Black-Scholes
    assumptions. Discrete monitoring uses the standard BGK continuity correction
    implemented by the composed barrier and one-touch engines, so daily
    observation schedules are approximated by shifting the barrier away from spot.
    """

    engine_type = EngineType.ANALYTICAL

    MIN_MATURITY = 1e-10
    MAX_MATURITY = 50.0
    MIN_VOL = 0.001
    MAX_VOL = 5.0

    def __init__(self, params: Optional[EngineParams] = None):
        super().__init__(params)
        self._barrier_engine = BarrierAnalyticalEngine(params)
        self._one_touch_engine = OneTouchAnalyticalEngine(params)

    def price(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        """
        Price a single sharkfin option analytically.

        Args:
            product: SingleSharkfinOption instance.
            pricing_env: Market data environment.

        Returns:
            Present value scaled by product.contract_multiplier.

        Raises:
            PricingError: If product type or observation mode is unsupported.
            ValidationError: If pricing inputs are invalid.
        """
        if not isinstance(product, SingleSharkfinOption):
            raise PricingError(
                "SingleSharkfinOptionAnalyticalEngine only supports "
                f"SingleSharkfinOption, got {type(product).__name__}"
            )

        spot = pricing_env.spot
        maturity = product.get_maturity(pricing_env)
        rate = pricing_env.get_rate(maturity)
        div = pricing_env.get_div_yield(maturity)
        vol = pricing_env.get_vol(product.strike, maturity)

        self._validate_inputs(
            spot=spot,
            strike=product.strike,
            barrier=product.barrier,
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

        ko_option_value = self._price_no_rebate_knock_out(product, pricing_env)
        knock_out_rebate_value = self._price_touch_leg(
            product=product,
            pricing_env=pricing_env,
            rebate=product.knock_out_rebate,
            touch_type=TouchType.ONE_TOUCH,
        )
        no_hit_rebate_value = self._price_touch_leg(
            product=product,
            pricing_env=pricing_env,
            rebate=product.no_hit_rebate,
            touch_type=TouchType.NO_TOUCH,
        )

        value = (
            product.participation_rate * ko_option_value
            + knock_out_rebate_value
            + no_hit_rebate_value
        )
        return max(value, 0.0) * product.contract_multiplier

    def _price_no_rebate_knock_out(
        self, product: SingleSharkfinOption, pricing_env: PricingEnvironment
    ) -> float:
        """Value the capped sharkfin participation leg as a knock-out option."""
        barrier_option = BarrierOption(
            strike=product.strike,
            option_type=product.option_type,
            barrier=product.barrier,
            barrier_type=self._barrier_type(product),
            maturity=product.maturity,
            exercise_date=product.exercise_date,
            settlement_date=product.settlement_date,
            rebate=0.0,
            participation_rate=1.0,
            pay_at_hit=False,
            observation_type=product.observation_type,
            observation_dates=product.observation_dates,
            observation_schedule=product.observation_schedule,
            contract_multiplier=1.0,
        )
        return self._barrier_engine.price(barrier_option, pricing_env)

    def _price_touch_leg(
        self,
        product: SingleSharkfinOption,
        pricing_env: PricingEnvironment,
        rebate: float,
        touch_type: TouchType,
    ) -> float:
        """Value a fixed cash leg conditional on touch or no-touch."""
        if rebate <= 0.0:
            return 0.0

        touch_option = OneTouchOption(
            barrier=product.barrier,
            barrier_direction=self._barrier_direction(product),
            maturity=product.maturity,
            exercise_date=product.exercise_date,
            settlement_date=product.settlement_date,
            rebate=rebate,
            payment_at_hit=(
                product.pay_at_hit if touch_type == TouchType.ONE_TOUCH else False
            ),
            touch_type=touch_type,
            observation_type=product.observation_type,
            observation_dates=product.observation_dates,
            observation_schedule=product.observation_schedule,
        )
        return self._one_touch_engine.price(touch_option, pricing_env)

    def _barrier_type(self, product: SingleSharkfinOption) -> BarrierType:
        """Map sharkfin orientation to a knock-out barrier type."""
        return BarrierType.UP_OUT if product.is_call() else BarrierType.DOWN_OUT

    def _barrier_direction(self, product: SingleSharkfinOption) -> BarrierDirection:
        """Map sharkfin orientation to a one-touch barrier direction."""
        return BarrierDirection.UP if product.is_call() else BarrierDirection.DOWN

    def _validate_inputs(
        self,
        spot: float,
        strike: float,
        barrier: float,
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
        validate_positive(barrier, "barrier")
        validate_non_negative(maturity, "maturity")
        validate_positive(vol, "volatility")
        validate_non_negative(participation_rate, "participation_rate")
        validate_non_negative(knock_out_rebate, "knock_out_rebate")
        validate_non_negative(no_hit_rebate, "no_hit_rebate")
        validate_positive(contract_multiplier, "contract_multiplier")

        if vol < self.MIN_VOL or vol > self.MAX_VOL:
            raise ValidationError(
                f"Volatility {vol} outside supported range "
                f"[{self.MIN_VOL}, {self.MAX_VOL}]"
            )
        if maturity > self.MAX_MATURITY:
            raise ValidationError(
                f"Maturity too long for analytical sharkfin pricing: {maturity}"
            )
        if div < 0:
            raise ValidationError(f"Dividend yield must be non-negative, got {div}")
        if abs(rate) > 1.0:
            raise ValidationError(f"Risk-free rate outside reasonable bounds: {rate}")

    def __repr__(self):
        return "SingleSharkfinOptionAnalyticalEngine()"
