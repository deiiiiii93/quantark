"""
Analytical pricing engine for single-barrier options.

Supports:
- Continuous monitoring (closed-form barrier formulas)
- Discrete monitoring with barrier shift (requires regular schedule)
- Expiry-only monitoring via vanilla/digital decomposition
"""

import math
from typing import Optional

from scipy import stats

from asset.equity.engine.base_engine import BaseEngine
from asset.equity.product.base_equity_product import BaseEquityProduct
from asset.equity.product.option import (
    BarrierOption,
    EuropeanVanillaOption,
    OneTouchOption,
)
from asset.equity.param import EngineParams
from priceenv import PricingEnvironment
from util.barrier_shift import apply_barrier_shift
from util.enum import ObservationType, BarrierDirection, TouchType
from util.enum.engine_enums import EngineType
from util.exceptions import ValidationError, PricingError

from .black_scholes_engine import BlackScholesEngine
from .one_touch_analytical_engine import OneTouchAnalyticalEngine


class BarrierAnalyticalEngine(BaseEngine):
    """
    Closed-form pricing engine for single-barrier options.

    Notes:
        - Discrete monitoring uses Broadie-Glasserman-Kou barrier shift assuming
          a regular observation grid and fixed payoff across observations.
        - Rebates for continuous/discrete monitoring are valued via OneTouchAnalyticalEngine.
    """

    engine_type = EngineType.ANALYTICAL

    MIN_VOL = 0.001
    MAX_VOL = 5.0
    MIN_MATURITY = 1e-10
    MAX_MATURITY = 50.0

    def __init__(self, params: Optional[EngineParams] = None):
        super().__init__(params)
        self._bs_engine = BlackScholesEngine(params)
        self._one_touch_engine = OneTouchAnalyticalEngine(params)

    def price(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        if not isinstance(product, BarrierOption):
            raise PricingError(
                f"BarrierAnalyticalEngine only supports BarrierOption, "
                f"got {type(product).__name__}"
            )

        spot = pricing_env.spot
        strike = product.strike
        maturity = product.get_maturity(pricing_env)
        rate = pricing_env.get_rate(maturity)
        div = pricing_env.get_div_yield(maturity)
        vol = pricing_env.get_vol(strike, maturity)
        participation = product.participation_rate
        multiplier = product.contract_multiplier

        self._validate_inputs(spot, strike, maturity, rate, div, vol, product.barrier)

        # Immediate handling for zero maturity
        if maturity < self.MIN_MATURITY:
            if product.is_knock_out and product.is_barrier_hit(spot):
                return product.rebate * multiplier
            if product.is_knock_in and product.is_barrier_hit(spot):
                return product.get_payoff(spot) * participation
            if product.is_knock_in:
                return 0.0
            return product.get_payoff(spot) * participation

        obs_type = product.observation_type

        # Knock-out already hit under continuous monitoring
        if obs_type != ObservationType.EXPIRY and product.is_barrier_hit(spot):
            if product.is_knock_out:
                # Pay rebate depending on timing preference
                if product.pay_at_hit:
                    return product.rebate * multiplier
                return product.rebate * math.exp(-rate * maturity) * multiplier
            vanilla = EuropeanVanillaOption(
                strike=product.strike,
                option_type=product.option_type,
                maturity=product.maturity,
                exercise_date=product.exercise_date,
                settlement_date=product.settlement_date,
            )
            vanilla_price = self._bs_engine.price(vanilla, pricing_env)
            return participation * vanilla_price * multiplier

        if obs_type == ObservationType.EXPIRY:
            return self._price_expiry(
                product, pricing_env, spot, maturity, rate, div, vol
            )

        if obs_type == ObservationType.DISCRETE:
            schedule = product.observation_schedule
            if schedule is None or not schedule.records:
                raise PricingError(
                    "Discrete barrier monitoring requires ObservationSchedule."
                )
            schedule.assert_analytical_ready(default_payoff=product.rebate)
            freq = schedule.ensure_regular_frequency(schedule.times)
            shifted_barrier = apply_barrier_shift(
                barrier=product.barrier,
                is_up_barrier=product.is_up_barrier,
                volatility=vol,
                observation_interval=freq,
            )
            return self._price_continuous(
                product=product,
                pricing_env=pricing_env,
                barrier=shifted_barrier,
                rate=rate,
                div=div,
                vol=vol,
                maturity=maturity,
                participation=participation,
            )

        if obs_type == ObservationType.CONTINUOUS:
            return self._price_continuous(
                product=product,
                pricing_env=pricing_env,
                barrier=product.barrier,
                rate=rate,
                div=div,
                vol=vol,
                maturity=maturity,
                participation=participation,
            )

        raise PricingError(f"Unsupported observation type: {obs_type}")

    def _price_continuous(
        self,
        product: BarrierOption,
        pricing_env: PricingEnvironment,
        barrier: float,
        rate: float,
        div: float,
        vol: float,
        maturity: float,
        participation: float,
    ) -> float:
        b = rate - div
        ko_price = self._price_knock_out_closed_form(
            spot=pricing_env.spot,
            strike=product.strike,
            barrier=barrier,
            maturity=maturity,
            rate=rate,
            carry=b,
            vol=vol,
            is_call=product.is_call(),
            is_up=product.is_up_barrier,
        )

        rebate_val = self._price_rebate_leg(product, pricing_env)

        if product.is_knock_out:
            value = participation * max(ko_price, 0.0) + rebate_val
            return value * product.contract_multiplier

        vanilla = EuropeanVanillaOption(
            strike=product.strike,
            option_type=product.option_type,
            maturity=product.maturity,
            exercise_date=product.exercise_date,
            settlement_date=product.settlement_date,
        )
        vanilla_price = self._bs_engine.price(vanilla, pricing_env)
        ki_price = max(vanilla_price - max(ko_price, 0.0), 0.0)
        value = participation * ki_price + rebate_val
        return value * product.contract_multiplier

    def _price_knock_out_closed_form(
        self,
        spot: float,
        strike: float,
        barrier: float,
        maturity: float,
        rate: float,
        carry: float,
        vol: float,
        is_call: bool,
        is_up: bool,
    ) -> float:
        sqrt_t = math.sqrt(maturity)
        mu = (carry - 0.5 * vol * vol) / (vol * vol)

        log_s_k = math.log(spot / strike)
        log_s_b = math.log(spot / barrier)
        pow_term = (barrier / spot) ** (2 * mu)
        pow_term_mu = (barrier / spot) ** (2 * (mu + 1))

        x1 = log_s_k / (vol * sqrt_t) + (1 + mu) * vol * sqrt_t
        x2 = log_s_b / (vol * sqrt_t) + (1 + mu) * vol * sqrt_t
        y1 = (
            math.log((barrier * barrier) / (spot * strike)) / (vol * sqrt_t)
            + (1 + mu) * vol * sqrt_t
        )
        y2 = math.log(barrier / spot) / (vol * sqrt_t) + (1 + mu) * vol * sqrt_t

        eta = -1.0 if is_up else 1.0

        def A(phi: float) -> float:
            return phi * spot * math.exp((carry - rate) * maturity) * stats.norm.cdf(
                phi * x1
            ) - phi * strike * math.exp(-rate * maturity) * stats.norm.cdf(
                phi * x1 - phi * vol * sqrt_t
            )

        def B(phi: float) -> float:
            return phi * spot * math.exp((carry - rate) * maturity) * stats.norm.cdf(
                phi * x2
            ) - phi * strike * math.exp(-rate * maturity) * stats.norm.cdf(
                phi * x2 - phi * vol * sqrt_t
            )

        def C(phi: float) -> float:
            return phi * spot * math.exp(
                (carry - rate) * maturity
            ) * pow_term_mu * stats.norm.cdf(eta * y1) - phi * strike * math.exp(
                -rate * maturity
            ) * pow_term * stats.norm.cdf(
                eta * y1 - eta * vol * sqrt_t
            )

        def D(phi: float) -> float:
            return phi * spot * math.exp(
                (carry - rate) * maturity
            ) * pow_term_mu * stats.norm.cdf(eta * y2) - phi * strike * math.exp(
                -rate * maturity
            ) * pow_term * stats.norm.cdf(
                eta * y2 - eta * vol * sqrt_t
            )

        if is_call:
            if is_up:
                if barrier <= strike:
                    return 0.0
                return A(1.0) - B(1.0) + C(1.0) - D(1.0)
            if barrier < strike:
                return A(1.0) - C(1.0)
            return B(1.0) - D(1.0)

        # Put
        if is_up:
            if barrier > strike:
                return A(-1.0) - C(-1.0)
            return B(-1.0) - D(-1.0)

        if barrier < strike:
            return A(-1.0) - B(-1.0) + C(-1.0) - D(-1.0)
        return 0.0

    def _price_expiry(
        self,
        product: BarrierOption,
        pricing_env: PricingEnvironment,
        spot: float,
        maturity: float,
        rate: float,
        div: float,
        vol: float,
    ) -> float:
        discount = math.exp(-rate * maturity)
        d1, d2 = self._d1_d2(spot, product.barrier, maturity, rate, div, vol)
        prob_up = stats.norm.cdf(d2)
        prob_down = stats.norm.cdf(-d2)
        asset_up = spot * math.exp(-div * maturity) * stats.norm.cdf(d1)
        asset_down = spot * math.exp(-div * maturity) * stats.norm.cdf(-d1)

        vanilla = EuropeanVanillaOption(
            strike=product.strike,
            option_type=product.option_type,
            maturity=product.maturity,
            exercise_date=product.exercise_date,
            settlement_date=product.settlement_date,
        )
        vanilla_price = self._bs_engine.price(vanilla, pricing_env)

        if product.is_call():
            ko_no_rebate = self._expiry_call_knock_out(
                vanilla_price=vanilla_price,
                strike=product.strike,
                barrier=product.barrier,
                discount=discount,
                prob_up=prob_up,
                asset_up=asset_up,
            )
        else:
            ko_no_rebate = self._expiry_put_knock_out(
                vanilla_price=vanilla_price,
                strike=product.strike,
                barrier=product.barrier,
                discount=discount,
                prob_up=prob_up,
                prob_down=prob_down,
                asset_up=asset_up,
                asset_down=asset_down,
            )

        participation = product.participation_rate

        if product.is_up_barrier:
            prob_hit = prob_up
        else:
            prob_hit = prob_down
        prob_survive = 1.0 - prob_hit

        if product.is_knock_out:
            rebate_component = product.rebate * discount * prob_hit
            value = max(participation * ko_no_rebate + rebate_component, 0.0)
            return value * product.contract_multiplier

        ki_no_rebate = max(vanilla_price - ko_no_rebate, 0.0)
        rebate_component = product.rebate * discount * prob_survive
        value = max(participation * ki_no_rebate + rebate_component, 0.0)
        return value * product.contract_multiplier

    def _expiry_call_knock_out(
        self,
        vanilla_price: float,
        strike: float,
        barrier: float,
        discount: float,
        prob_up: float,
        asset_up: float,
    ) -> float:
        # Up-and-out call observed at expiry
        if barrier > strike and barrier > 0:
            portion_above = asset_up - strike * discount * prob_up
            return max(vanilla_price - portion_above, 0.0)
        if barrier <= strike:
            return 0.0 if barrier > 0 else vanilla_price
        # Down-and-out call
        if barrier < strike:
            return vanilla_price
        portion_above = asset_up - strike * discount * prob_up
        return max(portion_above, 0.0)

    def _expiry_put_knock_out(
        self,
        vanilla_price: float,
        strike: float,
        barrier: float,
        discount: float,
        prob_up: float,
        prob_down: float,
        asset_up: float,
        asset_down: float,
    ) -> float:
        # Up-and-out put observed at expiry
        if barrier <= strike:
            return max(strike * discount * prob_down - asset_down, 0.0)
        if barrier > strike:
            return vanilla_price
        # Down-and-out put
        if barrier >= strike:
            return 0.0
        portion_below = strike * discount * prob_down - asset_down
        return max(vanilla_price - portion_below, 0.0)

    def _d1_d2(
        self,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        div: float,
        vol: float,
    ) -> tuple:
        sqrt_t = math.sqrt(maturity)
        if spot <= 0 or strike <= 0 or vol <= 0 or maturity <= 0:
            raise ValidationError("Invalid inputs for d1/d2 calculation.")
        d1 = (math.log(spot / strike) + (rate - div + 0.5 * vol * vol) * maturity) / (
            vol * sqrt_t
        )
        d2 = d1 - vol * sqrt_t
        return d1, d2

    def _price_rebate_leg(
        self, product: BarrierOption, pricing_env: PricingEnvironment
    ) -> float:
        """Value rebate via one-touch/no-touch pricing for continuous/discrete."""
        if product.rebate <= 0:
            return 0.0

        touch_type = TouchType.ONE_TOUCH if product.is_knock_out else TouchType.NO_TOUCH
        barrier_direction = (
            BarrierDirection.UP if product.is_up_barrier else BarrierDirection.DOWN
        )
        pay_at_hit = product.pay_at_hit if product.is_knock_out else False

        rebate_leg = OneTouchOption(
            barrier=product.barrier,
            barrier_direction=barrier_direction,
            maturity=product.maturity,
            exercise_date=product.exercise_date,
            settlement_date=product.settlement_date,
            rebate=product.rebate,
            payment_at_hit=pay_at_hit,
            touch_type=touch_type,
            observation_type=product.observation_type,
            observation_dates=product.observation_dates,
            observation_schedule=product.observation_schedule,
        )
        return self._one_touch_engine.price(rebate_leg, pricing_env)

    def _validate_inputs(
        self,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        div: float,
        vol: float,
        barrier: float,
    ) -> None:
        if spot <= 0:
            raise ValidationError(f"Spot price must be positive, got {spot}")
        if strike <= 0:
            raise ValidationError(f"Strike price must be positive, got {strike}")
        if barrier <= 0:
            raise ValidationError(f"Barrier must be positive, got {barrier}")
        if maturity < 0:
            raise ValidationError(f"Maturity must be non-negative, got {maturity}")
        if vol <= 0:
            raise ValidationError(f"Volatility must be positive, got {vol}")
        if vol < self.MIN_VOL or vol > self.MAX_VOL:
            raise ValidationError(
                f"Volatility {vol} outside supported range [{self.MIN_VOL}, {self.MAX_VOL}]"
            )
        if maturity > self.MAX_MATURITY:
            raise ValidationError(
                f"Maturity too long for analytical barrier pricing: {maturity}"
            )

    def __repr__(self):
        return "BarrierAnalyticalEngine()"
