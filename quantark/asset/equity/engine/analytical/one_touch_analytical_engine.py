"""
Analytical pricing engine for one-touch and no-touch options.
"""

import math
from typing import Optional

from scipy import stats

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.engine.capabilities import SettlementSupport
from quantark.asset.equity.engine.settlement_support import (
    pending_receivable_pv,
    resolve_terminal_timing,
    terminal_lifecycle_pv,
    validate_settlement_capability,
)
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option import CashOrNothingDigitalOption, OneTouchOption
from quantark.asset.equity.settlement import (
    CashflowKind,
    SettlementRequest,
    SettlementResolver,
)
from quantark.execution.errors import CapabilityError
from quantark.asset.equity.param import EngineParams
from quantark.priceenv import PricingEnvironment
from quantark.util.barrier_shift import apply_barrier_shift
from quantark.util.enum import ObservationType, OptionType, TouchType
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import PricingError, ValidationError

from .digital_option_engine import DigitalOptionAnalyticalEngine


class OneTouchAnalyticalEngine(BaseEngine):
    """
    Closed-form pricing engine for one-touch and no-touch options.

    Supports:
        - Continuous monitoring
        - Discrete monitoring via Broadie-Glasserman-Kou barrier shift (regular grids)
        - Expiry-only monitoring via digital-option fallback

    Note:
        No tenor/365 scaling is applied at the engine level.
    """

    engine_type = EngineType.ANALYTICAL
    settlement_support = SettlementSupport.EVENT_AND_TERMINAL
    supports_lifecycle_state = True

    MIN_VOL = 0.001
    MAX_VOL = 5.0
    MIN_MATURITY = 1e-10
    MAX_MATURITY = 50.0

    def __init__(self, params: Optional[EngineParams] = None):
        super().__init__(params)
        self._digital_engine = DigitalOptionAnalyticalEngine(params)

    def price(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        *,
        lifecycle_state=None,
    ) -> float:
        if not isinstance(product, OneTouchOption):
            raise PricingError(
                f"OneTouchAnalyticalEngine only supports OneTouchOption, "
                f"got {type(product).__name__}"
            )
        validate_settlement_capability(self, product, lifecycle_state)
        fixed_pv = terminal_lifecycle_pv(lifecycle_state, pricing_env)
        if fixed_pv is not None:
            return fixed_pv
        terminal_timing = resolve_terminal_timing(product, pricing_env)
        pending_pv = pending_receivable_pv(lifecycle_state, pricing_env)

        spot = pricing_env.spot
        maturity = product.get_maturity(pricing_env)
        rate = pricing_env.get_rate(maturity)
        div = pricing_env.get_div_yield(maturity)
        vol = pricing_env.get_vol(product.barrier, maturity)
        rebate = product.rebate

        pay_at_hit = product.payment_at_hit if product.is_one_touch else False

        self._validate_inputs(spot, product.barrier, maturity, vol, rebate)
        if (
            pay_at_hit
            and product.observation_type != ObservationType.EXPIRY
            and not product.is_barrier_hit(spot)
            and self._requests_delayed_hit_payment(product)
        ):
            raise CapabilityError(
                "OneTouchAnalyticalEngine cannot represent a delayed "
                "continuous/discrete first-hit payment"
            )

        # Immediate handling for near-expiry or already-hit barriers
        if maturity < self.MIN_MATURITY:
            value = self._instantaneous_payoff(
                product=product,
                spot=spot,
                maturity=maturity,
                rate=rate,
                pay_at_hit=pay_at_hit,
                pricing_env=pricing_env,
                terminal_payment_df=terminal_timing.payment_df,
            )
            return value + pending_pv

        obs_type = product.observation_type

        if obs_type != ObservationType.EXPIRY and product.is_barrier_hit(spot):
            if product.is_one_touch:
                if pay_at_hit:
                    event_timing = SettlementResolver.resolve_contingent(
                        product,
                        SettlementRequest(
                            kind=CashflowKind.HIT,
                            determination_date=pricing_env.valuation_date,
                            determination_time=0.0,
                            cashflow_id="touch:already_hit",
                        ),
                        pricing_env,
                    )
                    return rebate * event_timing.payment_df + pending_pv
                return rebate * terminal_timing.payment_df + pending_pv
            return pending_pv

        if obs_type == ObservationType.EXPIRY:
            return self._price_expiry(product, pricing_env) + pending_pv

        if obs_type == ObservationType.DISCRETE:
            schedule = product.observation_schedule
            if schedule is None or not schedule.records:
                raise PricingError(
                    "Discrete monitoring requires a populated ObservationSchedule."
                )
            schedule.assert_analytical_ready(default_payoff=rebate)
            frequency = schedule.ensure_regular_frequency(schedule.times)
            barrier = apply_barrier_shift(
                barrier=product.barrier,
                is_up_barrier=product.is_up_barrier,
                volatility=vol,
                observation_interval=frequency,
            )
        elif obs_type == ObservationType.CONTINUOUS:
            barrier = product.barrier
        else:
            raise PricingError(f"Unsupported observation type: {obs_type}")

        if product.is_one_touch:
            value = self._one_touch_price(
                spot=spot,
                barrier=barrier,
                maturity=maturity,
                rate=rate,
                div=div,
                vol=vol,
                rebate=rebate,
                pay_at_hit=pay_at_hit,
                is_up=product.is_up_barrier,
            )
            if not pay_at_hit:
                value *= terminal_timing.delay_df
            return value + pending_pv

        # No-touch: pay only at expiry if not hit
        prob_touch = self._touch_probability(
            spot=spot,
            barrier=barrier,
            maturity=maturity,
            rate=rate,
            div=div,
            vol=vol,
            is_up=product.is_up_barrier,
        )
        prob_touch = min(max(prob_touch, 0.0), 1.0)
        return (
            rebate
            * terminal_timing.payment_df
            * max(0.0, 1.0 - prob_touch)
            + pending_pv
        )

    def _price_expiry(
        self, product: OneTouchOption, pricing_env: PricingEnvironment
    ) -> float:
        """Price with expiry-only monitoring using digital option fallback."""
        option_type = self._digital_direction(product)
        digital = CashOrNothingDigitalOption(
            strike=product.barrier,
            payout=product.rebate,
            option_type=option_type,
            maturity=product.maturity,
            exercise_date=product.exercise_date,
            settlement_date=product.settlement_date,
            settlement_convention=product.settlement_convention,
        )
        return self._digital_engine.price(digital, pricing_env)

    @staticmethod
    def _requests_delayed_hit_payment(product: OneTouchOption) -> bool:
        convention = product.settlement_convention
        if convention is not None and convention.lag != 0.0:
            return True
        schedule = product.observation_schedule
        if schedule is None:
            return False
        return any(
            record.settlement_date is not None
            or (
                record.settlement_time is not None
                and record.observation_time is not None
                and record.settlement_time != record.observation_time
            )
            for record in schedule.records
        )

    def _digital_direction(self, product: OneTouchOption) -> OptionType:
        """Map one-touch/no-touch direction to an equivalent digital payoff."""
        if product.is_one_touch:
            return OptionType.CALL if product.is_up_barrier else OptionType.PUT
        # No-touch pays if terminal spot stays on the non-breach side
        return OptionType.PUT if product.is_up_barrier else OptionType.CALL

    def _one_touch_price(
        self,
        spot: float,
        barrier: float,
        maturity: float,
        rate: float,
        div: float,
        vol: float,
        rebate: float,
        pay_at_hit: bool,
        is_up: bool,
    ) -> float:
        """Closed-form one-touch price for continuous or shifted discrete barriers."""
        if pay_at_hit:
            return rebate * self._instant_touch_term(
                spot=spot,
                barrier=barrier,
                maturity=maturity,
                rate=rate,
                div=div,
                vol=vol,
                is_up=is_up,
            )

        return rebate * math.exp(-rate * maturity) * self._expiry_touch_term(
            spot=spot,
            barrier=barrier,
            maturity=maturity,
            rate=rate,
            div=div,
            vol=vol,
            is_up=is_up,
        )

    def _touch_probability(
        self,
        spot: float,
        barrier: float,
        maturity: float,
        rate: float,
        div: float,
        vol: float,
        is_up: bool,
    ) -> float:
        """Probability of touching the barrier before expiry (used for no-touch)."""
        return self._expiry_touch_term(
            spot=spot,
            barrier=barrier,
            maturity=maturity,
            rate=rate,
            div=div,
            vol=vol,
            is_up=is_up,
        )

    def _instant_touch_term(
        self,
        spot: float,
        barrier: float,
        maturity: float,
        rate: float,
        div: float,
        vol: float,
        is_up: bool,
    ) -> float:
        b = rate - div
        mu = (b - 0.5 * vol * vol) / (vol * vol)
        lam = math.sqrt(mu * mu + 2.0 * rate / (vol * vol))
        sqrt_t = math.sqrt(maturity)
        z = math.log(barrier / spot) / (vol * sqrt_t) + lam * vol * sqrt_t
        eta = -1.0 if is_up else 1.0

        term1 = math.pow(barrier / spot, mu + lam) * stats.norm.cdf(eta * z)
        term2 = math.pow(barrier / spot, mu - lam) * stats.norm.cdf(
            eta * z - 2 * eta * lam * vol * sqrt_t
        )
        return term1 + term2

    def _expiry_touch_term(
        self,
        spot: float,
        barrier: float,
        maturity: float,
        rate: float,
        div: float,
        vol: float,
        is_up: bool,
    ) -> float:
        b = rate - div
        mu = (b - 0.5 * vol * vol) / (vol * vol)
        sqrt_t = math.sqrt(maturity)
        log_s_b = math.log(spot / barrier)
        x2 = log_s_b / (vol * sqrt_t) + (1 + mu) * vol * sqrt_t
        y2 = -log_s_b / (vol * sqrt_t) + (1 + mu) * vol * sqrt_t
        phi = 1.0 if is_up else -1.0
        eta = -1.0 if is_up else 1.0
        pow_term = math.pow(barrier / spot, 2 * mu)
        return stats.norm.cdf(phi * x2 - phi * vol * sqrt_t) + pow_term * stats.norm.cdf(
            eta * y2 - eta * vol * sqrt_t
        )

    def _instantaneous_payoff(
        self,
        product: OneTouchOption,
        spot: float,
        maturity: float,
        rate: float,
        pay_at_hit: bool,
        pricing_env: PricingEnvironment,
        terminal_payment_df: float,
    ) -> float:
        """Handle payoffs when maturity is effectively zero."""
        touched = product.is_barrier_hit(spot)
        if product.is_one_touch:
            if touched:
                if pay_at_hit:
                    event_timing = SettlementResolver.resolve_contingent(
                        product,
                        SettlementRequest(
                            kind=CashflowKind.HIT,
                            determination_date=pricing_env.valuation_date,
                            determination_time=0.0,
                            cashflow_id="touch:expiry",
                        ),
                        pricing_env,
                    )
                    return product.rebate * event_timing.payment_df
                return product.rebate * terminal_payment_df
            return 0.0
        return product.rebate * terminal_payment_df if not touched else 0.0

    def _validate_inputs(
        self,
        spot: float,
        barrier: float,
        maturity: float,
        vol: float,
        rebate: float,
    ) -> None:
        if spot <= 0:
            raise ValidationError(f"Spot price must be positive, got {spot}")
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
                f"Maturity too long for analytical one-touch pricing: {maturity}"
            )
        if rebate < 0:
            raise ValidationError(f"Rebate must be non-negative, got {rebate}")

    def __repr__(self):
        return "OneTouchAnalyticalEngine()"
