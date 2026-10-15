"""
Analytical pricing engine for one-touch and no-touch options.
"""

import math
from typing import Optional

import numpy as np
from scipy.special import ndtr

from quantark.asset.equity.engine.barrier_schedule_rules import (
    forward_barrier_level,
    forward_knock_out_cash,
    refuse_mixed_forward_levels,
    valuation_barrier_terms,
)
from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.engine.capabilities import SettlementSupport
from quantark.asset.equity.engine.settlement_support import (
    constant_hit_lag_year_fraction,
    pending_receivable_pv,
    resolve_terminal_timing,
    terminal_lifecycle_pv,
    validate_settlement_capability,
)
from quantark.asset.equity.engine.monitoring_ended import monitoring_ended, remaining_replication
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option import CashOrNothingDigitalOption, OneTouchOption
from quantark.asset.equity.settlement import (
    CashflowKind,
    SettlementRequest,
    SettlementResolver,
)
from quantark.asset.equity.param import EngineParams
from quantark.priceenv import PricingEnvironment
from quantark.util.barrier_shift import apply_barrier_shift
from quantark.util.enum import ObservationType, OptionType, TouchType
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import PricingError, ValidationError
from quantark.util.numerical import pow_elementwise

from .digital_option_engine import DigitalOptionAnalyticalEngine
from .first_observation import (
    first_future_observation,
    has_future_observation,
    observed_at_valuation,
    resolve_current_records,
    through_first_observation,
)


class OneTouchAnalyticalEngine(BaseEngine):
    """
    Closed-form pricing engine for one-touch and no-touch options.

    Supports:
        - Continuous monitoring
        - Discrete monitoring via Broadie-Glasserman-Kou barrier shift (regular grids), valued through the next
          observation when none decides now (inside or beyond the barrier), the shifted formula only after it
        - Expiry-only monitoring via digital-option fallback

    Note:
        No tenor/365 scaling is applied at the engine level.
    """

    engine_type = EngineType.ANALYTICAL
    # params-only engine: instance state == construction arguments (patch spec 2026-09-03 §7)
    MODEL_FINGERPRINT_ATTRS = ('params', '_digital_engine')
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
        if monitoring_ended(product) and product.get_maturity(pricing_env) > 0.0:
            # no observation left (spec 2026-10-07 theta boundary 3.3): the remaining contract, by replication
            knocked_in = bool(getattr(lifecycle_state, "knocked_in", False))
            return (remaining_replication(product, knocked_in).price(pricing_env)
                    + pending_receivable_pv(lifecycle_state, pricing_env))
        terminal_timing = resolve_terminal_timing(product, pricing_env)
        pending_pv = pending_receivable_pv(lifecycle_state, pricing_env)

        spot = pricing_env.spot
        maturity = product.get_maturity(pricing_env)
        obs_type = product.observation_type
        discrete = obs_type == ObservationType.DISCRETE
        resolved = None
        if discrete:
            # W18c (spec 2026-10-13 I4): a DISCRETE schedule is touched now only by a record at the valuation instant,
            # at its own level; forward, the closed form monitors the level every remaining record shares (None:
            # they differ -- no closed form, refused before any forward value)
            resolved = self._resolved_records(product, pricing_env)    # once per price
            hit_now = self._breached_at_valuation(product, pricing_env, spot, resolved)
            forward_level = forward_barrier_level(product, pricing_env)
            beyond_forward = forward_level is not None and self._beyond(product, spot, forward_level)
        else:
            hit_now = obs_type != ObservationType.EXPIRY and product.is_barrier_hit(spot)
            forward_level = product.barrier
            beyond_forward = hit_now
        rate = pricing_env.get_rate(maturity)
        div = pricing_env.get_div_yield(maturity)
        # the volatility at the monitored level (mixed remaining levels price no forward value: product.barrier then
        # feeds only the input checks)
        vol = pricing_env.get_vol(forward_level if forward_level is not None else product.barrier, maturity)
        rebate = product.rebate
        # the cash of a future touch: a DISCRETE one-touch pays the touching record's payoff (W18c follow-up); a
        # no-touch pays its rebate when nothing touches
        forward_cash = (
            forward_knock_out_cash(product, pricing_env) if discrete and product.is_one_touch else rebate
        )

        pay_at_hit = product.payment_at_hit if product.is_one_touch else False

        self._validate_inputs(spot, product.barrier, maturity, vol, rebate)
        # A hit-paid rebate with a CONSTANT settlement lag prices exactly as
        # exp(-rate * lag) times the pay-at-hit value; hit-date-dependent lags
        # raise here, before pricing (patch spec 2026-09-03 §5).
        hit_lag = 0.0
        if (
            pay_at_hit
            and obs_type != ObservationType.EXPIRY
            and not (hit_now or beyond_forward)
        ):
            hit_lag = constant_hit_lag_year_fraction(product, pricing_env, maturity)

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

        if discrete and not hit_now:
            if forward_level is None:
                refuse_mixed_forward_levels(product.observation_schedule, pricing_env, default_payoff=rebate)
            if beyond_forward or has_future_observation(resolved):
                # No observation deciding now: the next observation decides, the shifted formula only after it
                return (
                    self._price_beyond_discrete(
                        product, pricing_env, spot, maturity, rate, div, vol, terminal_timing, forward_level,
                        forward_cash, resolved,
                    )
                    + pending_pv
                )

        if hit_now:
            if product.is_one_touch:
                # the deciding record's payoff on a DISCRETE schedule (W18c), the rebate otherwise
                if discrete:
                    rebate = valuation_barrier_terms(product, pricing_env)[1]
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
            schedule.assert_analytical_ready(default_payoff=rebate, pricing_env=pricing_env)
            frequency = schedule.ensure_regular_frequency(schedule.times)
            barrier = apply_barrier_shift(
                barrier=forward_level,
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
                rebate=forward_cash,
                pay_at_hit=pay_at_hit,
                is_up=product.is_up_barrier,
                hit_lag=hit_lag,
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

    def _resolved_records(self, product: OneTouchOption, pricing_env: PricingEnvironment):
        return resolve_current_records(
            product.observation_schedule,
            pricing_env,
            default_barrier=product.barrier,
            default_payoff=product.rebate,
            require_single=True,
            product=product,
        )

    @staticmethod
    def _beyond(product: OneTouchOption, spot: float, level: float) -> bool:
        return (spot >= level) if product.is_up_barrier else (spot <= level)

    def _breached_at_valuation(
        self, product: OneTouchOption, pricing_env: PricingEnvironment, spot: float, resolved=None
    ) -> bool:
        """Whether an observation at the valuation instant is breached by the spot, at its own level (``resolved``:
        the records already resolved for this price)."""
        if resolved is None:
            resolved = self._resolved_records(product, pricing_env)
        return any(self._beyond(product, spot, rec.barrier) for rec in observed_at_valuation(resolved))

    def _price_beyond_discrete(
        self,
        product: OneTouchOption,
        pricing_env: PricingEnvironment,
        spot: float,
        maturity: float,
        rate: float,
        div: float,
        vol: float,
        terminal_timing,
        forward_level: float,
        forward_cash: float,
        resolved=None,
    ) -> float:
        """The value through the first future observation (``first_observation``), no observation deciding now,
        the spot inside or beyond the barrier.

        A touch at the first future observation ``t1`` pays the one-touch's rebate (at the hit, after the constant
        hit lag, or at the terminal payment time) and ends the no-touch; otherwise the continuation is this
        engine's shifted formula from ``t1`` (no observation left: no touch), on ``forward_level``, the level the
        remaining records share; a one-touch pays ``forward_cash``, their payoff (W18c). ``resolved``: the records
        already resolved for this price. The closures take an array of spots at ``t1``.
        """
        schedule = product.observation_schedule
        if schedule is None or not schedule.records:
            raise PricingError("Discrete monitoring requires a populated ObservationSchedule.")
        schedule.assert_analytical_ready(default_payoff=product.rebate, pricing_env=pricing_env)
        rebate = forward_cash
        frequency = schedule.ensure_regular_frequency(schedule.times)
        first = first_future_observation(
            resolved if resolved is not None else self._resolved_records(product, pricing_env),
            schedule,
            single=True,
            is_up=product.is_up_barrier,
        )
        tau = maturity - first.time
        shifted = apply_barrier_shift(
            barrier=forward_level,
            is_up_barrier=product.is_up_barrier,
            volatility=vol,
            observation_interval=frequency,
        )
        terminal_df = math.exp(-rate * tau) * terminal_timing.delay_df
        pay_at_hit = product.payment_at_hit if product.is_one_touch else False
        hit_lag = 0.0
        if pay_at_hit:
            hit_lag = constant_hit_lag_year_fraction(product, pricing_env, maturity)

        def touch_probability(s: np.ndarray) -> np.ndarray:
            return np.minimum(np.maximum(self._touch_probability(
                spot=s, barrier=shifted, maturity=tau, rate=rate, div=div, vol=vol,
                is_up=product.is_up_barrier,
            ), 0.0), 1.0)

        if product.is_one_touch:
            def hit_value(s: np.ndarray) -> float:
                return rebate * math.exp(-rate * hit_lag) if pay_at_hit else rebate * terminal_df

            def continuation(s: np.ndarray) -> np.ndarray:
                if first.remaining == 0:
                    return 0.0
                value = self._one_touch_price(
                    spot=s, barrier=shifted, maturity=tau, rate=rate, div=div, vol=vol,
                    rebate=rebate, pay_at_hit=pay_at_hit, is_up=product.is_up_barrier, hit_lag=hit_lag,
                )
                return value if pay_at_hit else value * terminal_timing.delay_df
        else:
            def hit_value(s: np.ndarray) -> float:
                return 0.0

            def continuation(s: np.ndarray) -> np.ndarray:
                if first.remaining == 0:
                    return rebate * terminal_df
                return rebate * terminal_df * np.maximum(0.0, 1.0 - touch_probability(s))

        return through_first_observation(
            spot=spot,
            first=first,
            rate=rate,
            div=div,
            vol=vol,
            hit_value=hit_value,
            continuation=continuation,
            remaining_time=tau,
            hit_is_constant=True,
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
        hit_lag: float = 0.0,
    ) -> float:
        """Closed-form one-touch price for continuous or shifted discrete barriers.

        ``hit_lag`` is the constant year-fraction delay of a hit-paid rebate:
        E[R e^{-r(tau+L)} 1{tau<=T}] = e^{-rL} E[R e^{-r tau} 1{tau<=T}] under the
        formula's flat ``rate`` (the rate read at ``maturity``; under a term
        structure the factor inherits the flat-r assumption the hit-paid leg
        already makes). 0.0 leaves the unlagged path bitwise unchanged.
        """
        if pay_at_hit:
            value = rebate * self._instant_touch_term(
                spot=spot,
                barrier=barrier,
                maturity=maturity,
                rate=rate,
                div=div,
                vol=vol,
                is_up=is_up,
            )
            if hit_lag != 0.0:
                value *= math.exp(-rate * hit_lag)
            return value

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
        """``spot`` a float or an array of spots (the same operations elementwise)."""
        array = isinstance(spot, np.ndarray)
        log, power = (np.log, pow_elementwise) if array else (math.log, math.pow)
        b = rate - div
        mu = (b - 0.5 * vol * vol) / (vol * vol)
        lam = math.sqrt(mu * mu + 2.0 * rate / (vol * vol))
        sqrt_t = math.sqrt(maturity)
        z = log(barrier / spot) / (vol * sqrt_t) + lam * vol * sqrt_t
        eta = -1.0 if is_up else 1.0

        term1 = power(barrier / spot, mu + lam) * ndtr(eta * z)
        term2 = power(barrier / spot, mu - lam) * ndtr(
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
        """``spot`` a float or an array of spots (the same operations elementwise)."""
        array = isinstance(spot, np.ndarray)
        log, power = (np.log, pow_elementwise) if array else (math.log, math.pow)
        b = rate - div
        mu = (b - 0.5 * vol * vol) / (vol * vol)
        sqrt_t = math.sqrt(maturity)
        log_s_b = log(spot / barrier)
        x2 = log_s_b / (vol * sqrt_t) + (1 + mu) * vol * sqrt_t
        y2 = -log_s_b / (vol * sqrt_t) + (1 + mu) * vol * sqrt_t
        phi = 1.0 if is_up else -1.0
        eta = -1.0 if is_up else 1.0
        pow_term = power(barrier / spot, 2 * mu)
        return ndtr(phi * x2 - phi * vol * sqrt_t) + pow_term * ndtr(
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
        """Handle payoffs when maturity is effectively zero: a discrete schedule touches only through its records
        at the valuation instant, each at its own level, and a one-touch touched there pays that record's payoff
        (W18c)."""
        rebate = product.rebate
        if product.observation_type == ObservationType.DISCRETE:
            touched = self._breached_at_valuation(product, pricing_env, spot)
        else:
            touched = product.is_barrier_hit(spot)
        if product.is_one_touch:
            if touched:
                if product.observation_type == ObservationType.DISCRETE:
                    rebate = valuation_barrier_terms(product, pricing_env)[1]
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
                    return rebate * event_timing.payment_df
                return rebate * terminal_payment_df
            return 0.0
        return rebate * terminal_payment_df if not touched else 0.0

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

    def static_replication(self, product, pricing_env):
        """An ended schedule's remaining contract (spec 2026-10-07 theta boundary 3.3), else none."""
        if monitoring_ended(product):
            return remaining_replication(product)
        return super().static_replication(product, pricing_env)

    def __repr__(self):
        return "OneTouchAnalyticalEngine()"
