"""
Analytical pricing engine for single-barrier options.

Supports:
- Continuous monitoring (closed-form barrier formulas)
- Discrete monitoring with barrier shift (requires regular schedule)
- Expiry-only monitoring via vanilla/digital decomposition
"""

import math
from typing import Optional

import numpy as np
from scipy import stats
from scipy.special import ndtr

from quantark.asset.equity.engine.barrier_schedule_rules import (
    forward_barrier_level,
    forward_knock_out_cash,
    refuse_knock_in_record_payoff,
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
from quantark.asset.equity.engine.monitoring_ended import (
    knock_in_carried,
    remaining_contract_applies,
    remaining_replication,
)
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option import (
    BarrierOption,
    EuropeanVanillaOption,
    OneTouchOption,
)
from quantark.asset.equity.param import EngineParams
from quantark.priceenv import PricingEnvironment
from quantark.util.barrier_shift import apply_barrier_shift
from quantark.util.enum import ObservationType, BarrierDirection, TouchType
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import ValidationError, PricingError
from quantark.util.numerical import is_zero, pow_elementwise

from .black_scholes_engine import BlackScholesEngine
from .first_observation import (
    first_future_observation,
    has_future_observation,
    observed_at_valuation,
    resolve_current_records,
    through_first_observation,
)
from .one_touch_analytical_engine import OneTouchAnalyticalEngine


class BarrierAnalyticalEngine(BaseEngine):
    """
    Closed-form pricing engine for single-barrier options.

    Notes:
        - Discrete monitoring uses Broadie-Glasserman-Kou barrier shift assuming
          a regular observation grid and fixed payoff across observations. With no
          observation deciding now the option is valued through the next
          observation (``first_observation``), inside or beyond the barrier, and
          the shifted formula only after it; only an observation at the valuation
          instant knocks the option out or in on the spot.
        - At maturity a knock-in carried as triggered pays the payoff; otherwise
          only what is observed now decides: a discrete schedule's records at the
          valuation instant, each at its own level and cash.
        - Rebates for continuous/discrete monitoring are valued via OneTouchAnalyticalEngine.
    """

    engine_type = EngineType.ANALYTICAL
    # params-only engine: instance state == construction arguments (patch spec 2026-09-03 §7)
    MODEL_FINGERPRINT_ATTRS = ('params', '_bs_engine', '_one_touch_engine')
    settlement_support = SettlementSupport.EVENT_AND_TERMINAL
    supports_lifecycle_state = True

    MIN_VOL = 0.001
    MAX_VOL = 5.0
    MIN_MATURITY = 1e-10
    MAX_MATURITY = 50.0

    def __init__(self, params: Optional[EngineParams] = None):
        super().__init__(params)
        self._bs_engine = BlackScholesEngine(params)
        self._one_touch_engine = OneTouchAnalyticalEngine(params)

    def price(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        *,
        lifecycle_state=None,
    ) -> float:
        if not isinstance(product, BarrierOption):
            raise PricingError(
                f"BarrierAnalyticalEngine only supports BarrierOption, "
                f"got {type(product).__name__}"
            )
        validate_settlement_capability(self, product, lifecycle_state)
        refuse_knock_in_record_payoff(product)      # one rebate per knock-in, as on every route (W18 Q3, R19)
        fixed_pv = terminal_lifecycle_pv(lifecycle_state, pricing_env)
        if fixed_pv is not None:
            return fixed_pv
        if remaining_contract_applies(product, lifecycle_state) and product.get_maturity(pricing_env) > 0.0:
            # no observation left (spec 2026-10-07 theta boundary 3.3): the remaining contract, by replication
            return (remaining_replication(product, knock_in_carried(product, lifecycle_state)).price(pricing_env)
                    + pending_receivable_pv(lifecycle_state, pricing_env))
        terminal_timing = resolve_terminal_timing(product, pricing_env)
        pending_pv = pending_receivable_pv(lifecycle_state, pricing_env)

        spot = pricing_env.spot
        strike = product.strike
        maturity = product.get_maturity(pricing_env)
        rate = pricing_env.get_rate(maturity)
        div = pricing_env.get_div_yield(maturity)
        vol = pricing_env.get_vol(strike, maturity)
        participation = product.participation_rate
        multiplier = product.contract_multiplier

        self._validate_inputs(spot, strike, maturity, rate, div, vol, product.barrier)
        obs_type = product.observation_type
        discrete = obs_type == ObservationType.DISCRETE
        resolved = None
        if discrete:
            # W18c (spec 2026-10-13 I4): a DISCRETE schedule is hit now only by a record at the valuation instant, at
            # its own level; forward, the closed form monitors the level every remaining record shares (None: they
            # differ -- no closed form, refused before any forward value)
            resolved = self._resolved_records(product, pricing_env)    # once per price
            hit_now = self._breached_at_valuation(product, pricing_env, spot, resolved)
            forward_level = forward_barrier_level(product, pricing_env)
            beyond_forward = forward_level is not None and self._breached(product, spot, forward_level)
        else:
            hit_now = obs_type != ObservationType.EXPIRY and product.is_barrier_hit(spot)
            forward_level = product.barrier
            beyond_forward = hit_now
        # the cash of a future hit: a DISCRETE knock-out pays the hit record's payoff (W18c follow-up); a knock-in's
        # rebate is paid when nothing hits
        forward_cash = (
            forward_knock_out_cash(product, pricing_env) if discrete and product.is_knock_out else product.rebate
        )
        if (
            product.is_knock_out
            and product.pay_at_hit
            and forward_cash > 0
            and obs_type != ObservationType.EXPIRY
            and not (hit_now or beyond_forward)
        ):
            # Fail closed BEFORE pricing when the first-hit lag is not one constant;
            # the rebate leg (one-touch engine) applies the exp(-r L) factor itself
            # (patch spec 2026-09-03 §5).
            constant_hit_lag_year_fraction(product, pricing_env, maturity)

        # Immediate handling for zero maturity
        if maturity < self.MIN_MATURITY:
            return (
                self._price_at_maturity(product, pricing_env, spot, terminal_timing,
                                        knocked_in=knock_in_carried(product, lifecycle_state), resolved=resolved)
                + pending_pv
            )

        if discrete and not hit_now:
            if forward_level is None:
                refuse_mixed_forward_levels(product.observation_schedule, pricing_env, default_payoff=product.rebate)
            if beyond_forward or has_future_observation(resolved):
                # No observation deciding now: the next observation decides, the shifted formula only after it
                return (
                    self._price_through_first_observation(
                        product, pricing_env, spot, maturity, rate, div, vol, terminal_timing, forward_level,
                        forward_cash, resolved,
                    )
                    + pending_pv
                )

        # Barrier already hit: continuously, or by an observation at the valuation instant
        if hit_now:
            if product.is_knock_out:
                # the deciding record's payoff on a DISCRETE schedule (W18c), the rebate otherwise
                cash = valuation_barrier_terms(product, pricing_env)[1] if discrete else product.rebate
                # Pay rebate depending on timing preference
                if product.pay_at_hit:
                    return (
                        self._price_rebate_leg(product, pricing_env, rebate=cash)
                        * multiplier
                        + pending_pv
                    )
                return (
                    cash
                    * terminal_timing.payment_df
                    * multiplier
                    + pending_pv
                )
            vanilla = EuropeanVanillaOption(
                strike=product.strike,
                option_type=product.option_type,
                maturity=product.maturity,
                exercise_date=product.exercise_date,
            )
            vanilla_price = self._bs_engine.price(vanilla, pricing_env)
            return (
                participation
                * vanilla_price
                * terminal_timing.delay_df
                * multiplier
                + pending_pv
            )

        if obs_type == ObservationType.EXPIRY:
            value = self._price_expiry(
                product, pricing_env, spot, maturity, rate, div, vol
            )
            return value * terminal_timing.delay_df + pending_pv

        if obs_type == ObservationType.DISCRETE:
            schedule = product.observation_schedule
            if schedule is None or not schedule.records:
                raise PricingError(
                    "Discrete barrier monitoring requires ObservationSchedule."
                )
            schedule.assert_analytical_ready(default_payoff=product.rebate, pricing_env=pricing_env)
            freq = schedule.ensure_regular_frequency(schedule.times)
            shifted_barrier = apply_barrier_shift(
                barrier=forward_level,
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
                rebate=forward_cash,
            ) + pending_pv

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
            ) + pending_pv

        raise PricingError(f"Unsupported observation type: {obs_type}")

    def _resolved_records(self, product: BarrierOption, pricing_env: PricingEnvironment):
        return resolve_current_records(
            product.observation_schedule,
            pricing_env,
            default_barrier=product.barrier,
            default_payoff=product.rebate,
            require_single=True,
            product=product,
        )

    def _breached(self, product: BarrierOption, spot: float, level: float) -> bool:
        return spot >= level if product.is_up_barrier else spot <= level

    def _breached_at_valuation(
        self, product: BarrierOption, pricing_env: PricingEnvironment, spot: float, resolved=None
    ) -> bool:
        """Whether an observation at the valuation instant is breached by the spot, at its own level (``resolved``:
        the records already resolved for this price)."""
        if resolved is None:
            resolved = self._resolved_records(product, pricing_env)
        return any(self._breached(product, spot, rec.barrier) for rec in observed_at_valuation(resolved))

    def _price_at_maturity(
        self,
        product: BarrierOption,
        pricing_env: PricingEnvironment,
        spot: float,
        terminal_timing,
        knocked_in: bool = False,
        resolved=None,
    ) -> float:
        """The value at maturity: the carried knock-in state first, then what is observed now.

        ``knocked_in``: a knock-in carried as triggered (``knock_in_carried``, the state the T > 0 path prices as the
        knocked-in vanilla; ruling 2026-10-10) pays the payoff whatever is observed now. Otherwise continuous and
        expiry monitoring observe the spot at the product's barrier; a discrete schedule observes only its records at
        the valuation instant, each at its own level and paying its own cash (none observed: no hit). A knock-out hit
        pays its cash, a knock-in never knocked in its rebate, at the terminal payment time (hit-paid knock-out cash at
        the record's own payment time). ``resolved``: the records already resolved for this price.
        """
        multiplier = product.contract_multiplier
        payoff = product.get_payoff(spot) * product.participation_rate * terminal_timing.payment_df
        if product.observation_type == ObservationType.DISCRETE:
            if resolved is None:
                resolved = self._resolved_records(product, pricing_env)
            hits = [rec for rec in observed_at_valuation(resolved) if self._breached(product, spot, rec.barrier)]
            hit = hits[0] if hits else None
            if product.is_knock_in:
                return (payoff if (knocked_in or hit is not None)
                        else product.rebate * terminal_timing.payment_df * multiplier)
            if hit is None:
                return payoff
            cash_df = (
                pricing_env.get_discount_factor(hit.settlement_time)
                if product.pay_at_hit
                else terminal_timing.payment_df
            )
            return hit.payoff * cash_df * multiplier

        hit = product.is_barrier_hit(spot)
        if product.is_knock_in:
            return payoff if (knocked_in or hit) else product.rebate * terminal_timing.payment_df * multiplier
        if not hit:
            return payoff
        if product.pay_at_hit:
            return self._price_rebate_leg(product, pricing_env) * multiplier
        return product.rebate * terminal_timing.payment_df * multiplier

    def _price_through_first_observation(
        self,
        product: BarrierOption,
        pricing_env: PricingEnvironment,
        spot: float,
        maturity: float,
        rate: float,
        div: float,
        vol: float,
        terminal_timing,
        forward_level: float,
        rebate: float,
        resolved=None,
    ) -> float:
        """The value through the first future observation (``first_observation``), no observation deciding now,
        the spot inside or beyond the barrier.

        The breach at the first future observation ``t1`` pays the knock-out's rebate (at the hit, after the
        constant hit lag, or at the terminal payment time) or knocks the option in (Black-Scholes from ``t1``);
        otherwise the continuation is this engine's shifted formula from ``t1`` (no observation left: no barrier),
        on ``forward_level``, the level the remaining records share, a knock-out hit paying ``rebate``, their payoff
        (W18c). ``resolved``: the records already resolved for this price. The closures take an array of spots at
        ``t1``.
        """
        schedule = product.observation_schedule
        if schedule is None or not schedule.records:
            raise PricingError("Discrete barrier monitoring requires ObservationSchedule.")
        schedule.assert_analytical_ready(default_payoff=product.rebate, pricing_env=pricing_env)
        freq = schedule.ensure_regular_frequency(schedule.times)
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
            observation_interval=freq,
        )
        carry = rate - div
        delay = terminal_timing.delay_df
        participation = product.participation_rate
        is_call = product.is_call()
        terminal_df = math.exp(-rate * tau) * delay
        hit_lag = 0.0
        if product.is_knock_out and product.pay_at_hit and rebate > 0:
            hit_lag = constant_hit_lag_year_fraction(product, pricing_env, maturity)
        touch = self._one_touch_engine

        def vanilla(s: np.ndarray) -> np.ndarray:
            if is_zero(tau):
                return np.maximum(s - product.strike, 0.0) if is_call else np.maximum(product.strike - s, 0.0)
            d1, d2 = self._d1_d2(s, product.strike, tau, rate, div, vol)
            if is_call:
                return self._bs_engine._price_call(s, product.strike, tau, rate, div, d1, d2)
            return self._bs_engine._price_put(s, product.strike, tau, rate, div, d1, d2)

        def knock_out(s: np.ndarray) -> np.ndarray:
            return np.maximum(
                self._price_knock_out_closed_form(
                    spot=s,
                    strike=product.strike,
                    barrier=shifted,
                    maturity=tau,
                    rate=rate,
                    carry=carry,
                    vol=vol,
                    is_call=is_call,
                    is_up=product.is_up_barrier,
                ),
                0.0,
            )

        def touch_probability(s: np.ndarray) -> np.ndarray:
            return np.minimum(
                np.maximum(
                    touch._expiry_touch_term(
                        spot=s, barrier=shifted, maturity=tau, rate=rate, div=div, vol=vol,
                        is_up=product.is_up_barrier,
                    ),
                    0.0,
                ),
                1.0,
            )

        if product.is_knock_out:
            def hit_value(s: np.ndarray) -> float:
                if product.pay_at_hit:
                    return rebate * math.exp(-rate * hit_lag)
                return rebate * terminal_df

            def continuation(s: np.ndarray) -> np.ndarray:
                if first.remaining == 0:
                    return participation * vanilla(s) * delay
                value = participation * knock_out(s) * delay
                if rebate > 0:
                    if product.pay_at_hit:
                        value += rebate * touch._instant_touch_term(
                            spot=s, barrier=shifted, maturity=tau, rate=rate, div=div, vol=vol,
                            is_up=product.is_up_barrier,
                        ) * math.exp(-rate * hit_lag)
                    else:
                        value += rebate * terminal_df * touch._expiry_touch_term(
                            spot=s, barrier=shifted, maturity=tau, rate=rate, div=div, vol=vol,
                            is_up=product.is_up_barrier,
                        )
                return value
        else:
            def hit_value(s: np.ndarray) -> np.ndarray:
                return participation * vanilla(s) * delay

            def continuation(s: np.ndarray) -> np.ndarray:
                if first.remaining == 0:
                    return rebate * terminal_df
                value = participation * np.maximum(vanilla(s) - knock_out(s), 0.0) * delay
                if rebate > 0:
                    value += rebate * terminal_df * np.maximum(0.0, 1.0 - touch_probability(s))
                return value

        return through_first_observation(
            spot=spot,
            first=first,
            rate=rate,
            div=div,
            vol=vol,
            hit_value=hit_value,
            continuation=continuation,
            remaining_time=tau,
            hit_is_constant=product.is_knock_out,
            kinks=(product.strike,),      # the payoff's, with no time left after t1
        ) * product.contract_multiplier

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
        rebate: Optional[float] = None,
    ) -> float:
        """The closed form on ``barrier``; ``rebate``: the cash of the rebate leg (default ``product.rebate``; a
        DISCRETE knock-out's remaining records' payoff, W18c)."""
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

        rebate_val = self._price_rebate_leg(product, pricing_env, rebate=rebate)
        terminal_delay = resolve_terminal_timing(
            product,
            pricing_env,
        ).delay_df

        if product.is_knock_out:
            value = (
                participation * max(ko_price, 0.0) * terminal_delay
                + rebate_val
            )
            return value * product.contract_multiplier

        vanilla = EuropeanVanillaOption(
            strike=product.strike,
            option_type=product.option_type,
            maturity=product.maturity,
            exercise_date=product.exercise_date,
        )
        vanilla_price = self._bs_engine.price(vanilla, pricing_env)
        ki_price = max(vanilla_price - max(ko_price, 0.0), 0.0)
        value = participation * ki_price * terminal_delay + rebate_val
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
        """The continuously monitored knock-out (no rebate); ``spot`` a float or an array of spots (the same
        operations elementwise)."""
        array = isinstance(spot, np.ndarray)
        log = np.log if array else math.log
        power = pow_elementwise if array else (lambda base, exponent: base ** exponent)
        sqrt_t = math.sqrt(maturity)
        mu = (carry - 0.5 * vol * vol) / (vol * vol)

        log_s_k = log(spot / strike)
        log_s_b = log(spot / barrier)
        pow_term = power(barrier / spot, 2 * mu)
        pow_term_mu = power(barrier / spot, 2 * (mu + 1))

        x1 = log_s_k / (vol * sqrt_t) + (1 + mu) * vol * sqrt_t
        x2 = log_s_b / (vol * sqrt_t) + (1 + mu) * vol * sqrt_t
        y1 = (
            log((barrier * barrier) / (spot * strike)) / (vol * sqrt_t)
            + (1 + mu) * vol * sqrt_t
        )
        y2 = log(barrier / spot) / (vol * sqrt_t) + (1 + mu) * vol * sqrt_t
        no_value = np.zeros_like(spot) if array else 0.0

        eta = -1.0 if is_up else 1.0

        def A(phi: float) -> float:
            return phi * spot * math.exp((carry - rate) * maturity) * ndtr(
                phi * x1
            ) - phi * strike * math.exp(-rate * maturity) * ndtr(
                phi * x1 - phi * vol * sqrt_t
            )

        def B(phi: float) -> float:
            return phi * spot * math.exp((carry - rate) * maturity) * ndtr(
                phi * x2
            ) - phi * strike * math.exp(-rate * maturity) * ndtr(
                phi * x2 - phi * vol * sqrt_t
            )

        def C(phi: float) -> float:
            return phi * spot * math.exp(
                (carry - rate) * maturity
            ) * pow_term_mu * ndtr(eta * y1) - phi * strike * math.exp(
                -rate * maturity
            ) * pow_term * ndtr(
                eta * y1 - eta * vol * sqrt_t
            )

        def D(phi: float) -> float:
            return phi * spot * math.exp(
                (carry - rate) * maturity
            ) * pow_term_mu * ndtr(eta * y2) - phi * strike * math.exp(
                -rate * maturity
            ) * pow_term * ndtr(
                eta * y2 - eta * vol * sqrt_t
            )

        if is_call:
            if is_up:
                if barrier <= strike:
                    return no_value
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
        return no_value

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
        """``(d1, d2)``; ``spot`` a float or an array of spots."""
        array = isinstance(spot, np.ndarray)
        sqrt_t = math.sqrt(maturity)
        spot_invalid = bool(np.any(spot <= 0)) if array else spot <= 0
        if spot_invalid or strike <= 0 or vol <= 0 or maturity <= 0:
            raise ValidationError("Invalid inputs for d1/d2 calculation.")
        log = np.log if array else math.log
        d1 = (log(spot / strike) + (rate - div + 0.5 * vol * vol) * maturity) / (
            vol * sqrt_t
        )
        d2 = d1 - vol * sqrt_t
        return d1, d2

    def _price_rebate_leg(
        self, product: BarrierOption, pricing_env: PricingEnvironment, rebate: Optional[float] = None
    ) -> float:
        """Value rebate via one-touch/no-touch pricing for continuous/discrete (``rebate``: the cash, default
        ``product.rebate``; a knock-out decided now on a DISCRETE schedule pays its record's payoff, W18c)."""
        if rebate is None:
            rebate = product.rebate
        if rebate <= 0:
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
            settlement_convention=product.settlement_convention,
            rebate=rebate,
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

    def static_replication(self, product, pricing_env):
        """An ended schedule's remaining contract (spec 2026-10-07 theta boundary 3.3), else none."""
        if remaining_contract_applies(product):
            return remaining_replication(product, knock_in_carried(product))
        return super().static_replication(product, pricing_env)

    def __repr__(self):
        return "BarrierAnalyticalEngine()"
