"""
Analytical pricing engine for single sharkfin options.

The payoff is decomposed into:
- a no-rebate knock-out vanilla option,
- a one-touch cash leg for the knock-out rebate, and
- a no-touch cash leg for the no-hit rebate.

Discrete monitoring is handled by the underlying barrier and one-touch
analytical engines using the Broadie-Glasserman-Kou barrier shift.
"""

from dataclasses import replace
from typing import Optional

from quantark.asset.equity.engine.barrier_schedule_rules import forward_barrier_level, forward_knock_out_cash
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
from quantark.asset.equity.product.option import (
    BarrierOption,
    OneTouchOption,
    SingleSharkfinOption,
)
from quantark.asset.equity.product.option.observation_schedule import ObservationSchedule
from quantark.asset.equity.param import EngineParams
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import BarrierDirection, BarrierType, ObservationType, TouchType
from quantark.util.enum.engine_enums import DiscreteBarrierApproximation, EngineType
from quantark.util.exceptions import PricingError, ValidationError
from quantark.util.numerical import validate_non_negative, validate_positive

from .barrier_analytical_engine import BarrierAnalyticalEngine
from .first_observation import (
    observed_at_valuation,
    resolve_current_records,
    validate_discrete_barrier_approximation,
)
from .one_touch_analytical_engine import OneTouchAnalyticalEngine


class SingleSharkfinOptionAnalyticalEngine(BaseEngine):
    """
    Closed-form analytical engine for SingleSharkfinOption.

    Supports expiry-only and continuous monitoring exactly under Black-Scholes
    assumptions. Discrete monitoring uses the standard BGK continuity correction
    implemented by the composed barrier and one-touch engines, so daily
    observation schedules are approximated by shifting the barrier away from spot.
    Between observations the legs follow ``discrete_barrier_approximation``
    (``BGK``: the shifted closed form at today's spot on either side of the
    barrier; ``FIRST_OBSERVATION_BGK``: through the next observation); the
    composed engines are built with the same mode.
    """

    engine_type = EngineType.ANALYTICAL
    # params-only engine: instance state == construction arguments (patch spec 2026-09-03 §7)
    MODEL_FINGERPRINT_ATTRS = ('params', 'discrete_barrier_approximation', '_barrier_engine', '_one_touch_engine')
    settlement_support = SettlementSupport.EVENT_AND_TERMINAL
    supports_lifecycle_state = True

    MIN_MATURITY = 1e-10
    MAX_MATURITY = 50.0
    MIN_VOL = 0.001
    MAX_VOL = 5.0

    def __init__(
        self,
        params: Optional[EngineParams] = None,
        *,
        discrete_barrier_approximation: DiscreteBarrierApproximation = DiscreteBarrierApproximation.BGK,
    ):
        """
        Args:
            params: Engine configuration parameters.
            discrete_barrier_approximation: The value of a DISCRETE schedule between observations (see
                ``DiscreteBarrierApproximation``), handed to the knock-out and touch leg engines.
        """
        super().__init__(params)
        self.discrete_barrier_approximation = validate_discrete_barrier_approximation(discrete_barrier_approximation)
        self._barrier_engine = BarrierAnalyticalEngine(
            params, discrete_barrier_approximation=self.discrete_barrier_approximation
        )
        self._one_touch_engine = OneTouchAnalyticalEngine(
            params, discrete_barrier_approximation=self.discrete_barrier_approximation
        )

    def price(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        *,
        lifecycle_state=None,
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
        if product.observation_type == ObservationType.DISCRETE:
            # W18c (spec 2026-10-13 I4): hit now only by a record at the valuation instant, at its own level;
            # forward, the level every remaining record shares (None: they differ -- the legs refuse)
            hit_now = self._record_hit_at_valuation(product, pricing_env, spot) is not None
            forward_level = forward_barrier_level(product, pricing_env)
            # BGK: the legs price the closed form beyond the level too, so its hit lag is checked there as well
            hit_or_beyond = hit_now or (
                self.discrete_barrier_approximation is DiscreteBarrierApproximation.FIRST_OBSERVATION_BGK
                and forward_level is not None
                and (spot >= forward_level if self._barrier_direction(product) == BarrierDirection.UP
                     else spot <= forward_level)
            )
        else:
            hit_now = False
            hit_or_beyond = product.is_barrier_hit(spot)
        if (
            product.pay_at_hit
            and forward_knock_out_cash(product, pricing_env) > 0
            and product.observation_type != ObservationType.EXPIRY
            and not hit_or_beyond
        ):
            # Fail closed BEFORE pricing when the first-hit lag is not one constant;
            # the knock-out cash leg (one-touch engine) applies the exp(-r L) factor
            # itself (patch spec 2026-09-03 §5).
            constant_hit_lag_year_fraction(product, pricing_env, maturity)

        if maturity < self.MIN_MATURITY:
            return (
                self._price_at_maturity(product, pricing_env, spot, terminal_timing.payment_df)
                + pending_pv
            )

        if product.observation_type not in (
            ObservationType.EXPIRY,
            ObservationType.CONTINUOUS,
            ObservationType.DISCRETE,
        ):
            raise PricingError(
                f"Unsupported observation type: {product.observation_type}"
            )

        if product.observation_type == ObservationType.DISCRETE and not hit_now:
            # the legs' "no closed form" refusals on the sharkfin's own records (mixed payoffs or levels, A8): the
            # knock-out leg no longer carries the record payoffs, and a zero knock-out cash leg is skipped
            product.observation_schedule.assert_analytical_ready(
                default_payoff=product.knock_out_rebate, pricing_env=pricing_env)
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
        return max(value, 0.0) * product.contract_multiplier + pending_pv

    def _price_at_maturity(
        self,
        product: SingleSharkfinOption,
        pricing_env: PricingEnvironment,
        spot: float,
        terminal_payment_df: float,
    ) -> float:
        """The value at maturity: a discrete schedule is breached only by its records at the valuation instant,
        each at its own level and paying its own cash; continuous and expiry monitoring observe the spot."""
        if product.observation_type != ObservationType.DISCRETE:
            return product.get_payoff(spot) * terminal_payment_df
        hit = self._record_hit_at_valuation(product, pricing_env, spot)
        if hit is None:
            return product.get_no_hit_payoff(spot) * terminal_payment_df
        return hit.payoff * terminal_payment_df * product.contract_multiplier

    def _record_hit_at_valuation(
        self, product: SingleSharkfinOption, pricing_env: PricingEnvironment, spot: float
    ):
        """The first record at the valuation instant the spot breaches at its own level, or None."""
        records = observed_at_valuation(
            resolve_current_records(
                product.observation_schedule,
                pricing_env,
                default_barrier=product.barrier,
                default_payoff=product.knock_out_rebate,
                require_single=True,
                product=product,
            )
        )
        up = self._barrier_direction(product) == BarrierDirection.UP
        hits = [rec for rec in records if (spot >= rec.barrier if up else spot <= rec.barrier)]
        return hits[0] if hits else None

    def _price_no_rebate_knock_out(
        self, product: SingleSharkfinOption, pricing_env: PricingEnvironment
    ) -> float:
        """Value the capped sharkfin participation leg as a knock-out option: no cash on a hit, so its records pay
        nothing (the one-touch leg pays the knock-out cash; a record paying it here would count it twice, W18c)."""
        schedule = product.observation_schedule
        if schedule is not None:
            schedule = ObservationSchedule(
                records=[replace(rec, payoff=0.0) for rec in schedule.records],
                aggregation_mode=schedule.aggregation_mode,
                frequency=schedule.frequency,
                monitoring_ended=schedule.monitoring_ended,
            )
        barrier_option = BarrierOption(
            strike=product.strike,
            option_type=product.option_type,
            barrier=product.barrier,
            barrier_type=self._barrier_type(product),
            maturity=product.maturity,
            exercise_date=product.exercise_date,
            settlement_date=product.settlement_date,
            settlement_convention=product.settlement_convention,
            rebate=0.0,
            participation_rate=1.0,
            pay_at_hit=False,
            observation_type=product.observation_type,
            observation_dates=product.observation_dates,
            observation_schedule=schedule,
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
        """Value a fixed cash leg conditional on touch or no-touch. A DISCRETE touch pays the touching record's
        payoff (the one-touch engine reads it, W18c): the knock-out cash leg is skipped only when neither the product
        nor any record pays cash."""
        if rebate <= 0.0 and not (
            touch_type == TouchType.ONE_TOUCH
            and product.observation_type == ObservationType.DISCRETE
            and any(rec.payoff is not None and rec.payoff > 0.0 for rec in product.observation_schedule.records)
        ):
            return 0.0

        touch_option = OneTouchOption(
            barrier=product.barrier,
            barrier_direction=self._barrier_direction(product),
            maturity=product.maturity,
            exercise_date=product.exercise_date,
            settlement_date=product.settlement_date,
            settlement_convention=product.settlement_convention,
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
        if abs(rate) > 1.0:
            raise ValidationError(f"Risk-free rate outside reasonable bounds: {rate}")

    def static_replication(self, product, pricing_env):
        """An ended schedule's remaining contract (spec 2026-10-07 theta boundary 3.3), else none."""
        if monitoring_ended(product):
            return remaining_replication(product)
        return super().static_replication(product, pricing_env)

    def __repr__(self):
        if self.discrete_barrier_approximation is DiscreteBarrierApproximation.BGK:
            return "SingleSharkfinOptionAnalyticalEngine()"
        return ("SingleSharkfinOptionAnalyticalEngine("
                f"discrete_barrier_approximation={self.discrete_barrier_approximation.name})")
