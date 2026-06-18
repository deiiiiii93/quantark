"""
Analytical pricing engine for accumulator options.

The accumulator is statically replicated as a strip of up-and-out option legs,
one per observation date ``t_i``:

* a long up-and-out CALL struck at ``K`` (the linear gain leg), and
* a short, geared up-and-out PUT struck at ``K`` (the geared loss leg).

For :class:`AccumulatorKnockOutType.TERMINATION` both legs are knocked out on the
observation dates up to ``t_i`` (so a breach extinguishes all later accruals), and
an optional cash rebate is valued as a one-touch on the knock-out barrier. For
:class:`AccumulatorKnockOutType.SINGLE_DAY` only the call leg carries the barrier
(checked at ``t_i``) and the geared put leg is a plain vanilla, so a breach cancels
only that day's accrual.

Discrete daily monitoring inherits the Broadie-Glasserman-Kou barrier-shift
approximation implemented by the composed barrier and one-touch engines.
"""

from typing import List, Optional

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.param import EngineParams
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option import (
    AccumulatorOption,
    BarrierOption,
    EuropeanVanillaOption,
    OneTouchOption,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import (
    AccumulatorKnockOutType,
    BarrierDirection,
    BarrierType,
    ObservationType,
    OptionType,
    TouchType,
)
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import PricingError, ValidationError
from quantark.util.numerical import (
    safe_exp,
    validate_non_negative,
    validate_positive,
)

from .barrier_analytical_engine import BarrierAnalyticalEngine
from .black_scholes_engine import BlackScholesEngine
from .one_touch_analytical_engine import OneTouchAnalyticalEngine


class AccumulatorAnalyticalEngine(BaseEngine):
    """
    Closed-form decomposition engine for :class:`AccumulatorOption`.

    Composes :class:`BarrierAnalyticalEngine`, :class:`BlackScholesEngine`, and
    :class:`OneTouchAnalyticalEngine` to value each daily leg, the knock-out
    rebate, and the optional extra-shares-at-expiry leg.
    """

    engine_type = EngineType.ANALYTICAL

    MIN_VOL = 0.001
    MAX_VOL = 5.0
    MIN_MATURITY = 1e-10
    MAX_MATURITY = 50.0

    def __init__(self, params: Optional[EngineParams] = None):
        super().__init__(params)
        self._barrier_engine = BarrierAnalyticalEngine(params)
        self._bs_engine = BlackScholesEngine(params)
        self._one_touch_engine = OneTouchAnalyticalEngine(params)

    def price(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        """
        Price an accumulator option analytically.

        Args:
            product: An :class:`AccumulatorOption` instance.
            pricing_env: Market data environment.

        Returns:
            Present value of the accumulator.

        Raises:
            PricingError: If the product type is unsupported.
            ValidationError: If pricing inputs are invalid.
        """
        if not isinstance(product, AccumulatorOption):
            raise PricingError(
                "AccumulatorAnalyticalEngine only supports AccumulatorOption, "
                f"got {type(product).__name__}"
            )

        spot = pricing_env.spot
        strike = product.strike
        maturity = product.get_maturity(pricing_env)
        rate = pricing_env.get_rate(maturity)
        div = pricing_env.get_div_yield(maturity)
        vol = pricing_env.get_vol(strike, maturity)

        self._validate_inputs(spot, strike, maturity, rate, div, vol, product)

        # Realized accrual from past observations is locked in (rate-free part is
        # supplied by the product; discounting is applied here).
        realized = product.get_realized_accrual()
        if product.settlement_at_expiry and maturity >= self.MIN_MATURITY:
            realized *= safe_exp(-rate * maturity)

        if maturity < self.MIN_MATURITY:
            # At expiry only the locked-in accrual and the deterministic terminal
            # extra-shares leg remain.
            return realized + self._extra_shares_intrinsic(product, spot)

        times = product.get_observation_times()
        daily = product.daily_share_accumulation
        df_maturity = pricing_env.get_discount_factor(maturity)
        per_contract = 0.0
        for idx, t_i in enumerate(times):
            sub_dates = times[: idx + 1]
            leg = self._price_leg(product, pricing_env, sub_dates, t_i, daily)
            if product.settlement_at_expiry:
                # Defer the leg's payoff from t_i to T using curve discount
                # factors: DF(0, T) / DF(0, t_i) (curve-consistent, not flat-only).
                leg *= df_maturity / pricing_env.get_discount_factor(t_i)
            per_contract += leg

        # The rebate is conditional on a knock-out observation; with no accrual
        # fixings there is nothing to monitor.
        if times:
            per_contract += self._price_rebate_leg(product, pricing_env, times)
        per_contract += self._price_extra_shares_leg(
            product, pricing_env, times, maturity
        )

        value = per_contract * product.contract_multiplier + realized
        return value

    def _extra_shares_intrinsic(self, product: AccumulatorOption, spot: float) -> float:
        """Deterministic terminal value of the extra-shares leg at expiry."""
        extra = product.extra_shares_at_expiry
        if extra <= 0.0 or spot >= product.knock_out_barrier:
            return 0.0
        return -extra * max(product.strike - spot, 0.0) * product.contract_multiplier

    # ------------------------------------------------------------------
    # Leg pricing
    # ------------------------------------------------------------------

    def _price_leg(
        self,
        product: AccumulatorOption,
        pricing_env: PricingEnvironment,
        sub_dates: List[float],
        maturity_i: float,
        daily: float,
    ) -> float:
        """Value one observation's (long call - geared put) leg, per contract."""
        call_value = self._price_barrier_call(
            product, pricing_env, sub_dates, maturity_i, daily
        )
        put_value = self._price_put_leg(
            product, pricing_env, sub_dates, maturity_i, daily
        )
        return call_value - put_value

    def _price_barrier_call(
        self,
        product: AccumulatorOption,
        pricing_env: PricingEnvironment,
        sub_dates: List[float],
        maturity_i: float,
        daily: float,
    ) -> float:
        """Up-and-out call gain leg scaled by ``daily`` shares."""
        obs_type, obs_dates = self._call_leg_monitoring(product, sub_dates, maturity_i)
        call_leg = BarrierOption(
            strike=product.strike,
            option_type=OptionType.CALL,
            barrier=product.knock_out_barrier,
            barrier_type=BarrierType.UP_OUT,
            maturity=maturity_i,
            rebate=0.0,
            participation_rate=1.0,
            pay_at_hit=False,
            observation_type=obs_type,
            observation_dates=obs_dates,
            contract_multiplier=daily,
        )
        return self._barrier_engine.price(call_leg, pricing_env)

    def _price_put_leg(
        self,
        product: AccumulatorOption,
        pricing_env: PricingEnvironment,
        sub_dates: List[float],
        maturity_i: float,
        daily: float,
    ) -> float:
        """Geared loss leg: up-and-out put (TERMINATION) or vanilla put (SINGLE_DAY)."""
        geared = daily * product.gearing
        if product.knock_out_type == AccumulatorKnockOutType.SINGLE_DAY:
            # The lower put leg is unaffected by the upper knock-out barrier.
            put = EuropeanVanillaOption(
                strike=product.strike,
                option_type=OptionType.PUT,
                maturity=maturity_i,
                contract_multiplier=geared,
            )
            return self._bs_engine.price(put, pricing_env)

        obs_type, obs_dates = self._leg_monitoring(product, sub_dates)
        put_leg = BarrierOption(
            strike=product.strike,
            option_type=OptionType.PUT,
            barrier=product.knock_out_barrier,
            barrier_type=BarrierType.UP_OUT,
            maturity=maturity_i,
            rebate=0.0,
            participation_rate=1.0,
            pay_at_hit=False,
            observation_type=obs_type,
            observation_dates=obs_dates,
            contract_multiplier=geared,
        )
        return self._barrier_engine.price(put_leg, pricing_env)

    def _leg_monitoring(self, product: AccumulatorOption, sub_dates: List[float]):
        """Resolve barrier monitoring for a leg's observation sub-schedule.

        A single observation is monitored at expiry (a barrier shift needs at
        least two observation times); multiple observations are monitored
        discretely on the daily grid up to the leg's maturity.
        """
        if product.observation_type == ObservationType.CONTINUOUS:
            return ObservationType.CONTINUOUS, None
        if product.observation_type == ObservationType.EXPIRY:
            return ObservationType.EXPIRY, None
        if len(sub_dates) <= 1:
            return ObservationType.EXPIRY, None
        return ObservationType.DISCRETE, list(sub_dates)

    def _call_leg_monitoring(
        self, product: AccumulatorOption, sub_dates: List[float], maturity_i: float
    ):
        """Monitoring for the gain (call) leg of one observation.

        SINGLE_DAY only cancels that day's accrual, so the call leg's barrier is
        checked solely on its own observation date (expiry monitoring). For
        TERMINATION an earlier breach extinguishes the leg, so the cumulative
        observation sub-schedule is monitored.
        """
        if product.knock_out_type == AccumulatorKnockOutType.SINGLE_DAY:
            return ObservationType.EXPIRY, None
        return self._leg_monitoring(product, sub_dates)

    def _price_rebate_leg(
        self,
        product: AccumulatorOption,
        pricing_env: PricingEnvironment,
        times: List[float],
    ) -> float:
        """Value the TERMINATION knock-out cash rebate as a one-touch, per contract."""
        if product.knock_out_type != AccumulatorKnockOutType.TERMINATION:
            return 0.0
        rebate_cash = product.get_knock_out_rebate_cash()
        if rebate_cash <= 0.0:
            return 0.0

        obs_type, obs_dates = self._leg_monitoring(product, times)
        touch = OneTouchOption(
            barrier=product.knock_out_barrier,
            barrier_direction=BarrierDirection.UP,
            maturity=times[-1],
            rebate=rebate_cash,
            payment_at_hit=True,
            touch_type=TouchType.ONE_TOUCH,
            observation_type=obs_type,
            observation_dates=obs_dates,
        )
        return self._one_touch_engine.price(touch, pricing_env)

    def _price_extra_shares_leg(
        self,
        product: AccumulatorOption,
        pricing_env: PricingEnvironment,
        times: List[float],
        maturity: float,
    ) -> float:
        """Subtract the extra-shares-at-expiry up-and-out put leg, per contract."""
        extra = product.extra_shares_at_expiry
        if extra <= 0.0:
            return 0.0

        # The terminal extra-shares leg matures at the contract maturity and
        # follows the same knock-out treatment as a gain leg maturing at T:
        # expiry-only for SINGLE_DAY (no cumulative knockout), cumulative discrete
        # monitoring for TERMINATION.
        obs_type, obs_dates = self._call_leg_monitoring(product, times, maturity)
        put_leg = BarrierOption(
            strike=product.strike,
            option_type=OptionType.PUT,
            barrier=product.knock_out_barrier,
            barrier_type=BarrierType.UP_OUT,
            maturity=maturity,
            rebate=0.0,
            participation_rate=1.0,
            pay_at_hit=False,
            observation_type=obs_type,
            observation_dates=obs_dates,
            contract_multiplier=extra,
        )
        return -self._barrier_engine.price(put_leg, pricing_env)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_inputs(
        self,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        div: float,
        vol: float,
        product: AccumulatorOption,
    ) -> None:
        """Validate market and product inputs for analytical pricing."""
        validate_positive(spot, "spot")
        validate_positive(strike, "strike")
        validate_positive(product.knock_out_barrier, "knock_out_barrier")
        validate_non_negative(maturity, "maturity")
        validate_positive(vol, "volatility")
        validate_non_negative(product.gearing, "gearing")
        validate_non_negative(
            product.daily_share_accumulation, "daily_share_accumulation"
        )
        validate_positive(product.contract_multiplier, "contract_multiplier")

        if vol < self.MIN_VOL or vol > self.MAX_VOL:
            raise ValidationError(
                f"Volatility {vol} outside supported range "
                f"[{self.MIN_VOL}, {self.MAX_VOL}]"
            )
        if maturity > self.MAX_MATURITY:
            raise ValidationError(
                f"Maturity too long for analytical accumulator pricing: {maturity}"
            )
        if div < 0:
            raise ValidationError(f"Dividend yield must be non-negative, got {div}")
        if abs(rate) > 1.0:
            raise ValidationError(f"Risk-free rate outside reasonable bounds: {rate}")

    def __repr__(self) -> str:
        return "AccumulatorAnalyticalEngine()"
