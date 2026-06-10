"""
Adapter layer for building quadrature core inputs from product types.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence, TYPE_CHECKING

import numpy as np

from asset.equity.engine.quad.quad_core import QuadCoreInputs
from asset.equity.product.base_equity_product import BaseEquityProduct
from asset.equity.product.option import (
    BarrierOption,
    DoubleBarrierOption,
    DoubleOneTouchOption,
    EuropeanVanillaOption,
    OneTouchOption,
)
from asset.equity.product.option.observation_schedule import (
    ObservationSchedule,
    ResolvedObservationRecord,
)
from priceenv import PricingEnvironment
from util.enum import ObservationAggregation, ObservationType, TouchType
from util.exceptions import PricingError, ValidationError
from util.numerical import (
    Tolerance,
    is_close,
    is_zero,
    validate_non_negative,
    validate_positive,
)

if TYPE_CHECKING:
    from asset.equity.engine.quad.discrete_quad_engine import DiscreteQuadEngine


@dataclass(frozen=True)
class QuadPricingContext:
    """Shared pricing context for discrete quadrature adapters."""

    spot: float
    maturity: float
    rate: float
    div: float
    vol: float
    contract_multiplier: float


class QuadInputAdapter(ABC):
    """Interface for building quad core inputs from a product."""

    @abstractmethod
    def can_handle(self, product: BaseEquityProduct) -> bool:
        """Return True if adapter can handle the product."""

    @abstractmethod
    def build_pricing_context(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: "DiscreteQuadEngine",
    ) -> QuadPricingContext:
        """Build shared pricing context for the product."""

    @abstractmethod
    def early_price(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        context: QuadPricingContext,
        engine: "DiscreteQuadEngine",
    ) -> float | None:
        """Return a price if no core evaluation is needed; otherwise None."""

    @abstractmethod
    def resolve_schedule(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        context: QuadPricingContext,
    ) -> Sequence[ResolvedObservationRecord]:
        """Resolve observation schedule into concrete records."""

    @abstractmethod
    def build_inputs(
        self,
        product: BaseEquityProduct,
        resolved: Sequence[ResolvedObservationRecord],
        context: QuadPricingContext,
    ) -> QuadCoreInputs:
        """Construct quad core inputs for the product."""

    @abstractmethod
    def finalize_price(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        context: QuadPricingContext,
        core_price: float,
        engine: "DiscreteQuadEngine",
    ) -> float:
        """Post-process the core output into a final product price."""


class BaseDiscreteQuadAdapter(QuadInputAdapter):
    """Base adapter with shared schedule handling helpers."""

    def resolve_schedule(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        context: QuadPricingContext,
    ) -> Sequence[ResolvedObservationRecord]:
        maturity = context.maturity
        default_payoff = self._default_payoff(product)

        if product.observation_type == ObservationType.EXPIRY:
            schedule = ObservationSchedule.from_legacy(
                observation_dates=[maturity],
                default_barrier=product.barrier,
                default_payoff=default_payoff,
                aggregation_mode=ObservationAggregation.STOP_FIRST_HIT,
            )
        else:
            schedule = product.observation_schedule
            if schedule is None and product.observation_dates:
                schedule = ObservationSchedule.from_legacy(
                    observation_dates=product.observation_dates,
                    default_barrier=product.barrier,
                    default_payoff=default_payoff,
                    aggregation_mode=ObservationAggregation.STOP_FIRST_HIT,
                )

        if schedule is None or not schedule.records:
            raise PricingError("Discrete monitoring requires ObservationSchedule.")
        if schedule.aggregation_mode != ObservationAggregation.STOP_FIRST_HIT:
            raise PricingError("DiscreteQuadEngine requires STOP_FIRST_HIT aggregation.")

        resolved = schedule.resolve(
            pricing_env,
            default_barrier=product.barrier,
            default_payoff=default_payoff,
            require_single=True,
        )
        return resolved

    def _default_payoff(self, product: BaseEquityProduct) -> float:
        return float(getattr(product, "rebate", 0.0))

    def _extract_observations(
        self,
        resolved: Sequence[ResolvedObservationRecord],
        maturity: float,
        is_up_barrier: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obs_times = np.array([rec.observation_time for rec in resolved], dtype=float)
        barriers = np.array([rec.barrier for rec in resolved], dtype=float)
        payoffs = np.array([rec.payoff for rec in resolved], dtype=float)
        settlement_times = np.array(
            [
                rec.settlement_time if rec.settlement_time is not None else rec.observation_time
                for rec in resolved
            ],
            dtype=float,
        )

        if obs_times.size == 0:
            raise ValidationError("Resolved observation schedule is empty.")

        if maturity - obs_times[-1] > Tolerance.ZERO:
            obs_times = np.append(obs_times, maturity)
            barriers = np.append(barriers, math.inf if is_up_barrier else 0.0)
            payoffs = np.append(payoffs, 0.0)
            settlement_times = np.append(settlement_times, maturity)

        return obs_times, barriers, payoffs, settlement_times

    def _discount_rebates(
        self,
        payoffs: np.ndarray,
        observation_times: np.ndarray,
        settlement_times: np.ndarray,
        maturity: float,
        rate: float,
        pay_at_hit: bool,
    ) -> np.ndarray:
        if payoffs.size == 0:
            return payoffs
        if pay_at_hit:
            delays = np.maximum(settlement_times - observation_times, 0.0)
            discount = np.exp(-rate * delays)
        else:
            discount = np.exp(-rate * (maturity - observation_times))
        return payoffs * discount


class BarrierQuadInputAdapter(BaseDiscreteQuadAdapter):
    """Adapter for barrier options in discrete quadrature."""

    def can_handle(self, product: BaseEquityProduct) -> bool:
        return isinstance(product, BarrierOption)

    def build_pricing_context(
        self,
        product: BarrierOption,
        pricing_env: PricingEnvironment,
        engine: "DiscreteQuadEngine",
    ) -> QuadPricingContext:
        spot = pricing_env.spot
        maturity = product.get_maturity(pricing_env)
        rate = pricing_env.get_rate(maturity)
        div = pricing_env.get_div_yield(maturity)
        vol = pricing_env.get_vol(product.strike, maturity)
        contract_multiplier = product.contract_multiplier

        self._validate_inputs(
            spot, product.strike, maturity, rate, div, vol, product.barrier, engine
        )

        return QuadPricingContext(
            spot=spot,
            maturity=maturity,
            rate=rate,
            div=div,
            vol=vol,
            contract_multiplier=contract_multiplier,
        )

    def early_price(
        self,
        product: BarrierOption,
        pricing_env: PricingEnvironment,
        context: QuadPricingContext,
        engine: "DiscreteQuadEngine",
    ) -> float | None:
        if context.maturity < engine.MIN_MATURITY:
            value = self._barrier_expired_price(
                product, context.spot, context.rate, context.maturity
            )
            return value * context.contract_multiplier

        if (
            product.observation_type != ObservationType.EXPIRY
            and product.is_barrier_hit(context.spot)
        ):
            if product.is_knock_out:
                value = self._barrier_immediate_ko_price(
                    product, context.rate, context.maturity
                )
                return value * context.contract_multiplier
            value = self._vanilla_price(product, pricing_env, engine)
            return value * context.contract_multiplier

        if product.observation_type == ObservationType.CONTINUOUS:
            raise PricingError(
                "DiscreteQuadEngine supports discrete monitoring only. "
                "Use BarrierAnalyticalEngine for continuous barriers."
            )

        return None

    def resolve_schedule(
        self,
        product: BarrierOption,
        pricing_env: PricingEnvironment,
        context: QuadPricingContext,
    ) -> Sequence[ResolvedObservationRecord]:
        return super().resolve_schedule(product, pricing_env, context)

    def build_inputs(
        self,
        product: BarrierOption,
        resolved: Sequence[ResolvedObservationRecord],
        context: QuadPricingContext,
    ) -> QuadCoreInputs:
        obs_times, barriers, payoffs, settlement_times = self._extract_observations(
            resolved, context.maturity, product.is_up_barrier
        )

        if product.is_up_barrier:
            k_plus = np.array(barriers, dtype=float)
            k_minus = np.zeros_like(k_plus)
        else:
            k_minus = np.array(barriers, dtype=float)
            k_plus = np.full_like(k_minus, math.inf, dtype=float)

        a_minus = np.zeros_like(k_minus, dtype=float)
        b_minus = np.zeros_like(k_minus, dtype=float)
        a_plus = np.zeros_like(k_plus, dtype=float)
        b_plus = np.zeros_like(k_plus, dtype=float)

        if product.rebate > 0.0:
            rebate_values = self._discount_rebates(
                payoffs,
                obs_times,
                settlement_times,
                context.maturity,
                context.rate,
                product.pay_at_hit,
            )
            if product.is_up_barrier:
                b_plus = rebate_values
            else:
                b_minus = rebate_values

        maturity_barrier = self._resolve_maturity_barrier(
            obs_times, barriers, context.maturity
        )
        a_terminal, b_terminal = self._apply_terminal_payoff_structure(
            product,
            maturity_barrier,
            k_minus,
            k_plus,
            a_minus,
            b_minus,
            a_plus,
            b_plus,
        )

        return QuadCoreInputs(
            observation_times=obs_times,
            k_minus=k_minus,
            k_plus=k_plus,
            a_minus=a_minus,
            b_minus=b_minus,
            a_plus=a_plus,
            b_plus=b_plus,
            a_terminal=a_terminal,
            b_terminal=b_terminal,
        )

    def finalize_price(
        self,
        product: BarrierOption,
        pricing_env: PricingEnvironment,
        context: QuadPricingContext,
        core_price: float,
        engine: "DiscreteQuadEngine",
    ) -> float:
        if product.is_knock_in:
            vanilla_price = self._vanilla_price(product, pricing_env, engine)
            rebate_discount = product.rebate * math.exp(-context.rate * context.maturity)
            value = vanilla_price + rebate_discount - core_price
            return value * context.contract_multiplier
        return core_price * context.contract_multiplier

    def _validate_inputs(
        self,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        div: float,
        vol: float,
        barrier: float,
        engine: "DiscreteQuadEngine",
    ) -> None:
        if spot <= 0:
            raise ValidationError(f"Spot price must be positive, got {spot}")
        if strike <= 0:
            raise ValidationError(f"Strike price must be positive, got {strike}")
        if maturity < 0:
            raise ValidationError(f"Time to maturity must be non-negative, got {maturity}")
        if vol <= 0:
            raise ValidationError(f"Volatility must be positive, got {vol}")
        if barrier <= 0:
            raise ValidationError(f"Barrier must be positive, got {barrier}")
        if div < 0:
            raise ValidationError(f"Dividend yield must be non-negative, got {div}")
        if abs(rate) > 1.0:
            raise ValidationError(f"Risk-free rate outside reasonable bounds: {rate}")
        if vol > engine.MAX_VOL:
            raise ValidationError(f"Volatility too high for quadrature stability: {vol}")

    def _resolve_maturity_barrier(
        self, observation_times: np.ndarray, barriers: np.ndarray, maturity: float
    ) -> float | None:
        maturity_mask = np.isclose(
            observation_times, maturity, atol=Tolerance.ZERO, rtol=0.0
        )
        if not np.any(maturity_mask):
            return None
        value = float(barriers[np.where(maturity_mask)[0][-1]])
        if not math.isfinite(value) or value <= 0.0:
            return None
        return value

    def _apply_terminal_payoff_structure(
        self,
        product: BarrierOption,
        maturity_barrier: float | None,
        k_minus: np.ndarray,
        k_plus: np.ndarray,
        a_minus: np.ndarray,
        b_minus: np.ndarray,
        a_plus: np.ndarray,
        b_plus: np.ndarray,
    ) -> tuple[float, float]:
        participation = product.participation_rate
        strike = product.strike
        is_call = product.is_call()
        a_terminal = 0.0
        b_terminal = 0.0

        if maturity_barrier is None:
            if is_call:
                k_minus[-1] = strike
                a_terminal = participation
                b_terminal = -participation * strike
            else:
                k_plus[-1] = strike
                a_terminal = -participation
                b_terminal = participation * strike
            return a_terminal, b_terminal

        barrier = maturity_barrier

        if product.is_up_barrier:
            k_plus[-1] = barrier
            if is_call:
                if barrier > strike:
                    k_minus[-1] = strike
                    a_terminal = participation
                    b_terminal = -participation * strike
            else:
                if barrier > strike:
                    k_minus[-1] = strike
                    a_minus[-1] = -participation
                    b_minus[-1] = participation * strike
                else:
                    a_terminal = -participation
                    b_terminal = participation * strike
            return a_terminal, b_terminal

        k_minus[-1] = barrier
        if is_call:
            if barrier < strike:
                k_plus[-1] = strike
                a_plus[-1] = participation
                b_plus[-1] = -participation * strike
            else:
                a_terminal = participation
                b_terminal = -participation * strike
        else:
            if barrier < strike:
                k_plus[-1] = strike
                a_terminal = -participation
                b_terminal = participation * strike

        return a_terminal, b_terminal

    def _barrier_expired_price(
        self, product: BarrierOption, spot: float, rate: float, maturity: float
    ) -> float:
        hit = product.is_barrier_hit(spot)
        if product.is_call():
            intrinsic = max(spot - product.strike, 0.0)
        else:
            intrinsic = max(product.strike - spot, 0.0)
        vanilla = intrinsic * product.participation_rate
        if product.is_knock_out:
            value = product.rebate if hit else vanilla
        else:
            value = vanilla if hit else product.rebate
        if maturity <= 0.0:
            return value
        return value * math.exp(-rate * maturity)

    def _barrier_immediate_ko_price(
        self, product: BarrierOption, rate: float, maturity: float
    ) -> float:
        if product.pay_at_hit:
            return product.rebate
        return product.rebate * math.exp(-rate * maturity)

    def _vanilla_price(
        self,
        product: BarrierOption,
        pricing_env: PricingEnvironment,
        engine: "DiscreteQuadEngine",
    ) -> float:
        vanilla = EuropeanVanillaOption(
            strike=product.strike,
            option_type=product.option_type,
            maturity=product.maturity,
            exercise_date=product.exercise_date,
            settlement_date=product.settlement_date,
            contract_multiplier=1.0,
        )
        price = engine.vanilla_engine.price(vanilla, pricing_env)
        return price * product.participation_rate


class OneTouchQuadInputAdapter(BaseDiscreteQuadAdapter):
    """Adapter for one-touch/no-touch options in discrete quadrature."""

    def can_handle(self, product: BaseEquityProduct) -> bool:
        return isinstance(product, OneTouchOption)

    def build_pricing_context(
        self,
        product: OneTouchOption,
        pricing_env: PricingEnvironment,
        engine: "DiscreteQuadEngine",
    ) -> QuadPricingContext:
        spot = pricing_env.spot
        maturity = product.get_maturity(pricing_env)
        rate = pricing_env.get_rate(maturity)
        div = pricing_env.get_div_yield(maturity)
        vol = pricing_env.get_vol(product.barrier, maturity)
        contract_multiplier = getattr(product, "contract_multiplier", 1.0)

        self._validate_inputs(
            spot, product.barrier, maturity, rate, div, vol, product.rebate, engine
        )

        return QuadPricingContext(
            spot=spot,
            maturity=maturity,
            rate=rate,
            div=div,
            vol=vol,
            contract_multiplier=contract_multiplier,
        )

    def early_price(
        self,
        product: OneTouchOption,
        pricing_env: PricingEnvironment,
        context: QuadPricingContext,
        engine: "DiscreteQuadEngine",
    ) -> float | None:
        if is_zero(context.maturity, tol=engine.MIN_MATURITY):
            pay_at_hit = product.payment_at_hit if product.is_one_touch else False
            value = self._one_touch_instant_payoff(
                product, context.spot, context.maturity, context.rate, pay_at_hit
            )
            return value * context.contract_multiplier

        if (
            product.observation_type != ObservationType.EXPIRY
            and product.is_barrier_hit(context.spot)
        ):
            if product.is_one_touch:
                pay_at_hit = product.payment_at_hit
                value = product.rebate if pay_at_hit else product.rebate * math.exp(
                    -context.rate * context.maturity
                )
                return value * context.contract_multiplier
            return 0.0

        if product.observation_type == ObservationType.CONTINUOUS:
            raise PricingError(
                "DiscreteQuadEngine supports discrete or expiry monitoring only. "
                "Use OneTouchAnalyticalEngine for continuous monitoring."
            )

        if product.rebate <= 0.0:
            return 0.0

        return None

    def resolve_schedule(
        self,
        product: OneTouchOption,
        pricing_env: PricingEnvironment,
        context: QuadPricingContext,
    ) -> Sequence[ResolvedObservationRecord]:
        return super().resolve_schedule(product, pricing_env, context)

    def build_inputs(
        self,
        product: OneTouchOption,
        resolved: Sequence[ResolvedObservationRecord],
        context: QuadPricingContext,
    ) -> QuadCoreInputs:
        obs_times, barriers, payoffs, settlement_times = self._extract_observations(
            resolved, context.maturity, product.is_up_barrier
        )

        if product.is_up_barrier:
            k_plus = np.array(barriers, dtype=float)
            k_minus = np.zeros_like(k_plus)
        else:
            k_minus = np.array(barriers, dtype=float)
            k_plus = np.full_like(k_minus, math.inf, dtype=float)

        a_minus = np.zeros_like(k_minus, dtype=float)
        b_minus = np.zeros_like(k_minus, dtype=float)
        a_plus = np.zeros_like(k_plus, dtype=float)
        b_plus = np.zeros_like(k_plus, dtype=float)

        if product.rebate > 0.0:
            rebate_values = self._discount_rebates(
                payoffs,
                obs_times,
                settlement_times,
                context.maturity,
                context.rate,
                product.payment_at_hit,
            )
            if product.is_up_barrier:
                b_plus = rebate_values
            else:
                b_minus = rebate_values

        return QuadCoreInputs(
            observation_times=obs_times,
            k_minus=k_minus,
            k_plus=k_plus,
            a_minus=a_minus,
            b_minus=b_minus,
            a_plus=a_plus,
            b_plus=b_plus,
            a_terminal=0.0,
            b_terminal=0.0,
        )

    def finalize_price(
        self,
        product: OneTouchOption,
        pricing_env: PricingEnvironment,
        context: QuadPricingContext,
        core_price: float,
        engine: "DiscreteQuadEngine",
    ) -> float:
        if product.is_no_touch:
            rebate_discount = product.rebate * math.exp(-context.rate * context.maturity)
            value = rebate_discount - core_price
            return max(0.0, value) * context.contract_multiplier
        return core_price * context.contract_multiplier

    def _validate_inputs(
        self,
        spot: float,
        barrier: float,
        maturity: float,
        rate: float,
        div: float,
        vol: float,
        rebate: float,
        engine: "DiscreteQuadEngine",
    ) -> None:
        validate_positive(spot, "spot")
        validate_positive(barrier, "barrier")
        validate_positive(vol, "volatility")
        validate_positive(maturity, "maturity", allow_zero=True)
        validate_non_negative(rebate, "rebate")
        validate_non_negative(div, "dividend_yield")

        if abs(rate) > 1.0:
            raise ValidationError(f"Risk-free rate outside reasonable bounds: {rate}")
        if vol > engine.MAX_VOL:
            raise ValidationError(f"Volatility too high for quadrature stability: {vol}")

    def _one_touch_instant_payoff(
        self,
        product: OneTouchOption,
        spot: float,
        maturity: float,
        rate: float,
        pay_at_hit: bool,
    ) -> float:
        touched = product.is_barrier_hit(spot)
        discount = math.exp(-rate * maturity)
        if product.is_one_touch:
            if touched:
                return product.rebate if pay_at_hit else product.rebate * discount
            return 0.0
        return product.rebate * discount if not touched else 0.0


class DoubleBarrierQuadInputAdapter(BaseDiscreteQuadAdapter):
    """Adapter for double barrier options in discrete quadrature."""

    def can_handle(self, product: BaseEquityProduct) -> bool:
        return isinstance(product, DoubleBarrierOption)

    def build_pricing_context(
        self,
        product: DoubleBarrierOption,
        pricing_env: PricingEnvironment,
        engine: "DiscreteQuadEngine",
    ) -> QuadPricingContext:
        spot = pricing_env.spot
        maturity = product.get_maturity(pricing_env)
        rate = pricing_env.get_rate(maturity)
        div = pricing_env.get_div_yield(maturity)
        vol = pricing_env.get_vol(product.strike, maturity)
        contract_multiplier = product.contract_multiplier

        self._validate_inputs(
            spot,
            product.strike,
            maturity,
            rate,
            div,
            vol,
            product.lower_barrier,
            product.upper_barrier,
            engine,
        )

        return QuadPricingContext(
            spot=spot,
            maturity=maturity,
            rate=rate,
            div=div,
            vol=vol,
            contract_multiplier=contract_multiplier,
        )

    def early_price(
        self,
        product: DoubleBarrierOption,
        pricing_env: PricingEnvironment,
        context: QuadPricingContext,
        engine: "DiscreteQuadEngine",
    ) -> float | None:
        if is_zero(context.maturity, tol=engine.MIN_MATURITY):
            value = self._double_barrier_expired_price(
                product, context.spot, context.rate, context.maturity
            )
            return value * context.contract_multiplier

        if (
            product.observation_type != ObservationType.EXPIRY
            and product.is_barrier_hit(context.spot)
        ):
            if product.is_knock_out:
                rebate_discount = math.exp(-context.rate * context.maturity)
                return product.rebate * rebate_discount * context.contract_multiplier
            vanilla_price = self._vanilla_price(product, pricing_env, engine)
            return vanilla_price * context.contract_multiplier

        if product.observation_type == ObservationType.CONTINUOUS:
            raise PricingError(
                "DiscreteQuadEngine supports discrete monitoring only. "
                "Use a continuous barrier engine for continuous barriers."
            )

        return None

    def resolve_schedule(
        self,
        product: DoubleBarrierOption,
        pricing_env: PricingEnvironment,
        context: QuadPricingContext,
    ) -> Sequence[ResolvedObservationRecord]:
        if product.observation_type == ObservationType.EXPIRY:
            schedule = ObservationSchedule.from_legacy(
                observation_dates=[context.maturity],
                default_barrier=None,
                default_payoff=product.rebate,
                aggregation_mode=ObservationAggregation.STOP_FIRST_HIT,
                upper_barrier=product.upper_barrier,
                lower_barrier=product.lower_barrier,
            )
        else:
            schedule = product.observation_schedule
            if schedule is None and product.observation_dates:
                schedule = ObservationSchedule.from_legacy(
                    observation_dates=product.observation_dates,
                    default_barrier=None,
                    default_payoff=product.rebate,
                    aggregation_mode=ObservationAggregation.STOP_FIRST_HIT,
                    upper_barrier=product.upper_barrier,
                    lower_barrier=product.lower_barrier,
                )

        if schedule is None or not schedule.records:
            raise PricingError("Discrete monitoring requires ObservationSchedule.")
        if schedule.aggregation_mode != ObservationAggregation.STOP_FIRST_HIT:
            raise PricingError("DiscreteQuadEngine requires STOP_FIRST_HIT aggregation.")

        return schedule.resolve(
            pricing_env,
            default_upper=product.upper_barrier,
            default_lower=product.lower_barrier,
            default_payoff=product.rebate,
            require_double=True,
        )

    def build_inputs(
        self,
        product: DoubleBarrierOption,
        resolved: Sequence[ResolvedObservationRecord],
        context: QuadPricingContext,
    ) -> QuadCoreInputs:
        (
            obs_times,
            upper_barriers,
            lower_barriers,
            payoffs,
            settlement_times,
        ) = self._extract_double_observations(resolved, context.maturity)

        k_plus = np.array(upper_barriers, dtype=float)
        k_minus = np.array(lower_barriers, dtype=float)
        a_minus = np.zeros_like(k_minus, dtype=float)
        b_minus = np.zeros_like(k_minus, dtype=float)
        a_plus = np.zeros_like(k_plus, dtype=float)
        b_plus = np.zeros_like(k_plus, dtype=float)

        if np.any(payoffs > 0.0):
            pay_at_hit = self._pay_at_hit(settlement_times, context.maturity)
            rebate_values = self._discount_rebates(
                payoffs,
                obs_times,
                settlement_times,
                context.maturity,
                context.rate,
                pay_at_hit,
            )
            b_minus = rebate_values
            b_plus = rebate_values

        a_terminal, b_terminal = self._terminal_payoff_coefficients(
            product, k_minus, k_plus
        )

        return QuadCoreInputs(
            observation_times=obs_times,
            k_minus=k_minus,
            k_plus=k_plus,
            a_minus=a_minus,
            b_minus=b_minus,
            a_plus=a_plus,
            b_plus=b_plus,
            a_terminal=a_terminal,
            b_terminal=b_terminal,
        )

    def finalize_price(
        self,
        product: DoubleBarrierOption,
        pricing_env: PricingEnvironment,
        context: QuadPricingContext,
        core_price: float,
        engine: "DiscreteQuadEngine",
    ) -> float:
        if product.is_knock_in:
            vanilla_price = self._vanilla_price(product, pricing_env, engine)
            rebate_discount = product.rebate * math.exp(-context.rate * context.maturity)
            value = vanilla_price + rebate_discount - core_price
            return value * context.contract_multiplier
        return core_price * context.contract_multiplier

    def _extract_double_observations(
        self, resolved: Sequence[ResolvedObservationRecord], maturity: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obs_times = np.array([rec.observation_time for rec in resolved], dtype=float)
        upper = np.array([rec.upper_barrier for rec in resolved], dtype=float)
        lower = np.array([rec.lower_barrier for rec in resolved], dtype=float)
        payoffs = np.array([rec.payoff for rec in resolved], dtype=float)
        settlement_times = np.array(
            [
                rec.settlement_time if rec.settlement_time is not None else rec.observation_time
                for rec in resolved
            ],
            dtype=float,
        )

        if obs_times.size == 0:
            raise ValidationError("Resolved observation schedule is empty.")

        if maturity - obs_times[-1] > Tolerance.ZERO:
            obs_times = np.append(obs_times, maturity)
            upper = np.append(upper, math.inf)
            lower = np.append(lower, 0.0)
            payoffs = np.append(payoffs, 0.0)
            settlement_times = np.append(settlement_times, maturity)

        return obs_times, upper, lower, payoffs, settlement_times

    def _terminal_payoff_coefficients(
        self,
        product: DoubleBarrierOption,
        k_minus: np.ndarray,
        k_plus: np.ndarray,
    ) -> tuple[float, float]:
        strike = product.strike
        lower = float(k_minus[-1])
        upper = float(k_plus[-1])

        if product.is_call():
            if strike <= lower:
                return 1.0, -strike
            if strike >= upper:
                return 0.0, 0.0
            k_minus[-1] = strike
            return 1.0, -strike

        if strike >= upper:
            return -1.0, strike
        if strike <= lower:
            return 0.0, 0.0
        k_plus[-1] = strike
        return -1.0, strike

    def _double_barrier_expired_price(
        self,
        product: DoubleBarrierOption,
        spot: float,
        rate: float,
        maturity: float,
    ) -> float:
        hit = product.is_barrier_hit(spot)
        if product.is_call():
            intrinsic = max(spot - product.strike, 0.0)
        else:
            intrinsic = max(product.strike - spot, 0.0)
        if product.is_knock_out:
            value = product.rebate if hit else intrinsic
        else:
            value = intrinsic if hit else product.rebate
        if maturity <= 0.0:
            return value
        return value * math.exp(-rate * maturity)

    def _pay_at_hit(self, settlement_times: np.ndarray, maturity: float) -> bool:
        if settlement_times.size == 0:
            return False
        return not np.all(
            [is_close(t, maturity, abs_tol=Tolerance.PRECISION) for t in settlement_times]
        )

    def _vanilla_price(
        self,
        product: DoubleBarrierOption,
        pricing_env: PricingEnvironment,
        engine: "DiscreteQuadEngine",
    ) -> float:
        vanilla = EuropeanVanillaOption(
            strike=product.strike,
            option_type=product.option_type,
            maturity=product.maturity,
            exercise_date=product.exercise_date,
            settlement_date=product.settlement_date,
            contract_multiplier=1.0,
        )
        return engine.vanilla_engine.price(vanilla, pricing_env)

    def _validate_inputs(
        self,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        div: float,
        vol: float,
        lower: float,
        upper: float,
        engine: "DiscreteQuadEngine",
    ) -> None:
        validate_positive(spot, "spot")
        validate_positive(strike, "strike")
        validate_positive(vol, "volatility")
        validate_positive(maturity, "maturity", allow_zero=True)
        validate_positive(lower, "lower_barrier")
        validate_positive(upper, "upper_barrier")
        validate_non_negative(div, "dividend_yield")

        if lower >= upper:
            raise ValidationError(
                f"lower_barrier ({lower}) must be less than upper_barrier ({upper})"
            )
        if abs(rate) > 1.0:
            raise ValidationError(f"Risk-free rate outside reasonable bounds: {rate}")
        if vol > engine.MAX_VOL:
            raise ValidationError(f"Volatility too high for quadrature stability: {vol}")


class DoubleOneTouchQuadInputAdapter(BaseDiscreteQuadAdapter):
    """Adapter for double one-touch/no-touch options in discrete quadrature."""

    def can_handle(self, product: BaseEquityProduct) -> bool:
        return isinstance(product, DoubleOneTouchOption)

    def build_pricing_context(
        self,
        product: DoubleOneTouchOption,
        pricing_env: PricingEnvironment,
        engine: "DiscreteQuadEngine",
    ) -> QuadPricingContext:
        spot = pricing_env.spot
        maturity = product.get_maturity(pricing_env)
        rate = pricing_env.get_rate(maturity)
        div = pricing_env.get_div_yield(maturity)
        barrier_ref = math.sqrt(product.upper_barrier * product.lower_barrier)
        vol = pricing_env.get_vol(barrier_ref, maturity)
        contract_multiplier = getattr(product, "contract_multiplier", 1.0)

        self._validate_inputs(
            spot,
            product.lower_barrier,
            product.upper_barrier,
            maturity,
            rate,
            div,
            vol,
            product.rebate,
            engine,
        )

        return QuadPricingContext(
            spot=spot,
            maturity=maturity,
            rate=rate,
            div=div,
            vol=vol,
            contract_multiplier=contract_multiplier,
        )

    def early_price(
        self,
        product: DoubleOneTouchOption,
        pricing_env: PricingEnvironment,
        context: QuadPricingContext,
        engine: "DiscreteQuadEngine",
    ) -> float | None:
        if is_zero(context.maturity, tol=engine.MIN_MATURITY):
            value = self._double_touch_instant_payoff(
                product, context.spot, context.maturity, context.rate
            )
            return value * context.contract_multiplier

        touched = (
            context.spot >= product.upper_barrier
            or context.spot <= product.lower_barrier
        )
        if product.observation_type != ObservationType.EXPIRY and touched:
            if product.touch_type == TouchType.DOUBLE_ONE_TOUCH:
                pay_at_hit = product.payment_at_hit
                value = product.rebate if pay_at_hit else product.rebate * math.exp(
                    -context.rate * context.maturity
                )
                return value * context.contract_multiplier
            return 0.0

        if product.observation_type == ObservationType.CONTINUOUS:
            raise PricingError(
                "DiscreteQuadEngine supports discrete monitoring only. "
                "Use a continuous engine for continuous barriers."
            )

        if product.rebate <= 0.0:
            return 0.0

        return None

    def resolve_schedule(
        self,
        product: DoubleOneTouchOption,
        pricing_env: PricingEnvironment,
        context: QuadPricingContext,
    ) -> Sequence[ResolvedObservationRecord]:
        if product.observation_type == ObservationType.EXPIRY:
            schedule = ObservationSchedule.from_legacy(
                observation_dates=[context.maturity],
                default_barrier=None,
                default_payoff=product.rebate,
                aggregation_mode=ObservationAggregation.STOP_FIRST_HIT,
                upper_barrier=product.upper_barrier,
                lower_barrier=product.lower_barrier,
            )
        else:
            schedule = product.observation_schedule
            if schedule is None and product.observation_dates:
                schedule = ObservationSchedule.from_legacy(
                    observation_dates=product.observation_dates,
                    default_barrier=None,
                    default_payoff=product.rebate,
                    aggregation_mode=ObservationAggregation.STOP_FIRST_HIT,
                    upper_barrier=product.upper_barrier,
                    lower_barrier=product.lower_barrier,
                )

        if schedule is None or not schedule.records:
            raise PricingError("Discrete monitoring requires ObservationSchedule.")
        if schedule.aggregation_mode != ObservationAggregation.STOP_FIRST_HIT:
            raise PricingError("DiscreteQuadEngine requires STOP_FIRST_HIT aggregation.")

        return schedule.resolve(
            pricing_env,
            default_upper=product.upper_barrier,
            default_lower=product.lower_barrier,
            default_payoff=product.rebate,
            require_double=True,
        )

    def build_inputs(
        self,
        product: DoubleOneTouchOption,
        resolved: Sequence[ResolvedObservationRecord],
        context: QuadPricingContext,
    ) -> QuadCoreInputs:
        (
            obs_times,
            upper_barriers,
            lower_barriers,
            payoffs,
            settlement_times,
        ) = self._extract_double_observations(resolved, context.maturity)

        k_plus = np.array(upper_barriers, dtype=float)
        k_minus = np.array(lower_barriers, dtype=float)
        a_minus = np.zeros_like(k_minus, dtype=float)
        b_minus = np.zeros_like(k_minus, dtype=float)
        a_plus = np.zeros_like(k_plus, dtype=float)
        b_plus = np.zeros_like(k_plus, dtype=float)

        if np.any(payoffs > 0.0):
            rebate_values = self._discount_rebates(
                payoffs,
                obs_times,
                settlement_times,
                context.maturity,
                context.rate,
                product.payment_at_hit,
            )
            b_minus = rebate_values
            b_plus = rebate_values

        return QuadCoreInputs(
            observation_times=obs_times,
            k_minus=k_minus,
            k_plus=k_plus,
            a_minus=a_minus,
            b_minus=b_minus,
            a_plus=a_plus,
            b_plus=b_plus,
            a_terminal=0.0,
            b_terminal=0.0,
        )

    def finalize_price(
        self,
        product: DoubleOneTouchOption,
        pricing_env: PricingEnvironment,
        context: QuadPricingContext,
        core_price: float,
        engine: "DiscreteQuadEngine",
    ) -> float:
        if product.touch_type == TouchType.DOUBLE_NO_TOUCH:
            rebate_discount = product.rebate * math.exp(-context.rate * context.maturity)
            value = rebate_discount - core_price
            return max(0.0, value) * context.contract_multiplier
        return core_price * context.contract_multiplier

    def _extract_double_observations(
        self, resolved: Sequence[ResolvedObservationRecord], maturity: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obs_times = np.array([rec.observation_time for rec in resolved], dtype=float)
        upper = np.array([rec.upper_barrier for rec in resolved], dtype=float)
        lower = np.array([rec.lower_barrier for rec in resolved], dtype=float)
        payoffs = np.array([rec.payoff for rec in resolved], dtype=float)
        settlement_times = np.array(
            [
                rec.settlement_time if rec.settlement_time is not None else rec.observation_time
                for rec in resolved
            ],
            dtype=float,
        )

        if obs_times.size == 0:
            raise ValidationError("Resolved observation schedule is empty.")

        if maturity - obs_times[-1] > Tolerance.ZERO:
            obs_times = np.append(obs_times, maturity)
            upper = np.append(upper, math.inf)
            lower = np.append(lower, 0.0)
            payoffs = np.append(payoffs, 0.0)
            settlement_times = np.append(settlement_times, maturity)

        return obs_times, upper, lower, payoffs, settlement_times

    def _double_touch_instant_payoff(
        self,
        product: DoubleOneTouchOption,
        spot: float,
        maturity: float,
        rate: float,
    ) -> float:
        touched = spot >= product.upper_barrier or spot <= product.lower_barrier
        discount = math.exp(-rate * maturity)
        if product.touch_type == TouchType.DOUBLE_ONE_TOUCH:
            if touched:
                return product.rebate if product.payment_at_hit else product.rebate * discount
            return 0.0
        return product.rebate * discount if not touched else 0.0

    def _validate_inputs(
        self,
        spot: float,
        lower: float,
        upper: float,
        maturity: float,
        rate: float,
        div: float,
        vol: float,
        rebate: float,
        engine: "DiscreteQuadEngine",
    ) -> None:
        validate_positive(spot, "spot")
        validate_positive(lower, "lower_barrier")
        validate_positive(upper, "upper_barrier")
        validate_positive(vol, "volatility")
        validate_positive(maturity, "maturity", allow_zero=True)
        validate_non_negative(rebate, "rebate")
        validate_non_negative(div, "dividend_yield")

        if lower >= upper:
            raise ValidationError(
                f"lower_barrier ({lower}) must be less than upper_barrier ({upper})"
            )
        if abs(rate) > 1.0:
            raise ValidationError(f"Risk-free rate outside reasonable bounds: {rate}")
        if vol > engine.MAX_VOL:
            raise ValidationError(f"Volatility too high for quadrature stability: {vol}")


class QuadInputAdapterRegistry:
    """Registry for quad input adapters."""

    def __init__(self) -> None:
        self._adapters: list[QuadInputAdapter] = []

    def register(self, adapter: QuadInputAdapter) -> None:
        self._adapters.append(adapter)

    def resolve(self, product: BaseEquityProduct) -> QuadInputAdapter:
        for adapter in self._adapters:
            if adapter.can_handle(product):
                return adapter
        raise PricingError(
            f"DiscreteQuadEngine has no adapter for product type {type(product).__name__}"
        )


_ADAPTER_REGISTRY = QuadInputAdapterRegistry()
_ADAPTER_REGISTRY.register(BarrierQuadInputAdapter())
_ADAPTER_REGISTRY.register(OneTouchQuadInputAdapter())
_ADAPTER_REGISTRY.register(DoubleBarrierQuadInputAdapter())
_ADAPTER_REGISTRY.register(DoubleOneTouchQuadInputAdapter())


def register_quad_adapter(adapter: QuadInputAdapter) -> None:
    """Register a custom quad input adapter."""
    _ADAPTER_REGISTRY.register(adapter)


def resolve_quad_adapter(product: BaseEquityProduct) -> QuadInputAdapter:
    """Resolve the quad input adapter for a product."""
    return _ADAPTER_REGISTRY.resolve(product)
