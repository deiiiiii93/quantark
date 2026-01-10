"""
Quadrature pricing engine for discretely monitored single-barrier options.
"""

import math
from typing import Optional, Sequence, Tuple

import numpy as np

from asset.equity.engine.base_engine import BaseEngine
from asset.equity.engine.quad.barrier_quad_solver import _BarrierQuadratureSolver
from asset.equity.engine.quad.european_quad_engine import EuropeanQuadEngine
from asset.equity.engine.quad.one_touch_quad_engine import OneTouchQuadEngine
from asset.equity.param import QuadParams
from asset.equity.product.base_equity_product import BaseEquityProduct
from asset.equity.product.option import BarrierOption, OneTouchOption
from asset.equity.product.option.observation_schedule import ObservationSchedule
from priceenv import PricingEnvironment
from util.enum import (
    BarrierDirection,
    BarrierType,
    ObservationAggregation,
    ObservationType,
    TouchType,
)
from util.enum.engine_enums import EngineType
from util.exceptions import PricingError, ValidationError


class BarrierQuadEngine(BaseEngine):
    """
    Quadrature pricing engine for discretely monitored single-barrier options.

    Supports UP_OUT, DOWN_OUT, UP_IN, DOWN_IN barrier types with discrete
    observation schedules using FFT-based convolution and Simpson integration.
    """

    engine_type = EngineType.QUADRATURE
    MIN_MATURITY = 1e-10

    def __init__(self, params: Optional[QuadParams] = None) -> None:
        """
        Initialize the barrier quadrature engine.

        Args:
            params: Quadrature configuration parameters (QuadParams).
        """
        if params is None:
            params = QuadParams()
        if not isinstance(params, QuadParams):
            raise ValidationError(
                f"params must be QuadParams instance, got {type(params).__name__}"
            )
        super().__init__(params)
        self._vanilla_engine = EuropeanQuadEngine(params=params)
        self._one_touch_engine = OneTouchQuadEngine(params=params)

    def price(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        """
        Price a discretely monitored single-barrier option using quadrature.

        Args:
            product: BarrierOption instance.
            pricing_env: Pricing environment with market data.

        Returns:
            Option price.
        """
        if not isinstance(product, BarrierOption):
            raise PricingError(
                f"BarrierQuadEngine only supports BarrierOption, got {type(product).__name__}"
            )

        S = pricing_env.spot
        K = product.strike
        T = product.get_maturity(pricing_env)
        r = pricing_env.get_rate(T)
        q = pricing_env.get_div_yield(T)
        sigma = pricing_env.get_vol(K, T)

        self._validate_inputs(S, K, T, r, q, sigma, product.barrier)

        if T < self.MIN_MATURITY:
            return self._price_expired(product, S, r, T) * product.contract_multiplier

        if (
            product.observation_type != ObservationType.EXPIRY
            and product.is_barrier_hit(S)
        ):
            if product.is_knock_out:
                return (
                    self._price_immediate_knock_out(product, r, T)
                    * product.contract_multiplier
                )
            vanilla_price = self._vanilla_price(product, pricing_env)
            return vanilla_price * product.contract_multiplier

        if product.observation_type == ObservationType.CONTINUOUS:
            raise PricingError(
                "BarrierQuadEngine supports discrete monitoring only. "
                "Use BarrierAnalyticalEngine for continuous barriers."
            )

        schedule, resolved = self._resolve_schedule(product, pricing_env, T)
        if schedule.aggregation_mode != ObservationAggregation.STOP_FIRST_HIT:
            raise PricingError(
                "BarrierQuadEngine requires STOP_FIRST_HIT observation aggregation."
            )

        if product.is_knock_in:
            ko_price = self._price_knock_out(
                self._to_knock_out(product),
                pricing_env,
                S,
                K,
                T,
                r,
                q,
                sigma,
                resolved,
            )
            vanilla_price = self._vanilla_price(product, pricing_env)
            rebate_discount = product.rebate * math.exp(-r * T)
            return (
                vanilla_price + rebate_discount - ko_price
            ) * product.contract_multiplier

        ko_price = self._price_knock_out(
            product, pricing_env, S, K, T, r, q, sigma, resolved
        )
        return ko_price * product.contract_multiplier

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

    def _price_expired(
        self, product: BarrierOption, spot: float, r: float, T: float
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
        if T <= 0.0:
            return value
        return value * math.exp(-r * T)

    def _price_immediate_knock_out(
        self, product: BarrierOption, r: float, T: float
    ) -> float:
        if product.pay_at_hit:
            return product.rebate
        return product.rebate * math.exp(-r * T)

    def _vanilla_price(
        self, product: BarrierOption, pricing_env: PricingEnvironment
    ) -> float:
        from asset.equity.product.option import EuropeanVanillaOption

        vanilla = EuropeanVanillaOption(
            strike=product.strike,
            option_type=product.option_type,
            maturity=product.maturity,
            exercise_date=product.exercise_date,
            settlement_date=product.settlement_date,
            contract_multiplier=1.0,
        )
        price = self._vanilla_engine.price(vanilla, pricing_env)
        return price * product.participation_rate

    def _resolve_schedule(
        self, product: BarrierOption, pricing_env: PricingEnvironment, maturity: float
    ) -> Tuple[ObservationSchedule, list]:
        if product.observation_type == ObservationType.EXPIRY:
            schedule = ObservationSchedule.from_legacy(
                observation_dates=[maturity],
                default_barrier=product.barrier,
                default_payoff=product.rebate,
                aggregation_mode=ObservationAggregation.STOP_FIRST_HIT,
            )
        else:
            schedule = product.observation_schedule
            if schedule is None and product.observation_dates:
                schedule = ObservationSchedule.from_legacy(
                    observation_dates=product.observation_dates,
                    default_barrier=product.barrier,
                    default_payoff=product.rebate,
                    aggregation_mode=ObservationAggregation.STOP_FIRST_HIT,
                )
        if schedule is None or not schedule.records:
            raise PricingError("Discrete monitoring requires ObservationSchedule.")

        resolved = schedule.resolve(
            pricing_env,
            default_barrier=product.barrier,
            default_payoff=product.rebate,
            require_single=True,
        )
        return schedule, resolved

    def _price_knock_out(
        self,
        product: BarrierOption,
        pricing_env: PricingEnvironment,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        div: float,
        vol: float,
        resolved: Sequence,
    ) -> float:
        obs_times = np.array([rec.observation_time for rec in resolved], dtype=float)
        if np.any(obs_times < 0.0) or np.any(obs_times > maturity + 1e-12):
            raise ValidationError("Observation times must be within [0, maturity].")
        if len(obs_times) > 1 and np.any(np.diff(obs_times) <= 0.0):
            raise ValidationError("Observation times must be strictly increasing.")

        barriers = np.array([rec.barrier for rec in resolved], dtype=float)
        payoffs = np.array([rec.payoff for rec in resolved], dtype=float)
        settlement_times = np.array(
            [
                rec.settlement_time
                if rec.settlement_time is not None
                else rec.observation_time
                for rec in resolved
            ],
            dtype=float,
        )

        grid_t = self._select_time_steps(maturity, len(obs_times))
        time_grid = np.linspace(0.0, maturity, grid_t + 1)
        obs_indices = np.searchsorted(time_grid, obs_times, side="left")
        obs_indices = np.clip(obs_indices, 0, grid_t)

        # Numerical Stability Check:
        # Ensure grid spacing h is small enough to resolve the Gaussian kernel width (sigma * sqrt(dt)).
        # Heuristic: h <= 0.5 * sigma * sqrt(dt)
        # Range width approx: 2 * log_c (calculated inside solver, but approximated here)
        dt = maturity / grid_t
        vol_max = vol if np.isscalar(vol) else float(np.max(vol))
        
        # Approximate log_c from solver logic
        approx_log_c = 10.0 * vol_max * math.sqrt(maturity) + (1.0 + 0.5 * vol_max**2) * maturity
        range_width = 2.0 * approx_log_c
        
        target_h = 0.5 * vol_max * math.sqrt(dt)
        safe_grid_x = int(range_width / target_h) + 1
        
        # Use the larger of user-specified points or safe points
        effective_grid_x = max(self.params.grid_points, safe_grid_x)

        solver = _BarrierQuadratureSolver(
            grid_x=effective_grid_x,
            grid_t=grid_t,
            maturity=maturity,
            spot=spot,
            r=rate,
            q=div,
            vol=vol,
        )
        upper_bounds = np.full(grid_t + 1, spot * solver.constant_c)
        lower_bounds = np.full(grid_t + 1, spot / solver.constant_c)

        if product.is_up_barrier:
            upper_bounds[obs_indices] = barriers
        else:
            lower_bounds[obs_indices] = barriers

        factors = {
            "asset1": np.zeros(grid_t + 1),
            "asset2": np.zeros(grid_t + 1),
            "asset3": np.zeros(grid_t + 1),
            "cash1": np.zeros(grid_t + 1),
            "cash2": np.zeros(grid_t + 1),
            "cash3": np.zeros(grid_t + 1),
        }

        maturity_barrier = None
        maturity_tol = 1e-10
        maturity_mask = np.isclose(obs_times, maturity, atol=maturity_tol, rtol=0.0)
        if np.any(maturity_mask):
            maturity_barrier = float(barriers[np.where(maturity_mask)[0][-1]])

        payoff_lower = None
        payoff_upper = None
        if product.is_call():
            payoff_lower = strike
            if maturity_barrier is not None:
                if product.is_up_barrier:
                    payoff_upper = maturity_barrier
                else:
                    payoff_lower = max(strike, maturity_barrier)
        else:
            payoff_upper = strike
            if maturity_barrier is not None:
                if product.is_up_barrier:
                    payoff_upper = min(strike, maturity_barrier)
                else:
                    payoff_lower = maturity_barrier

        payoff_active = True
        if payoff_upper is not None and payoff_lower is not None:
            if payoff_upper <= payoff_lower:
                payoff_active = False

        if payoff_active:
            if payoff_lower is not None:
                lower_bounds[-1] = payoff_lower
            if payoff_upper is not None:
                upper_bounds[-1] = payoff_upper
            if product.is_call():
                payoff_asset = 1.0
                payoff_cash = -strike
            else:
                payoff_asset = -1.0
                payoff_cash = strike
            payoff_asset *= product.participation_rate
            payoff_cash *= product.participation_rate
            factors["asset2"][-1] = payoff_asset
            factors["cash2"][-1] = payoff_cash

        rebate_leg = 0.0
        rebate_in_factors = False
        if product.rebate > 0.0:
            try:
                rebate_leg = self._price_rebate_leg(product, pricing_env)
            except (ValidationError, PricingError):
                rebate_in_factors = True

        if rebate_in_factors:
            discount_T = math.exp(-rate * maturity)
            if product.pay_at_hit:
                rebate_discount = np.exp(-rate * settlement_times)
            else:
                rebate_discount = discount_T * np.ones_like(settlement_times)
            rebate_values = payoffs * rebate_discount
            if product.is_up_barrier:
                factors["cash3"][obs_indices] = rebate_values
            else:
                factors["cash1"][obs_indices] = rebate_values

        quad_value = solver.price(
            upper_bounds=upper_bounds,
            lower_bounds=lower_bounds,
            upper_indices=obs_indices if product.is_up_barrier else [],
            lower_indices=obs_indices if product.is_down_barrier else [],
            factors=factors,
        )
        return quad_value + rebate_leg

    def _price_rebate_leg(
        self, product: BarrierOption, pricing_env: PricingEnvironment
    ) -> float:
        if product.rebate <= 0.0:
            return 0.0

        barrier_direction = (
            BarrierDirection.UP if product.is_up_barrier else BarrierDirection.DOWN
        )
        touch_type = TouchType.ONE_TOUCH if product.is_knock_out else TouchType.NO_TOUCH
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

    def _select_time_steps(self, maturity: float, obs_count: int) -> int:
        steps = max(int(self.params.bus_days_in_year * maturity), obs_count * 4, 10)
        return max(steps, 3)

    def _to_knock_out(self, product: BarrierOption) -> BarrierOption:
        """Convert knock-in to knock-out for barrier parity decomposition.

        Uses the same rebate with pay-at-expiry to align with KI parity.
        """
        if product.barrier_type == BarrierType.UP_IN:
            ko_type = BarrierType.UP_OUT
        elif product.barrier_type == BarrierType.DOWN_IN:
            ko_type = BarrierType.DOWN_OUT
        else:
            ko_type = product.barrier_type

        return BarrierOption(
            strike=product.strike,
            option_type=product.option_type,
            barrier=product.barrier,
            barrier_type=ko_type,
            maturity=product.maturity,
            exercise_date=product.exercise_date,
            settlement_date=product.settlement_date,
            rebate=product.rebate,
            participation_rate=product.participation_rate,
            pay_at_hit=False,
            observation_type=product.observation_type,
            observation_dates=product.observation_dates,
            observation_schedule=product.observation_schedule,
            contract_multiplier=1.0,
        )

    def __repr__(self) -> str:
        return "BarrierQuadEngine()"
