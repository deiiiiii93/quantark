"""
Quadrature pricing engine for discretely monitored one-touch options.
"""

import math
from typing import Optional, Sequence, Tuple

import numpy as np

from asset.equity.engine.base_engine import BaseEngine
from asset.equity.engine.quad.barrier_quad_solver import _BarrierQuadratureSolver
from asset.equity.param import QuadParams
from asset.equity.product.base_equity_product import BaseEquityProduct
from asset.equity.product.option import OneTouchOption
from asset.equity.product.option.observation_schedule import ObservationSchedule
from priceenv import PricingEnvironment
from util.enum import ObservationAggregation, ObservationType
from util.enum.engine_enums import EngineType
from util.exceptions import PricingError, ValidationError
from util.numerical import Tolerance, is_zero, validate_non_negative, validate_positive


class OneTouchQuadEngine(BaseEngine):
    """
    Quadrature pricing engine for discretely monitored one-touch/no-touch options.

    Uses the FFT-based quadrature recursion from the barrier quadrature solver
    and maps one-touch payoffs to the barrier-hit cash factors.
    """

    engine_type = EngineType.QUADRATURE
    MIN_MATURITY = 1e-10
    MAX_VOL = 5.0

    def __init__(self, params: Optional[QuadParams] = None) -> None:
        """
        Initialize the one-touch quadrature engine.

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

    def price(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        """
        Price a discretely monitored one-touch/no-touch option using quadrature.

        Args:
            product: OneTouchOption instance.
            pricing_env: Pricing environment with market data.

        Returns:
            Option price.
        """
        if not isinstance(product, OneTouchOption):
            raise PricingError(
                f"OneTouchQuadEngine only supports OneTouchOption, "
                f"got {type(product).__name__}"
            )

        spot = pricing_env.spot
        maturity = product.get_maturity(pricing_env)
        rate = pricing_env.get_rate(maturity)
        div = pricing_env.get_div_yield(maturity)
        vol = pricing_env.get_vol(product.barrier, maturity)
        rebate = product.rebate
        pay_at_hit = product.payment_at_hit if product.is_one_touch else False
        contract_multiplier = getattr(product, "contract_multiplier", 1.0)

        self._validate_inputs(spot, product.barrier, maturity, rate, div, vol, rebate)

        if is_zero(maturity, tol=self.MIN_MATURITY):
            value = self._instantaneous_payoff(product, spot, maturity, rate, pay_at_hit)
            return value * contract_multiplier

        if (
            product.observation_type != ObservationType.EXPIRY
            and product.is_barrier_hit(spot)
        ):
            if product.is_one_touch:
                value = rebate if pay_at_hit else rebate * math.exp(-rate * maturity)
                return value * contract_multiplier
            return 0.0

        if product.observation_type == ObservationType.CONTINUOUS:
            raise PricingError(
                "OneTouchQuadEngine supports discrete or expiry monitoring only. "
                "Use OneTouchAnalyticalEngine for continuous monitoring."
            )

        schedule, resolved = self._resolve_schedule(product, pricing_env, maturity)
        if schedule.aggregation_mode != ObservationAggregation.STOP_FIRST_HIT:
            raise PricingError(
                "OneTouchQuadEngine requires STOP_FIRST_HIT observation aggregation."
            )

        if rebate <= 0.0:
            return 0.0

        if product.is_no_touch:
            payoffs = np.full(len(resolved), rebate, dtype=float)
            touch_value = self._price_one_touch(
                product=product,
                spot=spot,
                maturity=maturity,
                rate=rate,
                div=div,
                vol=vol,
                resolved=resolved,
                pay_at_hit=False,
                payoffs=payoffs,
            )
            rebate_discount = rebate * math.exp(-rate * maturity)
            value = rebate_discount - touch_value
            return max(0.0, value) * contract_multiplier

        payoffs = np.array([rec.payoff for rec in resolved], dtype=float)
        value = self._price_one_touch(
            product=product,
            spot=spot,
            maturity=maturity,
            rate=rate,
            div=div,
            vol=vol,
            resolved=resolved,
            pay_at_hit=pay_at_hit,
            payoffs=payoffs,
        )
        return value * contract_multiplier

    def _validate_inputs(
        self,
        spot: float,
        barrier: float,
        maturity: float,
        rate: float,
        div: float,
        vol: float,
        rebate: float,
    ) -> None:
        validate_positive(spot, "spot")
        validate_positive(barrier, "barrier")
        validate_positive(vol, "volatility")
        validate_positive(maturity, "maturity", allow_zero=True)
        validate_non_negative(rebate, "rebate")
        validate_non_negative(div, "dividend_yield")

        if abs(rate) > 1.0:
            raise ValidationError(f"Risk-free rate outside reasonable bounds: {rate}")
        if vol > self.MAX_VOL:
            raise ValidationError(
                f"Volatility too high for quadrature stability: {vol}"
            )

    def _instantaneous_payoff(
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

    def _resolve_schedule(
        self, product: OneTouchOption, pricing_env: PricingEnvironment, maturity: float
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

    def _price_one_touch(
        self,
        product: OneTouchOption,
        spot: float,
        maturity: float,
        rate: float,
        div: float,
        vol: float,
        resolved: Sequence,
        pay_at_hit: bool,
        payoffs: np.ndarray,
    ) -> float:
        obs_times = np.array([rec.observation_time for rec in resolved], dtype=float)
        if np.any(obs_times < -Tolerance.ZERO) or np.any(
            obs_times > maturity + Tolerance.ZERO
        ):
            raise ValidationError("Observation times must be within [0, maturity].")
        if len(obs_times) > 1 and np.any(np.diff(obs_times) <= Tolerance.ZERO):
            raise ValidationError("Observation times must be strictly increasing.")

        if payoffs.size != len(obs_times):
            raise ValidationError("Payoff array size does not match observations.")

        barriers = np.array([rec.barrier for rec in resolved], dtype=float)
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
        grid_times = time_grid[obs_indices]

        dt = maturity / grid_t
        vol_max = vol if np.isscalar(vol) else float(np.max(vol))
        approx_log_c = (
            10.0 * vol_max * math.sqrt(maturity)
            + (1.0 + 0.5 * vol_max * vol_max) * maturity
        )
        range_width = 2.0 * approx_log_c
        target_h = 0.5 * vol_max * math.sqrt(dt)
        safe_grid_x = int(range_width / target_h) + 1
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

        if pay_at_hit:
            delay = np.maximum(settlement_times - grid_times, 0.0)
            discount = np.exp(-rate * delay)
        else:
            discount = np.exp(-rate * (maturity - grid_times))

        rebate_values = payoffs * discount
        if product.is_up_barrier:
            factors["cash3"][obs_indices] = rebate_values
        else:
            factors["cash1"][obs_indices] = rebate_values

        return solver.price(
            upper_bounds=upper_bounds,
            lower_bounds=lower_bounds,
            upper_indices=obs_indices if product.is_up_barrier else [],
            lower_indices=obs_indices if product.is_down_barrier else [],
            factors=factors,
        )

    def _select_time_steps(self, maturity: float, obs_count: int) -> int:
        steps = max(int(self.params.bus_days_in_year * maturity), obs_count * 4, 10)
        return max(steps, 3)

    def __repr__(self) -> str:
        return "OneTouchQuadEngine()"
