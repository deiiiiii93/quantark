"""
PDE solver for single barrier options.

Implements the finite difference method for knock-in and knock-out
barrier options with continuous or discrete monitoring.
"""

from typing import Dict, List, Optional, Set

import numpy as np

from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option.barrier_option import BarrierOption
from quantark.asset.equity.product.option.observation_schedule import ResolvedObservationRecord
from quantark.asset.equity.param import PDEParams
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType, ObservationAggregation
from quantark.util.exceptions import PricingError
from quantark.util.numerical import is_close, safe_divide

from .base_pde_solver import BasePDESolver


class BarrierPDESolver(BasePDESolver):
    """
    PDE solver for single barrier options.

    Handles all four barrier types:
        - DOWN_OUT: Knock-out when price goes below barrier
        - DOWN_IN:  Knock-in when price goes below barrier
        - UP_OUT:   Knock-out when price goes above barrier
        - UP_IN:    Knock-in when price goes above barrier

    For knock-out options, the grid boundary at the barrier is set to
    the rebate value (or 0 if no rebate).

    For knock-in options, we use the identity:
        Knock-in = Vanilla - Knock-out

    This solver also handles discrete observation by only checking
    the barrier at specified observation times.
    """

    def __init__(self, params: Optional[PDEParams] = None):
        """
        Initialize barrier option PDE solver.

        Args:
            params: PDE engine configuration parameters
        """
        super().__init__(params)
        self._observation_indices: Set[int] = set()
        self._schedule_records: Dict[int, List[ResolvedObservationRecord]] = {}
        self._schedule_aggregation: ObservationAggregation = (
            ObservationAggregation.STOP_FIRST_HIT
        )
        self._total_tau: float = 0.0
        self._terminal_schedule_records: List[ResolvedObservationRecord] = []
        self._has_terminal_observation: bool = False

    @staticmethod
    def _current_time(total_tau: float, tau_remaining: float) -> float:
        return max(total_tau - tau_remaining, 0.0)

    @staticmethod
    def _df_between_times(
        pricing_env: PricingEnvironment, start_time: float, end_time: float
    ) -> float:
        if end_time <= start_time:
            return 1.0
        df_end = pricing_env.get_discount_factor(end_time)
        df_start = pricing_env.get_discount_factor(start_time)
        return float(safe_divide(df_end, df_start, fallback=0.0))

    def _cashflow_value_at_time(
        self,
        pricing_env: PricingEnvironment,
        cashflow: float,
        current_time: float,
        settlement_time: Optional[float],
    ) -> float:
        if settlement_time is None or settlement_time <= current_time:
            return float(cashflow)
        df = self._df_between_times(pricing_env, current_time, settlement_time)
        return float(cashflow) * df

    def price(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        """
        Price a barrier option using PDE method.

        For knock-in options, uses: Knock-in = Vanilla - Knock-out

        Args:
            product: Barrier option
            pricing_env: Pricing environment

        Returns:
            Option price

        Raises:
            PricingError: If product is not a barrier option
        """
        if not isinstance(product, BarrierOption):
            raise PricingError(
                f"BarrierPDESolver only supports BarrierOption, "
                f"got {type(product).__name__}"
            )

        if getattr(product, "observation_type", None) == ObservationType.EXPIRY:
            raise PricingError(
                "BarrierPDESolver does not support EXPIRY observation_type. "
                "Use BarrierAnalyticalEngine for expiry-only monitoring."
            )

        # Check if barrier is already hit
        spot = pricing_env.spot
        if product.is_barrier_hit(spot):
            if product.is_knock_out:
                maturity = product.get_maturity(pricing_env)
                settlement_time = 0.0 if product.pay_at_hit else maturity
                return self._cashflow_value_at_time(
                    pricing_env=pricing_env,
                    cashflow=product.rebate,
                    current_time=0.0,
                    settlement_time=settlement_time,
                )
            else:
                # Knocked in, price as vanilla
                return self._price_vanilla(product, pricing_env)

        if product.is_knock_in:
            # Knock-in = Vanilla - Knock-out (with no rebate)
            vanilla_price = self._price_vanilla(product, pricing_env)

            # Create a temporary knock-out version with no rebate
            ko_price = self._price_knock_out(product, pricing_env)

            return vanilla_price - ko_price
        else:
            # Direct knock-out pricing
            return super().price(product, pricing_env)

    def _price_vanilla(
        self, product: BarrierOption, pricing_env: PricingEnvironment
    ) -> float:
        """
        Price the underlying vanilla option.

        Args:
            product: Barrier option (used for strike, type, maturity)
            pricing_env: Pricing environment

        Returns:
            Vanilla option price
        """
        from quantark.asset.equity.product.option import EuropeanVanillaOption
        from .european_pde_solver import EuropeanPDESolver

        vanilla = EuropeanVanillaOption(
            strike=product.strike,
            option_type=product.option_type,
            maturity=product.maturity,
            exercise_date=product.exercise_date,
            settlement_date=product.settlement_date,
        )

        solver = EuropeanPDESolver(self.params)
        return product.participation_rate * solver.price(vanilla, pricing_env)

    def _price_knock_out(
        self, product: BarrierOption, pricing_env: PricingEnvironment
    ) -> float:
        """
        Price as a knock-out option (for knock-in decomposition).

        Creates a temporary knock-out version and prices it.

        Args:
            product: Original barrier option
            pricing_env: Pricing environment

        Returns:
            Knock-out option price
        """
        # For knock-in decomposition, we need knock-out with zero rebate
        from quantark.util.enum import BarrierType

        # Convert knock-in type to knock-out
        if product.barrier_type == BarrierType.UP_IN:
            ko_type = BarrierType.UP_OUT
        elif product.barrier_type == BarrierType.DOWN_IN:
            ko_type = BarrierType.DOWN_OUT
        else:
            ko_type = product.barrier_type

        ko_product = BarrierOption(
            strike=product.strike,
            option_type=product.option_type,
            barrier=product.barrier,
            barrier_type=ko_type,
            maturity=product.maturity,
            exercise_date=product.exercise_date,
            settlement_date=product.settlement_date,
            rebate=0.0,  # Zero rebate for decomposition
            participation_rate=product.participation_rate,
            pay_at_hit=product.pay_at_hit,
            observation_type=product.observation_type,
            observation_dates=product.observation_dates,
            observation_schedule=product.observation_schedule,
        )

        return super().price(ko_product, pricing_env)

    def calculate_greeks(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> Dict[str, float]:
        """
        Calculate Greeks for a barrier option using PDE method.

        For knock-in options, uses: Greeks_KI = Greeks_Vanilla - Greeks_KO

        Args:
            product: Barrier option
            pricing_env: Pricing environment

        Returns:
            Dictionary with price, delta, gamma

        Raises:
            PricingError: If product is not a barrier option
        """
        if not isinstance(product, BarrierOption):
            raise PricingError(
                f"BarrierPDESolver only supports BarrierOption, "
                f"got {type(product).__name__}"
            )

        if getattr(product, "observation_type", None) == ObservationType.EXPIRY:
            raise PricingError(
                "BarrierPDESolver does not support EXPIRY observation_type. "
                "Use BarrierAnalyticalEngine for expiry-only monitoring."
            )

        spot = pricing_env.spot
        tau = product.get_maturity(pricing_env)

        # Handle expired case
        if tau <= 0:
            return {
                "price": self._calculate_intrinsic(product, spot),
                "delta": self._intrinsic_delta(product, spot),
                "gamma": 0.0,
            }

        # Check if barrier is already hit
        if product.is_barrier_hit(spot):
            if product.is_knock_out:
                # Knocked out: return rebate Greeks (rebate is constant, so delta=gamma=0)
                maturity = product.get_maturity(pricing_env)
                settlement_time = 0.0 if product.pay_at_hit else maturity
                rebate_value = self._cashflow_value_at_time(
                    pricing_env=pricing_env,
                    cashflow=product.rebate,
                    current_time=0.0,
                    settlement_time=settlement_time,
                )
                return {"price": rebate_value, "delta": 0.0, "gamma": 0.0}
            else:
                # Knocked in: return vanilla Greeks
                return self._calculate_greeks_vanilla(product, pricing_env)

        if product.is_knock_in:
            # Knock-in = Vanilla - Knock-out decomposition
            vanilla_greeks = self._calculate_greeks_vanilla(product, pricing_env)
            ko_greeks = self._calculate_greeks_knock_out(product, pricing_env)

            return {
                "price": vanilla_greeks["price"] - ko_greeks["price"],
                "delta": vanilla_greeks["delta"] - ko_greeks["delta"],
                "gamma": vanilla_greeks["gamma"] - ko_greeks["gamma"],
            }
        else:
            # Direct knock-out pricing
            return super().calculate_greeks(product, pricing_env)

    def _calculate_greeks_vanilla(
        self, product: BarrierOption, pricing_env: PricingEnvironment
    ) -> Dict[str, float]:
        """
        Calculate Greeks for the underlying vanilla option.

        Args:
            product: Barrier option (used for strike, type, maturity)
            pricing_env: Pricing environment

        Returns:
            Dictionary with price, delta, gamma for vanilla option
        """
        from quantark.asset.equity.product.option import EuropeanVanillaOption
        from .european_pde_solver import EuropeanPDESolver

        vanilla = EuropeanVanillaOption(
            strike=product.strike,
            option_type=product.option_type,
            maturity=product.maturity,
            exercise_date=product.exercise_date,
            settlement_date=product.settlement_date,
        )

        solver = EuropeanPDESolver(self.params)
        greeks = solver.calculate_greeks(vanilla, pricing_env)

        # Apply participation rate
        pr = product.participation_rate
        return {
            "price": pr * greeks["price"],
            "delta": pr * greeks["delta"],
            "gamma": pr * greeks["gamma"],
        }

    def _calculate_greeks_knock_out(
        self, product: BarrierOption, pricing_env: PricingEnvironment
    ) -> Dict[str, float]:
        """
        Calculate Greeks as a knock-out option (for knock-in decomposition).

        Creates a temporary knock-out version and calculates Greeks.

        Args:
            product: Original barrier option
            pricing_env: Pricing environment

        Returns:
            Dictionary with price, delta, gamma for knock-out option
        """
        from quantark.util.enum import BarrierType

        # Convert knock-in type to knock-out
        if product.barrier_type == BarrierType.UP_IN:
            ko_type = BarrierType.UP_OUT
        elif product.barrier_type == BarrierType.DOWN_IN:
            ko_type = BarrierType.DOWN_OUT
        else:
            ko_type = product.barrier_type

        ko_product = BarrierOption(
            strike=product.strike,
            option_type=product.option_type,
            barrier=product.barrier,
            barrier_type=ko_type,
            maturity=product.maturity,
            exercise_date=product.exercise_date,
            settlement_date=product.settlement_date,
            rebate=0.0,  # Zero rebate for decomposition
            participation_rate=product.participation_rate,
            pay_at_hit=product.pay_at_hit,
            observation_type=product.observation_type,
            observation_dates=product.observation_dates,
            observation_schedule=product.observation_schedule,
        )

        return super().calculate_greeks(ko_product, pricing_env)

    def set_terminal_condition(
        self,
        grid: np.ndarray,
        x_vec: np.ndarray,
        s_vec: np.ndarray,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
    ) -> None:
        """
        Set the terminal condition (payoff at maturity).

        For knock-out options:
            - Zero payoff if barrier already hit
            - Standard payoff otherwise

        Args:
            grid: Solution grid [num_x, num_t]
            x_vec: Log-price grid points
            s_vec: Price grid points
            product: Barrier option
            pricing_env: Pricing environment
        """
        K = product.strike
        barrier = product.barrier
        participation = product.participation_rate

        # Calculate base payoff
        if product.is_call():
            payoff = np.maximum(s_vec - K, 0.0)
        else:
            payoff = np.maximum(K - s_vec, 0.0)
        payoff = payoff * participation

        apply_terminal_barrier = product.observation_type != ObservationType.DISCRETE
        apply_terminal_barrier = (
            apply_terminal_barrier or self._has_terminal_observation
        )

        if apply_terminal_barrier:
            if (
                product.observation_type == ObservationType.DISCRETE
                and self._terminal_schedule_records
            ):
                current_time = self._total_tau
                for rec in self._terminal_schedule_records:
                    rec_barrier = rec.barrier if rec.barrier is not None else barrier
                    cashflow_value = self._cashflow_value_at_time(
                        pricing_env=pricing_env,
                        cashflow=rec.payoff,
                        current_time=current_time,
                        settlement_time=rec.settlement_time,
                    )
                    if self._schedule_aggregation == ObservationAggregation.ACCUMULATE:
                        if product.is_up_barrier:
                            payoff[s_vec >= rec_barrier] += cashflow_value
                        else:
                            payoff[s_vec <= rec_barrier] += cashflow_value
                    else:
                        if product.is_up_barrier:
                            payoff[s_vec >= rec_barrier] = cashflow_value
                        else:
                            payoff[s_vec <= rec_barrier] = cashflow_value
                        break
            else:
                if product.is_up_barrier:
                    payoff[s_vec >= barrier] = product.rebate
                else:
                    payoff[s_vec <= barrier] = product.rebate

        grid[:, -1] = payoff

    def set_boundary_conditions(
        self,
        grid: np.ndarray,
        x_vec: np.ndarray,
        s_vec: np.ndarray,
        t_idx: int,
        tau: float,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
    ) -> None:
        """
        Set boundary conditions at spatial edges.

        For barrier options, the boundary at the barrier level
        is set to the rebate (discounted if paid at expiry).

        Args:
            grid: Solution grid [num_x, num_t]
            x_vec: Log-price grid points
            s_vec: Price grid points
            t_idx: Current time index
            tau: Time remaining to maturity
            product: Barrier option
            pricing_env: Pricing environment
        """
        K = product.strike
        total_tau = (
            self._total_tau
            if self._total_tau > 0
            else product.get_maturity(pricing_env)
        )
        current_time = self._current_time(total_tau, tau)
        df_to_maturity = self._df_between_times(pricing_env, current_time, total_tau)

        q = pricing_env.get_div_yield(tau) if tau > 0 else 0.0
        df_div = np.exp(-q * tau) if tau > 0 else 1.0
        participation = product.participation_rate

        if product.observation_type == ObservationType.DISCRETE:
            if product.is_call():
                grid[0, t_idx] = 0.0
                grid[-1, t_idx] = (
                    max(s_vec[-1] * df_div - K * df_to_maturity, 0.0) * participation
                )
            else:
                grid[0, t_idx] = K * df_to_maturity * participation
                grid[-1, t_idx] = 0.0
            return

        settlement_time = current_time if product.pay_at_hit else total_tau
        rebate_value = self._cashflow_value_at_time(
            pricing_env=pricing_env,
            cashflow=product.rebate,
            current_time=current_time,
            settlement_time=settlement_time,
        )

        if product.is_up_barrier:
            if product.is_call():
                grid[0, t_idx] = 0.0
            else:
                grid[0, t_idx] = K * df_to_maturity * participation
            grid[-1, t_idx] = rebate_value
        else:
            grid[0, t_idx] = rebate_value
            if product.is_call():
                grid[-1, t_idx] = (
                    max(s_vec[-1] * df_div - K * df_to_maturity, 0.0) * participation
                )
            else:
                grid[-1, t_idx] = 0.0

    def _apply_step_modifications(
        self,
        grid: np.ndarray,
        x_vec: np.ndarray,
        s_vec: np.ndarray,
        t_idx: int,
        tau: float,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
    ) -> None:
        """
        Apply barrier checks at each time step.

        For discrete monitoring, only check at observation times.
        For continuous monitoring, check at every time step.

        Args:
            grid: Solution grid
            x_vec: Log-price grid points
            s_vec: Price grid points
            t_idx: Current time index
            tau: Time remaining to maturity
            product: Barrier option
            pricing_env: Pricing environment
        """
        # For continuous monitoring, always apply barrier
        # For discrete, only at observation times
        if product.observation_type == ObservationType.DISCRETE:
            if t_idx not in self._observation_indices:
                return

        total_tau = (
            self._total_tau
            if self._total_tau > 0
            else product.get_maturity(pricing_env)
        )
        current_time = self._current_time(total_tau, tau)
        schedule_records = self._schedule_records.get(t_idx)

        if schedule_records:
            for rec in schedule_records:
                barrier = rec.barrier if rec.barrier is not None else product.barrier
                cashflow_value = self._cashflow_value_at_time(
                    pricing_env=pricing_env,
                    cashflow=rec.payoff,
                    current_time=current_time,
                    settlement_time=rec.settlement_time,
                )
                if self._schedule_aggregation == ObservationAggregation.ACCUMULATE:
                    if product.is_up_barrier:
                        grid[s_vec >= barrier, t_idx] += cashflow_value
                    else:
                        grid[s_vec <= barrier, t_idx] += cashflow_value
                else:
                    if product.is_up_barrier:
                        grid[s_vec >= barrier, t_idx] = cashflow_value
                    else:
                        grid[s_vec <= barrier, t_idx] = cashflow_value
                    # Stop-first-hit semantics: once applied, exit early
                    return
            return

        settlement_time = current_time if product.pay_at_hit else total_tau
        rebate_value = self._cashflow_value_at_time(
            pricing_env=pricing_env,
            cashflow=product.rebate,
            current_time=current_time,
            settlement_time=settlement_time,
        )

        # Apply barrier knockout
        if product.is_up_barrier:
            # Up barrier: knockout at high prices
            grid[s_vec >= product.barrier, t_idx] = rebate_value
        else:
            # Down barrier: knockout at low prices
            grid[s_vec <= product.barrier, t_idx] = rebate_value

    def _get_barriers(self, product: BaseEquityProduct) -> List[float]:
        """Include schedule-specific barriers when building spatial bounds."""
        barriers = super()._get_barriers(product)
        schedule = getattr(product, "observation_schedule", None)
        if schedule is not None:
            for rec in schedule.records:
                if rec.barrier is not None:
                    barriers.append(rec.barrier)
        return barriers

    def _build_grids(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        spot: float,
        sigma: float,
        tau: float,
        r: float,
        q: float,
    ):
        """
        Build grids and setup observation indices for discrete monitoring.
        """
        result = super()._build_grids(product, pricing_env, spot, sigma, tau, r, q)
        x_vec, s_vec, dx_vec, t_vec, dt_vec = result

        # Setup observation time indices for discrete monitoring
        self._total_tau = tau
        self._observation_indices.clear()
        self._schedule_records.clear()
        self._schedule_aggregation = ObservationAggregation.STOP_FIRST_HIT
        self._terminal_schedule_records = []
        self._has_terminal_observation = False

        schedule = getattr(product, "observation_schedule", None)
        if schedule is not None:
            resolved_records = schedule.resolve(
                pricing_env=pricing_env,
                default_barrier=product.barrier,
                default_payoff=product.rebate,
                require_single=True,
            )
            self._schedule_aggregation = schedule.aggregation_mode
            if self._schedule_aggregation in (
                ObservationAggregation.BEST,
                ObservationAggregation.WORST,
            ):
                raise PricingError(
                    f"PDE solver does not support aggregation mode {self._schedule_aggregation.value}"
                )
            for rec in resolved_records:
                if is_close(rec.observation_time, 0.0):
                    idx = 0
                    self._observation_indices.add(idx)
                    self._schedule_records.setdefault(idx, []).append(rec)
                elif is_close(rec.observation_time, tau):
                    self._terminal_schedule_records.append(rec)
                    self._has_terminal_observation = True
                elif 0.0 < rec.observation_time < tau:
                    idx = int(np.argmin(np.abs(t_vec - rec.observation_time)))
                    self._observation_indices.add(idx)
                    self._schedule_records.setdefault(idx, []).append(rec)
        elif (
            hasattr(product, "observation_type")
            and product.observation_type == ObservationType.DISCRETE
            and hasattr(product, "observation_dates")
            and product.observation_dates is not None
        ):
            for obs_time in product.observation_dates:
                if is_close(obs_time, 0.0):
                    self._observation_indices.add(0)
                elif is_close(obs_time, tau):
                    self._has_terminal_observation = True
                elif 0.0 < obs_time < tau:
                    idx = int(np.argmin(np.abs(t_vec - obs_time)))
                    self._observation_indices.add(idx)

        return result

    def get_critical_points(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> List[float]:
        """
        Get critical prices for grid concentration.

        For barrier options, both strike and barrier are critical.

        Args:
            product: Barrier option
            pricing_env: Pricing environment

        Returns:
            List containing strike and barrier
        """
        points = [product.strike, product.barrier]
        schedule = getattr(product, "observation_schedule", None)
        if schedule is not None:
            for rec in schedule.records:
                if rec.barrier is not None:
                    points.append(rec.barrier)
        # sort and make unique before return
        points = sorted(set(points))
        return points

    def __repr__(self):
        return "BarrierPDESolver()"
