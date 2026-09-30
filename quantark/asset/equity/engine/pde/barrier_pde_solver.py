"""
PDE solver for single barrier options.

Implements the finite difference method for knock-in and knock-out
barrier options with continuous or discrete monitoring.
"""

from typing import Dict, List

import numpy as np

from quantark.asset.equity.engine.capabilities import SettlementSupport
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option.barrier_option import BarrierOption
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType, ObservationAggregation
from quantark.util.exceptions import PricingError

from .base_pde_solver import BasePDESolver
from .core.adapters.barrier import BarrierAdapter


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

    settlement_support = SettlementSupport.EVENT_AND_TERMINAL
    _adapter_class = BarrierAdapter

    def _uses_grid_layer(self) -> bool:
        return True

    def grid_request(self, product, market, tau):
        return self._generic_grid_request(product, market, tau)

    # Discrete-monitoring state (_observation_indices, _schedule_records,
    # _terminal_schedule_records, ...) is initialized by BasePDESolver and
    # populated by the shared _setup_observation_indices.

    # _current_time / _df_between_times / _cashflow_value_at_time are
    # inherited from BasePDESolver.

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
        terminal_delay_df = self._terminal_delay_df(product, pricing_env)
        payoff = payoff * participation * terminal_delay_df

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
                        settlement_time=(
                            rec.settlement_time
                            if product.pay_at_hit
                            else self._terminal_payment_time(
                                product, pricing_env
                            )
                        ),
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
                rebate_value = product.rebate * terminal_delay_df
                if product.is_up_barrier:
                    payoff[s_vec >= barrier] = rebate_value
                else:
                    payoff[s_vec <= barrier] = rebate_value

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
        terminal_delay_df = self._terminal_delay_df(product, pricing_env)

        q = pricing_env.get_div_yield(tau) if tau > 0 else 0.0
        df_div = np.exp(-q * tau) if tau > 0 else 1.0
        participation = product.participation_rate

        if product.observation_type == ObservationType.DISCRETE:
            if product.is_call():
                grid[0, t_idx] = 0.0
                grid[-1, t_idx] = (
                    max(s_vec[-1] * df_div - K * df_to_maturity, 0.0)
                    * participation
                    * terminal_delay_df
                )
            else:
                grid[0, t_idx] = (
                    K * df_to_maturity * participation * terminal_delay_df
                )
                grid[-1, t_idx] = 0.0
            return

        settlement_time = (
            self._event_payment_time(product, pricing_env, current_time)
            if product.pay_at_hit
            else self._terminal_payment_time(product, pricing_env)
        )
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
                grid[0, t_idx] = (
                    K * df_to_maturity * participation * terminal_delay_df
                )
            grid[-1, t_idx] = rebate_value
        else:
            grid[0, t_idx] = rebate_value
            if product.is_call():
                grid[-1, t_idx] = (
                    max(s_vec[-1] * df_div - K * df_to_maturity, 0.0)
                    * participation
                    * terminal_delay_df
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
                    settlement_time=(
                        rec.settlement_time
                        if product.pay_at_hit
                        else self._terminal_payment_time(
                            product, pricing_env
                        )
                    ),
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

        settlement_time = (
            self._event_payment_time(product, pricing_env, current_time)
            if product.pay_at_hit
            else self._terminal_payment_time(product, pricing_env)
        )
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
