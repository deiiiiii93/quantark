"""
PDE solver for one-touch (and no-touch) options.

Implements the finite difference method for digital barrier options
that pay a fixed rebate on touching (or not touching) a barrier.
"""

from typing import Dict, List
import numpy as np

from quantark.asset.equity.engine.capabilities import SettlementSupport
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option.one_touch_option import OneTouchOption
from quantark.asset.equity.param import PDEParams
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType, ObservationAggregation
from quantark.util.exceptions import PricingError

from .base_pde_solver import BasePDESolver


class OneTouchPDESolver(BasePDESolver):
    """
    PDE solver for one-touch and no-touch options.

    One-touch options pay a fixed rebate if the barrier is touched.
    No-touch options pay a fixed rebate if the barrier is NOT touched.

    For one-touch (payment at expiry):
        Terminal condition: rebate at barrier side, 0 elsewhere
        Boundary at barrier: rebate (discounted if payment at expiry)

    For no-touch:
        Terminal condition: rebate inside (away from barrier), 0 at barrier
        Boundary at barrier: 0

    For one-touch with payment at hit:
        We solve for the present value of the expected rebate,
        with the barrier being an absorbing boundary with value = rebate.
    """

    settlement_support = SettlementSupport.EVENT_AND_TERMINAL

    def _uses_grid_layer(self) -> bool:
        return True

    def grid_request(self, product, market, tau):
        return self._generic_grid_request(product, market, tau)

    # Discrete-monitoring state is initialized by BasePDESolver and populated
    # by the shared _setup_observation_indices.

    def price(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        """
        Price a one-touch or no-touch option using PDE method.

        Args:
            product: One-touch option
            pricing_env: Pricing environment

        Returns:
            Option price

        Raises:
            PricingError: If product is not a one-touch option
        """
        if not isinstance(product, OneTouchOption):
            raise PricingError(
                f"OneTouchPDESolver only supports OneTouchOption, "
                f"got {type(product).__name__}"
            )

        # Check if barrier already hit
        spot = pricing_env.spot
        if product.is_barrier_hit(spot):
            if product.is_one_touch:
                settlement_time = (
                    self._event_payment_time(product, pricing_env, 0.0)
                    if product.payment_at_hit
                    else self._terminal_payment_time(product, pricing_env)
                )
                return self._cashflow_value_at_time(
                    pricing_env,
                    product.rebate,
                    0.0,
                    settlement_time,
                )
            else:
                # No-touch already failed
                return 0.0

        return super().price(product, pricing_env)

    def calculate_greeks(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> Dict[str, float]:
        """
        Calculate Greeks for a one-touch or no-touch option.

        Args:
            product: One-touch option
            pricing_env: Pricing environment

        Returns:
            Dictionary with price, delta, gamma

        Raises:
            PricingError: If product is not a one-touch option
        """
        if not isinstance(product, OneTouchOption):
            raise PricingError(
                f"OneTouchPDESolver only supports OneTouchOption, "
                f"got {type(product).__name__}"
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

        # Check if barrier already hit
        if product.is_barrier_hit(spot):
            if product.is_one_touch:
                # Already touched, fixed rebate (delta=gamma=0)
                settlement_time = (
                    self._event_payment_time(product, pricing_env, 0.0)
                    if product.payment_at_hit
                    else self._terminal_payment_time(product, pricing_env)
                )
                price = self._cashflow_value_at_time(
                    pricing_env,
                    product.rebate,
                    0.0,
                    settlement_time,
                )
                return {"price": price, "delta": 0.0, "gamma": 0.0}
            else:
                # No-touch already failed
                return {"price": 0.0, "delta": 0.0, "gamma": 0.0}

        return super().calculate_greeks(product, pricing_env)

    def set_terminal_condition(
        self,
        grid: np.ndarray,
        x_vec: np.ndarray,
        s_vec: np.ndarray,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
    ) -> None:
        """
        Set the terminal condition at maturity.

        For one-touch (payment at expiry):
            - If barrier not touched by expiry, one-touch pays 0
            - This is handled by solving backwards

        For no-touch:
            - Pays rebate at expiry if barrier was never touched
            - So terminal is rebate away from barrier, 0 at barrier

        Args:
            grid: Solution grid [num_x, num_t]
            x_vec: Log-price grid points
            s_vec: Price grid points
            product: One-touch option
            pricing_env: Pricing environment
        """
        barrier = product.barrier
        rebate = product.rebate

        # Base terminal value assuming no touch happens exactly at maturity.
        if product.is_one_touch:
            grid[:, -1] = 0.0
        else:
            grid[:, -1] = rebate * self._terminal_delay_df(
                product, pricing_env
            )

        # Apply the touch check at maturity only for continuous monitoring or
        # when the discrete schedule actually observes at t=T; otherwise this
        # would insert a phantom terminal observation.
        apply_terminal_touch = (
            product.observation_type != ObservationType.DISCRETE
            or self._has_terminal_observation
        )
        if not apply_terminal_touch:
            return

        for rec, payoff in self._resolved_terminal_payoffs(
            product,
            pricing_env,
            default_payoff=rebate,
            event_payment=(
                product.is_one_touch and product.payment_at_hit
            ),
        ):
            rec_barrier = (
                rec.barrier if rec is not None and rec.barrier is not None else barrier
            )
            touched = (
                s_vec >= rec_barrier if product.is_up_barrier else s_vec <= rec_barrier
            )
            if product.is_one_touch:
                # Mirror the interior-step aggregation semantics.
                if self._schedule_aggregation == ObservationAggregation.ACCUMULATE:
                    grid[touched, -1] += payoff
                else:
                    grid[touched, -1] = payoff
                    break
            else:
                # No-touch: any touch region pays zero (aggregation-neutral).
                grid[touched, -1] = 0.0

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

        For one-touch:
            - Barrier boundary: rebate (discounted if payment at expiry)
            - Far boundary: 0 (never reach barrier from there)

        For no-touch:
            - Barrier boundary: 0 (touched = failed)
            - Far boundary: discounted rebate

        Args:
            grid: Solution grid [num_x, num_t]
            x_vec: Log-price grid points
            s_vec: Price grid points
            t_idx: Current time index
            tau: Time remaining to maturity
            product: One-touch option
            pricing_env: Pricing environment
        """
        rebate = product.rebate

        current_time = self._current_time(self._total_tau, tau)
        terminal_payment_time = self._terminal_payment_time(
            product, pricing_env
        )
        terminal_df = self._df_between_times(
            pricing_env, current_time, terminal_payment_time
        )

        if product.is_one_touch:
            if product.payment_at_hit:
                payment_time = self._event_payment_time(
                    product, pricing_env, current_time
                )
                barrier_value = self._cashflow_value_at_time(
                    pricing_env,
                    rebate,
                    current_time,
                    payment_time,
                )
            else:
                barrier_value = rebate * terminal_df

            far_value = 0.0  # Too far from barrier to ever reach it
        else:
            # No-touch
            barrier_value = 0.0  # Touched = no payout
            far_value = rebate * terminal_df

        if product.is_up_barrier:
            # Up barrier: upper boundary is barrier
            grid[0, t_idx] = far_value
            grid[-1, t_idx] = barrier_value
        else:
            # Down barrier: lower boundary is barrier
            grid[0, t_idx] = barrier_value
            grid[-1, t_idx] = far_value

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

        Args:
            grid: Solution grid
            x_vec: Log-price grid points
            s_vec: Price grid points
            t_idx: Current time index
            tau: Time remaining to maturity
            product: One-touch option
            pricing_env: Pricing environment
        """
        # For discrete monitoring, only check at observation times
        if product.observation_type == ObservationType.DISCRETE:
            if t_idx not in self._observation_indices:
                return

        barrier = product.barrier
        rebate = product.rebate
        current_time = self._current_time(self._total_tau, tau)
        terminal_payment_time = self._terminal_payment_time(
            product, pricing_env
        )
        terminal_df = self._df_between_times(
            pricing_env, current_time, terminal_payment_time
        )

        schedule_records = self._schedule_records.get(t_idx)
        if schedule_records:
            for rec in schedule_records:
                barrier = rec.barrier if rec.barrier is not None else product.barrier
                payoff = rec.payoff
                if product.is_one_touch:
                    if product.payment_at_hit:
                        barrier_value = self._cashflow_value_at_time(
                            pricing_env=pricing_env,
                            cashflow=payoff,
                            current_time=current_time,
                            settlement_time=rec.settlement_time,
                        )
                    else:
                        barrier_value = payoff * terminal_df
                else:
                    barrier_value = 0.0
                if self._schedule_aggregation == ObservationAggregation.ACCUMULATE:
                    if product.is_up_barrier:
                        grid[s_vec >= barrier, t_idx] += barrier_value
                    else:
                        grid[s_vec <= barrier, t_idx] += barrier_value
                else:
                    if product.is_up_barrier:
                        grid[s_vec >= barrier, t_idx] = barrier_value
                    else:
                        grid[s_vec <= barrier, t_idx] = barrier_value
                    return
            return

        if product.is_one_touch:
            if product.payment_at_hit:
                barrier_value = self._cashflow_value_at_time(
                    pricing_env,
                    rebate,
                    current_time,
                    self._event_payment_time(
                        product, pricing_env, current_time
                    ),
                )
            else:
                barrier_value = rebate * terminal_df
        else:
            barrier_value = 0.0

        # Apply barrier value where barrier is hit
        if product.is_up_barrier:
            grid[s_vec >= barrier, t_idx] = barrier_value
        else:
            grid[s_vec <= barrier, t_idx] = barrier_value

    def _populate_observation_maps(self, product, pricing_env, layout, tau):
        # Discrete-monitoring bookkeeping (search-based; works on layer
        # grids). The bespoke legacy grid construction this solver carried
        # was removed with the declarative layer (0.4.0).
        self._setup_observation_indices(
            product,
            pricing_env,
            tau,
            layout.time.t,
            resolve_kwargs={
                "default_barrier": product.barrier,
                "default_payoff": product.rebate,
                "require_single": True,
            },
        )

    def get_critical_points(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> List[float]:
        """
        Get critical prices for grid concentration.

        For one-touch, only the barrier is critical.

        Args:
            product: One-touch option
            pricing_env: Pricing environment

        Returns:
            List containing the barrier
        """
        points = [product.barrier]
        schedule = getattr(product, "observation_schedule", None)
        if schedule is not None:
            for rec in schedule.records:
                if rec.barrier is not None:
                    points.append(rec.barrier)
        # sort and make unique before return
        points = sorted(set(points))
        return points

    def __repr__(self):
        return "OneTouchPDESolver()"
