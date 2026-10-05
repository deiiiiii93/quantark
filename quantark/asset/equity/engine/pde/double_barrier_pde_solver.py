"""
PDE solver for double barrier options.

Implements the finite difference method for double knock-in and
knock-out barrier options (corridor options).
"""

from typing import Dict, Optional, List
import numpy as np

from quantark.asset.equity.engine.capabilities import SettlementSupport
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option.double_barrier_option import DoubleBarrierOption
from quantark.asset.equity.param import PDEParams
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import DoubleBarrierType, ObservationType, ObservationAggregation
from quantark.util.exceptions import PricingError

from .base_pde_solver import BasePDESolver
from .core.adapters.barrier import DoubleBarrierAdapter
from .grid import DiscreteEvent, EventOp, EventRecord


class DoubleBarrierPDESolver(BasePDESolver):
    """
    PDE solver for double barrier (corridor) options.

    Double barrier options have both upper and lower barriers.
    The option knocks out (or in) if either barrier is hit.

    For knock-out corridor options:
        - Terminal condition: payoff inside corridor, zero outside
        - Boundary conditions: rebate at both barriers

    For knock-in corridor options, we use:
        Knock-in = Vanilla - Knock-out
    """

    settlement_support = SettlementSupport.EVENT_AND_TERMINAL
    _adapter_class = DoubleBarrierAdapter

    def _uses_grid_layer(self) -> bool:
        return True

    def grid_request(self, product, market, tau):
        return self._generic_grid_request(product, market, tau)

    def __init__(self, params: Optional[PDEParams] = None):
        """
        Initialize double barrier option PDE solver.

        Args:
            params: PDE engine configuration parameters
        """
        super().__init__(params)
        # Discrete-monitoring state is initialized by BasePDESolver and
        # populated by the shared _setup_observation_indices.

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

        For double knock-out:
            - Payoff inside corridor (between barriers)
            - Rebate outside corridor

        Args:
            grid: Solution grid [num_x, num_t]
            x_vec: Log-price grid points
            s_vec: Price grid points
            product: Double barrier option
            pricing_env: Pricing environment
        """
        K = product.strike
        terminal_delay_df = self._terminal_delay_df(product, pricing_env)

        # Calculate base payoff
        if product.is_call():
            payoff = np.maximum(s_vec - K, 0.0)
        else:
            payoff = np.maximum(K - s_vec, 0.0)
        payoff *= terminal_delay_df
        payoff *= self._multiplier(product)

        # Apply the corridor check at maturity only for continuous monitoring
        # or when the discrete schedule actually observes at t=T; otherwise
        # this would insert a phantom terminal observation.
        apply_terminal_barrier = (
            product.observation_type != ObservationType.DISCRETE
            or self._has_terminal_observation
        )

        if apply_terminal_barrier:
            payoff = self._apply_discrete_event(
                payoff, x_vec, s_vec, self._terminal_event(product, pricing_env),
                nodal=product.observation_type != ObservationType.DISCRETE)
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

        For double barrier:
            - Lower boundary: rebate (at/below lower barrier)
            - Upper boundary: rebate (at/above upper barrier)

        Args:
            grid: Solution grid [num_x, num_t]
            x_vec: Log-price grid points
            s_vec: Price grid points
            t_idx: Current time index
            tau: Time remaining to maturity
            product: Double barrier option
            pricing_env: Pricing environment
        """
        rebate = product.rebate
        current_time = self._current_time(self._total_tau, tau)
        discounted_rebate = rebate * self._df_between_times(
            pricing_env,
            current_time,
            self._terminal_payment_time(product, pricing_env),
        ) * self._multiplier(product)

        # Continuous monitoring: edges sit AT the barriers (absorbing).
        # Discrete monitoring: edges sit >=5 sigma outside the corridor
        # (see grid_request), where knockout at the next observation is
        # almost sure, so the discounted rebate is the correct far-field
        # limit between observations. After the LAST observation the exact
        # far-field would relax to the vanilla continuation; the resulting
        # boundary mismatch reaches the interior only with probability
        # ~P(|5 sigma| move), which the wide domain makes negligible.
        grid[0, t_idx] = discounted_rebate
        grid[-1, t_idx] = discounted_rebate

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
            product: Double barrier option
            pricing_env: Pricing environment
        """
        # For discrete monitoring, only check at observation times
        if product.observation_type == ObservationType.DISCRETE:
            if t_idx not in self._observation_indices:
                return

        # _setup_observation_indices (the adapter's observation maps) always sets _total_tau
        # before any time step runs.
        current_time = self._current_time(self._total_tau, tau)
        event = self._step_event(product, pricing_env, t_idx, current_time)
        self._apply_discrete_event_at(
            grid, t_idx, x_vec, s_vec, event,
            nodal=product.observation_type != ObservationType.DISCRETE or t_idx == 0)

    def _terminal_event(self, product, pricing_env) -> DiscreteEvent:
        """The maturity observation's records and operations (spec 3A §5.4 table)."""
        accumulate = self._schedule_aggregation == ObservationAggregation.ACCUMULATE
        out = []
        for rec, cashflow_value in self._resolved_terminal_payoffs(
            product, pricing_env, default_payoff=product.rebate, event_payment=False,
        ):
            upper = rec.upper_barrier if rec is not None and rec.upper_barrier is not None else product.upper_barrier
            lower = rec.lower_barrier if rec is not None and rec.lower_barrier is not None else product.lower_barrier
            cashflow_value = cashflow_value * self._multiplier(product)
            if rec is not None and accumulate:
                out.append(EventRecord(lower, upper, cashflow_value, EventOp.ADD))
            else:
                out.append(EventRecord(lower, upper, cashflow_value, EventOp.SET))   # the default record: SET
                break
        return DiscreteEvent(tuple(out))

    def _step_event(self, product, pricing_env, t_idx, current_time) -> DiscreteEvent:
        """The interior observation's records and operations at node ``t_idx`` (spec 3A §5.4 table)."""
        schedule_records = self._schedule_records.get(t_idx)
        if schedule_records:
            accumulate = self._schedule_aggregation == ObservationAggregation.ACCUMULATE
            out = []
            for rec in schedule_records:
                upper = rec.upper_barrier if rec.upper_barrier is not None else product.upper_barrier
                lower = rec.lower_barrier if rec.lower_barrier is not None else product.lower_barrier
                value = self._cashflow_value_at_time(
                    pricing_env=pricing_env, cashflow=rec.payoff, current_time=current_time,
                    settlement_time=self._terminal_payment_time(product, pricing_env)) * self._multiplier(product)
                out.append(EventRecord(lower, upper, value, EventOp.ADD if accumulate else EventOp.SET))
                if not accumulate:
                    break
            return DiscreteEvent(tuple(out))
        discounted_rebate = product.rebate * self._df_between_times(
            pricing_env, current_time, self._terminal_payment_time(product, pricing_env)) * self._multiplier(product)
        return DiscreteEvent((EventRecord(product.lower_barrier, product.upper_barrier, discounted_rebate,
                                          EventOp.SET),))

    def _get_barriers(self, product: BaseEquityProduct) -> List[float]:
        """Include schedule-specific barriers when building spatial bounds."""
        barriers = super()._get_barriers(product)
        schedule = getattr(product, "observation_schedule", None)
        if schedule is not None:
            for rec in schedule.records:
                if rec.lower_barrier is not None:
                    barriers.append(rec.lower_barrier)
                if rec.upper_barrier is not None:
                    barriers.append(rec.upper_barrier)
        return barriers

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
                "default_upper": product.upper_barrier,
                "default_lower": product.lower_barrier,
                "default_payoff": product.rebate,
                "require_double": True,
            },
        )

    def get_critical_points(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> List[float]:
        """
        Get critical prices for grid concentration.

        For double barrier: strike and both barriers are critical.

        Args:
            product: Double barrier option
            pricing_env: Pricing environment

        Returns:
            List containing strike and both barriers
        """
        points = [product.strike, product.lower_barrier, product.upper_barrier]
        schedule = getattr(product, "observation_schedule", None)
        if schedule is not None:
            for rec in schedule.records:
                if rec.lower_barrier is not None:
                    points.append(rec.lower_barrier)
                if rec.upper_barrier is not None:
                    points.append(rec.upper_barrier)
        # sort and make unique before return
        points = sorted(set(points))
        return points

    def __repr__(self):
        return "DoubleBarrierPDESolver()"
