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
from .core.adapters.barrier import OneTouchAdapter
from .grid import DiscreteEvent, EventOp, EventRecord


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
    _adapter_class = OneTouchAdapter

    def _uses_grid_layer(self) -> bool:
        return True

    def grid_request(self, product, market, tau):
        return self._generic_grid_request(product, market, tau)

    # Discrete-monitoring state is initialized by BasePDESolver and populated
    # by the shared _setup_observation_indices.

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
        # Base terminal value assuming no touch happens exactly at maturity.
        grid[:, -1], _ = self._terminal_payoff(product, pricing_env, s_vec)

        # Apply the touch check at maturity only for continuous monitoring or
        # when the discrete schedule actually observes at t=T; otherwise this
        # would insert a phantom terminal observation.
        apply_terminal_touch = (
            product.observation_type != ObservationType.DISCRETE
            or self._has_terminal_observation
        )
        if not apply_terminal_touch:
            return

        grid[:, -1] = self._apply_discrete_event(
            grid[:, -1], x_vec, s_vec, self._terminal_event(product, pricing_env),
            nodal=product.observation_type != ObservationType.DISCRETE)

    def _terminal_payoff(self, product, pricing_env, s_vec):
        """The value the march starts from, before the maturity event (spec 3C decision 30): nothing for a
        paying touch, the rebate with its terminal payment discount otherwise. ``(value, slope)``."""
        if product.is_one_touch:
            return np.zeros_like(s_vec), np.zeros_like(s_vec)
        return (np.full_like(s_vec, product.rebate * self._terminal_delay_df(product, pricing_env), dtype=float),
                np.zeros_like(s_vec))

    def _observation_resolve_kwargs(self, product) -> dict:
        """The ``resolve_kwargs`` of this solver's observation maps."""
        return {"default_barrier": product.barrier, "default_payoff": product.rebate, "require_single": True}

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
            # index 0 is the valuation instant: its observation is the readout's decision, never written into the
            # nodes (spec 3C decision 25); the surface at t = 0 is the continuation
            if t_idx == 0 or t_idx not in self._observation_indices:
                return

        current_time = self._current_time(self._total_tau, tau)
        event = self._step_event(product, pricing_env, t_idx, current_time)
        self._apply_discrete_event_at(
            grid, t_idx, x_vec, s_vec, event,
            nodal=product.observation_type != ObservationType.DISCRETE or t_idx == 0)

    @staticmethod
    def _sides(product, barrier):
        """(lower, upper) as ``_apply_step_modifications`` compared them: up >= b, down <= b."""
        return (None, barrier) if product.is_up_barrier else (barrier, None)

    def _terminal_event(self, product, pricing_env) -> DiscreteEvent:
        """The maturity observation's records and operations (spec 3A §5.4 table)."""
        accumulate = self._schedule_aggregation == ObservationAggregation.ACCUMULATE
        out = []
        for rec, payoff in self._resolved_terminal_payoffs(
            product, pricing_env, default_payoff=product.rebate,
            event_payment=(product.is_one_touch and product.payment_at_hit),
        ):
            rec_barrier = rec.barrier if rec is not None and rec.barrier is not None else product.barrier
            sides = self._sides(product, rec_barrier)
            if product.is_one_touch:
                if accumulate:
                    out.append(EventRecord(*sides, payoff, EventOp.ADD))
                else:
                    out.append(EventRecord(*sides, payoff, EventOp.SET))
                    break
            else:
                out.append(EventRecord(*sides, 0.0, EventOp.SET))   # no-touch: every record zeroes its region
        return DiscreteEvent(tuple(out))

    def _step_event(self, product, pricing_env, t_idx, current_time) -> DiscreteEvent:
        """The interior observation's records and operations at node ``t_idx`` (spec 3A §5.4 table)."""
        return self._event_for_records(product, pricing_env, self._schedule_records.get(t_idx), current_time)

    def _event_for_records(self, product, pricing_env, records, current_time, aggregation=None) -> DiscreteEvent:
        """The event of one observation from its resolved ``records`` at ``current_time``; empty or ``None``
        records give the default record (spec 3C decision 25: one builder for the march and the decision).
        ``aggregation=None`` reads the solver's schedule aggregation."""
        terminal_df = self._df_between_times(
            pricing_env, current_time, self._terminal_payment_time(product, pricing_env))
        schedule_records = records
        if schedule_records:
            accumulate = (self._schedule_aggregation if aggregation is None
                          else aggregation) == ObservationAggregation.ACCUMULATE
            out = []
            for rec in schedule_records:
                barrier = rec.barrier if rec.barrier is not None else product.barrier
                if product.is_one_touch:
                    if product.payment_at_hit:
                        value = self._cashflow_value_at_time(pricing_env=pricing_env, cashflow=rec.payoff,
                                                             current_time=current_time,
                                                             settlement_time=rec.settlement_time)
                    else:
                        value = rec.payoff * terminal_df
                else:
                    value = 0.0
                out.append(EventRecord(*self._sides(product, barrier), value,
                                       EventOp.ADD if accumulate else EventOp.SET))
                if not accumulate:
                    break
            return DiscreteEvent(tuple(out))
        if product.is_one_touch:
            if product.payment_at_hit:
                value = self._cashflow_value_at_time(
                    pricing_env, product.rebate, current_time,
                    self._event_payment_time(product, pricing_env, current_time))
            else:
                value = product.rebate * terminal_df
        else:
            value = 0.0
        return DiscreteEvent((EventRecord(*self._sides(product, product.barrier), value, EventOp.SET),))

    def _populate_observation_maps(self, product, pricing_env, layout, tau):
        # Discrete-monitoring bookkeeping (search-based; works on layer
        # grids). The bespoke legacy grid construction this solver carried
        # was removed with the declarative layer (0.4.0).
        self._setup_observation_indices(
            product,
            pricing_env,
            tau,
            layout.time.t,
            resolve_kwargs=self._observation_resolve_kwargs(product),
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
