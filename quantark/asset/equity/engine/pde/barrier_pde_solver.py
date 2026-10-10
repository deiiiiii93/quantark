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
from quantark.util.numerical import is_close, is_greater_than_or_close

from .base_pde_solver import BasePDESolver
from .core.adapters.barrier import BarrierAdapter
from .grid import DiscreteEvent, EventOp, EventRecord


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
        payoff, _ = self._terminal_payoff(product, pricing_env, s_vec)

        # continuous monitoring, EXPIRY (its one observation is maturity), a schedule that observes at maturity
        apply_terminal_barrier = (self._monitored_continuously(product)
                                  or product.observation_type == ObservationType.EXPIRY
                                  or self._has_terminal_observation)

        if apply_terminal_barrier:
            payoff = self._apply_discrete_event(
                payoff, x_vec, s_vec, self._terminal_event(product, pricing_env),
                nodal=self._monitored_continuously(product))
        grid[:, -1] = payoff

    def _vanilla_solver(self):
        """The solver of a knock-in's vanilla part (spec 3C §5.3: the local-vol barrier overrides it)."""
        from .european_pde_solver import EuropeanPDESolver
        return EuropeanPDESolver(self.params)

    def _terminal_payoff(self, product, pricing_env, s_vec):
        """The payoff the march starts from, before the maturity event (spec 3C decision 30): per contract, with
        the participation and the terminal payment discount. ``(value, slope)`` on ``s_vec``."""
        K = product.strike
        participation = product.participation_rate
        terminal_delay_df = self._terminal_delay_df(product, pricing_env)
        m = self._multiplier(product)
        if product.is_call():
            payoff, slope = np.maximum(s_vec - K, 0.0), np.where(s_vec > K, 1.0, 0.0)
        else:
            payoff, slope = np.maximum(K - s_vec, 0.0), np.where(s_vec < K, -1.0, 0.0)
        return payoff * participation * terminal_delay_df * m, slope * participation * terminal_delay_df * m

    @staticmethod
    def _monitored_continuously(product) -> bool:
        """Continuous monitoring: the barrier is the domain edge. DISCRETE and EXPIRY observe on dates (P6)."""
        return product.observation_type == ObservationType.CONTINUOUS

    def _observation_timeline(self, product, pricing_env):
        """(time, level, cash, settlement time) of each observation after the valuation instant, in time order: the
        first record of a node (stop-first-hit), a bare observation date's product defaults, EXPIRY's one record at
        maturity (spec 3C decision 9, P7, P15)."""
        terminal_pay = self._terminal_payment_time(product, pricing_env)
        out = []
        if product.observation_type == ObservationType.EXPIRY:
            out.append((self._total_tau, product.barrier, product.rebate, terminal_pay))
            return tuple(out)
        schedule = getattr(product, "observation_schedule", None)
        if schedule is not None:
            nodes = [recs[0] for idx, recs in sorted(self._schedule_records.items()) if idx != 0 and recs]
            if self._terminal_schedule_records:
                nodes.append(self._terminal_schedule_records[0])
            for rec in nodes:
                level = rec.barrier if rec.barrier is not None else product.barrier
                settle = rec.settlement_time if product.pay_at_hit else terminal_pay
                out.append((float(rec.observation_time), float(level), float(rec.payoff), float(settle)))
            return tuple(sorted(out))
        for t in sorted(float(x) for x in (getattr(product, "observation_dates", None) or ())):
            if is_close(t, 0.0) or t < 0.0 or (t > self._total_tau and not is_close(t, self._total_tau)):
                continue
            settle = self._event_payment_time(product, pricing_env, t) if product.pay_at_hit else terminal_pay
            out.append((t, float(product.barrier), float(product.rebate), float(settle)))
        return tuple(out)

    def _next_observation(self, current_time):
        """(level, cash, settlement time) of the first observation at or after ``current_time``, else None."""
        for t, level, cash, settle in getattr(self, "_far_field_timeline", ()):
            if is_greater_than_or_close(t, current_time):
                return level, cash, settle
        return None

    def _certain_knock_out_row(self, grid, s_vec, t_idx, current_time, product, pricing_env) -> None:
        """Beyond a discretely monitored barrier the knock-out at the next observation is certain (spec 3C §5.2 (b)):
        the row on the barrier's side is that record's cash from its payment time, per contract, when the edge lies
        at or beyond its level. First-hit only: under ACCUMULATE nothing knocks out and the row stays today's."""
        if not product.is_knock_out or self._schedule_aggregation == ObservationAggregation.ACCUMULATE:
            return
        nxt = self._next_observation(current_time)
        if nxt is None:
            return
        level, cash, settle = nxt
        value = self._cashflow_value_at_time(pricing_env, cash, current_time, settle) * self._multiplier(product)
        if product.is_up_barrier and s_vec[-1] >= level:
            grid[-1, t_idx] = value
        elif not product.is_up_barrier and s_vec[0] <= level:
            grid[0, t_idx] = value

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

        # forward carry over [t, T] (spec 3A §5.10): exp(-(q(T) T - q(t) t)), never exp(-q(tau) tau)
        df_div = self._carry_df_between_times(pricing_env, current_time, total_tau)
        participation = product.participation_rate
        m = self._multiplier(product)

        if not self._monitored_continuously(product):
            if product.is_call():
                grid[0, t_idx] = 0.0
                grid[-1, t_idx] = (
                    max(s_vec[-1] * df_div - K * df_to_maturity, 0.0)
                    * participation
                    * terminal_delay_df
                    * m
                )
            else:
                grid[0, t_idx] = (
                    max(K * df_to_maturity - s_vec[0] * df_div, 0.0)
                    * participation
                    * terminal_delay_df
                    * m
                )
                grid[-1, t_idx] = 0.0
            self._certain_knock_out_row(grid, s_vec, t_idx, current_time, product, pricing_env)
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
        ) * m

        if product.is_up_barrier:
            if product.is_call():
                grid[0, t_idx] = 0.0
            else:
                grid[0, t_idx] = (
                    max(K * df_to_maturity - s_vec[0] * df_div, 0.0)
                    * participation
                    * terminal_delay_df
                    * m
                )
            grid[-1, t_idx] = rebate_value
        else:
            grid[0, t_idx] = rebate_value
            if product.is_call():
                grid[-1, t_idx] = (
                    max(s_vec[-1] * df_div - K * df_to_maturity, 0.0)
                    * participation
                    * terminal_delay_df
                    * m
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
        if not self._monitored_continuously(product):
            # EXPIRY observes at maturity only (the terminal); DISCRETE at its nodes; index 0 is the decision (the
            # valuation instant's observation is never written into the nodes, spec 3C decision 25)
            if (product.observation_type == ObservationType.EXPIRY or t_idx == 0
                    or t_idx not in self._observation_indices):
                return

        total_tau = (
            self._total_tau
            if self._total_tau > 0
            else product.get_maturity(pricing_env)
        )
        current_time = self._current_time(total_tau, tau)
        event = self._step_event(product, pricing_env, t_idx, current_time)
        self._apply_discrete_event_at(
            grid, t_idx, x_vec, s_vec, event,
            nodal=self._monitored_continuously(product))

    @staticmethod
    def _sides(product, barrier):
        """(lower, upper) as ``_apply_step_modifications`` compared them: up >= b, down <= b."""
        return (None, barrier) if product.is_up_barrier else (barrier, None)

    def _terminal_event(self, product, pricing_env) -> DiscreteEvent:
        """The maturity observation's records and operations (spec 3A §5.4 table)."""
        if product.observation_type == ObservationType.DISCRETE and self._terminal_schedule_records:
            accumulate = self._schedule_aggregation == ObservationAggregation.ACCUMULATE
            out = []
            for rec in self._terminal_schedule_records:
                rec_barrier = rec.barrier if rec.barrier is not None else product.barrier
                value = self._cashflow_value_at_time(
                    pricing_env=pricing_env, cashflow=rec.payoff, current_time=self._total_tau,
                    settlement_time=(rec.settlement_time if product.pay_at_hit
                                     else self._terminal_payment_time(product, pricing_env))) * self._multiplier(product)
                out.append(EventRecord(*self._sides(product, rec_barrier), value,
                                       EventOp.ADD if accumulate else EventOp.SET))
                if not accumulate:
                    break                                   # stop-first-hit: the first record only
            return DiscreteEvent(tuple(out))
        rebate_value = product.rebate * self._terminal_delay_df(product, pricing_env) * self._multiplier(product)
        return DiscreteEvent((EventRecord(*self._sides(product, product.barrier), rebate_value, EventOp.SET),))

    def _step_event(self, product, pricing_env, t_idx, current_time) -> DiscreteEvent:
        """The interior observation's records and operations at node ``t_idx`` (spec 3A §5.4 table)."""
        return self._event_for_records(product, pricing_env, self._schedule_records.get(t_idx), current_time)

    def _event_for_records(self, product, pricing_env, records, current_time, aggregation=None) -> DiscreteEvent:
        """The event of one observation from its resolved ``records`` at ``current_time``; empty or ``None``
        records give the default record (spec 3C decision 25: one builder for the march and the decision).
        ``aggregation=None`` reads the solver's schedule aggregation."""
        schedule_records = records
        if schedule_records:
            accumulate = (self._schedule_aggregation if aggregation is None
                          else aggregation) == ObservationAggregation.ACCUMULATE
            out = []
            for rec in schedule_records:
                barrier = rec.barrier if rec.barrier is not None else product.barrier
                value = self._cashflow_value_at_time(
                    pricing_env=pricing_env, cashflow=rec.payoff, current_time=current_time,
                    settlement_time=(rec.settlement_time if product.pay_at_hit
                                     else self._terminal_payment_time(product, pricing_env))) * self._multiplier(product)
                out.append(EventRecord(*self._sides(product, barrier), value,
                                       EventOp.ADD if accumulate else EventOp.SET))
                if not accumulate:
                    break
            return DiscreteEvent(tuple(out))
        settlement_time = (
            self._event_payment_time(product, pricing_env, current_time)
            if product.pay_at_hit
            else self._terminal_payment_time(product, pricing_env)
        )
        rebate_value = self._cashflow_value_at_time(
            pricing_env=pricing_env, cashflow=product.rebate, current_time=current_time,
            settlement_time=settlement_time) * self._multiplier(product)
        return DiscreteEvent((EventRecord(*self._sides(product, product.barrier), rebate_value, EventOp.SET),))

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


def barrier_expiry_outcome(product, pricing_env, greeks: bool = False):
    """This route on a barrier option at its exercise instant: its maturity outcome (the payoff at the spot, then the
    decision there -- knocked out, knocked in or not -- times the terminal payment discount; delta the slope, gamma
    0), decided before any grid or vol is read. The vol-model 2D barrier PDE routes settle on it before their model
    runs (W9, ruling 2026-10-10)."""
    from quantark.asset.equity.param import PDEParams

    solver = BarrierPDESolver(PDEParams())
    if greeks:
        return solver.calculate_greeks(product, pricing_env)
    return solver.price(product, pricing_env)
