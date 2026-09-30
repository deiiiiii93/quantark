"""American (``american_pde_solver.py``): exercise dates, payment timing and the valuation obstacle."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np

from quantark.util.exceptions import PricingError

from ..prepare import Operation, TerminalOutcome
from .base import FamilyAdapter

__all__ = ["AmericanAdapter"]


class AmericanAdapter(FamilyAdapter):
    """``price`` moved verbatim; ``_prepare_solve_state`` runs on every operation (the
    base ``solve_state``), so Greeks on a fresh solver resolve the exercise dates too."""

    def validate(self, product, env, request, lifecycle_state) -> None:
        from quantark.asset.equity.product.option.american_option import AmericanOption

        if request.operation is Operation.PRICE and not isinstance(product, AmericanOption):
            raise PricingError(
                f"AmericanPDESolver only supports AmericanOption, "
                f"got {type(product).__name__}"
            )

    def terminal_outcome(self, product, env, request, lifecycle_state):
        if request.operation is not Operation.PRICE:
            # calculate_greeks and the session seam: the base intrinsic, no delay factor
            return super().terminal_outcome(product, env, request, lifecycle_state)
        from quantark.asset.equity.engine.settlement_support import resolve_american_exercise_timings

        solver = self.solver
        tau = product.get_maturity(env)
        if tau > 0.0:
            return None
        timings = resolve_american_exercise_timings(
            product,
            env,
            np.array([0.0]),
            exercise_dates=solver._exercise_dates,
        )
        solver._exercise_timings = timings
        return TerminalOutcome("expired", product.get_payoff(env.spot) * timings.delay_dfs[0])

    def prepare_events(self, product, env, layout, tau, request) -> tuple:
        # The deleted _time_stepping override: resolve exercise payment timing once
        from quantark.asset.equity.engine.settlement_support import (american_exercise_requires_dates,
                                                                     resolve_american_exercise_timings)
        solver = self.solver
        t_vec = layout.time.t
        node_dates = None
        if american_exercise_requires_dates(product):
            dates = solver._exercise_dates
            date_times = solver._exercise_date_times
            if dates is None or date_times is None:
                raise PricingError(
                    "American exercise dates must be resolved before "
                    "building the PDE grid"
                )
            mapped: list[Optional[datetime]] = [None] * len(t_vec)
            for date, date_time in zip(dates, date_times):
                index = int(np.searchsorted(t_vec, date_time))
                if (
                    index >= len(t_vec)
                    or float(t_vec[index]) != float(date_time)
                ):
                    raise PricingError(
                        "American exercise date is not an exact PDE time node"
                    )
                mapped[index] = date
            node_dates = tuple(mapped)

        timings = resolve_american_exercise_timings(
            product,
            env,
            t_vec,
            exercise_dates=node_dates,
        )
        solver._exercise_timings = timings

        eligible_indices = np.flatnonzero(timings.eligible)
        boundary_delay_dfs = np.empty(len(t_vec), dtype=float)
        for index, node_time in enumerate(t_vec):
            positions = eligible_indices[eligible_indices >= index]
            next_index = int(
                positions[0] if positions.size else eligible_indices[-1]
            )
            boundary_delay_dfs[index] = (
                timings.payment_dfs[next_index]
                / env.get_discount_factor(float(node_time))
            )
        solver._exercise_boundary_delay_dfs = boundary_delay_dfs
        return ()

    def read(self, prepared, result):
        value = super().read(prepared, result)
        if prepared.request.operation is not Operation.PRICE:
            return value            # calculate_greeks: the base read, no obstacle max
        timings = self.solver._exercise_timings
        if timings is None:
            raise PricingError(
                "American exercise timings were not retained by the PDE solve"
            )
        valuation_obstacle = (
            prepared.product.intrinsic_value(prepared.env.spot)
            * timings.delay_dfs[0]
        )
        return max(value, valuation_obstacle)
