"""
Put dates on a convertible-bond time grid.

A put is exercisable on its date, which rarely coincides with a grid step. Matching steps to put dates by
calendar equality drops every put that falls between two steps. Instead each put is placed on the first step
at or after its date: a coupon dated on or before the put date is paid in the step that ends there, so the
holder who puts still receives it. The snapping error is under one step and vanishes as the grid refines.
"""
from datetime import datetime
from typing import TYPE_CHECKING, Dict

import numpy as np

if TYPE_CHECKING:  # the product module imports this package for its cash flows
    from quantark.asset.bond.product.convertible import ConvertibleBond


def put_prices_by_step(
    bond: "ConvertibleBond", valuation_date: datetime, step_times: np.ndarray
) -> Dict[int, float]:
    """
    Map each future put of ``bond`` onto the grid step where it is exercised.

    Put times count whole calendar days from the valuation date over 365, the same clock as the engines'
    ``valuation_date + int(t * 365)`` node dates. A put lands on the first step at or after its time; a put
    after the last step (possible for an intraday valuation) lands on the last step. Puts before the valuation
    date are dropped. When several puts land on one step the holder takes the largest price.

    Args:
        bond: Convertible bond carrying the put schedule
        valuation_date: Valuation date (time 0 of the grid)
        step_times: Ascending step times in years from valuation, ``step_times[0] = 0``, last = maturity

    Returns:
        Mapping of step index to put price
    """
    step_times = np.asarray(step_times, dtype=float)
    last = len(step_times) - 1
    prices: Dict[int, float] = {}
    for entry in bond.put_schedule:
        days = (entry.put_date.date() - valuation_date.date()).days
        if days < 0:
            continue
        # First step with t_k >= t_put, no tolerance: the same comparison that puts a coupon
        # at c <= t_k into an earlier step, so a coupon paid on the put date is never lost
        k = min(int(np.searchsorted(step_times, days / 365.0, side="left")), last)
        prices[k] = max(prices.get(k, entry.put_price), entry.put_price)
    return prices
