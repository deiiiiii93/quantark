"""Statistics of a product knocked out at valuation: a terminal outcome, no march (spec 3B §10 step 3).

Under the cell-average projection the legacy readout overrides every column at the spot when the
valuation date's knock-out fires, so the march contributes nothing to the result. This builds the same
overridden values without a layout and hands them to the shared readout tail.
"""
from __future__ import annotations

import numpy as np

from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.util.numerical import is_close

from . import legacy
from .readout import assemble_autocallable

__all__ = ["knocked_out_statistics"]


def knocked_out_statistics(solver, product, env, request):
    """The legacy readout's numbers for a knock-out that fires at valuation, from ``request``'s streams
    and pv (``external_pv``, else the immediate knock-out payoff)."""
    tau = float(product.get_maturity(env))
    ko_records = solver._filter_observations_by_tau(product.resolve_ko_observations(env), tau)
    n_ko = len(ko_records)
    rec0_pos = next(k for k, rec in enumerate(ko_records) if is_close(float(rec.observation_time), 0.0))
    rec0 = ko_records[rec0_pos]

    streams = request.streams
    if streams is None:
        want_ki = True
        want_coupon = True
    else:
        from quantark.cashleg.event_distribution import EventType

        want_ki = bool(streams & {EventType.KI, EventType.MATURITY_WITH_KI})
        want_coupon = EventType.COUPON in streams

    # the legacy passes t_vec[0], which is 0.0 on every layout
    df_delay0 = solver._cashflow_value_at_time(
        pricing_env=env, cashflow=1.0, current_time=0.0, settlement_time=rec0.settlement_time,
    )
    ed_unit = np.array([df_delay0 if i == rec0_pos else 0.0 for i in range(n_ko)], dtype=float)

    already_knocked_in = bool(getattr(product, "_otc_lifecycle_knocked_in", False))
    if already_knocked_in:
        ki = (1.0, 1.0, np.array([0.0], dtype=float), np.array([1.0], dtype=float),
              np.array([0.0], dtype=float))
    elif want_ki:
        df_T = env.get_discount_factor(float(tau))
        ki_probability = float(0.0 / df_T) if df_T > 0.0 else 0.0
        ki = (ki_probability, ki_probability, np.array([], dtype=float), np.array([], dtype=float),
              np.array([], dtype=float))
    else:
        ki = (0.0, 0.0, np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float))

    extra_fields = {}
    if want_coupon and isinstance(product, PhoenixOption):
        # the arrays the solve would have registered (the legacy read used those)
        (solver._coupon_barriers, solver._coupon_amounts, solver._coupon_cumulative,
         solver._initial_coupon_arrears) = solver._coupon_schedule(product, env)
        overrides = legacy._t0_extra_indicator_overrides(
            solver, product, env, env.spot, n_ko, rec0_pos, rec0, np.array([0.0]), True, df_delay0,
        )
        # every coupon column is overridden at a fired knock-out, so no grid is read
        extra_fields = legacy._extract_extra_event_stats(
            solver, None, None, None, n_ko, ko_records, env, product, col_overrides=overrides,
        )

    pv = request.external_pv if request.external_pv is not None else solver._get_immediate_ko_payoff(product, env)
    return assemble_autocallable(solver, product, env, pv=pv, ko_records=ko_records, ed_unit=ed_unit,
                                 ki=ki, extra_fields=extra_fields)
