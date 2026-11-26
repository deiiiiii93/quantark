"""``EventStatistics`` of the QUAD V1 engines (unified event statistics spec §Contracts, T17).

The V1 statistics recursion marches one column per termination outcome through the price's own operators
(``SnowballQuadEngine._compute_event_stats``, ``KOResetSnowballQuadEngine._compute_event_stats``): per knock-out
record its discounted unit (one paid at its settlement), the maturity unit and cash (never knocked in) and the
knocked-in settlement unit and cash, read at the spot; the Phoenix coupon columns (``_CouponCashRows``). This module
turns them into the one result type:

* every outcome from its own column (DESIGN §4.7), the maturity row marched directly (never ``pv - Σ KO``);
* cash to pay ``discounted / DF(payment)`` on rows paid at a deterministic time; EXPIRY coupons are path-dependent
  (the coupon earned at the row, undiscounted) and the ledger books them at the ending events (the payment buckets);
* the knock-in path statistics by the valuation-date rules (DESIGN §4.5); elsewhere P(knock in before any knock-out),
  the expected first knock-in time and the per-date knock-in table from the forward adjoint pass (``v1_adjoint.py``),
  run only when the request asks for the knock-in stream (spec D17);
* the expected knocked-in loss ``principal DF(payment) P(KI settlement) - KI settlement cash``.

``pv`` is ``price()``'s value (spec D23).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from quantark.asset.equity.engine.event_stats import (
    CashLedger,
    CouponTable,
    EventStatistics,
    KnockInTable,
    ObservationTable,
    StatisticsProvenance,
)
from quantark.asset.equity.engine.knock_in_days import knock_in_day_axis, not_a_grid_time
from quantark.asset.equity.engine.pde.core.statistics import event_statistics as es
from quantark.asset.equity.engine.pde.core.statistics.adjoint import AdjointUnavailable, ZERO_TO_ROUND_OFF
from quantark.util.exceptions import PricingError
from quantark.util.numerical import Tolerance, is_zero

from .continuous_ki_days import crossed_by, split_interval_mass, survival_after_rows
from .v1_adjoint import MarchLog, forward_pass

__all__ = [
    "BGK_REASON",
    "FIRST_PASSAGE",
    "coupon_table",
    "knock_in_table",
    "statistics",
]

#: the continuous knock-in day rows' interpolation scheme: each interval's bridge mass split by the interval's own
#: Brownian first-passage law (QUAD V2's scheme, ``quad/v2/statistics.py``)
FIRST_PASSAGE = "interval_first_passage"
#: a discrete knock-in priced by the opt-in BGK continuous-barrier approximation has no per-observation mass
BGK_REASON = (
    "undefined: KnockInMonitoringMode.BGK_APPROXIMATION prices a shifted continuously monitored barrier; its "
    "knock-in mass falls between the product's observation dates and has no per-observation row"
)
PROVENANCE = StatisticsProvenance("quad_v1", "quad_v1")


def _df(env, t: float) -> float:
    return float(env.get_discount_factor(float(t)))


def knock_in_table(product, env, log: MarchLog, *, horizon: float, continuous: bool,
                   ki_records: Sequence, refusal: Optional[str] = None, result=None):
    """``(probability, expected_time, KnockInTable, expected_time_reason)`` from the forward pass (DESIGN §4.5): rows
    on the product's KI schedule (discrete), or one per trading day of the window (continuous: each interval's bridge
    mass split by its first-passage law, rows ``interpolated`` where a day boundary is not a step time).

    Raises:
        AdjointUnavailable: ``refusal`` given, or the pass does not transpose this march (reason given).
        PricingError: first knock-in mass off the product's KI schedule (an invariant).
    """
    if refusal is not None:
        raise AdjointUnavailable(refusal)
    if result is None:
        result = forward_pass(log)
    t = result.times
    df = np.array([_df(env, x) for x in t])
    event_p = result.first_knock_in / df
    interval_p = result.interval_knock_in / df
    ko_p = result.clean_knock_out / df
    if continuous:
        days, dates = knock_in_day_axis(product, env, horizon)
        steps = list(reversed(log.steps))
        starts_t = np.concatenate([[0.0], t[:-1]])
        day_starts = np.concatenate([[0.0], days[:-1]]) if days.size else np.zeros(0)

        def crossed_at(i, inside):
            s0, s1 = float(starts_t[i]), float(t[i])
            tau, alpha = steps[i].diffusion[0], steps[i].diffusion[1]
            log_barrier, _, _, reverse = steps[i].bridge
            start = result.interval_start_dual[i]
            # an exact first step starts at the spot: its law is a point mass there (``(nodes, weights)``)
            nodes, weights = start if isinstance(start, tuple) else (log.math_utils.grid, start)
            # the step's coefficients are constant: its variance accrues linearly in time
            fractions = np.clip((days[inside] - s0) / (s1 - s0), 0.0, 1.0)
            law = ((nodes, weights), 2.0 * tau * alpha, 2.0 * tau, log_barrier, reverse)
            # the dual is signed at round-off level where the note is all but knocked out (its weights carry the
            # projection's ringing): the split is linear in it, so only a law with no mass at all is refused
            return crossed_by(law, np.r_[fractions, 1.0])

        day_mass = split_interval_mass(days, t, interval_p, event_p, crossed_at, engine="QUAD V1")
        probability = day_mass
        weighted = float(np.sum(0.5 * (day_starts + days) * probability))
        grid = np.unique(np.concatenate([[0.0], t]))
        interpolated = ((not_a_grid_time(days, grid) | not_a_grid_time(day_starts, grid)) if days.size
                        else np.zeros(0, dtype=bool))
        times = days
        scheme = FIRST_PASSAGE
    else:
        if np.any(interval_p != 0.0):
            raise PricingError("the QUAD V1 forward pass found bridge mass on a discretely monitored knock-in")
        records = [r for r in ki_records
                   if r.observation_time is not None
                   and -Tolerance.ZERO <= float(r.observation_time) <= horizon + Tolerance.ZERO]
        times = np.array([float(r.observation_time) for r in records], dtype=float)
        columns = []
        for x in times:
            hit = np.flatnonzero(np.abs(t - x) < Tolerance.ZERO)
            columns.append(int(hit[0]) if hit.size else None)
        on_schedule = np.zeros(t.size, dtype=bool)
        on_schedule[[c for c in columns if c is not None]] = True
        if np.any(event_p[~on_schedule] != 0.0):
            raise PricingError("the QUAD V1 forward pass found first knock-in mass off the product's KI schedule at "
                               f"t = {t[~on_schedule & (event_p != 0.0)].tolist()}")
        probability = np.array([0.0 if c is None else event_p[c] for c in columns], dtype=float)
        dated = tuple(r.observation_date for r in records)
        dates = dated if dated and all(d is not None for d in dated) else None
        interpolated = np.zeros(times.size, dtype=bool)
        weighted = float(np.sum(times * probability))
        scheme = None
    if np.any(probability < -Tolerance.PROBABILITY):
        k = int(np.argmin(probability))
        raise AdjointUnavailable(f"undefined: the recursion's first knock-in mass is negative on the row at t = "
                                 f"{float(times[k])!r} ({float(probability[k])!r}); refine the grid")
    survival = survival_after_rows(times, probability, t, ko_p)
    table = KnockInTable(times=times, payment_times=times, dates=dates, payment_dates=dates,
                         probability=probability, survival=survival, interpolated=interpolated,
                         interpolation_scheme=scheme if bool(np.any(interpolated)) else None)
    total = float(np.sum(probability))
    if total == 0.0:
        return 0.0, None, table, es.NO_KNOCK_IN
    if is_zero(total):
        return total, None, table, ZERO_TO_ROUND_OFF
    return total, weighted / total, table, None


def coupon_table(env, terminal, observations: ObservationTable, *, probability, cash, expiry: bool,
                 earned: Optional[np.ndarray], buckets: Optional[np.ndarray], streams, unavailable):
    """The Phoenix coupon table on the observation grid and, under EXPIRY pay, the ledger's payment buckets.

    INSTANT: each coupon settles with its record, cash to pay ``discounted / DF(settlement)``. EXPIRY: the row's
    payment is path-dependent (placeholder: the terminal payment), its cash to pay the coupon earned at the row
    (``earned``, undiscounted); the ledger holds the coupon cash at each ending event (``buckets``: per knock-out
    record at its settlement, then the maturity at the terminal payment)."""
    n = observations.size
    survival = np.concatenate([[1.0], observations.survival[:-1]]) if n else np.zeros(0)
    coupon_probability = probability if es.want_coupon_probability(streams) else None
    if coupon_probability is None:
        unavailable["coupons.probability"] = es.NOT_REQUESTED
    if not expiry:
        payment = observations.payment_times
        return CouponTable(probability=coupon_probability, survival=survival, payment_times=payment,
                           payment_dates=observations.payment_dates, discounted_cash=cash,
                           cash_to_pay=es.to_pay(env, cash, payment) if n else cash,
                           payment_is_path_dependent=np.zeros(n, dtype=bool)), None
    terminal_pay = float(terminal.payment_time)
    coupons = CouponTable(
        probability=coupon_probability, survival=survival, payment_times=np.full(n, terminal_pay),
        payment_dates=None if terminal.payment_date is None else (terminal.payment_date,) * n,
        discounted_cash=cash, cash_to_pay=earned, payment_is_path_dependent=np.full(n, True),
    )
    determination = np.concatenate([observations.times, [float(terminal.determination_time)]])
    payment = np.concatenate([observations.payment_times, [terminal_pay]])
    determination_dates = (None if observations.dates is None or terminal.determination_date is None
                           else observations.dates + (terminal.determination_date,))
    payment_dates = (None if observations.payment_dates is None or terminal.payment_date is None
                     else observations.payment_dates + (terminal.payment_date,))
    ledger = CashLedger(determination_times=determination, payment_times=payment,
                        determination_dates=determination_dates, payment_dates=payment_dates,
                        discounted_cash=np.asarray(buckets, dtype=float),
                        cash_to_pay=es.to_pay(env, buckets, payment))
    return coupons, ledger


def statistics(product, env, *, family: str, pv: float, observations: ObservationTable,
               coupons: Optional[CouponTable], maturity, ki_settlement, rule: str, want_knock_in: bool,
               forward, unavailable: Dict[str, str],
               path_dependent_cash: Optional[CashLedger] = None,
               knock_in_refusal: Optional[str] = None) -> EventStatistics:
    """The result: knock-in fields by ``rule`` (``forward`` the bound pass), ledger from the tables."""
    loss = es.expected_loss(product, env, probability=ki_settlement.probability,
                            discounted_cash=ki_settlement.discounted_cash, payment_time=ki_settlement.payment_time)
    if knock_in_refusal is not None and rule == es.KnockInRule.FORWARD:
        def forward():
            raise AdjointUnavailable(knock_in_refusal)
    knock_in, ki_table = es.knock_in_fields(rule, want=want_knock_in, probability_no_ko=ki_settlement.probability,
                                            expected_loss=loss, unavailable=unavailable, forward=forward,
                                            valuation_dates=es.at_valuation_dates(product, env))
    lifetime = (float(np.sum(observations.probability * observations.times)) + maturity.probability * maturity.time
                + ki_settlement.probability * ki_settlement.time)
    return es.assemble(family=family, pv=pv, observations=observations, coupons=coupons, maturity=maturity,
                       ki_settlement=ki_settlement, knock_in=knock_in, ki_table=ki_table, expected_lifetime=lifetime,
                       provenance=PROVENANCE, unavailable=unavailable, path_dependent_cash=path_dependent_cash)
