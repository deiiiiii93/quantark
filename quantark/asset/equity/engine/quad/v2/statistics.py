"""``EventStatistics`` of the QUAD V2 engines (unified event statistics spec §Contracts, T16).

The statistics context compiles one backward channel per termination outcome beside the price (``adapters.py``,
``components="events"``): per knock-out record its probability (``1 / DF(t)`` paid on the knock-out) and redemption
cash, per Phoenix row the coupon probability, the coupon cash (memory arrears and the coupon paid with a knock-out
included, attributed to the earning row), the coupon paid at each ending event and, under EXPIRY pay, the coupon earned
at the row undiscounted; the maturity (never knocked in) and knocked-in settlement units and cash. This module reads
them at the spot into the one result type:

* every outcome from its own channel (DESIGN §4.7); cash to pay ``discounted / DF(payment)`` on rows paid at a
  deterministic time, the undiscounted earned-coupon channel on EXPIRY rows (path-dependent payment), whose cash the
  ledger books at the ending events;
* the knock-in path statistics by the valuation-date rules (DESIGN §4.5); elsewhere P(knock in before any knock-out),
  the expected first knock-in time and the per-date knock-in table from the forward adjoint pass (``adjoint.py``), run
  only when the request asks for the knock-in stream (spec D17);
* the expected knocked-in loss ``principal DF(payment) P(KI settlement) - KI settlement cash``;
* receivables (DESIGN §4.9c): the lifecycle state's pending flows, inside the cash identity, never events; a
  terminated note reports them alone, or nothing when they are settled (spec AC17).

``pv`` is the statistics context's price channel, the value ``price()`` reads from the same recursion (spec D23).
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Dict, Optional

import numpy as np

from quantark.asset.equity.engine.event_stats import (
    CashLedger,
    CouponTable,
    EventStatistics,
    KnockInStatistics,
    KnockInTable,
    ObservationTable,
    OutcomeRow,
    ReceivablesTable,
    StatisticsProvenance,
    build_cash_ledger,
)
from quantark.asset.equity.engine.knock_in_days import knock_in_day_axis, not_a_grid_time
from quantark.asset.equity.engine.pde.core.statistics import event_statistics as es
from quantark.asset.equity.engine.pde.core.statistics.adjoint import (
    ZERO_TO_ROUND_OFF,
    AdjointUnavailable,
    split_same_instant_rows,
)
from quantark.asset.equity.engine.settlement_support import first_schedule_settlement, resolve_terminal_timing
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.util.exceptions import PricingError, ValidationError
from quantark.util.numerical import Tolerance, is_close, is_zero

from ..continuous_ki_days import crossed_by, split_interval_mass, survival_after_rows
from .adjoint import forward_pass
from .context import interval_moments

__all__ = [
    "FIRST_PASSAGE",
    "family_of",
    "knock_in_table",
    "receivables_table",
    "terminated_statistics",
    "v2_event_statistics",
]

#: the continuous knock-in day rows' interpolation scheme: each interval's crossing mass split by the interval's own
#: Brownian first-passage law (the law the reflection kernel prices with)
FIRST_PASSAGE = "interval_first_passage"


def family_of(product) -> str:
    if isinstance(product, PhoenixOption):
        return "phoenix"
    if isinstance(product, KnockOutResetSnowballOption):
        return "ko_reset"
    return "snowball"


def _df(env, t: float) -> float:
    return float(env.get_discount_factor(float(t)))


# --- receivables (DESIGN §4.9c) -------------------------------------------------------------------------------------

#: a pending lifecycle flow's receivable source
_RECEIVABLE_SOURCE = {"KNOCK_OUT": "ko", "COUPON": "coupon"}


def receivables_table(lifecycle_state, env, *, knocked_in: bool) -> ReceivablesTable:
    """The lifecycle state's pending flows as receivables: cash decided at or before the valuation and paid after it,
    discounted with the pending-receivable timing ``price()`` adds (``SettlementResolver.resolve_pending``).

    A maturity / expiry flow is the knocked-in settlement when the state is knocked in, else the maturity.

    Raises:
        ValidationError: a pending flow of a kind no receivable source names (a knock-in carries no cash).
    """
    point = getattr(lifecycle_state, "valuation_point", None)
    if lifecycle_state is None or point is None:
        return ReceivablesTable.empty()
    from quantark.asset.equity.settlement import SettlementResolver
    from quantark.util.calendar.day_counter import calculate_year_fraction

    rows = []
    for cashflow in lifecycle_state.ledger.pending(point):
        name = getattr(cashflow.event_type, "name", None)
        if name in ("MATURITY", "EXPIRY"):
            source = "ki_settlement" if knocked_in else "maturity"
        else:
            source = _RECEIVABLE_SOURCE.get(name)
        if source is None:
            raise ValidationError(
                f"pending lifecycle flow {cashflow.cashflow_id!r} is a {cashflow.event_type!r} receivable; an "
                "autocallable's receivables are knock-out, coupon and maturity flows"
            )
        timing = SettlementResolver.resolve_pending(cashflow, env, valuation_point=point)
        if point.date is not None:
            decided = cashflow.determination_date
            decision = 0.0 if decided == env.valuation_date else -float(calculate_year_fraction(
                decided, env.valuation_date, env.day_count_convention, env.bus_days_in_year,
                calendar=getattr(env, "calendar", None)))
        else:
            decided = None
            decision = float(cashflow.determination_time) - float(point.time)
        rows.append((source, min(decision, 0.0), decided, float(timing.payment_time), timing.payment_date,
                     float(cashflow.amount) * float(timing.payment_df), float(cashflow.amount)))
    if not rows:
        return ReceivablesTable.empty()

    def dates(column):
        values = tuple(row[column] for row in rows)
        return values if all(v is not None for v in values) else None

    return ReceivablesTable(
        source=tuple(row[0] for row in rows), decision_times=[row[1] for row in rows], decision_dates=dates(2),
        payment_times=[row[3] for row in rows], payment_dates=dates(4),
        discounted_cash=[row[5] for row in rows], cash_to_pay=[row[6] for row in rows],
    )


def terminated_statistics(product, env, lifecycle_state, *, pv: float, provenance: StatisticsProvenance,
                          knocked_in: bool) -> Optional[EventStatistics]:
    """A knocked-out or matured lifecycle state (spec AC17, DESIGN §4.9c): its receivables only, or ``None`` when
    nothing is left unpaid. Every termination probability is 0 (the type does not check the probability identity of
    a terminated note); the knock-in path statistics follow the rule of a note knocked out at valuation."""
    receivables = receivables_table(lifecycle_state, env, knocked_in=knocked_in)
    if receivables.size == 0:
        return None
    family = family_of(product)
    observations = ObservationTable(times=[], payment_times=[], is_ko_obs=[], is_coupon_obs=[],
                                    schedule=() if family == "ko_reset" else None, probability=[], survival=[],
                                    discounted_cash=[], cash_to_pay=[], payment_is_path_dependent=[])
    coupons = (CouponTable(probability=[], survival=[], payment_times=[], discounted_cash=[], cash_to_pay=[],
                           payment_is_path_dependent=[]) if family == "phoenix" else None)
    nothing = dict(time=0.0, payment_time=0.0, probability=0.0, discounted_cash=0.0, cash_to_pay=0.0)
    unavailable: Dict[str, str] = {}
    if knocked_in:
        dates = es.at_valuation_dates(product, env)
        ki_table = KnockInTable(times=[0.0], payment_times=[0.0], dates=dates, payment_dates=dates, probability=[1.0],
                                survival=[0.0], interpolated=[False])
    else:
        ki_table = KnockInTable.empty()
        unavailable["knock_in.expected_time"] = es.NO_KNOCK_IN
    knock_in = KnockInStatistics(probability=1.0 if knocked_in else 0.0, probability_no_ko=0.0,
                                 expected_time=0.0 if knocked_in else None, expected_loss=0.0)
    ledger = CashLedger(
        determination_times=receivables.decision_times, payment_times=receivables.payment_times,
        determination_dates=receivables.decision_dates, payment_dates=receivables.payment_dates,
        discounted_cash=receivables.discounted_cash, cash_to_pay=receivables.cash_to_pay,
    )
    return EventStatistics(
        product_family=family, pv=float(pv), observations=observations, coupons=coupons,
        maturity=OutcomeRow(**nothing), ki_settlement=OutcomeRow(**nothing), knock_in=knock_in, ki_table=ki_table,
        receivables=receivables, ledger=ledger, expected_lifetime=0.0, provenance=provenance,
        unavailable=unavailable,
    )


# --- the valuation-date rule (DESIGN §4.5) --------------------------------------------------------------------------


def _valuation_decisions(context, spot):
    """``(knocked_out_now, knocked_in_now, carried)`` at the valuation instant, from the recursion's own first event
    (an event at t = 0 is read deterministically at the spot) and continuous monitoring through the spot."""
    log = context._march
    contract = log.contract
    n_memory = int(context.diagnostics["n_memory"])
    init = contract.initial_state
    carried = init >= n_memory
    x0 = float(np.log(spot))
    events = contract.events
    channels = contract.channels
    ko_channels = [i for i, name in enumerate(channels) if name.startswith("ko_probability:")]
    ki_terminal = channels.index("terminal_ki_probability")
    pairs = dict(contract.ki_pairs)
    state = init
    knocked_in_now = False
    continuous = contract.continuous_barrier is not None
    if continuous and not carried:
        barrier, reverse = contract.continuous_barrier, contract.reverse
        mean = float(log.means[0]) if float(log.variances[0]) == 0 else 0.0
        boundary = (min(barrier, barrier - mean) if reverse else max(barrier, barrier - mean))
        through = x0 >= boundary if reverse else x0 <= boundary
        if through:
            knocked_in_now = True
            state = pairs[init]
    first = events[0]
    if not (first.time == 0.0 and float(log.variances[0]) == 0):
        return False, knocked_in_now, carried
    y = x0 + float(log.means[0])
    breaks = tuple(first.breaks)
    if y in breaks and first.boundary_rows and not (continuous and not carried):
        action = first.boundary_rows[state][breaks.index(y)]
    else:
        action = first.rows[state][int(np.searchsorted(np.asarray(breaks, dtype=float), y, side="right"))]
    knocked_out_now = any(action.cash[c] != 0.0 for c in ko_channels)
    if not carried and state < n_memory:
        n_note = 2 * n_memory
        knocked_in_now = knocked_in_now or any(n_memory <= k < n_note for k, _ in action.terms) or \
            action.cash[ki_terminal] != 0.0
    return knocked_out_now, knocked_in_now, carried


# --- the knock-in table from the forward pass -----------------------------------------------------------------------


def knock_in_table(product, env, context, spot):
    """``(probability, expected_time, KnockInTable, expected_time_reason)`` from the forward pass (DESIGN §4.5): rows
    on the product's KI schedule (discrete), or one per trading day of the window (continuous, each interval's crossing
    mass split by its first-passage law, rows ``interpolated`` where a day boundary is not an event time).

    Raises:
        AdjointUnavailable: the pass does not transpose this route (reason given).
        PricingError: first knock-in mass off the product's KI schedule (an invariant).
    """
    log = context._march
    metadata = context.diagnostics
    contract = log.contract
    result = forward_pass(log, spot)
    t = result.times
    df = np.array([_df(env, x) for x in t])
    event_p = result.first_knock_in / df
    interval_p = result.interval_knock_in / df
    ko_p = result.clean_knock_out / df
    horizon = float(metadata["ki_horizon"])
    if contract.continuous_barrier is not None:
        days, dates = knock_in_day_axis(product, env, horizon)
        barrier, reverse = contract.continuous_barrier, contract.reverse
        # the cumulative variance on the environment's own clock at every day end and event time (one call)
        knots = np.unique(np.concatenate([days, t[t > 0.0]]))
        cumulative = dict(zip(knots.tolist(), np.cumsum(interval_moments(env, product.strike, knots.tolist())[1])))
        cumulative[0.0] = 0.0

        starts_t = np.concatenate([[0.0], t[:-1]])
        day_starts = np.concatenate([[0.0], days[:-1]]) if days.size else np.zeros(0)

        def crossed_at(i, inside):
            (points, weights), mean, variance = result.interval_law[i]
            v0 = cumulative[float(starts_t[i])]
            fractions = np.clip((np.array([cumulative[float(days[d])] for d in inside]) - v0) / variance, 0.0, 1.0)
            return crossed_by(((points, weights), mean, variance, barrier, reverse), np.r_[fractions, 1.0])

        day_mass = split_interval_mass(days, t, interval_p, event_p, crossed_at, engine="QUAD V2")
        probability = day_mass
        midpoint_mass = float(np.sum(0.5 * (day_starts + days) * probability))
        grid = np.unique(np.concatenate([[0.0], t]))
        interpolated = (not_a_grid_time(days, grid) | not_a_grid_time(day_starts, grid)) if days.size else \
            np.zeros(0, dtype=bool)
        times = days
        total = float(np.sum(probability))
        scheme = FIRST_PASSAGE
    else:
        if np.any(interval_p != 0.0):
            raise PricingError("the QUAD V2 forward pass found continuous crossing mass on a discretely monitored "
                               "knock-in")
        records = [r for r in metadata["ki_records"] if r[1] <= horizon + Tolerance.ZERO]
        index = {float(x): i for i, x in enumerate(t)}
        columns = [index.get(float(r[0])) for r in records]
        if any(c is None for c in columns):
            raise PricingError("a QUAD V2 knock-in observation is not an event of the recursion")
        off = np.ones(t.size, dtype=bool)
        off[columns] = False
        if np.any(event_p[off] != 0.0):
            raise PricingError("the QUAD V2 forward pass found first knock-in mass off the product's KI schedule at "
                               f"t = {t[off & (event_p != 0.0)].tolist()}")
        times = np.array([float(r[1]) for r in records], dtype=float)
        probability = np.array([event_p[c] for c in columns], dtype=float)
        # records sharing an env instant (0.16.0 zero-gap rule): the instant's first knock-in on the first record
        # (contract order) whose level is breached -- each band's own mass (``Event.ki_split``)
        probability = split_same_instant_rows(
            product, [SimpleNamespace(observation_time=float(r[1]), barrier=r[3]) for r in records], columns,
            probability, event_p,
            lambda c: None if c not in result.knock_in_parts
            else np.cumsum(result.knock_in_parts[c] / df[c])[:-1])
        dated = tuple(r[2] for r in records)
        dates = dated if dated and all(d is not None for d in dated) else None
        interpolated = np.zeros(times.size, dtype=bool)
        total = float(np.sum(probability))
        midpoint_mass = float(np.sum(times * probability))
        scheme = None
    if np.any(probability < -Tolerance.PROBABILITY):
        k = int(np.argmin(probability))
        raise AdjointUnavailable(f"undefined: the recursion's first knock-in mass is negative on the row at t = "
                                 f"{float(times[k])!r} ({float(probability[k])!r}); refine the mesh")
    survival = survival_after_rows(times, probability, t, ko_p)
    table = KnockInTable(times=times, payment_times=times, dates=dates, payment_dates=dates,
                         probability=probability, survival=survival, interpolated=interpolated,
                         interpolation_scheme=scheme if bool(np.any(interpolated)) else None)
    if total == 0.0:
        return 0.0, None, table, es.NO_KNOCK_IN
    if is_zero(total):
        return total, None, table, ZERO_TO_ROUND_OFF
    return total, midpoint_mass / total, table, None


# --- the result -----------------------------------------------------------------------------------------------------


def v2_event_statistics(product, env, context, values: Dict[str, float], *, pv: float, spot: float,
                        lifecycle_state, streams, provenance: StatisticsProvenance) -> EventStatistics:
    """``EventStatistics`` of a statistics context read at ``spot`` (``values``: its channels there)."""
    metadata = context.diagnostics
    family = family_of(product)
    phoenix, reset = family == "phoenix", family == "ko_reset"
    records = list(metadata["ko_metadata"])
    terminal = resolve_terminal_timing(product, env)
    delay = float(metadata["terminal_delay"])

    # knock-out table: one row per record (KO-reset: both schedules, the pre record first at a tie)
    times = np.array([float(r[4]) for r in records], dtype=float)
    payment = np.array([float(r[2]) for r in records], dtype=float)
    probability = np.array([values[f"ko_probability:{r[0]}"] for r in records], dtype=float)
    discounted = np.array([values[f"ko_cash:{r[0]}"] for r in records], dtype=float)
    tags = np.array([0 if r[3] == "pre" else 1 for r in records])
    order = np.lexsort((tags, times)) if records else np.zeros(0, dtype=int)
    records = [records[i] for i in order]
    times, payment, probability, discounted = times[order], payment[order], probability[order], discounted[order]
    n = len(records)
    pay_df = np.array([_df(env, p) for p in payment])
    observations = ObservationTable(
        times=times, payment_times=payment, dates=es.all_dates(r[5] for r in records),
        payment_dates=es.all_dates(r[6] for r in records), is_ko_obs=np.ones(n, dtype=bool),
        is_coupon_obs=np.full(n, phoenix), schedule=tuple(r[3] for r in records) if reset else None,
        probability=probability, survival=1.0 - np.cumsum(probability), discounted_cash=discounted,
        cash_to_pay=discounted / pay_df if n else discounted, payment_is_path_dependent=np.zeros(n, dtype=bool),
    )

    # maturity (never knocked in) and knocked-in settlement rows
    maturity_t, pre_t = float(metadata["maturity"]), float(metadata["pre_maturity"])
    post_timing = dict(time=maturity_t, payment_time=maturity_t + delay, date=terminal.determination_date,
                       payment_date=terminal.payment_date)
    if reset:
        last_pre = [r for r in records if r[3] == "pre"]
        pre_date = (last_pre[-1][5] if last_pre and is_close(float(last_pre[-1][4]), pre_t) else None)
        pre_timing = dict(time=pre_t, payment_time=pre_t + delay, date=pre_date,
                          payment_date=pre_date if is_close(delay, 0.0) else None)
        settlement = first_schedule_settlement(product, env, terminal)
        if settlement.settlement_date_governs:
            # an explicit settlement date pays the note that ends with schedule 1 (spec D33)
            pre_timing.update(payment_time=settlement.payment_time, payment_date=settlement.payment_date)
    else:
        pre_timing = post_timing
    maturity = es.outcome_row(env, probability=values["terminal_alive_probability"],
                              discounted=values["terminal_alive_cash"], **pre_timing)
    ki_settlement = es.outcome_row(env, probability=values["terminal_ki_probability"],
                                   discounted=values["terminal_ki_cash"], **post_timing)
    if reset and n and maturity.time < float(times[-1]) - Tolerance.ZERO:
        # F5: the first schedule's maturity mass leaves the survival at its own determination
        observations = replace(observations, survival=observations.survival - np.where(
            times >= maturity.time, maturity.probability, 0.0))

    unavailable: Dict[str, str] = {}
    coupons, path_dependent_cash = None, None
    if phoenix:
        coupons, path_dependent_cash = _coupon_table(env, terminal, metadata, records, observations, values, streams,
                                                     unavailable, pre_timing, post_timing)

    # knock-in path statistics
    knocked_out_now, knocked_in_now, carried = _valuation_decisions(context, spot)
    rule = es.knock_in_rule(product, knocked_out_now=knocked_out_now,
                            knocked_in_now=carried if knocked_out_now else (carried or knocked_in_now))
    loss = es.expected_loss(product, env, probability=ki_settlement.probability,
                            discounted_cash=ki_settlement.discounted_cash, payment_time=ki_settlement.payment_time)
    knock_in, ki_table = es.knock_in_fields(
        rule, want=es.want_knock_in_path(streams), probability_no_ko=ki_settlement.probability, expected_loss=loss,
        unavailable=unavailable, forward=lambda: knock_in_table(product, env, context, spot),
        valuation_dates=es.at_valuation_dates(product, env),
    )

    receivables = receivables_table(lifecycle_state, env, knocked_in=carried)
    ledger = build_cash_ledger(observations=observations, coupons=coupons, maturity=maturity,
                               ki_settlement=ki_settlement, receivables=receivables,
                               path_dependent_cash=path_dependent_cash)
    lifetime = (float(np.sum(observations.probability * observations.times)) + maturity.probability * maturity.time
                + ki_settlement.probability * ki_settlement.time)
    return EventStatistics(
        product_family=family, pv=float(pv), observations=observations, coupons=coupons, maturity=maturity,
        ki_settlement=ki_settlement, knock_in=knock_in, ki_table=ki_table, receivables=receivables, ledger=ledger,
        expected_lifetime=lifetime, provenance=provenance, unavailable=unavailable,
    )


def _coupon_table(env, terminal, metadata, records, observations, values, streams, unavailable, pre_timing,
                  post_timing):
    """The Phoenix coupon table on the observation grid and, under EXPIRY pay, the ledger's payment-event buckets.

    INSTANT: each coupon settles with its record, cash to pay ``discounted / DF(settlement)``. EXPIRY: the row's
    payment is path-dependent (placeholder: the terminal payment), its cash to pay the earned-coupon channel; the
    ledger holds the coupon cash at each ending event (a knock-out's settlement, the two maturities)."""
    n = len(records)
    probability = np.array([values[f"coupon_probability:{r[0]}"] for r in records], dtype=float)
    cash = np.array([values[f"coupon_cash:{r[0]}"] for r in records], dtype=float)
    survival = np.concatenate([[1.0], observations.survival[:-1]]) if n else np.zeros(0)
    coupon_probability = probability if es.want_coupon_probability(streams) else None
    if coupon_probability is None:
        unavailable["coupons.probability"] = es.NOT_REQUESTED
    if not metadata["expiry_coupons"]:
        payment = observations.payment_times
        coupons = CouponTable(probability=coupon_probability, survival=survival, payment_times=payment,
                              payment_dates=observations.payment_dates, discounted_cash=cash,
                              cash_to_pay=es.to_pay(env, cash, payment) if n else cash,
                              payment_is_path_dependent=np.zeros(n, dtype=bool))
        return coupons, None
    terminal_pay = float(terminal.payment_time)
    if any(f"coupon_earned:{r[0]}" in values for r in records):
        earned = np.array([values[f"coupon_earned:{r[0]}"] for r in records], dtype=float)
    else:
        # without memory a row pays exactly its own coupon when its barrier is met
        amounts = dict(zip([r[0] for r in metadata["ko_metadata"]], metadata["coupon_amounts"]))
        earned = np.array([amounts[r[0]] for r in records], dtype=float) * probability
    coupons = CouponTable(
        probability=coupon_probability, survival=survival, payment_times=np.full(n, terminal_pay),
        payment_dates=None if terminal.payment_date is None else (terminal.payment_date,) * n,
        discounted_cash=cash, cash_to_pay=earned, payment_is_path_dependent=np.full(n, True),
    )
    buckets = [values[f"coupon_payment:{r[0]}"] for r in records] + [values["terminal_alive_coupon"],
                                                                       values["terminal_ki_coupon"]]
    determination = np.concatenate([observations.times, [pre_timing["time"], post_timing["time"]]])
    payment = np.concatenate([observations.payment_times, [pre_timing["payment_time"], post_timing["payment_time"]]])
    obs_dates, pay_dates = observations.dates, observations.payment_dates
    determination_dates = (None if obs_dates is None or pre_timing["date"] is None or post_timing["date"] is None
                           else obs_dates + (pre_timing["date"], post_timing["date"]))
    payment_dates = (None if pay_dates is None or pre_timing["payment_date"] is None
                     or post_timing["payment_date"] is None
                     else pay_dates + (pre_timing["payment_date"], post_timing["payment_date"]))
    ledger = CashLedger(determination_times=determination, payment_times=payment,
                        determination_dates=determination_dates, payment_dates=payment_dates,
                        discounted_cash=np.asarray(buckets, dtype=float),
                        cash_to_pay=es.to_pay(env, buckets, payment))
    return coupons, ledger
