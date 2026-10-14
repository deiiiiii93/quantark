"""Run lifecycle of a DCN (W14b, spec I1 / C3 / A1 / A5 / A7').

``DCNLifecycleTracker`` is the DCN's run tracker: standalone (not an autocallable subclass), duck-typed to the API the
run consumers call (``quantark.backtest.replay`` via ``tracker_for``, and ``PortfolioLifecycleManager`` for the
equity backtest and the dynamic scenario).

Decisions. Each close decides the BOOKED schedule's own dates (never a rebuilt schedule): the daily knock-in on a
daily-KI date (sticky), then on a monthly row the knock-out (KO-eligible row, spot >= KO; it ends the note and nothing
is observed after it), else the fixed coupon (coupon-eligible row, not knocked in, spot >= the coupon barrier), and on
the final row the loss leg of a knocked-in survivor -- the MC path kernel's order. Cash is a dated
``RealizedCashflow`` on ``lifecycle.ledger`` keyed by its observation date (``dcn:<kind>:<YYYY-MM-DD>``), SIGNED by
``direction_sign`` and scaled by the position quantity, payable on ``DCNOption.payment_dates()`` (C4).

Run calendars (A7'). Any run day prices (a closed day of the schedule calendar has no row and no KI test). A schedule
date the run never visits -- a holiday of the run calendar, or a date between path instants -- is still decided, at
the spot of the run instant before it, flagged ``gap=True`` (the signed W6 run-fixings rule). A schedule date before
the run start is history (signed rule 3, 2026-10-09): no barrier state is assumed and a past coupon is not owed. A
first run day after a run-start-or-later schedule date has no earlier instant to decide it at and is refused
(``TODO(dcn-run-start)``).

Pricing views (A1). Pre-close (``initial=True``): ``spec.build(valuation_date=d)``, row d kept and decided by the
engine at t = 0. Post-close (after ``observe(d)``): the same ``.without_observed_through(d)``; day d's cash is in the
ledger. Run consumers value the contingent claim from these copies and the ledger separately (live MTM + pending PV +
paid cash), as for every tracked product; the engines' own ``lifecycle_state`` route (unit ledgers) is the
engine-level contract and is not what the runs pass.
"""

from __future__ import annotations

import dataclasses
import warnings
from datetime import datetime
from typing import Any, Callable, List, Optional

import pandas as pd

from quantark.util.exceptions import ValidationError

from .autocallable import MaturedBeforeRunWarning
from .cashflows import RealizedCashflow
from .events import LifecycleEvent, LifecycleEventType
from .state import DCNLifecycleState

__all__ = ["DCNLifecycleTracker", "refuse_dcn_pnl_explain"]


def _day(value) -> datetime:
    return pd.Timestamp(value).normalize().to_pydatetime()


def _require_date_instant(instant) -> None:
    """W6 Q2(a): a run instant holding a DCN must be a date (midnight); a time of day is refused."""
    stamp = pd.Timestamp(instant)
    if stamp != stamp.normalize():
        # TODO(run-fixings): intraday run instants -- a fixing-time term on the DCN's observations
        raise ValidationError(
            f"run instant {stamp} carries a time of day: a DCN's observations are dated closes with no fixing "
            "time, so a run holding a DCN must step on dates (signed W6 Q2(a), run fixings R8)"
        )


def refuse_dcn_pnl_explain(products, consumer: str) -> None:
    """PnL explain of a run holding a DCN is not built (out of scope, R22): refused, never half-explained."""
    from quantark.asset.equity.product.option.dcn_option import DCNOption

    if any(isinstance(p, DCNOption) for p in products):
        # TODO(dcn-explain): explain a DCN run (pre/post-close views, ledger receivables) in quantark.pnlexplain
        raise ValidationError(
            f"{consumer}: PnL explain does not support a DCNOption position yet (TODO(dcn-explain)); "
            "run it with pnl_explain=None"
        )


class DCNLifecycleTracker:
    """Realized lifecycle of one DCN position through a run (see the module docstring)."""

    def __init__(
        self,
        *,
        product: Any,
        quantity: float,
        lifecycle: Optional[DCNLifecycleState] = None,
        start_date: Optional[pd.Timestamp] = None,
        date_resolver: Optional[Callable] = None,
        has_lifecycle: bool = True,
    ) -> None:
        from quantark.asset.equity.product.option.dcn_grid import build_dcn_grid_context
        from quantark.asset.equity.product.option.dcn_option import DCNOption

        if not isinstance(product, DCNOption):
            raise ValidationError(f"DCNLifecycleTracker tracks a DCNOption, got {type(product).__name__}")
        if lifecycle is not None and not isinstance(lifecycle, DCNLifecycleState):
            raise ValidationError(f"a DCN's lifecycle is a DCNLifecycleState, got {type(lifecycle).__name__}")
        self.product = product
        self.quantity = float(quantity)
        self.lifecycle = (
            lifecycle if lifecycle is not None
            else DCNLifecycleState(knocked_in=bool(product.knocked_in_at_valuation))
        )
        self.start_date = start_date
        self.has_lifecycle = has_lifecycle
        self._date_resolver = date_resolver            # accepted for the tracker_for signature; DCN dates are dated
        self._booked = product.schedule
        self._ctx = build_dcn_grid_context(product)
        self._payments = product.payment_dates()
        self._row_of = {m.observation_date: j for j, m in enumerate(self._booked.monthly)}
        self._ki_dates = frozenset(self._booked.daily_ki_dates)
        self._decision_dates = sorted(self._ki_dates | set(self._row_of))
        self._last: Optional[tuple] = None             # (day, spot) of the last observed run instant
        self._observed_through: Optional[datetime] = None
        self._matured_before_run = False

    # -- clock and run start ---------------------------------------------------------------------------------------
    @property
    def clock_origin(self) -> pd.Timestamp:
        """The schedule's initial date (A5)."""
        return pd.Timestamp(self._booked.spec.initial_date)

    @property
    def final_observation_date(self) -> datetime:
        return self._booked.monthly[-1].observation_date

    def matured_before_run(self, env=None) -> bool:
        """The final observation date is strictly before the run start (W1 rule)."""
        if self.start_date is None:
            return False
        return self.final_observation_date < _day(self.start_date)

    def retire_if_matured_before_run(self, env=None) -> bool:
        """Warn once and mark the note matured with no cash (valued at zero, W1); True when retired by this rule."""
        if self._matured_before_run:
            return True
        if not self.lifecycle.alive or not self.matured_before_run(env):
            return False
        warnings.warn(
            MaturedBeforeRunWarning(
                f"DCNOption's final observation {self.final_observation_date.date()} is before the run start "
                f"{_day(self.start_date).date()}: it is not settled in the run, books no cash and is valued at zero"
            ),
            stacklevel=2,
        )
        self.lifecycle.mark_matured_before_run(self.final_observation_date)
        self._matured_before_run = True
        return True

    # -- pricing copies --------------------------------------------------------------------------------------------
    @property
    def lifecycle_contract(self) -> Any:
        return self.product

    def product_for_lifecycle(self) -> Any:
        """The booked contract (decisions read its own schedule)."""
        return self.product

    def remaining_maturity(self, date) -> float:
        """ACT/365F years from ``date`` to the final observation, for maturity-sampling consumers (the replay's
        surface / term-carry probe). Never refuses and builds no copy: 0.0 on or after the final observation; the
        booked contract's own maturity before its valuation date (as ``product_for_pricing``). A dead (knocked-out)
        note keeps the real time to its final observation, the sample it gave before (R28): the probe drives a
        book-wide column and must not move with one product's state."""
        _require_date_instant(date)
        d = _day(date)
        if d < self._booked.valuation_date:
            return float(self.product.get_maturity())
        if d >= self.final_observation_date:
            return 0.0
        return (self.final_observation_date - d).days / 365.0

    def product_for_pricing(self, date, pricing_env=None, *, initial: bool = False) -> Any:
        """The DCN to price on ``date`` (A1); a pure accessor (the state is not touched).

        Contract: a copy priced together with the state on day d needs ``observe(d)`` first (post-close view). The
        only pre-close use is the run's first valuation (``initial=True``) before its first close, or a run instant
        right after the last observed one. Maturity sampling goes through ``remaining_maturity``, never here.

        * before the booking's valuation date: the booked contract, untouched (I1);
        * a dead note (knocked out / matured): a terminal copy, never a refusal (it is never priced): the post-close
          view of ``date`` while a row remains after it, else the contract on its final observation date
          (``get_maturity`` 0.0);
        * a live note past its final observation before ``observe(date)``: refused (that copy would still carry the
          undecided final row);
        * pre-close (``initial``, or ``date`` not yet observed): ``spec.build(valuation_date=date)``; refused while a
          run-time schedule date before ``date`` is still undecided (it would be in neither the copy nor the ledger);
        * post-close (after ``observe(date)``): the same ``.without_observed_through(date)``.

        A live copy is checked against the ledger (the double-count guard).
        """
        from quantark.asset.equity.engine.dcn_valuation import check_dcn_ledger

        _require_date_instant(date)
        d = _day(date)
        if d < self._booked.valuation_date:
            return self.product
        state = self.lifecycle
        final = self.final_observation_date
        if state.alive and d > final:
            raise ValidationError(
                f"the DCN's final observation {final.date()} is before {d.date()} and not yet decided: "
                "observe the run instant first (a copy now would carry the undecided final row)"
            )
        if not state.alive:
            if d < final and any(t > d for t in self._row_of):
                schedule = dataclasses.replace(self._booked.spec, valuation_date=d).build(self._booked.calendar)
                schedule = schedule.without_observed_through(d)
            else:
                schedule = dataclasses.replace(self._booked.spec, valuation_date=final).build(self._booked.calendar)
            return _rebuild(self.product, schedule=schedule, knocked_in_at_valuation=bool(state.knocked_in))
        schedule = dataclasses.replace(self._booked.spec, valuation_date=d).build(self._booked.calendar)
        post = (not initial and self.has_lifecycle and self._observed_through is not None
                and self._observed_through >= d)
        if post:
            schedule = schedule.without_observed_through(d)
            knocked_in = state.knocked_in
        else:
            self._require_decided_before(d)
            knocked_in = state.knocked_in and (state.ki_date is None or state.ki_date < d)
        product = _rebuild(self.product, schedule=schedule, knocked_in_at_valuation=bool(knocked_in))
        check_dcn_ledger(product, d, state.ledger, alive=state.alive)
        return product

    def _require_decided_before(self, d: datetime) -> None:
        """A pre-close copy at ``d`` needs every run-time schedule date before ``d`` decided (CP5)."""
        if not self.has_lifecycle:
            return
        lower = self._last[0] if self._last is not None else None
        start = _day(self.start_date) if self.start_date is not None else None
        pending = [t for t in self._decision_dates
                   if t < d and (lower is None or t > lower) and (start is None or t >= start)]
        if pending:
            raise ValidationError(
                f"DCN schedule date(s) {', '.join(f'{t:%Y-%m-%d}' for t in pending)} before {d.date()} are "
                "undecided: observe the run instants first (a pre-close copy would drop them)"
            )

    # -- decisions -------------------------------------------------------------------------------------------------
    def observe(self, date, product: Any = None, env=None, spot: float = None, **_ignored) -> List[LifecycleEvent]:
        """Decide this close and every schedule date since the last run instant (A7' gap rule)."""
        _require_date_instant(date)
        d = _day(date)
        state = self.lifecycle
        state.advance_to(d)
        if self.retire_if_matured_before_run(env):
            return []
        if not state.alive or not self.has_lifecycle:
            self._last = (d, float(spot))
            return []
        lower = self._last[0] if self._last is not None else None
        start = _day(self.start_date) if self.start_date is not None else None
        events: List[LifecycleEvent] = []
        for t in self._decision_dates:
            if t > d or (lower is not None and t <= lower):
                continue
            if start is not None and t < start:
                # before the run (signed rule 3): no barrier state assumed, a past coupon not owed
                if t in self._row_of:
                    state.observed_row_indices.add(self._row_of[t])
                continue
            if t < d:
                if self._last is None:
                    # TODO(dcn-run-start): a run whose first instant is after a schedule date on or after its start
                    # has no earlier spot to decide that date at; not decided rather than invented
                    raise ValidationError(
                        f"DCN observation {t.date()} is on/after the run start but before the first run instant "
                        f"{d.date()}: no spot to decide it at (TODO(dcn-run-start))"
                    )
                s, gap = self._last[1], True
            else:
                s, gap = float(spot), False
            events.extend(self._decide(t, s, gap, d))
            if not state.alive:
                break
        self._last = (d, float(spot))
        self._observed_through = d
        return events

    def settle_maturity_if_due(self, date, product: Any = None, env=None, spot: float = None, *,
                               settled_product: Any = None, previous_spot: Optional[float] = None):
        """The loss leg is decided by ``observe`` on the final row (with the gap rule); nothing to do here."""
        return None

    # -- one schedule date -----------------------------------------------------------------------------------------
    def _snapshot(self) -> dict:
        s = self.lifecycle
        return {"alive": s.alive, "knocked_in": s.knocked_in, "knocked_out": s.knocked_out, "matured": s.matured}

    def _cash(self, kind: str, event_type: LifecycleEventType, t: datetime, paid: datetime, unit: float, gap: bool,
              decided_on: datetime) -> tuple:
        signed = float(self.product.direction_sign) * unit
        cash = self.quantity * signed
        flow = RealizedCashflow(
            cashflow_id=f"dcn:{kind}:{t:%Y-%m-%d}", event_type=event_type, amount=cash,
            determination_date=t, payment_date=paid,
            metadata={"gap": gap, "decided_on": decided_on, "observation_date": t},
        )
        self.lifecycle.ledger.register(flow)
        return signed, cash, flow

    def _event(self, event_type, t, s, j, barrier, payoff, cash, flow, terminates, before, gap, decided_on):
        return LifecycleEvent(
            event_type=event_type, date=pd.Timestamp(t), spot=float(s), observation_index=j, barrier=barrier,
            payoff=payoff, cashflow=cash, realized_cashflow=flow, terminates_position=terminates,
            state_before=before, state_after=self._snapshot(),
            metadata={"gap": gap, "decided_on": pd.Timestamp(decided_on),
                      "settlement_date": None if flow is None else flow.payment_date},
        )

    def _decide(self, t: datetime, s: float, gap: bool, decided_on: datetime) -> List[LifecycleEvent]:
        p, state, ctx = self.product, self.lifecycle, self._ctx
        events: List[LifecycleEvent] = []
        # 1) daily knock-in (sticky), before the monthly row
        if t in self._ki_dates and not state.knocked_in and s <= p.ki_barrier:
            before = self._snapshot()
            state.knocked_in, state.ki_date = True, t
            events.append(self._event(LifecycleEventType.KNOCK_IN, t, s, None, float(p.ki_barrier), 0.0, 0.0,
                                      None, False, before, gap, decided_on))
        j = self._row_of.get(t)
        if j is None:
            return events
        state.observed_row_indices.add(j)
        row = self._booked.monthly[j]
        part, notional = p.participation, p.notional
        # 2) knock-out has priority (either knock-in state); it ends the note
        if row.is_ko_obs and s >= p.ko_barrier:
            before = self._snapshot()
            unit = part * p.ko_coupon_rate * float(ctx.ko_accruals[j]) * notional
            payoff, cash, flow = self._cash("knock-out", LifecycleEventType.KNOCK_OUT, t,
                                            self._payments.ko_payment_dates[j], unit, gap, decided_on)
            state.knocked_out, state.ko_date = True, t
            events.append(self._event(LifecycleEventType.KNOCK_OUT, t, s, j, float(p.ko_barrier), payoff, cash, flow,
                                      True, before, gap, decided_on))
            return events
        # 3) the fixed coupon: not knocked in, at or above the coupon barrier
        if row.is_coupon_obs and not state.knocked_in and s >= p.coupon_barrier:
            before = self._snapshot()
            unit = part * p.coupon_rate * float(ctx.coupon_accruals[j]) * notional
            payoff, cash, flow = self._cash("coupon", LifecycleEventType.COUPON, t,
                                            self._payments.coupon_payment_dates[j], unit, gap, decided_on)
            events.append(self._event(LifecycleEventType.COUPON, t, s, j, float(p.coupon_barrier), payoff, cash, flow,
                                      False, before, gap, decided_on))
        # 4) the final row: maturity; a knocked-in survivor pays the loss leg
        if j == len(self._booked.monthly) - 1:
            before = self._snapshot()
            payoff = cash = 0.0
            flow = None
            if state.knocked_in:
                unit = -(notional / p.initial_price) * part * max(p.k_loss - s, 0.0)
                payoff, cash, flow = self._cash("loss", LifecycleEventType.MATURITY, t,
                                                self._payments.loss_payment_date, unit, gap, decided_on)
            state.matured = True
            events.append(self._event(LifecycleEventType.MATURITY, t, s, j, float(p.k_loss), payoff, cash, flow,
                                      True, before, gap, decided_on))
        return events


def _rebuild(product, **changes):
    """A DCNOption copy with ``changes`` (validated by the constructor)."""
    from quantark.asset.equity.product.option.dcn_option import DCNOption

    names = ("notional", "initial_price", "direction", "coupon_barrier_ratio", "ko_barrier_ratio", "ki_barrier_ratio",
             "ki_put_strike_ratio", "coupon_rate", "ko_coupon_rate", "participation", "coupon_counted_days",
             "coupon_days_denom", "schedule", "settlement_date", "knocked_in_at_valuation", "settlement_convention")
    kwargs = {name: getattr(product, name) for name in names}
    kwargs.update(changes)
    return DCNOption(**kwargs)
