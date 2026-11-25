"""A DCN monthly observation at the valuation instant (W10 / W10b, user rulings 2026-10-10).

Spec: ``docs/superpowers/specs/2026-10-10-dcn-final-observation-proposal.md``. The row observed at the valuation
instant is decided on today's spot (the close, Q1), pointwise at barrier ties (Q2), in the MC path kernel's order
(``engine/mc/dcn_payoff.py``): knock-in = carried OR spot <= KI; knock-out (spot >= KO, in either knock-in state) pays
the knock-out coupon and ends the contract; the fixed coupon pays when not knocked out, not knocked in and spot >= the
coupon barrier; on the FINAL row the loss leg pays when knocked in and not knocked out. Every flow is discounted from
the valuation date to its own payment date (DF = 1 when paid today). Leg amounts are the kernel's arithmetic, so the MC
route equals the kernel's one-column row bit for bit.

Neutral module: no Monte Carlo import, so the PDE routes call it too. Decided-but-unpaid cash from earlier rows and a
knocked-out or matured contract come in through a ``DCNLifecycleState`` (W14, ``resolve_dcn_lifecycle_state``). Payment
dates are the grid context's, i.e.
``DCNOption.payment_dates()``: a ``settlement_convention`` times today's row as it times every later one (W15).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from quantark.asset.equity.engine.settlement_support import at_exercise_instant
from quantark.asset.equity.product.option.dcn_grid import DCNGridContext, build_dcn_grid_context
from quantark.priceenv.term_sampling import make_df_fn
from quantark.execution.errors import CapabilityError
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_close

__all__ = [
    "DCNFoldCash",
    "DCNFoldSlots",
    "DCNStateValuation",
    "DCNValuationRow",
    "at_final_observation",
    "bind_dcn_fold",
    "check_dcn_ledger",
    "dcn_fold_slots",
    "folded_row_cash",
    "observe_valuation_row",
    "resolve_dcn_lifecycle_state",
    "settle_final_observation",
    "refuse_terminal_fold",
    "settled_row_delta",
]


@dataclass(frozen=True)
class DCNValuationRow:
    """The monthly row observed at the valuation instant.

    Leg PVs are UNSIGNED (buyer perspective, as the MC kernel's) and discounted to the valuation date; ``loss_pv`` is
    non-zero only on the final row. ``pv`` applies ``direction_sign`` leg by leg in the MC result's order.
    """

    is_final: bool
    knocked_in: bool
    knocked_out: bool
    coupon_paid: bool
    coupon_pv: float
    ko_pv: float
    loss_pv: float
    direction_sign: float
    loss_df: float

    @property
    def ends_contract(self) -> bool:
        """Nothing is left to price after today: the final row, or a knock-out today."""
        return self.is_final or self.knocked_out

    @property
    def pv(self) -> float:
        sign = self.direction_sign
        return float(sign * self.coupon_pv) + float(sign * self.ko_pv) + float(sign * self.loss_pv)


def at_final_observation(product, pricing_env) -> bool:
    """Whether the DCN is valued at its final observation instant on ``pricing_env``'s clock (``get_maturity`` 0.0:
    the final observation date, or a later date the env places at the valuation instant)."""
    return at_exercise_instant(product.get_maturity(pricing_env))


def observe_valuation_row(
    product, pricing_env, ctx: Optional[DCNGridContext] = None, df=None,
) -> Optional[DCNValuationRow]:
    """The monthly row observed at the valuation instant, or ``None`` when the valuation date is not one."""
    ctx = build_dcn_grid_context(product, pricing_env) if ctx is None else ctx
    if ctx.obs_cols.size == 0 or int(ctx.obs_cols[0]) != 0:
        return None
    df = make_df_fn(pricing_env) if df is None else df
    spot = float(pricing_env.spot)
    part, notional, s0 = product.participation, product.notional, product.initial_price
    is_final = ctx.times.size == 1
    # the contract's inclusive inequalities, exactly as the kernel and the PDE masks state them (Q2: pointwise)
    knocked_in = bool(product.knocked_in_at_valuation) or spot <= product.ki_barrier
    knocked_out = bool(ctx.obs_is_ko[0]) and spot >= product.ko_barrier
    coupon_paid = (bool(ctx.obs_is_coupon[0]) and not knocked_out and not knocked_in
                   and spot >= product.coupon_barrier)
    ko_pv = coupon_pv = loss_pv = 0.0
    if knocked_out:
        dfs_k = np.array([df(t) for t in ctx.ko_pay_times])
        ko_pv = float(part * product.ko_coupon_rate * ctx.ko_accruals[0] * notional * dfs_k[0])
    if coupon_paid:
        dfs_c = np.array([df(t) for t in ctx.coupon_pay_times])
        coupon_pv = float(part * product.coupon_rate * ctx.coupon_accruals[0] * notional * dfs_c[0])
    loss_df = float(df(ctx.loss_pay_time))
    if is_final and knocked_in and not knocked_out:
        loss_pv = float(-(notional / s0) * part * np.maximum(product.k_loss - spot, 0.0) * loss_df)
    return DCNValuationRow(
        is_final=is_final, knocked_in=knocked_in, knocked_out=knocked_out, coupon_paid=coupon_paid,
        coupon_pv=coupon_pv, ko_pv=ko_pv, loss_pv=loss_pv, direction_sign=float(product.direction_sign),
        loss_df=loss_df,
    )


def settle_final_observation(product, pricing_env, ctx: Optional[DCNGridContext] = None) -> DCNValuationRow:
    """The DCN on its final observation date (T = 0): the settled final row."""
    ctx = build_dcn_grid_context(product, pricing_env) if ctx is None else ctx
    if ctx.times.size != 1:
        raise ValidationError("settle_final_observation needs a DCN valued on its final observation date")
    return observe_valuation_row(product, pricing_env, ctx)


def settled_row_delta(product, pricing_env, row: DCNValuationRow) -> float:
    """Q6: the slope of the settled final row at the spot (signed); 0 off the loss leg and at a tie, where the row
    jumps or kinks (the KO barrier; the coupon barrier when not knocked in; the KI barrier when not carried; K_loss on
    the loss leg)."""
    if not row.is_final:
        raise ValidationError("settled_row_delta is the final row's slope")
    spot = float(pricing_env.spot)
    ties = [product.ko_barrier]
    if not row.knocked_in:
        ties.append(product.coupon_barrier)
    if not product.knocked_in_at_valuation:
        ties.append(product.ki_barrier)
    on_loss_leg = row.knocked_in and not row.knocked_out
    if on_loss_leg:
        ties.append(product.k_loss)
    if any(is_close(spot, float(level)) for level in ties):
        return 0.0
    if not on_loss_leg or spot >= product.k_loss:
        return 0.0
    slope = (product.notional / product.initial_price) * product.participation * row.loss_df
    return float(row.direction_sign * slope)


#: a DCN ledger flow's receivable source (spec ReceivablesTable): the run tracker's knock-out, coupon and loss flows
_RECEIVABLE_SOURCE = {"KNOCK_OUT": "ko", "COUPON": "coupon", "MATURITY": "dcn_loss"}


@dataclass(frozen=True)
class DCNStateValuation:
    """A validated ``DCNLifecycleState`` reduced to what a DCN engine adds.

    ``pending_pv``: the ledger's pending receivable PV (signed, via
    ``LifecycleCashflowLedger.pending_pv``). ``terminal``: knocked out or
    matured -- the price IS ``pending_pv`` and the engine does not run.
    ``pending``: the pending flows behind ``pending_pv``, each with the payment timing it was discounted at
    (``SettlementResolver.resolve_pending``), in the ledger's order.
    """

    pending_pv: float
    terminal: bool
    pending: tuple = ()
    #: the environment the state was resolved against (its clock times the receivables)
    pricing_env: object = None

    def receivables_table(self, valuation_date):
        """The pending flows as the statistics' receivables (spec §4.9c): cash decided at or before the valuation
        and paid after it, ledger rows and never events.

        Per row: the source (a DCN ledger carries knock-out, coupon and loss flows), the decision date and its time
        on the environment's axis (<= 0, ``env_time`` from ``valuation_date``; 0.14.0 retired the DCN's own ACT/365F
        axis), the payment date and the time ``price()`` discounted it at, the discounted cash ``amount x DF``
        (signed, as ``pending_pv``) and the cash to pay ``amount``.

        Raises:
            ValidationError: a pending flow of another kind (a DCN ledger has no other receivable).
        """
        from quantark.asset.equity.engine.event_stats import ReceivablesTable
        from quantark.priceenv.date_axis import env_times

        if not self.pending:
            return ReceivablesTable.empty()
        sources = []
        for cashflow, _timing in self.pending:
            source = _RECEIVABLE_SOURCE.get(getattr(cashflow.event_type, "name", None))
            if source is None:
                raise ValidationError(
                    f"DCN ledger flow {cashflow.cashflow_id!r} is a {cashflow.event_type!r} receivable; a DCN ledger "
                    "carries knock-out, coupon and loss (MATURITY) flows only"
                )
            sources.append(source)
        return ReceivablesTable(
            source=tuple(sources),
            decision_times=list(env_times(self.pricing_env, [cashflow.determination_date
                                                             for cashflow, _ in self.pending], anchor=valuation_date)),
            decision_dates=tuple(cashflow.determination_date for cashflow, _ in self.pending),
            payment_times=[float(timing.payment_time) for _, timing in self.pending],
            payment_dates=tuple(cashflow.payment_date for cashflow, _ in self.pending),
            discounted_cash=[cashflow.amount * timing.payment_df for cashflow, timing in self.pending],
            cash_to_pay=[cashflow.amount for cashflow, _ in self.pending],
        )


def resolve_dcn_lifecycle_state(product, pricing_env, lifecycle_state) -> Optional[DCNStateValuation]:
    """Validate a DCN lifecycle state against the product and environment (W14, C2 / A1); ``None`` for no state.

    Contract: the state is as of the environment's valuation date d, so a run tracker must have run ``observe(d)``
    before a copy is priced with it on day d (``DCNLifecycleTracker.product_for_pricing``).

    Raises:
        CapabilityError: a state that is not a ``DCNLifecycleState``.
        ValidationError: a time-based valuation point or ledger flow (a DCN is dated); a valuation point that is not
            the environment's (and the schedule's) valuation date; ``state.knocked_in`` differing from
            ``product.knocked_in_at_valuation``; a ledger flow determined after the valuation date; or, for a state
            that is still alive, a flow determined on a date that still has a monthly row in the product's schedule
            (the engine would pay that row again -- the double-count guard). A terminal state runs no engine, so it
            is exempt from the row check.
    """
    if lifecycle_state is None:
        return None
    from quantark.asset.equity.engine.settlement_support import pending_receivable_pv
    from quantark.asset.equity.lifecycle.state import DCNLifecycleState

    if not isinstance(lifecycle_state, DCNLifecycleState):
        raise CapabilityError(
            f"DCN engines take a DCNLifecycleState, got {type(lifecycle_state).__name__}"
        )
    valuation_date = pricing_env.valuation_date
    if product.schedule.valuation_date != valuation_date:
        raise ValidationError(
            "a DCN lifecycle state needs the schedule's valuation date "
            f"{product.schedule.valuation_date:%Y-%m-%d} to equal the environment's {valuation_date:%Y-%m-%d}"
        )
    point = lifecycle_state.valuation_point
    if point is not None:
        if point.date is None:
            raise ValidationError("a DCN lifecycle state needs a date valuation point (a DCN is dated)")
        if point.date != valuation_date:
            raise ValidationError(
                f"lifecycle valuation point {point.date:%Y-%m-%d} is not the valuation date {valuation_date:%Y-%m-%d}"
            )
    if bool(lifecycle_state.knocked_in) != bool(product.knocked_in_at_valuation):
        raise ValidationError(
            f"lifecycle state knocked_in={bool(lifecycle_state.knocked_in)} contradicts the product's "
            f"knocked_in_at_valuation={bool(product.knocked_in_at_valuation)}"
        )
    terminal = not lifecycle_state.alive
    check_dcn_ledger(product, valuation_date, lifecycle_state.ledger, alive=not terminal)
    if lifecycle_state.valuation_point is None:
        pending_receivable_pv(lifecycle_state, pricing_env)   # refuses a non-empty ledger without a point
    pending = _pending_flows(lifecycle_state, pricing_env, valuation_date)
    # ``LifecycleCashflowLedger.pending_pv``'s arithmetic (amount x DF, summed in ledger order) on the DCN's own clock
    total = 0.0
    for cashflow, timing in pending:
        total += cashflow.amount * timing.payment_df
    return DCNStateValuation(pending_pv=float(total), terminal=terminal, pending=pending, pricing_env=pricing_env)


#: a pending flow's settlement kind, exactly as ``SettlementResolver.resolve_pending`` maps it (the price accepts every
#: kind the resolver does; only the statistics' receivables table refuses a non-DCN kind)
_PENDING_KIND = {"COUPON": "COUPON", "KNOCK_OUT": "REDEMPTION", "KNOCK_IN": "HIT", "MATURITY": "TERMINAL",
                 "EXPIRY": "TERMINAL"}


def _pending_flows(lifecycle_state, pricing_env, valuation_date) -> tuple:
    """``(cashflow, timing)`` of every pending flow, timed on the environment's axis from the valuation date (spec
    D15): ``env_time`` of the payment date -- the time ``price()`` gives the same date, honouring
    ``holiday_time_adjustment`` -- and its DF. A flow paid after the valuation date that the env's clock places at the
    valuation instant (a holiday right after it, PRECEDING) is still a receivable, discounted at DF(0). On
    CALENDAR_DAYS / ACT_365 this is ``SettlementResolver.resolve_pending``'s timing bit for bit."""
    from quantark.asset.equity.settlement import CashflowKind, ResolvedPaymentTiming, SettlementResolver
    from quantark.priceenv.date_axis import env_times

    point = lifecycle_state.valuation_point
    if point is None:
        return ()
    flows = lifecycle_state.ledger.pending(point)
    if not flows:
        return ()
    times = env_times(pricing_env, [cashflow.payment_date for cashflow in flows], anchor=valuation_date)
    determination_df = SettlementResolver._checked_df(pricing_env, 0.0, "pending valuation")
    out = []
    for cashflow, t in zip(flows, times):
        kind = _PENDING_KIND.get(getattr(cashflow.event_type, "name", None))
        if kind is None:
            raise ValidationError(
                f"pending cashflow id={cashflow.cashflow_id!r} has unsupported event type {cashflow.event_type!r}"
            )
        if t < 0.0:
            raise ValidationError(
                f"pending cashflow id={cashflow.cashflow_id!r} is paid before the valuation instant on the "
                "environment's clock"
            )
        payment_df = SettlementResolver._checked_df(pricing_env, float(t), "pending payment")
        out.append((cashflow, ResolvedPaymentTiming(
            kind=CashflowKind[kind], determination_date=point.date, determination_time=0.0,
            payment_date=cashflow.payment_date, payment_time=float(t), determination_df=determination_df,
            payment_df=payment_df, delay_df=payment_df / determination_df,
        )))
    return tuple(out)


def check_dcn_ledger(product, valuation_date, ledger, *, alive: bool) -> None:
    """The double-count guard of ``resolve_dcn_lifecycle_state`` (A1), also run by the run tracker on every pricing
    copy: every flow is dated, none is determined after ``valuation_date``, and -- while the contract is alive -- none
    is determined on a date that still has a monthly row in ``product.schedule``. It assumes the ledger is complete
    through ``valuation_date`` (``observe`` ran first); an undecided date is the tracker's own refusal."""
    rows = {m.observation_date for m in product.schedule.monthly}
    for cashflow in ledger.cashflows:
        determined = cashflow.determination_date
        if determined is None or cashflow.payment_date is None:
            raise ValidationError(
                f"DCN ledger flow {cashflow.cashflow_id!r} needs a determination and a payment date"
            )
        if determined > valuation_date:
            raise ValidationError(
                f"DCN ledger flow {cashflow.cashflow_id!r} is determined {determined:%Y-%m-%d}, after the valuation "
                f"date {valuation_date:%Y-%m-%d}"
            )
        if alive and determined in rows:
            raise ValidationError(
                f"DCN ledger flow {cashflow.cashflow_id!r} is determined {determined:%Y-%m-%d}, a date that still "
                "has a monthly row in the product's schedule (it would be paid twice); price the post-close view "
                "(DCNSchedule.without_observed_through)"
            )


# --- cash-leg folding (ship spec 2026-10-21 D9, D13) ----------------------------------------------------------------

@dataclass(frozen=True)
class DCNFoldCash:
    """A ``FoldOverlay`` bound to one DCN solve: buyer-view cash per unit, paid at the event's payment date.

    ``ko[j]`` is paid at monthly observation ``j``'s knock-out payment by a path knocking out there (0 on an
    observation that is not a knock-out observation, whose knock-out probability is zero on every route);
    ``no_ki`` / ``ki`` by a survivor of the final observation, never / ever knocked in, at the loss-leg
    settlement. The engines add it where they add the knock-out coupon and the loss leg, before
    ``direction_sign``.
    """

    ko: np.ndarray
    no_ki: float
    ki: float


@dataclass(frozen=True)
class DCNFoldSlots:
    """The DCN's event axis for folding: the shared monthly observation grid and the survivor settlement.

    ``ko`` holds one ``("main", t_obs, ko_pay_time)`` record per monthly observation, the KO stream's axis of
    the DCN statistics (DESIGN §4.2: the KO and coupon rows share one grid; a coupon-only row carries KO
    probability 0). Both maturity branches are determined at ``determination`` (the final observation) and
    paid at ``payment`` (the loss-leg settlement). Times are on the price's axis (the environment's).
    """

    ko: tuple
    determination: float
    payment: float


def dcn_fold_slots(ctx: DCNGridContext) -> DCNFoldSlots:
    """The fold structure's and the binder's one source of the DCN event axis."""
    obs = np.asarray(ctx.times[ctx.obs_cols], dtype=float)
    return DCNFoldSlots(
        ko=tuple(("main", float(t), float(p)) for t, p in zip(obs, ctx.ko_pay_times)),
        determination=float(obs[-1]),
        payment=float(ctx.loss_pay_time),
    )


def bind_dcn_fold(overlay, ctx: DCNGridContext) -> DCNFoldCash:
    """``overlay`` bound one to one to the DCN's monthly rows and its two survivor branches (D5, fail closed).

    Every monthly observation is a knock-out slot at ``(t_obs, ko_pay_time)`` (the KO stream's axis,
    ``dcn_fold_slots``); both maturity branches are determined at the final observation and paid at
    ``loss_pay_time``. A slot on an observation that is not a knock-out observation binds and carries nothing:
    the event it is conditional on cannot happen (a leg's own overlay does carry cash there,
    ``v_e - v_anchor``, which its zero probability removes from the event route's leg value).
    """
    slots = dcn_fold_slots(ctx)
    amounts = overlay.bind_ko(list(slots.ko))
    return DCNFoldCash(
        ko=np.where(ctx.obs_is_ko, amounts, 0.0),
        no_ki=float(overlay.bind_terminal("no_ki", slots.determination, slots.payment)),
        ki=float(overlay.bind_terminal("ki", slots.determination, slots.payment)),
    )


def folded_row_cash(row: DCNValuationRow, fold: DCNFoldCash, ctx: DCNGridContext, df) -> float:
    """The folded cash a valuation-date row that ends the contract pays (D13), buyer-view, discounted as the row
    discounts its own legs: a knock-out today at the knock-out payment; the final row's survivor branch at the
    loss-leg settlement; nothing on a row that does not end the contract."""
    if row.knocked_out:
        return float(fold.ko[0]) * float(df(ctx.ko_pay_times[0]))
    if row.is_final:
        return float(fold.ki if row.knocked_in else fold.no_ki) * row.loss_df
    return 0.0


def refuse_terminal_fold() -> None:
    """A knocked-out or matured lifecycle state prices its receivables alone: no event is left to carry cash."""
    raise CapabilityError(
        "a knocked-out or matured DCN lifecycle state has no knock-out or maturity event left to fold cash onto; "
        "its cash legs are constants on the event route"
    )
