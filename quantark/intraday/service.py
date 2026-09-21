"""value_intraday: resolve once, dispatch once, report everything.

The design's sequence: (1) resolve the context; (2) look up the capability row
for the product/engine/monitoring; (3) price the twin through the route;
(4) add pending receivables from the time-based ledger; (5) itemise cashflows
with provenance; (6) greeks and a local theta re-evaluating the SAME resolved price function.
With a ``PricingSession``, an engine-invoking route is re-dispatched through
the execution kernel and must agree to 1e-12 (two paths to one number).
"""
from __future__ import annotations

from quantark.asset.equity.engine.settlement_support import pending_receivable_pv
from quantark.execution.contracts import PricingRequest
from quantark.execution.errors import CapabilityError, DeterminismViolation
from quantark.intraday.capability import engine_class_path, require_capability
from quantark.intraday.context import resolve_context
from quantark.intraday.engines import route_for
from quantark.intraday.events import EventKind, monitoring_of
from quantark.intraday.provisional import ASSUMED
from quantark.intraday.request import IntradayValuationRequest
from quantark.intraday.result import CashflowComponent, IntradayValuationResult
from quantark.intraday.timestamp import to_utc
from quantark.intraday.twin import event_for_cashflow
from quantark.util.numerical import is_close

_LIFECYCLE_FIELDS = ("alive", "knocked_in", "knocked_out", "matured", "expired", "coupon_memory_count")
#: Route methods that never call the engine; there is nothing for the kernel to re-dispatch. A claim decided at the
#: valuation instant is resolved by the runtime's own events, and QUAD V2's compiler has no zero-maturity twin to take.
_ENGINE_FREE_METHODS = ("terminated", "deterministic_zero_variance", "decided_at_valuation")
_PARITY_TOL = 1e-12


def _monitoring(ctx) -> str:
    return monitoring_of(ctx.timeline)


def _cashflow_origin(cf, ctx):
    """(event_id, provenance, depends_on) of a ledger flow.

    A flow is provisional when an assumption can change it, not only when the
    assumption determined its own event (review 2026-09-16 R7). Replay is
    chronological, so an assumed instant reaches every flow determined after it
    through the state it leaves behind:

    * a KO observation decides SURVIVAL, so every later flow exists only on its
      assumed outcome;
    * a coupon observation decides coupon MEMORY: whether a later coupon pays
      arrears, and which (a memory coupon settles every period missed since the
      last one paid);
    * a KI observation decides the maturity payoff, and every later flow where a
      knock-in changes the KO rule (``disable_ko_after_ki``, or a KO-reset
      schedule).

    A flow determined before every assumption, or carried in from the
    checkpoint, is confirmed history.
    """
    if cf.metadata.get(ASSUMED):
        # a continuous-barrier hit assumed at the valuation instant
        return None, "provisional", ("continuous_history",)
    event = event_for_cashflow(cf, ctx.timeline, ctx.request.session_calendar.tz)
    if event is None:
        return None, "confirmed", ()          # carried in from the authoritative checkpoint
    kinds = {e.event_id: e.kind for e in ctx.timeline.events}
    when = to_utc(event.timestamp)
    product = ctx.request.product
    ki_moves_later_cash = (bool(getattr(getattr(product, "barrier_config", None), "disable_ko_after_ki", False))
                           or getattr(product, "post_barrier_config", None) is not None)
    depends = []
    for assumption in ctx.reconstruction.assumptions:
        assumed = set(assumption.event_ids)
        if event.event_id in assumed:
            depends.extend(assumption.event_ids)
            continue
        if to_utc(assumption.scheduled_at) >= when:
            continue
        moved = {kinds.get(eid) for eid in assumed}
        if (EventKind.KO in moved or EventKind.COUPON in moved
                or (EventKind.KI in moved and (event.kind is EventKind.TERMINAL or ki_moves_later_cash))):
            depends.extend(assumption.event_ids)
    depends = tuple(dict.fromkeys(depends))
    return event.event_id, ("provisional" if depends else "confirmed"), depends


def value_intraday(engine, request: IntradayValuationRequest, *, session=None) -> IntradayValuationResult:
    """Value one contract at an intraday timestamp; see the module docstring for the sequence."""
    ctx = resolve_context(request)
    if request.greeks and request.greek_convention not in ("desk_bump", "point"):
        raise CapabilityError(f"{request.greek_convention} greeks have no intraday route")
    require_capability(request.product, engine, monitoring=_monitoring(ctx), outputs=("price",) + request.greeks)
    outcome = route_for(ctx, engine).price(ctx, engine)
    state = ctx.numerical.lifecycle_state
    pending_pv = float(pending_receivable_pv(state, ctx.pricing_env)) if state is not None else 0.0
    price = outcome.contingent_pv + pending_pv
    records = list(outcome.records) + [f"context:{ctx.identity}"]
    if session is not None:
        if outcome.method in _ENGINE_FREE_METHODS:
            records.append(f"manifest:not-dispatched ({outcome.method} route calls no engine)")
        else:
            dispatched = outcome.engine_used if outcome.engine_used is not None else engine
            records.append(_dispatch_and_compare(session, dispatched, request, ctx, outcome, price))
    cashflows = [CashflowComponent("contingent", outcome.contingent_pv, "model")]
    if state is not None:
        for cf in state.ledger.cashflows:
            event_id, provenance, depends_on = _cashflow_origin(cf, ctx)
            if cf.payment_time > 0.0:
                pv = float(cf.amount) * float(ctx.pricing_env.get_discount_factor(cf.payment_time))
                cashflows.append(CashflowComponent("pending_receivable", pv, provenance, event_id=event_id,
                                                   cashflow_id=cf.cashflow_id, payment_tau=float(cf.payment_time),
                                                   depends_on=depends_on))
            else:
                cashflows.append(CashflowComponent("paid", float(cf.amount), provenance, event_id=event_id,
                                                   cashflow_id=cf.cashflow_id, payment_tau=float(cf.payment_time),
                                                   depends_on=depends_on))
    lifecycle = {k: getattr(state, k) for k in _LIFECYCLE_FIELDS if hasattr(state, k)} if state is not None else {}
    numerical = dict(outcome.numerical)
    by_name = {}
    sensitivities = tuple(g for g in request.greeks if g != "theta")
    if sensitivities and request.greek_convention == "point":
        from quantark.intraday.greeks import point_greek_values
        values, point_records = point_greek_values(ctx, engine, sensitivities)
        by_name.update((g.name, g) for g in values)
        records.extend(point_records)
    elif sensitivities:
        from quantark.intraday.greeks import assemble_desk_greeks, bump_config_for, desk_bump_cells
        cells = desk_bump_cells(ctx, engine, sensitivities)
        values = assemble_desk_greeks(cells, sensitivities, spot=ctx.spot, bump_config=bump_config_for(engine))
        by_name.update((g.name, g) for g in values)
        records.extend(f"cell:{bump_id}" for bump_id in cells if bump_id != "base")
    if "theta" in request.greeks:
        from quantark.intraday.greeks import intraday_theta, resolve_theta_step, theta_metadata
        step = resolve_theta_step(ctx, request.theta_step, request.theta_unit)
        theta = intraday_theta(ctx, engine, route_for(ctx, engine), step, price_base=price, base_exact=outcome.exact,
                               convention=request.greek_convention)
        by_name["theta"] = theta
        numerical.update(theta_metadata(step))
        if request.greek_convention == "point":
            # the desk step above only locates the segment; the derivative's own stencil step is this one
            numerical["theta_point_step_s"] = theta.bump
    greeks = tuple(by_name[g] for g in request.greeks)
    return IntradayValuationResult(
        price=price, contingent_pv=outcome.contingent_pv, pending_receivable_pv=pending_pv,
        paid_cash=ctx.numerical.paid_cash, units="price_per_contract", valuation_timestamp=ctx.valuation_timestamp,
        phase=ctx.phase, provisional=ctx.provisional, assumptions=ctx.reconstruction.assumptions,
        continuous_assumption=ctx.reconstruction.continuous_assumption, lifecycle=lifecycle,
        cashflows=tuple(cashflows), greeks=greeks, profile_identity=request.variance_profile.identity(),
        session_identity=request.session_calendar.identity(), market_snapshot_id=ctx.market_snapshot_id,
        context_identity=ctx.identity, engine=engine_class_path(engine), method=outcome.method,
        numerical=numerical, records=tuple(records))


def _dispatch_and_compare(session, engine, request, ctx, outcome, price) -> str:
    """Re-dispatch the twin through the execution kernel; the value must match the route's."""
    state = ctx.numerical.lifecycle_state
    stateful = bool(getattr(engine, "supports_lifecycle_state", False))
    req = PricingRequest(product=ctx.numerical.product, pricing_env=ctx.pricing_env, request_id=request.request_id,
                         lifecycle_state=state if stateful else None)
    fw = session.execute(engine, req)
    # A stateful engine prices the pending receivables itself; a stateless one returns the contingent claim only.
    fw_value = float(fw.value) if stateful else float(fw.value) + (price - outcome.contingent_pv)
    if not is_close(fw_value, price, rel_tol=_PARITY_TOL, abs_tol=_PARITY_TOL):
        raise DeterminismViolation(f"framework dispatch {fw_value!r} != intraday route {price!r}")
    return f"manifest:{fw.manifest.request_fingerprint}"
