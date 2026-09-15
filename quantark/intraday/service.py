"""value_intraday: resolve once, dispatch once, report everything.

The design's sequence: (1) resolve the context; (2) look up the capability row
for the product/engine/monitoring; (3) price the twin through the route;
(4) add pending receivables from the time-based ledger; (5) itemise cashflows
with provenance; (6) greeks — delivered by Plan 3, fail closed until then.
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
from quantark.intraday.events import EventKind
from quantark.intraday.request import IntradayValuationRequest
from quantark.intraday.result import CashflowComponent, IntradayValuationResult
from quantark.intraday.twin import event_for_cashflow
from quantark.util.numerical import is_close

_LIFECYCLE_FIELDS = ("alive", "knocked_in", "knocked_out", "matured", "coupon_memory_count")
#: Route methods that never call the engine; there is nothing for the kernel to re-dispatch.
_ENGINE_FREE_METHODS = ("terminated", "deterministic_zero_variance")
_PARITY_TOL = 1e-12


def _monitoring(ctx) -> str:
    if ctx.timeline.continuous_ki_barrier is not None:
        return "continuous"
    if all(e.kind is EventKind.TERMINAL for e in ctx.timeline.events):
        return "terminal"
    return "discrete"


def _cashflow_origin(cf, ctx):
    """(event_id, provenance) of a ledger flow: provisional iff its event was determined by an assumption."""
    event = event_for_cashflow(cf, ctx.timeline)
    if event is None:
        return None, "confirmed"          # carried in from the authoritative checkpoint
    assumed = {eid for a in ctx.reconstruction.assumptions for eid in a.event_ids}
    return event.event_id, ("provisional" if event.event_id in assumed else "confirmed")


def value_intraday(engine, request: IntradayValuationRequest, *, session=None) -> IntradayValuationResult:
    """Value one contract at an intraday timestamp; see the module docstring for the sequence."""
    ctx = resolve_context(request)
    if request.greeks:
        raise CapabilityError("intraday greeks are delivered by plan 3; request price only")
    require_capability(request.product, engine, monitoring=_monitoring(ctx), outputs=("price",))
    outcome = route_for(ctx, engine).price(ctx, engine)
    state = ctx.numerical.lifecycle_state
    pending_pv = float(pending_receivable_pv(state, ctx.pricing_env)) if state is not None else 0.0
    price = outcome.contingent_pv + pending_pv
    records = list(outcome.records) + [f"context:{ctx.identity}"]
    if session is not None:
        if outcome.method in _ENGINE_FREE_METHODS:
            records.append(f"manifest:not-dispatched ({outcome.method} route calls no engine)")
        else:
            records.append(_dispatch_and_compare(session, engine, request, ctx, outcome, price))
    cashflows = [CashflowComponent("contingent", outcome.contingent_pv, "model")]
    if state is not None:
        for cf in state.ledger.cashflows:
            event_id, provenance = _cashflow_origin(cf, ctx)
            if cf.payment_time > 0.0:
                pv = float(cf.amount) * float(ctx.pricing_env.get_discount_factor(cf.payment_time))
                cashflows.append(CashflowComponent("pending_receivable", pv, provenance, event_id=event_id,
                                                   cashflow_id=cf.cashflow_id, payment_tau=float(cf.payment_time)))
            else:
                cashflows.append(CashflowComponent("paid", float(cf.amount), provenance, event_id=event_id,
                                                   cashflow_id=cf.cashflow_id, payment_tau=float(cf.payment_time)))
    lifecycle = {k: getattr(state, k) for k in _LIFECYCLE_FIELDS} if state is not None else {}
    return IntradayValuationResult(
        price=price, contingent_pv=outcome.contingent_pv, pending_receivable_pv=pending_pv,
        paid_cash=ctx.numerical.paid_cash, units="price_per_contract", valuation_timestamp=ctx.valuation_timestamp,
        phase=ctx.phase, provisional=ctx.provisional, assumptions=ctx.reconstruction.assumptions,
        continuous_assumption=ctx.reconstruction.continuous_assumption, lifecycle=lifecycle,
        cashflows=tuple(cashflows), greeks=(), profile_identity=request.variance_profile.identity(),
        session_identity=request.session_calendar.identity(), market_snapshot_id=ctx.market_snapshot_id,
        context_identity=ctx.identity, engine=engine_class_path(engine), method=outcome.method,
        numerical=dict(outcome.numerical), records=tuple(records))


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
