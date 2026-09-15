"""Batch valuation, spot curves and aggregation that keep every provenance flag.

- ``value_intraday_many``: caller-ordered; with ``collect_errors`` a failure is a typed ``IntradayFailure``, never a
  silent ``None``.
- ``spot_curve``: ONE resolved context. Its confirmed and assumed fixings come from the request's base snapshot and
  are shared by every point — a curve never rebuilds an assumption from a curve spot. QUAD V2 prepares the operator
  once over the declared spots and reads price, delta and gamma from it; other routes price each spot on the same
  context with only the spot moved.
- ``aggregate_intraday``: quantities scale prices, paid cash and Greeks; the book is provisional if any position is,
  and a Greek is summed only when every position reports it ``ok`` under one convention and unit.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Optional, Sequence, Tuple

from quantark.intraday.context import resolve_context
from quantark.intraday.engines import route_for
from quantark.intraday.fixings import AssumedFixing
from quantark.intraday.request import IntradayValuationRequest
from quantark.intraday.result import IntradayValuationResult


@dataclass(frozen=True)
class IntradayFailure:
    item_id: str
    error_type: str
    message: str


def value_intraday_many(items: Sequence[Tuple[object, IntradayValuationRequest]], *, session=None,
                        collect_errors: bool = False) -> list:
    """Value (engine, request) pairs in caller order; fail fast unless ``collect_errors``."""
    from quantark.intraday.service import value_intraday

    out = []
    for index, (engine, request) in enumerate(items):
        if not collect_errors:
            out.append(value_intraday(engine, request, session=session))
            continue
        try:
            out.append(value_intraday(engine, request, session=session))
        except Exception as exc:  # noqa: BLE001 - typed into the failure record
            out.append(IntradayFailure(request.request_id or str(index), type(exc).__name__, str(exc)))
    return out


@dataclass(frozen=True)
class SpotCurvePoint:
    spot: float
    price: float
    delta: Optional[float]
    gamma: Optional[float]
    status: str                  # "ok" | "undefined" | "not_requested"
    reason: str
    assumptions: Tuple[AssumedFixing, ...] = ()


def _spot_context(ctx, spot: float):
    from quantark.intraday.greeks import with_pricing_env
    env = deepcopy(ctx.pricing_env)
    env.spot_quote.spot = float(spot)
    return with_pricing_env(ctx, env, f"curve_spot:{float(spot)!r}")


def spot_curve(engine, request: IntradayValuationRequest, spots: Sequence[float]) -> Tuple[SpotCurvePoint, ...]:
    """Price (and, on QUAD V2, point delta/gamma) at each spot of one resolved context, in the given order."""
    from quantark.intraday.capability import require_capability
    from quantark.intraday.engines.quad_v2 import QuadV2Route
    from quantark.intraday.greeks import cell_price, discontinuity_at_spot
    from quantark.intraday.service import _monitoring

    ctx = resolve_context(request)
    require_capability(request.product, engine, monitoring=_monitoring(ctx))
    spots = [float(s) for s in spots]
    assumptions = ctx.reconstruction.assumptions
    route = route_for(ctx, engine)
    jumps = [discontinuity_at_spot(ctx, s) for s in spots]
    num = ctx.numerical
    if isinstance(route, QuadV2Route) and not num.terminated:
        prepared = engine.prepare(num.product, ctx.pricing_env, spot_levels=spots, event_phase=ctx.phase.value,
                                  lifecycle_state=num.lifecycle_state)
        values = prepared.evaluate(spots)
        points = []
        for i, s in enumerate(spots):
            delta, gamma = float(values["delta"][i]), float(values["gamma"][i])
            if jumps[i] or not (isfinite(delta) and isfinite(gamma)):
                points.append(SpotCurvePoint(s, float(values["price"][i]), None, None, "undefined",
                                             jumps[i] or "payoff discontinuity of an unfixed event at the query spot", assumptions))
            else:
                points.append(SpotCurvePoint(s, float(values["price"][i]), delta, gamma, "ok", "", assumptions))
        return tuple(points)
    points = []
    for s, jump in zip(spots, jumps):
        # one context at a time: each carries a copied market whose engine memos live as long as it does
        price = cell_price(_spot_context(ctx, s), engine)
        if num.terminated and not jump:
            points.append(SpotCurvePoint(s, price, 0.0, 0.0, "ok", "", assumptions))
        elif jump:
            points.append(SpotCurvePoint(s, price, None, None, "undefined", jump, assumptions))
        else:
            points.append(SpotCurvePoint(s, price, None, None, "not_requested",
                                         f"{type(route).__name__} curves price each spot; request point greeks per spot",
                                         assumptions))
    return tuple(points)


@dataclass(frozen=True)
class IntradayPortfolioResult:
    total_price: float
    total_paid_cash: float
    provisional: bool
    provisional_positions: Tuple[str, ...]
    greeks: Mapping[str, Optional[float]]
    greek_status: Mapping[str, str]
    positions: Tuple[Tuple[str, float, IntradayValuationResult], ...]


def aggregate_intraday(results: Sequence[Tuple[str, float, IntradayValuationResult]]) -> IntradayPortfolioResult:
    """Position-weighted totals; a Greek is summed only when every position reports it ok in one convention/unit."""
    positions = tuple((str(pid), float(q), r) for pid, q, r in results)
    names = []
    for _, _, r in positions:
        for g in r.greeks:
            if g.name not in names:
                names.append(g.name)
    greeks, status = {}, {}
    for name in names:
        problems, total, labels = [], 0.0, set()
        for pid, q, r in positions:
            found = [g for g in r.greeks if g.name == name]
            if not found:
                problems.append(f"{pid}: not requested")
                continue
            g = found[0]
            labels.add((g.convention, g.unit))
            if g.status != "ok":
                problems.append(f"{pid}: {g.status} ({g.reason})")
            else:
                total += q * float(g.value)
        if len(labels) > 1:
            problems.append(f"mixed conventions/units {sorted(labels)}")
        greeks[name] = None if problems else total
        status[name] = "; ".join(problems) if problems else "ok"
    provisional_positions = tuple(pid for pid, _, r in positions if r.provisional)
    return IntradayPortfolioResult(
        total_price=sum(q * r.price for _, q, r in positions),
        total_paid_cash=sum(q * r.paid_cash for _, q, r in positions),
        provisional=bool(provisional_positions), provisional_positions=provisional_positions,
        greeks=greeks, greek_status=status, positions=positions)
