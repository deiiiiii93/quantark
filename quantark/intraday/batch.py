"""Batch valuation, spot curves and aggregation that keep every provenance flag.

- ``value_intraday_many``: caller-ordered; with ``collect_errors`` a failure is a typed ``IntradayFailure``, never a
  silent ``None``.
- ``spot_curve``: ONE resolved context. Its confirmed and assumed fixings come from the request's base snapshot and
  are shared by every point — a curve never rebuilds an assumption from a curve spot. QUAD V2 prepares the operator
  once over the declared spots and reads price, delta and gamma from it, under the same Gate C certificate as a
  single point Greek; other routes price each spot on the same context with only the spot moved.
- ``aggregate_intraday``: quantities scale prices, paid cash and Greeks; the book is provisional if any position is,
  and a Greek is summed only when every position reports it ``ok`` under one convention and unit.
"""
from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from math import isfinite
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

from quantark.intraday.context import resolve_context
from quantark.intraday.engines import route_for
from quantark.intraday.fixings import AssumedFixing, ContinuousHistoryAssumption
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
    status: str                  # greek status: "ok" | "undefined" | "unqualified" | "not_requested"
    reason: str
    assumptions: Tuple[AssumedFixing, ...] = ()
    #: This spot's own price evidence (PDE resolution, MC standard error, ...). A route that
    #: prices each spot separately can resolve one and not the next, so it belongs per point.
    numerical: Mapping[str, object] = field(default_factory=dict)
    method: str = ""

    def __post_init__(self):
        object.__setattr__(self, "numerical", MappingProxyType(dict(self.numerical)))

    def to_dict(self) -> dict:
        """JSON-serialisable view (``dataclasses.asdict`` cannot copy the frozen mappings)."""
        from quantark.intraday.result import _jsonable
        return _jsonable(self)


@dataclass(frozen=True)
class SpotCurve(Sequence):
    """A curve of one resolved context, carrying that context's provenance ONCE.

    The points alone cannot say whether the curve is provisional, what history was
    assumed to build it, or which clock and market it was resolved on — and a
    consumer that cannot tell a confirmed curve from one resting on an uncovered
    touch interval will treat them alike (review 2026-09-16 finding 15). It is a
    Sequence so existing callers keep indexing and iterating the points.
    """

    points: Tuple[SpotCurvePoint, ...]
    valuation_timestamp: datetime
    phase: object
    provisional: bool
    assumptions: Tuple[AssumedFixing, ...]
    continuous_assumption: Optional[ContinuousHistoryAssumption]
    lifecycle: Mapping[str, object]
    profile_identity: tuple
    session_identity: tuple
    market_snapshot_id: str
    context_identity: str
    engine: str
    method: str
    numerical: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "points", tuple(self.points))
        object.__setattr__(self, "lifecycle", MappingProxyType(dict(self.lifecycle)))
        object.__setattr__(self, "numerical", MappingProxyType(dict(self.numerical)))

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index):
        return self.points[index]

    def to_dict(self) -> dict:
        """JSON-serialisable view (``dataclasses.asdict`` cannot copy the frozen mappings)."""
        from quantark.intraday.result import _jsonable
        return _jsonable(self)


def _spot_price(ctx, engine, spot: float):
    """(price, price evidence, method) at one curve spot, holding nothing that owns a market.

    A route outcome carries the engine instance that produced it, and a PDE solver
    owns its solved layout and this spot's coefficient memos. Returning plain data
    lets each spot's market die before the next one is built: a 101-point PDE curve
    that kept them all alive grew past 8 GiB.
    """
    from quantark.intraday.greeks import cell_outcome
    price, outcome = cell_outcome(_spot_context(ctx, spot), engine)
    return price, dict(outcome.numerical), outcome.method


def _spot_context(ctx, spot: float):
    from quantark.intraday.greeks import with_pricing_env
    env = deepcopy(ctx.pricing_env)
    env.spot_quote.spot = float(spot)
    return with_pricing_env(ctx, env, f"curve_spot:{float(spot)!r}")


_LIFECYCLE_FIELDS = ("alive", "knocked_in", "knocked_out", "matured", "expired", "coupon_memory_count")


def _curve(ctx, engine, points, method: str, numerical) -> SpotCurve:
    """Wrap the points in the resolved context's provenance — the same contract a single value reports."""
    from quantark.intraday.capability import engine_class_path

    state = ctx.numerical.lifecycle_state
    return SpotCurve(
        points=tuple(points), valuation_timestamp=ctx.valuation_timestamp, phase=ctx.phase,
        provisional=ctx.provisional, assumptions=ctx.reconstruction.assumptions,
        continuous_assumption=ctx.reconstruction.continuous_assumption,
        lifecycle={k: getattr(state, k) for k in _LIFECYCLE_FIELDS if hasattr(state, k)} if state is not None else {},
        profile_identity=ctx.request.variance_profile.identity(),
        session_identity=ctx.request.session_calendar.identity(),
        market_snapshot_id=ctx.market_snapshot_id, context_identity=ctx.identity,
        engine=engine_class_path(engine), method=method, numerical=numerical)


def spot_curve(engine, request: IntradayValuationRequest, spots: Sequence[float]) -> SpotCurve:
    """Price (and, on QUAD V2, point delta/gamma) at each spot of one resolved context, in the given order."""
    from quantark.intraday.capability import require_capability
    from quantark.intraday.engines.quad_v2 import QuadV2Route
    from quantark.intraday.greeks import discontinuity_at_spot, point_certificate_gap
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
        shared = dict(getattr(prepared, "diagnostics", {}) or {})
        # The conditional economics are shared, but each spot must remain in
        # the demonstrated spatial envelope (review R4).
        points = []
        for i, s in enumerate(spots):
            gap = point_certificate_gap(ctx, engine, route, spot=s)
            delta, gamma = float(values["delta"][i]), float(values["gamma"][i])
            if jumps[i] or not (isfinite(delta) and isfinite(gamma)):
                points.append(SpotCurvePoint(s, float(values["price"][i]), None, None, "undefined",
                                             jumps[i] or "payoff discontinuity of an unfixed event at the query spot",
                                             assumptions, method="quad_v2_prepared"))
            elif gap:
                points.append(SpotCurvePoint(s, float(values["price"][i]), None, None, "unqualified",
                                             f"kernel derivative of the discretised value: {gap}", assumptions,
                                             method="quad_v2_prepared"))
            else:
                points.append(SpotCurvePoint(s, float(values["price"][i]), delta, gamma, "ok", "", assumptions,
                                             method="quad_v2_prepared"))
        # one prepared operator serves every spot, so its evidence is the curve's, not a point's
        return _curve(ctx, engine, points, "quad_v2_prepared", shared)
    points = []
    for s, jump in zip(spots, jumps):
        # one context at a time: each carries a copied market whose engine memos live as long as it does
        price, numerical, method = _spot_price(ctx, engine, s)
        common = dict(assumptions=assumptions, numerical=numerical, method=method)
        if num.terminated and not jump:
            points.append(SpotCurvePoint(s, price, 0.0, 0.0, "ok", "", **common))
        elif jump:
            points.append(SpotCurvePoint(s, price, None, None, "undefined", jump, **common))
        else:
            points.append(SpotCurvePoint(s, price, None, None, "not_requested",
                                         f"{type(route).__name__} curves price each spot; request point greeks per spot",
                                         **common))
    return _curve(ctx, engine, points, points[0].method if points else "", {})


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
