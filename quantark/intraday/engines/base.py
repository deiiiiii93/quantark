"""Route contract: a route prices the resolved context's twin through one engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional, Protocol, Tuple


@dataclass(frozen=True)
class EnginePriceOutcome:
    contingent_pv: float
    method: str                                   # e.g. "analytical_bs_effective_variance", "quad_v2_direct"
    numerical: Mapping[str, object] = field(default_factory=dict)
    components: Mapping[str, float] = field(default_factory=dict)   # engine-native components when available
    records: Tuple[str, ...] = ()
    #: The engine instance that produced the value when the route priced on a
    #: refined clone; the kernel parity dispatch must use the same instance.
    engine_used: object = field(default=None, compare=False)
    #: True when the value carries no discretisation or sampling error: a closed form the route has proven
    #: exact here, or a claim already decided (a fixed ledger, an outcome decided on the known spot). A finite
    #: difference of exact values is exact as a MOVE; any other value carries its own discretisation or sampling error.
    exact: bool = False

    def __post_init__(self):
        object.__setattr__(self, "numerical", MappingProxyType(dict(self.numerical)))
        object.__setattr__(self, "components", MappingProxyType(dict(self.components)))
        object.__setattr__(self, "records", tuple(self.records))


@dataclass(frozen=True)
class PointGreeks:
    """Spot derivatives of the conditional price function at the valuation spot.

    ``status`` and ``reason`` apply to both outputs unless ``statuses`` / ``reasons`` override one:
    a finite delta survives a gamma that could not be computed.
    """

    delta: Optional[float]
    gamma: Optional[float]
    status: str                   # "ok" | "undefined" | "failed"
    reason: str
    evidence: str                 # "kernel_derivative" | "closed_form" | "closed_form_fd" | "grid_stencil" | "paired_rqmc" | "terminated"
    uncertainty: Mapping[str, float] = field(default_factory=dict)
    statuses: Mapping[str, str] = field(default_factory=dict)
    reasons: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "uncertainty", MappingProxyType(dict(self.uncertainty)))
        object.__setattr__(self, "statuses", MappingProxyType(dict(self.statuses)))
        object.__setattr__(self, "reasons", MappingProxyType(dict(self.reasons)))
        for name in ("delta", "gamma"):
            if self.status_of(name) != "ok":
                object.__setattr__(self, name, None)

    def status_of(self, name: str) -> str:
        return self.statuses.get(name, self.status)

    def reason_of(self, name: str) -> str:
        return self.reasons.get(name, self.reason)


def point_greeks_from_estimates(delta: float, gamma: float, evidence: str, *, note: str = "",
                                uncertainty: Optional[Mapping[str, float]] = None) -> PointGreeks:
    """Validate each estimate on its own: a finite one is ``ok``, a non-finite one is ``failed``.

    ``undefined`` is never inferred from a non-finite number here. A payoff discontinuity at the query
    spot is established separately, before the route is asked (``greeks.discontinuity_at_spot``).
    """
    from math import isfinite
    statuses, reasons = {}, {}
    for name, value in (("delta", delta), ("gamma", gamma)):
        if not isfinite(value):
            statuses[name] = "failed"
            reasons[name] = f"non-finite {name} estimate ({value!r}); no discontinuity is established at the query spot"
    return PointGreeks(delta, gamma, "ok", note, evidence, uncertainty or {}, statuses, reasons)


TERMINATED_POINT_GREEKS = PointGreeks(0.0, 0.0, "ok", "", "terminated")


class IntradayEngineRoute(Protocol):
    def price(self, ctx, engine) -> EnginePriceOutcome: ...
