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
    #: difference of exact values is exact as a MOVE; anything else needs a certificate after differencing.
    exact: bool = False

    def __post_init__(self):
        object.__setattr__(self, "numerical", MappingProxyType(dict(self.numerical)))
        object.__setattr__(self, "components", MappingProxyType(dict(self.components)))
        object.__setattr__(self, "records", tuple(self.records))


@dataclass(frozen=True)
class PointGreeks:
    """Spot derivatives of the conditional price function at the valuation spot."""

    delta: Optional[float]
    gamma: Optional[float]
    status: str                   # "ok" | "undefined" | "unqualified" | "failed"
    reason: str
    evidence: str                 # "kernel_derivative" | "closed_form" | "closed_form_fd" | "grid_stencil" | "paired_rqmc" | "terminated"
    uncertainty: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "uncertainty", MappingProxyType(dict(self.uncertainty)))
        if self.status != "ok":
            object.__setattr__(self, "delta", None)
            object.__setattr__(self, "gamma", None)


TERMINATED_POINT_GREEKS = PointGreeks(0.0, 0.0, "ok", "", "terminated")

#: ``point_greeks(ctx, engine, *, certify=True)``: with ``certify`` a numerical route publishes a value only inside a
#: Gate C certificate. ``certify=False`` returns the raw estimator with the route's OWN evidence only (resolution,
#: batches, finiteness) -- for the Gate C ladder, which produces the certificates and so must never read them.


class IntradayEngineRoute(Protocol):
    def price(self, ctx, engine) -> EnginePriceOutcome: ...
