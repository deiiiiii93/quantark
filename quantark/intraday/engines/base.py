"""Route contract: a route prices the resolved context's twin through one engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Protocol, Tuple


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

    def __post_init__(self):
        object.__setattr__(self, "numerical", MappingProxyType(dict(self.numerical)))
        object.__setattr__(self, "components", MappingProxyType(dict(self.components)))
        object.__setattr__(self, "records", tuple(self.records))


class IntradayEngineRoute(Protocol):
    def price(self, ctx, engine) -> EnginePriceOutcome: ...
