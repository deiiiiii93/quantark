"""Engine routes for intraday valuation, dispatched on the exact engine class path."""
from __future__ import annotations

from quantark.execution.errors import CapabilityError
from quantark.intraday.capability import engine_class_path
from quantark.intraday.engines.base import EnginePriceOutcome, IntradayEngineRoute

_ROUTES = {}


def register_route(engine_class_path_: str, factory) -> None:
    _ROUTES[engine_class_path_] = factory


def route_for(ctx, engine) -> IntradayEngineRoute:
    path = engine_class_path(engine)
    try:
        return _ROUTES[path]()
    except KeyError:
        raise CapabilityError(f"no intraday route registered for {path}") from None


from quantark.intraday.engines.analytical_digital import AnalyticalDigitalRoute  # noqa: E402
from quantark.intraday.engines.quad_v2 import QuadV2Route  # noqa: E402

register_route("quantark.asset.equity.engine.analytical.digital_option_engine.DigitalOptionAnalyticalEngine",
               AnalyticalDigitalRoute)
for _name in ("SnowballQuadEngineV2", "PhoenixQuadEngineV2"):
    register_route(f"quantark.asset.equity.engine.quad.v2.engine.{_name}", QuadV2Route)

__all__ = ["EnginePriceOutcome", "IntradayEngineRoute", "register_route", "route_for"]
