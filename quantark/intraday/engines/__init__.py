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


from quantark.intraday.engines.analytical_barrier import AnalyticalBarrierRoute  # noqa: E402
from quantark.intraday.engines.analytical_digital import AnalyticalDigitalRoute  # noqa: E402
from quantark.intraday.engines.quad_v2 import QuadV2Route  # noqa: E402

register_route("quantark.asset.equity.engine.analytical.digital_option_engine.DigitalOptionAnalyticalEngine",
               AnalyticalDigitalRoute)
register_route("quantark.asset.equity.engine.analytical.barrier_analytical_engine.BarrierAnalyticalEngine",
               AnalyticalBarrierRoute)
register_route("quantark.asset.equity.engine.analytical.one_touch_analytical_engine.OneTouchAnalyticalEngine",
               AnalyticalBarrierRoute)
for _name in ("SnowballQuadEngineV2", "PhoenixQuadEngineV2", "KOResetSnowballQuadEngineV2"):
    register_route(f"quantark.asset.equity.engine.quad.v2.engine.{_name}", QuadV2Route)

from quantark.intraday.engines.pde import PDERoute  # noqa: E402

PDE_SOLVER_PATHS = {
    "SnowballPDESolver": "quantark.asset.equity.engine.pde.snowball_pde_solver.SnowballPDESolver",
    "PhoenixPDESolver": "quantark.asset.equity.engine.pde.phoenix_pde_solver.PhoenixPDESolver",
    "KOResetSnowballPDESolver": "quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver.KOResetSnowballPDESolver",
    "BarrierPDESolver": "quantark.asset.equity.engine.pde.barrier_pde_solver.BarrierPDESolver",
    "OneTouchPDESolver": "quantark.asset.equity.engine.pde.one_touch_pde_solver.OneTouchPDESolver",
}
for _path in PDE_SOLVER_PATHS.values():
    register_route(_path, PDERoute)

__all__ = ["EnginePriceOutcome", "IntradayEngineRoute", "register_route", "route_for"]
