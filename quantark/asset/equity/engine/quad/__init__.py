"""
Quadrature-based pricing engines for equity derivatives.
"""

from .european_quad_engine import EuropeanQuadEngine
from .discrete_quad_engine import DiscreteQuadEngine, BarrierQuadEngine, OneTouchQuadEngine
from .snowball_quad_engine import SnowballQuadEngine
from .ko_reset_snowball_quad_engine import KOResetSnowballQuadEngine
from .phoenix_quad_engine import PhoenixQuadEngine
from .v2 import SnowballQuadEngineV2, PhoenixQuadEngineV2, KOResetSnowballQuadEngineV2
from .quad_adapters import QuadInputAdapter, register_quad_adapter, resolve_quad_adapter

__all__ = [
    "EuropeanQuadEngine",
    "DiscreteQuadEngine",
    "BarrierQuadEngine",
    "OneTouchQuadEngine",
    "SnowballQuadEngine",
    "KOResetSnowballQuadEngine",
    "PhoenixQuadEngine",
    "SnowballQuadEngineV2",
    "PhoenixQuadEngineV2",
    "KOResetSnowballQuadEngineV2",
    "QuadInputAdapter",
    "register_quad_adapter",
    "resolve_quad_adapter",
]
