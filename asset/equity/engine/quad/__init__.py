"""
Quadrature-based pricing engines for equity derivatives.
"""

from .european_quad_engine import EuropeanQuadEngine
from .discrete_quad_engine import DiscreteQuadEngine, BarrierQuadEngine, OneTouchQuadEngine
from .snowball_quad_engine import SnowballQuadEngine
from .phoenix_quad_engine import PhoenixQuadEngine
from .quad_adapters import QuadInputAdapter, register_quad_adapter, resolve_quad_adapter

__all__ = [
    "EuropeanQuadEngine",
    "DiscreteQuadEngine",
    "BarrierQuadEngine",
    "OneTouchQuadEngine",
    "SnowballQuadEngine",
    "PhoenixQuadEngine",
    "QuadInputAdapter",
    "register_quad_adapter",
    "resolve_quad_adapter",
]
