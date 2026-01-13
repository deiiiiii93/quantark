"""
Quadrature-based pricing engines for equity derivatives.
"""

from .european_quad_engine import EuropeanQuadEngine
from .discrete_quad_engine import DiscreteQuadEngine, BarrierQuadEngine, OneTouchQuadEngine
from .quad_adapters import QuadInputAdapter, register_quad_adapter, resolve_quad_adapter

__all__ = [
    "EuropeanQuadEngine",
    "DiscreteQuadEngine",
    "BarrierQuadEngine",
    "OneTouchQuadEngine",
    "QuadInputAdapter",
    "register_quad_adapter",
    "resolve_quad_adapter",
]
