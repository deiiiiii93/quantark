"""
PDE pricing engines for equity derivatives.

This module provides finite difference solvers for pricing various
option types using the Black-Scholes PDE framework.
"""

from .time_grid import TimeGrid
from .spatial_grid import SpatialGrid
from .base_pde_solver import BasePDESolver
from .european_pde_solver import EuropeanPDESolver
from .american_pde_solver import AmericanPDESolver
from .barrier_pde_solver import BarrierPDESolver
from .double_barrier_pde_solver import DoubleBarrierPDESolver
from .one_touch_pde_solver import OneTouchPDESolver
from .double_one_touch_pde_solver import DoubleOneTouchPDESolver
from .snowball_pde_solver import SnowballPDESolver
from .phoenix_pde_solver import PhoenixPDESolver

__all__ = [
    # Grid utilities
    'TimeGrid',
    'SpatialGrid',
    # Solvers
    'BasePDESolver',
    'EuropeanPDESolver',
    'AmericanPDESolver',
    'BarrierPDESolver',
    'DoubleBarrierPDESolver',
    'OneTouchPDESolver',
    'DoubleOneTouchPDESolver',
    'SnowballPDESolver',
    'PhoenixPDESolver',
]
