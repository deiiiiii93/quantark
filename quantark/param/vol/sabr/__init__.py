"""
SABR volatility smile/surface (Hagan lognormal approximation).

Public API:
    - Hagan implied-vol formulas: ``sabr_implied_vol_black``,
      ``sabr_atm_implied_vol_black``, ``sabr_implied_vol_black_shifted``,
      ``sabr_generate_vol_surface``
    - Calibration: ``calibrate_sabr_slice``, ``calibrate_sabr_surface``
    - Surface adapter: ``SABRVolSurface``
"""

from .calibration import calibrate_sabr_slice, calibrate_sabr_surface
from .hagan import (
    sabr_atm_implied_vol_black,
    sabr_generate_vol_surface,
    sabr_implied_vol_black,
    sabr_implied_vol_black_shifted,
)
from .diagnostics import ArbitrageReport, butterfly_density, check_arbitrage
from .sabr_surface import SABRVolSurface

__all__ = [
    "sabr_implied_vol_black",
    "sabr_atm_implied_vol_black",
    "sabr_implied_vol_black_shifted",
    "sabr_generate_vol_surface",
    "calibrate_sabr_slice",
    "calibrate_sabr_surface",
    "SABRVolSurface",
    "ArbitrageReport",
    "check_arbitrage",
    "butterfly_density",
]
