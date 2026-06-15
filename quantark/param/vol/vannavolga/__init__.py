"""
Vanna-Volga FX smile construction (param layer).

Public API:
    - Conventions: ``FXEnv``, ``DeltaConvention``, ``atm_strike``,
      ``strikes_25d``, ``bs_delta``, ``choose_delta_convention``
    - GK pricing/greeks: ``GKInput``, ``price_gk``, ``greeks_gk``
    - Smile inputs: ``SmileQuotes``, ``rr_bf_costs``,
      ``broker_strangle_sigma_1vol``
    - VV core: ``InstrumentGreeks``, ``compute_omega``,
      ``vv_adjustment_simple``, ``vv_adjustment_matrix``
    - Surface adapter: ``VannaVolgaVolSurface``
"""

from .bs_fx import GKInput, greeks_gk, price_gk
from .market_conventions import (
    DeltaConvention,
    FXEnv,
    atm_strike,
    bs_delta,
    choose_delta_convention,
    strike_for_delta,
    strikes_25d,
)
from .smile_builder import SmileQuotes, broker_strangle_sigma_1vol, rr_bf_costs
from .vv_core import (
    InstrumentGreeks,
    compute_omega,
    vv_adjustment_matrix,
    vv_adjustment_simple,
)
from .vv_surface import VannaVolgaVolSurface

__all__ = [
    "FXEnv",
    "DeltaConvention",
    "atm_strike",
    "strike_for_delta",
    "strikes_25d",
    "bs_delta",
    "choose_delta_convention",
    "GKInput",
    "price_gk",
    "greeks_gk",
    "SmileQuotes",
    "rr_bf_costs",
    "broker_strangle_sigma_1vol",
    "InstrumentGreeks",
    "compute_omega",
    "vv_adjustment_simple",
    "vv_adjustment_matrix",
    "VannaVolgaVolSurface",
]
