"""Greek (re-)valuation conventions for smile-aware risk.

Defines HOW the smile moves when the underlying is bumped:

- STICKY_STRIKE: implied vol at a fixed strike is held constant on a spot bump.
- STICKY_MONEYNESS: the smile is re-anchored to the new spot (vol at fixed
  moneyness K/S is held): sigma'(K) = sigma(K * S / S').
- STICKY_DELTA : vol at fixed Black delta is held. NOTE: this is NOT the
  moneyness re-anchor (true sticky delta needs a per-expiry delta-strike
  solver). VannaVolgaVolSurface.rebound() historically implements the
  moneyness re-anchor under this name (FX legacy).
- MODEL        : bump the model's own state/parameters and re-derive the smile.
- BARTLETT     : SABR's vega + the rho-induced backbone shift (Bartlett delta).
"""

from enum import Enum


class GreekConvention(Enum):
    STICKY_STRIKE = "sticky_strike"
    STICKY_MONEYNESS = "sticky_moneyness"
    STICKY_DELTA = "sticky_delta"
    MODEL = "model"
    BARTLETT = "bartlett"
