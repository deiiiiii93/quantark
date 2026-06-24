"""Greek (re-)valuation conventions for smile-aware risk.

Defines HOW the smile moves when the underlying is bumped:

- STICKY_STRIKE: implied vol at a fixed strike is held constant on a spot bump.
- STICKY_DELTA : the smile is re-anchored to the new spot (vol at fixed delta /
  moneyness is held). This is what VannaVolgaVolSurface.rebound() implements.
- MODEL        : bump the model's own state/parameters and re-derive the smile.
- BARTLETT     : SABR's vega + the rho-induced backbone shift (Bartlett delta).
"""

from enum import Enum


class GreekConvention(Enum):
    STICKY_STRIKE = "sticky_strike"
    STICKY_DELTA = "sticky_delta"
    MODEL = "model"
    BARTLETT = "bartlett"
