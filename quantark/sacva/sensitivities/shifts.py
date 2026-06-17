"""Market-risk bump helpers for SA-CVA sensitivities (spec §3.4).

One-sided relative bumps for equity/FX spot and volatility (MAR50.63/50.70):
``s_cva = (CVA(x·(1+shift)) − CVA(x)) / shift`` with ``shift`` the divisor below.
Bumps clone the pricing environment (spot quote / vol surface) so the base trade
objects are untouched and re-runs share common random numbers (same MC seed).
"""

from dataclasses import replace

from quantark.param import SpotQuote
from quantark.param.vol.vol_surface import VolatilitySurface
from quantark.util.exceptions import ValidationError

# relative shift sizes / divisors (MAR50.63): 1% for equity/FX/commodity spot & all vega
EQUITY_SPOT_SHIFT = 0.01
EQUITY_VOL_SHIFT = 0.01


class ScaledVolSurface(VolatilitySurface):
    """Multiplicative wrapper: ``get_vol = factor · base.get_vol`` (relative vega bump)."""

    def __init__(self, base: VolatilitySurface, factor: float):
        if not (factor > 0):
            raise ValidationError("vol scale factor must be > 0")
        self.base = base
        self.factor = float(factor)

    def get_vol(self, strike: float, time_to_maturity: float, spot: float) -> float:
        return self.factor * self.base.get_vol(strike, time_to_maturity, spot)


def bump_spot_env(env, factor):
    """Clone ``env`` with spot scaled by ``factor`` (e.g. 1.01 for a +1% bump)."""
    if env.spot_quote is None:
        raise ValidationError("cannot bump spot: env has no spot_quote")
    new_spot = SpotQuote(spot=float(env.spot) * float(factor),
                         asset_name=getattr(env.spot_quote, "asset_name", None))
    return replace(env, spot_quote=new_spot)


def bump_vol_env(env, factor):
    """Clone ``env`` with the whole vol surface scaled by ``factor``."""
    if env.vol_surface is None:
        raise ValidationError("cannot bump vol: env has no vol_surface")
    return replace(env, vol_surface=ScaledVolSurface(env.vol_surface, factor))
