"""Spot-shock surface conventions for equity smiles (spec WP4.4).

``shocked_surface`` returns the surface to price with AFTER a spot move
``s_base -> s_shocked``, per convention:

- STICKY_STRIKE: the surface is unchanged in K (held fixed: vol at each
  absolute strike). Returns the input surface itself.
- STICKY_MONEYNESS: the smile rides the spot (held fixed: vol at each
  moneyness K/S): sigma'(K) = sigma(K * s_base / s_shocked).
- STICKY_DELTA: rejected in this iteration — the moneyness re-anchor is NOT
  sticky delta (true sticky delta preserves Black delta and needs a
  per-expiry delta-strike solver); refusing beats silently mislabeling.
"""
from __future__ import annotations

from quantark.util.enum.greek_conventions import GreekConvention
from quantark.util.exceptions import ValidationError


class _StickyMoneynessView:
    """get_vol-compatible re-anchored view; everything else delegates."""

    def __init__(self, base, s_base: float, s_shocked: float):
        if s_base <= 0.0 or s_shocked <= 0.0:
            raise ValidationError("spots must be positive")
        self._base = base
        self._ratio = float(s_base) / float(s_shocked)

    def get_vol(self, strike, time_to_maturity, spot=None):
        try:
            return self._base.get_vol(
                float(strike) * self._ratio, time_to_maturity, spot
            )
        except TypeError:
            # base surface without the spot argument (e.g. SVIVolSurface)
            return self._base.get_vol(
                float(strike) * self._ratio, time_to_maturity
            )

    def __getattr__(self, name):
        return getattr(self._base, name)


def shocked_surface(surface, s_base: float, s_shocked: float,
                    convention: GreekConvention):
    if convention is GreekConvention.STICKY_STRIKE:
        return surface
    if convention is GreekConvention.STICKY_MONEYNESS:
        return _StickyMoneynessView(surface, s_base, s_shocked)
    if convention is GreekConvention.STICKY_DELTA:
        raise ValidationError(
            "sticky delta is out of scope in this iteration; it requires a "
            "per-expiry delta-strike solver (the moneyness re-anchor is not "
            "sticky delta) — use STICKY_MONEYNESS or STICKY_STRIKE"
        )
    raise ValidationError(f"unsupported spot-shock convention: {convention!r}")
