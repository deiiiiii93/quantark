"""Sticky spot-shock convention tests (spec WP4.4)."""
import numpy as np
import pytest

from quantark.param.vol.sticky import shocked_surface
from quantark.util.enum.greek_conventions import GreekConvention
from quantark.util.exceptions import ValidationError


class _SkewedSurface:
    """sigma(K) = 0.2 - 0.1 * ln(K / 6000): simple skew stub."""

    def get_vol(self, strike, time_to_maturity, spot=None):
        return 0.2 - 0.1 * float(np.log(strike / 6000.0))


def test_sticky_strike_returns_same_object():
    s = _SkewedSurface()
    assert shocked_surface(s, 6000.0, 6600.0, GreekConvention.STICKY_STRIKE) is s


def test_sticky_moneyness_re_anchors_atm():
    s = _SkewedSurface()
    view = shocked_surface(s, 6000.0, 6600.0, GreekConvention.STICKY_MONEYNESS)
    # ATM vol rides the spot: vol at the NEW spot equals the old ATM vol
    assert view.get_vol(6600.0, 1.0) == pytest.approx(0.2)
    # and a fixed strike now maps to a different point on the base smile
    assert view.get_vol(6000.0, 1.0) != pytest.approx(
        s.get_vol(6000.0, 1.0)
    )


def test_sticky_delta_rejected():
    with pytest.raises(ValidationError):
        shocked_surface(
            _SkewedSurface(), 6000.0, 6600.0, GreekConvention.STICKY_DELTA
        )


def test_conventions_differ_for_skewed_smile_delta():
    # integration: FD delta of a European-style payoff differs by convention
    # for a skewed smile. Use Black price at sigma(K_ref) as the payoff.
    from quantark.param.vol.marketquotes import black_price

    s = _SkewedSurface()
    k, t, df = 6000.0, 1.0, 0.97

    def pv(spot, convention):
        surf = shocked_surface(s, 6000.0, spot, convention)
        sigma = surf.get_vol(k, t)
        return black_price(spot * 1.0, k, t, sigma, df, True)

    h = 60.0
    deltas = {}
    for conv in (GreekConvention.STICKY_STRIKE,
                 GreekConvention.STICKY_MONEYNESS):
        deltas[conv] = (pv(6000.0 + h, conv) - pv(6000.0 - h, conv)) / (2 * h)
    assert abs(
        deltas[GreekConvention.STICKY_STRIKE]
        - deltas[GreekConvention.STICKY_MONEYNESS]
    ) > 1e-4
