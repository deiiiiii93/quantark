"""Named long-end extrapolation schemes (spec WP3.5).

Observable market data is often far shorter than contract tenors, so the
long end is an assumption, not a market bucket. Each curve family gets at
least two named schemes, all continuous at ``last_observable_tenor``;
extrapolation-scheme risk is reported as its own line item via
``extrapolation_scheme_risk`` — never folded into market buckets.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from quantark.util.exceptions import ValidationError


class RateExtrapolation(Enum):
    FLAT_FORWARD_RATE = "flat_forward_rate"
    FLAT_ZERO_RATE = "flat_zero_rate"


class CarryExtrapolation(Enum):
    FLAT_FORWARD_CARRY = "flat_forward_carry"
    ZERO_FORWARD_CARRY = "zero_forward_carry"


class VolExtrapolation(Enum):
    FLAT_FORWARD_VOL = "flat_forward_vol"
    FLAT_TOTAL_IMPLIED_VOL = "flat_total_implied_vol"


def extrapolated_zero_rate(curve, T: float, scheme: RateExtrapolation) -> float:
    """Zero rate at T beyond the curve's last pillar, per the named scheme.

    FLAT_ZERO_RATE holds the last zero rate; FLAT_FORWARD_RATE holds the
    last inter-pillar instantaneous forward. Both equal the curve at the
    last pillar (continuity).
    """
    tenors = [float(t) for t in getattr(curve, "tenors", [])]
    if len(tenors) < 2:
        raise ValidationError(
            "extrapolated_zero_rate needs a curve with >= 2 pillars"
        )
    T = float(T)
    t_last, t_prev = tenors[-1], tenors[-2]
    r_last = float(curve.get_rate(t_last))
    if T <= t_last:
        return float(curve.get_rate(T))
    if scheme is RateExtrapolation.FLAT_ZERO_RATE:
        return r_last
    if scheme is RateExtrapolation.FLAT_FORWARD_RATE:
        r_prev = float(curve.get_rate(t_prev))
        fwd = (r_last * t_last - r_prev * t_prev) / (t_last - t_prev)
        return (r_last * t_last + fwd * (T - t_last)) / T
    raise ValidationError(f"unknown rate extrapolation scheme: {scheme!r}")


def extrapolated_total_variance(
    vol_surface,
    ref_strike: float,
    T: float,
    scheme: VolExtrapolation,
    last_observable_tenor: float,
    penultimate_tenor: Optional[float] = None,
) -> float:
    """Total variance V(T) = sigma(T)^2 * T beyond the last observable tenor.

    FLAT_TOTAL_IMPLIED_VOL holds sigma constant (V = sigma_last^2 * T);
    FLAT_FORWARD_VOL holds the last inter-node forward variance
    (V = V_last + fwd_var * (T - T_last)). Both equal V at the last node.
    ``penultimate_tenor`` is required for FLAT_FORWARD_VOL.
    """
    T = float(T)
    t_obs = float(last_observable_tenor)
    sigma_last = float(vol_surface.get_vol(float(ref_strike), t_obs))
    v_last = sigma_last * sigma_last * t_obs
    if T <= t_obs:
        sig = float(vol_surface.get_vol(float(ref_strike), T))
        return sig * sig * T
    if scheme is VolExtrapolation.FLAT_TOTAL_IMPLIED_VOL:
        return sigma_last * sigma_last * T
    if scheme is VolExtrapolation.FLAT_FORWARD_VOL:
        if penultimate_tenor is None:
            raise ValidationError(
                "FLAT_FORWARD_VOL needs penultimate_tenor to form the last "
                "forward variance"
            )
        t_pen = float(penultimate_tenor)
        sig_pen = float(vol_surface.get_vol(float(ref_strike), t_pen))
        v_pen = sig_pen * sig_pen * t_pen
        fwd_var = (v_last - v_pen) / (t_obs - t_pen)
        if fwd_var < 0.0:
            raise ValidationError(
                "negative forward variance at the long end (calendar "
                "arbitrage in the input surface)"
            )
        return v_last + fwd_var * (T - t_obs)
    raise ValidationError(f"unknown vol extrapolation scheme: {scheme!r}")


def extrapolation_scheme_risk(price_fn, default_env, alt_env) -> dict:
    """Scheme risk = PV(alternative) - PV(default), as its own line item."""
    pv_default = float(price_fn(default_env))
    pv_alternative = float(price_fn(alt_env))
    return {
        "pv_default": pv_default,
        "pv_alternative": pv_alternative,
        "scheme_risk": pv_alternative - pv_default,
    }
