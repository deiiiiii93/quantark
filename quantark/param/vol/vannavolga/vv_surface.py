"""
Vanna-Volga FX volatility surface.

Adapts the Vanna-Volga construction to the quant-ark ``VolatilitySurface``
interface. The surface is a *single-expiry* FX smile keyed to an ``FXEnv``
market snapshot and ATM/RR/BF quotes. ``get_vol`` returns the VV-adjusted
implied volatility at a strike by computing the VV-adjusted vanilla price and
inverting Black-Scholes/Garman-Kohlhagen.

Like ``FlatVolSurface`` / ``TermStructureVolSurface``, the ``spot`` and (here)
``time_to_maturity`` arguments of ``get_vol`` are advisory: the smile is a fixed
market object evaluated at its stored ``FXEnv``. The free variable is the strike.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from quantark.param.vol.vol_surface import VolatilitySurface
from quantark.util.exceptions import NumericalError, ValidationError

from .bs_fx import GKInput, greeks_gk, price_gk
from .market_conventions import DeltaConvention, FXEnv
from .smile_builder import SmileQuotes
from .vv_core import compute_omega, vv_adjustment_matrix

# Bracket for Black-Scholes implied-vol inversion.
_VOL_LOW = 1e-4
_VOL_HIGH = 5.0
# Convergence tolerance on the vol bracket width (in vol units).
_VOL_INVERSION_TOL = 1e-12
# Relative-to-spot BS vega below which the implied-vol inversion is ill-posed
# (extreme moneyness, vol-insensitive price) and the VV smile is capped at ATM.
_DEGENERATE_VEGA_FLOOR = 1e-8


@dataclass
class VannaVolgaVolSurface(VolatilitySurface):
    """
    Single-expiry Vanna-Volga FX smile.

    Attributes:
        env: FX market snapshot (spot, domestic/foreign rates, tau).
        quotes: ATM vol, 25d risk-reversal, 25d butterfly (2-vol).
        conv: Delta convention used to locate 25d strikes.
        premium_included_atm: Whether the ATM strike is premium-included.
    """

    env: FXEnv
    quotes: SmileQuotes
    conv: DeltaConvention = DeltaConvention.SPOT
    premium_included_atm: bool = False
    _omega: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.quotes.sigma_atm <= 0.0:
            raise ValidationError("sigma_atm must be positive")
        if self.env.tau <= 0.0:
            raise ValidationError("tau must be positive")
        self.conv = DeltaConvention(self.conv)
        omega, _ = compute_omega(
            self.env, self.quotes, self.conv, premium_included_atm=self.premium_included_atm
        )
        self._omega = omega

    def rebound(
        self, spot: float, rd: float, rf: float, tau: float
    ) -> "VannaVolgaVolSurface":
        """Return a new surface re-anchored to a different market snapshot.

        The intrinsic smile data (quotes, delta convention, premium flag) is
        preserved; only the FXEnv anchor changes. Immutable: the original
        surface is left untouched. This is how the risk chain re-anchors the
        smile under spot/rate bumps (sticky-delta).
        """
        return VannaVolgaVolSurface(
            env=FXEnv(spot=spot, rd=rd, rf=rf, tau=tau),
            quotes=self.quotes,
            conv=self.conv,
            premium_included_atm=self.premium_included_atm,
        )

    def with_quotes(self, quotes: SmileQuotes) -> "VannaVolgaVolSurface":
        """Return a new surface with shifted ATM/RR/BF quotes (vega bump).

        The market anchor is preserved; only the three quotes change. This is
        the full-quote vega path used by the chain's vol-bump helpers.
        """
        return VannaVolgaVolSurface(
            env=self.env,
            quotes=quotes,
            conv=self.conv,
            premium_included_atm=self.premium_included_atm,
        )

    def _vv_price(self, strike: float) -> Tuple[float, float]:
        """VV-adjusted call price at the strike and the BS vega (at ATM vol)."""
        sigma_atm = self.quotes.sigma_atm
        g = greeks_gk(
            True, GKInput(self.env.spot, strike, self.env.rd, self.env.rf, sigma_atm, self.env.tau)
        )
        adjustment = vv_adjustment_matrix(g["vega"], g["vanna"], g["volga"], self._omega)
        base = price_gk(
            True, GKInput(self.env.spot, strike, self.env.rd, self.env.rf, sigma_atm, self.env.tau)
        )
        return base + adjustment, g["vega"]

    def _implied_vol(self, strike: float, target_price: float) -> float:
        """Invert the GK call price to an implied vol via bisection.

        Convergence is on the volatility bracket width, not on an absolute price
        residual: for deep OTM/ITM strikes the option price itself can be far
        below machine epsilon, so an absolute-price stop would accept almost any
        small vol. The price is monotonic in vol, so bracketing the sign change
        and shrinking the vol interval recovers the true implied vol.
        """

        def call_price(vol: float) -> float:
            return price_gk(
                True, GKInput(self.env.spot, strike, self.env.rd, self.env.rf, vol, self.env.tau)
            )

        lo, hi = _VOL_LOW, _VOL_HIGH
        f_lo = call_price(lo) - target_price
        f_hi = call_price(hi) - target_price
        if np.sign(f_lo) == np.sign(f_hi):
            raise NumericalError(
                f"VV implied vol for strike {strike} not bracketed in "
                f"[{_VOL_LOW}, {_VOL_HIGH}] (target price {target_price})"
            )
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            f_mid = call_price(mid) - target_price
            if np.sign(f_mid) == np.sign(f_lo):
                lo, f_lo = mid, f_mid
            else:
                hi, f_hi = mid, f_mid
            if (hi - lo) < _VOL_INVERSION_TOL:
                break
        return 0.5 * (lo + hi)

    def get_vol(self, strike: float, time_to_maturity: float, spot: float) -> float:
        """
        VV-adjusted implied volatility at the strike.

        Args:
            strike: Strike price (the smile's free variable).
            time_to_maturity: Advisory (smile is single-expiry; uses stored tau).
            spot: Advisory (uses the stored FXEnv spot).

        Returns:
            Black-Scholes/GK implied volatility (annualized).
        """
        strike = float(strike)
        if strike <= 0.0:
            raise ValidationError(f"strike must be positive, got {strike}")
        target, vega = self._vv_price(strike)
        # Implied-vol inversion is only well-posed where the price is sensitive to
        # vol, i.e. vega is non-negligible. At extreme moneyness vega -> 0 (deep
        # OTM prices underflow; deep ITM prices sit at intrinsic), the inversion
        # is ill-conditioned, and the VV smile is known to break down. There we
        # cap at the ATM vol (exact for a flat smile, whose correction is zero).
        if vega < self.env.spot * _DEGENERATE_VEGA_FLOOR:
            return self.quotes.sigma_atm
        return float(self._implied_vol(strike, target))

    def __repr__(self) -> str:
        return (
            f"VannaVolgaVolSurface(atm={self.quotes.sigma_atm:.4f}, "
            f"rr25={self.quotes.rr25:.4f}, bf25={self.quotes.bf25_2vol:.4f})"
        )


__all__ = ["VannaVolgaVolSurface"]
