"""
Term structure of Vanna-Volga FX smiles.

Holds one single-expiry :class:`VannaVolgaVolSurface` per quoted tenor and
serves ``get_vol(strike, t)`` for *any* maturity by interpolating the per-strike
total implied variance ``w(K, t) = sigma(K, t)^2 * t`` across tenors:

    sigma(K, t) = sqrt( interp_t[ sigma(K, tau_i)^2 * tau_i ] / t )

This matches the total-variance convention of ``TermStructureVolSurface`` and
``GridVolSurface``: it reproduces each input smile exactly at its quoted tenor,
interpolates linearly in total variance between tenors, and extrapolates flat in
volatility outside the quoted range. Each ``sigma(K, tau_i)`` is the full
VV-adjusted smile vol from that tenor's slice, so the smile shape (RR/BF skew)
is carried through per tenor.

Unlike a single-tenor surface, the ``time_to_maturity`` argument of ``get_vol``
is *honoured* here — that is the whole point: a multi-fixing product sees the
correct smile at each fixing date.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np

from quantark.param.vol.vol_surface import VolatilitySurface
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_close, safe_divide, safe_sqrt

from .smile_builder import SmileQuotes
from .vv_surface import VannaVolgaVolSurface


@dataclass
class TermStructureVannaVolgaVolSurface(VolatilitySurface):
    """
    Term structure of single-expiry Vanna-Volga smiles.

    Attributes:
        slices: Single-tenor VV smiles in strictly increasing tenor order. All
            slices must share the same spot (a term structure is one market
            snapshot); rates may differ by tenor.
        check_calendar_arbitrage: When True (default), validate at construction
            that the per-strike total variance is non-decreasing across tenors
            (no negative forward variance). Set False to bypass for slightly
            arbitrageable market data.
    """

    # Log-moneyness grid (in std units) on which calendar arbitrage is checked.
    _CAL_ARB_GRID_STD = (-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0)
    # Tolerance on the total-variance non-decrease test (absolute + relative).
    _CAL_ARB_ABS_TOL = 1e-12
    _CAL_ARB_REL_TOL = 1e-7

    slices: List[VannaVolgaVolSurface]
    check_calendar_arbitrage: bool = True
    _taus: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.slices:
            raise ValidationError("at least one VV smile slice is required")
        for s in self.slices:
            if not isinstance(s, VannaVolgaVolSurface):
                raise ValidationError(
                    f"every slice must be a VannaVolgaVolSurface, got "
                    f"{type(s).__name__}"
                )
        taus = [s.env.tau for s in self.slices]
        if any(taus[i] >= taus[i + 1] for i in range(len(taus) - 1)):
            raise ValidationError("slice tenors must be strictly increasing")
        spot0 = self.slices[0].env.spot
        for s in self.slices[1:]:
            if not is_close(s.env.spot, spot0, rel_tol=1e-9):
                raise ValidationError(
                    f"all slices must share the same spot; got {s.env.spot} "
                    f"vs {spot0}"
                )
        self._taus = np.asarray(taus, dtype=float)

        if self.check_calendar_arbitrage and len(self.slices) >= 2:
            self._assert_no_calendar_arbitrage()

    def _assert_no_calendar_arbitrage(self) -> None:
        """Reject negative forward variance between adjacent quoted tenors.

        Under linear total-variance interpolation an interval is arbitrage-free
        iff its endpoints are, so checking the quoted tenors pairwise on a
        log-moneyness strike grid is sufficient. Strikes far from the money cap
        at the ATM vol, so the wings reduce to the ATM-variance comparison.
        """
        spot = self.slices[0].env.spot
        sigma_ref = max(s.quotes.sigma_atm for s in self.slices)
        width = sigma_ref * math.sqrt(float(self._taus[-1]))
        strikes = sorted(
            {spot * math.exp(z * width) for z in self._CAL_ARB_GRID_STD}
            | {s.env.forward for s in self.slices}
        )
        for strike in strikes:
            prev_w = None
            prev_tau = None
            for i, s in enumerate(self.slices):
                tau = float(self._taus[i])
                sigma = s.get_vol(strike, tau, spot)
                w = sigma * sigma * tau
                if prev_w is not None:
                    tol = self._CAL_ARB_ABS_TOL + self._CAL_ARB_REL_TOL * prev_w
                    if w < prev_w - tol:
                        raise ValidationError(
                            f"Calendar arbitrage: total variance decreases from "
                            f"tenor {prev_tau:g} to {tau:g} at strike "
                            f"{strike:.6f} ({prev_w:.6e} -> {w:.6e}). Pass "
                            f"check_calendar_arbitrage=False to bypass."
                        )
                prev_w, prev_tau = w, tau

    @property
    def tenors(self) -> List[float]:
        """Quoted tenors (years), strictly increasing."""
        return [float(t) for t in self._taus]

    def get_vol(self, strike: float, time_to_maturity: float, spot: float) -> float:
        """
        VV-adjusted implied vol at ``strike`` for maturity ``time_to_maturity``.

        Flat-vol extrapolation outside the quoted tenors; linear total-variance
        interpolation inside.
        """
        t = float(time_to_maturity)
        if t <= 0.0:
            raise ValidationError(f"time_to_maturity must be positive, got {t}")
        taus = self._taus

        if t <= taus[0]:
            return self.slices[0].get_vol(strike, taus[0], spot)
        if t >= taus[-1]:
            return self.slices[-1].get_vol(strike, taus[-1], spot)

        # Total variance w_i = sigma(K, tau_i)^2 * tau_i at each tenor, then
        # linear interpolation in t (matching GridVolSurface).
        total_var = np.array(
            [
                self.slices[i].get_vol(strike, taus[i], spot) ** 2 * taus[i]
                for i in range(taus.size)
            ]
        )
        w = float(np.interp(t, taus, total_var))
        return float(safe_sqrt(safe_divide(w, t)))

    def reanchor(
        self,
        spot: float,
        rd_fn: Callable[[float], float],
        rf_fn: Callable[[float], float],
    ) -> "TermStructureVannaVolgaVolSurface":
        """Re-anchor every slice to a new spot and rate environment (sticky-delta).

        ``rd_fn`` / ``rf_fn`` map a tenor to the domestic / foreign rate; each
        slice is rebound at its own tenor. Intrinsic smile quotes are preserved.
        """
        return TermStructureVannaVolgaVolSurface(
            [
                s.rebound(spot=spot, rd=rd_fn(s.env.tau), rf=rf_fn(s.env.tau), tau=s.env.tau)
                for s in self.slices
            ],
            # Derived from an already-validated surface by a small market bump;
            # skip the (costly) re-check, which the transform preserves.
            check_calendar_arbitrage=False,
        )

    def with_scaled_quotes(
        self, factor: float
    ) -> "TermStructureVannaVolgaVolSurface":
        """Scale every slice's ATM/RR/BF quotes by ``factor`` (full-quote vega bump)."""
        scaled = []
        for s in self.slices:
            q = s.quotes
            scaled.append(
                s.with_quotes(
                    SmileQuotes(
                        sigma_atm=q.sigma_atm * factor,
                        rr25=q.rr25 * factor,
                        bf25_2vol=q.bf25_2vol * factor,
                    )
                )
            )
        return TermStructureVannaVolgaVolSurface(
            scaled, check_calendar_arbitrage=False
        )

    def __repr__(self) -> str:
        return (
            f"TermStructureVannaVolgaVolSurface(tenors={self.tenors}, "
            f"atm={[round(s.quotes.sigma_atm, 4) for s in self.slices]})"
        )


__all__ = ["TermStructureVannaVolgaVolSurface"]
