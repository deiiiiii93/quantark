"""SVI implied-vol surface (spec WP4.2).

Slices are raw-SVI fits in forward log-moneyness; between expiries the
surface interpolates LINEARLY IN TOTAL VARIANCE at fixed y; before the
first expiry total variance scales proportionally; beyond the last expiry
the configured vol extrapolation scheme applies (spec WP3.5). Wings follow
the fitted SVI slopes by construction.

Forward-curve context (normative): the surface stores its own forward
provider FROZEN at build time, so ``get_vol(strike, T)`` converts
``y = log(K / F(0,T))`` internally. Spot/rate/carry bumps do NOT move the
stored forwards; re-anchoring under spot shocks is exclusively the job of
the sticky-convention wrappers (the bare surface is sticky-strike).

Dupire interop: exposes ``strikes`` / ``maturities`` / ``iv_grid`` sampled
on a configurable grid, so ``build_dupire_local_vol`` consumes it with no
change to the builder.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import numpy as np

from quantark.param.extrapolation import VolExtrapolation
from quantark.util.exceptions import NumericalError, ValidationError

from .svi_fit import SVISliceFit

CALENDAR_TOL = -1e-8          # min allowed w(y,T2)-w(y,T1) (spec WP4.2)
_W_FLOOR = 1e-12
_CHECK_Y = np.arange(-1.5, 1.5 + 1e-12, 0.01)
_DEFAULT_N_STRIKES = 31
_DEFAULT_N_MATURITIES = 9


class SVIVolSurface:
    def __init__(
        self,
        slice_fits: Sequence[SVISliceFit],
        forward_provider: Callable[[float], float],
        last_observable_tenor: Optional[float] = None,
        vol_extrapolation: VolExtrapolation = VolExtrapolation.FLAT_FORWARD_VOL,
        sample_strikes: Optional[Sequence[float]] = None,
        sample_maturities: Optional[Sequence[float]] = None,
    ):
        if not slice_fits:
            raise ValidationError("SVIVolSurface needs at least one slice fit")
        fits = sorted(slice_fits, key=lambda f: f.expiry_t)
        ts = [f.expiry_t for f in fits]
        if any(t2 <= t1 for t1, t2 in zip(ts, ts[1:])):
            raise ValidationError("slice expiries must be strictly increasing")
        self._fits: List[SVISliceFit] = list(fits)
        self._ts = np.asarray(ts, dtype=float)
        self._forward = forward_provider
        self.last_observable_tenor = (
            float(last_observable_tenor)
            if last_observable_tenor is not None
            else float(self._ts[-1])
        )
        self.vol_extrapolation = vol_extrapolation
        self._sample_strikes = sample_strikes
        self._sample_maturities = sample_maturities
        self._calendar_check()

    # -- construction ------------------------------------------------------
    @classmethod
    def fit_from_quotes(cls, cleaned, carry_curve, spot,
                        **kwargs) -> "SVIVolSurface":
        """Fit one raw-SVI slice per cleaned expiry (spec WP4.2). The
        discount curve is not needed here: cleaning already priced D/F into
        the (y, iv) coordinates."""
        from .svi_fit import fit_svi_slice

        fits = []
        for expiry_t in sorted(cleaned.slices):
            quotes = cleaned.slices[expiry_t]
            y = np.array([q.log_moneyness for q in quotes])
            w = np.array([q.iv * q.iv * expiry_t for q in quotes])
            weights = np.array([q.weight for q in quotes])
            fits.append(fit_svi_slice(y, w, weights, expiry_t))
        spot = float(spot)

        def forward(t: float) -> float:
            return float(carry_curve.forward(spot, t))

        return cls(fits, forward_provider=forward, **kwargs)

    # -- core --------------------------------------------------------------
    def _slice_w(self, i: int, y) -> np.ndarray:
        return np.asarray(self._fits[i].params.total_variance(y), dtype=float)

    def total_variance(self, y, T: float) -> np.ndarray:
        T = float(T)
        if T <= 0.0:
            raise ValidationError(f"T must be positive, got {T}")
        ts = self._ts
        if T <= ts[0]:
            return self._slice_w(0, y) * (T / ts[0])
        if T >= ts[-1]:
            w_last = self._slice_w(len(ts) - 1, y)
            if T == ts[-1]:
                return w_last
            if self.vol_extrapolation is VolExtrapolation.FLAT_TOTAL_IMPLIED_VOL:
                return w_last * (T / ts[-1])
            if self.vol_extrapolation is VolExtrapolation.FLAT_FORWARD_VOL:
                if len(ts) < 2:
                    return w_last * (T / ts[-1])  # single slice: proportional
                w_prev = self._slice_w(len(ts) - 2, y)
                fwd_var = (w_last - w_prev) / (ts[-1] - ts[-2])
                return w_last + np.maximum(fwd_var, 0.0) * (T - ts[-1])
            raise ValidationError(
                f"unknown vol extrapolation: {self.vol_extrapolation!r}"
            )
        i = int(np.searchsorted(ts, T, side="right")) - 1
        w_lo, w_hi = self._slice_w(i, y), self._slice_w(i + 1, y)
        lam = (T - ts[i]) / (ts[i + 1] - ts[i])
        return w_lo + lam * (w_hi - w_lo)

    def get_vol(self, strike: float, time_to_maturity: float,
                spot: Optional[float] = None) -> float:
        """Black implied vol at (K, T); ``spot`` accepted for interface
        compatibility and deliberately ignored (frozen forwards)."""
        T = float(time_to_maturity)
        f = float(self._forward(T))
        y = float(np.log(float(strike) / f))
        w = float(self.total_variance(y, T))
        if w < CALENDAR_TOL:
            raise NumericalError(
                f"negative total variance {w:.3e} at (K={strike}, T={T})"
            )
        return float(np.sqrt(max(w, _W_FLOOR) / T))

    def calendar_check(self) -> dict:
        """Min Δw per adjacent slice pair on the dense y grid."""
        out = {}
        for i in range(len(self._ts) - 1):
            dw = self._slice_w(i + 1, _CHECK_Y) - self._slice_w(i, _CHECK_Y)
            out[(float(self._ts[i]), float(self._ts[i + 1]))] = float(dw.min())
        return out

    def _calendar_check(self) -> None:
        for pair, min_dw in self.calendar_check().items():
            if min_dw < CALENDAR_TOL:
                raise NumericalError(
                    f"calendar arbitrage between T={pair[0]:g} and "
                    f"T={pair[1]:g}: min dw = {min_dw:.3e}"
                )

    # -- Dupire / GridVolSurface interop ------------------------------------
    @property
    def maturities(self) -> List[float]:
        if self._sample_maturities is not None:
            return [float(t) for t in self._sample_maturities]
        return list(
            np.linspace(
                float(self._ts[0]), float(self._ts[-1]), _DEFAULT_N_MATURITIES
            )
        )

    @property
    def strikes(self) -> List[float]:
        if self._sample_strikes is not None:
            return [float(k) for k in self._sample_strikes]
        f_last = float(self._forward(float(self._ts[-1])))
        y = np.linspace(-1.0, 1.0, _DEFAULT_N_STRIKES)
        return list(f_last * np.exp(y))

    @property
    def iv_grid(self) -> np.ndarray:
        ks, ts = self.strikes, self.maturities
        return np.array([[self.get_vol(k, t) for k in ks] for t in ts])

    def slice_fits(self) -> List[SVISliceFit]:
        return list(self._fits)

    def to_dict(self) -> dict:
        return {
            "slices": [f.to_dict() for f in self._fits],
            "last_observable_tenor": self.last_observable_tenor,
            "vol_extrapolation": self.vol_extrapolation.value,
            "calendar_min_dw": {
                f"{a:g}->{b:g}": v
                for (a, b), v in self.calendar_check().items()
            },
        }
