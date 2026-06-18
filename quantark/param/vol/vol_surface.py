"""
Volatility surface representations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from quantark.util.exceptions import ValidationError
from quantark.util.numerical import safe_divide, safe_sqrt, validate_positive
import numpy as np


class VolatilitySurface(ABC):
    """
    Abstract base class for volatility surfaces.

    Volatility surfaces provide implied volatility as a function of
    strike and time to maturity.
    """

    @abstractmethod
    def get_vol(self, strike: float, time_to_maturity: float, spot: float) -> float:
        """
        Get implied volatility for given strike and maturity.

        Args:
            strike: Strike price
            time_to_maturity: Time to maturity in years
            spot: Current spot price

        Returns:
            Implied volatility (annualized)
        """
        pass


@dataclass
class FlatVolSurface(VolatilitySurface):
    """
    Flat (constant) volatility surface.

    Returns the same volatility regardless of strike and maturity.
    Suitable for Black-Scholes models with constant volatility.

    Attributes:
        volatility: Constant annualized volatility
    """

    volatility: float

    def __post_init__(self) -> None:
        """Validate volatility."""
        if not isinstance(self.volatility, (int, float)):
            raise ValidationError(f"Volatility must be numeric, got {self.volatility}")
        validate_positive(float(self.volatility), name="volatility")
        if self.volatility > 5.0:  # 500% vol - sanity check
            raise ValidationError(
                f"Volatility seems unreasonably high: {self.volatility}"
            )

    def get_vol(self, strike: float, time_to_maturity: float, spot: float) -> float:
        """
        Return constant volatility.

        Args:
            strike: Strike price (ignored)
            time_to_maturity: Time to maturity (ignored)
            spot: Spot price (ignored)

        Returns:
            Constant volatility
        """
        return self.volatility

    def __repr__(self):
        return f"FlatVolSurface(vol={self.volatility:.2%})"


@dataclass
class TermStructureVolSurface(VolatilitySurface):
    """
    Term-structure volatility surface (ATM by maturity).

    Provides a time-dependent volatility via total variance interpolation on maturity.
    Strike is ignored (ATM term structure).

    Attributes:
        times: Increasing maturities (year fractions).
        vols: Implied volatilities for each maturity.
    """

    times: list[float]
    vols: list[float]

    def __post_init__(self) -> None:
        if len(self.times) != len(self.vols):
            raise ValidationError("times and vols must have the same length.")
        if len(self.times) < 2:
            raise ValidationError("times must have at least 2 points.")
        if any(t <= 0 for t in self.times):
            raise ValidationError("times must be positive.")
        if any(self.times[i] >= self.times[i + 1] for i in range(len(self.times) - 1)):
            raise ValidationError("times must be strictly increasing.")
        for v in self.vols:
            validate_positive(float(v), name="volatility")

    def get_vol(self, strike: float, time_to_maturity: float, spot: float) -> float:
        t = float(time_to_maturity)
        if t <= self.times[0]:
            return float(self.vols[0])
        if t >= self.times[-1]:
            return float(self.vols[-1])
        total_variances = [float(v) ** 2 * float(tt) for v, tt in zip(self.vols, self.times)]
        interp_total_var = float(np.interp(t, self.times, total_variances))
        return float(safe_sqrt(safe_divide(interp_total_var, t)))

    def __repr__(self):
        return "TermStructureVolSurface(points=%d)" % len(self.times)


@dataclass
class GridVolSurface(VolatilitySurface):
    """Market implied-volatility surface on a strike x maturity grid.

    Interpolation: linear in total variance w(T)=sigma^2*T across maturities; linear in
    strike at each maturity (``strike_interpolation`` reserved for future schemes). Flat
    extrapolation (clamp to nearest edge) on both axes. This is MARKET DATA (implied vols),
    the input to the Dupire builder; it is NOT a derived local-vol surface.

    Attributes:
        strikes: strictly increasing strike grid (length nK >= 2).
        maturities: strictly increasing maturity grid in years (length nT >= 2).
        iv_grid: implied vols, shape (nT, nK), axis 0 = maturity, axis 1 = strike.
        strike_interpolation: only "linear" supported in v1.
    """

    strikes: list[float]
    maturities: list[float]
    iv_grid: np.ndarray
    strike_interpolation: str = "linear"

    def __post_init__(self) -> None:
        self.strikes = [float(k) for k in self.strikes]
        self.maturities = [float(t) for t in self.maturities]
        self.iv_grid = np.asarray(self.iv_grid, dtype=float)
        nT, nK = len(self.maturities), len(self.strikes)
        if nK < 2 or nT < 2:
            raise ValidationError("GridVolSurface needs >= 2 strikes and >= 2 maturities")
        if self.iv_grid.shape != (nT, nK):
            raise ValidationError(
                f"iv_grid shape {self.iv_grid.shape} must equal (nT, nK)=({nT}, {nK})"
            )
        if not all(np.isfinite(k) and k > 0 for k in self.strikes):
            raise ValidationError("strikes must be finite and positive")
        if not all(np.isfinite(t) for t in self.maturities):
            raise ValidationError("maturities must be finite")
        if any(self.strikes[i] >= self.strikes[i + 1] for i in range(nK - 1)):
            raise ValidationError("strikes must be strictly increasing")
        if any(self.maturities[i] >= self.maturities[i + 1] for i in range(nT - 1)):
            raise ValidationError("maturities must be strictly increasing")
        if any(t <= 0 for t in self.maturities):
            raise ValidationError("maturities must be positive")
        if not np.all(np.isfinite(self.iv_grid)) or np.any(self.iv_grid <= 0):
            raise ValidationError("iv_grid must be finite and strictly positive")
        if self.strike_interpolation != "linear":
            raise ValidationError("only 'linear' strike_interpolation is supported in v1")

    def get_vol(self, strike: float, time_to_maturity: float, spot: float) -> float:
        k, t = float(strike), float(time_to_maturity)
        if not (np.isfinite(k) and np.isfinite(t)):
            raise ValidationError("strike and time_to_maturity must be finite")
        strikes = np.asarray(self.strikes)
        mats = np.asarray(self.maturities)
        k_clamped = min(max(k, strikes[0]), strikes[-1])
        vols_by_T = np.array(
            [float(np.interp(k_clamped, strikes, self.iv_grid[i])) for i in range(mats.size)]
        )
        if t <= mats[0]:
            return float(vols_by_T[0])
        if t >= mats[-1]:
            return float(vols_by_T[-1])
        w = float(np.interp(t, mats, vols_by_T ** 2 * mats))
        return float(safe_sqrt(safe_divide(w, t)))

    def __repr__(self):
        return f"GridVolSurface(nK={len(self.strikes)}, nT={len(self.maturities)})"
