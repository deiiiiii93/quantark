"""
Dividend yield representations.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import math

import numpy as np
from quantark.util.exceptions import ValidationError


class DividendYield(ABC):
    """
    Abstract base class for dividend yields.
    
    Dividend yields can be continuous or discrete.
    """
    
    @abstractmethod
    def get_yield(self, time_to_maturity: float) -> float:
        """
        Get dividend yield for given maturity.
        
        Args:
            time_to_maturity: Time to maturity in years
            
        Returns:
            Annualized dividend yield (continuously compounded)
        """
        pass


@dataclass
class ContinuousDividendYield(DividendYield):
    """
    Continuous dividend yield.
    
    Models dividends as a continuous yield, suitable for index options
    or stocks with frequent dividend payments.
    
    Attributes:
        div_yield: Constant annualized dividend yield (continuously compounded)
    """
    div_yield: float
    
    def __post_init__(self):
        """Validate dividend yield (signed carry allowed, symmetric bound)."""
        if not math.isfinite(self.div_yield):
            raise ValidationError(f"Dividend yield must be finite, got {self.div_yield}")
        if abs(self.div_yield) > 0.20:  # symmetric sanity bound
            raise ValidationError(
                f"Dividend yield magnitude seems unreasonably high: {self.div_yield}"
            )
    
    def get_yield(self, time_to_maturity: float) -> float:
        """
        Return constant dividend yield.
        
        Args:
            time_to_maturity: Time to maturity (ignored)
            
        Returns:
            Constant dividend yield
        """
        return self.div_yield
    
    def __repr__(self):
        return f"ContinuousDividendYield(yield={self.div_yield:.2%})"


@dataclass
class TermStructureDividendYield(DividendYield):
    """
    Term-structure dividend yield (by maturity).

    Provides a time-dependent dividend yield via linear interpolation.
    """

    times: list[float]
    yields: list[float]

    def __post_init__(self) -> None:
        if len(self.times) != len(self.yields):
            raise ValidationError("times and yields must have the same length.")
        if len(self.times) < 2:
            raise ValidationError("times must have at least 2 points.")
        if any(t <= 0 for t in self.times):
            raise ValidationError("times must be positive.")
        if any(self.times[i] >= self.times[i + 1] for i in range(len(self.times) - 1)):
            raise ValidationError("times must be strictly increasing.")
        if any(not math.isfinite(y) for y in self.yields):
            raise ValidationError("dividend yields must be finite.")
        if any(abs(y) > 1.0 for y in self.yields):
            raise ValidationError("dividend yield magnitude must be <= 1.0.")

    def get_yield(self, time_to_maturity):
        """Zero carry to maturity; accepts a scalar or an ndarray of times.

        np.interp clips to the endpoint values, which IS the documented flat
        extrapolation — array and scalar paths are exactly equivalent.
        """
        t = np.asarray(time_to_maturity, dtype=float)
        out = np.interp(t, self.times, self.yields)
        if t.ndim == 0:
            return float(out)
        return out

    def __repr__(self):
        return "TermStructureDividendYield(points=%d)" % len(self.times)


@dataclass
class NoDividend(DividendYield):
    """
    Zero dividend yield.
    
    Convenience class for stocks that pay no dividends.
    """
    
    def get_yield(self, time_to_maturity: float) -> float:
        """
        Return zero dividend yield.
        
        Args:
            time_to_maturity: Time to maturity (ignored)
            
        Returns:
            Zero
        """
        return 0.0
    
    def __repr__(self):
        return "NoDividend()"


@dataclass(frozen=True)
class CarryInvariantDividendYield(DividendYield):
    """Dividend yield re-derived so the forward F(0,T) is invariant under a
    discount-curve bump (spec WP3.3 desk convention).

    With cumulative carry B(T) = (r(T) - q(T)) * T held fixed, a rate bump
    Delta_r(T) implies q'(T) = q(T) + Delta_r(T) pointwise, so the drift
    r_fwd - q_fwd (and hence every simulated path) is unchanged and the bump
    is pure discounting.
    """

    base: DividendYield
    base_rate_curve: object      # RateCurve before the bump
    bumped_rate_curve: object    # RateCurve after the bump

    def get_yield(self, time_to_maturity):
        t = time_to_maturity
        base_q = self.base.get_yield(t)
        try:
            return base_q + (
                self.bumped_rate_curve.get_rate(t)
                - self.base_rate_curve.get_rate(t)
            )
        except TypeError:
            # array input: evaluate pointwise (RateCurve API is scalar)
            import numpy as np

            arr = np.asarray(t, dtype=float)
            delta = np.array(
                [
                    self.bumped_rate_curve.get_rate(float(x))
                    - self.base_rate_curve.get_rate(float(x))
                    for x in arr.ravel()
                ]
            ).reshape(arr.shape)
            return base_q + delta
