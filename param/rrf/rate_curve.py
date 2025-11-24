"""
Risk-free rate curve representations.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple
import math
from util.exceptions import ValidationError


class RateCurve(ABC):
    """
    Abstract base class for risk-free rate curves.
    
    Rate curves provide risk-free interest rates as a function of maturity.
    """
    
    @abstractmethod
    def get_rate(self, time_to_maturity: float) -> float:
        """
        Get risk-free rate for given maturity.
        
        Args:
            time_to_maturity: Time to maturity in years
            
        Returns:
            Annualized risk-free rate (continuously compounded)
        """
        pass
    
    @abstractmethod
    def get_discount_factor(self, time_to_maturity: float) -> float:
        """
        Get discount factor for given maturity.
        
        Args:
            time_to_maturity: Time to maturity in years
            
        Returns:
            Discount factor exp(-r*T)
        """
        pass
    
    def get_forward_rate(self, t1: float, t2: float) -> float:
        """
        Calculate forward rate between two times.
        
        Args:
            t1: Start time (years)
            t2: End time (years)
            
        Returns:
            Forward rate from t1 to t2
        """
        if t2 <= t1:
            raise ValidationError(f"t2 ({t2}) must be greater than t1 ({t1})")
        
        df1 = self.get_discount_factor(t1)
        df2 = self.get_discount_factor(t2)
        
        return -math.log(df2 / df1) / (t2 - t1)


@dataclass
class FlatRateCurve(RateCurve):
    """
    Flat (constant) rate curve.
    
    Returns the same rate regardless of maturity.
    
    Attributes:
        rate: Constant annualized risk-free rate (continuously compounded)
    """
    rate: float
    
    def __post_init__(self):
        """Validate rate - allow negative rates but warn if extreme."""
        if self.rate < -0.10:  # -10% rate - sanity check
            raise ValidationError(f"Rate seems unreasonably low: {self.rate}")
        if self.rate > 0.50:  # 50% rate - sanity check
            raise ValidationError(f"Rate seems unreasonably high: {self.rate}")
    
    def get_rate(self, time_to_maturity: float) -> float:
        """
        Return constant rate.
        
        Args:
            time_to_maturity: Time to maturity (ignored)
            
        Returns:
            Constant risk-free rate
        """
        return self.rate
    
    def get_discount_factor(self, time_to_maturity: float) -> float:
        """
        Calculate discount factor.
        
        Args:
            time_to_maturity: Time to maturity in years
            
        Returns:
            Discount factor exp(-r*T)
        """
        import math
        if time_to_maturity < 0:
            raise ValidationError(f"Time to maturity must be non-negative, got {time_to_maturity}")
        return math.exp(-self.rate * time_to_maturity)
    
    def __repr__(self):
        return f"FlatRateCurve(rate={self.rate:.2%})"


class InterpolatedRateCurve(RateCurve):
    """
    Base class for interpolated rate curves.
    
    Stores rate pillars (time, rate pairs) and interpolates between them.
    
    Attributes:
        pillars: List of (time, rate) tuples, sorted by time
    """
    
    def __init__(self, pillars: List[Tuple[float, float]]):
        """
        Initialize interpolated rate curve.
        
        Args:
            pillars: List of (time_to_maturity, rate) tuples
            
        Raises:
            ValidationError: If pillars are invalid
        """
        if not pillars:
            raise ValidationError("Must provide at least one pillar")
        
        # Sort pillars by time
        self.pillars = sorted(pillars, key=lambda x: x[0])
        
        # Validate pillars
        for i, (time, rate) in enumerate(self.pillars):
            if time < 0:
                raise ValidationError(f"Pillar {i}: time must be non-negative, got {time}")
            if i > 0 and time == self.pillars[i-1][0]:
                raise ValidationError(f"Duplicate time pillar at {time}")
    
    def _find_bracketing_pillars(self, time: float) -> Tuple[int, int]:
        """
        Find indices of pillars that bracket the given time.
        
        Returns:
            Tuple of (left_index, right_index)
            If time is outside range, returns appropriate boundary indices.
        """
        if time <= self.pillars[0][0]:
            return (0, 0)  # Flat extrapolation on left
        
        if time >= self.pillars[-1][0]:
            return (len(self.pillars)-1, len(self.pillars)-1)  # Flat extrapolation on right
        
        # Binary search for bracketing pillars
        left = 0
        right = len(self.pillars) - 1
        
        while right - left > 1:
            mid = (left + right) // 2
            if self.pillars[mid][0] <= time:
                left = mid
            else:
                right = mid
        
        return (left, right)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(pillars={len(self.pillars)})"


class LinearRateCurve(InterpolatedRateCurve):
    """
    Linear interpolated rate curve.
    
    Performs linear interpolation on rates between pillars.
    Uses flat extrapolation outside the pillar range.
    """
    
    def get_rate(self, time_to_maturity: float) -> float:
        """
        Get rate using linear interpolation.
        
        Args:
            time_to_maturity: Time to maturity in years
            
        Returns:
            Interpolated rate
        """
        if time_to_maturity < 0:
            raise ValidationError(f"Time to maturity must be non-negative, got {time_to_maturity}")
        
        left_idx, right_idx = self._find_bracketing_pillars(time_to_maturity)
        
        # Flat extrapolation
        if left_idx == right_idx:
            return self.pillars[left_idx][1]
        
        # Linear interpolation
        t1, r1 = self.pillars[left_idx]
        t2, r2 = self.pillars[right_idx]
        
        # Linear interpolation formula
        weight = (time_to_maturity - t1) / (t2 - t1)
        rate = r1 + weight * (r2 - r1)
        
        return rate
    
    def get_discount_factor(self, time_to_maturity: float) -> float:
        """
        Get discount factor.
        
        Args:
            time_to_maturity: Time to maturity in years
            
        Returns:
            Discount factor exp(-r*T)
        """
        rate = self.get_rate(time_to_maturity)
        return math.exp(-rate * time_to_maturity)


class LogLinearRateCurve(InterpolatedRateCurve):
    """
    Log-linear interpolated rate curve.
    
    Performs linear interpolation on log(discount factors) between pillars.
    This ensures smooth forward rates and is the market standard for
    discount curve interpolation.
    """
    
    def get_discount_factor(self, time_to_maturity: float) -> float:
        """
        Get discount factor using log-linear interpolation.
        
        Args:
            time_to_maturity: Time to maturity in years
            
        Returns:
            Interpolated discount factor
        """
        if time_to_maturity < 0:
            raise ValidationError(f"Time to maturity must be non-negative, got {time_to_maturity}")
        
        if time_to_maturity == 0:
            return 1.0
        
        left_idx, right_idx = self._find_bracketing_pillars(time_to_maturity)
        
        # Calculate discount factors at pillars
        t1, r1 = self.pillars[left_idx]
        df1 = math.exp(-r1 * t1) if t1 > 0 else 1.0
        
        # Flat extrapolation
        if left_idx == right_idx:
            if t1 == 0:
                return math.exp(-r1 * time_to_maturity)
            # Flat forward rate extrapolation
            return df1 * math.exp(-r1 * (time_to_maturity - t1))
        
        # Log-linear interpolation
        t2, r2 = self.pillars[right_idx]
        df2 = math.exp(-r2 * t2)
        
        # Linear interpolation on log(DF)
        log_df1 = math.log(df1)
        log_df2 = math.log(df2)
        
        weight = (time_to_maturity - t1) / (t2 - t1)
        log_df = log_df1 + weight * (log_df2 - log_df1)
        
        return math.exp(log_df)
    
    def get_rate(self, time_to_maturity: float) -> float:
        """
        Get rate from discount factor.
        
        Args:
            time_to_maturity: Time to maturity in years
            
        Returns:
            Zero rate
        """
        if time_to_maturity <= 0:
            if len(self.pillars) > 0:
                return self.pillars[0][1]
            return 0.0
        
        df = self.get_discount_factor(time_to_maturity)
        return -math.log(df) / time_to_maturity


class CubicSplineRateCurve(InterpolatedRateCurve):
    """
    Cubic spline interpolated rate curve.
    
    Uses natural cubic spline interpolation on rates.
    Provides smooth first and second derivatives.
    """
    
    def __init__(self, pillars: List[Tuple[float, float]]):
        """Initialize and compute spline coefficients."""
        super().__init__(pillars)
        self._compute_spline_coefficients()
    
    def _compute_spline_coefficients(self):
        """
        Compute natural cubic spline coefficients.
        
        For each interval [x_i, x_{i+1}], the spline is:
        S_i(x) = a_i + b_i*(x-x_i) + c_i*(x-x_i)^2 + d_i*(x-x_i)^3
        """
        n = len(self.pillars)
        
        if n < 2:
            # Not enough points for spline, treat as constant
            self.coefficients = [(self.pillars[0][1], 0, 0, 0)]
            return
        
        # Extract times and rates
        times = [p[0] for p in self.pillars]
        rates = [p[1] for p in self.pillars]
        
        # Compute intervals
        h = [times[i+1] - times[i] for i in range(n-1)]
        
        # Set up tridiagonal system for second derivatives
        # Natural spline: second derivative is 0 at endpoints
        alpha = [0] * n
        for i in range(1, n-1):
            alpha[i] = (3.0/h[i]) * (rates[i+1] - rates[i]) - (3.0/h[i-1]) * (rates[i] - rates[i-1])
        
        # Solve tridiagonal system
        l = [1.0] * n
        mu = [0.0] * n
        z = [0.0] * n
        
        for i in range(1, n-1):
            l[i] = 2.0 * (times[i+1] - times[i-1]) - h[i-1] * mu[i-1]
            mu[i] = h[i] / l[i]
            z[i] = (alpha[i] - h[i-1] * z[i-1]) / l[i]
        
        # Back substitution
        c = [0.0] * n
        b = [0.0] * (n-1)
        d = [0.0] * (n-1)
        
        for j in range(n-2, -1, -1):
            c[j] = z[j] - mu[j] * c[j+1]
            b[j] = (rates[j+1] - rates[j]) / h[j] - h[j] * (c[j+1] + 2.0*c[j]) / 3.0
            d[j] = (c[j+1] - c[j]) / (3.0 * h[j])
        
        # Store coefficients (a, b, c, d) for each interval
        self.coefficients = []
        for i in range(n-1):
            self.coefficients.append((rates[i], b[i], c[i], d[i]))
    
    def get_rate(self, time_to_maturity: float) -> float:
        """
        Get rate using cubic spline interpolation.
        
        Args:
            time_to_maturity: Time to maturity in years
            
        Returns:
            Interpolated rate
        """
        if time_to_maturity < 0:
            raise ValidationError(f"Time to maturity must be non-negative, got {time_to_maturity}")
        
        left_idx, right_idx = self._find_bracketing_pillars(time_to_maturity)
        
        # Flat extrapolation
        if left_idx == right_idx:
            return self.pillars[left_idx][1]
        
        # Cubic spline evaluation
        a, b, c, d = self.coefficients[left_idx]
        t0 = self.pillars[left_idx][0]
        dt = time_to_maturity - t0
        
        rate = a + b*dt + c*dt*dt + d*dt*dt*dt
        
        return rate
    
    def get_discount_factor(self, time_to_maturity: float) -> float:
        """
        Get discount factor.
        
        Args:
            time_to_maturity: Time to maturity in years
            
        Returns:
            Discount factor exp(-r*T)
        """
        rate = self.get_rate(time_to_maturity)
        return math.exp(-rate * time_to_maturity)

