"""
Black-Scholes-Merton process with continuous dividend yield.
"""

from dataclasses import dataclass
from quantark.util.exceptions import ValidationError


@dataclass
class BSMProcess:
    """
    Black-Scholes-Merton stochastic process for equity pricing.

    The BSM model assumes the underlying asset follows geometric Brownian motion:
        dS/S = (r - q) dt + σ dW

    where:
        S = spot price
        r = risk-free rate
        q = continuous dividend yield
        σ = volatility
        W = Wiener process

    Attributes:
        spot: Current spot price
        volatility: Annualized volatility (σ)
        risk_free_rate: Risk-free interest rate (r)
        dividend_yield: Continuous dividend yield (q)
    """

    spot: float
    volatility: float
    risk_free_rate: float
    dividend_yield: float = 0.0

    def __post_init__(self):
        """Validate process parameters."""
        if self.spot <= 0:
            raise ValidationError(f"Spot must be positive, got {self.spot}")
        if self.volatility <= 0:
            raise ValidationError(f"Volatility must be positive, got {self.volatility}")
        if self.volatility > 5.0:  # 500% vol - sanity check
            raise ValidationError(
                f"Volatility seems unreasonably high: {self.volatility}"
            )
        if abs(self.dividend_yield) > 1.0:  # 100% carry - unit-error sanity check
            raise ValidationError(
                f"Dividend yield magnitude seems unreasonably high: {self.dividend_yield}"
            )
        # Allow negative rates
        if self.risk_free_rate < -0.10 or self.risk_free_rate > 0.50:
            raise ValidationError(
                f"Risk-free rate outside reasonable bounds: {self.risk_free_rate}"
            )

    @property
    def drift(self) -> float:
        """
        Get the risk-neutral drift rate.

        Under risk-neutral measure: μ = r - q

        Returns:
            Drift rate (r - q)
        """
        return self.risk_free_rate - self.dividend_yield

    @property
    def forward_price(self) -> float:
        """
        Calculate the forward price.

        F = S * exp((r - q) * T)

        For T=1 year, returns the 1-year forward.

        Returns:
            Forward price (1 year)
        """
        import math

        return self.spot * math.exp(self.drift)

    def get_forward_price(self, time_to_maturity: float) -> float:
        """
        Calculate forward price for a given maturity.

        F(T) = S * exp((r - q) * T)

        Args:
            time_to_maturity: Time to maturity in years

        Returns:
            Forward price
        """
        import math

        if time_to_maturity < 0:
            raise ValidationError(
                f"Time to maturity must be non-negative, got {time_to_maturity}"
            )
        return self.spot * math.exp(self.drift * time_to_maturity)

    def __repr__(self):
        return (
            f"BSMProcess(S={self.spot:.2f}, σ={self.volatility:.2%}, "
            f"r={self.risk_free_rate:.2%}, q={self.dividend_yield:.2%})"
        )
