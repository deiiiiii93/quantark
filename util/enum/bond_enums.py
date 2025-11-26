"""
Enumerations for bond products.
"""

from enum import Enum


class PaymentFrequency(Enum):
    """
    Payment frequency for coupon bonds.

    Value represents the number of payments per year.
    """

    ANNUAL = 1
    SEMI_ANNUAL = 2
    QUARTERLY = 4
    MONTHLY = 12
    WEEKLY = 52
    DAILY = 365

    @property
    def periods_per_year(self) -> int:
        """Get number of payment periods per year."""
        return self.value

    @property
    def months_between_payments(self) -> int:
        """Get approximate months between payments."""
        return 12 // self.value if self.value <= 12 else 0

    def __repr__(self):
        return f"PaymentFrequency.{self.name}"


class StubType(Enum):
    """
    Type of stub period in a bond schedule.

    - SHORT_FRONT: Short first period (< regular period)
    - SHORT_BACK: Short last period (< regular period)
    - LONG_FRONT: Long first period (> regular period)
    - LONG_BACK: Long last period (> regular period)
    - NONE: No stub, all regular periods
    """

    SHORT_FRONT = "short_front"
    SHORT_BACK = "short_back"
    LONG_FRONT = "long_front"
    LONG_BACK = "long_back"
    NONE = "none"


class CompoundingType(Enum):
    """
    Compounding convention for interest rates.

    - CONTINUOUS: Continuous compounding (e^rt)
    - ANNUAL: Annual compounding
    - SEMI_ANNUAL: Semi-annual compounding
    - QUARTERLY: Quarterly compounding
    - MONTHLY: Monthly compounding
    - SIMPLE: Simple interest (no compounding)
    """

    CONTINUOUS = "continuous"
    ANNUAL = "annual"
    SEMI_ANNUAL = "semi_annual"
    QUARTERLY = "quarterly"
    MONTHLY = "monthly"
    SIMPLE = "simple"


class BondType(Enum):
    """
    Type of bond instrument.
    """

    FIXED_RATE = "fixed_rate"
    FLOATING_RATE = "floating_rate"
    ZERO_COUPON = "zero_coupon"
    CONVERTIBLE = "convertible"
    CALLABLE = "callable"
    PUTABLE = "putable"


class BondDerivativeType(Enum):
    """
    Type of bond derivative instrument.

    - FORWARD: OTC forward contract on a bond
    - FUTURES: Exchange-traded bond futures contract
    - OPTION: Option on a bond or bond futures
    - REPO: Repurchase agreement
    """

    FORWARD = "forward"
    FUTURES = "futures"
    OPTION = "option"
    REPO = "repo"
