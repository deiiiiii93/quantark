"""Configuration for credit risk factors used by the credit VaR engines."""

import warnings
from typing import Optional

_SPREAD_DEPRECATION = (
    "CreditRiskFactorConfig.include_spread is deprecated; use 'include_hazard'. "
    "The credit curve-level factor is the hazard intensity (Hazard01); the old "
    "'spread' name was a misnomer."
)


class CreditRiskFactorConfig:
    """
    Which credit risk factors to include in VaR.

    Credit positions carry a hazard-intensity (default-intensity) factor and an
    interest-rate factor per reference entity. The hazard factor is the canonical
    shared curve-level risk factor (corresponding to Hazard01).

    ``include_spread`` is retained as a deprecated alias of ``include_hazard``
    (construction, read and assignment all work, with a ``DeprecationWarning``)
    so existing VaR configurations keep disabling the credit factor as before.
    """

    def __init__(
        self,
        include_hazard: bool = True,
        include_rate: bool = True,
        include_spread: Optional[bool] = None,
    ):
        if include_spread is not None:
            warnings.warn(_SPREAD_DEPRECATION, DeprecationWarning, stacklevel=2)
            include_hazard = include_spread
        self.include_hazard = include_hazard
        self.include_rate = include_rate

    @property
    def include_spread(self) -> bool:
        """Deprecated alias for :attr:`include_hazard`."""
        warnings.warn(_SPREAD_DEPRECATION, DeprecationWarning, stacklevel=2)
        return self.include_hazard

    @include_spread.setter
    def include_spread(self, value: bool) -> None:
        warnings.warn(_SPREAD_DEPRECATION, DeprecationWarning, stacklevel=2)
        self.include_hazard = value

    def __repr__(self) -> str:
        return (
            f"CreditRiskFactorConfig(include_hazard={self.include_hazard}, "
            f"include_rate={self.include_rate})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CreditRiskFactorConfig):
            return NotImplemented
        return (
            self.include_hazard == other.include_hazard
            and self.include_rate == other.include_rate
        )
