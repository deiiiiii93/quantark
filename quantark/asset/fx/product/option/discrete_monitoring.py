"""
Discrete barrier fixings of an FX option, on either clock.

A numeric (``maturity``) product books its fixings as ``observation_times``, year fractions from the valuation date.
A dated (``expiry_date``) product books them as ``observation_dates``, resolved against the pricing environment's
valuation date and day count like ``expiry_date`` is. Both given: the dates win and the floats are ignored
(clock rule 2026-10-26).
"""

from datetime import datetime
from typing import List, Optional, TYPE_CHECKING

from quantark.util.exceptions import ValidationError
from ..base_fx_product import BaseFxProduct

if TYPE_CHECKING:
    from quantark.priceenv import FxPricingEnvironment


class DiscreteMonitoring(BaseFxProduct):
    """An FX product with ``observation_times`` / ``observation_dates`` barrier fixings."""

    observation_times: Optional[List[float]]
    observation_dates: Optional[List[datetime]]

    def _validate_observation_times(self) -> None:
        if self.observation_dates:
            self._validate_dates(list(self.observation_dates))
            return
        times = self.observation_times
        if not times:
            raise ValidationError(
                "DISCRETE monitoring requires a non-empty observation_times (or observation_dates) list"
            )
        self._validate_ordering(list(times), "observation_times")
        T = self.maturity
        for t in times:
            if t <= 0.0:
                raise ValidationError("observation_times must be strictly positive")
            if T is not None and t > T:
                raise ValidationError("observation_times must not exceed maturity")

    def _validate_dates(self, dates: List[datetime]) -> None:
        self._validate_ordering(dates, "observation_dates")
        if self.expiry_date is not None and dates[-1] > self.expiry_date:
            raise ValidationError("observation_dates must not be after expiry_date")

    @staticmethod
    def _validate_ordering(values: list, name: str) -> None:
        if values != sorted(values):
            raise ValidationError(f"{name} must be ascending")
        if len(set(values)) != len(values):
            raise ValidationError(f"{name} must be unique")

    def observation_times_at(self, fx_env: Optional["FxPricingEnvironment"] = None) -> List[float]:
        """The fixing times in years from ``fx_env``'s valuation date.

        A dated product resolves its ``observation_dates`` on the environment's day count; a fixing on or before the
        valuation date is refused, as a numeric time that is not strictly positive is.
        """
        if not self.observation_dates:
            return list(self.observation_times or ())
        if fx_env is not None and self.observation_dates[0] <= fx_env.valuation_date:
            raise ValidationError(
                "observation_dates must fall after the valuation date (observation_times must be strictly positive)"
            )
        return [self._year_fraction_to(d, fx_env) for d in self.observation_dates]
