"""Mathematical helpers for SA-CCR (supervisory duration, maturity factors, delta).

Reference: Basel Committee SA-CCR document, paragraphs 155-164.

Numerical conventions (QuantArk):
- Regulatory inputs are domain-validated *before* any protected-math call, so the
  ``quantark.util.numerical.safe_*`` helpers never silently make an invalid input
  computable. Invalid inputs raise :class:`ValidationError`.
- The supervisory option delta uses the EXACT normal CDF ``scipy.stats.norm.cdf``.
  The original standalone implementation used a hand-rolled Abramowitz-Stegun-style
  approximation that mixed erf coefficients without the ``/sqrt(2)`` argument scaling,
  giving errors up to ~3.7e-2 versus the true normal CDF (e.g. at the Annex 4a
  Example-1 swaption it returned ~0.2324 vs the correct ~0.2694). Switching to SciPy
  is therefore a CORRECTION, not merely a refinement.
"""

from typing import Optional

from scipy.stats import norm

from quantark.saccr.models.enums import Position
from quantark.util.enum.option_enums import OptionType
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import (
    safe_log, safe_exp, safe_sqrt, safe_divide, validate_positive)

#: Minimum time period: 10 business days, assuming 250 business days per year.
MIN_PERIOD = 10 / 250


def supervisory_duration(start_date: float, end_date: float) -> float:
    """Supervisory duration for IR and Credit derivatives (paragraph 157).

    ``SD = (exp(-0.05 * S) - exp(-0.05 * E)) / 0.05``

    Validation order (flooring must not mask invalid raw inputs):
      1. raw ``start_date >= 0`` and raw ``end_date >= 0``;
      2. raw ``end_date >= start_date`` (catches inverted dates BEFORE flooring);
      3. floor ``E = max(end_date, 10/250)``;
      4. require ``E > start_date``.

    Example (Annex 4a, Example 1): 10Y swap S=0, E=10 ->
    ``(1 - exp(-0.5)) / 0.05 = 7.8694``.
    """
    if start_date < 0:
        raise ValidationError(f"start_date must be non-negative, got {start_date}")
    if end_date < 0:
        raise ValidationError(f"end_date must be non-negative, got {end_date}")
    if end_date < start_date:
        raise ValidationError(
            f"end_date ({end_date}) must be >= start_date ({start_date})")

    e = max(end_date, MIN_PERIOD)
    if e <= start_date:
        raise ValidationError(
            f"floored end_date ({e}) must exceed start_date ({start_date})")

    return (float(safe_exp(-0.05 * start_date)) - float(safe_exp(-0.05 * e))) / 0.05


def maturity_factor_unmargined(maturity: float) -> float:
    """Maturity factor for unmargined transactions (paragraph 164).

    ``MF = sqrt(min(max(M, 10/250), 1 year) / 1 year)``
    """
    if maturity is None:
        raise ValidationError("maturity is required for the unmargined maturity factor")
    m = max(maturity, MIN_PERIOD)
    m_capped = min(m, 1.0)
    return float(safe_sqrt(m_capped))


def maturity_factor_margined(mpor_years: float) -> float:
    """Maturity factor for margined transactions (paragraph 164).

    ``MF = 1.5 * sqrt(MPOR / 1 year)``.

    ``mpor_years`` is the margin period of risk expressed in years
    (e.g. ``mpor_days / 250``); it is independent of the trade-maturity floor.
    """
    if mpor_years is None:
        raise ValidationError("mpor_years is required for the margined maturity factor")
    if mpor_years < 0:
        raise ValidationError(f"mpor_years must be non-negative, got {mpor_years}")
    return 1.5 * float(safe_sqrt(mpor_years))


def supervisory_delta(
    position: Position,
    is_option: bool = False,
    option_type: Optional[OptionType] = None,
    underlying_price: Optional[float] = None,
    strike_price: Optional[float] = None,
    exercise_date: Optional[float] = None,
    supervisory_volatility: Optional[float] = None,
    is_cdo_tranche: bool = False,
    attachment_point: Optional[float] = None,
    detachment_point: Optional[float] = None,
) -> float:
    """Supervisory delta adjustment (paragraph 159), in [-1, 1].

    - Linear (non-option, non-CDO): +1 long / -1 short.
    - Option: Black-Scholes-style delta sign using the EXACT normal CDF.
    - CDO tranche: ``+/- 15 / ((1 + 14 A)(1 + 14 D))``.

    Domain inputs are validated before any protected math (raises
    :class:`ValidationError`).
    """
    # CDO tranche
    if is_cdo_tranche:
        if attachment_point is None or detachment_point is None:
            raise ValidationError(
                "CDO tranche requires attachment_point and detachment_point")
        delta_abs = 15.0 / ((1 + 14 * attachment_point) * (1 + 14 * detachment_point))
        return -delta_abs if position == Position.LONG else delta_abs

    # Option
    if is_option:
        if option_type is None:
            raise ValidationError("Options require option_type")
        if (underlying_price is None or strike_price is None
                or exercise_date is None or supervisory_volatility is None):
            raise ValidationError(
                "Options require underlying_price, strike_price, exercise_date, "
                "and supervisory_volatility")
        # All present; validate_positive also rejects NaN/inf and non-positive values.
        p = validate_positive(underlying_price, "underlying_price")
        k = validate_positive(strike_price, "strike_price")
        t = validate_positive(exercise_date, "exercise_date")
        sigma = validate_positive(supervisory_volatility, "supervisory_volatility")

        # d1 = (ln(P/K) + 0.5 * sigma^2 * T) / (sigma * sqrt(T))
        numerator = safe_log(p / k) + 0.5 * sigma ** 2 * t
        denominator = sigma * float(safe_sqrt(t))
        d1 = safe_divide(numerator, denominator)

        if option_type == OptionType.CALL:
            delta_abs = float(norm.cdf(d1))
        else:  # PUT
            delta_abs = -float(norm.cdf(-d1))

        return delta_abs if position == Position.LONG else -delta_abs

    # Linear
    return 1.0 if position == Position.LONG else -1.0


def supervisory_delta_option(
    option_type: OptionType,
    underlying_price: float,
    strike_price: float,
    exercise_date: float,
    supervisory_volatility: float,
    is_bought: bool = True,
) -> float:
    """Convenience wrapper for option supervisory delta (paragraph 159)."""
    position = Position.LONG if is_bought else Position.SHORT
    return supervisory_delta(
        position=position,
        is_option=True,
        option_type=option_type,
        underlying_price=underlying_price,
        strike_price=strike_price,
        exercise_date=exercise_date,
        supervisory_volatility=supervisory_volatility,
    )
