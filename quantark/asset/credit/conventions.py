"""
Market conventions for translating between CDS quoted spreads and hazard
intensities.

Under the credit triangle ``s = lambda (1 - R)`` a quoted-spread move maps to a
hazard-intensity move only once a recovery rate is fixed. The hazard curve is
shared across positions that may carry different contractual recoveries, so a
*curve-level* spread shock is undefined unless a recovery convention is supplied
explicitly; this module provides the ISDA-standard default and the conversion.
"""
from quantark.util.exceptions import ValidationError

#: ISDA-standard senior-unsecured recovery, used as the default convention when
#: a curve-level spread shock must be converted to a hazard shift and no
#: instrument-specific recovery is supplied.
STANDARD_RECOVERY = 0.40


def spread_shift_to_hazard_shift(
    spread_shift: float, recovery: float = STANDARD_RECOVERY
) -> float:
    """Absolute spread move -> hazard move: ``Δlambda = Δs / (1 - R)``."""
    lgd = 1.0 - recovery
    if lgd <= 1e-6:
        raise ValidationError(
            "Spread-to-hazard conversion is undefined at full recovery (R=1): "
            "the spread s = lambda (1 - R) is identically zero."
        )
    return spread_shift / lgd
