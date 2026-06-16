"""CVA sensitivity input record for SA-CVA (MAR50.51, 50.54-50.77).

One contribution to the net sensitivity of a single regulatory risk factor.
Values are decimals (sensitivities in reporting-currency units). Risk-factor
netting is performed by the engine (see design spec §2.0).
"""

from dataclasses import dataclass
from typing import Optional

from quantark.sacva.models.enums import RiskClass, RiskType, CreditQuality
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_valid_number


def _finite(x) -> bool:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return False
    return is_valid_number(x)


# index buckets per risk class (optional treatment, MAR50.50)
_INDEX_BUCKETS = {
    RiskClass.REFERENCE_CREDIT: {16, 17},
    RiskClass.EQUITY: {12, 13},
}

# Counterparty credit spread regulatory tenors (MAR50.65(1)).
_COUNTERPARTY_TENORS = frozenset({0.5, 1.0, 3.0, 5.0, 10.0})


def _ref_expected_quality(bucket: int):
    """Credit quality implied by a reference-credit bucket (None for bucket 15)."""
    if bucket in (1, 2, 3, 4, 5, 6, 7, 16):
        return CreditQuality.IG
    if bucket in (8, 9, 10, 11, 12, 13, 14, 17):
        return CreditQuality.HY_NR
    return None  # bucket 15 (other) has no defined quality


@dataclass(frozen=True)
class CVASensitivity:
    """A single CVA/hedge sensitivity contribution to risk factor ``k``."""

    risk_class: RiskClass
    risk_type: RiskType
    bucket: int
    risk_factor: str
    s_cva: float
    s_hdg: float = 0.0
    currency: Optional[str] = None
    tenor: Optional[float] = None
    is_inflation: bool = False
    name: Optional[str] = None
    legal_entity_group: Optional[str] = None
    credit_quality: Optional[CreditQuality] = None
    sub_bucket: Optional[str] = None
    is_index: bool = False
    index_series: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.risk_class, RiskClass):
            raise ValidationError(f"risk_class must be RiskClass, got {self.risk_class!r}")
        if not isinstance(self.risk_type, RiskType):
            raise ValidationError(f"risk_type must be RiskType, got {self.risk_type!r}")
        for flag in ("is_inflation", "is_index"):
            if not isinstance(getattr(self, flag), bool):
                raise ValidationError(f"{flag} must be a bool, got {getattr(self, flag)!r}")
        for amt in ("s_cva", "s_hdg"):
            if not _finite(getattr(self, amt)):
                raise ValidationError(f"{amt} must be a finite number, got {getattr(self, amt)!r}")
        if isinstance(self.bucket, bool) or not isinstance(self.bucket, int):
            raise ValidationError(f"bucket must be an int, got {self.bucket!r}")
        if self.tenor is not None and not _finite(self.tenor):
            raise ValidationError(f"tenor must be a finite number, got {self.tenor!r}")
        if self.index_series is not None and not self.is_index:
            raise ValidationError("index_series is only valid when is_index=True")

        rc = self.risk_class
        # sub_bucket only applies to counterparty credit bucket 1.
        if self.sub_bucket is not None and not (
                rc == RiskClass.COUNTERPARTY_CREDIT and self.bucket == 1):
            raise ValidationError(
                "sub_bucket is only valid for counterparty credit bucket 1")

        if rc == RiskClass.COUNTERPARTY_CREDIT:
            if self.risk_type == RiskType.VEGA:
                raise ValidationError("Counterparty credit has no vega (MAR50.63)")
            if not self.name:
                raise ValidationError("Counterparty credit requires name")
            if self.tenor is None:
                raise ValidationError("Counterparty credit requires tenor")
            if self.tenor not in _COUNTERPARTY_TENORS:
                raise ValidationError(
                    f"Counterparty tenor must be one of {sorted(_COUNTERPARTY_TENORS)}, "
                    f"got {self.tenor}")
            if not isinstance(self.credit_quality, CreditQuality):
                raise ValidationError("Counterparty credit requires credit_quality")
            if self.bucket == 1 and self.sub_bucket not in ("a", "b"):
                raise ValidationError("Counterparty bucket 1 requires sub_bucket 'a'/'b'")
            if self.bucket == 8 and not self.is_index:
                raise ValidationError("Counterparty bucket 8 is index-only (is_index=True)")
            if self.bucket == 8 and not self.index_series:
                raise ValidationError("Counterparty bucket 8 requires index_series")
            if self.is_index and self.bucket != 8:
                raise ValidationError("Counterparty is_index is only valid in bucket 8")
        elif rc == RiskClass.INTEREST_RATE:
            if not self.currency:
                raise ValidationError("IR requires currency")
        elif rc == RiskClass.FX:
            if not self.currency:
                raise ValidationError("FX requires currency")
        elif rc in (RiskClass.REFERENCE_CREDIT, RiskClass.EQUITY, RiskClass.COMMODITY):
            idx = _INDEX_BUCKETS.get(rc, set())
            if self.bucket in idx and not self.is_index:
                raise ValidationError(f"{rc.name} bucket {self.bucket} requires is_index=True")
            if self.is_index and self.bucket not in idx:
                raise ValidationError(f"{rc.name} bucket {self.bucket} is not an index bucket")
            if rc == RiskClass.REFERENCE_CREDIT and self.credit_quality is not None:
                expected = _ref_expected_quality(self.bucket)
                if expected is not None and self.credit_quality != expected:
                    raise ValidationError(
                        f"reference-credit bucket {self.bucket} is {expected.name}, "
                        f"got credit_quality {self.credit_quality.name}")
