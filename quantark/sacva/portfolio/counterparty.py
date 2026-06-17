"""Counterparty — pure data (provider/engine logic lives in the adapter, §3.5)."""

from dataclasses import dataclass
from typing import List, Optional

from quantark.sacva.models.enums import CreditQuality
from quantark.sacva.portfolio.netting import NettingSet
from quantark.util.exceptions import ValidationError


@dataclass
class Counterparty:
    """One counterparty: netting sets + supplied credit curve + SA-CVA bucketing."""

    name: str
    netting_sets: List[NettingSet]
    credit_curve: object  # PD term structure + ELGD (CreditPricingEnvironment-like)
    bucket: int
    credit_quality: CreditQuality
    sub_bucket: Optional[str] = None
    legal_entity_group: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValidationError("Counterparty requires a name")
        if not self.netting_sets:
            raise ValidationError(f"{self.name}: no netting sets")
        if self.credit_curve is None:
            raise ValidationError(f"{self.name}: credit_curve required (PD/ELGD)")
        if isinstance(self.bucket, bool) or not isinstance(self.bucket, int):
            raise ValidationError(f"{self.name}: bucket must be int")
        if not isinstance(self.credit_quality, CreditQuality):
            raise ValidationError(f"{self.name}: credit_quality must be CreditQuality")
