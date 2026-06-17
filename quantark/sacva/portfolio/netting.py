"""Netting set (MAR50.35). v1: uncollateralized only (spec §6)."""

from dataclasses import dataclass
from typing import List

from quantark.sacva.portfolio.trade import CVATrade
from quantark.util.exceptions import ValidationError


@dataclass
class NettingSet:
    """A set of trades; netting is applied only when legally enforceable."""

    set_id: str
    trades: List[CVATrade]
    netting_enforceable: bool = True
    collateralized: bool = False

    def __post_init__(self) -> None:
        if not self.set_id:
            raise ValidationError("NettingSet requires a set_id")
        if not self.trades:
            raise ValidationError(f"{self.set_id}: netting set has no trades")
        if self.collateralized:
            raise ValidationError(
                f"{self.set_id}: collateralized sets are deferred (v1, spec §6)"
            )
