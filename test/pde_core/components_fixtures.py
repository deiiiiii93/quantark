"""Component stand-ins for the slot tests (spec 2 §4.1; Task 13 extends them)."""
from dataclasses import dataclass

from quantark.asset.equity.engine.pde.core.components import CentralDifference


@dataclass(frozen=True)
class ShiftedOperatorStub(CentralDifference):
    """A CentralDifference that is not the built-in instance: one float field."""

    shift: float = 0.0
