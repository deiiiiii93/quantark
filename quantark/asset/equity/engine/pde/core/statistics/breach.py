"""What an event writes where it fires, per column (spec 3B §5.1 "Events").

The value passes ``breach=None``: each stage writes its scalar cash as before. A statistics group
passes a ``Breach``: the stage asks it for a target shaped like the block it transforms.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

__all__ = ["Breach"]


class Breach(Protocol):
    def ko(self, record, block: np.ndarray, surface: str) -> np.ndarray:
        """The block's values where ``record``'s knock-out fires (its unit in the record's column,
        knock-in-ever its own value, everything else as the family's columns say)."""

    def seed(self, block: np.ndarray) -> np.ndarray:
        """The not-knocked-in block at a staged KO-reset's pre-maturity index."""
