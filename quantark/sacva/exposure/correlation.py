"""Gaussian-copula correlation over the simulated risk-factor drivers (spec §3.2).

v1 requires a positive-definite correlation matrix (explicit choice; singular PSD
inputs raise rather than being silently jittered).
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from quantark.util.exceptions import ValidationError


@dataclass
class CorrelationModel:
    keys: List[str]
    matrix: object

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=float)
        n = len(self.keys)
        if self.matrix.shape != (n, n):
            raise ValidationError("correlation matrix shape must match keys")
        if not np.allclose(self.matrix, self.matrix.T):
            raise ValidationError("correlation matrix must be symmetric")

    def cholesky(self) -> np.ndarray:
        try:
            return np.linalg.cholesky(self.matrix)
        except np.linalg.LinAlgError:
            raise ValidationError("correlation matrix must be positive definite")
