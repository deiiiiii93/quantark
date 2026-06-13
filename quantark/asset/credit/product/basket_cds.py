"""
Basket Credit Default Swap (FtD / NtD / tranche) product.
"""
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np

from quantark.util.exceptions import ValidationError
from .base_credit_product import BaseCreditProduct
from .cds import ProtectionSide


class BasketType(Enum):
    """Settlement structure of a basket CDS."""

    FTD = "FirstToDefault"
    NTD = "NthToDefault"
    TRANCHE = "Tranche"


class CopulaType(Enum):
    """Dependence structure for correlated default times."""

    GAUSSIAN = "Gaussian"
    STUDENT_T = "StudentT"


@dataclass
class BasketCDS(BaseCreditProduct):
    """
    Basket Credit Default Swap on multiple reference entities.

    Protection is triggered by the first default (FTD), the n-th default
    (NTD), or losses falling within a tranche [attachment, detachment].
    Per-entity hazard curves are supplied at pricing time via a
    :class:`~quantark.priceenv.BasketCreditPricingEnvironment`.

    Attributes:
        notional: Basket notional.
        maturity: Time to maturity in years.
        recovery_rates: Recovery rate per entity, each in [0, 1].
        coupon_spread: Contractual running spread (decimal).
        payment_freq: Premium payments per year.
        basket_type: FTD / NTD / TRANCHE.
        n_to_default: Default rank that triggers an NTD basket.
        attachment_point: Tranche lower loss bound in [0, 1].
        detachment_point: Tranche upper loss bound in [0, 1].
        copula_type: Gaussian or Student-t copula.
        correlation_matrix: Entity correlation matrix (defaults to 0.3 flat).
        student_t_df: Degrees of freedom for the t-copula.
        side: Protection side held.
    """

    notional: float
    maturity: float
    recovery_rates: List[float]
    coupon_spread: float = 0.0
    payment_freq: int = 4
    basket_type: BasketType = BasketType.FTD
    n_to_default: int = 1
    attachment_point: float = 0.0
    detachment_point: float = 1.0
    copula_type: CopulaType = CopulaType.GAUSSIAN
    correlation_matrix: Optional[np.ndarray] = None
    student_t_df: int = 4
    side: ProtectionSide = ProtectionSide.BUY

    def __post_init__(self) -> None:
        n = len(self.recovery_rates)
        if self.correlation_matrix is None:
            m = np.full((n, n), 0.3)
            np.fill_diagonal(m, 1.0)
            self.correlation_matrix = m
        self.validate()

    def validate(self) -> None:
        n = self.n_entities
        if self.notional <= 0:
            raise ValidationError(f"Notional must be positive, got {self.notional}")
        if self.maturity <= 0:
            raise ValidationError(f"Maturity must be positive, got {self.maturity}")
        if n == 0:
            raise ValidationError("At least one reference entity is required")
        if any(not 0.0 <= r <= 1.0 for r in self.recovery_rates):
            raise ValidationError("All recovery rates must be in [0, 1]")
        if self.coupon_spread < 0:
            raise ValidationError("Coupon spread must be non-negative")
        if self.payment_freq <= 0:
            raise ValidationError("Payment frequency must be positive")
        corr = np.asarray(self.correlation_matrix, dtype=float)
        if corr.shape != (n, n):
            raise ValidationError(f"Correlation matrix must be {n}x{n}")
        if not np.allclose(corr, corr.T, atol=1e-8):
            raise ValidationError("Correlation matrix must be symmetric")
        if not np.allclose(np.diag(corr), 1.0, atol=1e-8):
            raise ValidationError("Correlation matrix must have a unit diagonal")
        # Must be positive *definite*: the basket engine draws correlated
        # default times via a Cholesky factorisation, which requires a strictly
        # positive-definite matrix. A merely positive-semidefinite (singular)
        # matrix such as [[1, 1], [1, 1]] would pass a PSD check but break the
        # Cholesky, so a singular or indefinite matrix is rejected here.
        min_eigenvalue = float(np.linalg.eigvalsh(corr).min())
        if min_eigenvalue <= 1e-10:
            raise ValidationError(
                "Correlation matrix must be positive definite for the Cholesky "
                "factorisation used to draw correlated default times "
                f"(smallest eigenvalue {min_eigenvalue:.2e})"
            )
        if self.basket_type == BasketType.NTD and not 1 <= self.n_to_default <= n:
            raise ValidationError(
                f"n_to_default must be in [1, {n}], got {self.n_to_default}"
            )
        if self.basket_type == BasketType.TRANCHE:
            if not 0.0 <= self.attachment_point < self.detachment_point <= 1.0:
                raise ValidationError(
                    "Require 0 <= attachment < detachment <= 1"
                )
        if not isinstance(self.side, ProtectionSide):
            raise ValidationError("side must be a ProtectionSide")

    @property
    def n_entities(self) -> int:
        return len(self.recovery_rates)

    @property
    def side_sign(self) -> float:
        """+1 for protection buyer, -1 for protection seller."""
        return 1.0 if self.side == ProtectionSide.BUY else -1.0
