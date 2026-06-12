"""
Multi-target, multi-instrument hedge optimization.

Solves for hedge instrument quantities that move a vector of portfolio
Greeks to their target levels. Given hedging targets (rows) and hedge
instruments (columns), builds the sensitivity matrix A where
A[i, j] = per-unit Greek i of instrument j, and solves:

    A @ x = b,   b[i] = target_i - current_i

Three regimes are handled with exact, documented semantics:

- Square (n_targets == n_instruments): direct linear solve. Raises
  NumericalError if the matrix is singular or ill-conditioned.
- Underdetermined (more instruments than targets): all targets are met
  exactly; among the infinitely many solutions, the one minimizing the
  cost-weighted trade norm sum((c_j * x_j)^2) is returned (minimum-norm
  solution when all costs are equal).
- Overdetermined (more targets than instruments): targets cannot all be
  met exactly; the weighted least-squares solution minimizing
  sum((w_i * (A@x - b)_i)^2) is returned.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from quantark.util.exceptions import NumericalError, ValidationError
from quantark.util.numerical import is_zero


@dataclass(frozen=True)
class HedgeTarget:
    """
    One Greek to control in a multi-target hedging strategy.

    Attributes:
        greek: Greek name as it appears in portfolio Greeks dicts
            (e.g. 'delta', 'gamma', 'vega')
        target: Desired portfolio level for this Greek (default 0 = neutral)
        threshold: Absolute deviation from target that triggers a hedge
        weight: Relative importance in the overdetermined (least-squares)
            regime; ignored when targets can be met exactly
    """

    greek: str
    target: float = 0.0
    threshold: float = 0.0
    weight: float = 1.0

    def __post_init__(self):
        if not self.greek:
            raise ValidationError("HedgeTarget greek name must be non-empty")
        if self.threshold < 0:
            raise ValidationError(
                f"HedgeTarget threshold must be non-negative, got {self.threshold}"
            )
        if self.weight <= 0:
            raise ValidationError(
                f"HedgeTarget weight must be positive, got {self.weight}"
            )


class HedgeOptimizer:
    """
    Solves the multi-target, multi-instrument hedge sizing problem.

    Stateless: each call to solve() is independent.

    Attributes:
        max_condition_number: Conditioning limit above which the linear
            system is treated as numerically singular
    """

    def __init__(self, max_condition_number: float = 1e12):
        """
        Initialize hedge optimizer.

        Args:
            max_condition_number: Conditioning limit for the linear solve
        """
        if max_condition_number <= 0:
            raise ValidationError(
                f"max_condition_number must be positive, got {max_condition_number}"
            )
        self.max_condition_number = max_condition_number

    def solve(
        self,
        portfolio_greeks: Dict[str, float],
        instrument_greeks: Dict[str, Dict[str, float]],
        targets: List[HedgeTarget],
        instrument_costs: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Solve for hedge quantities per instrument.

        Args:
            portfolio_greeks: Current portfolio Greeks
            instrument_greeks: Per-unit Greeks of each tradeable hedge
                instrument, keyed by instrument name
            targets: Greeks to control and their target levels
            instrument_costs: Optional positive cost weight per instrument
                name; used only in the underdetermined regime to prefer
                cheaper instruments. Missing names default to 1.0.

        Returns:
            Dictionary mapping instrument name to quantity to trade
            (positive = buy, negative = sell)

        Raises:
            ValidationError: If inputs are inconsistent
            NumericalError: If the system is singular or ill-conditioned
        """
        if not targets:
            raise ValidationError("At least one HedgeTarget is required")
        if not instrument_greeks:
            raise ValidationError("At least one hedge instrument is required")

        greek_names = [t.greek for t in targets]
        if len(set(greek_names)) != len(greek_names):
            raise ValidationError(f"Duplicate greeks in targets: {greek_names}")

        names = list(instrument_greeks.keys())
        n_targets, n_instruments = len(targets), len(names)

        # Build sensitivity matrix A (targets x instruments) and residual b
        matrix = np.empty((n_targets, n_instruments))
        residual = np.empty(n_targets)
        for i, target in enumerate(targets):
            for j, name in enumerate(names):
                greeks = instrument_greeks[name]
                if target.greek not in greeks:
                    raise ValidationError(
                        f"Instrument '{name}' is missing greek "
                        f"'{target.greek}' required by hedging targets"
                    )
                matrix[i, j] = greeks[target.greek]
            residual[i] = target.target - portfolio_greeks.get(target.greek, 0.0)

        # Already at target on every controlled Greek: no trade needed
        if all(is_zero(r) for r in residual):
            return {name: 0.0 for name in names}

        if n_targets == n_instruments:
            quantities = self._solve_square(matrix, residual)
        elif n_instruments > n_targets:
            quantities = self._solve_underdetermined(
                matrix, residual, names, instrument_costs
            )
        else:
            quantities = self._solve_overdetermined(matrix, residual, targets)

        return dict(zip(names, quantities.tolist()))

    def _solve_square(self, matrix: np.ndarray, residual: np.ndarray) -> np.ndarray:
        """Exact solve for the square system."""
        self._check_conditioning(matrix, "instrument Greeks matrix")
        return np.linalg.solve(matrix, residual)

    def _solve_underdetermined(
        self,
        matrix: np.ndarray,
        residual: np.ndarray,
        names: List[str],
        instrument_costs: Optional[Dict[str, float]],
    ) -> np.ndarray:
        """
        Meet all targets exactly with minimal cost-weighted trade norm.

        Solves min sum((c_j x_j)^2) subject to A x = b, whose closed form
        is x = D^-1 A^T (A D^-1 A^T)^-1 b with D = diag(c^2).
        """
        costs = np.ones(len(names))
        if instrument_costs is not None:
            for j, name in enumerate(names):
                cost = instrument_costs.get(name, 1.0)
                if cost <= 0:
                    raise ValidationError(
                        f"Instrument cost for '{name}' must be positive, got {cost}"
                    )
                costs[j] = cost

        d_inv = 1.0 / costs**2
        gram = matrix @ (d_inv[:, None] * matrix.T)
        self._check_conditioning(gram, "cost-weighted Gram matrix")
        multipliers = np.linalg.solve(gram, residual)
        return d_inv * (matrix.T @ multipliers)

    def _solve_overdetermined(
        self,
        matrix: np.ndarray,
        residual: np.ndarray,
        targets: List[HedgeTarget],
    ) -> np.ndarray:
        """Weighted least-squares fit when targets cannot all be met."""
        weights = np.array([t.weight for t in targets])
        weighted_matrix = weights[:, None] * matrix
        weighted_residual = weights * residual
        solution, _, rank, _ = np.linalg.lstsq(
            weighted_matrix, weighted_residual, rcond=None
        )
        if rank < matrix.shape[1]:
            raise NumericalError(
                "Hedge instrument Greeks are linearly dependent: least-squares "
                f"system has rank {rank} < {matrix.shape[1]} instruments"
            )
        return solution

    def _check_conditioning(self, matrix: np.ndarray, description: str):
        """Raise NumericalError if the matrix is singular or ill-conditioned."""
        condition = np.linalg.cond(matrix)
        if not np.isfinite(condition) or condition > self.max_condition_number:
            raise NumericalError(
                f"Hedge optimization failed: {description} is singular or "
                f"ill-conditioned (condition number {condition:.3e}). "
                "Choose hedge instruments with more distinct Greek profiles "
                "(e.g. different tenors or strikes)."
            )
