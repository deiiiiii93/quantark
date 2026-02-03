import numpy as np
from scipy.linalg import solve_banded
from typing import Tuple, Callable, Optional

class PDESystemState:
    """
    Manages the N-state grid for the PDE solver.
    
    Encapsulates the data for 1 or more surfaces (channels) and provides
    vectorized time-stepping capabilities.
    """
    def __init__(self, num_x: int, num_t: int, num_states: int = 1):
        """
        Initialize the PDE system state.
        
        Args:
            num_x: Number of spatial steps
            num_t: Number of time steps
            num_states: Number of parallel states (surfaces)
        """
        self.num_x = num_x
        self.num_t = num_t
        self.num_states = num_states
        
        # Shape: (num_x, num_t, num_states)
        # Grids are stored with time on the second axis to match BasePDESolver conventions,
        # but expanded to 3D for multi-state support.
        self.grids = np.zeros((num_x, num_t, num_states), dtype=float)
        
    def get_slice(self, t_idx: int) -> np.ndarray:
        """
        Get the grid slice at a specific time index.
        
        Returns:
            ndarray of shape (num_x, num_states)
        """
        return self.grids[:, t_idx, :]
    
    def set_slice(self, t_idx: int, values: np.ndarray):
        """
        Set the grid slice at a specific time index.
        
        Args:
            t_idx: Time index
            values: ndarray of shape (num_x, num_states) or broadcastable
        """
        self.grids[:, t_idx, :] = values

    def solve_step_banded(
        self,
        t_idx_curr: int,
        t_idx_next: int,
        banded_lhs: np.ndarray,
        rhs_coeffs: Tuple[np.ndarray, np.ndarray, np.ndarray],
        boundary_injector: Callable[[np.ndarray, int], None],
        rhs_buffer: Optional[np.ndarray] = None,
    ) -> None:
        """
        Perform one backward time step using vectorized banded solver.
        
        Solves: M2 * V_curr = M1 * V_next + BC
        
        Args:
            t_idx_curr: Current time index (target)
            t_idx_next: Previous time index (source, since backward)
            banded_lhs: LHS matrix M2 in banded format (3, N-2)
            rhs_coeffs: Tuple (lower, main, upper) diagonals of RHS matrix M1
            boundary_injector: Function(rhs, t_idx) to add boundary terms to RHS
        """
        # 1. Get source values (interior points)
        # Shape: (N-2, num_states)
        v_next = self.grids[1:-1, t_idx_next, :]
        
        # 2. Compute RHS explicitly: M1 * v_next
        lower, main, upper = rhs_coeffs
        
        # Vectorized calculation using broadcasting
        # lower: (N-2,), v_next: (N-2, K) -> broadcasts column-wise
        # Equivalent to matrix multiplication for tridiagonal M1
        rhs = rhs_buffer
        if rhs is None:
            rhs = np.empty_like(v_next)
        elif rhs.shape != v_next.shape:
            raise ValueError(
                f"rhs_buffer shape {rhs.shape} does not match v_next shape {v_next.shape}"
            )
        rhs[:] = main[:, None] * v_next
        rhs[1:] += lower[:, None] * v_next[:-1]
        rhs[:-1] += upper[:, None] * v_next[1:]
        
        # 3. Inject boundary conditions
        # boundary_injector modifies rhs in-place
        boundary_injector(rhs, t_idx_curr)
        
        # 4. Solve system
        # solve_banded supports (3, N) matrix and (N, K) RHS
        # result shape: (N-2, K)
        # check_finite=False for speed
        v_curr = solve_banded((1, 1), banded_lhs, rhs, overwrite_b=True, check_finite=False)
        
        # 5. Store result
        self.grids[1:-1, t_idx_curr, :] = v_curr
