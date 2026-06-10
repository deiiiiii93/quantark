"""
Spatial grid utilities for PDE solvers.

Provides methods for generating spatial discretizations in log-price space,
including the Tavella-Randall transformation for concentrating grid points
near critical prices (strikes, barriers, etc.).

The beta parameter in Tavella-Randall controls grid concentration:
- Smaller beta = more concentration near critical point
- Larger beta = more uniform grid

Beta is automatically calculated to achieve a target local spacing (epsilon)
near the critical point using bisection. Default epsilon values:
- Pricing only: 0.5% (eps_crit=0.005)
- Pricing + Greeks: 0.2-0.3% (eps_crit=0.002-0.003)
"""

import numpy as np
from typing import List, Optional, Tuple


class SpatialGrid:
    """
    Static utility class for generating spatial grids.

    All methods work in log-price space (x = ln(S)) for numerical stability.
    The Tavella-Randall transformation allows concentration of grid points
    near critical prices like strikes and barriers.

    The transformation is:
        x(w) = x_crit + beta * sinh(c1*(1-w) + c2*w),  w in [0, 1]

    where:
        c1 = arcsinh((x_min - x_crit) / beta)
        c2 = arcsinh((x_max - x_crit) / beta)

    The local spacing near the critical point is approximately:
        dx_crit ≈ beta * (c2 - c1) / N

    Beta is chosen to achieve a target relative spacing epsilon:
        dx_crit ≈ ln(1 + epsilon) ≈ epsilon
    """

    # Default target relative spacing near critical point
    # 0.3% gives good pricing + Greeks accuracy
    DEFAULT_EPSILON_CRIT = 0.003

    # Maximum allowed ratio of max/min grid spacing (tail coarseness guard)
    DEFAULT_MAX_DX_RATIO = 100.0

    # Bisection parameters for beta calculation
    _BISECTION_ITERS = 80
    _BETA_LO = 1e-12

    @staticmethod
    def _beta_for_target_dx(
        x_min: float,
        x_max: float,
        x_crit: float,
        num_points: int,
        dx_target: float,
        beta_lo: float = _BETA_LO,
        beta_hi: Optional[float] = None,
        iters: int = _BISECTION_ITERS,
    ) -> float:
        """
        Solve for beta to achieve a target local spacing near the critical point.

        Uses bisection to find beta such that:
            dx_crit = beta * (c2 - c1) / N ≈ dx_target

        where N = num_points - 1, and c1, c2 are the Tavella-Randall parameters.

        Args:
            x_min: Minimum log-price
            x_max: Maximum log-price
            x_crit: Critical log-price (concentration point)
            num_points: Number of grid points
            dx_target: Target local spacing near critical point (in log-space)
            beta_lo: Lower bound for bisection (default: 1e-12)
            beta_hi: Upper bound for bisection (default: auto-calculated)
            iters: Maximum bisection iterations (default: 80)

        Returns:
            Beta value that achieves (approximately) the target spacing
        """
        N = num_points - 1
        if beta_hi is None:
            # Large enough that grid is near-uniform
            beta_hi = 1e3 * (x_max - x_min)

        def dxcrit(beta: float) -> float:
            """Compute local spacing at critical point for given beta."""
            c1 = np.arcsinh((x_min - x_crit) / beta)
            c2 = np.arcsinh((x_max - x_crit) / beta)
            return beta * (c2 - c1) / N

        # Check bracket validity
        f_lo = dxcrit(beta_lo) - dx_target
        f_hi = dxcrit(beta_hi) - dx_target

        if f_lo > 0:
            # dx_target is smaller than achievable with maximum concentration
            return beta_lo
        if f_hi < 0:
            # dx_target is larger than uniform spacing; use near-uniform
            return beta_hi

        # Bisection search
        lo, hi = beta_lo, beta_hi
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            f_mid = dxcrit(mid) - dx_target
            if f_mid >= 0:
                hi = mid
            else:
                lo = mid

        return 0.5 * (lo + hi)

    @staticmethod
    def _compute_dx_at_crit(
        x_min: float, x_max: float, x_crit: float, num_points: int, beta: float
    ) -> float:
        """
        Compute the local grid spacing at the critical point.

        This is useful for verifying that the grid achieves the target spacing.

        Args:
            x_min: Minimum log-price
            x_max: Maximum log-price
            x_crit: Critical log-price
            num_points: Number of grid points
            beta: Concentration parameter

        Returns:
            Local spacing dx at the critical point
        """
        N = num_points - 1
        c1 = np.arcsinh((x_min - x_crit) / beta)
        c2 = np.arcsinh((x_max - x_crit) / beta)
        return beta * (c2 - c1) / N

    @staticmethod
    def check_grid_quality(
        dx_vec: np.ndarray, max_ratio: float = DEFAULT_MAX_DX_RATIO
    ) -> Tuple[bool, float]:
        """
        Check grid quality by examining the tail coarseness ratio.

        A very high ratio of max/min spacing indicates extreme concentration
        that may cause numerical issues in the tails.

        Args:
            dx_vec: Array of grid spacings
            max_ratio: Maximum acceptable ratio of max/min spacing

        Returns:
            Tuple of (is_acceptable, actual_ratio)
        """
        if len(dx_vec) == 0:
            return True, 1.0

        dx_min = np.min(dx_vec)
        dx_max = np.max(dx_vec)

        if dx_min <= 0:
            return False, np.inf

        ratio = dx_max / dx_min
        return ratio <= max_ratio, ratio

    @staticmethod
    def build_uniform_log(
        s_min: float, s_max: float, num_points: int
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Build a uniform grid in log-price space.

        Args:
            s_min: Minimum spot price (S space)
            s_max: Maximum spot price (S space)
            num_points: Number of grid points

        Returns:
            Tuple of (x_vec, s_vec, dx):
                - x_vec: Log-price grid points, shape (num_points,)
                - s_vec: Price grid points (exp(x_vec)), shape (num_points,)
                - dx: Uniform grid spacing in log-space
        """
        if s_min <= 0:
            raise ValueError(f"s_min must be positive, got {s_min}")
        if s_max <= s_min:
            raise ValueError(f"s_max ({s_max}) must be greater than s_min ({s_min})")
        if num_points < 3:
            raise ValueError(f"num_points must be at least 3, got {num_points}")

        x_min = np.log(s_min)
        x_max = np.log(s_max)

        x_vec = np.linspace(x_min, x_max, num_points)
        s_vec = np.exp(x_vec)
        dx = (x_max - x_min) / (num_points - 1)

        return x_vec, s_vec, dx

    @staticmethod
    def build_tavella_randall(
        s_min: float,
        s_max: float,
        num_points: int,
        critical_point: float,
        beta: Optional[float] = None,
        eps_crit: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build a non-uniform grid using Tavella-Randall transformation.

        The transformation concentrates grid points near a critical price
        (e.g., strike or barrier) while maintaining smooth transitions.

        The transformation is:
            x(w) = x_crit + beta * sinh(c1*(1-w) + c2*w),  w = i/N

        where:
            c1 = arcsinh((x_min - x_crit) / beta)
            c2 = arcsinh((x_max - x_crit) / beta)

        Beta controls concentration strength (smaller = more concentrated).
        When beta is None, it is automatically calculated to achieve a target
        relative spacing eps_crit near the critical point.

        Args:
            s_min: Minimum spot price (S space)
            s_max: Maximum spot price (S space)
            num_points: Number of grid points
            critical_point: Price to concentrate around (e.g., strike)
            beta: Concentration parameter. If None, auto-calculated from eps_crit.
            eps_crit: Target relative spacing near critical point (e.g., 0.003 for 0.3%).
                      Only used when beta is None. Default: 0.003 (0.3%).

        Returns:
            Tuple of (x_vec, s_vec, dx_vec):
                - x_vec: Log-price grid points
                - s_vec: Price grid points
                - dx_vec: Variable grid spacings

        Notes:
            The local spacing near the critical point is approximately:
                dx_crit ≈ beta * (c2 - c1) / N

            In spot space, this corresponds to:
                dS/S ≈ dx_crit ≈ eps_crit

            Typical values for eps_crit:
                - Pricing only: 0.005 (0.5%)
                - Pricing + Greeks: 0.002-0.003 (0.2-0.3%)
        """
        if s_min <= 0:
            raise ValueError(f"s_min must be positive, got {s_min}")
        if s_max <= s_min:
            raise ValueError(f"s_max ({s_max}) must be greater than s_min ({s_min})")
        if num_points < 3:
            raise ValueError(f"num_points must be at least 3, got {num_points}")
        if critical_point < s_min or critical_point > s_max:
            raise ValueError(
                f"critical_point ({critical_point}) must be in [{s_min}, {s_max}]"
            )

        # Convert to log space
        x_min = np.log(s_min)
        x_max = np.log(s_max)
        x_crit = np.log(critical_point)

        # Auto-calculate beta if not provided
        # Beta is solved via bisection to achieve target local spacing
        if beta is None:
            if eps_crit is None:
                eps_crit = SpatialGrid.DEFAULT_EPSILON_CRIT

            # Target spacing in log-space: ln(1 + eps) ≈ eps for small eps
            dx_target = np.log1p(eps_crit)

            beta = SpatialGrid._beta_for_target_dx(
                x_min, x_max, x_crit, num_points, dx_target
            )

        # Calculate Tavella-Randall transformation parameters
        c1 = np.arcsinh((x_min - x_crit) / beta)
        c2 = np.arcsinh((x_max - x_crit) / beta)

        # Generate grid
        N = num_points - 1
        i_vec = np.arange(num_points)

        # Transformation: x_i = x_crit + beta * sinh(c1 * (1 - i/N) + c2 * (i/N))
        weight = i_vec / N
        x_vec = x_crit + beta * np.sinh(c1 * (1.0 - weight) + c2 * weight)

        # Convert back to price space
        s_vec = np.exp(x_vec)

        # Calculate variable grid spacings
        dx_vec = np.diff(x_vec)

        return x_vec, s_vec, dx_vec

    # ============================================================
    # ODE-based Tavella-Randall helper methods for multiple critical points
    # ============================================================

    @staticmethod
    def _ode_f(y: float, A: float, beta: float, crits: np.ndarray) -> float:
        """
        Right-hand side of the ODE for multi-critical-point grid generation.

        ODE: dY/de = A * (sum_k J_k^-2)^(-0.5)
        where J_k = sqrt(beta^2 + (Y - B_k)^2)

        Args:
            y: Current position in log-space
            A: Scaling constant (found via shooting)
            beta: Concentration parameter
            crits: Array of critical points in log-space

        Returns:
            dY/de value at current position
        """
        j_sq = beta * beta + (y - crits) ** 2
        s = np.sum(1.0 / j_sq)
        return A / np.sqrt(s)

    @staticmethod
    def _ode_rk4_step(
        y: float, h: float, A: float, beta: float, crits: np.ndarray
    ) -> float:
        """
        Single RK4 (Runge-Kutta 4th order) integration step.

        Args:
            y: Current position
            h: Step size in parameter space e
            A: Scaling constant
            beta: Concentration parameter
            crits: Critical points array

        Returns:
            New position after one RK4 step
        """
        f = SpatialGrid._ode_f
        k1 = f(y, A, beta, crits)
        k2 = f(y + 0.5 * h * k1, A, beta, crits)
        k3 = f(y + 0.5 * h * k2, A, beta, crits)
        k4 = f(y + h * k3, A, beta, crits)
        return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    @staticmethod
    def _ode_integrate(
        y_min: float, N: int, A: float, beta: float, crits: np.ndarray
    ) -> np.ndarray:
        """
        Integrate ODE from e=0 to e=1 with N steps.

        Args:
            y_min: Starting position (Y(0) = y_min)
            N: Number of steps (generates N+1 points)
            A: Scaling constant
            beta: Concentration parameter
            crits: Critical points array

        Returns:
            Array of N+1 grid points
        """
        h = 1.0 / N
        mesh = np.empty(N + 1, dtype=float)
        mesh[0] = y_min
        y = y_min
        for i in range(1, N + 1):
            y = SpatialGrid._ode_rk4_step(y, h, A, beta, crits)
            mesh[i] = y
        return mesh

    @staticmethod
    def _ode_find_A(
        y_min: float, y_max: float, N: int, beta: float, crits: np.ndarray
    ) -> float:
        """
        Find scaling constant A via shooting method (bisection).

        Solves for A such that Y(1) = y_max given Y(0) = y_min.

        Args:
            y_min: Lower boundary
            y_max: Upper boundary
            N: Number of grid intervals
            beta: Concentration parameter
            crits: Critical points array

        Returns:
            Scaling constant A
        """
        a_lo = 0.0
        a_hi = max(4.0 * abs(y_max), abs(y_max - y_min))

        def residual(A: float) -> float:
            mesh = SpatialGrid._ode_integrate(y_min, N, A, beta, crits)
            return mesh[-1] - y_max

        f_lo = residual(a_lo)
        f_hi = residual(a_hi)

        # Expand bracket if needed
        for _ in range(20):
            if f_lo * f_hi <= 0:
                break
            a_hi *= 2.0
            f_hi = residual(a_hi)

        # Bisection
        tol = 1e-10
        for _ in range(100):
            if abs(a_hi - a_lo) <= tol:
                break
            a_mid = 0.5 * (a_lo + a_hi)
            f_mid = residual(a_mid)
            if f_mid == 0.0:
                return a_mid
            if f_lo * f_mid < 0:
                a_hi, f_hi = a_mid, f_mid
            else:
                a_lo, f_lo = a_mid, f_mid

        return 0.5 * (a_lo + a_hi)

    @staticmethod
    def _calculate_beta_for_multi_crit(
        x_min: float,
        x_max: float,
        x_crits: np.ndarray,
        num_points: int,
        eps_crit: float,
        use_heuristic_beta: bool = False,
    ) -> float:
        """
        Find beta that achieves target spacing at critical points.

        Uses bisection to find beta such that the minimum spacing
        near any critical point is approximately eps_crit.

        Optimization: Uses a coarse grid for the expensive search,
        scaling the target spacing accordingly.
        """
        if use_heuristic_beta:
            M = len(x_crits)
            # Heuristic: beta ~ ln(1+eps) * sqrt(M/12)
            # Adjusts concentration based on number of observation dates
            beta = np.log1p(eps_crit) * np.sqrt(max(M, 1) / 12.0)

            # Check grid quality and adjust if necessary
            # Ensure spacing ratio is acceptable (avoid tail coarseness)
            N_check = num_points - 1
            max_check_iters = 5
            
            for _ in range(max_check_iters):
                # Generate grid to check spacing ratio
                A = SpatialGrid._ode_find_A(x_min, x_max, N_check, beta, x_crits)
                mesh = SpatialGrid._ode_integrate(x_min, N_check, A, beta, x_crits)
                dx = np.diff(mesh)
                
                is_acceptable, _ = SpatialGrid.check_grid_quality(dx, max_ratio=100.0)
                if is_acceptable:
                    break
                
                # If ratio too high, increase beta (smoother grid)
                beta *= 1.2
            
            return beta

        dx_target = np.log1p(eps_crit)
        N = num_points - 1

        # Use coarse grid for beta search to improve performance
        # Beta is a shape parameter, so we can estimate it on a coarser mesh
        N_coarse = min(N, 64)
        dx_target_adj = dx_target * (N / N_coarse)

        # Bracket for beta (log-scale search)
        beta_lo = 1e-6 * (x_max - x_min)
        beta_hi = 10.0 * (x_max - x_min)

        def min_spacing_near_crits(beta: float) -> float:
            A = SpatialGrid._ode_find_A(x_min, x_max, N_coarse, beta, x_crits)
            mesh = SpatialGrid._ode_integrate(x_min, N_coarse, A, beta, x_crits)
            dx = np.diff(mesh)

            # Find minimum spacing near any critical point
            min_dx = float("inf")
            for xc in x_crits:
                idx = np.searchsorted(mesh, xc)
                idx = max(0, min(idx, len(dx) - 1))
                if idx > 0:
                    min_dx = min(min_dx, dx[idx - 1])
                if idx < len(dx):
                    min_dx = min(min_dx, dx[idx])
            return min_dx

        # Bisection with geometric mean (log-scale)
        # Reduced iterations (20) as precise beta is not critical for grid quality
        for _ in range(20):
            beta_mid = np.sqrt(beta_lo * beta_hi)
            current_dx = min_spacing_near_crits(beta_mid)
            if current_dx > dx_target_adj:
                beta_hi = beta_mid  # Need tighter concentration
            else:
                beta_lo = beta_mid  # Loosen concentration

        return np.sqrt(beta_lo * beta_hi)

    @staticmethod
    def _snap_critical_points(x_vec: np.ndarray, x_crits: np.ndarray) -> np.ndarray:
        """
        Snap nearest grid points to exact critical values.

        Ensures critical points are exactly included in the grid,
        which is important for barrier boundary conditions.

        Args:
            x_vec: Grid points array
            x_crits: Critical points to snap to

        Returns:
            Grid with critical points exactly included
        """
        x_vec = x_vec.copy()
        for xc in x_crits:
            # Find nearest grid index
            idx = np.argmin(np.abs(x_vec - xc))
            # Snap to exact critical value
            x_vec[idx] = xc
        # Re-sort to maintain monotonicity (shouldn't change much)
        x_vec = np.sort(x_vec)
        return x_vec

    @staticmethod
    def build_tavella_randall_multi(
        s_min: float,
        s_max: float,
        num_points: int,
        critical_points: List[float],
        beta: Optional[float] = None,
        eps_crit: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build a non-uniform grid with concentration near multiple critical points.

        Uses the ODE-based Tavella-Randall method:
            dY/de = A * (sum_k J_k^-2)^(-0.5)
        where J_k = sqrt(beta^2 + (Y - B_k)^2), solved via RK4 with shooting.

        This method provides C-infinity smooth grids that naturally concentrate
        around all critical points simultaneously.

        Args:
            s_min: Minimum spot price (S space)
            s_max: Maximum spot price (S space)
            num_points: Total number of grid points
            critical_points: List of prices to concentrate around
            beta: Concentration parameter. If None, auto-calculated from eps_crit.
            eps_crit: Target relative spacing near critical points (e.g., 0.003 for 0.3%).
                      Only used when beta is None. Default: 0.003 (0.3%).

        Returns:
            Tuple of (x_vec, s_vec, dx_vec):
                - x_vec: Log-price grid points
                - s_vec: Price grid points
                - dx_vec: Variable grid spacings

        Notes:
            Critical points are exactly included in the final grid via post-processing,
            which is important for barrier boundary conditions.
        """
        # Validation
        if s_min <= 0:
            raise ValueError(f"s_min must be positive, got {s_min}")
        if s_max <= s_min:
            raise ValueError(f"s_max ({s_max}) must be greater than s_min ({s_min})")
        if num_points < 3:
            raise ValueError(f"num_points must be at least 3, got {num_points}")

        # Handle empty critical points -> uniform grid
        if not critical_points:
            x_vec, s_vec, dx = SpatialGrid.build_uniform_log(s_min, s_max, num_points)
            dx_vec = np.full(num_points - 1, dx)
            return x_vec, s_vec, dx_vec

        # Filter and sort critical points within bounds
        crits = np.array([c for c in critical_points if s_min < c < s_max])
        if len(crits) == 0:
            # No valid critical points, use uniform grid
            x_vec, s_vec, dx = SpatialGrid.build_uniform_log(s_min, s_max, num_points)
            dx_vec = np.full(num_points - 1, dx)
            return x_vec, s_vec, dx_vec

        crits = np.sort(np.unique(crits))

        # If only one critical point, use single Tavella-Randall (optimization)
        if len(crits) == 1:
            return SpatialGrid.build_tavella_randall(
                s_min, s_max, num_points, crits[0], beta, eps_crit
            )

        # Convert to log-space
        x_min = np.log(s_min)
        x_max = np.log(s_max)
        x_crits = np.log(crits)
        N = num_points - 1

        # Auto-calculate beta if not provided
        if beta is None:
            if eps_crit is None:
                eps_crit = SpatialGrid.DEFAULT_EPSILON_CRIT
            beta = SpatialGrid._calculate_beta_for_multi_crit(
                x_min, x_max, x_crits, num_points, eps_crit
            )

        # Safeguard against extreme concentration
        beta = max(beta, 1e-10 * (x_max - x_min))

        # Find scaling constant A via shooting method
        A = SpatialGrid._ode_find_A(x_min, x_max, N, beta, x_crits)

        # Generate mesh via RK4 integration
        x_vec = SpatialGrid._ode_integrate(x_min, N, A, beta, x_crits)

        # Snap critical points to grid (ensure exact inclusion)
        x_vec = SpatialGrid._snap_critical_points(x_vec, x_crits)

        # Convert to price space and compute spacings
        s_vec = np.exp(x_vec)
        dx_vec = np.diff(x_vec)

        return x_vec, s_vec, dx_vec

    @staticmethod
    def calculate_auto_bounds(
        spot: float,
        sigma: float,
        tau: float,
        r: float = 0.0,
        q: float = 0.0,
        strike: Optional[float] = None,
        barriers: Optional[List[float]] = None,
        num_std: float = 4.0,
    ) -> Tuple[float, float]:
        """
        Automatically calculate appropriate grid bounds.

        Uses volatility-based expansion to ensure the grid captures
        the relevant price range for the option.

        Args:
            spot: Current spot price
            sigma: Volatility
            tau: Time to maturity in years
            r: Risk-free rate (default: 0)
            q: Dividend yield (default: 0)
            strike: Strike price (optional, ensures it's in bounds)
            barriers: Barrier prices (optional, ensures they're in bounds)
            num_std: Number of standard deviations for bounds (default: 4)

        Returns:
            Tuple of (s_min, s_max)
        """
        if spot <= 0:
            raise ValueError(f"spot must be positive, got {spot}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}")

        # Calculate expected drift and volatility range
        drift = (r - q) * tau
        vol_range = num_std * sigma * np.sqrt(tau)

        # Calculate bounds
        s_min = spot * np.exp(drift - vol_range)
        s_max = spot * np.exp(drift + vol_range)

        # Ensure strike is within bounds
        if strike is not None and strike > 0:
            s_min = min(s_min, strike * 0.5)
            s_max = max(s_max, strike * 2.0)

        # Ensure barriers are within bounds
        if barriers is not None:
            for b in barriers:
                if b > 0:
                    s_min = min(s_min, b * 0.8)
                    s_max = max(s_max, b * 1.2)

        # Sanity bounds
        s_min = max(s_min, spot * 0.01)  # At least 1% of spot
        s_max = min(s_max, spot * 100.0)  # At most 100x spot

        return s_min, s_max

    @staticmethod
    def build(
        s_min: float,
        s_max: float,
        num_points: int,
        critical_points: Optional[List[float]] = None,
        use_adaptive: bool = False,
        beta: Optional[float] = None,
        eps_crit: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build a spatial grid using the appropriate method.

        This is a convenience method that chooses between uniform and
        Tavella-Randall based on the parameters.

        Args:
            s_min: Minimum spot price
            s_max: Maximum spot price
            num_points: Number of grid points
            critical_points: Points to concentrate around (optional)
            use_adaptive: Whether to use Tavella-Randall transformation
            beta: Concentration parameter for Tavella-Randall. If None, auto-calculated.
            eps_crit: Target relative spacing near critical points (e.g., 0.003 for 0.3%).
                      Only used when beta is None and use_adaptive is True.

        Returns:
            Tuple of (x_vec, s_vec, dx_vec)
        """
        if use_adaptive and critical_points:
            return SpatialGrid.build_tavella_randall_multi(
                s_min, s_max, num_points, critical_points, beta, eps_crit
            )
        elif use_adaptive and not critical_points:
            # Use midpoint as critical point
            mid = np.sqrt(s_min * s_max)  # Geometric mean
            return SpatialGrid.build_tavella_randall(
                s_min, s_max, num_points, mid, beta, eps_crit
            )
        else:
            x_vec, s_vec, dx = SpatialGrid.build_uniform_log(s_min, s_max, num_points)
            dx_vec = np.full(num_points - 1, dx)
            return x_vec, s_vec, dx_vec
