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

        Uses piecewise Tavella-Randall segments between critical points.
        Points are allocated proportionally to segment length.

        Args:
            s_min: Minimum spot price (S space)
            s_max: Maximum spot price (S space)
            num_points: Total number of grid points
            critical_points: List of prices to concentrate around
            beta: Concentration parameter. If None, auto-calculated from eps_crit.
            eps_crit: Target relative spacing near critical points (e.g., 0.003 for 0.3%).
                      Only used when beta is None. Default: 0.003 (0.3%).

        Returns:
            Tuple of (x_vec, s_vec, dx_vec)
        """
        if not critical_points:
            # Fall back to uniform grid if no critical points
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

        # If only one critical point, use single Tavella-Randall
        if len(crits) == 1:
            return SpatialGrid.build_tavella_randall(
                s_min, s_max, num_points, crits[0], beta, eps_crit
            )

        # Build segments: [s_min, c1], [c1, c2], ..., [cn, s_max]
        boundaries = np.concatenate([[s_min], crits, [s_max]])
        n_segments = len(boundaries) - 1

        # Allocate points proportionally to segment length (in log space)
        log_boundaries = np.log(boundaries)
        segment_lengths = np.diff(log_boundaries)
        total_length = segment_lengths.sum()

        min_points_per_segment = 3
        if num_points < min_points_per_segment * n_segments:
            # Not enough points to guarantee stable Tavella-Randall segments.
            # Fall back to uniform log grid to avoid invalid small segments.
            x_vec, s_vec, dx = SpatialGrid.build_uniform_log(s_min, s_max, num_points)
            dx_vec = np.full(num_points - 1, dx)
            return x_vec, s_vec, dx_vec

        # Points per segment (proportional, minimum 3 per segment)
        points_per_segment = np.maximum(
            min_points_per_segment,
            np.round(segment_lengths / total_length * num_points).astype(int),
        )

        # Adjust to match total, without violating the minimum
        diff = int(num_points - points_per_segment.sum())
        if diff > 0:
            # Add points to the longest segments first
            order = np.argsort(-segment_lengths)
            for i in range(diff):
                points_per_segment[order[i % n_segments]] += 1
        elif diff < 0:
            to_remove = -diff
            for _ in range(to_remove):
                eligible = np.where(points_per_segment > min_points_per_segment)[0]
                if eligible.size == 0:
                    break
                idx = int(eligible[np.argmax(points_per_segment[eligible])])
                points_per_segment[idx] -= 1

            # If we couldn't remove enough without breaking the minimum, fall back.
            if points_per_segment.sum() != num_points:
                x_vec, s_vec, dx = SpatialGrid.build_uniform_log(s_min, s_max, num_points)
                dx_vec = np.full(num_points - 1, dx)
                return x_vec, s_vec, dx_vec

        # Build each segment
        all_x = []
        for i in range(n_segments):
            seg_smin = boundaries[i]
            seg_smax = boundaries[i + 1]
            seg_n = points_per_segment[i]

            # Critical point for this segment is the midpoint or boundary
            if i == 0:
                seg_crit = boundaries[i + 1]  # First critical point
            elif i == n_segments - 1:
                seg_crit = boundaries[i]  # Last critical point
            else:
                # Middle segments: use the ending critical point
                seg_crit = boundaries[i + 1]

            # Clamp critical point to be within segment
            seg_crit = max(seg_smin * 1.001, min(seg_smax * 0.999, seg_crit))

            x_seg, _, _ = SpatialGrid.build_tavella_randall(
                seg_smin, seg_smax, seg_n, seg_crit, beta, eps_crit
            )

            # Avoid duplicating boundary points
            if i > 0:
                x_seg = x_seg[1:]

            all_x.append(x_seg)

        x_vec = np.concatenate(all_x)
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
