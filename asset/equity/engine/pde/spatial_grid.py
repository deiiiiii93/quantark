"""
Spatial grid utilities for PDE solvers.

Provides methods for generating spatial discretizations in log-price space,
including the Tavella-Randall transformation for concentrating grid points
near critical prices (strikes, barriers, etc.).
"""

import numpy as np
from typing import List, Optional, Tuple
import math


class SpatialGrid:
    """
    Static utility class for generating spatial grids.
    
    All methods work in log-price space (x = ln(S)) for numerical stability.
    The Tavella-Randall transformation allows concentration of grid points
    near critical prices like strikes and barriers.
    """
    
    @staticmethod
    def build_uniform_log(
        s_min: float,
        s_max: float,
        num_points: int
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
        beta: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build a non-uniform grid using Tavella-Randall transformation.
        
        The transformation concentrates grid points near a critical price
        (e.g., strike or barrier) while maintaining smooth transitions.
        
        The transformation is:
            x_i = K + beta * sinh(c1 * (1 - i/N) + c2 * (i/N))
        
        where K is the critical point and beta controls concentration strength.
        
        Args:
            s_min: Minimum spot price (S space)
            s_max: Maximum spot price (S space)
            num_points: Number of grid points
            critical_point: Price to concentrate around (e.g., strike)
            beta: Concentration parameter (auto-calculated if None)
            
        Returns:
            Tuple of (x_vec, s_vec, dx_vec):
                - x_vec: Log-price grid points
                - s_vec: Price grid points
                - dx_vec: Variable grid spacings
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
        # Beta controls concentration: smaller beta = more concentration
        if beta is None:
            # Heuristic: beta proportional to grid range, scaled down
            beta = (x_max - x_min) / 4.0
        
        # Calculate Tavella-Randall transformation parameters
        a1 = (x_min - x_crit) / beta
        a2 = (x_max - x_crit) / beta
        c1 = np.arcsinh(a1)
        c2 = np.arcsinh(a2)
        
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
        beta: Optional[float] = None
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
            beta: Concentration parameter (auto-calculated if None)
            
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
                s_min, s_max, num_points, crits[0], beta
            )
        
        # Build segments: [s_min, c1], [c1, c2], ..., [cn, s_max]
        boundaries = np.concatenate([[s_min], crits, [s_max]])
        n_segments = len(boundaries) - 1
        
        # Allocate points proportionally to segment length (in log space)
        log_boundaries = np.log(boundaries)
        segment_lengths = np.diff(log_boundaries)
        total_length = segment_lengths.sum()
        
        # Points per segment (proportional, minimum 3 per segment)
        points_per_segment = np.maximum(
            3,
            np.round(segment_lengths / total_length * num_points).astype(int)
        )
        
        # Adjust to match total (distribute remainder)
        diff = num_points - points_per_segment.sum()
        for i in range(abs(diff)):
            idx = i % n_segments
            points_per_segment[idx] += np.sign(diff)
        
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
                seg_smin, seg_smax, seg_n, seg_crit, beta
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
        num_std: float = 4.0
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
        beta: Optional[float] = None
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
            beta: Concentration parameter for Tavella-Randall
            
        Returns:
            Tuple of (x_vec, s_vec, dx_vec)
        """
        if use_adaptive and critical_points:
            return SpatialGrid.build_tavella_randall_multi(
                s_min, s_max, num_points, critical_points, beta
            )
        elif use_adaptive and not critical_points:
            # Use midpoint as critical point
            mid = np.sqrt(s_min * s_max)  # Geometric mean
            return SpatialGrid.build_tavella_randall(
                s_min, s_max, num_points, mid, beta
            )
        else:
            x_vec, s_vec, dx = SpatialGrid.build_uniform_log(s_min, s_max, num_points)
            dx_vec = np.full(num_points - 1, dx)
            return x_vec, s_vec, dx_vec

