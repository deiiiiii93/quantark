"""
Time grid utilities for PDE solvers.

Provides various methods for generating temporal discretizations
suitable for different option types and observation schedules.
"""

import numpy as np
from typing import Tuple, List, Optional


class TimeGrid:
    """
    Static utility class for generating time discretizations.
    
    The time grid is defined backward from maturity (tau) to valuation date (0).
    All methods return:
        - t_vec: Time points from 0 to tau
        - dt_vec: Time step sizes (length = len(t_vec) - 1)
    """
    
    @staticmethod
    def build_uniform(tau: float, num_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build a uniform time grid with constant step size.
        
        Args:
            tau: Total time to maturity in years
            num_steps: Number of time steps
            
        Returns:
            Tuple of (t_vec, dt_vec):
                - t_vec: Time points from 0 to tau, shape (num_steps + 1,)
                - dt_vec: Constant time steps, shape (num_steps,)
        """
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}")
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        
        t_vec = np.linspace(0.0, tau, num_steps + 1)
        dt = tau / num_steps
        dt_vec = np.full(num_steps, dt)
        
        return t_vec, dt_vec
    
    @staticmethod
    def build_graded(
        tau: float, 
        num_steps: int, 
        exponent: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build a graded time grid with power-law clustering near maturity.
        
        Uses transformation: t_i = tau * (i / N)^p
        
        Higher exponent means more steps clustered near maturity,
        which is useful for capturing payoff discontinuities.
        
        Args:
            tau: Total time to maturity in years
            num_steps: Number of time steps
            exponent: Power-law exponent (p > 1 clusters near maturity)
            
        Returns:
            Tuple of (t_vec, dt_vec)
        """
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}")
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if exponent <= 0:
            raise ValueError(f"exponent must be positive, got {exponent}")
        
        # Generate graded points: more clustering near maturity (tau)
        # We use (i/N)^p which clusters near t=0, then flip to cluster near tau
        normalized = np.linspace(0.0, 1.0, num_steps + 1)
        # Cluster near tau by using 1 - (1 - x)^p
        t_vec = tau * (1.0 - (1.0 - normalized) ** exponent)
        
        # Calculate step sizes
        dt_vec = np.diff(t_vec)
        
        return t_vec, dt_vec
    
    @staticmethod
    def build_event_clustered(
        tau: float,
        event_times: List[float],
        steps_per_interval: int = 10,
        min_steps_total: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build a time grid that exactly includes specified event times.
        
        This is essential for discretely-observed barrier options, autocallables,
        and other path-dependent products with specific observation dates.
        
        Steps are allocated proportionally between events, with a minimum
        number of steps per interval.
        
        Args:
            tau: Total time to maturity in years
            event_times: List of event times (in years) to include exactly
            steps_per_interval: Minimum steps between consecutive events
            min_steps_total: Minimum total number of steps
            
        Returns:
            Tuple of (t_vec, dt_vec)
        """
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}")
        if steps_per_interval <= 0:
            raise ValueError(f"steps_per_interval must be positive, got {steps_per_interval}")
        
        # Clean and sort event times
        events = np.array([e for e in event_times if 0 < e < tau])
        events = np.unique(events)
        events = np.sort(events)
        
        # Build intervals: [0, e1], [e1, e2], ..., [en, tau]
        boundaries = np.concatenate([[0.0], events, [tau]])
        
        # Allocate steps proportionally to interval length, with minimum per interval
        interval_lengths = np.diff(boundaries)
        total_length = tau
        
        # Calculate steps per interval (proportional + minimum)
        num_intervals = len(interval_lengths)
        target_total_steps = max(min_steps_total, num_intervals * steps_per_interval)
        
        steps_per = np.maximum(
            steps_per_interval,
            np.round(interval_lengths / total_length * target_total_steps).astype(int)
        )
        
        # Ensure at least 1 step per interval
        steps_per = np.maximum(steps_per, 1)
        
        # Build the full grid
        all_points = [np.array([0.0])]  # Start with t=0
        
        for i, (start, end, n_steps) in enumerate(zip(
            boundaries[:-1], boundaries[1:], steps_per
        )):
            # Linspace excluding the start (already included) but including end
            interval_points = np.linspace(start, end, n_steps + 1)[1:]
            all_points.append(interval_points)
        
        t_vec = np.concatenate(all_points)
        dt_vec = np.diff(t_vec)
        
        return t_vec, dt_vec
    
    @staticmethod
    def build(
        tau: float,
        num_steps: int,
        method: str = "uniform",
        event_times: Optional[List[float]] = None,
        grade_exponent: float = 2.0,
        steps_per_interval: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build a time grid using the specified method.
        
        This is a convenience method that dispatches to the appropriate
        grid-building method based on the 'method' parameter.
        
        Args:
            tau: Total time to maturity in years
            num_steps: Number of time steps (used for uniform and graded)
            method: Grid generation method - "uniform", "graded", or "event_clustered"
            event_times: Event times for event_clustered method
            grade_exponent: Exponent for graded method
            steps_per_interval: Steps per interval for event_clustered method
            
        Returns:
            Tuple of (t_vec, dt_vec)
        """
        if method == "uniform":
            return TimeGrid.build_uniform(tau, num_steps)
        elif method == "graded":
            return TimeGrid.build_graded(tau, num_steps, grade_exponent)
        elif method == "event_clustered":
            if event_times is None or len(event_times) == 0:
                # Fall back to uniform if no events specified
                return TimeGrid.build_uniform(tau, num_steps)
            return TimeGrid.build_event_clustered(
                tau, event_times, steps_per_interval, num_steps
            )
        else:
            raise ValueError(
                f"Unknown time grid method: {method}. "
                f"Must be 'uniform', 'graded', or 'event_clustered'"
            )

