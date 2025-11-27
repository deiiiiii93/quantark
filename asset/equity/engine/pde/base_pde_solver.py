"""
Base PDE solver providing common infrastructure for finite difference pricing.

Implements the Crank-Nicolson scheme for solving the Black-Scholes PDE
backward in time, with support for Rannacher smoothing.
"""

from abc import abstractmethod
from typing import Dict, Optional, Tuple, List
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from asset.equity.engine.base_engine import BaseEngine
from asset.equity.product.base_equity_product import BaseEquityProduct
from asset.equity.param import PDEParams
from priceenv import PricingEnvironment
from util.exceptions import PricingError, NumericalError

from .time_grid import TimeGrid
from .spatial_grid import SpatialGrid


class BasePDESolver(BaseEngine):
    """
    Abstract base class for PDE-based option pricing.
    
    Solves the Black-Scholes PDE using finite difference methods:
        dV/dt + (r-q)S*dV/dS + 0.5*sigma^2*S^2*d2V/dS^2 - r*V = 0
    
    In log-price space (x = ln(S)):
        dV/dt + (r-q-0.5*sigma^2)*dV/dx + 0.5*sigma^2*d2V/dx^2 - r*V = 0
    
    The PDE is solved backward in time from maturity to valuation date.
    
    Subclasses must implement:
        - set_terminal_condition(): Define payoff at maturity
        - set_boundary_conditions(): Define behavior at spatial boundaries
        - get_critical_points(): Return prices for grid concentration
    """
    
    def __init__(self, params: Optional[PDEParams] = None):
        """
        Initialize the PDE solver.
        
        Args:
            params: PDE engine configuration parameters
        """
        super().__init__(params if params is not None else PDEParams())
        self._matrix_cache: Dict[Tuple[float, float], Tuple] = {}
    
    @abstractmethod
    def set_terminal_condition(
        self,
        grid: np.ndarray,
        x_vec: np.ndarray,
        s_vec: np.ndarray,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment
    ) -> None:
        """
        Set the terminal condition (payoff at maturity).
        
        Modifies grid[:, -1] in place.
        
        Args:
            grid: Solution grid [num_x, num_t]
            x_vec: Log-price grid points
            s_vec: Price grid points
            product: The option product
            pricing_env: Pricing environment
        """
        pass
    
    @abstractmethod
    def set_boundary_conditions(
        self,
        grid: np.ndarray,
        x_vec: np.ndarray,
        s_vec: np.ndarray,
        t_idx: int,
        tau: float,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment
    ) -> None:
        """
        Set boundary conditions at the spatial edges.
        
        Modifies grid[0, t_idx] and grid[-1, t_idx] in place.
        
        Args:
            grid: Solution grid [num_x, num_t]
            x_vec: Log-price grid points
            s_vec: Price grid points
            t_idx: Current time index
            tau: Time to maturity for this time step
            product: The option product
            pricing_env: Pricing environment
        """
        pass
    
    def get_critical_points(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment
    ) -> List[float]:
        """
        Get critical prices for grid concentration.
        
        Override this method to specify strikes, barriers, etc.
        Default returns strike if available.
        
        Args:
            product: The option product
            pricing_env: Pricing environment
            
        Returns:
            List of critical prices
        """
        points = []
        if hasattr(product, 'strike') and product.strike > 0:
            points.append(product.strike)
        return points
    
    def price(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment
    ) -> float:
        """
        Price the option using PDE method.
        
        Args:
            product: The option product
            pricing_env: Pricing environment
            
        Returns:
            Option price
        """
        # Extract market parameters
        spot = pricing_env.spot
        tau = product.get_maturity(pricing_env)
        
        if tau <= 0:
            # Option has expired, return intrinsic value
            return self._calculate_intrinsic(product, spot)
        
        strike = getattr(product, 'strike', spot)
        r = pricing_env.get_rate(tau)
        q = pricing_env.get_div_yield(tau)
        sigma = pricing_env.get_vol(strike, tau)
        
        # Build grids
        x_vec, s_vec, dx_vec, t_vec, dt_vec = self._build_grids(
            product, pricing_env, spot, sigma, tau, r, q
        )
        
        num_x = len(x_vec)
        num_t = len(t_vec)
        
        # Initialize solution grid
        grid = np.zeros((num_x, num_t))
        
        # Set terminal condition
        self.set_terminal_condition(grid, x_vec, s_vec, product, pricing_env)
        
        # Calculate finite difference coefficients
        dx = dx_vec[0] if len(np.unique(np.round(dx_vec, 10))) == 1 else None
        l, c, u = self._calculate_coefficients(r, q, sigma, dx_vec, num_x)
        
        # Build spatial operator matrix
        A = self._build_operator_matrix(l, c, u, num_x)
        
        # Time stepping (backward in time)
        self._time_stepping(
            grid, A, x_vec, s_vec, t_vec, dt_vec,
            product, pricing_env, r, q, sigma, tau
        )
        
        # Interpolate price at current spot
        spot_log = np.log(spot)
        price = self._interpolate_price(grid[:, 0], x_vec, spot_log)
        
        return price
    
    def calculate_greeks(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment
    ) -> Dict[str, float]:
        """
        Calculate Greeks using finite differences on the PDE solution.
        
        For delta and gamma, uses the solution surface directly.
        For other Greeks, uses bump-and-reprice.
        
        Args:
            product: The option product
            pricing_env: Pricing environment
            
        Returns:
            Dictionary of Greeks
        """
        # Get base price and solution surface
        spot = pricing_env.spot
        tau = product.get_maturity(pricing_env)
        
        if tau <= 0:
            intrinsic = self._calculate_intrinsic(product, spot)
            return {
                "price": intrinsic,
                "delta": self._intrinsic_delta(product, spot),
                "gamma": 0.0
            }
        
        strike = getattr(product, 'strike', spot)
        r = pricing_env.get_rate(tau)
        q = pricing_env.get_div_yield(tau)
        sigma = pricing_env.get_vol(strike, tau)
        
        # Build grids and solve
        x_vec, s_vec, dx_vec, t_vec, dt_vec = self._build_grids(
            product, pricing_env, spot, sigma, tau, r, q
        )
        
        num_x = len(x_vec)
        num_t = len(t_vec)
        
        grid = np.zeros((num_x, num_t))
        self.set_terminal_condition(grid, x_vec, s_vec, product, pricing_env)
        
        l, c, u = self._calculate_coefficients(r, q, sigma, dx_vec, num_x)
        A = self._build_operator_matrix(l, c, u, num_x)
        
        self._time_stepping(
            grid, A, x_vec, s_vec, t_vec, dt_vec,
            product, pricing_env, r, q, sigma, tau
        )
        
        # Calculate Greeks from solution surface
        spot_log = np.log(spot)
        price = self._interpolate_price(grid[:, 0], x_vec, spot_log)
        delta, gamma = self._calculate_delta_gamma(grid[:, 0], x_vec, spot_log, spot)
        
        return {
            "price": price,
            "delta": delta,
            "gamma": gamma
        }
    
    def _build_grids(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        spot: float,
        sigma: float,
        tau: float,
        r: float,
        q: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build spatial and temporal grids.
        
        Returns:
            Tuple of (x_vec, s_vec, dx_vec, t_vec, dt_vec)
        """
        params: PDEParams = self.params
        
        # Determine spatial bounds
        if params.s_min > 0 and params.s_max > 0:
            s_min, s_max = params.s_min, params.s_max
        else:
            strike = getattr(product, 'strike', spot)
            barriers = self._get_barriers(product)
            s_min, s_max = SpatialGrid.calculate_auto_bounds(
                spot, sigma, tau, r, q,
                strike=strike,
                barriers=barriers
            )
        
        # Get critical points for grid concentration
        critical_points = self.get_critical_points(product, pricing_env)
        
        # Build spatial grid
        x_vec, s_vec, dx_vec = SpatialGrid.build(
            s_min, s_max, params.grid_size,
            critical_points=critical_points,
            use_adaptive=params.adaptive_grid
        )
        
        # Get event times for time grid
        event_times = self._get_event_times(product, tau)
        
        # Build time grid
        t_vec, dt_vec = TimeGrid.build(
            tau, params.time_steps,
            method=params.time_grid_type,
            event_times=event_times,
            grade_exponent=params.grade_exponent
        )
        
        return x_vec, s_vec, dx_vec, t_vec, dt_vec
    
    def _get_barriers(self, product: BaseEquityProduct) -> List[float]:
        """Extract barrier levels from product if any."""
        barriers = []
        for attr in ('barrier', 'upper_barrier', 'lower_barrier'):
            if hasattr(product, attr):
                val = getattr(product, attr)
                if val is not None and val > 0:
                    barriers.append(val)
        return barriers
    
    def _get_event_times(
        self,
        product: BaseEquityProduct,
        tau: float
    ) -> Optional[List[float]]:
        """Extract observation times from product if any."""
        for attr in ('observation_dates', 'obs_times', 'event_times'):
            if hasattr(product, attr):
                times = getattr(product, attr)
                if times is not None and len(times) > 0:
                    return [t for t in times if 0 < t < tau]
        return None
    
    def _calculate_coefficients(
        self,
        r: float,
        q: float,
        sigma: float,
        dx_vec: np.ndarray,
        num_x: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate finite difference coefficients for the log-price PDE.
        
        The PDE in log-price space:
            dV/dt + mu*dV/dx + 0.5*sigma^2*d2V/dx^2 - r*V = 0
        
        where mu = r - q - 0.5*sigma^2
        
        Using central differences:
            dV/dx ≈ (V[i+1] - V[i-1]) / (2*dx)
            d2V/dx^2 ≈ (V[i+1] - 2*V[i] + V[i-1]) / dx^2
        
        Returns coefficients (l, c, u) for the tridiagonal system:
            l[i] * V[i-1] + c[i] * V[i] + u[i] * V[i+1] = dV/dt
        
        Args:
            r: Risk-free rate
            q: Dividend yield
            sigma: Volatility
            dx_vec: Grid spacings
            num_x: Number of spatial points
            
        Returns:
            Tuple of (l, c, u) coefficient arrays
        """
        # For uniform grid, use average spacing
        dx = np.mean(dx_vec)
        
        # Drift and diffusion terms
        mu = r - q - 0.5 * sigma * sigma
        diffusion = 0.5 * sigma * sigma
        
        # Coefficients
        drift_term = mu / (2.0 * dx)
        diff_term = diffusion / (dx * dx)
        
        l = np.full(num_x, diff_term - drift_term)  # Lower diagonal
        c = np.full(num_x, -2.0 * diff_term - r)    # Main diagonal
        u = np.full(num_x, diff_term + drift_term)  # Upper diagonal
        
        return l, c, u
    
    def _build_operator_matrix(
        self,
        l: np.ndarray,
        c: np.ndarray,
        u: np.ndarray,
        num_x: int
    ) -> sp.csc_matrix:
        """
        Build the sparse spatial operator matrix A.
        
        The interior points use the tridiagonal stencil.
        Boundary rows are set to identity (boundary conditions applied separately).
        
        Args:
            l: Lower diagonal coefficients
            c: Center diagonal coefficients
            u: Upper diagonal coefficients
            num_x: Number of spatial points
            
        Returns:
            Sparse CSC matrix
        """
        # Build tridiagonal matrix for interior points
        diagonals = [
            l[2:],      # Lower diagonal (offset -1)
            c[1:-1],    # Main diagonal
            u[:-2]      # Upper diagonal (offset +1)
        ]
        
        # Create sparse matrix
        A = sp.diags(
            diagonals,
            offsets=[-1, 0, 1],
            shape=(num_x - 2, num_x - 2),
            format='csc'
        )
        
        return A
    
    def _time_stepping(
        self,
        grid: np.ndarray,
        A: sp.csc_matrix,
        x_vec: np.ndarray,
        s_vec: np.ndarray,
        t_vec: np.ndarray,
        dt_vec: np.ndarray,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        r: float,
        q: float,
        sigma: float,
        tau: float
    ) -> None:
        """
        Perform backward time stepping to solve the PDE.
        
        Uses Crank-Nicolson scheme (theta = 0.5) with optional
        Rannacher smoothing (theta = 1.0 for first few steps).
        
        Scheme: (I - theta*dt*A) * V^n = (I + (1-theta)*dt*A) * V^{n+1}
        
        Args:
            grid: Solution grid, modified in place
            A: Spatial operator matrix
            x_vec: Log-price grid points
            s_vec: Price grid points
            t_vec: Time points
            dt_vec: Time step sizes
            product: The option product
            pricing_env: Pricing environment
            r: Risk-free rate
            q: Dividend yield
            sigma: Volatility
            tau: Total time to maturity
        """
        params: PDEParams = self.params
        num_t = len(t_vec)
        num_x = len(x_vec)
        
        # Identity matrix for interior points
        I = sp.eye(num_x - 2, format='csc')
        
        # Clear matrix cache for new solve
        self._matrix_cache.clear()
        
        # Time step backward from maturity
        for j in range(num_t - 2, -1, -1):
            dt = dt_vec[j]
            current_tau = tau - t_vec[j]  # Time remaining to maturity
            
            # Determine theta (Rannacher smoothing at start)
            steps_from_end = num_t - 1 - j
            if params.use_rannacher and steps_from_end < params.rannacher_steps:
                theta = 1.0  # Backward Euler for smoothing
            else:
                theta = params.theta
            
            # Get or compute matrices for this (dt, theta) combination
            M1, M2_lu = self._get_matrices(I, A, dt, theta)
            
            # Set boundary conditions for current time step
            self.set_boundary_conditions(
                grid, x_vec, s_vec, j, current_tau, product, pricing_env
            )
            
            # Extract interior values at next time step
            V_next = grid[1:-1, j + 1]
            
            # Right-hand side: (I + (1-theta)*dt*A) * V^{n+1}
            rhs = M1 @ V_next
            
            # Add boundary contributions
            # These come from the first and last rows of the full system
            # For simplicity, we assume Dirichlet BCs are already set
            
            # Solve: (I - theta*dt*A) * V^n = rhs
            V_curr = M2_lu.solve(rhs)
            
            # Store interior solution
            grid[1:-1, j] = V_curr
            
            # Apply any product-specific modifications (e.g., early exercise)
            self._apply_step_modifications(
                grid, x_vec, s_vec, j, current_tau, product, pricing_env
            )
    
    def _get_matrices(
        self,
        I: sp.csc_matrix,
        A: sp.csc_matrix,
        dt: float,
        theta: float
    ) -> Tuple[sp.csc_matrix, spla.SuperLU]:
        """
        Get or compute matrices for time stepping.
        
        Caches LU factorizations for efficiency when dt and theta repeat.
        
        Args:
            I: Identity matrix
            A: Spatial operator matrix
            dt: Time step size
            theta: Scheme parameter (0.5 = CN, 1.0 = BE)
            
        Returns:
            Tuple of (M1, M2_lu) where:
                M1 = I + (1-theta)*dt*A (for RHS)
                M2_lu = LU factorization of I - theta*dt*A (for LHS)
        """
        # Round dt to avoid floating point comparison issues
        key = (round(dt, 12), round(theta, 6))
        
        if key in self._matrix_cache:
            return self._matrix_cache[key]
        
        # Build matrices
        M1 = I + (1.0 - theta) * dt * A
        M2 = I - theta * dt * A
        
        # LU factorization of M2
        try:
            M2_lu = spla.splu(M2)
        except Exception as e:
            raise NumericalError(f"Failed to factorize matrix: {e}")
        
        self._matrix_cache[key] = (M1, M2_lu)
        return M1, M2_lu
    
    def _apply_step_modifications(
        self,
        grid: np.ndarray,
        x_vec: np.ndarray,
        s_vec: np.ndarray,
        t_idx: int,
        tau: float,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment
    ) -> None:
        """
        Apply product-specific modifications after each time step.
        
        Override this method for American options (early exercise)
        or barrier options (barrier checks).
        
        Args:
            grid: Solution grid
            x_vec: Log-price grid points
            s_vec: Price grid points
            t_idx: Current time index
            tau: Time remaining to maturity
            product: The option product
            pricing_env: Pricing environment
        """
        pass  # Default: no modifications
    
    def _interpolate_price(
        self,
        v_vec: np.ndarray,
        x_vec: np.ndarray,
        x_target: float
    ) -> float:
        """
        Interpolate option value at target log-price.
        
        Uses linear interpolation between nearest grid points.
        
        Args:
            v_vec: Option values at grid points
            x_vec: Log-price grid points
            x_target: Target log-price
            
        Returns:
            Interpolated option value
        """
        return float(np.interp(x_target, x_vec, v_vec))
    
    def _calculate_delta_gamma(
        self,
        v_vec: np.ndarray,
        x_vec: np.ndarray,
        x_target: float,
        spot: float
    ) -> Tuple[float, float]:
        """
        Calculate delta and gamma from the solution vector.
        
        In log-space:
            dV/dS = (1/S) * dV/dx
            d2V/dS2 = (1/S^2) * (d2V/dx^2 - dV/dx)
        
        Args:
            v_vec: Option values at grid points
            x_vec: Log-price grid points
            x_target: Target log-price (ln(spot))
            spot: Current spot price
            
        Returns:
            Tuple of (delta, gamma)
        """
        # Find nearest grid points
        idx = np.searchsorted(x_vec, x_target)
        idx = max(1, min(idx, len(x_vec) - 2))
        
        # Local grid spacing
        dx_left = x_vec[idx] - x_vec[idx - 1]
        dx_right = x_vec[idx + 1] - x_vec[idx]
        dx_avg = (dx_left + dx_right) / 2.0
        
        # Central differences for derivatives in log-space
        dv_dx = (v_vec[idx + 1] - v_vec[idx - 1]) / (dx_left + dx_right)
        d2v_dx2 = (v_vec[idx + 1] - 2 * v_vec[idx] + v_vec[idx - 1]) / (dx_avg ** 2)
        
        # Convert to price-space derivatives
        delta = dv_dx / spot
        gamma = (d2v_dx2 - dv_dx) / (spot ** 2)
        
        return delta, gamma
    
    def _calculate_intrinsic(
        self,
        product: BaseEquityProduct,
        spot: float
    ) -> float:
        """
        Calculate intrinsic value of the option.
        
        Args:
            product: The option product
            spot: Current spot price
            
        Returns:
            Intrinsic value
        """
        if hasattr(product, 'get_payoff'):
            return product.get_payoff(spot)
        return 0.0
    
    def _intrinsic_delta(
        self,
        product: BaseEquityProduct,
        spot: float
    ) -> float:
        """
        Calculate delta of intrinsic value.
        
        Args:
            product: The option product
            spot: Current spot price
            
        Returns:
            Intrinsic delta
        """
        if hasattr(product, 'is_call') and hasattr(product, 'strike'):
            if product.is_call():
                return 1.0 if spot > product.strike else 0.0
            else:
                return -1.0 if spot < product.strike else 0.0
        return 0.0

