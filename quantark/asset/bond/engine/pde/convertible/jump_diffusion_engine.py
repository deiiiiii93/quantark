"""
Jump-diffusion PDE engine for convertible bond pricing.

Implements the Bloomberg OVCV model where the stock follows a jump-diffusion
process with credit risk modeled via hazard rate.
"""
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from quantark.asset.bond.product.convertible.convertible_bond import ConvertibleBond
from quantark.asset.bond.engine.pde.convertible.pde_params import ConvertibleBondPDEParams
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import ValidationError, PricingError
from quantark.util.numerical import Tolerance, safe_exp, safe_sqrt, safe_log, is_zero


@dataclass
class ConvertibleBondJumpDiffusionResult:
    """
    Result container for jump-diffusion PDE engine pricing.

    Attributes:
        price: Clean price of the convertible bond
        dirty_price: Dirty price including accrued interest
        delta: Price sensitivity to stock price
        gamma: Second derivative of price with respect to stock
        theta: Time decay (daily)
        conversion_probability: Risk-neutral probability of eventual conversion
    """

    price: float
    dirty_price: float
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    conversion_probability: float = 0.0


class ConvertibleBondJumpDiffusionEngine:
    """
    Bloomberg OVCV jump-diffusion model for convertible bonds.

    This engine implements a finite-difference PDE solver where the stock
    follows:
        dS = (r - q - lambda * eta) * S * dt + sigma * S * dW - eta * S * dN

    where:
        lambda = hazard rate (default intensity)
        eta = stock price drop on default (stock_jump_on_default)
        dN = Poisson process with intensity lambda

    The convertible bond PDE becomes:
        V_t + 0.5 * sigma^2 * S^2 * V_SS + (r - q - lambda*eta) * S * V_S
            - r * V - lambda * (V - recovery) = 0

    Boundary conditions:
        - At S=0: V = recovery * face_value
        - At S=S_max: V = conversion_ratio * S (pure equity)
        - At maturity: V = max(face_value, conversion_ratio * S)
    """

    def __init__(
        self,
        pricing_env: PricingEnvironment,
        params: Optional[ConvertibleBondPDEParams] = None,
    ):
        """
        Initialize the jump-diffusion PDE engine.

        Args:
            pricing_env: Pricing environment with market data
            params: PDE configuration parameters (optional)
        """
        if pricing_env is None:
            raise ValidationError("Pricing environment is required")

        self.pricing_env = pricing_env
        self.params = params if params is not None else ConvertibleBondPDEParams()

    def price(self, bond: ConvertibleBond) -> float:
        """
        Calculate the clean price of the convertible bond.

        Args:
            bond: Convertible bond to price

        Returns:
            Clean price of the bond
        """
        result = self.price_with_details(bond)
        return result.price

    def price_with_details(
        self, bond: ConvertibleBond
    ) -> ConvertibleBondJumpDiffusionResult:
        """
        Calculate price with detailed results including Greeks.

        Args:
            bond: Convertible bond to price

        Returns:
            ConvertibleBondJumpDiffusionResult with full pricing details
        """
        valuation_date = self.pricing_env.valuation_date

        # Validate inputs
        if bond.is_expired(valuation_date):
            raise PricingError("Cannot price expired bond")

        # Get market data
        spot = self.pricing_env.spot
        T = bond.time_to_maturity(valuation_date)

        # Credit parameters from bond
        lambda_h = bond.hazard_rate  # Hazard rate
        eta = bond.stock_jump_on_default  # Stock drop on default
        recovery = bond.recovery_rate

        # Dividend yield
        q = bond.continuous_dividend_yield

        # Build grid
        S_min = spot * self.params.min_stock_multiple
        S_max = spot * self.params.max_stock_multiple
        N_s = self.params.num_space_steps
        N_t = self.params.num_time_steps
        dt = T / N_t

        # Use log-space grid for better accuracy
        log_S_min = safe_log(S_min)
        log_S_max = safe_log(S_max)
        log_S = np.linspace(log_S_min, log_S_max, N_s + 1)
        S = np.exp(log_S)
        dS = S[1:] - S[:-1]  # Variable spacing

        # Initialize solution
        V = self._terminal_condition(bond, S)
        P = self._terminal_conversion_probability(bond, S)

        # Get coupon schedule
        coupon_schedule = self._build_coupon_schedule(bond, valuation_date)

        # Time stepping - backward from T to 0
        # All times are measured in years from valuation date
        for n in range(N_t - 1, -1, -1):
            t = n * dt  # Current time (years from valuation)
            t_next = t + dt  # End of this step (closer to maturity)
            node_date = valuation_date + timedelta(days=int(t * 365))

            # Use Rannacher smoothing for first few steps
            use_implicit = n >= N_t - self.params.rannacher_steps

            # Query time-local forward rate for this step
            r_local = self.pricing_env.rate_curve.get_forward_rate(t, t_next)

            # Query time-local effective volatility for this step
            vol_local = self.pricing_env.get_step_volatility(spot, t, t_next)

            # Apply coupon payments (jump condition) to the known later-time state
            coupon_amount = 0.0
            for ct, ca in coupon_schedule:
                if t < ct <= t_next:
                    coupon_amount += ca
            if coupon_amount > 0.0:
                V = V + coupon_amount

            # Build discretization matrices with time-local parameters
            A, b = self._build_matrices(
                S, V, r_local, q, vol_local, lambda_h, eta, recovery, bond, dt, use_implicit
            )

            # Solve system
            if use_implicit or self.params.scheme == "implicit_euler":
                # Implicit: solve A * V_new = b
                V = spsolve(A.tocsr(), b)
            else:
                # Crank-Nicolson or explicit
                V = spsolve(A.tocsr(), b)

            conversion_possible = bond.conversion_end_date >= node_date
            p_upper_boundary = 1.0 if conversion_possible else 0.0
            A_p, b_p = self._build_probability_matrices(
                S,
                P,
                r_local,
                q,
                vol_local,
                lambda_h,
                eta,
                dt,
                use_implicit,
                p_upper_boundary=p_upper_boundary,
            )
            P = spsolve(A_p.tocsr(), b_p)
            P = np.clip(P, 0.0, 1.0)

            # Apply early exercise constraints to both value and probability
            V, P = self._apply_constraints(bond, S, V, node_date, P)
            P = np.clip(P, 0.0, 1.0)

        # Interpolate to get value at spot
        spot_idx = np.searchsorted(S, spot)
        if spot_idx == 0:
            dirty_price = V[0]
            conv_prob = P[0]
        elif spot_idx >= len(S):
            dirty_price = V[-1]
            conv_prob = P[-1]
        else:
            # Linear interpolation
            w = (spot - S[spot_idx - 1]) / (S[spot_idx] - S[spot_idx - 1])
            dirty_price = (1 - w) * V[spot_idx - 1] + w * V[spot_idx]
            conv_prob = (1 - w) * P[spot_idx - 1] + w * P[spot_idx]
        conv_prob = float(np.clip(conv_prob, 0.0, 1.0))

        # Calculate Greeks
        delta, gamma = self._calculate_greeks(S, V, spot, spot_idx)
        theta = self._calculate_theta(bond, dirty_price)

        # Calculate accrued interest
        accrued = bond.calculate_accrued_interest(valuation_date)
        clean_price = dirty_price - accrued

        return ConvertibleBondJumpDiffusionResult(
            price=clean_price,
            dirty_price=dirty_price,
            delta=delta,
            gamma=gamma,
            theta=theta,
            conversion_probability=conv_prob,
        )

    def _terminal_condition(
        self, bond: ConvertibleBond, S: np.ndarray
    ) -> np.ndarray:
        """
        Compute terminal condition at maturity.

        At maturity: V = max(face_value, conversion_ratio * S)

        Args:
            bond: Convertible bond
            S: Stock price grid

        Returns:
            Array of terminal values
        """
        face_value = bond.face_value
        conversion_value = bond.conversion_ratio * S
        if bond.is_convertible_at(bond.maturity_date):
            return np.maximum(face_value, conversion_value)
        return face_value * np.ones_like(S)

    def _terminal_conversion_probability(
        self, bond: ConvertibleBond, S: np.ndarray
    ) -> np.ndarray:
        """
        Terminal condition for the eventual conversion probability.

        At maturity: P = 1 if conversion is optimal and allowed, else 0.
        """
        if not bond.is_convertible_at(bond.maturity_date):
            return np.zeros_like(S)
        conversion_value = bond.conversion_ratio * S
        return (conversion_value > bond.face_value).astype(float)

    def _build_matrices(
        self,
        S: np.ndarray,
        V: np.ndarray,
        r: float,
        q: float,
        vol: float,
        lambda_h: float,
        eta: float,
        recovery: float,
        bond: ConvertibleBond,
        dt: float,
        use_implicit: bool,
    ) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """
        Build finite difference matrices for the PDE.

        Args:
            S: Stock price grid
            V: Current solution
            r, q: Rate and dividend yield
            vol: Volatility
            lambda_h: Hazard rate
            eta: Stock jump on default
            recovery: Recovery rate
            bond: Convertible bond
            dt: Time step
            use_implicit: Whether to use fully implicit scheme

        Returns:
            Tuple of (A matrix, b vector) for A*V_new = b
        """
        N = len(S)
        recovery_value = recovery * bond.face_value

        # Coefficient for drift term (adjusted for default jump)
        drift = r - q - lambda_h * eta

        # Build tridiagonal matrix coefficients
        # Using central differences for interior points
        diag = np.zeros(N)
        lower = np.zeros(N - 1)
        upper = np.zeros(N - 1)

        for i in range(1, N - 1):
            h_minus = S[i] - S[i - 1]
            h_plus = S[i + 1] - S[i]
            h = 0.5 * (h_minus + h_plus)

            # Diffusion coefficient: 0.5 * sigma^2 * S^2
            D = 0.5 * vol * vol * S[i] * S[i]

            # Convection coefficient: drift * S
            C = drift * S[i]

            # Reaction coefficient: -(r + lambda)
            R = -(r + lambda_h)

            # Central difference approximation
            lower[i - 1] = D / (h_minus * h) - C / (2 * h)
            diag[i] = -2 * D / (h_minus * h_plus) + R
            upper[i] = D / (h_plus * h) + C / (2 * h)

        # Boundary conditions
        # At S=0: V = recovery_value (Dirichlet)
        diag[0] = 1.0
        upper[0] = 0.0

        # At S_max: V = conversion_ratio * S (Dirichlet for deep ITM)
        diag[-1] = 1.0
        lower[-1] = 0.0

        # Build matrices based on scheme
        if use_implicit:
            # Fully implicit: (I - dt*A) * V_new = V + dt * source
            theta_scheme = 1.0
        elif self.params.scheme == "explicit_euler":
            theta_scheme = 0.0
        else:  # crank_nicolson
            theta_scheme = 0.5

        # A_new = I - theta * dt * L
        # A_old = I + (1 - theta) * dt * L
        # A_new * V_new = A_old * V + dt * source

        L = sparse.diags(
            [lower, diag, upper], [-1, 0, 1], shape=(N, N), format="csr"
        )

        I = sparse.eye(N, format="csr")

        A = I - theta_scheme * dt * L

        # Right-hand side
        if theta_scheme < 1.0:
            A_old = I + (1 - theta_scheme) * dt * L
            b = A_old @ V
        else:
            b = V.copy()

        # Add jump term contribution (recovery on default)
        b[1:-1] += dt * lambda_h * recovery_value

        # Apply boundary conditions
        b[0] = recovery_value
        b[-1] = bond.conversion_ratio * S[-1]

        return A, b

    def _build_probability_matrices(
        self,
        S: np.ndarray,
        P: np.ndarray,
        r: float,
        q: float,
        vol: float,
        lambda_h: float,
        eta: float,
        dt: float,
        use_implicit: bool,
        p_upper_boundary: float,
    ) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """
        Build finite difference matrices for the eventual conversion probability.

        The probability PDE incorporates default as a killing term (-lambda * P)
        and uses Dirichlet boundaries at the grid endpoints.
        """
        N = len(S)

        drift = r - q - lambda_h * eta

        diag = np.zeros(N)
        lower = np.zeros(N - 1)
        upper = np.zeros(N - 1)

        for i in range(1, N - 1):
            h_minus = S[i] - S[i - 1]
            h_plus = S[i + 1] - S[i]
            h = 0.5 * (h_minus + h_plus)

            D = 0.5 * vol * vol * S[i] * S[i]
            C = drift * S[i]
            R = -lambda_h

            lower[i - 1] = D / (h_minus * h) - C / (2 * h)
            diag[i] = -2 * D / (h_minus * h_plus) + R
            upper[i] = D / (h_plus * h) + C / (2 * h)

        # Dirichlet boundaries: enforce via zero operator rows
        diag[0] = 0.0
        diag[-1] = 0.0
        upper[0] = 0.0
        lower[-1] = 0.0

        if use_implicit:
            theta_scheme = 1.0
        elif self.params.scheme == "explicit_euler":
            theta_scheme = 0.0
        else:  # crank_nicolson
            theta_scheme = 0.5

        L = sparse.diags(
            [lower, diag, upper], [-1, 0, 1], shape=(N, N), format="csr"
        )
        I = sparse.eye(N, format="csr")

        A = I - theta_scheme * dt * L

        if theta_scheme < 1.0:
            A_old = I + (1 - theta_scheme) * dt * L
            b = A_old @ P
        else:
            b = P.copy()

        b[0] = 0.0
        b[-1] = p_upper_boundary

        return A, b

    def _apply_constraints(
        self,
        bond: ConvertibleBond,
        S: np.ndarray,
        V: np.ndarray,
        node_date: datetime,
        P: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply early exercise constraints (conversion, call, put).

        Args:
            bond: Convertible bond
            S: Stock price grid
            V: Current values
            node_date: Current date in backward induction

        Returns:
            Tuple of (updated values, updated conversion probability) with
            constraints applied
        """
        V_new = V.copy()
        P_new = P.copy()

        # Conversion constraint (holder's right)
        if bond.is_convertible_at(node_date):
            conversion_value = bond.conversion_ratio * S
            convert_mask = conversion_value >= (V_new - Tolerance.PRECISION)
            V_new = np.maximum(V_new, conversion_value)
            if np.any(convert_mask):
                P_new[convert_mask] = 1.0

        # Call constraint (issuer's right)
        call_price = bond.get_call_price_at(node_date)
        if call_price is not None:
            for i, stock in enumerate(S):
                if bond.is_callable_at(node_date, stock):
                    # Issuer calls if V > call_price
                    # Holder chooses max(conversion, call_price)
                    if V_new[i] > call_price:
                        conversion_value = bond.conversion_ratio * stock
                        if bond.is_convertible_at(node_date):
                            V_new[i] = max(conversion_value, call_price)
                            if conversion_value > call_price:
                                P_new[i] = 1.0
                            else:
                                P_new[i] = 0.0
                        else:
                            V_new[i] = call_price
                            P_new[i] = 0.0

        # Put constraint (holder's right)
        put_price = bond.get_put_price_at(node_date)
        if put_price is not None:
            put_mask = put_price >= (V_new - Tolerance.PRECISION)
            V_new = np.maximum(V_new, put_price)
            if np.any(put_mask):
                P_new[put_mask] = 0.0

        return V_new, P_new

    def _build_coupon_schedule(
        self, bond: ConvertibleBond, valuation_date: datetime
    ) -> list:
        """
        Build list of (time, amount) tuples for coupon payments.

        Args:
            bond: Convertible bond
            valuation_date: Valuation date

        Returns:
            List of (time_to_payment, coupon_amount) tuples
        """
        T = bond.time_to_maturity(valuation_date)
        schedule = []

        for cf in bond.get_all_cashflows():
            cf_time = (cf.payment_date - valuation_date).days / 365.0
            if 0 < cf_time <= T:
                # Extract just the coupon (not principal)
                coupon_amount = cf.amount
                if cf.payment_date >= bond.maturity_date:
                    coupon_amount -= bond.face_value
                if coupon_amount > 0:
                    schedule.append((cf_time, coupon_amount))

        return schedule

    def _calculate_greeks(
        self,
        S: np.ndarray,
        V: np.ndarray,
        spot: float,
        spot_idx: int,
    ) -> Tuple[float, float]:
        """
        Calculate delta and gamma from the PDE grid.

        Args:
            S: Stock price grid
            V: Solution values
            spot: Current spot price
            spot_idx: Index of spot in grid

        Returns:
            Tuple of (delta, gamma)
        """
        if spot_idx <= 1 or spot_idx >= len(S) - 1:
            return 0.0, 0.0

        # Use central differences
        i = spot_idx
        h_minus = S[i] - S[i - 1]
        h_plus = S[i + 1] - S[i]

        # Delta: dV/dS
        delta = (V[i + 1] - V[i - 1]) / (h_plus + h_minus)

        # Gamma: d2V/dS2
        gamma = 2.0 * (
            V[i + 1] / (h_plus * (h_plus + h_minus))
            - V[i] / (h_plus * h_minus)
            + V[i - 1] / (h_minus * (h_plus + h_minus))
        )

        return delta, gamma

    def _calculate_theta(
        self, bond: ConvertibleBond, current_price: float
    ) -> float:
        """
        Calculate theta (time decay) by finite difference.

        Args:
            bond: Convertible bond
            current_price: Current dirty price

        Returns:
            Daily theta
        """
        # Theta calculation would require pricing at t+dt
        # For now, return 0 (could be implemented with bump-and-reprice)
        return 0.0

    def __repr__(self):
        return (
            f"ConvertibleBondJumpDiffusionEngine("
            f"scheme={self.params.scheme})"
        )
