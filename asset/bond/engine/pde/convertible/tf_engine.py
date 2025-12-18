"""
Tsiveriotis-Fernandes decomposition PDE engine for convertible bond pricing.

Implements the TF model which decomposes the convertible bond value into
equity-like and debt-like components, each discounted at appropriate rates.
"""
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from asset.bond.product.convertible.convertible_bond import ConvertibleBond
from asset.bond.engine.pde.convertible.pde_params import ConvertibleBondPDEParams
from priceenv import PricingEnvironment
from util.exceptions import ValidationError, PricingError
from util.numerical import Tolerance, safe_exp, safe_sqrt, safe_log, is_zero


@dataclass
class ConvertibleBondTFResult:
    """
    Result container for Tsiveriotis-Fernandes PDE engine pricing.

    Attributes:
        price: Clean price of the convertible bond
        dirty_price: Dirty price including accrued interest
        equity_component: Equity-like component (u) - value conditional on conversion
        bond_component: Bond-like component (v) - value conditional on redemption
        delta: Price sensitivity to stock price
        gamma: Second derivative of price with respect to stock
        theta: Time decay (daily)
        conversion_probability: Risk-neutral probability of eventual conversion
    """

    price: float
    dirty_price: float
    equity_component: float
    bond_component: float
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    conversion_probability: float = 0.0


class ConvertibleBondTFEngine:
    """
    Tsiveriotis-Fernandes decomposition model for convertible bonds.

    This engine implements the TF model which splits the convertible bond
    value V into two components:
        V = u + v

    where:
        u = equity component (discounted at risk-free rate r)
        v = bond/debt component (discounted at risky rate r + credit_spread)

    The coupled PDE system is:
        u_t + 0.5*sigma^2*S^2*u_SS + (r-q)*S*u_S - r*u = 0
        v_t + 0.5*sigma^2*S^2*v_SS + (r-q)*S*v_S - (r+credit_spread)*v = 0

    with boundary conditions:
        - At conversion: u = conversion_value, v = 0
        - At redemption: u = 0, v = face_value
        - At maturity: u = max(0, conversion_value - face_value), v = min(face_value, conversion_value)

    This decomposition is particularly useful for analyzing the COCB
    (cash-only component of the bond) which is just v.
    """

    def __init__(
        self,
        pricing_env: PricingEnvironment,
        params: Optional[ConvertibleBondPDEParams] = None,
    ):
        """
        Initialize the TF PDE engine.

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
    ) -> ConvertibleBondTFResult:
        """
        Calculate price with detailed results including component decomposition.

        Args:
            bond: Convertible bond to price

        Returns:
            ConvertibleBondTFResult with full pricing details and decomposition
        """
        valuation_date = self.pricing_env.valuation_date

        # Validate inputs
        if bond.is_expired(valuation_date):
            raise PricingError("Cannot price expired bond")

        # Get market data
        spot = self.pricing_env.spot
        T = bond.time_to_maturity(valuation_date)
        vol = self.pricing_env.get_vol(spot, T)
        r = self.pricing_env.rate_curve.get_rate(T)

        # Credit parameters
        credit_spread = bond.credit_spread
        r_risky = r + credit_spread

        # Dividend yield
        q = bond.continuous_dividend_yield

        # Build grid
        S_min = spot * self.params.min_stock_multiple
        S_max = spot * self.params.max_stock_multiple
        N_s = self.params.num_space_steps
        N_t = self.params.num_time_steps
        dt = T / N_t

        # Use log-space grid
        log_S_min = safe_log(S_min)
        log_S_max = safe_log(S_max)
        log_S = np.linspace(log_S_min, log_S_max, N_s + 1)
        S = np.exp(log_S)

        # Initialize solutions for both components
        u, v = self._terminal_condition(bond, S)
        P = self._terminal_conversion_probability(bond, S)

        # Get coupon schedule
        coupon_schedule = self._build_coupon_schedule(bond, valuation_date)

        # Time stepping - backward from T to 0
        for n in range(N_t - 1, -1, -1):
            t = n * dt
            node_date = valuation_date + timedelta(days=int(t * 365))

            # Use Rannacher smoothing for first few steps
            use_implicit = n >= N_t - self.params.rannacher_steps

            # Solve for equity component u (discounted at r)
            A_u, b_u = self._build_matrices(
                S, u, r, q, vol, r, dt, use_implicit
            )
            u = spsolve(A_u.tocsr(), b_u)

            # Apply coupon payments (jump condition) to the bond component's
            # known later-time state (coupons paid only if not converted).
            coupon_amount = 0.0
            for ct, ca in coupon_schedule:
                if t < ct <= t + dt:
                    coupon_amount += ca
            if coupon_amount > 0.0:
                v = v + coupon_amount

            # Solve for bond component v (discounted at r + credit_spread)
            A_v, b_v = self._build_matrices(
                S, v, r, q, vol, r_risky, dt, use_implicit
            )

            v = spsolve(A_v.tocsr(), b_v)

            conversion_possible = bond.conversion_end_date >= node_date
            P[0] = 0.0
            P[-1] = 1.0 if conversion_possible else 0.0
            A_p, b_p = self._build_matrices(
                S, P, r, q, vol, 0.0, dt, use_implicit
            )
            P = spsolve(A_p.tocsr(), b_p)
            P = np.clip(P, 0.0, 1.0)

            # Apply constraints and update components and probability
            u, v, P = self._apply_constraints(bond, S, u, v, node_date, P)
            P = np.clip(P, 0.0, 1.0)

        # Total value
        V = u + v

        # Interpolate to get values at spot
        spot_idx = np.searchsorted(S, spot)
        if spot_idx == 0:
            u_spot = u[0]
            v_spot = v[0]
            V_spot = V[0]
            conv_prob = P[0]
        elif spot_idx >= len(S):
            u_spot = u[-1]
            v_spot = v[-1]
            V_spot = V[-1]
            conv_prob = P[-1]
        else:
            w = (spot - S[spot_idx - 1]) / (S[spot_idx] - S[spot_idx - 1])
            u_spot = (1 - w) * u[spot_idx - 1] + w * u[spot_idx]
            v_spot = (1 - w) * v[spot_idx - 1] + w * v[spot_idx]
            V_spot = (1 - w) * V[spot_idx - 1] + w * V[spot_idx]
            conv_prob = (1 - w) * P[spot_idx - 1] + w * P[spot_idx]

        dirty_price = V_spot
        conv_prob = float(np.clip(conv_prob, 0.0, 1.0))

        # Calculate Greeks
        delta, gamma = self._calculate_greeks(S, V, spot, spot_idx)

        # Calculate accrued interest
        accrued = bond.calculate_accrued_interest(valuation_date)
        clean_price = dirty_price - accrued

        return ConvertibleBondTFResult(
            price=clean_price,
            dirty_price=dirty_price,
            equity_component=u_spot,
            bond_component=v_spot,
            delta=delta,
            gamma=gamma,
            theta=0.0,
            conversion_probability=conv_prob,
        )

    def _terminal_condition(
        self, bond: ConvertibleBond, S: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute terminal conditions for u and v at maturity.

        At maturity:
        - If conversion: u = conversion_value, v = 0
        - If redemption: u = 0, v = face_value
        - Decision: max(face_value, conversion_value)

        Args:
            bond: Convertible bond
            S: Stock price grid

        Returns:
            Tuple of (u, v) arrays
        """
        face_value = bond.face_value
        conversion_value = bond.conversion_ratio * S

        # u = equity component (value from conversion)
        # v = bond component (value from redemption)
        u = np.zeros_like(S)
        v = np.zeros_like(S)

        if not bond.is_convertible_at(bond.maturity_date):
            v[:] = face_value
            return u, v

        for i in range(len(S)):
            if conversion_value[i] > face_value:
                u[i] = conversion_value[i]
                v[i] = 0.0
            else:
                u[i] = 0.0
                v[i] = face_value

        return u, v

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
        discount_rate: float,
        dt: float,
        use_implicit: bool,
    ) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """
        Build finite difference matrices for the PDE.

        Args:
            S: Stock price grid
            V: Current solution
            r: Risk-free rate (for drift)
            q: Dividend yield
            vol: Volatility
            discount_rate: Rate for discounting (r for u, r+spread for v)
            dt: Time step
            use_implicit: Whether to use fully implicit scheme

        Returns:
            Tuple of (A matrix, b vector) for A*V_new = b
        """
        N = len(S)
        drift = r - q

        # Build tridiagonal matrix coefficients
        diag = np.zeros(N)
        lower = np.zeros(N - 1)
        upper = np.zeros(N - 1)

        for i in range(1, N - 1):
            h_minus = S[i] - S[i - 1]
            h_plus = S[i + 1] - S[i]
            h = 0.5 * (h_minus + h_plus)

            # Diffusion: 0.5 * sigma^2 * S^2
            D = 0.5 * vol * vol * S[i] * S[i]

            # Convection: drift * S
            C = drift * S[i]

            # Reaction: -discount_rate
            R = -discount_rate

            lower[i - 1] = D / (h_minus * h) - C / (2 * h)
            diag[i] = -2 * D / (h_minus * h_plus) + R
            upper[i] = D / (h_plus * h) + C / (2 * h)

        # Boundary conditions
        diag[0] = 1.0
        upper[0] = 0.0
        diag[-1] = 1.0
        lower[-1] = 0.0

        # Scheme selection
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
            b = A_old @ V
        else:
            b = V.copy()

        # Boundary values will be set by constraints
        b[0] = V[0]
        b[-1] = V[-1]

        return A, b

    def _apply_constraints(
        self,
        bond: ConvertibleBond,
        S: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        node_date: datetime,
        P: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply early exercise constraints and update component decomposition.

        Args:
            bond: Convertible bond
            S: Stock price grid
            u: Equity component
            v: Bond component
            node_date: Current date

        Returns:
            Tuple of updated (u, v, conversion_probability)
        """
        u_new = u.copy()
        v_new = v.copy()
        P_new = P.copy()
        V = u + v  # Total value

        face_value = bond.face_value

        for i in range(len(S)):
            stock = S[i]
            conversion_value = bond.conversion_ratio * stock
            total_value = V[i]

            # Track the optimal action
            action = "hold"
            optimal_value = total_value

            # Check conversion (holder's right)
            if bond.is_convertible_at(node_date):
                if conversion_value >= (optimal_value - Tolerance.PRECISION):
                    action = "convert"
                    optimal_value = conversion_value

            # Check call (issuer's right)
            call_price = bond.get_call_price_at(node_date)
            if call_price is not None and bond.is_callable_at(node_date, stock):
                if optimal_value > call_price:
                    # Issuer calls; holder chooses max(conversion, call)
                    if bond.is_convertible_at(node_date) and conversion_value > call_price:
                        action = "convert"
                        optimal_value = conversion_value
                    else:
                        action = "call"
                        optimal_value = call_price

            # Check put (holder's right)
            put_price = bond.get_put_price_at(node_date)
            if put_price is not None:
                if put_price > optimal_value:
                    action = "put"
                    optimal_value = put_price

            # Update components based on action
            if action == "convert":
                u_new[i] = conversion_value
                v_new[i] = 0.0
                P_new[i] = 1.0
            elif action == "call":
                u_new[i] = 0.0
                v_new[i] = call_price
                P_new[i] = 0.0
            elif action == "put":
                u_new[i] = 0.0
                v_new[i] = put_price
                P_new[i] = 0.0
            # else: hold - keep current decomposition

        # Boundary conditions
        # At S=0: pure bond component
        u_new[0] = 0.0
        v_new[0] = bond.recovery_rate * face_value
        P_new[0] = 0.0

        # At S_max: pure equity component
        u_new[-1] = bond.conversion_ratio * S[-1]
        v_new[-1] = 0.0
        conversion_possible = bond.conversion_end_date >= node_date
        P_new[-1] = 1.0 if conversion_possible else 0.0

        return u_new, v_new, P_new

    def _build_coupon_schedule(
        self, bond: ConvertibleBond, valuation_date: datetime
    ) -> list:
        """
        Build list of (time, amount) tuples for coupon payments.
        """
        T = bond.time_to_maturity(valuation_date)
        schedule = []

        for cf in bond.get_all_cashflows():
            cf_time = (cf.payment_date - valuation_date).days / 365.0
            if 0 < cf_time <= T:
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
        """
        if spot_idx <= 1 or spot_idx >= len(S) - 1:
            return 0.0, 0.0

        i = spot_idx
        h_minus = S[i] - S[i - 1]
        h_plus = S[i + 1] - S[i]

        delta = (V[i + 1] - V[i - 1]) / (h_plus + h_minus)

        gamma = 2.0 * (
            V[i + 1] / (h_plus * (h_plus + h_minus))
            - V[i] / (h_plus * h_minus)
            + V[i - 1] / (h_minus * (h_plus + h_minus))
        )

        return delta, gamma

    def get_cocb(self, bond: ConvertibleBond) -> float:
        """
        Get the Cash-Only Component of Bond (COCB).

        The COCB is the bond component v in the TF decomposition,
        which represents the present value of cash flows assuming
        no conversion ever occurs.

        Args:
            bond: Convertible bond

        Returns:
            COCB value
        """
        result = self.price_with_details(bond)
        return result.bond_component

    def __repr__(self):
        return f"ConvertibleBondTFEngine(scheme={self.params.scheme})"
