"""
Trinomial tree engine for convertible bond pricing.

Implements the Hull-White trinomial model with explicit default branch,
where at each node there's a probability of default leading to recovery
value.
"""
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import numpy as np

from asset.bond.product.convertible.convertible_bond import ConvertibleBond
from asset.bond.engine.tree.convertible.tree_params import ConvertibleBondTreeParams
from priceenv import PricingEnvironment
from util.exceptions import ValidationError, PricingError
from util.numerical import safe_exp, safe_sqrt, is_zero


@dataclass
class ConvertibleBondTrinomialResult:
    """
    Result container for trinomial engine pricing.

    Attributes:
        price: Clean price of the convertible bond
        dirty_price: Dirty price including accrued interest
        conversion_probability: Probability of eventual conversion
        default_probability: Probability of default before maturity
        delta: Price sensitivity to stock price
        gamma: Second derivative of price with respect to stock
        equity_component: Equity-like component of value
        bond_component: Bond-like component of value
        recovery_component: Recovery value component
    """

    price: float
    dirty_price: float
    conversion_probability: float
    default_probability: float
    delta: float = 0.0
    gamma: float = 0.0
    equity_component: float = 0.0
    bond_component: float = 0.0
    recovery_component: float = 0.0


class ConvertibleBondTrinomialEngine:
    """
    Hull-White trinomial model for convertible bonds with explicit default.

    This engine implements a trinomial tree where:
    1. Stock price can go up, down, or default at each node
    2. Default probability is derived from the hazard rate
    3. Upon default, bondholder receives recovery value
    4. Early exercise (conversion, call, put) is handled via backward induction

    The default probability per time step is:
        p_default = 1 - exp(-lambda * dt)

    where lambda is the hazard rate.

    Upon default:
    - Stock price jumps to eta * S (stock_jump_on_default)
    - Bondholder receives recovery_rate * face_value
    """

    def __init__(
        self,
        pricing_env: PricingEnvironment,
        params: Optional[ConvertibleBondTreeParams] = None,
    ):
        """
        Initialize the trinomial engine.

        Args:
            pricing_env: Pricing environment with market data
            params: Tree configuration parameters (optional)
        """
        if pricing_env is None:
            raise ValidationError("Pricing environment is required")

        self.pricing_env = pricing_env
        self.params = params if params is not None else ConvertibleBondTreeParams()

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
    ) -> ConvertibleBondTrinomialResult:
        """
        Calculate price with detailed results including Greeks and components.

        Args:
            bond: Convertible bond to price

        Returns:
            ConvertibleBondTrinomialResult with full pricing details
        """
        valuation_date = self.pricing_env.valuation_date

        # Validate inputs
        if bond.is_expired(valuation_date):
            raise PricingError("Cannot price expired bond")

        # Get market data
        spot = self.pricing_env.spot
        T = bond.time_to_maturity(valuation_date)
        vol = self.pricing_env.get_vol(spot, T)

        # Calculate time parameters
        dt = T / self.params.num_steps

        # Build tree parameters
        r = self.pricing_env.rate_curve.get_rate(T)
        q = bond.continuous_dividend_yield
        hazard_rate = bond.hazard_rate

        # Trinomial parameters
        tree_params = self._calculate_tree_params(vol, r, q, hazard_rate, dt)

        # Build stock price tree (non-recombining due to default branch)
        # Use a simplified approach with a recombining tree for the survival paths
        stock_tree = self._build_stock_tree(
            spot, tree_params["u"], tree_params["d"], self.params.num_steps
        )

        # Build value tree with backward induction
        results = self._backward_induction(bond, stock_tree, tree_params, dt)

        value_tree = results["value_tree"]
        conv_prob_tree = results["conv_prob_tree"]
        default_prob_tree = results["default_prob_tree"]

        # Extract results
        dirty_price = value_tree[0, 0]
        accrued = bond.calculate_accrued_interest(valuation_date)
        clean_price = dirty_price - accrued

        # Calculate conversion and default probabilities at root
        conv_prob = conv_prob_tree[0, 0]
        default_prob = default_prob_tree[0, 0]

        # Calculate delta and gamma from tree
        delta, gamma = self._calculate_greeks(
            stock_tree, value_tree, spot, self.params.num_steps
        )

        # Estimate components
        equity_component = conv_prob * dirty_price
        recovery_component = default_prob * bond.recovery_rate * bond.face_value
        bond_component = dirty_price - equity_component - recovery_component

        return ConvertibleBondTrinomialResult(
            price=clean_price,
            dirty_price=dirty_price,
            conversion_probability=conv_prob,
            default_probability=default_prob,
            delta=delta,
            gamma=gamma,
            equity_component=max(0, equity_component),
            bond_component=max(0, bond_component),
            recovery_component=recovery_component,
        )

    def _calculate_tree_params(
        self, vol: float, r: float, q: float, hazard_rate: float, dt: float
    ) -> Dict[str, float]:
        """
        Calculate trinomial tree parameters.

        Args:
            vol: Volatility
            r: Risk-free rate
            q: Dividend yield
            hazard_rate: Default intensity
            dt: Time step

        Returns:
            Dictionary with tree parameters
        """
        # Default probability per step
        p_default = 1.0 - safe_exp(-hazard_rate * dt)

        # Survival probability
        p_survive = 1.0 - p_default

        # For surviving paths, use CRR-style binomial within trinomial
        # Trinomial with stretching factor
        stretch = math.sqrt(3.0)
        u = safe_exp(vol * stretch * safe_sqrt(dt))
        d = 1.0 / u

        # Risk-neutral probabilities for up/down (conditional on survival)
        drift = safe_exp((r - q) * dt)
        var = safe_exp(vol * vol * dt) - 1.0

        # Match first two moments for trinomial
        # E[S_1/S_0] = drift
        # Var[S_1/S_0] = S_0^2 * (exp(sigma^2 * dt) - 1)

        # For trinomial: p_u * u + p_m * 1 + p_d * d = drift
        # p_u * u^2 + p_m * 1 + p_d * d^2 - drift^2 = var

        # Simplified: equal up/down probability around middle
        # p_u = p_d = (drift^2 + var - 1) / (2 * (u^2 - 1))
        # Capped to ensure valid probabilities

        u2 = u * u
        d2 = d * d

        # Solve for probabilities
        # Using standard trinomial parameterization
        p_up = (
            (drift - d) / (u - d) - (1.0 - d) * (drift - 1.0) / ((u - d) * (u - 1.0))
        ) * 0.5 + 0.5 * (drift - 1.0) / (u - 1.0)

        p_down = (
            (u - drift) / (u - d) - (u - 1.0) * (1.0 - drift) / ((u - d) * (1.0 - d))
        ) * 0.5 + 0.5 * (1.0 - drift) / (1.0 - d)

        # Simpler approach: standard CRR-like probabilities
        # adjusted for trinomial
        exp_growth = safe_exp((r - q) * dt)
        p_up = ((exp_growth - d) / (u - d)) * 0.5
        p_down = ((u - exp_growth) / (u - d)) * 0.5
        p_mid = 1.0 - p_up - p_down

        # Ensure valid probabilities
        p_up = max(0.0, min(1.0, p_up))
        p_down = max(0.0, min(1.0, p_down))
        p_mid = max(0.0, 1.0 - p_up - p_down)

        # Renormalize
        total = p_up + p_mid + p_down
        if total > 0:
            p_up /= total
            p_mid /= total
            p_down /= total

        return {
            "u": u,
            "d": d,
            "m": 1.0,  # Middle move is no change
            "p_up": p_up * p_survive,
            "p_mid": p_mid * p_survive,
            "p_down": p_down * p_survive,
            "p_default": p_default,
            "discount": safe_exp(-r * dt),
        }

    def _build_stock_tree(
        self, spot: float, u: float, d: float, n_steps: int
    ) -> np.ndarray:
        """
        Build recombining trinomial stock price tree.

        For simplicity, we use a recombining structure where the middle
        node stays at the same price. The tree has 2*n_steps + 1 nodes
        at the final time step.

        Args:
            spot: Initial stock price
            u: Up factor
            d: Down factor
            n_steps: Number of time steps

        Returns:
            2D array where tree[i, j] = stock price at time i, node j
        """
        # At time step i, there are 2*i + 1 nodes
        max_nodes = 2 * n_steps + 1
        tree = np.zeros((n_steps + 1, max_nodes))

        # Fill tree
        for i in range(n_steps + 1):
            num_nodes = 2 * i + 1
            mid_idx = i  # Middle index at time i

            for j in range(num_nodes):
                # j - mid_idx gives the net up moves
                net_ups = j - mid_idx
                if net_ups >= 0:
                    tree[i, j] = spot * (u ** net_ups)
                else:
                    tree[i, j] = spot * (d ** (-net_ups))

        return tree

    def _backward_induction(
        self,
        bond: ConvertibleBond,
        stock_tree: np.ndarray,
        tree_params: Dict[str, float],
        dt: float,
    ) -> Dict[str, np.ndarray]:
        """
        Perform backward induction with default branch.

        Args:
            bond: Convertible bond
            stock_tree: Pre-built stock price tree
            tree_params: Tree parameters
            dt: Time step

        Returns:
            Dictionary containing value_tree, conv_prob_tree, default_prob_tree
        """
        n_steps = self.params.num_steps
        valuation_date = self.pricing_env.valuation_date
        max_nodes = 2 * n_steps + 1

        # Initialize trees
        value_tree = np.zeros((n_steps + 1, max_nodes))
        conv_prob_tree = np.zeros((n_steps + 1, max_nodes))
        default_prob_tree = np.zeros((n_steps + 1, max_nodes))

        # Get coupon schedule
        all_cashflows = bond.get_all_cashflows()
        coupon_times = []
        coupon_amounts = []
        for cf in all_cashflows:
            cf_time = (cf.payment_date - valuation_date).days / 365.0
            if cf_time > 0:
                coupon_times.append(cf_time)
                coupon_amount = cf.amount
                if cf.payment_date >= bond.maturity_date:
                    coupon_amount -= bond.face_value
                coupon_amounts.append(max(0, coupon_amount))

        T = bond.time_to_maturity(valuation_date)
        r = self.pricing_env.rate_curve.get_rate(T)

        # Extract parameters
        u = tree_params["u"]
        d = tree_params["d"]
        p_up = tree_params["p_up"]
        p_mid = tree_params["p_mid"]
        p_down = tree_params["p_down"]
        p_default = tree_params["p_default"]
        discount = tree_params["discount"]

        # Recovery value on default
        recovery_value = bond.recovery_rate * bond.face_value

        # Terminal values at maturity
        num_terminal = 2 * n_steps + 1
        for j in range(num_terminal):
            stock = stock_tree[n_steps, j]
            conversion_value = bond.parity(stock)

            if conversion_value > bond.face_value:
                value_tree[n_steps, j] = conversion_value
                conv_prob_tree[n_steps, j] = 1.0
            else:
                value_tree[n_steps, j] = bond.face_value
                conv_prob_tree[n_steps, j] = 0.0

            default_prob_tree[n_steps, j] = 0.0  # No default at maturity node

        # Backward induction
        for i in range(n_steps - 1, -1, -1):
            t = i * dt
            node_date = valuation_date + timedelta(days=int(t * 365))
            num_nodes = 2 * i + 1

            for j in range(num_nodes):
                stock = stock_tree[i, j]

                # Child indices for trinomial
                j_up = j + 1  # Up move
                j_mid = j  # Middle (no move)
                j_down = j  # Down move (actually j - 1 in next layer, but mapped)

                # Map to next layer's indices
                # At time i+1, node indices are 0 to 2*(i+1)
                # Up from j goes to j+2, mid to j+1, down to j
                idx_up = j + 2
                idx_mid = j + 1
                idx_down = j

                # Get child values
                v_up = value_tree[i + 1, idx_up]
                v_mid = value_tree[i + 1, idx_mid]
                v_down = value_tree[i + 1, idx_down]

                # Get child probabilities
                cp_up = conv_prob_tree[i + 1, idx_up]
                cp_mid = conv_prob_tree[i + 1, idx_mid]
                cp_down = conv_prob_tree[i + 1, idx_down]

                dp_up = default_prob_tree[i + 1, idx_up]
                dp_mid = default_prob_tree[i + 1, idx_mid]
                dp_down = default_prob_tree[i + 1, idx_down]

                # Expected continuation value (weighted by survival probabilities)
                survival_prob = 1.0 - p_default
                continuation = discount * (
                    p_up * v_up + p_mid * v_mid + p_down * v_down
                    + p_default * recovery_value
                )

                # Expected conversion probability
                expected_conv_prob = (
                    p_up * cp_up + p_mid * cp_mid + p_down * cp_down
                ) / survival_prob if survival_prob > 0 else 0.0

                # Expected default probability (cumulative)
                expected_default_prob = (
                    p_default
                    + survival_prob
                    * (p_up * dp_up + p_mid * dp_mid + p_down * dp_down)
                    / survival_prob if survival_prob > 0 else p_default
                )

                # Add any coupon payments in this period
                for k, ct in enumerate(coupon_times):
                    if t < ct <= t + dt:
                        continuation += coupon_amounts[k] * safe_exp(-r * (ct - t))

                # Conversion value
                conversion_value = bond.parity(stock)

                # Check if convertible
                can_convert = bond.is_convertible_at(node_date)

                # Check for call
                can_call = bond.is_callable_at(node_date, stock)
                call_price = bond.get_call_price_at(node_date)

                # Check for put
                can_put = bond.is_puttable_at(node_date)
                put_price = bond.get_put_price_at(node_date)

                # Determine optimal action
                value = continuation
                is_converted = False

                # Holder's conversion right
                if can_convert and conversion_value > value:
                    value = conversion_value
                    is_converted = True

                # Issuer's call right
                if can_call and call_price is not None:
                    if value > call_price:
                        if can_convert and conversion_value > call_price:
                            value = conversion_value
                            is_converted = True
                        else:
                            value = call_price
                            is_converted = False

                # Holder's put right
                if can_put and put_price is not None:
                    if put_price > value:
                        value = put_price
                        is_converted = False

                value_tree[i, j] = value
                conv_prob_tree[i, j] = 1.0 if is_converted else expected_conv_prob
                default_prob_tree[i, j] = (
                    0.0 if is_converted else expected_default_prob
                )

        return {
            "value_tree": value_tree,
            "conv_prob_tree": conv_prob_tree,
            "default_prob_tree": default_prob_tree,
        }

    def _calculate_greeks(
        self,
        stock_tree: np.ndarray,
        value_tree: np.ndarray,
        spot: float,
        n_steps: int,
    ) -> Tuple[float, float]:
        """
        Calculate delta and gamma from the tree.

        Args:
            stock_tree: Stock price tree
            value_tree: Value tree
            spot: Initial spot price
            n_steps: Number of steps

        Returns:
            Tuple of (delta, gamma)
        """
        if n_steps < 2:
            return 0.0, 0.0

        # At time step 1, nodes are at indices 0, 1, 2 (down, mid, up)
        S_d = stock_tree[1, 0]
        S_m = stock_tree[1, 1]
        S_u = stock_tree[1, 2]
        V_d = value_tree[1, 0]
        V_m = value_tree[1, 1]
        V_u = value_tree[1, 2]

        # Delta using up and down nodes
        delta = (V_u - V_d) / (S_u - S_d)

        # Gamma from second derivative
        delta_up = (V_u - V_m) / (S_u - S_m)
        delta_down = (V_m - V_d) / (S_m - S_d)
        h = 0.5 * (S_u - S_d)
        gamma = (delta_up - delta_down) / h

        return delta, gamma

    def calculate_delta(self, bond: ConvertibleBond) -> float:
        """
        Calculate delta.

        Args:
            bond: Convertible bond

        Returns:
            Delta (price sensitivity to stock price)
        """
        result = self.price_with_details(bond)
        return result.delta

    def calculate_gamma(self, bond: ConvertibleBond) -> float:
        """
        Calculate gamma.

        Args:
            bond: Convertible bond

        Returns:
            Gamma (second derivative with respect to stock)
        """
        result = self.price_with_details(bond)
        return result.gamma

    def __repr__(self):
        return (
            f"ConvertibleBondTrinomialEngine("
            f"num_steps={self.params.num_steps})"
        )
