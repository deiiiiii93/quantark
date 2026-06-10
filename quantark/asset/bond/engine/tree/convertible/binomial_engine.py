"""
Binomial tree engine for convertible bond pricing.

Implements the Goldman Sachs credit-adjusted binomial model where the discount
rate at each node is adjusted based on the probability of conversion vs.
continuing as debt.
"""
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import numpy as np

from quantark.asset.bond.product.convertible.convertible_bond import ConvertibleBond
from quantark.asset.bond.engine.tree.convertible.tree_params import ConvertibleBondTreeParams
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import ValidationError, PricingError
from quantark.util.numerical import safe_exp, safe_sqrt, is_zero

logger = logging.getLogger(__name__)


@dataclass
class ConvertibleBondBinomialResult:
    """
    Result container for binomial engine pricing.

    Attributes:
        price: Clean price of the convertible bond
        dirty_price: Dirty price including accrued interest
        conversion_probability: Probability of eventual conversion
        delta: Price sensitivity to stock price
        gamma: Second derivative of price with respect to stock
        equity_component: Equity-like component of value
        bond_component: Bond-like component of value
    """

    price: float
    dirty_price: float
    conversion_probability: float
    delta: float = 0.0
    gamma: float = 0.0
    equity_component: float = 0.0
    bond_component: float = 0.0


class ConvertibleBondBinomialEngine:
    """
    Goldman Sachs credit-adjusted binomial model for convertible bonds.

    This engine implements a binomial tree where:
    1. Stock price evolves according to GBM with up/down moves
    2. At each node, the discount rate is credit-adjusted based on
       conversion probability
    3. Early exercise (conversion, call, put) is handled via backward induction

    The credit-adjusted discount rate at each node is:
        y = p * r + (1 - p) * (r + credit_spread)

    where:
        p = probability of conversion (equity-like outcome)
        r = risk-free rate
        credit_spread = issuer credit spread

    This reflects that cash flows are discounted at the risk-free rate when
    conversion is likely, and at the credit-risky rate when bond redemption
    is likely.
    """

    def __init__(
        self,
        pricing_env: PricingEnvironment,
        params: Optional[ConvertibleBondTreeParams] = None,
    ):
        """
        Initialize the binomial engine.

        Args:
            pricing_env: Pricing environment with market data
            params: Tree configuration parameters (optional)
        """
        if pricing_env is None:
            raise ValidationError("Pricing environment is required")

        self.pricing_env = pricing_env
        self.params = params if params is not None else ConvertibleBondTreeParams()

    def _warn_if_non_flat_curves(self) -> None:
        """
        Log warning if volatility surface is non-flat.

        This binomial implementation supports non-flat risk-free curves via
        per-step forward rates. It still assumes constant volatility (uses
        ATM volatility to maturity), and does not apply volatility term structure.
        """
        from quantark.param.vol.vol_surface import FlatVolSurface

        vol_surface = self.pricing_env.vol_surface

        if vol_surface is not None and not isinstance(vol_surface, FlatVolSurface):
            logger.warning(
                "Binomial GS engine supports non-flat risk-free curves, but "
                "approximates volatility term structure using ATM vol to maturity. "
                "Use PDE or Trinomial engines for better accuracy under vol term structure."
            )

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
    ) -> ConvertibleBondBinomialResult:
        """
        Calculate price with detailed results including Greeks and components.

        Args:
            bond: Convertible bond to price

        Returns:
            ConvertibleBondBinomialResult with full pricing details
        """
        # Warn if non-flat curves are detected
        self._warn_if_non_flat_curves()

        valuation_date = self.pricing_env.valuation_date

        # Validate inputs
        if bond.is_expired(valuation_date):
            raise PricingError("Cannot price expired bond")

        # Get market data
        spot = self.pricing_env.spot
        T = bond.time_to_maturity(valuation_date)
        # Use ATM volatility (strike = spot)
        vol = self.pricing_env.get_vol(spot, T)

        # Calculate time parameters
        dt = T / self.params.num_steps

        # Build tree parameters
        # Use per-step forward rates for piecewise curve support
        r_steps = np.zeros(self.params.num_steps)
        for i in range(self.params.num_steps):
            t = i * dt
            r_steps[i] = self.pricing_env.rate_curve.get_forward_rate(t, t + dt)

        q = bond.continuous_dividend_yield  # Dividend yield from bond

        # CRR binomial parameters
        u, d, _ = self._calculate_tree_params(vol, r_steps[0], q, dt)

        # Build stock price tree
        stock_tree = self._build_stock_tree(spot, u, d, self.params.num_steps)

        # Build value tree with backward induction
        value_tree, conv_prob_tree = self._backward_induction(
            bond, stock_tree, u, d, r_steps, q, dt
        )

        # Extract results
        dirty_price = value_tree[0, 0]
        accrued = bond.calculate_accrued_interest(valuation_date)
        clean_price = dirty_price - accrued

        # Calculate conversion probability at root
        conv_prob = conv_prob_tree[0, 0]

        # Calculate delta and gamma from tree
        delta, gamma = self._calculate_greeks(
            stock_tree, value_tree, spot, self.params.num_steps
        )

        # Estimate equity and bond components
        equity_component = conv_prob * dirty_price
        bond_component = (1 - conv_prob) * dirty_price

        return ConvertibleBondBinomialResult(
            price=clean_price,
            dirty_price=dirty_price,
            conversion_probability=conv_prob,
            delta=delta,
            gamma=gamma,
            equity_component=equity_component,
            bond_component=bond_component,
        )

    def _calculate_tree_params(
        self, vol: float, r: float, q: float, dt: float
    ) -> Tuple[float, float, float]:
        """
        Calculate binomial tree parameters using Cox-Ross-Rubinstein (CRR).

        Args:
            vol: Volatility
            r: Risk-free rate
            q: Dividend yield
            dt: Time step

        Returns:
            Tuple of (u, d, p_up) - up factor, down factor, up probability
        """
        # CRR parameters
        u = safe_exp(vol * safe_sqrt(dt))
        d = 1.0 / u

        # Risk-neutral probability
        drift = safe_exp((r - q) * dt)
        p_up = (drift - d) / (u - d)

        # Ensure probability is valid
        if p_up < 0 or p_up > 1:
            raise PricingError(
                f"Invalid risk-neutral probability: {p_up}. "
                "Check volatility and rate inputs."
            )

        return u, d, p_up

    def _build_stock_tree(
        self, spot: float, u: float, d: float, n_steps: int
    ) -> np.ndarray:
        """
        Build recombining binomial stock price tree.

        Args:
            spot: Initial stock price
            u: Up factor
            d: Down factor
            n_steps: Number of time steps

        Returns:
            2D array where tree[i, j] = stock price at time i, node j
        """
        tree = np.zeros((n_steps + 1, n_steps + 1))

        for i in range(n_steps + 1):
            for j in range(i + 1):
                # j up moves, (i - j) down moves
                tree[i, j] = spot * (u ** j) * (d ** (i - j))

        return tree

    def _backward_induction(
        self,
        bond: ConvertibleBond,
        stock_tree: np.ndarray,
        u: float,
        d: float,
        r_steps: np.ndarray,
        q: float,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform backward induction to compute option values.

        Implements the GS credit-adjusted discount rate model.

        Args:
            bond: Convertible bond
            stock_tree: Pre-built stock price tree
            u: Up factor (constant)
            d: Down factor (constant)
            r_steps: Forward rate per step
            q: Dividend yield
            dt: Time step

        Returns:
            Tuple of (value_tree, conversion_probability_tree)
        """
        n_steps = self.params.num_steps
        valuation_date = self.pricing_env.valuation_date

        # Initialize value and conversion probability trees
        value_tree = np.zeros((n_steps + 1, n_steps + 1))
        conv_prob_tree = np.zeros((n_steps + 1, n_steps + 1))

        # Get coupon schedule for intermediate payments
        all_cashflows = bond.get_all_cashflows()
        coupon_times = []
        coupon_amounts = []
        for cf in all_cashflows:
            cf_time = (cf.payment_date - valuation_date).days / 365.0
            if cf_time > 0:
                coupon_times.append(cf_time)
                # Only coupon, not principal (which is handled at maturity)
                coupon_amount = cf.amount
                if cf.payment_date >= bond.maturity_date:
                    coupon_amount -= bond.face_value  # Subtract principal
                coupon_amounts.append(max(0, coupon_amount))

        T = bond.time_to_maturity(valuation_date)

        # Terminal values at maturity
        for j in range(n_steps + 1):
            stock = stock_tree[n_steps, j]
            conversion_value = bond.parity(stock)

            # At maturity: max(face_value, conversion_value)
            if conversion_value > bond.face_value:
                value_tree[n_steps, j] = conversion_value
                conv_prob_tree[n_steps, j] = 1.0
            else:
                value_tree[n_steps, j] = bond.face_value
                conv_prob_tree[n_steps, j] = 0.0

        # Backward induction
        for i in range(n_steps - 1, -1, -1):
            t = i * dt
            node_date = valuation_date + timedelta(days=int(t * 365))
            r_local = float(r_steps[i])

            drift = safe_exp((r_local - q) * dt)
            p_up = (drift - d) / (u - d)
            if p_up < 0 or p_up > 1:
                raise PricingError(
                    f"Invalid risk-neutral probability at step {i}: {p_up}. "
                    "Check volatility, rate curve, and num_steps."
                )

            for j in range(i + 1):
                stock = stock_tree[i, j]

                # Expected continuation value (risk-neutral)
                v_up = value_tree[i + 1, j + 1]
                v_down = value_tree[i + 1, j]
                p_up_child = conv_prob_tree[i + 1, j + 1]
                p_down_child = conv_prob_tree[i + 1, j]

                # Expected conversion probability of children
                expected_conv_prob = p_up * p_up_child + (1 - p_up) * p_down_child

                # Credit-adjusted discount rate (GS model)
                credit_spread = bond.credit_spread
                discount_rate = (
                    expected_conv_prob * r_local
                    + (1 - expected_conv_prob) * (r_local + credit_spread)
                )

                # Discount continuation value
                continuation = safe_exp(-discount_rate * dt) * (
                    p_up * v_up + (1 - p_up) * v_down
                )

                # Add any coupon payments in this period
                for k, ct in enumerate(coupon_times):
                    if t < ct <= t + dt:
                        continuation += coupon_amounts[k] * safe_exp(
                            -discount_rate * (ct - t)
                        )

                # Conversion value
                conversion_value = bond.parity(stock)

                # Check if convertible at this node
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

                # Issuer's call right (forces holder to choose conversion or call price)
                if can_call and call_price is not None:
                    if value > call_price:
                        # Issuer calls; holder chooses max(conversion, call)
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

        return value_tree, conv_prob_tree

    def _calculate_greeks(
        self,
        stock_tree: np.ndarray,
        value_tree: np.ndarray,
        spot: float,
        n_steps: int,
    ) -> Tuple[float, float]:
        """
        Calculate delta and gamma from the tree.

        Uses the first two time steps for finite difference approximation.

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

        # At time step 1
        S_u = stock_tree[1, 1]
        S_d = stock_tree[1, 0]
        V_u = value_tree[1, 1]
        V_d = value_tree[1, 0]

        # Delta from step 1
        delta = (V_u - V_d) / (S_u - S_d)

        # For gamma, use step 2
        if n_steps >= 2:
            S_uu = stock_tree[2, 2]
            S_ud = stock_tree[2, 1]
            S_dd = stock_tree[2, 0]
            V_uu = value_tree[2, 2]
            V_ud = value_tree[2, 1]
            V_dd = value_tree[2, 0]

            delta_up = (V_uu - V_ud) / (S_uu - S_ud)
            delta_down = (V_ud - V_dd) / (S_ud - S_dd)

            h = 0.5 * (S_uu - S_dd)
            gamma = (delta_up - delta_down) / h

        return delta, gamma

    def calculate_delta(self, bond: ConvertibleBond) -> float:
        """
        Calculate delta using finite difference.

        Args:
            bond: Convertible bond

        Returns:
            Delta (price sensitivity to stock price)
        """
        result = self.price_with_details(bond)
        return result.delta

    def calculate_gamma(self, bond: ConvertibleBond) -> float:
        """
        Calculate gamma using finite difference.

        Args:
            bond: Convertible bond

        Returns:
            Gamma (second derivative with respect to stock)
        """
        result = self.price_with_details(bond)
        return result.gamma

    def __repr__(self):
        return (
            f"ConvertibleBondBinomialEngine("
            f"num_steps={self.params.num_steps})"
        )
