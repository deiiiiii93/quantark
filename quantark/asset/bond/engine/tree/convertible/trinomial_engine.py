"""
Trinomial tree engine for convertible bond pricing.

Implements the Hull-White trinomial model with explicit default branch,
where at each node there's a probability of default leading to recovery
value.
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
from quantark.util.enum.engine_enums import ConvertibleBondTrinomialVolScheme
from quantark.util.exceptions import ValidationError, PricingError, NumericalError
from quantark.util.numerical import (
    safe_exp,
    safe_sqrt,
    safe_log,
    safe_divide,
    is_zero,
    is_close,
    Tolerance,
)

logger = logging.getLogger(__name__)


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

    Volatility schemes:
    - CONSTANT_VOL: CRR-style constant-vol grid (ignores term structure)
    - LOG_FIXED_DX: fixed log-price step with time-dependent probabilities
    - LOG_VARIABLE_DX: variable log-price step with re-gridding/interpolation
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

    def _warn_if_term_structure_ignored(self, bond: ConvertibleBond) -> None:
        """
        Warn if a non-flat volatility surface is used with constant-vol scheme.
        """
        if (
            self.params.trinomial_vol_scheme
            != ConvertibleBondTrinomialVolScheme.CONSTANT_VOL
        ):
            return

        if self._vol_surface_has_term_structure(bond):
            logger.warning(
                "Trinomial constant-vol scheme does not apply volatility term "
                "structure. Select LOG_FIXED_DX or LOG_VARIABLE_DX for "
                "time-dependent volatility."
            )

    def _vol_surface_has_term_structure(self, bond: ConvertibleBond) -> bool:
        """
        Detect whether the volatility surface varies over time at-the-money.
        """
        if self.pricing_env.vol_surface is None:
            return False

        valuation_date = self.pricing_env.valuation_date
        T = bond.time_to_maturity(valuation_date)
        if is_zero(T) or T < 0:
            return False

        spot = self.pricing_env.spot
        sample_times = np.linspace(0.01, T, num=max(3, self.params.num_steps // 50))
        ref_vol = self.pricing_env.get_vol(spot, sample_times[0])

        for t in sample_times[1:]:
            vol = self.pricing_env.get_vol(spot, t)
            if not is_close(vol, ref_vol):
                return True
        return False

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

    def _calculate_max_vol_for_grid(self, bond: ConvertibleBond) -> float:
        """
        Find maximum volatility over bond life for stable grid construction.

        Using max vol for grid spacing (u, d factors) ensures transition
        probabilities remain valid even when vol varies over time.

        Args:
            bond: Convertible bond

        Returns:
            Maximum volatility to use for dx calculation
        """
        T = bond.time_to_maturity(self.pricing_env.valuation_date)
        spot = self.pricing_env.spot

        # Sample volatility at multiple time points
        num_samples = max(10, self.params.num_steps // 10)
        times = np.linspace(0.01, T, num_samples)  # Avoid t=0

        max_vol = 0.0
        for t in times:
            vol = self.pricing_env.get_vol(spot, t)
            max_vol = max(max_vol, vol)

        return max_vol

    def _calculate_max_step_volatility(
        self, bond: ConvertibleBond, dt: float
    ) -> float:
        """
        Find maximum step volatility using forward variance over each time step.

        Args:
            bond: Convertible bond
            dt: Time step size

        Returns:
            Maximum step volatility to use for log-price spacing
        """
        T = bond.time_to_maturity(self.pricing_env.valuation_date)
        if dt <= 0:
            raise ValidationError("Time step must be positive")

        n_steps = self.params.num_steps
        spot = self.pricing_env.spot
        max_vol = 0.0

        for i in range(n_steps):
            t_start = i * dt
            t_end = min(T, t_start + dt)
            vol = self.pricing_env.get_step_volatility(spot, t_start, t_end)
            max_vol = max(max_vol, vol)

        return max_vol

    def _precompute_log_step_params(
        self, bond: ConvertibleBond, q: float, dt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Precompute step-local parameters for log-price trinomial schemes.

        Returns arrays for each time step i:
        - r_steps[i]: forward rate over [t_i, t_{i+1}]
        - vol_steps[i]: effective step volatility over [t_i, t_{i+1}]
        - alpha_steps[i]: deterministic log-drift over the step
        - drift_cum[i]: cumulative log-drift up to time t_i
        """
        n_steps = self.params.num_steps
        spot = self.pricing_env.spot

        r_steps = np.zeros(n_steps)
        vol_steps = np.zeros(n_steps)
        alpha_steps = np.zeros(n_steps)
        drift_cum = np.zeros(n_steps + 1)

        for i in range(n_steps):
            t = i * dt
            t_next = t + dt
            r_local = self.pricing_env.rate_curve.get_forward_rate(t, t_next)
            vol_local = self.pricing_env.get_step_volatility(spot, t, t_next)

            r_steps[i] = r_local
            vol_steps[i] = vol_local

            alpha = (r_local - q - 0.5 * vol_local * vol_local) * dt
            alpha_steps[i] = alpha
            drift_cum[i + 1] = drift_cum[i] + alpha

        return r_steps, vol_steps, alpha_steps, drift_cum

    def _calculate_variance_probabilities(
        self, vol: float, dt: float, dx: float
    ) -> Tuple[float, float, float]:
        """
        Calculate symmetric trinomial probabilities matching step variance.

        Uses a drift-in-node construction: the deterministic log drift is
        embedded in the lattice, and probabilities only match variance.
        """
        if is_zero(dx):
            raise NumericalError("Log-price step size dx must be positive")

        p = (vol * vol * dt) / (2.0 * dx * dx)
        if p < -Tolerance.PRECISION or p > 0.5 + Tolerance.PRECISION:
            raise NumericalError(
                "Invalid variance-matching probability; consider increasing "
                "num_steps or switching scheme."
            )

        p = max(0.0, min(0.5, p))
        p_mid = 1.0 - 2.0 * p
        return p, p_mid, p

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

        # Calculate time parameters
        dt = T / self.params.num_steps
        if dt <= 0:
            raise PricingError("Time step must be positive for trinomial engine")

        q = bond.continuous_dividend_yield
        hazard_rate = bond.hazard_rate
        scheme = self.params.trinomial_vol_scheme

        # Warn if the selected scheme ignores term structure
        self._warn_if_term_structure_ignored(bond)

        log_spot = None
        dx_grid = None
        drift_cum = None
        alpha_steps = None
        dx_steps = None

        if scheme == ConvertibleBondTrinomialVolScheme.CONSTANT_VOL:
            # Use maximum volatility over bond life for stable grid construction
            max_vol = self._calculate_max_vol_for_grid(bond)

            # Calculate u, d factors using max_vol for grid stability
            stretch = math.sqrt(3.0)
            u = safe_exp(max_vol * stretch * safe_sqrt(dt))
            d = 1.0 / u

            # Build stock price tree
            stock_tree = self._build_stock_tree(
                spot, u, d, self.params.num_steps
            )

            # Build value tree with backward induction
            results = self._backward_induction_constant(
                bond, stock_tree, u, d, hazard_rate, q, dt
            )
        elif scheme == ConvertibleBondTrinomialVolScheme.LOG_FIXED_DX:
            r_steps, vol_steps, alpha_steps, drift_cum = self._precompute_log_step_params(
                bond, q, dt
            )
            max_vol = float(np.max(vol_steps))
            dx = max_vol * safe_sqrt(3.0 * dt)
            if is_zero(dx):
                raise NumericalError("Fixed log-price step size dx must be positive")

            n_steps = self.params.num_steps
            dx_grid = np.full(n_steps + 1, dx)
            stock_tree, _, log_spot = self._build_log_stock_tree(
                spot, dx_grid, drift_cum, n_steps
            )

            results = self._backward_induction_log_fixed_dx(
                bond, stock_tree, dx, hazard_rate, dt, r_steps, vol_steps
            )
        elif scheme == ConvertibleBondTrinomialVolScheme.LOG_VARIABLE_DX:
            n_steps = self.params.num_steps
            r_steps, vol_steps, alpha_steps, drift_cum = self._precompute_log_step_params(
                bond, q, dt
            )

            dx_steps = vol_steps * safe_sqrt(3.0 * dt)
            if np.any(dx_steps <= Tolerance.ZERO):
                raise NumericalError(
                    "Step volatility produced non-positive dx in LOG_VARIABLE_DX scheme"
                )

            dx_grid = np.zeros(n_steps + 1)
            dx_grid[0] = dx_steps[0]
            if n_steps > 1:
                dx_grid[1:n_steps] = dx_steps[1:]
            dx_grid[n_steps] = dx_steps[-1]

            stock_tree, log_grid, log_spot = self._build_log_stock_tree(
                spot, dx_grid, drift_cum, n_steps
            )
            results = self._backward_induction_log_variable_dx(
                bond,
                stock_tree,
                log_grid,
                log_spot,
                dx_steps,
                dx_grid,
                drift_cum,
                hazard_rate,
                dt,
                r_steps,
                vol_steps,
                alpha_steps,
            )
        else:
            raise ValidationError(
                f"Unsupported trinomial volatility scheme: {scheme}"
            )

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
        if scheme == ConvertibleBondTrinomialVolScheme.LOG_VARIABLE_DX:
            assert log_spot is not None
            assert dx_grid is not None
            assert drift_cum is not None
            assert dx_steps is not None

            log_mid = log_spot + drift_cum[1]
            log_up = log_mid + float(dx_steps[0])
            log_down = log_mid - float(dx_steps[0])

            v_up = self._interpolate_tree_value(
                value_tree, log_spot, dx_grid, drift_cum, 1, log_up
            )
            v_mid = self._interpolate_tree_value(
                value_tree, log_spot, dx_grid, drift_cum, 1, log_mid
            )
            v_down = self._interpolate_tree_value(
                value_tree, log_spot, dx_grid, drift_cum, 1, log_down
            )

            s_up = safe_exp(log_up)
            s_mid = safe_exp(log_mid)
            s_down = safe_exp(log_down)

            delta = (v_up - v_down) / (s_up - s_down)
            delta_up = (v_up - v_mid) / (s_up - s_mid)
            delta_down = (v_mid - v_down) / (s_mid - s_down)
            h = 0.5 * (s_up - s_down)
            gamma = (delta_up - delta_down) / h
        else:
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

    def _build_log_stock_tree(
        self,
        spot: float,
        dx_grid: np.ndarray,
        drift_cum: np.ndarray,
        n_steps: int,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Build a log-price stock tree with per-time-step spacing.

        Args:
            spot: Initial stock price
            dx_grid: Log-price step size per time level
            drift_cum: Cumulative deterministic log drift per time level
            n_steps: Number of time steps

        Returns:
            Tuple of (stock_tree, log_grid, log_spot)
        """
        if len(drift_cum) != n_steps + 1:
            raise ValidationError(
                "drift_cum length must equal n_steps + 1 for log tree construction"
            )

        max_nodes = 2 * n_steps + 1
        stock_tree = np.zeros((n_steps + 1, max_nodes))
        log_grid = np.zeros((n_steps + 1, max_nodes))
        log_spot = safe_log(spot)

        for i in range(n_steps + 1):
            num_nodes = 2 * i + 1
            dx = dx_grid[i]
            for j in range(num_nodes):
                log_s = log_spot + drift_cum[i] + (j - i) * dx
                log_grid[i, j] = log_s
                stock_tree[i, j] = safe_exp(log_s)

        return stock_tree, log_grid, log_spot

    def _interpolate_tree_value(
        self,
        values: np.ndarray,
        log_spot: float,
        dx_grid: np.ndarray,
        drift_cum: np.ndarray,
        time_index: int,
        log_s: float,
    ) -> float:
        """
        Linearly interpolate a tree value at a given log-price.
        """
        dx = dx_grid[time_index]
        if is_zero(dx):
            raise NumericalError("Log-price grid spacing must be positive")

        max_idx = 2 * time_index
        center = log_spot + drift_cum[time_index]
        idx_float = safe_divide(log_s - center, dx, fallback=0.0) + time_index

        if idx_float <= 0.0:
            return values[time_index, 0]
        if idx_float >= max_idx:
            return values[time_index, max_idx]

        idx_low = int(math.floor(idx_float))
        idx_high = idx_low + 1
        weight_high = idx_float - idx_low
        weight_low = 1.0 - weight_high

        return (
            values[time_index, idx_low] * weight_low
            + values[time_index, idx_high] * weight_high
        )

    def _backward_induction_constant(
        self,
        bond: ConvertibleBond,
        stock_tree: np.ndarray,
        u: float,
        d: float,
        hazard_rate: float,
        q: float,
        dt: float,
    ) -> Dict[str, np.ndarray]:
        """
        Perform backward induction with constant-vol CRR-style probabilities.

        Args:
            bond: Convertible bond
            stock_tree: Pre-built stock price tree
            u: Up factor (constant, based on max_vol)
            d: Down factor (constant, based on max_vol)
            hazard_rate: Default intensity
            q: Dividend yield
            dt: Time step

        Returns:
            Dictionary containing value_tree, conv_prob_tree, default_prob_tree
        """
        n_steps = self.params.num_steps
        valuation_date = self.pricing_env.valuation_date
        spot = self.pricing_env.spot
        max_nodes = 2 * n_steps + 1
        T = bond.time_to_maturity(valuation_date)

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

        # Backward induction with time-local parameters
        for i in range(n_steps - 1, -1, -1):
            t = i * dt  # Current time (years from valuation)
            t_next = t + dt  # End of this step
            node_date = valuation_date + timedelta(days=int(t * 365))
            num_nodes = 2 * i + 1

            # Query time-local forward rate for this step
            r_local = self.pricing_env.rate_curve.get_forward_rate(t, t_next)

            # Calculate time-local transition probabilities
            # Note: u, d remain constant (based on max_vol) to maintain recombination
            # Probabilities only adjust for local rate under constant-vol scheme
            p_default = 1.0 - safe_exp(-hazard_rate * dt)
            p_survive = 1.0 - p_default

            # Risk-neutral probabilities using local rate
            exp_growth = safe_exp((r_local - q) * dt)
            p_up_raw = ((exp_growth - d) / (u - d)) * 0.5
            p_down_raw = ((u - exp_growth) / (u - d)) * 0.5
            p_mid_raw = 1.0 - p_up_raw - p_down_raw

            # Ensure valid probabilities
            p_up_raw = max(0.0, min(1.0, p_up_raw))
            p_down_raw = max(0.0, min(1.0, p_down_raw))
            p_mid_raw = max(0.0, 1.0 - p_up_raw - p_down_raw)

            # Renormalize
            total = p_up_raw + p_mid_raw + p_down_raw
            if total > 0:
                p_up_raw /= total
                p_mid_raw /= total
                p_down_raw /= total

            # Scale by survival probability
            p_up = p_up_raw * p_survive
            p_mid = p_mid_raw * p_survive
            p_down = p_down_raw * p_survive

            # Local discount factor
            discount_local = safe_exp(-r_local * dt)

            for j in range(num_nodes):
                stock = stock_tree[i, j]

                # Map to next layer's indices
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

                # Expected continuation value with local discount
                continuation = discount_local * (
                    p_up * v_up + p_mid * v_mid + p_down * v_down
                    + p_default * recovery_value
                )

                # Expected conversion probability
                expected_conv_prob = (
                    p_up_raw * cp_up + p_mid_raw * cp_mid + p_down_raw * cp_down
                ) if p_survive > 0 else 0.0

                # Expected default probability (cumulative)
                expected_default_prob = (
                    p_default
                    + p_survive
                    * (p_up_raw * dp_up + p_mid_raw * dp_mid + p_down_raw * dp_down)
                ) if p_survive > 0 else p_default

                # Add any coupon payments in this period
                for k, ct in enumerate(coupon_times):
                    if t < ct <= t_next:
                        continuation += coupon_amounts[k] * safe_exp(-r_local * (ct - t))

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

    def _backward_induction_log_fixed_dx(
        self,
        bond: ConvertibleBond,
        stock_tree: np.ndarray,
        dx: float,
        hazard_rate: float,
        dt: float,
        r_steps: np.ndarray,
        vol_steps: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Perform backward induction with fixed-dx log-price probabilities.

        Args:
            bond: Convertible bond
            stock_tree: Pre-built stock price tree
            dx: Log-price step size
            hazard_rate: Default intensity
            dt: Time step
            r_steps: Forward rate per step
            vol_steps: Step volatility per step

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

        # Backward induction with time-local parameters
        for i in range(n_steps - 1, -1, -1):
            t = i * dt  # Current time (years from valuation)
            t_next = t + dt  # End of this step
            node_date = valuation_date + timedelta(days=int(t * 365))
            num_nodes = 2 * i + 1

            r_local = float(r_steps[i])
            vol_local = float(vol_steps[i])

            # Calculate time-local transition probabilities
            p_default = 1.0 - safe_exp(-hazard_rate * dt)
            p_survive = 1.0 - p_default

            p_up_raw, p_mid_raw, p_down_raw = self._calculate_variance_probabilities(
                vol_local, dt, dx
            )

            # Scale by survival probability
            p_up = p_up_raw * p_survive
            p_mid = p_mid_raw * p_survive
            p_down = p_down_raw * p_survive

            # Local discount factor
            discount_local = safe_exp(-r_local * dt)

            for j in range(num_nodes):
                stock = stock_tree[i, j]

                # Map to next layer's indices
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

                # Expected continuation value with local discount
                continuation = discount_local * (
                    p_up * v_up
                    + p_mid * v_mid
                    + p_down * v_down
                    + p_default * recovery_value
                )

                # Expected conversion probability
                expected_conv_prob = (
                    p_up_raw * cp_up + p_mid_raw * cp_mid + p_down_raw * cp_down
                ) if p_survive > 0 else 0.0

                # Expected default probability (cumulative)
                expected_default_prob = (
                    p_default
                    + p_survive
                    * (p_up_raw * dp_up + p_mid_raw * dp_mid + p_down_raw * dp_down)
                ) if p_survive > 0 else p_default

                # Add any coupon payments in this period
                for k, ct in enumerate(coupon_times):
                    if t < ct <= t_next:
                        continuation += coupon_amounts[k] * safe_exp(
                            -r_local * (ct - t)
                        )

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

    def _backward_induction_log_variable_dx(
        self,
        bond: ConvertibleBond,
        stock_tree: np.ndarray,
        log_grid: np.ndarray,
        log_spot: float,
        dx_steps: np.ndarray,
        dx_grid: np.ndarray,
        drift_cum: np.ndarray,
        hazard_rate: float,
        dt: float,
        r_steps: np.ndarray,
        vol_steps: np.ndarray,
        alpha_steps: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Perform backward induction with variable-dx log-price re-gridding.

        Args:
            bond: Convertible bond
            stock_tree: Pre-built stock price tree
            log_grid: Log-price grid per time step
            log_spot: Log of initial spot
            dx_steps: Log step size per time interval
            dx_grid: Log step size per time level
            drift_cum: Cumulative deterministic log drift per time level
            hazard_rate: Default intensity
            dt: Time step
            r_steps: Forward rate per step
            vol_steps: Step volatility per step
            alpha_steps: Deterministic log drift per step

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

        # Backward induction with time-local parameters
        for i in range(n_steps - 1, -1, -1):
            t = i * dt  # Current time (years from valuation)
            t_next = t + dt  # End of this step
            node_date = valuation_date + timedelta(days=int(t * 365))
            num_nodes = 2 * i + 1

            r_local = float(r_steps[i])
            vol_local = float(vol_steps[i])
            alpha = float(alpha_steps[i])

            # Calculate time-local transition probabilities
            p_default = 1.0 - safe_exp(-hazard_rate * dt)
            p_survive = 1.0 - p_default

            dx_step = float(dx_steps[i])
            p_up_raw, p_mid_raw, p_down_raw = self._calculate_variance_probabilities(
                vol_local, dt, dx_step
            )

            # Scale by survival probability
            p_up = p_up_raw * p_survive
            p_mid = p_mid_raw * p_survive
            p_down = p_down_raw * p_survive

            # Local discount factor
            discount_local = safe_exp(-r_local * dt)

            for j in range(num_nodes):
                stock = stock_tree[i, j]
                log_s = log_grid[i, j]

                # Child log prices for this step
                log_mid = log_s + alpha
                log_up = log_mid + dx_step
                log_down = log_mid - dx_step

                # Interpolate child values on the next grid
                v_up = self._interpolate_tree_value(
                    value_tree, log_spot, dx_grid, drift_cum, i + 1, log_up
                )
                v_mid = self._interpolate_tree_value(
                    value_tree, log_spot, dx_grid, drift_cum, i + 1, log_mid
                )
                v_down = self._interpolate_tree_value(
                    value_tree, log_spot, dx_grid, drift_cum, i + 1, log_down
                )

                cp_up = self._interpolate_tree_value(
                    conv_prob_tree, log_spot, dx_grid, drift_cum, i + 1, log_up
                )
                cp_mid = self._interpolate_tree_value(
                    conv_prob_tree, log_spot, dx_grid, drift_cum, i + 1, log_mid
                )
                cp_down = self._interpolate_tree_value(
                    conv_prob_tree, log_spot, dx_grid, drift_cum, i + 1, log_down
                )

                dp_up = self._interpolate_tree_value(
                    default_prob_tree, log_spot, dx_grid, drift_cum, i + 1, log_up
                )
                dp_mid = self._interpolate_tree_value(
                    default_prob_tree, log_spot, dx_grid, drift_cum, i + 1, log_mid
                )
                dp_down = self._interpolate_tree_value(
                    default_prob_tree, log_spot, dx_grid, drift_cum, i + 1, log_down
                )

                # Expected continuation value with local discount
                continuation = discount_local * (
                    p_up * v_up
                    + p_mid * v_mid
                    + p_down * v_down
                    + p_default * recovery_value
                )

                # Expected conversion probability
                expected_conv_prob = (
                    p_up_raw * cp_up + p_mid_raw * cp_mid + p_down_raw * cp_down
                ) if p_survive > 0 else 0.0

                # Expected default probability (cumulative)
                expected_default_prob = (
                    p_default
                    + p_survive
                    * (p_up_raw * dp_up + p_mid_raw * dp_mid + p_down_raw * dp_down)
                ) if p_survive > 0 else p_default

                # Add any coupon payments in this period
                for k, ct in enumerate(coupon_times):
                    if t < ct <= t_next:
                        continuation += coupon_amounts[k] * safe_exp(
                            -r_local * (ct - t)
                        )

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
            f"num_steps={self.params.num_steps}, "
            f"vol_scheme={self.params.trinomial_vol_scheme})"
        )
