"""
Tsiveriotis-Fernandes binomial tree engine for convertible bond pricing.

The lattice twin of the TF PDE (``ConvertibleBondTFEngine``): each node
carries the total value V and its cash-only component B (COCB), discounting
B at the risky rate r + credit_spread and the equity part V - B at r.
"""
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import brentq

from quantark.asset.bond.engine.tree.convertible.tree_params import ConvertibleBondTreeParams
from quantark.asset.bond.product.convertible.convertible_bond import ConvertibleBond
from quantark.asset.bond.schedule.put_schedule import put_prices_by_step
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import PricingError, ValidationError
from quantark.util.numerical import Tolerance, is_close, safe_exp, safe_sqrt


@dataclass
class ConvertibleBondTFTreeResult:
    """
    Result container for the TF binomial tree.

    Attributes:
        price: Clean price
        dirty_price: Dirty price (total value V at the root)
        equity_component: Equity component V - B at the root (discounted at r)
        bond_component: Cash-only component B at the root (COCB, discounted at r + spread)
        delta: Price sensitivity to stock price (from steps 1 and 2)
        gamma: Second derivative with respect to stock price
        conversion_probability: Risk-neutral probability of eventual conversion
    """

    price: float
    dirty_price: float
    equity_component: float
    bond_component: float
    delta: float = 0.0
    gamma: float = 0.0
    conversion_probability: float = 0.0


class ConvertibleBondTFTreeEngine:
    """
    Tsiveriotis-Fernandes (1998) convertible bond model on a CRR binomial tree.

    Each node holds the total value V and the cash-only part B. Holding over a
    step of length dt_i:

        B_hold = exp(-(r_i + s) dt_i) E[B] + coupons in the step (risky-discounted)
        V_hold = exp(-r_i dt_i) E[V - B] + B_hold

    Then, in the TF PDE's order: the holder converts (V = parity, B = 0), the
    issuer calls (holder takes parity or the call price), the holder puts
    (V = B = put price).

    Time steps are equal in total variance w(t) = sigma(spot, t)^2 t, so a vol
    term structure keeps u = exp(sqrt(w(T) / N)) constant and the tree
    recombining; flat vol gives equal steps in time. Rates enter per step as
    forward rates; q is the bond's continuous dividend yield.
    """

    def __init__(
        self,
        pricing_env: PricingEnvironment,
        params: Optional[ConvertibleBondTreeParams] = None,
    ):
        """
        Initialize the TF tree engine.

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
            Clean price
        """
        return self.price_with_details(bond).price

    def price_with_details(self, bond: ConvertibleBond) -> ConvertibleBondTFTreeResult:
        """
        Calculate price with the TF decomposition, Greeks and conversion probability.

        Args:
            bond: Convertible bond to price

        Returns:
            ConvertibleBondTFTreeResult
        """
        valuation_date = self.pricing_env.valuation_date
        if bond.is_expired(valuation_date):
            raise PricingError("Cannot price expired bond")

        spot = self.pricing_env.spot
        T = bond.time_to_maturity(valuation_date)
        n_steps = self.params.num_steps
        times = self._time_grid(T)
        u = safe_exp(safe_sqrt(self._total_variance(T) / n_steps))
        d = 1.0 / u

        q = bond.continuous_dividend_yield
        spread = bond.credit_spread
        ratio = bond.conversion_ratio
        coupons = self._coupon_schedule(bond)
        put_by_step = put_prices_by_step(bond, valuation_date, times)

        # Terminal: convert or redeem; a put in the last step is exercised at maturity
        stock = self._stock_level(spot, u, n_steps)
        parity = ratio * stock
        redemption = bond.maturity_redemption
        converted = (parity > redemption) & bond.is_convertible_at(bond.maturity_date)
        V = np.where(converted, parity, redemption)
        B = np.where(converted, 0.0, redemption)
        P = converted.astype(float)
        V, B, P = self._apply_put(V, B, P, put_by_step.get(n_steps))

        # (stock, value) at steps 0-2 for delta and gamma
        early: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        if n_steps <= 2:
            early[n_steps] = (stock, V)
        for i in range(n_steps - 1, -1, -1):
            t, t_next = times[i], times[i + 1]
            dt = t_next - t
            r = self.pricing_env.rate_curve.get_forward_rate(t, t_next)
            p = (safe_exp((r - q) * dt) - d) / (u - d)
            if p < 0.0 or p > 1.0:
                raise PricingError(
                    f"Invalid risk-neutral probability at step {i}: {p}. "
                    "Check volatility, rate curve and num_steps."
                )

            risky_df = safe_exp(-(r + spread) * dt)
            B_hold = risky_df * (p * B[1:] + (1.0 - p) * B[:-1])
            for coupon_time, amount in coupons:
                if t < coupon_time <= t_next:
                    B_hold = B_hold + amount * safe_exp(-(r + spread) * (coupon_time - t))
            equity = V - B
            V_hold = safe_exp(-r * dt) * (p * equity[1:] + (1.0 - p) * equity[:-1]) + B_hold
            P_hold = p * P[1:] + (1.0 - p) * P[:-1]

            stock = self._stock_level(spot, u, i)
            node_date = valuation_date + timedelta(days=int(t * 365))
            V, B, P = self._apply_decisions(
                bond, stock, node_date, V_hold, B_hold, P_hold, put_by_step.get(i)
            )
            if i <= 2:
                early[i] = (stock, V)

        dirty_price = float(V[0])
        bond_component = float(B[0])
        delta, gamma = self._calculate_greeks(early)
        accrued = bond.calculate_accrued_interest(valuation_date)

        return ConvertibleBondTFTreeResult(
            price=dirty_price - accrued,
            dirty_price=dirty_price,
            equity_component=dirty_price - bond_component,
            bond_component=bond_component,
            delta=delta,
            gamma=gamma,
            conversion_probability=float(np.clip(P[0], 0.0, 1.0)),
        )

    def _total_variance(self, t: float) -> float:
        """Total implied variance w(t) = sigma(spot, t)^2 t (strike = spot, as the TF PDE)."""
        if t <= 0.0:
            return 0.0
        vol = self.pricing_env.get_vol(self.pricing_env.spot, t)
        return vol * vol * t

    def _time_grid(self, T: float) -> np.ndarray:
        """
        Step times with equal total-variance increments, from 0 to T.

        Variance affine in t (flat vol) gives exactly t_i = i T / N; otherwise
        each t_i solves w(t_i) = i w(T) / N.

        Raises:
            ValidationError: If total variance is not increasing on [0, T]
        """
        n_steps = self.params.num_steps
        dt = T / n_steps
        uniform = np.arange(n_steps + 1) * dt
        uniform[-1] = T
        w_uniform = np.array([self._total_variance(t) for t in uniform])
        w_T = w_uniform[-1]

        if not np.all(np.diff(w_uniform) > 0.0):
            raise ValidationError(
                "TF tree needs total implied variance increasing in time; "
                "the volatility term structure has calendar arbitrage or zero vol"
            )
        if all(is_close(w, w_T * i / n_steps) for i, w in enumerate(w_uniform)):
            return uniform

        times = np.empty(n_steps + 1)
        times[0], times[-1] = 0.0, T
        for i in range(1, n_steps):
            target = w_T * i / n_steps
            # w is increasing, so the root lies between the uniform nodes that bracket target
            k = int(np.searchsorted(w_uniform, target))
            times[i] = brentq(
                lambda t: self._total_variance(t) - target,
                uniform[k - 1],
                uniform[k],
                xtol=Tolerance.ZERO * T,
                rtol=4 * np.finfo(float).eps,
            )
        return times

    @staticmethod
    def _stock_level(spot: float, u: float, step: int) -> np.ndarray:
        """Stock prices at a time step, node j having j up moves."""
        return spot * u ** (2.0 * np.arange(step + 1) - step)

    def _coupon_schedule(self, bond: ConvertibleBond) -> List[Tuple[float, float]]:
        """(time, amount) of future coupons paid within a step, on the TF PDE's clock."""
        valuation_date = self.pricing_env.valuation_date
        T = bond.time_to_maturity(valuation_date)
        schedule = []
        for cf in bond.get_all_cashflows():
            cf_time = bond.coupon_time(cf, valuation_date)
            if 0 < cf_time <= T:
                amount = bond.coupon_amount(cf)
                if amount > 0:
                    schedule.append((cf_time, amount))
        return schedule

    def _apply_decisions(
        self,
        bond: ConvertibleBond,
        stock: np.ndarray,
        node_date,
        V: np.ndarray,
        B: np.ndarray,
        P: np.ndarray,
        put_price: Optional[float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Conversion, call and put at one time step, in the TF PDE's order.

        Args:
            bond: Convertible bond
            stock: Stock prices at the step's nodes
            node_date: Calendar date of the step (whole days, as the TF PDE)
            V: Hold value
            B: Hold cash-only component
            P: Hold conversion probability
            put_price: Put price exercisable at this step, if any

        Returns:
            Tuple of (V, B, P) after the holder's and issuer's decisions
        """
        parity = bond.conversion_ratio * stock
        can_convert = bond.is_convertible_at(node_date)

        # Holder's conversion right
        if can_convert:
            convert = parity >= V - Tolerance.PRECISION
            V = np.where(convert, parity, V)
            B = np.where(convert, 0.0, B)
            P = np.where(convert, 1.0, P)

        # Issuer's call right: holder then takes parity if convertible and above the call price
        call_price = bond.get_call_price_at(node_date)
        if call_price is not None:
            if bond.is_callable_at(node_date):
                callable_nodes = np.ones_like(stock, dtype=bool)
            else:  # soft call: the trigger depends on the stock price
                callable_nodes = np.array([bond.is_callable_at(node_date, s) for s in stock])
            called = callable_nodes & (V > call_price)
            converts = called & can_convert & (parity > call_price)
            V = np.where(converts, parity, np.where(called, call_price, V))
            B = np.where(converts, 0.0, np.where(called, call_price, B))
            P = np.where(converts, 1.0, np.where(called, 0.0, P))

        return self._apply_put(V, B, P, put_price)

    @staticmethod
    def _apply_put(
        V: np.ndarray, B: np.ndarray, P: np.ndarray, put_price: Optional[float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Holder's put: V = B = put price where it beats holding (no-op without a put)."""
        if put_price is None:
            return V, B, P
        put = put_price > V
        return (
            np.where(put, put_price, V),
            np.where(put, put_price, B),
            np.where(put, 0.0, P),
        )

    @staticmethod
    def _calculate_greeks(
        levels: Dict[int, Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[float, float]:
        """
        Delta from step 1 and gamma from step 2, as the GS binomial engine.

        Args:
            levels: Mapping of step (0, 1, 2) to (stock prices, values) at that step

        Returns:
            Tuple of (delta, gamma)
        """
        if 2 not in levels:
            return 0.0, 0.0
        (S1, V1), (S2, V2) = levels[1], levels[2]
        delta = (V1[1] - V1[0]) / (S1[1] - S1[0])
        delta_up = (V2[2] - V2[1]) / (S2[2] - S2[1])
        delta_down = (V2[1] - V2[0]) / (S2[1] - S2[0])
        gamma = (delta_up - delta_down) / (0.5 * (S2[2] - S2[0]))
        return float(delta), float(gamma)

    def calculate_delta(self, bond: ConvertibleBond) -> float:
        """Delta of the convertible bond."""
        return self.price_with_details(bond).delta

    def calculate_gamma(self, bond: ConvertibleBond) -> float:
        """Gamma of the convertible bond."""
        return self.price_with_details(bond).gamma

    def get_cocb(self, bond: ConvertibleBond) -> float:
        """
        Cash-Only Component of Bond (COCB): the B component at the root.

        Args:
            bond: Convertible bond

        Returns:
            COCB value
        """
        return self.price_with_details(bond).bond_component

    def __repr__(self):
        return f"ConvertibleBondTFTreeEngine(num_steps={self.params.num_steps})"
