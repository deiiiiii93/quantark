"""
Analytical pricing engine for Asian options.

This module implements multiple approximation methods for pricing Asian options:
- KEMNA_VORST: Exact closed-form for geometric average options
- GEOMETRIC_DISCRETE: Discrete geometric average-rate using term-structure vols
- TURNBULL_WAKEMAN: Moment matching for arithmetic average options
- LEVY: Alternative arithmetic approximation (requires b != 0)
- CURRAN: Geometric conditioning approximation
- DISCRETE_HHM: Discrete arithmetic (Haug-Haug-Margrabe)

References:
    [1] Kemna, A.G.Z. and Vorst, A.C.F. (1990) "A Pricing Method for Options
        Based on Average Asset Values"
    [2] Turnbull, S.M. and Wakeman, L.M. (1991) "A Quick Algorithm for Pricing
        European Average Options"
    [3] Levy, E. (1992) "Pricing European Average Rate Currency Options"
    [4] Curran, M. (1992) "Valuing Asian and Portfolio Options by Conditioning
        on the Geometric Mean Price"
    [5] Haug, E.G. "The Complete Guide to Option Pricing Formulas" (Section 4.20)
    [6] Henderson, V. and Wojakowski, R. (2001) "On the Equivalence of Floating
        and Fixed Strike Asian Options"
"""

import numpy as np
from typing import Optional, Union, Dict, Tuple, List

from scipy.stats import norm

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.product.option.asian_option import AsianOption
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.param import EngineParams
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import ValidationError, NumericalError, PricingError
from quantark.util.enum.engine_enums import AsianAnalyticalMethod, EngineType
from quantark.util.numerical import is_zero, is_close


class AsianOptionAnalyticalEngine(BaseEngine):
    """
    Analytical pricing engine for Asian options.

    Supports six approximation/exact methods:
    - KEMNA_VORST (default for geometric): Exact closed-form for geometric average
    - GEOMETRIC_DISCRETE: Discrete geometric average-rate using term-structure vols
    - TURNBULL_WAKEMAN (default for arithmetic): Moment matching approximation
    - LEVY: Alternative arithmetic approximation (requires b != 0)
    - CURRAN: Geometric conditioning approximation
    - DISCRETE_HHM: Discrete arithmetic (Haug-Haug-Margrabe)

    For floating-strike options, Henderson-Wojakowski symmetry is applied
    to transform to fixed-strike pricing.

    Usage:
        # Preferred: Two-level enum pattern
        engine = AsianOptionAnalyticalEngine(
            method=EngineType.ANALYTICAL(AsianAnalyticalMethod.TURNBULL_WAKEMAN)
        )

        # Alternative: Direct method enum
        engine = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.CURRAN)

        # Backward compatibility: String
        engine = AsianOptionAnalyticalEngine(method="levy")
    """

    engine_type = EngineType.ANALYTICAL

    DEFAULT_METHOD = AsianAnalyticalMethod.TURNBULL_WAKEMAN
    DEFAULT_GEOMETRIC_METHOD = AsianAnalyticalMethod.KEMNA_VORST

    MIN_VOL = 0.001
    MAX_VOL = 5.0
    MIN_MATURITY = 1e-10
    MAX_MATURITY = 30.0

    def __init__(
        self,
        params: Optional[EngineParams] = None,
        method: Union[str, AsianAnalyticalMethod, tuple, None] = None,
    ):
        """
        Initialize Asian option analytical engine.

        Args:
            params: Engine configuration parameters
            method: Pricing method, can be:
                - AsianAnalyticalMethod enum
                - String "kemna_vorst"/"geometric_discrete"/"turnbull_wakeman"/"levy"/"curran"/"discrete_hhm"
                - Tuple from EngineType.ANALYTICAL(AsianAnalyticalMethod.X)
                - None (auto-selects based on averaging type)

        Raises:
            ValidationError: If invalid method is specified
        """
        super().__init__(params)

        if method is None:
            self.method = None  # Auto-select based on product
        elif isinstance(method, tuple):
            engine_type, analytical_method = method
            if engine_type != EngineType.ANALYTICAL:
                raise ValidationError(
                    f"Expected EngineType.ANALYTICAL, got {engine_type}"
                )
            if not isinstance(analytical_method, AsianAnalyticalMethod):
                raise ValidationError(
                    f"Expected AsianAnalyticalMethod, got {type(analytical_method).__name__}"
                )
            self.method = analytical_method
        elif isinstance(method, AsianAnalyticalMethod):
            self.method = method
        elif isinstance(method, str):
            try:
                self.method = AsianAnalyticalMethod(method.lower())
            except ValueError:
                valid_methods = ", ".join([m.value for m in AsianAnalyticalMethod])
                raise ValidationError(
                    f"Invalid method '{method}'. Valid methods are: {valid_methods}"
                )
        else:
            raise ValidationError(
                f"Method must be AsianAnalyticalMethod enum, string, or EngineType tuple, "
                f"got {type(method).__name__}"
            )

    def price(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        """
        Price an Asian option using the selected analytical method.

        Args:
            product: Asian option
            pricing_env: Pricing environment with market data

        Returns:
            Option price

        Raises:
            PricingError: If product is not an Asian option
            ValidationError: If input parameters are invalid
            NumericalError: If numerical computation fails
        """
        if not isinstance(product, AsianOption):
            raise PricingError(
                f"AsianOptionAnalyticalEngine only supports AsianOption, "
                f"got {type(product).__name__}"
            )

        # Extract parameters
        params = self._extract_params(product, pricing_env)

        # Validate inputs
        self._validate_inputs(params)

        # Handle near-expiry case
        if params["T"] < self.MIN_MATURITY:
            price = self._handle_near_expiry(product, params)
            return price * product.contract_multiplier

        # Clamp parameters for numerical stability
        params["sigma"] = np.clip(params["sigma"], self.MIN_VOL, self.MAX_VOL)
        params["T"] = np.clip(params["T"], self.MIN_MATURITY, self.MAX_MATURITY)

        # Select method
        method = self._select_method(product)

        # Check method compatibility
        self._check_method_compatibility(product, method, params)

        # Handle floating-strike via symmetry
        if product.is_floating_strike():
            price = self._price_floating_strike(product, params, method)
            return price * product.contract_multiplier

        # Price fixed-strike option
        price = self._price_fixed_strike(product, params, method)
        return price * product.contract_multiplier

    def _extract_params(
        self, product: AsianOption, pricing_env: PricingEnvironment
    ) -> dict:
        """Extract and validate all pricing parameters."""
        S = pricing_env.spot
        K = product.strike
        T = product.get_maturity(pricing_env)
        r = pricing_env.get_rate(T)
        q = pricing_env.get_div_yield(T)
        b = r - q
        sigma = pricing_env.get_vol(K, T)

        # Observation schedule
        past_prices, past_weights, future_times, future_weights, n_total = (
            product.resolve_observations(pricing_env)
        )
        n = n_total
        m = len(past_prices)
        S_A = product.get_past_average(pricing_env) if m > 0 else 0.0
        t1 = future_times[0] if future_times else 0.0

        return {
            "S": S,
            "K": K,
            "T": T,
            "r": r,
            "q": q,
            "b": b,
            "sigma": sigma,
            "n": n,
            "m": m,
            "S_A": S_A,
            "t1": t1,
            "future_times": future_times,
            "future_weights": future_weights,
            "past_prices": past_prices,
            "past_weights": past_weights,
            "vol_times": [pricing_env.get_vol(K, t) for t in future_times],
            "is_continuous": product.is_continuous(),
        }

    def _validate_inputs(self, params: dict) -> None:
        """Validate input parameters."""
        if params["S"] <= 0:
            raise ValidationError(f"Spot price must be positive, got {params['S']}")
        if params["K"] <= 0:
            raise ValidationError(f"Strike price must be positive, got {params['K']}")
        if params["T"] < 0:
            raise ValidationError(
                f"Time to maturity must be non-negative, got {params['T']}"
            )
        if params["sigma"] <= 0:
            raise ValidationError(f"Volatility must be positive, got {params['sigma']}")
        if not params["is_continuous"] and params["n"] < 1:
            raise ValidationError(
                f"Number of observations must be >= 1, got {params['n']}"
            )

    def _select_method(self, product: AsianOption) -> AsianAnalyticalMethod:
        """Select appropriate method based on product and user preference."""
        if self.method is not None:
            return self.method

        # Auto-select based on averaging type
        if product.is_geometric():
            if product.is_continuous():
                return self.DEFAULT_GEOMETRIC_METHOD
            return AsianAnalyticalMethod.GEOMETRIC_DISCRETE
        else:
            return self.DEFAULT_METHOD

    # Analytical methods that admit a verified weighted-moment generalization.
    _WEIGHTED_METHODS = (
        AsianAnalyticalMethod.TURNBULL_WAKEMAN,
        AsianAnalyticalMethod.GEOMETRIC_DISCRETE,
    )

    def _check_method_compatibility(
        self, product: AsianOption, method: AsianAnalyticalMethod, params: dict
    ) -> None:
        """Check if method is compatible with product."""
        # Non-uniform averaging weights: only a subset of methods can price them
        # correctly. Reject the rest (and floating-strike / continuous) rather
        # than silently ignoring the weights.
        if product.has_nonuniform_weights():
            if product.is_continuous():
                raise ValidationError(
                    "Non-uniform averaging weights require discrete observations."
                )
            if product.is_floating_strike():
                raise ValidationError(
                    "Non-uniform averaging weights are not supported for "
                    "floating-strike Asian options analytically; use the MC engine."
                )
            if method not in self._WEIGHTED_METHODS:
                raise ValidationError(
                    f"Method {method.name} cannot price non-uniform averaging "
                    "weights. Use TURNBULL_WAKEMAN (arithmetic) / GEOMETRIC_DISCRETE "
                    "(geometric), or the MC engine."
                )

        # KEMNA_VORST only works for geometric averaging
        if method in (
            AsianAnalyticalMethod.KEMNA_VORST,
            AsianAnalyticalMethod.GEOMETRIC_DISCRETE,
        ) and product.is_arithmetic():
            raise ValidationError(
                "Geometric methods only apply to geometric averaging. "
                "Use TURNBULL_WAKEMAN, LEVY, CURRAN, or DISCRETE_HHM for arithmetic."
            )

        # GEOMETRIC_DISCRETE only applies to discrete averaging
        if method == AsianAnalyticalMethod.GEOMETRIC_DISCRETE and product.is_continuous():
            raise ValidationError(
                "GEOMETRIC_DISCRETE only applies to discrete averaging. "
                "Use KEMNA_VORST for continuous geometric averaging."
            )

        # LEVY does not support b=0
        if method == AsianAnalyticalMethod.LEVY and is_zero(params["b"]):
            raise ValidationError(
                "LEVY approximation does not support b=0 (cost-of-carry). "
                "Use TURNBULL_WAKEMAN or DISCRETE_HHM instead."
            )

    def _handle_near_expiry(self, product: AsianOption, params: dict) -> float:
        """Handle near-expiry case by returning intrinsic value."""
        S = params["S"]
        K = params["K"]
        m = params["m"]
        n = params["n"]
        S_A = params["S_A"]

        if m > 0 and m == n:
            # All observations realized
            avg = S_A
        elif m > 0:
            # Some observations realized - estimate final average
            avg = (m * S_A + (n - m) * S) / n
        else:
            avg = S

        return product.get_payoff(S, average=avg) / product.contract_multiplier

    def _price_fixed_strike(
        self,
        product: AsianOption,
        params: dict,
        method: AsianAnalyticalMethod,
    ) -> float:
        """Price fixed-strike Asian option."""
        is_call = product.is_call()
        m = params["m"]
        n = params["n"]
        if m > 0 and m == n:
            # All averaging observations are already realized: payoff is deterministic.
            avg = params["S_A"]
            if avg is None:
                raise PricingError(
                    "Expected S_A (past average) to be available when all observations are realized."
                )
            payoff = max(avg - params["K"], 0.0) if is_call else max(params["K"] - avg, 0.0)
            return payoff * self._safe_exp(-params["r"] * params["T"])

        if method == AsianAnalyticalMethod.KEMNA_VORST:
            return self._price_kemna_vorst(params, is_call)
        elif method == AsianAnalyticalMethod.GEOMETRIC_DISCRETE:
            return self._price_geometric_discrete(params, is_call)
        elif method == AsianAnalyticalMethod.TURNBULL_WAKEMAN:
            return self._price_turnbull_wakeman(params, is_call)
        elif method == AsianAnalyticalMethod.LEVY:
            return self._price_levy(params, is_call)
        elif method == AsianAnalyticalMethod.CURRAN:
            return self._price_curran(params, is_call)
        elif method == AsianAnalyticalMethod.DISCRETE_HHM:
            return self._price_discrete_hhm(params, is_call)
        else:
            raise ValidationError(f"Unknown method: {method}")

    def _price_floating_strike(
        self,
        product: AsianOption,
        params: dict,
        method: AsianAnalyticalMethod,
    ) -> float:
        """
        Price floating-strike Asian option using Henderson-Wojakowski symmetry.

        Floating call = Fixed put with: r → r-b, b → -b, K → S
        Floating put = Fixed call with same transformation
        """
        is_call = product.is_call()
        m = params["m"]
        n = params["n"]
        if m > 0 and m == n:
            # All averaging observations are already realized: average strike is known,
            # so the payoff reduces to a vanilla European option on spot with strike=average.
            avg = params["S_A"]
            if avg is None:
                raise PricingError(
                    "Expected S_A (past average) to be available when all observations are realized."
                )
            return self._bsm_price(
                params["S"], avg, params["T"], params["r"], params["b"], params["sigma"], is_call
            )

        scale = 1.0
        params_transformed = params.copy()
        if (
            product.is_arithmetic()
            and not params["is_continuous"]
            and params["future_times"]
            and is_close(max(params["future_times"]), params["T"])
        ):
            # Symmetry assumes the average excludes the terminal fixing; adjust if it is included.
            if n <= 1:
                return 0.0
            scale = (n - 1) / n
            params_transformed = params.copy()
            max_time = max(params["future_times"])
            max_index = params["future_times"].index(max_time)
            params_transformed["future_times"] = [
                t
                for idx, t in enumerate(params["future_times"])
                if idx != max_index
            ]
            params_transformed["vol_times"] = [
                v
                for idx, v in enumerate(params["vol_times"])
                if idx != max_index
            ]
            # Drop the terminal fixing's weight and renormalize ALL remaining
            # fixings (past + kept future) so they still sum to 1. Custom-weighted
            # floating strike is rejected upstream, so this only rescales uniform
            # weights, but past weights must be scaled too (not just future).
            dropped_weight = params["future_weights"][max_index]
            remaining = 1.0 - dropped_weight
            if remaining <= 0:
                return 0.0
            params_transformed["future_weights"] = [
                w / remaining
                for idx, w in enumerate(params["future_weights"])
                if idx != max_index
            ]
            params_transformed["past_weights"] = [
                w / remaining for w in params["past_weights"]
            ]
            params_transformed["n"] = n - 1

        # Transform parameters
        params_transformed["r"] = params["r"] - params["b"]
        params_transformed["b"] = -params["b"]
        params_transformed["K"] = params["S"]

        # Price as opposite option type
        if is_call:
            return scale * self._price_fixed_strike_internal(
                params_transformed, method, is_call=False
            )
        else:
            return scale * self._price_fixed_strike_internal(
                params_transformed, method, is_call=True
            )

    def _price_fixed_strike_internal(
        self,
        params: dict,
        method: AsianAnalyticalMethod,
        is_call: bool,
    ) -> float:
        """Internal fixed-strike pricing with explicit is_call flag."""
        if method == AsianAnalyticalMethod.KEMNA_VORST:
            return self._price_kemna_vorst(params, is_call)
        elif method == AsianAnalyticalMethod.GEOMETRIC_DISCRETE:
            return self._price_geometric_discrete(params, is_call)
        elif method == AsianAnalyticalMethod.TURNBULL_WAKEMAN:
            return self._price_turnbull_wakeman(params, is_call)
        elif method == AsianAnalyticalMethod.LEVY:
            return self._price_levy(params, is_call)
        elif method == AsianAnalyticalMethod.CURRAN:
            return self._price_curran(params, is_call)
        elif method == AsianAnalyticalMethod.DISCRETE_HHM:
            return self._price_discrete_hhm(params, is_call)
        else:
            raise ValidationError(f"Unknown method: {method}")

    # =========================================================================
    # KEMNA-VORST: Geometric Average (Exact)
    # =========================================================================

    def _price_kemna_vorst(self, params: dict, is_call: bool) -> float:
        """
        Price geometric average Asian option using Kemna-Vorst formula.

        This is an exact closed-form solution for geometric average options.

        Formula:
            σ_A = σ / √3
            b_A = (b - σ²/6) / 2
            Then apply BSM with adjusted parameters.
        """
        S = params["S"]
        K = params["K"]
        T = params["T"]
        r = params["r"]
        b = params["b"]
        sigma = params["sigma"]

        # Adjusted volatility and cost-of-carry
        sigma_A = sigma / np.sqrt(3)
        b_A = 0.5 * (b - sigma**2 / 6)

        # Apply BSM formula with adjusted parameters
        return self._bsm_price(S, K, T, r, b_A, sigma_A, is_call)

    # =========================================================================
    # GEOMETRIC DISCRETE: Geometric Average (Discrete Fixings)
    # =========================================================================

    def _price_geometric_discrete(self, params: dict, is_call: bool) -> float:
        """
        Price discrete geometric average Asian option using term-structure volatilities.

        Uses Haug-Haug-Margrabe term-structure formulation for geometric average.
        """
        S = params["S"]
        K = params["K"]
        T = params["T"]
        r = params["r"]
        b = params["b"]
        m = params["m"]
        past_prices = params["past_prices"]
        past_weights = params["past_weights"]

        # Sort future fixings ascending, carrying each fixing's vol and weight.
        triples = sorted(
            zip(params["future_times"], params["vol_times"], params["future_weights"]),
            key=lambda tvw: tvw[0],
        )
        future_times = [t for t, _, _ in triples]
        future_vols = [v for _, v, _ in triples]
        future_weights = [w for _, _, w in triples]

        n_future = len(future_times)
        if n_future <= 0:
            avg = params["S_A"] if m > 0 else S
            payoff = max(avg - K, 0.0) if is_call else max(K - avg, 0.0)
            return payoff * self._safe_exp(-r * T)

        # Weighted log-average  ln A = sum_i w_i ln S_ti, weights global (sum to 1).
        # E[ln A] = sum_past w_p ln(S_p) + sum_future w_f [ln S + (b - 0.5 sigma_f^2) t_f]
        log_S = self._safe_log(S)
        m_total = sum(
            wp * self._safe_log(p) for wp, p in zip(past_weights, past_prices)
        )
        m_total += sum(
            wf * (log_S + (b - 0.5 * sigma_i**2) * ti)
            for wf, sigma_i, ti in zip(future_weights, future_vols, future_times)
        )

        # Var[ln A] = sum_i sum_j w_i w_j Cov(ln S_ti, ln S_tj), future only.
        # With term-structure vols and ascending times, Cov(i,j) ~= sigma_i^2 t_i
        # for i <= j, giving  sum_i sigma_i^2 t_i ( w_i^2 + 2 w_i * sum_{j>i} w_j ).
        suffix_w = np.zeros(n_future + 1)  # suffix_w[i] = sum_{j>=i} w_j
        for i in range(n_future - 1, -1, -1):
            suffix_w[i] = suffix_w[i + 1] + future_weights[i]

        v_total = 0.0
        for i in range(n_future):
            sigma2_t = future_vols[i] ** 2 * future_times[i]
            wi = future_weights[i]
            v_total += sigma2_t * (wi * wi + 2.0 * wi * suffix_w[i + 1])

        if v_total < 1e-20:
            v_total = 1e-20

        return self._price_lognormal(m_total, v_total, K, r, T, is_call)

    # =========================================================================
    # TURNBULL-WAKEMAN: Arithmetic Average (Moment Matching)
    # =========================================================================

    def _price_turnbull_wakeman(self, params: dict, is_call: bool) -> float:
        """
        Price arithmetic average Asian option using Turnbull-Wakeman approximation.

        This method matches the first two moments of the arithmetic average
        to a lognormal distribution.
        """
        S = params["S"]
        K = params["K"]
        T = params["T"]
        r = params["r"]
        b = params["b"]
        sigma = params["sigma"]
        m = params["m"]
        n = params["n"]
        S_A = params["S_A"]
        t1 = params["t1"]
        is_continuous = params["is_continuous"]

        if m > 0 and m == n:
            # All observations already realized
            avg = S_A
            payoff = max(avg - K, 0.0) if is_call else max(K - avg, 0.0)
            return payoff * self._safe_exp(-r * T)

        # 1. Compute Moments M1 and M2
        if is_continuous:
            # Continuous averaging period (T2 in Haug's notation)
            if m > 0:
                T2 = T * n / (n - m) if (n - m) > 0 else T
                tau = T2 - T
            else:
                T2 = T
                tau = 0.0
                t1 = max(0.0, t1)

            # First moment M1 (continuous)
            if is_zero(b):
                M1 = 1.0
            else:
                if T - t1 > 1e-10:
                    M1 = (self._safe_exp(b * T) - self._safe_exp(b * t1)) / (b * (T - t1))
                else:
                    M1 = self._safe_exp(b * T)

            # Second moment M2 (continuous)
            M2 = self._compute_M2_turnbull_wakeman(T, t1, b, sigma)
        else:
            # Discrete averaging
            future_times = params["future_times"]
            M1, M2 = self._compute_M1_M2_discrete(
                S, b, sigma, future_times, params["future_weights"],
                params["past_prices"], params["past_weights"],
            )
            T2 = T # Use T as maturity for discrete case adjustment

        # 2. Adjust for in-period results
        if m > 0 and not is_continuous:
            # For discrete in-period, M1 and M2 already include past prices
            # Just adjust strike to match the BSM pricing on future spot
            # Actually, TW usually approximates the sum.
            pass

        # 3. Compute adjusted Black-Scholes parameters
        b_A = self._safe_log(M1) / T if T > 0 else 0.0
        var_term = self._safe_log(M2) / T - 2 * b_A
        if var_term < 0:
            var_term = 1e-10
        sigma_A = np.sqrt(var_term)

        # 4. Handle in-period adjustment (Haug's method for ITM check)
        if m > 0:
            if is_continuous:
                tau = T2 - T
                X_adj = (T2 / T) * K - (tau / T) * S_A
                # Certain exercise check
                if X_adj < 0:
                    if is_call:
                        E_SA = S_A * (T2 - T) / T2 + S * M1 * T / T2
                        return max(0.0, E_SA - K) * self._safe_exp(-r * T)
                    else:
                        return 0.0
                price = self._bsm_price(S, X_adj, T, r, b_A, sigma_A, is_call)
                return price * (T / T2)
            else:
                # Discrete in-period: Average = (sum_past + sum_future) / n
                # E[Average] = (m*S_A + sum E[S_ti]) / n = M1 * S
                # Here M1 is already (sum_past/S + sum_future_E/S) / n
                # So E[Average] = M1 * S.
                # Strike adjustment for BSM: Price(S, K_adj) where Average > K
                # (sum_past + sum_future) / n > K  => sum_future > n*K - m*S_A
                # sum_future / (n-m) > (n*K - m*S_A) / (n-m)
                # However, TW maps the whole average to a lognormal.
                # If we map (sum_past + sum_future)/n to lognormal, its mean is M1*S.
                # So we can just price with S_eff = M1*S and K_eff = K? 
                # No, standard TW uses S and K and adjusts b and sigma.
                # To match Haug's discrete HHM:
                X_adj = (n * K - m * S_A) / (n - m) if n > m else K
                # Price future part: (n-m)/n * Price(S, X_adj)
                # But we must ensure M1 and M2 are for the FUTURE part only if we do this.
                
                # REFINED DISCRETE TW: Treat the WHOLE average as lognormal.
                # Mean = M1*S, Mean^2 = M2*S^2.
                # Price = exp(-rT) * E[max(A-K, 0)]
                # This is equivalent to BSM price with S_0 = M1*S*exp(-b_A*T), K=K, etc.
                # Or more simply: S_0 = M1*S, K=K, r=r, q=r-b_A, sigma=sigma_A.
                return self._bsm_price(S * M1 * self._safe_exp(-b_A * T), K, T, r, b_A, sigma_A, is_call)
        else:
            return self._bsm_price(S, K, T, r, b_A, sigma_A, is_call)

    def _compute_M1_M2_discrete(
        self,
        S: float,
        b: float,
        sigma: float,
        future_times: List[float],
        future_weights: List[float],
        past_prices: List[float],
        past_weights: List[float],
    ) -> Tuple[float, float]:
        """First and second moments of the weighted discrete arithmetic average.

        Average A = sum_i w_i S_ti with weights normalized so that
        sum(past_weights) + sum(future_weights) == 1. Uniform weights (w_i = 1/n)
        reproduce the equal-weight moments exactly.

        M1 = E[A]/S = sum_past w_p (S_p/S) + sum_future w_f e^{b t_f}
        M2 = E[A^2]/S^2 = P_w^2 + 2 P_w F_w + FF_w, where
            P_w = sum_past w_p (S_p/S),  F_w = sum_future w_f e^{b t_f},
            FF_w = sum_i sum_j w_i w_j e^{b t_i + b t_j + sigma^2 min(t_i,t_j)}.
        """
        # Deterministic past contribution (weighted), as a ratio to S
        P_w = (
            sum(wp * (p / S) for wp, p in zip(past_weights, past_prices))
            if S > 0
            else 0.0
        )
        # Stochastic future first moment (weighted)
        F_w = sum(
            wf * self._safe_exp(b * t) for wf, t in zip(future_weights, future_times)
        )
        M1 = P_w + F_w

        # FF_w: weighted future-future double sum in O(n) via a weighted suffix sum.
        # Sort fixings ascending so min(t_i, t_j) = t_i for i <= j.
        pairs = sorted(zip(future_times, future_weights), key=lambda tw: tw[0])
        times = [t for t, _ in pairs]
        wts = [w for _, w in pairs]
        n_future = len(times)

        # suffix[i] = sum_{j >= i} w_j e^{b t_j}
        suffix = np.zeros(n_future + 1)
        for i in range(n_future - 1, -1, -1):
            suffix[i] = suffix[i + 1] + wts[i] * self._safe_exp(b * times[i])

        FF_w = 0.0
        for i in range(n_future):
            ti = times[i]
            wi = wts[i]
            diag = wi * wi * self._safe_exp((2 * b + sigma**2) * ti)
            cross = 2 * wi * self._safe_exp((b + sigma**2) * ti) * suffix[i + 1]
            FF_w += diag + cross

        M2 = P_w**2 + 2 * P_w * F_w + FF_w

        return M1, M2

    def _compute_M2_turnbull_wakeman(
        self, T: float, t1: float, b: float, sigma: float
    ) -> float:
        """Compute second moment M2 for Turnbull-Wakeman."""
        if T - t1 < 1e-10:
            return self._safe_exp(sigma**2 * T)

        if is_zero(b):
            # Special case for b = 0
            sigma2 = sigma**2
            denom = sigma2**2 * (T - t1) ** 2
            if denom < 1e-20:
                return 1.0
            M2 = (
                2 * self._safe_exp(sigma2 * T)
                - 2 * self._safe_exp(sigma2 * t1) * (1 + sigma2 * (T - t1))
            ) / denom
        else:
            # General case
            term1_denom = (b + sigma**2) * (2 * b + sigma**2) * (T - t1) ** 2
            term1 = 2 * self._safe_exp((2 * b + sigma**2) * T) / term1_denom

            term2_factor = 1.0 / (2 * b + sigma**2) - self._safe_exp(b * (T - t1)) / (
                b + sigma**2
            )
            term2 = (
                2 * self._safe_exp((2 * b + sigma**2) * t1) / (b * (T - t1) ** 2)
            ) * term2_factor

            M2 = term1 + term2

        return max(M2, 1e-20)

    # =========================================================================
    # LEVY: Arithmetic Average Approximation
    # =========================================================================

    def _price_levy(self, params: dict, is_call: bool) -> float:
        """
        Price arithmetic average Asian option using Levy approximation.

        Note: This method does not support b = 0.
        """
        S = params["S"]
        K = params["K"]
        T = params["T"]
        r = params["r"]
        b = params["b"]
        sigma = params["sigma"]
        m = params["m"]
        n = params["n"]
        S_A = params["S_A"]
        is_continuous = params["is_continuous"]

        if m > 0 and m == n:
            # All observations realized
            avg = S_A
            payoff = max(avg - K, 0.0) if is_call else max(K - avg, 0.0)
            return payoff * self._safe_exp(-r * T)

        if is_continuous:
            # Original Levy formula (continuous)
            if m > 0:
                T2 = T * n / (n - m)
            else:
                T2 = T

            S_E = (S / (T * b)) * (self._safe_exp((b - r) * T2) - self._safe_exp(-r * T2))
            X_star = K - ((T2 - T) / T2) * S_A if m > 0 else K
            
            M = (2 * S**2 / (b + sigma**2)) * (
                (self._safe_exp((2 * b + sigma**2) * T2) - 1) / (2 * b + sigma**2)
                - (self._safe_exp(b * T2) - 1) / b
            )
            D = M / (T2**2)
            # Levy uses T2 for maturity in BSM
            V = self._safe_log(D) - 2 * (r * T2 + self._safe_log(S_E))
            if V < 1e-20: V = 1e-20
            sqrt_V = np.sqrt(V)
            
            d1 = (0.5 * self._safe_log(D) - self._safe_log(X_star)) / sqrt_V
            d2 = d1 - sqrt_V
            
            call_price = S_E * norm.cdf(d1) - X_star * self._safe_exp(-r * T2) * norm.cdf(d2)
        else:
            # Discrete Levy: Same moment matching logic as TW but using Levy's BSM parameterization
            # Levy matches E[A] and E[A^2] to lognormal.
            future_times = params["future_times"]
            M1, M2 = self._compute_M1_M2_discrete(
                S, b, sigma, future_times, params["future_weights"],
                params["past_prices"], params["past_weights"],
            )
            
            E_A = M1 * S
            E_A2 = M2 * S**2
            
            # Map E[A], E[A^2] to BSM S_E, D
            # Levy's method is essentially TW but with different ITM adjustment.
            # For consistency, we use the same core logic for discrete TW and Levy.
            # But Levy formula usually doesn't have the ITM "certain exercise" check.
            
            V = self._safe_log(E_A2) - 2 * self._safe_log(E_A)
            if V < 1e-20: V = 1e-20
            
            d1 = (self._safe_log(E_A / K) + 0.5 * V) / np.sqrt(V)
            d2 = d1 - np.sqrt(V)
            
            call_price = self._safe_exp(-r * T) * (E_A * norm.cdf(d1) - K * norm.cdf(d2))

        if is_call:
            return max(call_price, 0.0)
        else:
            # Put via parity: Put = Call - discounted(E[A] - K)
            E_A = (M1 * S) if not is_continuous else (S_E * self._safe_exp(r * T2))
            X_eff = K if not is_continuous else X_star
            T_eff = T if not is_continuous else T2
            put_price = call_price - (E_A - X_eff) * self._safe_exp(-r * T_eff)
            return max(put_price, 0.0)

    # =========================================================================
    # CURRAN: Geometric Conditioning Approximation
    # =========================================================================

    def _price_curran(self, params: dict, is_call: bool) -> float:
        """
        Price arithmetic average Asian option using Curran's approximation.

        Based on geometric conditioning approach.
        """
        S = params["S"]
        K = params["K"]
        T = params["T"]
        r = params["r"]
        b = params["b"]
        sigma = params["sigma"]
        n = params["n"]
        m = params["m"]
        S_A = params["S_A"]
        t1 = params["t1"]

        # Number of future observations
        n_future = n - m
        if n_future <= 0:
            # All observations are past, return intrinsic
            avg = S_A
            if is_call:
                return max(avg - K, 0.0) * self._safe_exp(-r * T)
            else:
                return max(K - avg, 0.0) * self._safe_exp(-r * T)

        # Time step between observations
        if n_future > 1:
            dt = (T - t1) / (n_future - 1) if n_future > 1 else T - t1
        else:
            dt = T - t1

        # Compute E[A] for certain exercise check
        if is_zero(b):
            E_A = S
        else:
            E_A = (
                S
                / n_future
                * self._safe_exp(b * t1)
                * (1 - self._safe_exp(b * dt * n_future))
                / (1 - self._safe_exp(b * dt))
            )

        # Handle in-period adjustment
        X_adj = K
        if m > 0:
            # Check for certain exercise
            if S_A > (n / m) * K:
                if is_call:
                    final_avg = S_A * m / n + E_A * (n - m) / n
                    return (final_avg - K) * self._safe_exp(-r * T)
                else:
                    return 0.0
            X_adj = (n * K - m * S_A) / (n - m)

        # Single fixing left - use BSM
        if n_future == 1:
            X_bsm = n * K - (n - 1) * S_A if m > 0 else K
            price = self._bsm_price(S, X_bsm, T, r, b, sigma, is_call)
            return price / n

        # Compute Curran parameters
        mu = self._safe_log(S) + (b - 0.5 * sigma**2) * (t1 + (n_future - 1) * dt / 2)
        vx = sigma * np.sqrt(t1 + dt * (n_future - 1) * (2 * n_future - 1) / (6 * n_future))

        # Compute sum1 for Km
        sum1 = 0.0
        for i in range(1, n_future + 1):
            ti = t1 + (i - 1) * dt
            vi = sigma * np.sqrt(ti) if ti > 0 else 0.0
            vxi = sigma**2 * (t1 + dt * ((i - 1) - i * (i - 1) / (2 * n_future)))
            mui = self._safe_log(S) + (b - 0.5 * sigma**2) * ti

            exp_term = mui + (vxi / vx**2) * (self._safe_log(X_adj) - mu) + 0.5 * (
                vi**2 - vxi**2 / vx**2
            )
            sum1 += self._safe_exp(exp_term)

        Km = 2 * X_adj - sum1 / n_future

        # Compute sum2 for option value
        sum2 = 0.0
        z = 1 if is_call else -1
        for i in range(1, n_future + 1):
            ti = t1 + (i - 1) * dt
            vi = sigma * np.sqrt(ti) if ti > 0 else 0.0
            vxi = sigma**2 * (t1 + dt * ((i - 1) - i * (i - 1) / (2 * n_future)))
            mui = self._safe_log(S) + (b - 0.5 * sigma**2) * ti

            d_term = z * ((mu - self._safe_log(Km)) / vx + vxi / vx)
            sum2 += self._safe_exp(mui + 0.5 * vi**2) * norm.cdf(d_term)

        d_main = z * (mu - self._safe_log(Km)) / vx
        price = self._safe_exp(-r * T) * z * (
            sum2 / n_future - X_adj * norm.cdf(d_main)
        )

        # Apply in-period multiplier
        if m > 0:
            price *= (n - m) / n

        return max(price, 0.0)

    # =========================================================================
    # DISCRETE HHM: Discrete Arithmetic (Haug-Haug-Margrabe)
    # =========================================================================

    def _price_discrete_hhm(self, params: dict, is_call: bool) -> float:
        """
        Price discrete arithmetic average Asian option using HHM approximation.

        This is a modified Levy formula from Haug, Haug, and Margrabe.
        """
        S = params["S"]
        K = params["K"]
        T = params["T"]
        r = params["r"]
        b = params["b"]
        sigma = params["sigma"]
        n = params["n"]
        m = params["m"]
        S_A = params["S_A"]
        t1 = params["t1"]

        # Number of future observations
        n_future = n - m
        if n_future <= 0:
            avg = S_A
            if is_call:
                return max(avg - K, 0.0) * self._safe_exp(-r * T)
            else:
                return max(K - avg, 0.0) * self._safe_exp(-r * T)

        # Time step between observations
        h = (T - t1) / (n_future - 1) if n_future > 1 else T - t1

        # Compute E[A_T]
        if is_zero(b):
            E_A = S
        else:
            if abs(1 - self._safe_exp(b * h)) < 1e-10:
                E_A = S * self._safe_exp(b * t1)
            else:
                E_A = (
                    S
                    / n_future
                    * self._safe_exp(b * t1)
                    * (1 - self._safe_exp(b * h * n_future))
                    / (1 - self._safe_exp(b * h))
                )

        # Handle certain exercise
        if m > 0 and S_A > (n / m) * K:
            if is_call:
                final_avg = S_A * m / n + E_A * (n - m) / n
                return (final_avg - K) * self._safe_exp(-r * T)
            else:
                return 0.0

        # Single fixing left
        if n_future == 1:
            X_bsm = n * K - (n - 1) * S_A if m > 0 else K
            price = self._bsm_price(S, X_bsm, T, r, b, sigma, is_call)
            return price / n

        # Compute E[A_T^2]
        E_A2 = self._compute_E_A2_hhm(S, t1, h, n_future, b, sigma)

        # Compute volatility of average
        log_ratio = self._safe_log(E_A2) - 2 * self._safe_log(E_A)
        if log_ratio < 0:
            log_ratio = 1e-10
        sigma_A = np.sqrt(log_ratio / T) if T > 0 else sigma

        # Adjust strike for in-period
        X_adj = K
        if m > 0:
            X_adj = (n * K - m * S_A) / (n - m)

        # Compute d1, d2
        d1 = (self._safe_log(E_A / X_adj) + 0.5 * sigma_A**2 * T) / (
            sigma_A * np.sqrt(T)
        )
        d2 = d1 - sigma_A * np.sqrt(T)

        # Compute option price
        if is_call:
            price = self._safe_exp(-r * T) * (
                E_A * norm.cdf(d1) - X_adj * norm.cdf(d2)
            )
        else:
            price = self._safe_exp(-r * T) * (
                X_adj * norm.cdf(-d2) - E_A * norm.cdf(-d1)
            )

        # Apply in-period multiplier
        if m > 0:
            price *= (n - m) / n

        return max(price, 0.0)

    def _compute_E_A2_hhm(
        self, S: float, t1: float, h: float, n: int, b: float, sigma: float
    ) -> float:
        """Compute E[A_T^2] for discrete HHM."""
        sigma2 = sigma**2

        if is_zero(b):
            # b = 0 case
            exp_vh = self._safe_exp(sigma2 * h)
            if abs(1 - exp_vh) < 1e-10:
                return S**2 * self._safe_exp(sigma2 * t1)

            term1 = (1 - self._safe_exp(sigma2 * h * n)) / (1 - exp_vh)
            term2 = (n - term1) * 2 / (1 - exp_vh)

            E_A2 = S**2 * self._safe_exp(sigma2 * t1) / (n**2) * (term1 + term2)
        else:
            # General case
            exp_2bv_h = self._safe_exp((2 * b + sigma2) * h)
            exp_bv_h = self._safe_exp((b + sigma2) * h)
            exp_b_h = self._safe_exp(b * h)

            if abs(1 - exp_2bv_h) < 1e-10 or abs(1 - exp_bv_h) < 1e-10 or abs(1 - exp_b_h) < 1e-10:
                return S**2 * self._safe_exp((2 * b + sigma2) * t1)

            term1 = (1 - self._safe_exp((2 * b + sigma2) * h * n)) / (1 - exp_2bv_h)
            term2_a = (1 - self._safe_exp(b * h * n)) / (1 - exp_b_h)
            term2_b = (1 - self._safe_exp((2 * b + sigma2) * h * n)) / (1 - exp_2bv_h)
            term2 = 2 / (1 - exp_bv_h) * (term2_a - term2_b)

            E_A2 = (
                S**2 * self._safe_exp((2 * b + sigma2) * t1) / (n**2) * (term1 + term2)
            )

        return max(E_A2, 1e-20)

    # =========================================================================
    # Helper Functions
    # =========================================================================

    def _price_lognormal(
        self, m: float, v: float, K: float, r: float, T: float, is_call: bool
    ) -> float:
        """Price option on a lognormal variable with log-mean m and variance v."""
        if v < 1e-12:
            intrinsic = self._safe_exp(m) - K
            payoff = max(intrinsic, 0.0) if is_call else max(-intrinsic, 0.0)
            return payoff * self._safe_exp(-r * T)

        sqrt_v = np.sqrt(v)
        d1 = (m - self._safe_log(K) + v) / sqrt_v
        d2 = d1 - sqrt_v
        forward = self._safe_exp(m + 0.5 * v)
        df = self._safe_exp(-r * T)

        if is_call:
            price = df * (forward * norm.cdf(d1) - K * norm.cdf(d2))
        else:
            price = df * (K * norm.cdf(-d2) - forward * norm.cdf(-d1))

        return max(price, 0.0)

    def _bsm_price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        b: float,
        sigma: float,
        is_call: bool,
    ) -> float:
        """Generalized Black-Scholes-Merton formula."""
        if T < 1e-10:
            if is_call:
                return max(S - K, 0.0)
            else:
                return max(K - S, 0.0)

        sqrt_T = np.sqrt(T)
        d1 = (self._safe_log(S / K) + (b + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        if is_call:
            price = S * self._safe_exp((b - r) * T) * norm.cdf(d1) - K * self._safe_exp(
                -r * T
            ) * norm.cdf(d2)
        else:
            price = K * self._safe_exp(-r * T) * norm.cdf(-d2) - S * self._safe_exp(
                (b - r) * T
            ) * norm.cdf(-d1)

        return max(price, 0.0)

    def _safe_log(self, x: float) -> float:
        """Safe logarithm to avoid log(0) or log(negative)."""
        return np.log(max(x, 1e-16))

    def _safe_exp(self, x: float) -> float:
        """Safe exponential to avoid overflow/underflow."""
        if x > 700:
            return np.exp(700)
        if x < -700:
            return 0.0
        return np.exp(x)

    def __repr__(self):
        method_str = self.method.value if self.method else "auto"
        return f"AsianOptionAnalyticalEngine(method='{method_str}')"
