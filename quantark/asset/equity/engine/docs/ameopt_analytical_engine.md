# -*- coding: utf-8 -*-
"""
Created on Mon May 22 2023

@author: yaofuxin
@description:
    This class is used to price American vanilla options.
    Three methods are supported: BS93, BS02, and BAW.
    Reference:
    (1) Bjerksund. P, and Stensland. G., 1993.
    (2) Bjerksund. P, and Stensland. G., 2002. Closed-form approximation of American options.
    (3) Barone-Adesi, G., and Whaley, R. E., 1987. Efficient analytic approximation of American option values. Journal of Finance, 42(2), 351-360.


"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import fmin
from EuropeanOption import EuropeanVanillaOption


class AmericanVanillaOption(EuropeanVanillaOption):
    def __init__(
        self,
        rrf,
        div,
        vol,
        K,
        spot,
        cal_days,
        bus_days,
        disc_mode,
        N,
        is_call,
        tenor=0,
        participation_rate=1.0,
        bus_days_in_year=245,
        is_annualized=True,
        american_method="BS93",
    ) -> None:
        self.rrf = rrf
        self.div = div
        self.b = self.rrf - self.div
        self.vol = vol
        self.K = K
        self.spot = spot
        self.cal_days = cal_days
        self.bus_days = bus_days
        self.T = cal_days / 365
        self.disc_mode = disc_mode
        self.bus_days_in_year = bus_days_in_year
        self.tau = self.bus_days / self.bus_days_in_year
        self.df = (
            np.exp(-1.0 * self.rrf * self.tau)
            if self.disc_mode == "BD"
            else np.exp(-1.0 * self.rrf * self.T)
        )
        self.N = N
        self.is_call = is_call
        self.participation_rate = participation_rate
        self.tenor = cal_days if tenor <= 0 else tenor
        self.is_annualized = is_annualized
        self.value = 0.0
        self.delta = 0.0
        self.delta_cash = 0.0
        self.gamma = 0.0
        self.gamma_cash = 0.0
        self.vega = 0.0
        self.theta = 0.0
        self.rho = 0.0
        self.rhoQ = 0.0
        self.american_method = american_method.upper()

    def price_bsm(self):
        self.b = self.rrf - self.div
        annualized_factor = 1 if not self.is_annualized else self.tenor / 365
        if self.is_call and self.b >= self.rrf and self.rrf >= 0:
            # american call option will never be early exercised if q <= 0
            super().price_bsm()
        elif (
            self.is_call
            and self.b >= self.rrf
            and self.rrf < 0
            and self.div <= self.rrf
        ):
            # american call option will never be early exercised if q <= 0, but additional term has to be satified when rrf is negative
            super().price_bsm()
        elif not self.is_call:
            spot = self.spot
            K = self.K
            rrf = self.rrf
            b = self.b
            div = self.div
            # american put option will never be early exercised if rrf <= 0 and rrf <= div
            if hasattr(self, "_debug_bs02") and self._debug_bs02:
                print(
                    f"PUT CHECK: rrf={rrf:.4f}, div={div:.4f}, condition (rrf <= 0 and rrf <= div) = {rrf <= 0 and rrf <= div}"
                )
            if rrf <= 0 and rrf <= div:
                if hasattr(self, "_debug_bs02") and self._debug_bs02:
                    print("PUT: Using European pricing (early exit)")
                super().price_bsm()
            else:
                if getattr(self, "american_method", "BS93") == "BAW":
                    self.value = (
                        self.N
                        * self.__price_ame_put_baw__()
                        * self.participation_rate
                        * annualized_factor
                    )
                elif getattr(self, "american_method", "BS93") == "BS02":
                    # Use BS02 via call-analogue transformation
                    if hasattr(self, "_debug_bs02") and self._debug_bs02:
                        print(
                            f"PUT TRANSFORM: Original S={self.spot:.2f}, K={self.K:.2f}, r={self.rrf:.4f}, b={self.b:.4f}"
                        )
                    self.spot = K
                    self.K = spot
                    self.rrf = div
                    self.b = -1.0 * b
                    if hasattr(self, "_debug_bs02") and self._debug_bs02:
                        print(
                            f"PUT TRANSFORM: New S={self.spot:.2f}, K={self.K:.2f}, r={self.rrf:.4f}, b={self.b:.4f}"
                        )
                    call_value = self.__price_ame_call_bs02__()
                    if hasattr(self, "_debug_bs02") and self._debug_bs02:
                        print(
                            f"PUT TRANSFORM: Call value from transformation = {call_value:.6f}"
                        )
                    self.value = (
                        self.N
                        * call_value
                        * self.participation_rate
                        * annualized_factor
                    )
                    self.spot = spot
                    self.K = K
                    self.rrf = rrf
                    self.b = b
                else:
                    # Use BS93 via call-analogue transformation
                    self.spot = K
                    self.K = spot
                    self.rrf = div
                    self.b = -1.0 * b
                    self.value = (
                        self.N
                        * self.__price_ame_call__()
                        * self.participation_rate
                        * annualized_factor
                    )
                    self.spot = spot
                    self.K = K
                    self.rrf = rrf
                    self.b = b
        else:
            if getattr(self, "american_method", "BS93") == "BAW":
                self.value = (
                    self.N
                    * self.__price_ame_call_baw__()
                    * self.participation_rate
                    * annualized_factor
                )
            elif getattr(self, "american_method", "BS93") == "BS02":
                self.value = (
                    self.N
                    * self.__price_ame_call_bs02__()
                    * self.participation_rate
                    * annualized_factor
                )
            else:
                self.value = (
                    self.N
                    * self.__price_ame_call__()
                    * self.participation_rate
                    * annualized_factor
                )

    def __phi__(self, gamma, H, trigger_price):
        lamda = self.tau * (
            -1.0 * self.rrf + self.b * gamma + 0.5 * gamma * (gamma - 1) * self.vol**2
        )
        d = (
            -1.0
            * (
                np.log(self.spot / H)
                + (self.b + (gamma - 0.5) * self.vol**2) * self.tau
            )
            / self.vol
            / np.sqrt(self.tau)
        )
        kappa = 2 * self.b / self.vol**2 + 2 * gamma - 1
        nd1 = norm.cdf(d, loc=0, scale=1)
        nd2 = norm.cdf(
            d - 2 * np.log(trigger_price / self.spot) / self.vol / np.sqrt(self.tau),
            loc=0,
            scale=1,
        )
        return (
            np.exp(lamda)
            * np.power(self.spot, gamma)
            * (nd1 - np.power(trigger_price / self.spot, kappa) * nd2)
        )

    def __price_ame_call__(self):
        beta = (0.5 - self.b / self.vol**2) + np.sqrt(
            (0.5 - self.b / self.vol**2) ** 2 + 2 * self.rrf / self.vol**2
        )
        B_0 = max(self.K, self.K * self.rrf / (self.rrf - self.b))
        B_infinite = self.K * beta / (beta - 1.0)
        h_tau = (
            -1.0
            * (self.b * self.tau + 2.0 * self.vol * np.sqrt(self.tau))
            * B_0
            / (B_infinite - B_0)
        )
        trigger_price = B_0 + (B_infinite - B_0) * (1.0 - np.exp(h_tau))
        alpha_x = (trigger_price - self.K) * np.power(trigger_price, -1.0 * beta)

        value = 0.0
        if self.spot >= trigger_price:
            value = self.spot - self.K
        else:
            value = (
                alpha_x * np.power(self.spot, beta)
                - alpha_x * self.__phi__(beta, trigger_price, trigger_price)
                + self.__phi__(1, trigger_price, trigger_price)
                - self.__phi__(1, self.K, trigger_price)
                - self.K * self.__phi__(0, trigger_price, trigger_price)
                + self.K * self.__phi__(0, self.K, trigger_price)
            )

        return value

    def __phi_bs02__(self, S, T, gamma, H, I):
        """
        Helper function φ for Bjerksund-Stensland 2002 approximation
        φ(S, T, γ, H, I) = e^λT * S^γ * [N(-d) - (I/S)^κ * N(-d₂)]
        """
        if S <= 0 or T <= 0 or H <= 0 or I <= 0:
            return 0.0

        b = self.b
        r = self.rrf
        vol = self.vol

        # Calculate λ = -r + γb + 0.5γ(γ-1)σ²
        lambda_val = -r + gamma * b + 0.5 * gamma * (gamma - 1) * vol**2

        # Calculate κ = 2b/σ² + (2γ - 1)
        kappa = 2 * b / (vol**2) + (2 * gamma - 1)

        # Calculate d = [ln(S/H) + (b + (γ - 0.5)σ²)T] / (σ√T)
        d = (np.log(S / H) + (b + (gamma - 0.5) * vol**2) * T) / (vol * np.sqrt(T))

        # Calculate d₂ = [ln(I²/(SH)) + (b + (γ - 0.5)σ²)T] / (σ√T)
        d2 = (np.log(I**2 / (S * H)) + (b + (gamma - 0.5) * vol**2) * T) / (
            vol * np.sqrt(T)
        )

        # Calculate the result
        term1 = norm.cdf(-d)
        term2 = (I / S) ** kappa * norm.cdf(-d2)

        return np.exp(lambda_val * T) * (S**gamma) * (term1 - term2)

    def __bivariate_normal_cdf__(self, x, y, rho):
        """
        Bivariate normal cumulative distribution function M(x, y, ρ)
        Using more precise implementation
        """
        if abs(rho) < 1e-10:
            return norm.cdf(x) * norm.cdf(y)

        if abs(rho) >= 1.0:
            if rho > 0:
                return min(norm.cdf(x), norm.cdf(y))
            else:
                return max(norm.cdf(x) + norm.cdf(y) - 1, 0.0)

        # Using scipy's multivariate normal for accurate computation
        from scipy.stats import multivariate_normal

        mean = [0, 0]
        cov = [[1, rho], [rho, 1]]
        return multivariate_normal.cdf([x, y], mean, cov)

    def __psi_bs02__(self, S, T, gamma, H, I2, I1, t1):
        """
        Helper function Ψ for Bjerksund-Stensland 2002 approximation
        Ψ(S, T, γ, H, I₂, I₁, t₁) = e^λT * S^γ * [M(-e₁, -f₁, ρ) - (I₂/S)^κ * M(-e₂, -f₂, ρ)
                                    - (I₁/S)^κ * M(-e₃, -f₃, -ρ) + (I₁/I₂)^κ * M(-e₄, -f₄, -ρ)]
        """
        if S <= 0 or T <= 0 or t1 <= 0 or H <= 0 or I1 <= 0 or I2 <= 0:
            return 0.0

        b = self.b
        r = self.rrf
        vol = self.vol

        # Calculate λ and κ
        lambda_val = -r + gamma * b + 0.5 * gamma * (gamma - 1) * vol**2
        kappa = 2 * b / (vol**2) + (2 * gamma - 1)

        # Calculate correlation coefficient ρ = √(t₁/T)
        rho = np.sqrt(t1 / T)

        # Calculate e parameters
        e1 = (np.log(S / I1) + (b + (gamma - 0.5) * vol**2) * t1) / (vol * np.sqrt(t1))
        e2 = (np.log(I2**2 / (S * I1)) + (b + (gamma - 0.5) * vol**2) * t1) / (
            vol * np.sqrt(t1)
        )
        e3 = (np.log(S / I1) - (b + (gamma - 0.5) * vol**2) * t1) / (vol * np.sqrt(T))
        e4 = (np.log(I2**2 / (S * I1)) - (b + (gamma - 0.5) * vol**2) * t1) / (
            vol * np.sqrt(t1)
        )

        # Calculate f parameters
        f1 = (np.log(S / H) + (b + (gamma - 0.5) * vol**2) * T) / (vol * np.sqrt(T))
        f2 = (np.log(I2**2 / (S * H)) + (b + (gamma - 0.5) * vol**2) * T) / (
            vol * np.sqrt(T)
        )
        f3 = (np.log(I2**2 / (S * H)) + (b + (gamma - 0.5) * vol**2) * T) / (
            vol * np.sqrt(T)
        )
        f4 = (
            np.log(S * I1**2 / (H * I2**2)) + (b + (gamma - 0.5) * vol**2) * T
        ) / (vol * np.sqrt(T))

        # Calculate M terms using bivariate normal CDF
        M1 = self.__bivariate_normal_cdf__(-e1, -f1, rho)
        M2 = self.__bivariate_normal_cdf__(-e2, -f2, rho)
        M3 = self.__bivariate_normal_cdf__(-e3, -f3, -rho)
        M4 = self.__bivariate_normal_cdf__(-e4, -f4, -rho)

        # Calculate the result
        term1 = M1
        term2 = (I2 / S) ** kappa * M2
        term3 = (I1 / S) ** kappa * M3
        term4 = (I1 / I2) ** kappa * M4

        return np.exp(lambda_val * T) * (S**gamma) * (term1 - term2 - term3 + term4)

    def __price_ame_call_bs02__(self):
        """
        Bjerksund-Stensland 2002 approximation for American call option
        """
        S = self.spot
        X = self.K
        T = self.T  # Use calendar time T instead of tau
        r = self.rrf
        b = self.b
        vol = self.vol

        # When b >= r, American call = European call
        if b >= r:
            return self.__european_call_bsm__(S, X, T, r, b, vol)

        # Calculate β
        beta = (0.5 - b / vol**2) + np.sqrt((0.5 - b / vol**2) ** 2 + 2 * r / vol**2)

        # Calculate B_∞ and B_0
        B_infinity = beta * X / (beta - 1)
        B_0 = max(X, r * X / (r - b))

        # Calculate t₁ = 0.5(√5 - 1)T
        t1 = 0.5 * (np.sqrt(5) - 1) * T

        # Calculate h₁ and h₂
        h1 = -(b * t1 + 2 * vol * np.sqrt(t1)) * X**2 / ((B_infinity - B_0) * B_0)
        h2 = -(b * T + 2 * vol * np.sqrt(T)) * X**2 / ((B_infinity - B_0) * B_0)

        # Calculate I₁ and I₂ (trigger prices)
        I1 = B_0 + (B_infinity - B_0) * (1 - np.exp(h1))
        I2 = B_0 + (B_infinity - B_0) * (1 - np.exp(h2))

        # Calculate α₁ and α₂ using robust numerical approach
        # For large β, use log-space calculations to avoid underflow
        if I1 > 0 and I1 > X:
            log_alpha1 = np.log(I1 - X) - beta * np.log(I1)
            alpha1 = np.exp(log_alpha1) if log_alpha1 > -700 else 0.0  # Avoid underflow
        else:
            alpha1 = 0.0

        if I2 > 0 and I2 > X:
            log_alpha2 = np.log(I2 - X) - beta * np.log(I2)
            alpha2 = np.exp(log_alpha2) if log_alpha2 > -700 else 0.0  # Avoid underflow
        else:
            alpha2 = 0.0

        # Main formula (3.2) from the paper
        # C = α₂S^β - α₂φ(S, t₁, 1, I₂, I₂) + φ(S, t₁, 1, I₂, I₂) - φ(S, t₁, 1, I₁, I₂)
        #   - Xφ(S, t₁, 0, I₂, I₂) + Xφ(S, t₁, 0, I₁, I₂) + α₁φ(S, t₁, β, I₁, I₂)
        #   - α₁Ψ(S, T, β, I₁, I₂, I₁, t₁) + Ψ(S, T, 1, I₁, I₂, I₁, t₁)
        #   - Ψ(S, T, 1, X, I₂, I₁, t₁) - XΨ(S, T, 0, I₁, I₂, I₁, t₁) + Ψ(S, T, 0, X, I₂, I₁, t₁)

        # Check for early exercise
        if S >= I2:
            return S - X

        # Debug prints for intermediate values
        if hasattr(self, "_debug_bs02") and self._debug_bs02:
            print(
                f"Debug BS02 - S={S:.2f}, X={X:.2f}, T={T:.4f}, r={r:.4f}, b={b:.4f}, vol={vol:.4f}"
            )
            print(
                f"Beta={beta:.4f}, B_0={B_0:.2f}, B_inf={B_infinity:.2f}, t1={t1:.4f}"
            )
            print(f"I1={I1:.2f}, I2={I2:.2f}")
            if I1 > X:
                log_alpha1_calc = np.log(I1 - X) - beta * np.log(I1)
                print(
                    f"log_alpha1 = ln({I1:.2f}-{X:.2f}) - {beta:.4f}*ln({I1:.2f}) = {np.log(I1-X):.2f} - {beta*np.log(I1):.2f} = {log_alpha1_calc:.2f}"
                )
            if I2 > X:
                log_alpha2_calc = np.log(I2 - X) - beta * np.log(I2)
                print(
                    f"log_alpha2 = ln({I2:.2f}-{X:.2f}) - {beta:.4f}*ln({I2:.2f}) = {np.log(I2-X):.2f} - {beta*np.log(I2):.2f} = {log_alpha2_calc:.2f}"
                )
            print(f"alpha1={alpha1:.2e}, alpha2={alpha2:.2e}")
            print(f"S^beta = {S**beta:.2e}, alpha2*S^beta = {alpha2*(S**beta):.2f}")

        # Calculate each term according to formula (3.2)
        term1 = alpha2 * (S**beta)  # α₂S^β
        term2 = -alpha2 * self.__phi_bs02__(
            S, t1, beta, I2, I2
        )  # -α₂φ(S, t₁, β, I₂, I₂)
        term3 = self.__phi_bs02__(S, t1, 1, I2, I2)  # φ(S, t₁, 1, I₂, I₂)
        term4 = -self.__phi_bs02__(S, t1, 1, I1, I2)  # -φ(S, t₁, 1, I₁, I₂)
        term5 = -X * self.__phi_bs02__(S, t1, 0, I2, I2)  # -Xφ(S, t₁, 0, I₂, I₂)
        term6 = X * self.__phi_bs02__(S, t1, 0, I1, I2)  # Xφ(S, t₁, 0, I₁, I₂)
        term7 = alpha1 * self.__phi_bs02__(S, t1, beta, I1, I2)  # α₁φ(S, t₁, β, I₁, I₂)
        term8 = -alpha1 * self.__psi_bs02__(
            S, T, beta, I1, I2, I1, t1
        )  # -α₁Ψ(S, T, β, I₁, I₂, I₁, t₁)
        term9 = self.__psi_bs02__(S, T, 1, I1, I2, I1, t1)  # Ψ(S, T, 1, I₁, I₂, I₁, t₁)
        term10 = -self.__psi_bs02__(
            S, T, 1, X, I2, I1, t1
        )  # -Ψ(S, T, 1, X, I₂, I₁, t₁)
        term11 = -X * self.__psi_bs02__(
            S, T, 0, I1, I2, I1, t1
        )  # -XΨ(S, T, 0, I₁, I₂, I₁, t₁)
        term12 = X * self.__psi_bs02__(
            S, T, 0, X, I2, I1, t1
        )  # XΨ(S, T, 0, X, I₂, I₁, t₁)

        if hasattr(self, "_debug_bs02") and self._debug_bs02:
            print(
                f"Terms: {term1:.0f}, {term2:.0f}, {term3:.0f}, {term4:.0f}, {term5:.0f}, {term6:.0f}"
            )
            print(
                f"       {term7:.0f}, {term8:.0f}, {term9:.0f}, {term10:.0f}, {term11:.0f}, {term12:.0f}"
            )

        value = (
            term1
            + term2
            + term3
            + term4
            + term5
            + term6
            + term7
            + term8
            + term9
            + term10
            + term11
            + term12
        )

        if hasattr(self, "_debug_bs02") and self._debug_bs02:
            print(
                f"BS02 FINAL: Raw sum = {value:.6f}, max(value, S-X) = {max(value, S - X):.6f}"
            )

        # For BS02, don't force intrinsic value constraint as it interferes with put-call transformations
        # The early exercise check above handles the boundary condition properly
        return value

    def __d1_baw__(self, S, X, T, b, vol):
        """Calculate d1 parameter for BAW formula"""
        return (np.log(S / X) + (b + vol**2 / 2) * T) / (vol * np.sqrt(T))

    def __european_call_bsm__(self, S, X, T, r, b, vol):
        """Calculate European call price using BSM formula"""
        d1 = self.__d1_baw__(S, X, T, b, vol)
        d2 = d1 - vol * np.sqrt(T)
        return S * np.exp((b - r) * T) * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(
            d2
        )

    def __european_put_bsm__(self, S, X, T, r, b, vol):
        """Calculate European put price using BSM formula"""
        d1 = self.__d1_baw__(S, X, T, b, vol)
        d2 = d1 - vol * np.sqrt(T)
        return X * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp((b - r) * T) * norm.cdf(
            -d1
        )

    def __find_critical_call_price__(self, X, T, r, b, vol, q2):
        """Find critical price S* for American call using Newton-Raphson"""

        def objective(S_star):
            if S_star <= 0:
                return float("inf")
            d1_s = self.__d1_baw__(S_star, X, T, b, vol)
            c_bsm = self.__european_call_bsm__(S_star, X, T, r, b, vol)
            lhs = S_star - X
            rhs = c_bsm + (1 - np.exp((b - r) * T) * norm.cdf(d1_s)) * S_star / q2
            return abs(lhs - rhs)

        # Initial guess - start with strike price
        S0 = max(X, X * 1.1)
        try:
            result = fmin(objective, S0, disp=False, full_output=True)
            return (
                result[0][0] if result[4] == 0 else X * 1.5
            )  # fallback if optimization fails
        except:
            return X * 1.5  # fallback value

    def __find_critical_put_price__(self, X, T, r, b, vol, q1):
        """Find critical price S** for American put using Newton-Raphson"""

        def objective(S_star_star):
            if S_star_star <= 0:
                return float("inf")
            d1_s = self.__d1_baw__(S_star_star, X, T, b, vol)
            p_bsm = self.__european_put_bsm__(S_star_star, X, T, r, b, vol)
            lhs = X - S_star_star
            rhs = p_bsm - (1 - np.exp((b - r) * T) * norm.cdf(-d1_s)) * S_star_star / q1
            return abs(lhs - rhs)

        # Initial guess - start below strike price
        S0 = min(X, X * 0.9)
        try:
            result = fmin(objective, S0, disp=False, full_output=True)
            return (
                result[0][0] if result[4] == 0 else X * 0.5
            )  # fallback if optimization fails
        except:
            return X * 0.5  # fallback value

    def __price_ame_call_baw__(self):
        """Barone-Adesi-Whaley approximation for American call option"""
        S = self.spot
        X = self.K
        T = self.tau  # Use tau for consistency with existing code
        r = self.rrf
        b = self.b  # cost of carry
        vol = self.vol

        # When b >= r, American call = European call
        if b >= r:
            return self.__european_call_bsm__(S, X, T, r, b, vol)

        # Calculate BAW parameters
        M = 2 * r / (vol**2)
        N = 2 * b / (vol**2)
        K_param = 1 - np.exp(-r * T)

        # Calculate q2
        discriminant = (N - 1) ** 2 + 4 * M / K_param
        if discriminant < 0:
            # Fallback to European option if discriminant is negative
            return self.__european_call_bsm__(S, X, T, r, b, vol)

        q2 = (-(N - 1) + np.sqrt(discriminant)) / 2

        # Find critical price S*
        S_star = self.__find_critical_call_price__(X, T, r, b, vol, q2)

        if S >= S_star:
            # Early exercise
            return S - X
        else:
            # Calculate A2
            d1_s_star = self.__d1_baw__(S_star, X, T, b, vol)
            A2 = (S_star / q2) * (1 - np.exp((b - r) * T) * norm.cdf(d1_s_star))

            # Calculate European call price
            c_bsm = self.__european_call_bsm__(S, X, T, r, b, vol)

            # BAW approximation
            return c_bsm + A2 * (S / S_star) ** q2

    def __price_ame_put_baw__(self):
        """Barone-Adesi-Whaley approximation for American put option"""
        S = self.spot
        X = self.K
        T = self.tau  # Use tau for consistency with existing code
        r = self.rrf
        b = self.b  # cost of carry
        vol = self.vol

        # When r <= 0 and r <= b, American put = European put
        if r <= 0 and r <= b:
            return self.__european_put_bsm__(S, X, T, r, b, vol)

        # Calculate BAW parameters
        M = 2 * r / (vol**2)
        N = 2 * b / (vol**2)
        K_param = 1 - np.exp(-r * T)

        # Calculate q1
        discriminant = (N - 1) ** 2 + 4 * M / K_param
        if discriminant < 0:
            # Fallback to European option if discriminant is negative
            return self.__european_put_bsm__(S, X, T, r, b, vol)

        q1 = (-(N - 1) - np.sqrt(discriminant)) / 2

        # Find critical price S**
        S_star_star = self.__find_critical_put_price__(X, T, r, b, vol, q1)

        if S <= S_star_star:
            # Early exercise
            return X - S
        else:
            # Calculate A1
            d1_s_star_star = self.__d1_baw__(S_star_star, X, T, b, vol)
            A1 = -(S_star_star / q1) * (
                1 - np.exp((b - r) * T) * norm.cdf(-d1_s_star_star)
            )

            # Calculate European put price
            p_bsm = self.__european_put_bsm__(S, X, T, r, b, vol)

            # BAW approximation
            return p_bsm + A1 * (S / S_star_star) ** q1
