"""
Basket-CDS Monte-Carlo pricing engine using copula default-time models.

Correlated default times are drawn from a Gaussian or Student-t copula and the
FtD / NtD / tranche protection and premium legs are valued by simulation.
The per-simulation legs are vectorised across paths; only the tranche
waterfall, which is inherently path-sequential, loops over simulations.
"""
from typing import Dict, Optional

import numpy as np
from scipy.stats import norm, t as student_t

from quantark.asset.credit.engine.base_credit_engine import BaseCreditEngine
from quantark.asset.credit.product.basket_cds import BasketCDS, BasketType, CopulaType
from quantark.param.credit import FlatHazardCurve
from quantark.param.rrf import FlatRateCurve
from quantark.priceenv import BasketCreditPricingEnvironment
from quantark.util.exceptions import NumericalError

_U_EPS = 1e-12


class BasketCDSEngine(BaseCreditEngine):
    """Copula Monte-Carlo pricing engine for basket CDS."""

    def __init__(self, n_simulations: int = 10_000, seed: Optional[int] = None):
        self.n_simulations = n_simulations
        self.seed = seed

    def price(self, product: BasketCDS, env: BasketCreditPricingEnvironment) -> float:
        """Present value to the protection holder per ``product.side``."""
        return product.side_sign * self.calculate(product, env)["present_value"]

    def calculate(
        self, product: BasketCDS, env: BasketCreditPricingEnvironment
    ) -> Dict[str, float]:
        """Value both legs and the fair spread (buyer's perspective PV)."""
        default_times = self._simulate_default_times(product, env)
        protection_leg = self._protection_leg(product, env, default_times)
        rpv01 = self._premium_leg(product, env, default_times, spread=1.0)
        if rpv01 <= 0:
            raise NumericalError("Basket CDS risky annuity (RPV01) is non-positive")

        premium_leg = product.coupon_spread * rpv01
        present_value = protection_leg - premium_leg
        fair_spread = protection_leg / rpv01
        return {
            "premium_leg": premium_leg,
            "protection_leg": protection_leg,
            "fair_spread": fair_spread,
            "fair_spread_bps": fair_spread * 10_000.0,
            "present_value": present_value,
            "n_simulations": self.n_simulations,
        }

    def fair_spread(
        self, product: BasketCDS, env: BasketCreditPricingEnvironment
    ) -> float:
        return self.calculate(product, env)["fair_spread"]

    # ------------------------------------------------------------------ #
    # Simulation
    # ------------------------------------------------------------------ #
    def _simulate_default_times(
        self, product: BasketCDS, env: BasketCreditPricingEnvironment
    ) -> np.ndarray:
        """Correlated default times, shape (n_simulations, n_entities)."""
        rng = np.random.default_rng(self.seed)
        n = product.n_entities
        corr = np.asarray(product.correlation_matrix, dtype=float)
        chol = np.linalg.cholesky(corr)
        z = rng.standard_normal((self.n_simulations, n)) @ chol.T

        if product.copula_type == CopulaType.STUDENT_T:
            df = product.student_t_df
            chi2 = rng.chisquare(df, self.n_simulations)[:, None]
            u = student_t.cdf(z * np.sqrt(df / chi2), df)
        else:
            u = norm.cdf(z)
        u = np.clip(u, _U_EPS, 1.0 - _U_EPS)

        taus = np.empty_like(u)
        horizon = product.maturity * 3.0
        for i in range(n):
            taus[:, i] = self._inverse_survival(env.hazard_curves[i], u[:, i], horizon)
        return taus

    @staticmethod
    def _inverse_survival(curve, u: np.ndarray, horizon: float) -> np.ndarray:
        """Default time tau such that S(tau) = u (vectorised over u)."""
        if isinstance(curve, FlatHazardCurve):
            lam = curve.hazard_rate
            if lam <= 0:
                return np.full_like(u, np.inf)
            return -np.log(u) / lam
        grid = np.linspace(0.0, horizon, 600)
        surv = np.array([curve.get_survival_probability(float(t)) for t in grid])
        # surv is decreasing; reverse so the interp abscissa is increasing.
        return np.interp(u, surv[::-1], grid[::-1], left=grid[-1], right=0.0)

    # ------------------------------------------------------------------ #
    # Discounting
    # ------------------------------------------------------------------ #
    @staticmethod
    def _discount(env: BasketCreditPricingEnvironment, t: np.ndarray) -> np.ndarray:
        dc = env.discount_curve
        if isinstance(dc, FlatRateCurve):
            return np.exp(-dc.rate * np.asarray(t, dtype=float))
        return np.array([dc.get_discount_factor(float(x)) for x in np.atleast_1d(t)])

    # ------------------------------------------------------------------ #
    # Legs
    # ------------------------------------------------------------------ #
    def _trigger(self, product: BasketCDS, default_times: np.ndarray):
        """(trigger_time, triggering_entity_index) for FTD/NTD per simulation."""
        n_rank = 1 if product.basket_type == BasketType.FTD else product.n_to_default
        order = np.argsort(default_times, axis=1)
        trig_entity = order[:, n_rank - 1]
        trig_time = np.take_along_axis(
            default_times, trig_entity[:, None], axis=1
        ).ravel()
        return trig_time, trig_entity

    def _protection_leg(
        self, product: BasketCDS, env, default_times: np.ndarray
    ) -> float:
        if product.basket_type == BasketType.TRANCHE:
            return self._tranche_protection(product, env, default_times)
        trig_time, trig_entity = self._trigger(product, default_times)
        triggered = trig_time <= product.maturity
        recovery = np.asarray(product.recovery_rates)[trig_entity]
        lgd = product.notional * (1.0 - recovery)
        df = self._discount(env, np.where(triggered, trig_time, 0.0))
        pv = np.where(triggered, lgd * df, 0.0)
        return float(pv.mean())

    def _premium_leg(
        self, product: BasketCDS, env, default_times: np.ndarray, spread: float
    ) -> float:
        if product.basket_type == BasketType.TRANCHE:
            return self._tranche_premium(product, env, default_times, spread)
        trig_time, _ = self._trigger(product, default_times)
        triggered = trig_time <= product.maturity
        pi = 1.0 / product.payment_freq

        dates = np.arange(pi, product.maturity + pi / 2, pi)
        dates = dates[dates <= product.maturity]
        df_dates = self._discount(env, dates)

        # Scheduled coupons survive to a payment date when no trigger precedes it.
        survived = trig_time[:, None] >= dates[None, :]  # (n_sim, n_dates)
        scheduled = (survived * (pi * df_dates)[None, :]).sum(axis=1)

        # Accrual from the last coupon date to the trigger time, paid on trigger.
        last_coupon = np.floor(trig_time / pi) * pi
        accrual_period = np.where(triggered, trig_time - last_coupon, 0.0)
        df_trig = self._discount(env, np.where(triggered, trig_time, 0.0))
        accrued = np.where(triggered, accrual_period * df_trig, 0.0)

        annuity = float((scheduled + accrued).mean())
        return spread * product.notional * annuity

    # ------------------------------------------------------------------ #
    # Tranche (path-sequential waterfall)
    # ------------------------------------------------------------------ #
    def _tranche_protection(self, product, env, default_times: np.ndarray) -> float:
        attach, detach = product.attachment_point, product.detachment_point
        weights = np.ones(product.n_entities) / product.n_entities
        recoveries = np.asarray(product.recovery_rates)
        total = 0.0
        for sim in range(self.n_simulations):
            taus = default_times[sim]
            order = np.argsort(taus)
            cum = 0.0
            pv = 0.0
            for idx in order:
                if taus[idx] > product.maturity:
                    break
                loss = weights[idx] * (1.0 - recoveries[idx])  # fraction of notional
                prev = cum
                cum += loss
                impact = max(0.0, min(cum, detach) - max(prev, attach))
                if impact > 0:
                    df = self._discount(env, np.array([taus[idx]]))[0]
                    pv += impact * product.notional * df
            total += pv
        return total / self.n_simulations

    def _tranche_premium(self, product, env, default_times: np.ndarray, spread) -> float:
        attach, detach = product.attachment_point, product.detachment_point
        thickness = detach - attach
        weights = np.ones(product.n_entities) / product.n_entities
        recoveries = np.asarray(product.recovery_rates)
        pi = 1.0 / product.payment_freq
        dates = np.arange(pi, product.maturity + pi / 2, pi)
        dates = dates[dates <= product.maturity]

        total = 0.0
        for sim in range(self.n_simulations):
            taus = default_times[sim]
            order = sorted(range(product.n_entities), key=lambda j: taus[j])
            cum = 0.0
            remaining = 1.0
            pv = 0.0
            ptr = 0
            for d in dates:
                while ptr < len(order) and taus[order[ptr]] <= d and taus[order[ptr]] <= product.maturity:
                    idx = order[ptr]
                    loss = weights[idx] * (1.0 - recoveries[idx])
                    prev = cum
                    cum += loss
                    impact = max(0.0, min(cum, detach) - max(prev, attach))
                    if impact > 0:
                        new_remaining = max(0.0, (detach - max(cum, attach)) / thickness)
                        remaining = new_remaining
                    ptr += 1
                    if remaining <= 0:
                        break
                if remaining <= 0:
                    break
                df = self._discount(env, np.array([d]))[0]
                pv += spread * product.notional * thickness * remaining * pi * df
            total += pv
        return total / self.n_simulations
