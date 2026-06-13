"""
CDS pricing engine for a reduced-form model (hazard read from the curve).

The premium and protection legs are valued by numerical integration of the
survival and discount curves supplied by the
:class:`~quantark.priceenv.CreditPricingEnvironment`. The hazard curve may be
flat (constant intensity) or term-structured; this engine reads survival and
default-density from the curve and so handles both.
"""
from typing import Dict

import numpy as np

from quantark.asset.credit.engine.base_credit_engine import BaseCreditEngine
from quantark.asset.credit.product.cds import CDS
from quantark.priceenv import CreditPricingEnvironment
from quantark.util.exceptions import NumericalError

# Daily integration step (years) for the protection leg and premium accrual.
_DT = 1.0 / 365.0


class CDSReducedFormEngine(BaseCreditEngine):
    """Reduced-form CDS pricing engine valuing premium and protection legs."""

    def price(self, product: CDS, env: CreditPricingEnvironment) -> float:
        """Present value to the protection holder per ``product.side``."""
        res = self.calculate(product, env)
        return product.side_sign * res["present_value"]

    def calculate(self, product: CDS, env: CreditPricingEnvironment) -> Dict[str, float]:
        """
        Value both legs and the fair spread.

        Returns a dict with ``premium_leg``, ``protection_leg``,
        ``fair_spread``, ``fair_spread_bps`` and ``present_value`` — the last
        always from the protection *buyer*'s perspective
        (``protection_leg - premium_leg``).
        """
        protection_leg = self._protection_leg(product, env)
        rpv01 = self._premium_leg(product, env, spread=1.0)
        if rpv01 <= 0:
            raise NumericalError("CDS risky annuity (RPV01) is non-positive")

        premium_leg = product.coupon_spread * rpv01
        present_value = protection_leg - premium_leg
        fair_spread = protection_leg / rpv01
        return {
            "premium_leg": premium_leg,
            "protection_leg": protection_leg,
            "fair_spread": fair_spread,
            "fair_spread_bps": fair_spread * 10_000.0,
            "present_value": present_value,
        }

    def fair_spread(self, product: CDS, env: CreditPricingEnvironment) -> float:
        """Par (fair) running spread that sets the contract PV to zero."""
        return self.calculate(product, env)["fair_spread"]

    # ------------------------------------------------------------------ #
    # Legs
    # ------------------------------------------------------------------ #
    def _protection_leg(self, product: CDS, env: CreditPricingEnvironment) -> float:
        """PV of the default-contingent payment N*(1-R)*E[DF * default density]."""
        lgd = product.notional * (1.0 - product.recovery_rate)
        grid = np.arange(0.0, product.maturity + _DT, _DT)
        pv = 0.0
        for i in range(1, len(grid)):
            t = float(grid[i])
            density = env.get_default_density(t)
            discount = env.get_discount_factor(t)
            pv += lgd * discount * density * (t - float(grid[i - 1]))
        return pv

    def _premium_leg(
        self, product: CDS, env: CreditPricingEnvironment, spread: float
    ) -> float:
        """
        PV of the premium leg for a given running ``spread``.

        Linear in ``spread``; calling with ``spread=1.0`` yields the risky
        annuity (RPV01). Includes scheduled coupons (survival-weighted) plus
        premium accrued from the last coupon date to the default date.
        """
        notional = product.notional
        payment_interval = 1.0 / product.payment_freq

        # Scheduled coupon payments, conditional on survival.
        payment_times = np.arange(
            payment_interval, product.maturity + payment_interval / 2, payment_interval
        )
        payment_times = payment_times[payment_times <= product.maturity]
        pv_scheduled = 0.0
        for t in payment_times:
            t = float(t)
            survival = env.get_survival_probability(t)
            discount = env.get_discount_factor(t)
            pv_scheduled += spread * notional * payment_interval * discount * survival

        # Premium accrued to the default date (paid on default).
        grid = np.arange(0.0, product.maturity + _DT, _DT)
        pv_accrued = 0.0
        for i in range(1, len(grid)):
            t = float(grid[i])
            last_payment_time = payment_interval * int(t / payment_interval)
            if last_payment_time >= t:
                last_payment_time -= payment_interval
            accrual_period = t - last_payment_time
            density = env.get_default_density(t)
            discount = env.get_discount_factor(t)
            accrued = spread * notional * accrual_period
            pv_accrued += accrued * discount * density * (t - float(grid[i - 1]))

        return pv_scheduled + pv_accrued
