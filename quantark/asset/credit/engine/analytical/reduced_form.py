"""
CDS pricing engine for a reduced-form model (hazard read from the curve).

The premium and protection legs are valued by numerical integration of the
survival and discount curves supplied by the
:class:`~quantark.priceenv.CreditPricingEnvironment`. The hazard curve may be
flat (constant intensity) or term-structured; this engine reads survival and
default-density from the curve and so handles both.
"""
import math
from typing import Dict, List, Tuple

import numpy as np

from quantark.asset.credit.engine.base_credit_engine import BaseCreditEngine
from quantark.asset.credit.engine.schedule import (
    cds_coupon_schedule,
    cds_coupon_schedule_asof,
    year_fraction_act365,
)
from quantark.asset.credit.product.cds import CDS
from quantark.priceenv import CreditPricingEnvironment
from quantark.util.exceptions import NumericalError

# Daily integration step (years) for the protection leg and premium accrual.
_DT = 1.0 / 365.0
# Remaining maturities below this (years) are treated as a matured contract.
_EPS = 1e-9

# Zero result returned for a matured / expired CDS.
_EXPIRED: Dict[str, float] = {
    "premium_leg": 0.0,
    "protection_leg": 0.0,
    "fair_spread": 0.0,
    "fair_spread_bps": 0.0,
    "present_value": 0.0,
}


def _time_grid(start: float, end: float) -> np.ndarray:
    """Integration grid on ``[start, end]`` ending exactly at ``end``.

    ``start`` is the protection-start offset (years from the valuation date),
    nonzero only for a forward-starting contract whose effective date lies in
    the future; it is zero for a spot or seasoned trade.
    """
    span = max(end - start, 0.0)
    n_steps = max(1, int(math.ceil(span / _DT)))
    return np.linspace(start, end, n_steps + 1)


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

        For a *dated* contract (``product.effective_date`` set) the contract is
        valued as of ``env.valuation_date``: settled coupons are excluded, the
        remaining maturity is measured from the contractual dates, and a matured
        contract values to zero.
        """
        schedule, prot_start, prot_end, as_of_time = self._as_of_params(product, env)
        if prot_end <= prot_start + _EPS:  # matured — nothing left to value
            return dict(_EXPIRED)

        protection_leg = self._protection_leg(product, env, prot_start, prot_end)
        rpv01 = self._premium_leg(
            product, env, 1.0, schedule, prot_start, prot_end, as_of_time
        )
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
    # As-of resolution
    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_of_params(
        product: CDS, env: CreditPricingEnvironment
    ) -> Tuple[List[Tuple[float, float]], float, float, float]:
        """
        Resolve ``(schedule, protection_start, protection_end, as_of_time)`` for
        valuation, with all times measured *from the valuation date* (consistent
        with the as-of-dated discount and survival curves).

        * ``protection_start`` — years until protection begins. Zero for a spot
          or seasoned trade; positive for a forward-starting contract whose
          effective date is still in the future.
        * ``protection_end`` — years until the contract matures.
        * ``as_of_time`` — *signed* seasoning: years elapsed since the effective
          date (negative before the effective date). Used to anchor the
          accrued-on-default leg to the original coupon calendar.

        For an un-dated contract this is the legacy tenor-based schedule starting
        immediately (``protection_start = 0``, ``as_of_time = 0``).
        """
        if not product.is_dated:
            schedule = cds_coupon_schedule(product.maturity, product.payment_freq)
            return schedule, 0.0, product.maturity, 0.0

        as_of_time = year_fraction_act365(product.effective_date, env.valuation_date)
        total = year_fraction_act365(product.effective_date, product.maturity_date)
        protection_end = total - as_of_time
        protection_start = max(0.0, -as_of_time)  # forward-start delay, else 0
        if protection_end <= protection_start + _EPS:
            return [], protection_start, protection_end, as_of_time
        schedule = cds_coupon_schedule_asof(
            product.effective_date,
            product.maturity_date,
            product.payment_freq,
            env.valuation_date,
        )
        return schedule, protection_start, protection_end, as_of_time

    # ------------------------------------------------------------------ #
    # Legs
    # ------------------------------------------------------------------ #
    def _protection_leg(
        self,
        product: CDS,
        env: CreditPricingEnvironment,
        prot_start: float,
        prot_end: float,
    ) -> float:
        """PV of the default-contingent payment N*(1-R)*E[DF * default density].

        Integrated only over the live protection window ``[prot_start, prot_end]``
        (times from the valuation date). ``prot_start`` is nonzero only for a
        forward-starting contract, so no protection is credited before the
        effective date.
        """
        lgd = product.notional * (1.0 - product.recovery_rate)
        grid = _time_grid(prot_start, prot_end)
        pv = 0.0
        for i in range(1, len(grid)):
            t = float(grid[i])
            density = env.get_default_density(t)
            discount = env.get_discount_factor(t)
            pv += lgd * discount * density * (t - float(grid[i - 1]))
        return pv

    def _premium_leg(
        self,
        product: CDS,
        env: CreditPricingEnvironment,
        spread: float,
        schedule: List[Tuple[float, float]],
        prot_start: float,
        prot_end: float,
        as_of_time: float,
    ) -> float:
        """
        PV of the premium leg for a given running ``spread``.

        Linear in ``spread``; calling with ``spread=1.0`` yields the risky
        annuity (RPV01). Includes scheduled coupons (survival-weighted) plus
        premium accrued from the last coupon date to the default date. ``schedule``
        already holds only the unsettled coupons with payment times relative to
        the valuation date; ``as_of_time`` is the signed seasoning offset so the
        accrued-on-default leg references the correct contractual coupon dates,
        and accrual is only credited inside the live protection window.
        """
        notional = product.notional
        payment_interval = 1.0 / product.payment_freq

        # Scheduled coupon payments, conditional on survival. The schedule
        # includes the final (possibly short) stub coupon ending at maturity,
        # and each coupon's accrual is the actual period since the prior coupon.
        pv_scheduled = 0.0
        for t, accrual in schedule:
            survival = env.get_survival_probability(t)
            discount = env.get_discount_factor(t)
            pv_scheduled += spread * notional * accrual * discount * survival

        # Premium accrued to the default date (paid on default). The default
        # time ``t`` is measured from valuation; the contractual elapsed time is
        # ``as_of_time + t`` so the last coupon date is found on the original
        # calendar. Accrual is credited only over the live protection window
        # (a forward-starting trade accrues nothing before its effective date).
        grid = _time_grid(prot_start, prot_end)
        pv_accrued = 0.0
        for i in range(1, len(grid)):
            t = float(grid[i])
            contractual_t = as_of_time + t
            last_payment_time = payment_interval * int(contractual_t / payment_interval)
            if last_payment_time >= contractual_t:
                last_payment_time -= payment_interval
            accrual_period = contractual_t - last_payment_time
            density = env.get_default_density(t)
            discount = env.get_discount_factor(t)
            accrued = spread * notional * accrual_period
            pv_accrued += accrued * discount * density * (t - float(grid[i - 1]))

        return pv_scheduled + pv_accrued
