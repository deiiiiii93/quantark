"""The statistics readout's dataclass-and-ledger assembly (spec 3B §5.1, §5.2).

Shared by the legacy group (moved verbatim from its ``read``) and by the knocked-out-at-valuation
statistics (Task 9).
"""
from __future__ import annotations

import numpy as np

from quantark.asset.equity.engine.event_stats import payment_aware_cashflow_fields
from quantark.asset.equity.engine.settlement_support import resolve_terminal_timing
from quantark.util.enum import CouponPayType

__all__ = ["assemble_autocallable"]


def assemble_autocallable(solver, product, env, *, pv, ko_records, ed_unit, ki, extra_fields,
                          maturity=None, coupon_rows=None):
    """Probabilities, the maturity leg and the payment-aware ledger from the read columns.

    ``ed_unit`` is the discounted knock-out unit per record; ``ki`` is ``(ki_probability,
    ki_ever_probability, ki_times, ki_event_probability, ki_survival_probability)``;
    ``extra_fields`` the family's extra readout (the Phoenix coupon fields).

    ``maturity`` (spec 3B §5.2): the marched maturity leg; the leg is then that value and
    ``reconciliation_error = pv - Σ ko - Σ coupon - maturity``. ``None``: the legacy residual
    leg and an error of ``0.0``. ``coupon_rows``: ``(determination_times, payment_times, cash,
    determination_dates, payment_dates)``, the coupon ledger rows in place of the legacy block.
    """
    pricing_env = env
    ki_probability, ki_ever_probability, ki_times, ki_event_probability, ki_survival_probability = ki
    n_ko = len(ko_records)
    ko_times = np.array([float(rec.observation_time) for rec in ko_records], dtype=float)
    ko_probability = np.zeros(n_ko, dtype=float)
    ed_ko_cf = np.zeros(n_ko, dtype=float)

    for i, rec in enumerate(ko_records):
        obs_time = float(rec.observation_time)
        settle = rec.settlement_time if rec.settlement_time is not None else obs_time
        settle = float(settle)
        df0 = pricing_env.get_discount_factor(settle)
        if df0 > 0.0:
            ko_probability[i] = float(ed_unit[i] / df0)
        payoff = float(rec.payoff) if rec.payoff is not None else 0.0
        ed_ko_cf[i] = float(ed_unit[i] * payoff)

    survival_probability = np.ones(n_ko, dtype=float)
    cumulative = 0.0
    for i in range(n_ko):
        cumulative += ko_probability[i]
        survival_probability[i] = max(0.0, 1.0 - cumulative)

    pv = float(pv)   # the caller's: external, or the same march's value readout
    expected_discounted_maturity_cf = float(pv - float(np.sum(ed_ko_cf)))

    # The maturity cashflow is pv minus KO cashflows; for products with extra
    # cashflow streams (Phoenix coupons) also remove those so the decomposition
    # pv = sum(ko) + sum(coupon) + maturity stays correctly classified.
    coupon_cf = extra_fields.get("expected_discounted_coupon_cashflow")
    if coupon_cf is not None:
        expected_discounted_maturity_cf -= float(np.sum(coupon_cf))

    terminal = resolve_terminal_timing(product, pricing_env)
    determination_times = list(ko_times)
    payment_times = [
        float(
            record.settlement_time
            if record.settlement_time is not None
            else record.observation_time
        )
        for record in ko_records
    ]
    discounted_cashflows = list(ed_ko_cf)
    determination_dates = [
        record.observation_date for record in ko_records
    ]
    payment_dates = [
        record.settlement_date for record in ko_records
    ]

    coupon_ledger_cash = 0.0
    if coupon_rows is not None:
        c_det, c_pay, c_cash, c_det_dates, c_pay_dates = coupon_rows
        determination_times.extend(float(t) for t in c_det)
        payment_times.extend(float(t) for t in c_pay)
        discounted_cashflows.extend(float(c) for c in c_cash)
        determination_dates.extend(c_det_dates)
        payment_dates.extend(c_pay_dates)
        coupon_ledger_cash = float(np.sum(np.asarray(c_cash, dtype=float)))
    elif "coupon_probability" in extra_fields:
        coupon_cashflows = np.asarray(
            (
                extra_fields["expected_discounted_coupon_cashflow"]
                if coupon_cf is not None
                else np.zeros(n_ko, dtype=float)
            ),
            dtype=float,
        )
        coupon_at_expiry = (
            product.coupon_config.coupon_pay_type
            == CouponPayType.EXPIRY
        )
        determination_times.extend(ko_times)
        payment_times.extend(
            [
                (
                    float(terminal.payment_time)
                    if coupon_at_expiry
                    else float(
                        record.settlement_time
                        if record.settlement_time is not None
                        else record.observation_time
                    )
                )
                for record in ko_records
            ]
        )
        discounted_cashflows.extend(coupon_cashflows)
        determination_dates.extend(
            record.observation_date for record in ko_records
        )
        payment_dates.extend(
            (
                terminal.payment_date
                if coupon_at_expiry
                else record.settlement_date
            )
            for record in ko_records
        )

    reconciliation_error = 0.0
    if maturity is not None:
        # spec 3B §5.2: the marched leg, and the identity's residual reported, not absorbed
        coupon_cash = (coupon_ledger_cash if coupon_rows is not None
                       else (float(np.sum(coupon_cf)) if coupon_cf is not None else 0.0))
        expected_discounted_maturity_cf = float(maturity)
        reconciliation_error = float(pv - float(np.sum(ed_ko_cf)) - coupon_cash - expected_discounted_maturity_cf)

    determination_times.append(float(terminal.determination_time))
    payment_times.append(float(terminal.payment_time))
    discounted_cashflows.append(expected_discounted_maturity_cf)
    determination_dates.append(terminal.determination_date)
    payment_dates.append(terminal.payment_date)
    cashflow_fields = payment_aware_cashflow_fields(
        pricing_env,
        determination_times=determination_times,
        payment_times=payment_times,
        expected_discounted_cashflows=discounted_cashflows,
        determination_dates=determination_dates,
        payment_dates=payment_dates,
    )

    return solver._make_event_stats(
        pv=pv,
        ko_times=ko_times,
        ko_probability=ko_probability,
        survival_probability=survival_probability,
        expected_discounted_ko_cashflow=ed_ko_cf,
        ki_probability=ki_probability,
        expected_discounted_maturity_cashflow=expected_discounted_maturity_cf,
        reconciliation_error=reconciliation_error,
        ki_times=ki_times,
        ki_event_probability=ki_event_probability,
        ki_survival_probability=ki_survival_probability,
        # Two unambiguous, cross-engine-consistent KI fields. The legacy
        # `ki_probability` keeps the PDE's historical "settles knocked-in"
        # meaning (KI indicator absorbed to 0 on any KO), which equals
        # `ki_survive_knocked_in_probability`. `ki_ever_probability` comes from
        # the dedicated KI-ever column that carries no KO absorption.
        ki_ever_probability=ki_ever_probability,
        ki_survive_knocked_in_probability=ki_probability,
        **cashflow_fields,
        **extra_fields,
    )
