"""Neutral DCN grid/payment-time context — pure date arithmetic, no engine
imports. Shared by the MC payoff kernel and the PDE solver (spec WP1.3/1.4).

Times are ACT/365F year fractions from ``schedule.valuation_date`` (fixed by
the problem; ``annualized_days=365``). Payment times come from
``DCNOption.payment_dates()`` (W15): the one place the settlement convention
enters, so every route that reads this context honours it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from quantark.util.exceptions import ValidationError

from .dcn_option import DCNOption

DAYS_PER_YEAR = 365.0  # ACT/365F fixed by the problem


def _yf(d0, d1) -> float:
    return (d1 - d0).days / DAYS_PER_YEAR


@dataclass(frozen=True)
class DCNGridContext:
    times: np.ndarray                 # (n_grid,) t=0 first
    obs_cols: np.ndarray              # (n_obs,) grid column of each monthly obs
    obs_is_coupon: np.ndarray         # (n_obs,) bool
    obs_is_ko: np.ndarray             # (n_obs,) bool
    coupon_accruals: np.ndarray       # (n_obs,) sum a_k over coupon-eligible ks
    ko_accruals: np.ndarray           # (n_obs,) sum a_k over KO-eligible ks
    coupon_pay_times: np.ndarray      # (n_obs,)
    ko_pay_times: np.ndarray          # (n_obs,)
    loss_pay_time: float
    month_index_map: Tuple[Tuple[int, ...], ...]
    # Column 0 (t = 0, the valuation instant) is a daily-KI date. False when
    # the valuation date is not one: a closed day of the schedule calendar
    # (A7') or the post-close view (A1); t = 0 is then prepended and the
    # first knock-in test is column 1.
    valuation_monitored: bool = True


def build_dcn_grid_context(product: DCNOption) -> DCNGridContext:
    s = product.schedule
    v = s.valuation_date
    if not s.daily_ki_dates or s.daily_ki_dates[0] < v:
        raise ValidationError("DCN daily-KI dates must start on/after the valuation date")
    monitored = s.daily_ki_dates[0] == v
    first = 0 if monitored else 1  # unmonitored: t = 0 prepended
    times = np.array(([] if monitored else [0.0]) + [_yf(v, d) for d in s.daily_ki_dates])
    col_of = {d: i + first for i, d in enumerate(s.daily_ki_dates)}
    a = product.accrual_per_period
    # W15: every flow pays on the product's C4 dates (the convention's when set)
    pay_dates = product.payment_dates()
    obs_cols, is_c, is_k, acc_c, acc_k, pay_c, pay_k, kmap = (
        [], [], [], [], [], [], [], []
    )
    for j, m in enumerate(s.monthly):
        if m.observation_date not in col_of:
            raise ValidationError(
                f"observation {m.observation_date:%Y-%m-%d} not on the daily grid"
            )
        obs_cols.append(col_of[m.observation_date])
        is_c.append(m.is_coupon_obs)
        is_k.append(m.is_ko_obs)
        # merged-date convention (spec WP1.1): one a_k per merged eligible
        # month; the samples have exactly one k per row, so accrual == a.
        coupon_periods = sum(
            k >= s.spec.lock_months for k in m.month_indices
        )
        ko_periods = sum(
            k >= s.spec.ko_lock_months for k in m.month_indices
        )
        acc_c.append(a * coupon_periods if m.is_coupon_obs else 0.0)
        acc_k.append(a * ko_periods if m.is_ko_obs else 0.0)
        pay_c.append(_yf(v, pay_dates.coupon_payment_dates[j]))
        pay_k.append(_yf(v, pay_dates.ko_payment_dates[j]))
        kmap.append(m.month_indices)
    return DCNGridContext(
        times=times,
        obs_cols=np.asarray(obs_cols, dtype=int),
        obs_is_coupon=np.asarray(is_c, dtype=bool),
        obs_is_ko=np.asarray(is_k, dtype=bool),
        coupon_accruals=np.asarray(acc_c),
        ko_accruals=np.asarray(acc_k),
        coupon_pay_times=np.asarray(pay_c),
        ko_pay_times=np.asarray(pay_k),
        loss_pay_time=_yf(v, pay_dates.loss_payment_date),
        month_index_map=tuple(kmap),
        valuation_monitored=monitored,
    )
