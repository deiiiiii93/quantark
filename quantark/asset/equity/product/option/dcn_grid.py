"""Neutral DCN grid/payment-time context — pure date arithmetic, no engine
imports. Shared by the MC payoff kernel and the PDE solver (spec WP1.3/1.4).

Times are on the pricing environment's axis (0.14.0, retiring the DCN's own ACT/365F axis "D12"):
``env_time(env, date, anchor=schedule.valuation_date)`` -- the env's day-count convention, calendar,
``bus_days_in_year`` and ``holiday_time_adjustment``, measured from the SCHEDULE's valuation date (date-rolled theta
and ``theta_shift`` price with an env dated elsewhere). CALENDAR_DAYS / ACT_365 envs give ``num_days / 365.0`` as
before, bit for bit.

KI dates on one env instant share one grid column (``column_dates``); no zero-length step is ever built. On a merged
column the kernel's same-instant order (knock-in, knock-out, coupon, final loss) equals the dates' order because the
product's barriers are ordered (KI strictly below the coupon and KO barriers). A future date at env time 0 joins
column 0 and is decided at the valuation instant. Payment times come from ``DCNOption.payment_dates()`` (W15): the one
place the settlement convention enters, so every route that reads this context honours it.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Tuple

import numpy as np

from quantark.priceenv.date_axis import env_times
from quantark.util.exceptions import ValidationError
from quantark.util.numerical.time_nodes import same_instant

from .dcn_option import DCNOption

DAYS_PER_YEAR = 365.0  # the legacy ACT/365F axis (``DCNOption.get_maturity(None)`` only)


@dataclass(frozen=True)
class DCNGridContext:
    times: np.ndarray                 # (n_grid,) t=0 first, strictly increasing (one entry per env instant)
    obs_cols: np.ndarray              # (n_obs,) grid column of each monthly obs, strictly increasing
    obs_is_coupon: np.ndarray         # (n_obs,) bool
    obs_is_ko: np.ndarray             # (n_obs,) bool
    coupon_accruals: np.ndarray       # (n_obs,) sum a_k over coupon-eligible ks
    ko_accruals: np.ndarray           # (n_obs,) sum a_k over KO-eligible ks
    coupon_pay_times: np.ndarray      # (n_obs,)
    ko_pay_times: np.ndarray          # (n_obs,)
    loss_pay_time: float
    month_index_map: Tuple[Tuple[int, ...], ...]
    # Column 0 (t = 0, the valuation instant) holds a daily-KI date: the valuation date, or a later date the env
    # places at time 0. False when it holds none (a closed day of the schedule calendar (A7') or the post-close
    # view (A1)); t = 0 is then prepended and the first knock-in test is column 1.
    valuation_monitored: bool = True
    # The daily-KI dates of each column, in date order (``()`` for a prepended unmonitored t = 0).
    column_dates: Tuple[Tuple[datetime, ...], ...] = ()


def dcn_row_accruals(product: DCNOption) -> Tuple[np.ndarray, np.ndarray]:
    """Per monthly row: (coupon accrual, KO accrual) -- contract data, no environment.

    Merged-date convention (spec WP1.1): one a_k per merged eligible month; the samples have exactly one k per row,
    so accrual == a.
    """
    s = product.schedule
    a = product.accrual_per_period
    acc_c, acc_k = [], []
    for m in s.monthly:
        coupon_periods = sum(k >= s.spec.lock_months for k in m.month_indices)
        ko_periods = sum(k >= s.spec.ko_lock_months for k in m.month_indices)
        acc_c.append(a * coupon_periods if m.is_coupon_obs else 0.0)
        acc_k.append(a * ko_periods if m.is_ko_obs else 0.0)
    return np.asarray(acc_c), np.asarray(acc_k)


def _refuse_backwards(pricing_env, dates, times) -> None:
    """Spec D18: a convention whose times run backwards in date order has no grid."""
    order = sorted(range(len(dates)), key=lambda i: dates[i])
    for i, j in zip(order, order[1:]):
        if times[j] < times[i]:
            raise ValidationError(
                f"the environment's day count {pricing_env.day_count_convention.name} puts "
                f"{dates[j]:%Y-%m-%d} (t={times[j]!r}) before {dates[i]:%Y-%m-%d} (t={times[i]!r}); "
                "a DCN needs times that do not decrease in date order"
            )


def build_dcn_grid_context(product: DCNOption, pricing_env) -> DCNGridContext:
    """The DCN's grid on ``pricing_env``'s axis, anchored at the schedule's valuation date."""
    s = product.schedule
    v = s.valuation_date
    if not s.daily_ki_dates or s.daily_ki_dates[0] < v:
        raise ValidationError("DCN daily-KI dates must start on/after the valuation date")
    # W15: every flow pays on the product's C4 dates (the convention's when set)
    pay_dates = product.payment_dates()
    n_ki, n_obs = len(s.daily_ki_dates), len(s.monthly)
    dates = (list(s.daily_ki_dates) + list(pay_dates.coupon_payment_dates)
             + list(pay_dates.ko_payment_dates) + [pay_dates.loss_payment_date])
    all_times = env_times(pricing_env, dates, anchor=v)
    _refuse_backwards(pricing_env, [v] + dates, np.concatenate(([0.0], all_times)))
    ki_times = all_times[:n_ki]
    pay_c = all_times[n_ki:n_ki + n_obs]
    pay_k = all_times[n_ki + n_obs:n_ki + 2 * n_obs]
    loss_pay_time = float(all_times[-1])

    # one column per env instant; a KI date at time 0 joins the valuation instant
    times, columns = [], []
    for d, t in zip(s.daily_ki_dates, ki_times):
        if columns and same_instant(float(t), times[-1]):
            columns[-1].append(d)
        else:
            times.append(float(t))
            columns.append([d])
    monitored = same_instant(times[0], 0.0)
    if monitored:
        times[0] = 0.0
    else:  # unmonitored: t = 0 prepended
        times.insert(0, 0.0)
        columns.insert(0, [])
    col_of = {d: i for i, dates_i in enumerate(columns) for d in dates_i}

    acc_c, acc_k = dcn_row_accruals(product)
    obs_cols, is_c, is_k, kmap = [], [], [], []
    for m in s.monthly:
        if m.observation_date not in col_of:
            raise ValidationError(
                f"observation {m.observation_date:%Y-%m-%d} not on the daily grid"
            )
        col = col_of[m.observation_date]
        if obs_cols and col <= obs_cols[-1]:
            raise ValidationError(
                f"observations {s.monthly[len(obs_cols) - 1].observation_date:%Y-%m-%d} and "
                f"{m.observation_date:%Y-%m-%d} fall on one instant of the environment's clock; a DCN decides one "
                "monthly row per instant"
            )
        obs_cols.append(col)
        is_c.append(m.is_coupon_obs)
        is_k.append(m.is_ko_obs)
        kmap.append(m.month_indices)
    return DCNGridContext(
        times=np.asarray(times, dtype=float),
        obs_cols=np.asarray(obs_cols, dtype=int),
        obs_is_coupon=np.asarray(is_c, dtype=bool),
        obs_is_ko=np.asarray(is_k, dtype=bool),
        coupon_accruals=acc_c,
        ko_accruals=acc_k,
        coupon_pay_times=np.asarray(pay_c, dtype=float),
        ko_pay_times=np.asarray(pay_k, dtype=float),
        loss_pay_time=loss_pay_time,
        month_index_map=tuple(kmap),
        valuation_monitored=monitored,
        column_dates=tuple(tuple(c) for c in columns),
    )
