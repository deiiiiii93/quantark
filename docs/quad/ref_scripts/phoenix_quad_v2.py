# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 2025

Author: QuantArk
Description: An alternative quadrature-based Phoenix pricer that treats
coupon payments as interior step-function rewards at observation dates,
computed via backward convolution with survival to KO. Knock-in risk is
handled using existing barrier helpers. This avoids double-counting and
ad-hoc adjustments for coupons around KO.

This module does not modify existing phoenix_quad.py and can be used as
an independent implementation/validation.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple

from asset.equity.engine.bsm.quad.option_quad import (
    QuadratureOptionPricer,
    price_double_barrier,
    price_single_barrier,
    price_kiko,
)


class PhoenixCouponQuadrature:
    """Compute Phoenix coupon PV via backward quadrature with KO survival.

    The method uses the same Simpson + FFT convolution kernel as
    QuadratureOptionPricer but augments the state with interior rewards
    at observation dates: for each observation i, add coupon_i on states
    where S_{t_i} >= coupon_barrier_i. We split contributions into:
      - Interior reward on survival region (S < KO_i)
      - Boundary reward at KO_i (if KO implies coupon is paid at that date)

    Assumptions:
      - Coupons continue even after KI (typical Phoenix). If this is not
        the case, set coupon_after_ki=False and provide ki barrier data
        to exclude coupons on paths that have knocked-in earlier.
    """

    def __init__(
        self,
        grid_x: int,
        grid_t: int,
        maturity: float,
        spot: float,
        r: float,
        q: float,
        vol: float,
    ) -> None:
        self.base = QuadratureOptionPricer(
            grid_x=grid_x,
            grid_t=grid_t,
            maturity=maturity,
            spot=spot,
            r=r,
            q=q,
            vol=vol,
        )

        # Shorthands to underlying grid/params (read-only usage)
        self.grid_x = self.base.grid_x
        self.grid_t = self.base.grid_t
        self.maturity = self.base.maturity
        self.spot = self.base.spot
        self.r = self.base.r
        self.q = self.base.q
        self.vol = self.base.vol
        self.dt = self.base.dt
        self.tau = self.base.tau
        self.alpha = self.base.alpha
        self.beta = self.base.beta
        self.constant_c = self.base.constant_c
        self.grid = self.base.grid
        self.h = self.base.h

    # ---- Local copies/wrappers of primitives we need (no edits to original) ----
    def _omega(self, x: np.ndarray) -> np.ndarray:
        return np.exp(-(x**2) / self.tau / 4 - self.alpha * x)

    def _simpson_weights(
        self, values: np.ndarray, p_lr: int, p_ur: int, p0: int
    ) -> np.ndarray:
        U_array = np.zeros(2 * self.grid_x - 1)
        U_array[p_lr] = values[p_lr]
        U_array[p_ur + p0] = values[p_ur + p0]
        U_array[p_lr + 1 : p_ur + p0 : 2] = 4 * values[p_lr + 1 : p_ur + p0 : 2]
        U_array[p_lr + 2 : p_ur + p0 - 1 : 2] = 2 * values[p_lr + 2 : p_ur + p0 - 1 : 2]
        return U_array

    def _convolution_fft(
        self, omega_array: np.ndarray, U_array: np.ndarray
    ) -> np.ndarray:
        from scipy.fftpack import fft, ifft

        omega_array = np.asarray(omega_array).ravel()
        U_array = np.asarray(U_array).ravel()
        F_array = ifft(fft(omega_array) * fft(U_array)).real
        start_idx = self.grid_x - 1
        end_idx = min(2 * self.grid_x - 1, len(F_array))
        return F_array[start_idx:end_idx] * self.h / 3

    # ---- Core helpers ----
    def _indices_for_bound(
        self, bound_lr: float, bound_ur: float
    ) -> Tuple[int, int, int]:
        p_lr = int(np.argmax(self.grid >= bound_lr))
        p_ur = int(np.argmin(self.grid < bound_ur) - 1)
        p0 = (p_ur - p_lr) % 2
        return p_lr, p_ur, p0

    def _initialize_bounds(
        self, ko_array: np.ndarray, ki_array: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        integral_bound_upper = np.minimum(ko_array, self.spot * self.constant_c)
        integral_bound_lower = np.maximum(ki_array, self.spot / self.constant_c)
        return np.log(integral_bound_upper / self.spot), np.log(
            integral_bound_lower / self.spot
        )

    def _calc_M_minus_1(
        self,
        bound_upper: np.ndarray,
        bound_lower: np.ndarray,
    ) -> Tuple[int, int, int, float, float, float, float, np.ndarray, np.ndarray]:
        bound_ur = bound_upper[-2]
        bound_lr = bound_lower[-2]
        p_lr, p_ur, p0 = self._indices_for_bound(bound_lr, bound_ur)
        xee_lr = 0.5 * (self.grid[p_lr] + bound_lr)
        xee_ur = 0.5 * (self.grid[p_ur + p0] + bound_ur)
        return (
            p_lr,
            p_ur,
            p0,
            bound_lr,
            bound_ur,
            xee_lr,
            xee_ur,
            np.array([bound_lr, bound_ur, xee_lr, xee_ur]),
            self.spot * np.exp(self.grid),
        )

    def _step_back(
        self,
        m: int,
        y_array: np.ndarray,
        curr_p_lr: int,
        curr_p_ur: int,
        curr_p0: int,
        curr_bound_lr: float,
        curr_bound_ur: float,
        curr_xee_lr: float,
        curr_xee_ur: float,
        bound_upper: np.ndarray,
        bound_lower: np.ndarray,
        coupon_step: Tuple[float, float] | None,
        ko_coupon: float,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        Tuple[int, int, int],
        Tuple[float, float, float, float],
        Tuple[float, float, float, float],
    ]:
        # Interior reward: add coupon step on survival region BEFORE convolution,
        # using CURRENT step boundaries/points.
        if coupon_step is not None:
            coupon_amt, coupon_logK = coupon_step
            # 1{grid >= coupon_logK}
            mask = self.grid >= coupon_logK
            y_with_coupon = y_array.copy()
            y_with_coupon[mask] += coupon_amt
        else:
            y_with_coupon = y_array

        # Simpson integration weights over CURRENT survival interval
        U_array = self._simpson_weights(y_with_coupon, curr_p_lr, curr_p_ur, curr_p0)

        # Omega over the extended z stencil
        z = np.array(
            [
                -2 * np.log(self.constant_c) + (i - 1) * self.h
                for i in range(1, 2 * self.grid_x)
            ]
        )
        omega_array = self._omega(z)

        # Core convolution and edge corrections
        y1 = self._convolution_fft(omega_array, U_array)

        y2 = (self.grid[curr_p_lr] - curr_bound_lr) / 6
        y2 = y2 * (
            self._omega(self.grid - curr_bound_lr) * 0.0
            + 4 * self._omega(self.grid - curr_xee_lr) * 0.0
            + self._omega(self.grid - self.grid[curr_p_lr]) * y_with_coupon[curr_p_lr]
        )

        y3 = (curr_bound_ur - self.grid[curr_p_ur + curr_p0]) / 6
        y3 = y3 * (
            self._omega(self.grid - curr_bound_ur) * 0.0
            + 4 * self._omega(self.grid - curr_xee_ur) * 0.0
            + self._omega(self.grid - self.grid[curr_p_ur + curr_p0])
            * y_with_coupon[curr_p_ur + curr_p0]
        )

        y_new = (
            (y1 + y2 + y3)
            * np.exp(-self.beta * self.tau)
            / np.sqrt(np.pi * self.tau)
            / 2
        )

        # KO boundary coupon at this step (if KO implies coupon payment)
        if ko_coupon != 0.0:
            # Pay coupon at KO boundary via cash-digital at strike = KO[m]
            # Equivalent to adding factors["cash3"][m] in original engine
            S_grid = self.spot * np.exp(self.grid)
            ko_strike = self._ko_vals[m]
            # Compute binary cash price at time step m using base helper
            # Here we employ the same closed-form for one-step digital under kernel
            # using factor_value_at_m via the original engine API
            v_at_boundary = self.base._factor_value_at_m(
                S_grid, ko_strike, 1, "b"
            )  # type: ignore[attr-defined]
            y_new += ko_coupon * v_at_boundary

        # Determine NEXT step boundaries/points (m-1) if on barrier; otherwise keep current
        on_barrier = False
        if m in self._ko_idx or (m + 1) in self._ko_idx or (m - 1) in self._ko_idx:
            on_barrier = True
        if m in self._ki_idx or (m + 1) in self._ki_idx or (m - 1) in self._ki_idx:
            on_barrier = True

        if on_barrier:
            next_bound_lr = bound_lower[m - 1]
            next_bound_ur = bound_upper[m - 1]
            next_p_lr, next_p_ur, next_p0 = self._indices_for_bound(
                next_bound_lr, next_bound_ur
            )
            next_xee_lr = 0.5 * (self.grid[next_p_lr] + next_bound_lr)
            next_xee_ur = 0.5 * (self.grid[next_p_ur + next_p0] + next_bound_ur)
        else:
            next_bound_lr, next_bound_ur = curr_bound_lr, curr_bound_ur
            next_p_lr, next_p_ur, next_p0 = curr_p_lr, curr_p_ur, curr_p0
            next_xee_lr, next_xee_ur = curr_xee_lr, curr_xee_ur

        # Build Simpson weights for NEXT step
        U_array_new = self._simpson_weights(y_new, next_p_lr, next_p_ur, next_p0)

        return (
            y_new,
            U_array_new,
            (next_p_lr, next_p_ur, next_p0),
            (next_bound_lr, next_bound_ur, next_xee_lr, next_xee_ur),
            (curr_bound_lr, curr_bound_ur, curr_xee_lr, curr_xee_ur),
        )

    def price_coupons(
        self,
        ko_prices: np.ndarray,
        ko_dates: np.ndarray,
        coupon_prices: np.ndarray,
        coupon_dates: np.ndarray,
        coupon_amounts: np.ndarray,
        pay_type: str = "instant",
        coupon_after_ki: bool = True,
        ki_price: float | None = None,
        ki_dates: np.ndarray | None = None,
    ) -> float:
        # Map dates to grid indices
        grid_t_array = np.linspace(0, self.maturity, self.grid_t + 1)
        ko_idx = np.searchsorted(grid_t_array, ko_dates)
        cp_idx = np.searchsorted(grid_t_array, coupon_dates)

        self._ko_idx = set(int(i) for i in ko_idx)
        self._ki_idx = set(
            int(i)
            for i in (
                np.searchsorted(grid_t_array, ki_dates)
                if (ki_dates is not None)
                else []
            )
        )

        # Build KO/KI arrays over grid
        ko_arr = np.ones(self.grid_t + 1) * self.spot * self.constant_c
        ki_arr = np.ones(self.grid_t + 1) * (self.spot / self.constant_c)
        for i, k in zip(ko_idx, ko_prices):
            ko_arr[i] = k
        if (not coupon_after_ki) and (ki_price is not None) and (ki_dates is not None):
            for j in np.searchsorted(grid_t_array, ki_dates):
                ki_arr[int(j)] = ki_price

        bound_upper, bound_lower = self._initialize_bounds(ko_arr, ki_arr)

        # Initialize at M-1
        p_lr, p_ur, p0, b_lr, b_ur, xee_lr, xee_ur, x_pts, S_grid = (
            self._calc_M_minus_1(bound_upper, bound_lower)
        )

        # Start from terminal y (no coupon at T beyond observation processing)
        y_array = np.zeros_like(self.grid)
        U_array = self._simpson_weights(y_array, p_lr, p_ur, p0)

        # Keep KO levels for boundary digital
        self._ko_vals = ko_arr

        # Discounting convention for coupon amounts
        if pay_type.lower() == "instant":
            # coupons are paid at their observation times; we add them as immediate rewards
            # which will be naturally discounted by the kernel recursion
            coupon_df = np.ones_like(coupon_amounts)
        else:
            # coupons paid at maturity; pre-discount back to t_i under r
            coupon_df = np.exp(-self.r * (self.maturity - coupon_dates))

        # Build per-step coupon info
        cp_map: Dict[int, Tuple[float, float]] = {}
        for idx, c_amt, c_prc in zip(cp_idx, coupon_amounts * coupon_df, coupon_prices):
            # interior step threshold in log-space
            cp_map[int(idx)] = (float(c_amt), float(np.log(c_prc / self.spot)))

        # Backward recursion
        for m in range(self.grid_t - 1, 1, -1):
            # Interior coupon at m if any
            coupon_step = cp_map.get(m, None)

            # If KO happens at m and KO level >= coupon barrier, add KO boundary coupon only
            ko_coupon = 0.0
            if m in self._ko_idx:
                # KO coupon is the same coupon amount at this step, paid at KO boundary
                if coupon_step is not None:
                    ko_level = ko_arr[m]
                    _, cp_logK = coupon_step
                    if np.log(ko_level / self.spot) >= cp_logK:
                        ko_coupon = cp_map[m][0]

            # Perform one step back using CURRENT step geometry; receive NEXT step geometry
            y_array, U_array, (p_lr, p_ur, p0), (b_lr, b_ur, xee_lr, xee_ur), _ = (
                self._step_back(
                    m,
                    y_array,
                    p_lr,
                    p_ur,
                    p0,
                    b_lr,
                    b_ur,
                    xee_lr,
                    xee_ur,
                    bound_upper,
                    bound_lower,
                    coupon_step,
                    ko_coupon,
                )
            )

        # Final aggregation at time 0: integrate over [b_lr, b_ur] with omega(0-.)
        y1 = np.sum(
            self._omega(0 - self.grid[p_lr : p_ur + p0 + 1])
            * U_array[p_lr : p_ur + p0 + 1]
            * self.h
            / 3
        )

        y2 = (self.grid[p_lr] - b_lr) / 6
        y2 = y2 * (
            self._omega(0 - b_lr) * 0.0
            + 4 * self._omega(0 - xee_lr) * 0.0
            + self._omega(0 - self.grid[p_lr]) * y_array[p_lr]
        )

        y3 = (b_ur - self.grid[p_ur + p0]) / 6
        y3 = y3 * (
            self._omega(0 - b_ur) * 0.0
            + 4 * self._omega(0 - xee_ur) * 0.0
            + self._omega(0 - self.grid[p_ur + p0]) * y_array[p_ur + p0]
        )

        final_value = (
            (y1 + y2 + y3)
            * np.exp(-self.beta * self.tau)
            / np.sqrt(np.pi * self.tau)
            / 2
        )
        return float(final_value)


def price_phoenix_v2(
    bus_days: int,
    cal_days: int,
    ko_bus_days: List[int],
    ko_cal_days: List[int],
    ko_prices: List[float],
    div_prices: List[float],
    ki_price: float,
    spot: float,
    r: float,
    q: float,
    vol: float,
    notional: float,
    div_rate: float,
    initial: float,
    strike: float,
    grid_x: int = 1000,
    grid_t: int | None = None,
    participation_rate: float = 1.0,
    is_knocked_in: bool = False,
    protection_rate: float = 0.0,
    day_count_basis: Dict[str, int] = {"calendar": 365, "trading": 244},
    day_count_convention: str = "A365",
    ki_obs_type: str = "daily",
    ki_dates: List[str] | None = None,
    segment_cal_days: List[float] | None = None,
    is_coupon_annualized: bool = True,
    is_ki_annualized: bool = False,
    coupon_after_ki: bool = True,
    pay_type: str = "instant",
    market_spot: float = 0.0,
) -> Dict[str, float]:
    """Price Phoenix via alternative quadrature with survival-aware coupons.

    The interface mirrors price_phoenix in phoenix_quad.py, with a new
    parameter coupon_after_ki controlling whether coupons cease after KI.
    """
    maturity = bus_days / day_count_basis["trading"]
    df = np.exp(-r * maturity)

    if grid_t is None:
        grid_t = max(4, len(ko_bus_days) * 4)

    if market_spot == 0:
        market_spot = spot

    # Normalize to market spot for numerical stability
    ko_prices_arr = np.array(ko_prices, dtype=float) / market_spot
    div_prices_arr = np.array(div_prices, dtype=float) / market_spot
    ki_price_n = float(ki_price / market_spot)
    strike_n = float(strike / market_spot)
    initial_n = float(initial / market_spot)
    spot_n = float(spot / market_spot)

    ko_bus_days_annual = np.array(ko_bus_days, dtype=float) / day_count_basis["trading"]
    ko_cal_days_annual = (
        (np.array(ko_cal_days, dtype=float) / day_count_basis["calendar"])
        if is_coupon_annualized
        else np.ones(len(ko_bus_days), dtype=float)
    )

    # Build segment durations for coupon amount per period
    if segment_cal_days is None:
        if is_coupon_annualized:
            segment_cal_days = [ko_cal_days[0]]
            for i in range(1, len(ko_cal_days)):
                segment_cal_days.append(ko_cal_days[i] - ko_cal_days[i - 1])
        else:
            segment_cal_days = [1.0] * len(ko_cal_days)
    segment_cal_days_arr = (
        np.array(segment_cal_days, dtype=float) / day_count_basis["calendar"]
    )

    # Coupon amounts per observation period (annualized or not)
    coupon_amounts = div_rate * segment_cal_days_arr

    # Coupon pay dates (observation) in years
    coupon_dates = ko_bus_days_annual.copy()

    # Instantiate coupon pricer (works in normalized space)
    coupon_pricer = PhoenixCouponQuadrature(grid_x, grid_t, maturity, spot_n, r, q, vol)

    coupon_value = coupon_pricer.price_coupons(
        ko_prices=ko_prices_arr,
        ko_dates=ko_bus_days_annual,
        coupon_prices=div_prices_arr,
        coupon_dates=coupon_dates,
        coupon_amounts=coupon_amounts,
        pay_type=pay_type,
        coupon_after_ki=coupon_after_ki,
        ki_price=(ki_price_n if not coupon_after_ki else None),
        ki_dates=(
            np.array([i / day_count_basis["trading"] for i in range(bus_days)])
            if (not coupon_after_ki and ki_obs_type.lower() == "daily")
            else (
                np.array([maturity])
                if (not coupon_after_ki and ki_obs_type.lower() == "euro")
                else None
            )
        ),
    )

    # Knock-in risk component (use KIKO helper for cleaner expression)
    knockin_value = 0.0
    ki_obs_type_l = ki_obs_type.lower()
    ki_mapper = {
        "daily": np.array(
            [i / day_count_basis["trading"] for i in range(bus_days)], dtype=float
        ),
        "euro": np.array([maturity], dtype=float),
        "custom": (
            np.array(ki_dates, dtype=float) / day_count_basis["trading"]
            if (ki_dates is not None)
            else ValueError("knockin obs dates should be input, as obs type is custom.")
        ),
    }
    if ki_obs_type_l not in ki_mapper:
        raise ValueError(f"invalid obs type {ki_obs_type_l} input.")
    ki_bus_days_arr = ki_mapper[ki_obs_type_l]  # type: ignore[index]

    # KIKO put payoff represents down-and-in with up-and-out. If already
    # knocked in, drop the KI leg and keep only up-and-out put.
    if not np.isinf(ki_price_n):
        if is_knocked_in:
            knockin_value = -price_single_barrier(
                grid_x,
                grid_t,
                maturity,
                ko_prices_arr,
                ko_bus_days_annual,
                "put",
                strike_n,
                spot_n,
                r,
                q,
                vol,
                "up",
            )
        else:
            knockin_value = -price_kiko(
                grid_x,
                grid_t,
                maturity,
                ko_prices_arr,
                ko_bus_days_annual,
                ki_price_n,
                ki_bus_days_arr,
                "put",
                strike_n,
                spot_n,
                r,
                q,
                vol,
            )

    knockin_value *= df * participation_rate

    # Partial protection via put spread: protection_rate in [0,1]. If >0,
    # reduce terminal put exposure below protection floor P = protection_rate*initial.
    if protection_rate > 0.0 and protection_rate < 1.0:
        P = protection_rate * initial_n
        # Adjust KI payoff from put(K) to put spread put(K)-put(P)
        if not np.isinf(ki_price_n):
            protection_adj = 0.0
            if is_knocked_in:
                # Base already: -UpOutPut(K); add +UpOutPut(P)
                protection_adj = price_single_barrier(
                    grid_x,
                    grid_t,
                    maturity,
                    ko_prices_arr,
                    ko_bus_days_annual,
                    "put",
                    P,
                    spot_n,
                    r,
                    q,
                    vol,
                    "up",
                )
            else:
                # Base already: -KIKO(K); add +KIKO(P)
                protection_adj = price_kiko(
                    grid_x,
                    grid_t,
                    maturity,
                    ko_prices_arr,
                    ko_bus_days_annual,
                    ki_price_n,
                    ki_bus_days_arr,
                    "put",
                    P,
                    spot_n,
                    r,
                    q,
                    vol,
                )
            knockin_value += protection_adj * df * participation_rate

    if is_ki_annualized:
        knockin_value *= ko_cal_days[-1] / day_count_basis["calendar"]

    total_value = (coupon_value + knockin_value) * notional

    return {
        "total_value": float(total_value),
        "coupon_value": float(coupon_value * notional),
        "knockin_value": float(knockin_value * notional),
    }


if __name__ == "__main__":
    # Minimal smoke test (parameters illustrative only)
    maturity_in_days = 241
    maturity = 241 / 244
    time_resolution = 12 * 4
    grid_resolution = 512

    ko_prices = [1.03] * 12
    ko_dates_bus = np.array([22, 37, 58, 79, 96, 118, 140, 161, 182, 198, 220, 241])
    ko_dates_cal = np.array([34, 63, 92, 125, 153, 184, 216, 245, 278, 307, 337, 366])
    div_prices = [0.85] * 12

    spot = 1.00
    r = 0.03
    q = 0.02
    vol = 0.2
    notional = 1_000_000
    div_rate = 0.10
    initial = 1.0
    strike = 1.0

    out = price_phoenix_v2(
        bus_days=maturity_in_days,
        cal_days=int(maturity * 365),
        ko_bus_days=ko_dates_bus.tolist(),
        ko_cal_days=ko_dates_cal.tolist(),
        ko_prices=ko_prices,
        div_prices=div_prices,
        ki_price=0.7,
        spot=spot,
        r=r,
        q=q,
        vol=vol,
        notional=notional,
        div_rate=div_rate,
        initial=initial,
        strike=strike,
        grid_x=grid_resolution,
        grid_t=time_resolution,
        participation_rate=1.0,
        is_knocked_in=False,
        protection_rate=0.0,
        ki_obs_type="daily",
        segment_cal_days=None,
        is_coupon_annualized=True,
        is_ki_annualized=False,
        coupon_after_ki=True,
        pay_type="instant",
        market_spot=spot,
    )
    print({k: round(v, 6) for k, v in out.items()})
