"""
Quadrature solver for discretely monitored single-barrier structures.
"""

import math
from typing import Sequence, Tuple

import numpy as np
from scipy.stats import norm

from util.exceptions import NumericalError, ValidationError


class _BarrierQuadratureSolver:
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
        self.grid_x = int(grid_x)
        self.grid_t = int(grid_t)
        self.maturity = float(maturity)
        self.spot = float(spot)
        self.r = float(r)
        self.q = float(q)
        self.vol = float(vol)

        if self.grid_x < 3:
            raise ValidationError("grid_points must be at least 3.")
        if self.grid_t < 2:
            raise ValidationError("time_steps must be at least 2.")
        if self.maturity <= 0.0:
            raise ValidationError("maturity must be positive for quadrature solver.")
        if self.vol <= 0.0:
            raise ValidationError("volatility must be positive.")

        self.dt = self.maturity / self.grid_t
        self.tau = 0.5 * self.vol * self.vol * self.dt
        if self.tau <= 0.0:
            raise ValidationError("time step too small for quadrature solver.")

        self.alpha = (self.r - self.q - 0.5 * self.vol * self.vol) / (
            self.vol * self.vol
        )
        self.beta = (
            (self.r - self.q - 0.5 * self.vol * self.vol) ** 2 / self.vol**4
            + 2.0 * self.r / self.vol**2
        )

        self.constant_c = math.exp(self._factor_c())
        log_c = math.log(self.constant_c)
        self.grid = np.linspace(-log_c, log_c, self.grid_x)
        self.h = 2.0 * log_c / (self.grid_x - 1)

    def price(
        self,
        upper_bounds: np.ndarray,
        lower_bounds: np.ndarray,
        upper_indices: Sequence[int],
        lower_indices: Sequence[int],
        factors: dict,
    ) -> float:
        self.ko = np.asarray(upper_bounds, dtype=float)
        self.ki = np.asarray(lower_bounds, dtype=float)

        bound_upper = np.log(
            np.minimum(self.ko, self.spot * self.constant_c) / self.spot
        )
        bound_lower = np.log(
            np.maximum(self.ki, self.spot / self.constant_c) / self.spot
        )

        if np.any(bound_upper <= bound_lower):
            raise NumericalError("Invalid barrier bounds for quadrature grid.")

        ko_idx = np.array(upper_indices, dtype=int)
        ki_idx = np.array(lower_indices, dtype=int)

        values_at_m_minus_1 = self._calculate_values_at_M_minus_1(
            bound_upper, bound_lower, factors
        )
        return self._backward_recursion(
            values_at_m_minus_1, bound_upper, bound_lower, ko_idx, ki_idx, factors
        )

    def _factor_c(self) -> float:
        vol_max = self.vol if np.isscalar(self.vol) else float(np.max(self.vol))
        return (
            10.0 * vol_max * math.sqrt(self.maturity)
            + (1.0 + 0.5 * vol_max * vol_max) * self.maturity
        )

    def _factor_value_at_m(
        self, spot: np.ndarray | float, strike: float, epsilon: int, kind: str
    ) -> np.ndarray:
        if strike <= 0:
            if kind == "a":
                base = np.asarray(spot) * math.exp(-self.q * self.dt)
            else:
                base = math.exp(-self.r * self.dt)
            return base if epsilon > 0 else np.zeros_like(base, dtype=float)

        spot_array = np.asarray(spot, dtype=float)
        sqrt_dt = math.sqrt(self.dt)
        log_term = np.log(spot_array / strike)
        d1 = (log_term + (self.r - self.q + 0.5 * self.vol * self.vol) * self.dt) / (
            self.vol * sqrt_dt
        )

        if kind == "a":
            return spot_array * math.exp(-self.q * self.dt) * norm.cdf(epsilon * d1)

        d2 = d1 - self.vol * sqrt_dt
        return math.exp(-self.r * self.dt) * norm.cdf(epsilon * d2)

    def _option_value_at_previous_m(
        self,
        spot: np.ndarray | float,
        strike_upper: float,
        strike_lower: float,
        factors: dict,
        m: int,
    ) -> np.ndarray | float:
        v1 = self._factor_value_at_m(spot, strike_lower, -1, "a")
        v2 = self._factor_value_at_m(spot, strike_lower, -1, "b")
        v3 = self._factor_value_at_m(spot, strike_lower, 1, "a")
        v4 = self._factor_value_at_m(spot, strike_upper, 1, "a")
        v5 = self._factor_value_at_m(spot, strike_lower, 1, "b")
        v6 = self._factor_value_at_m(spot, strike_upper, 1, "b")

        return (
            factors["asset1"][m] * v1
            + factors["cash1"][m] * v2
            + factors["asset2"][m] * (v3 - v4)
            + factors["cash2"][m] * (v5 - v6)
            + factors["asset3"][m] * v4
            + factors["cash3"][m] * v6
        )

    def _omega(self, x: np.ndarray) -> np.ndarray:
        return np.exp(-(x**2) / (4.0 * self.tau) - self.alpha * x)

    def _calculate_integral_simpson(
        self, values: np.ndarray, p_lr: int, p_ur: int, p0: int
    ) -> np.ndarray:
        u_array = np.zeros(2 * self.grid_x - 1)
        u_array[p_lr] = values[p_lr]
        u_array[p_ur + p0] = values[p_ur + p0]
        u_array[p_lr + 1 : p_ur + p0 : 2] = 4.0 * values[p_lr + 1 : p_ur + p0 : 2]
        u_array[p_lr + 2 : p_ur + p0 - 1 : 2] = (
            2.0 * values[p_lr + 2 : p_ur + p0 - 1 : 2]
        )
        return u_array

    def _calculate_convolution_fft(
        self, omega_array: np.ndarray, u_array: np.ndarray
    ) -> np.ndarray:
        omega_array = np.asarray(omega_array).ravel()
        u_array = np.asarray(u_array).ravel()
        if len(omega_array) != len(u_array):
            raise NumericalError("omega_array and u_array must have the same length.")

        f_array = np.fft.ifft(np.fft.fft(omega_array) * np.fft.fft(u_array)).real
        start_idx = self.grid_x - 1
        end_idx = 2 * self.grid_x - 1
        return f_array[start_idx:end_idx] * self.h / 3.0

    def _select_simpson_indices(
        self, bound_lr: float, bound_ur: float
    ) -> Tuple[int, int, int]:
        p_lr = int(np.searchsorted(self.grid, bound_lr, side="left"))
        p_ur = int(np.searchsorted(self.grid, bound_ur, side="right")) - 1
        p_lr = max(0, min(p_lr, self.grid_x - 1))
        p_ur = max(0, min(p_ur, self.grid_x - 1))
        if p_ur <= p_lr:
            p_ur = min(p_lr + 1, self.grid_x - 1)
        p0 = (p_ur - p_lr) % 2
        if p_ur + p0 >= self.grid_x:
            p_ur -= 1
            p0 = (p_ur - p_lr) % 2
        return p_lr, p_ur, p0

    def _calculate_values_at_M_minus_1(
        self, bound_upper: np.ndarray, bound_lower: np.ndarray, factors: dict
    ) -> tuple:
        bound_ur_m = float(bound_upper[-2])
        bound_lr_m = float(bound_lower[-2])
        p_lr_m, p_ur_m, p0_m = self._select_simpson_indices(bound_lr_m, bound_ur_m)

        xee_lr_m = 0.5 * (self.grid[p_lr_m] + bound_lr_m)
        xee_ur_m = 0.5 * (self.grid[p_ur_m + p0_m] + bound_ur_m)
        x_m = np.array([bound_lr_m, bound_ur_m, xee_lr_m, xee_ur_m])

        spot_array = self.spot * np.exp(x_m)
        values = self._option_value_at_previous_m(
            spot_array, self.ko[-2], self.ki[-2], factors, -1
        )

        return (
            values,
            p_lr_m,
            p_ur_m,
            p0_m,
            x_m,
            bound_lr_m,
            bound_ur_m,
            xee_lr_m,
            xee_ur_m,
        )

    def _backward_recursion(
        self,
        m_minus_1_data: tuple,
        bound_upper: np.ndarray,
        bound_lower: np.ndarray,
        ko_idx: np.ndarray,
        ki_idx: np.ndarray,
        factors: dict,
    ) -> float:
        (
            values_at_m,
            p_lr_m,
            p_ur_m,
            p0_m,
            _x_m,
            bound_lr_m,
            bound_ur_m,
            xee_lr_m,
            xee_ur_m,
        ) = m_minus_1_data
        v_bound_lr_m, v_bound_ur_m, v_xee_lr_m, v_xee_ur_m = values_at_m

        spot_array = self.spot * np.exp(self.grid)
        y_array = self._option_value_at_previous_m(
            spot_array, self.ko[-1], self.ki[-1], factors, -1
        )
        u_array = self._calculate_integral_simpson(y_array, p_lr_m, p_ur_m, p0_m)

        z = -2.0 * math.log(self.constant_c) + np.arange(2 * self.grid_x - 1) * self.h
        omega_array = self._omega(z)

        ko_set = set(int(x) for x in ko_idx)
        ki_set = set(int(x) for x in ki_idx)

        for m in range(self.grid_t - 1, 1, -1):
            on_barrier = (
                m in ko_set
                or m + 1 in ko_set
                or m - 1 in ko_set
                or m in ki_set
                or m + 1 in ki_set
                or m - 1 in ki_set
            )
            values = self._calculate_values_in_process(
                m,
                u_array,
                omega_array,
                y_array,
                bound_upper,
                bound_lower,
                factors,
                (v_bound_lr_m, v_bound_ur_m, v_xee_lr_m, v_xee_ur_m),
                (p_lr_m, p_ur_m, p0_m),
                (bound_lr_m, bound_ur_m, xee_lr_m, xee_ur_m),
                on_barrier,
            )

            y_array = values["y_array"]
            u_array = values["u_array"]
            v_bound_lr_m, v_bound_ur_m, v_xee_lr_m, v_xee_ur_m = values["values"]
            p_lr_m, p_ur_m, p0_m = values["points"]
            bound_lr_m, bound_ur_m, xee_lr_m, xee_ur_m = values["boundaries"]

        return self._calculate_final_value(
            y_array,
            u_array,
            factors,
            (v_bound_lr_m, v_bound_ur_m, v_xee_lr_m, v_xee_ur_m),
            (p_lr_m, p_ur_m, p0_m),
            (bound_lr_m, bound_ur_m, xee_lr_m, xee_ur_m),
        )

    def _calculate_values_in_process(
        self,
        m: int,
        u_array: np.ndarray,
        omega_array: np.ndarray,
        y_array: np.ndarray,
        bound_upper: np.ndarray,
        bound_lower: np.ndarray,
        factors: dict,
        prev_values: tuple,
        prev_points: tuple,
        prev_boundaries: tuple,
        on_barrier: bool,
    ) -> dict:
        bound_lr_m, bound_ur_m, xee_lr_m, xee_ur_m = prev_boundaries
        v_bound_lr_m, v_bound_ur_m, v_xee_lr_m, v_xee_ur_m = prev_values
        p_lr_m, p_ur_m, p0_m = prev_points

        if on_barrier:
            bound_lr_m_n1 = float(bound_lower[m - 1])
            bound_ur_m_n1 = float(bound_upper[m - 1])
            p_lr_m_n1, p_ur_m_n1, p0_m_n1 = self._select_simpson_indices(
                bound_lr_m_n1, bound_ur_m_n1
            )
            xee_lr_m_n1 = 0.5 * (self.grid[p_lr_m_n1] + bound_lr_m_n1)
            xee_ur_m_n1 = 0.5 * (self.grid[p_ur_m_n1 + p0_m_n1] + bound_ur_m_n1)
            x_m_n1 = np.array(
                [bound_lr_m_n1, bound_ur_m_n1, xee_lr_m_n1, xee_ur_m_n1]
            )
        else:
            bound_lr_m_n1 = bound_lr_m
            bound_ur_m_n1 = bound_ur_m
            p_lr_m_n1 = p_lr_m
            p_ur_m_n1 = p_ur_m
            p0_m_n1 = p0_m
            xee_lr_m_n1 = xee_lr_m
            xee_ur_m_n1 = xee_ur_m
            x_m_n1 = np.array([bound_lr_m, bound_ur_m, xee_lr_m, xee_ur_m])

        y1 = self._calculate_convolution_fft(omega_array, u_array)

        y2 = (self.grid[p_lr_m] - bound_lr_m) / 6.0
        y2 = y2 * (
            self._omega(self.grid - bound_lr_m) * v_bound_lr_m
            + 4.0 * self._omega(self.grid - xee_lr_m) * v_xee_lr_m
            + self._omega(self.grid - self.grid[p_lr_m]) * y_array[p_lr_m]
        )

        y3 = (bound_ur_m - self.grid[p_ur_m + p0_m]) / 6.0
        y3 = y3 * (
            self._omega(self.grid - bound_ur_m) * v_bound_ur_m
            + 4.0 * self._omega(self.grid - xee_ur_m) * v_xee_ur_m
            + self._omega(self.grid - self.grid[p_ur_m + p0_m])
            * y_array[p_ur_m + p0_m]
        )

        a = []
        for s in x_m_n1:
            tmp = (
                self._omega(s - self.grid[p_lr_m : p_ur_m + p0_m + 1])
                * u_array[p_lr_m : p_ur_m + p0_m + 1]
            )
            a.append(float(np.sum(tmp) * self.h / 3.0))
        a = np.array(a)

        b = (self.grid[p_lr_m] - bound_lr_m) / 6.0
        b = b * (
            self._omega(x_m_n1 - bound_lr_m) * v_bound_lr_m
            + 4.0 * self._omega(x_m_n1 - xee_lr_m) * v_xee_lr_m
            + self._omega(x_m_n1 - self.grid[p_lr_m]) * y_array[p_lr_m]
        )

        c = (bound_ur_m - self.grid[p_ur_m + p0_m]) / 6.0
        c = c * (
            self._omega(x_m_n1 - bound_ur_m) * v_bound_ur_m
            + 4.0 * self._omega(x_m_n1 - xee_ur_m) * v_xee_ur_m
            + self._omega(x_m_n1 - self.grid[p_ur_m + p0_m])
            * y_array[p_ur_m + p0_m]
        )

        spot_array = self.spot * np.exp(x_m_n1)
        boundary_base = (
            (a + b + c) * math.exp(-self.beta * self.tau) / math.sqrt(math.pi * self.tau) / 2.0
        )
        boundary_add = (
            factors["asset3"][m]
            * self._factor_value_at_m(spot_array, self.ko[m], 1, "a")
            + factors["cash3"][m]
            * self._factor_value_at_m(spot_array, self.ko[m], 1, "b")
            + factors["asset1"][m]
            * self._factor_value_at_m(spot_array, self.ki[m], -1, "a")
            + factors["cash1"][m]
            * self._factor_value_at_m(spot_array, self.ki[m], -1, "b")
        )

        v_bound_lr_m, v_bound_ur_m, v_xee_lr_m, v_xee_ur_m = boundary_base + boundary_add

        y_array_new = (
            (y1 + y2 + y3)
            * math.exp(-self.beta * self.tau)
            / math.sqrt(math.pi * self.tau)
            / 2.0
        )
        spot_array = self.spot * np.exp(self.grid)
        y_array_new += (
            factors["asset3"][m]
            * self._factor_value_at_m(spot_array, self.ko[m], 1, "a")
            + factors["cash3"][m]
            * self._factor_value_at_m(spot_array, self.ko[m], 1, "b")
            + factors["asset1"][m]
            * self._factor_value_at_m(spot_array, self.ki[m], -1, "a")
            + factors["cash1"][m]
            * self._factor_value_at_m(spot_array, self.ki[m], -1, "b")
        )

        u_array_new = self._calculate_integral_simpson(
            y_array_new, p_lr_m_n1, p_ur_m_n1, p0_m_n1
        )

        return {
            "y_array": y_array_new,
            "u_array": u_array_new,
            "values": (v_bound_lr_m, v_bound_ur_m, v_xee_lr_m, v_xee_ur_m),
            "points": (p_lr_m_n1, p_ur_m_n1, p0_m_n1),
            "boundaries": (bound_lr_m_n1, bound_ur_m_n1, xee_lr_m_n1, xee_ur_m_n1),
        }

    def _calculate_final_value(
        self,
        y_array: np.ndarray,
        u_array: np.ndarray,
        factors: dict,
        final_values: tuple,
        final_points: tuple,
        final_boundaries: tuple,
    ) -> float:
        bound_lr_m, bound_ur_m, xee_lr_m, xee_ur_m = final_boundaries
        v_bound_lr_m, v_bound_ur_m, v_xee_lr_m, v_xee_ur_m = final_values
        p_lr_m, p_ur_m, p0_m = final_points

        y1 = np.sum(
            self._omega(0.0 - self.grid[p_lr_m : p_ur_m + p0_m + 1])
            * u_array[p_lr_m : p_ur_m + p0_m + 1]
            * self.h
            / 3.0
        )

        y2 = (self.grid[p_lr_m] - bound_lr_m) / 6.0
        y2 = y2 * (
            self._omega(0.0 - bound_lr_m) * v_bound_lr_m
            + 4.0 * self._omega(0.0 - xee_lr_m) * v_xee_lr_m
            + self._omega(0.0 - self.grid[p_lr_m]) * y_array[p_lr_m]
        )

        y3 = (bound_ur_m - self.grid[p_ur_m + p0_m]) / 6.0
        y3 = y3 * (
            self._omega(0.0 - bound_ur_m) * v_bound_ur_m
            + 4.0 * self._omega(0.0 - xee_ur_m) * v_xee_ur_m
            + self._omega(0.0 - self.grid[p_ur_m + p0_m])
            * y_array[p_ur_m + p0_m]
        )

        final_value = (
            (y1 + y2 + y3)
            * math.exp(-self.beta * self.tau)
            / math.sqrt(math.pi * self.tau)
            / 2.0
        )
        final_value += factors["asset3"][1] * self._factor_value_at_m(
            self.spot, self.ko[1], 1, "a"
        )
        final_value += factors["cash3"][1] * self._factor_value_at_m(
            self.spot, self.ko[1], 1, "b"
        )
        final_value += factors["asset1"][1] * self._factor_value_at_m(
            self.spot, self.ki[1], -1, "a"
        )
        final_value += factors["cash1"][1] * self._factor_value_at_m(
            self.spot, self.ki[1], -1, "b"
        )

        return float(final_value)
