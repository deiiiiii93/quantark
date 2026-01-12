"""
Product-agnostic quadrature core for discretely monitored options.

Implements the recursion in Eq. (3.5) from docs/quad/quad.md using
boundary levels and linear payoff coefficients at observation dates.
"""

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.stats import norm

from util.exceptions import NumericalError, ValidationError
from util.numerical import Tolerance


@dataclass(frozen=True)
class QuadCoreInputs:
    """Inputs for the quadrature core recursion."""

    observation_times: Sequence[float]
    k_minus: Sequence[float]
    k_plus: Sequence[float]
    a_minus: Sequence[float]
    b_minus: Sequence[float]
    a_plus: Sequence[float]
    b_plus: Sequence[float]
    a_terminal: float
    b_terminal: float


class QuadratureCore:
    """Quadrature core implementing Eq. (3.5) on a fixed log-price grid."""

    def __init__(
        self,
        grid_x: int,
        spot: float,
        observation_times: Sequence[float],
        rate: float | Sequence[float],
        div: float | Sequence[float],
        vol: float | Sequence[float],
    ) -> None:
        self.grid_x = int(grid_x)
        self.spot = float(spot)

        times = np.asarray(observation_times, dtype=float)
        if times.size == 0:
            raise ValidationError("observation_times must contain at least one entry.")
        if np.any(times <= Tolerance.ZERO):
            raise ValidationError("observation_times must be strictly positive.")
        if np.any(np.diff(times) <= Tolerance.ZERO):
            raise ValidationError("observation_times must be strictly increasing.")

        self.times = times
        self.grid_t = int(times.size)

        if self.grid_x < 3:
            raise ValidationError("grid_points must be at least 3.")
        if self.grid_t < 1:
            raise ValidationError("time_steps must be at least 1.")

        times_full = np.concatenate(([0.0], times))
        dt = np.diff(times_full)
        self.dt = np.concatenate(([0.0], dt))

        self.r = self._broadcast_param(rate, "rate")
        self.q = self._broadcast_param(div, "dividend_yield")
        self.vol = self._broadcast_param(vol, "volatility")

        if self.spot <= 0.0:
            raise ValidationError(f"spot must be positive, got {self.spot}.")
        if np.any(self.vol[1:] <= 0.0):
            raise ValidationError("volatility must be positive.")
        if np.any(self.dt[1:] <= 0.0):
            raise ValidationError("observation_times must be strictly increasing.")

        self.tau = 0.5 * self.vol * self.vol * self.dt
        if np.any(self.tau[1:] <= 0.0):
            raise ValidationError("time step too small for quadrature solver.")

        self.alpha = (self.r - self.q - 0.5 * self.vol * self.vol) / (
            self.vol * self.vol
        )
        self.beta = (
            (self.r - self.q - 0.5 * self.vol * self.vol) ** 2 / self.vol**4
            + 2.0 * self.r / self.vol**2
        )

        self.maturity = float(times[-1])
        self.constant_c = math.exp(self._factor_c())
        log_c = math.log(self.constant_c)
        self.grid = np.linspace(-log_c, log_c, self.grid_x)
        self.h = 2.0 * log_c / (self.grid_x - 1)
        self._z_grid = -2.0 * log_c + np.arange(2 * self.grid_x - 1) * self.h

    def _broadcast_param(self, param: float | Sequence[float], name: str) -> np.ndarray:
        values = np.asarray(param, dtype=float)
        if values.shape == () or values.size == 1:
            return np.full(self.grid_t + 1, float(values.ravel()[0]))
        if values.size == self.grid_t:
            return np.concatenate(([float(values[0])], values.astype(float)))
        if values.size == self.grid_t + 1:
            return values.astype(float)
        raise ValidationError(
            f"{name} length must be 1, {self.grid_t}, or {self.grid_t + 1}."
        )

    def _factor_c(self) -> float:
        vol_max = float(np.max(self.vol[1:]))
        return (
            10.0 * vol_max * math.sqrt(self.maturity)
            + (1.0 + 0.5 * vol_max * vol_max) * self.maturity
        )

    def price(self, inputs: QuadCoreInputs) -> float:
        obs_times = np.asarray(inputs.observation_times, dtype=float)
        if obs_times.size != self.grid_t:
            raise ValidationError("observation_times length mismatch in QuadCoreInputs.")

        ko = self._pad_series(inputs.k_plus, "k_plus", math.inf)
        ki = self._pad_series(inputs.k_minus, "k_minus", 0.0)
        a_minus = self._pad_series(inputs.a_minus, "a_minus", 0.0)
        b_minus = self._pad_series(inputs.b_minus, "b_minus", 0.0)
        a_plus = self._pad_series(inputs.a_plus, "a_plus", 0.0)
        b_plus = self._pad_series(inputs.b_plus, "b_plus", 0.0)

        self.ko = ko
        self.ki = ki

        bound_upper = np.log(
            np.minimum(self.ko, self.spot * self.constant_c) / self.spot
        )
        bound_lower = np.log(
            np.maximum(self.ki, self.spot / self.constant_c) / self.spot
        )

        if np.any(bound_upper[1:] <= bound_lower[1:]):
            raise NumericalError("Invalid barrier bounds for quadrature grid.")

        factors = {
            "asset1": a_minus,
            "cash1": b_minus,
            "asset2": np.zeros(self.grid_t + 1),
            "cash2": np.zeros(self.grid_t + 1),
            "asset3": a_plus,
            "cash3": b_plus,
        }
        factors["asset2"][-1] = float(inputs.a_terminal)
        factors["cash2"][-1] = float(inputs.b_terminal)

        if self.grid_t == 1:
            value = self._option_value_at_previous_m(
                self.spot,
                self.ko[1],
                self.ki[1],
                factors,
                factor_index=1,
                step_index=1,
            )
            return float(value)

        m_minus_1_data = self._calculate_values_at_M_minus_1(
            bound_upper, bound_lower, factors
        )
        return self._backward_recursion(
            m_minus_1_data, bound_upper, bound_lower, factors
        )

    def _pad_series(self, series: Sequence[float], name: str, default: float) -> np.ndarray:
        values = np.asarray(series, dtype=float)
        if values.size == self.grid_t:
            padded = np.concatenate(([default], values))
        elif values.size == self.grid_t + 1:
            padded = values
        else:
            raise ValidationError(
                f"{name} length must be {self.grid_t} or {self.grid_t + 1}."
            )
        return padded.astype(float)

    def _factor_value_at_m(
        self,
        spot: np.ndarray | float,
        strike: float,
        epsilon: int,
        kind: str,
        step_index: int,
    ) -> np.ndarray:
        dt = float(self.dt[step_index])
        r = float(self.r[step_index])
        q = float(self.q[step_index])
        vol = float(self.vol[step_index])

        if math.isinf(strike):
            if kind == "a":
                base = np.asarray(spot) * math.exp(-q * dt)
            else:
                base = math.exp(-r * dt)
            return np.zeros_like(base, dtype=float) if epsilon > 0 else base

        if strike <= 0:
            if kind == "a":
                base = np.asarray(spot) * math.exp(-q * dt)
            else:
                base = math.exp(-r * dt)
            return base if epsilon > 0 else np.zeros_like(base, dtype=float)

        spot_array = np.asarray(spot, dtype=float)
        sqrt_dt = math.sqrt(dt)
        log_term = np.log(spot_array / strike)
        d1 = (log_term + (r - q + 0.5 * vol * vol) * dt) / (vol * sqrt_dt)

        if kind == "a":
            return spot_array * math.exp(-q * dt) * norm.cdf(epsilon * d1)

        d2 = d1 - vol * sqrt_dt
        return math.exp(-r * dt) * norm.cdf(epsilon * d2)

    def _option_value_at_previous_m(
        self,
        spot: np.ndarray | float,
        strike_upper: float,
        strike_lower: float,
        factors: dict,
        factor_index: int,
        step_index: int,
    ) -> np.ndarray | float:
        v1 = self._factor_value_at_m(spot, strike_lower, -1, "a", step_index)
        v2 = self._factor_value_at_m(spot, strike_lower, -1, "b", step_index)
        v3 = self._factor_value_at_m(spot, strike_lower, 1, "a", step_index)
        v4 = self._factor_value_at_m(spot, strike_upper, 1, "a", step_index)
        v5 = self._factor_value_at_m(spot, strike_lower, 1, "b", step_index)
        v6 = self._factor_value_at_m(spot, strike_upper, 1, "b", step_index)

        return (
            factors["asset1"][factor_index] * v1
            + factors["cash1"][factor_index] * v2
            + factors["asset2"][factor_index] * (v3 - v4)
            + factors["cash2"][factor_index] * (v5 - v6)
            + factors["asset3"][factor_index] * v4
            + factors["cash3"][factor_index] * v6
        )

    def _omega(self, x: np.ndarray, step_index: int) -> np.ndarray:
        tau = float(self.tau[step_index])
        alpha = float(self.alpha[step_index])
        return np.exp(-(x**2) / (4.0 * tau) - alpha * x)

    def _prefactor(self, step_index: int) -> float:
        tau = float(self.tau[step_index])
        beta = float(self.beta[step_index])
        return math.exp(-beta * tau) / math.sqrt(math.pi * tau) / 2.0

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
    ) -> tuple[int, int, int]:
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

    def _calculate_tail_integral_left(
        self,
        x_array: np.ndarray,
        m: int,
        bound_lr: float,
        xee_lr: float,
        v_bound_lr: float,
        v_xee_lr: float,
        anchor_grid: float,
        anchor_value: float,
    ) -> np.ndarray:
        weight = (anchor_grid - bound_lr) / 6.0
        return weight * (
            self._omega(x_array - bound_lr, m) * v_bound_lr
            + 4.0 * self._omega(x_array - xee_lr, m) * v_xee_lr
            + self._omega(x_array - anchor_grid, m) * anchor_value
        )

    def _calculate_tail_integral_right(
        self,
        x_array: np.ndarray,
        m: int,
        bound_ur: float,
        xee_ur: float,
        v_bound_ur: float,
        v_xee_ur: float,
        anchor_grid: float,
        anchor_value: float,
    ) -> np.ndarray:
        weight = (bound_ur - anchor_grid) / 6.0
        return weight * (
            self._omega(x_array - bound_ur, m) * v_bound_ur
            + 4.0 * self._omega(x_array - xee_ur, m) * v_xee_ur
            + self._omega(x_array - anchor_grid, m) * anchor_value
        )

    def _calculate_values_at_M_minus_1(
        self, bound_upper: np.ndarray, bound_lower: np.ndarray, factors: dict
    ) -> tuple:
        m_index = self.grid_t - 1
        bound_ur_m = float(bound_upper[m_index])
        bound_lr_m = float(bound_lower[m_index])
        p_lr_m, p_ur_m, p0_m = self._select_simpson_indices(bound_lr_m, bound_ur_m)

        xee_lr_m = 0.5 * (self.grid[p_lr_m] + bound_lr_m)
        xee_ur_m = 0.5 * (self.grid[p_ur_m + p0_m] + bound_ur_m)
        x_m = np.array([bound_lr_m, bound_ur_m, xee_lr_m, xee_ur_m])

        spot_array = self.spot * np.exp(x_m)
        values = self._option_value_at_previous_m(
            spot_array,
            self.ko[m_index],
            self.ki[m_index],
            factors,
            factor_index=self.grid_t,
            step_index=self.grid_t,
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
            spot_array,
            self.ko[self.grid_t],
            self.ki[self.grid_t],
            factors,
            factor_index=self.grid_t,
            step_index=self.grid_t,
        )
        u_array = self._calculate_integral_simpson(y_array, p_lr_m, p_ur_m, p0_m)

        for m in range(self.grid_t - 1, 1, -1):
            omega_array = self._omega(self._z_grid, m)
            prefactor = self._prefactor(m)
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
                prefactor,
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
        prefactor: float,
    ) -> dict:
        bound_lr_m, bound_ur_m, xee_lr_m, xee_ur_m = prev_boundaries
        v_bound_lr_m, v_bound_ur_m, v_xee_lr_m, v_xee_ur_m = prev_values
        p_lr_m, p_ur_m, p0_m = prev_points

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

        y1 = self._calculate_convolution_fft(omega_array, u_array)
        anchor_left = self.grid[p_lr_m]
        anchor_right = self.grid[p_ur_m + p0_m]

        y2 = self._calculate_tail_integral_left(
            self.grid,
            m,
            bound_lr_m,
            xee_lr_m,
            v_bound_lr_m,
            v_xee_lr_m,
            anchor_left,
            y_array[p_lr_m],
        )

        y3 = self._calculate_tail_integral_right(
            self.grid,
            m,
            bound_ur_m,
            xee_ur_m,
            v_bound_ur_m,
            v_xee_ur_m,
            anchor_right,
            y_array[p_ur_m + p0_m],
        )

        a = []
        for s in x_m_n1:
            tmp = (
                self._omega(s - self.grid[p_lr_m : p_ur_m + p0_m + 1], m)
                * u_array[p_lr_m : p_ur_m + p0_m + 1]
            )
            a.append(float(np.sum(tmp) * self.h / 3.0))
        a = np.array(a)

        b = self._calculate_tail_integral_left(
            x_m_n1,
            m,
            bound_lr_m,
            xee_lr_m,
            v_bound_lr_m,
            v_xee_lr_m,
            anchor_left,
            y_array[p_lr_m],
        )

        c = self._calculate_tail_integral_right(
            x_m_n1,
            m,
            bound_ur_m,
            xee_ur_m,
            v_bound_ur_m,
            v_xee_ur_m,
            anchor_right,
            y_array[p_ur_m + p0_m],
        )

        spot_array = self.spot * np.exp(x_m_n1)
        boundary_base = (a + b + c) * prefactor
        boundary_add = (
            factors["asset3"][m]
            * self._factor_value_at_m(spot_array, self.ko[m], 1, "a", m)
            + factors["cash3"][m]
            * self._factor_value_at_m(spot_array, self.ko[m], 1, "b", m)
            + factors["asset1"][m]
            * self._factor_value_at_m(spot_array, self.ki[m], -1, "a", m)
            + factors["cash1"][m]
            * self._factor_value_at_m(spot_array, self.ki[m], -1, "b", m)
        )

        v_bound_lr_m, v_bound_ur_m, v_xee_lr_m, v_xee_ur_m = boundary_base + boundary_add

        y_array_new = (y1 + y2 + y3) * prefactor
        spot_array = self.spot * np.exp(self.grid)
        y_array_new += (
            factors["asset3"][m]
            * self._factor_value_at_m(spot_array, self.ko[m], 1, "a", m)
            + factors["cash3"][m]
            * self._factor_value_at_m(spot_array, self.ko[m], 1, "b", m)
            + factors["asset1"][m]
            * self._factor_value_at_m(spot_array, self.ki[m], -1, "a", m)
            + factors["cash1"][m]
            * self._factor_value_at_m(spot_array, self.ki[m], -1, "b", m)
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
        prefactor = self._prefactor(1)

        y1 = np.sum(
            self._omega(0.0 - self.grid[p_lr_m : p_ur_m + p0_m + 1], 1)
            * u_array[p_lr_m : p_ur_m + p0_m + 1]
            * self.h
            / 3.0
        )

        y2 = (self.grid[p_lr_m] - bound_lr_m) / 6.0
        y2 = y2 * (
            self._omega(0.0 - bound_lr_m, 1) * v_bound_lr_m
            + 4.0 * self._omega(0.0 - xee_lr_m, 1) * v_xee_lr_m
            + self._omega(0.0 - self.grid[p_lr_m], 1) * y_array[p_lr_m]
        )

        y3 = (bound_ur_m - self.grid[p_ur_m + p0_m]) / 6.0
        y3 = y3 * (
            self._omega(0.0 - bound_ur_m, 1) * v_bound_ur_m
            + 4.0 * self._omega(0.0 - xee_ur_m, 1) * v_xee_ur_m
            + self._omega(0.0 - self.grid[p_ur_m + p0_m], 1)
            * y_array[p_ur_m + p0_m]
        )

        final_value = (y1 + y2 + y3) * prefactor
        final_value += factors["asset3"][1] * self._factor_value_at_m(
            self.spot, self.ko[1], 1, "a", 1
        )
        final_value += factors["cash3"][1] * self._factor_value_at_m(
            self.spot, self.ko[1], 1, "b", 1
        )
        final_value += factors["asset1"][1] * self._factor_value_at_m(
            self.spot, self.ki[1], -1, "a", 1
        )
        final_value += factors["cash1"][1] * self._factor_value_at_m(
            self.spot, self.ki[1], -1, "b", 1
        )

        return float(final_value)
