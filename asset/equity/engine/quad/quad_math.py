"""
Shared quadrature math utilities for grid setup and convolution.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from util.exceptions import NumericalError, ValidationError


class QuadratureMath:
    """Reusable quadrature math utilities for log-price grids."""

    def __init__(
        self,
        grid_x: int,
        spot: float,
        maturity: float,
        vol_max: float,
    ) -> None:
        self.grid_x = int(grid_x)
        self.spot = float(spot)
        self.maturity = float(maturity)
        self.vol_max = float(vol_max)

        if self.grid_x < 3:
            raise ValidationError("grid_points must be at least 3.")
        if self.spot <= 0.0:
            raise ValidationError(f"spot must be positive, got {self.spot}.")
        if self.maturity <= 0.0:
            raise ValidationError(f"maturity must be positive, got {self.maturity}.")
        if self.vol_max <= 0.0:
            raise ValidationError(f"volatility must be positive, got {self.vol_max}.")

        self.constant_c = math.exp(self._factor_c())
        log_c = math.log(self.constant_c)
        self.grid = np.linspace(-log_c, log_c, self.grid_x)
        self.h = 2.0 * log_c / (self.grid_x - 1)
        self.z_grid = -2.0 * log_c + np.arange(2 * self.grid_x - 1) * self.h
        self._weights: np.ndarray | None = None

    def _factor_c(self) -> float:
        return (
            10.0 * self.vol_max * math.sqrt(self.maturity)
            + (1.0 + 0.5 * self.vol_max * self.vol_max) * self.maturity
        )

    def select_simpson_indices(self, bound_lr: float, bound_ur: float) -> Tuple[int, int, int]:
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

    def simpson_weights(
        self, values: np.ndarray, p_lr: int, p_ur: int, p0: int
    ) -> np.ndarray:
        u_array = np.zeros(2 * self.grid_x - 1)
        u_array[p_lr] = values[p_lr]
        u_array[p_ur + p0] = values[p_ur + p0]
        u_array[p_lr + 1 : p_ur + p0 : 2] = 4.0 * values[p_lr + 1 : p_ur + p0 : 2]
        u_array[p_lr + 2 : p_ur + p0 - 1 : 2] = 2.0 * values[p_lr + 2 : p_ur + p0 - 1 : 2]
        return u_array

    def simpson_weight_vector(self) -> np.ndarray:
        if self._weights is None:
            weights = np.ones(self.grid_x)
            weights[1:-1:2] = 4.0
            weights[2:-1:2] = 2.0
            self._weights = weights * self.h / 3.0
        return self._weights

    def convolution_fft(self, omega_array: np.ndarray, u_array: np.ndarray) -> np.ndarray:
        omega_array = np.asarray(omega_array).ravel()
        u_array = np.asarray(u_array).ravel()
        if len(omega_array) != len(u_array):
            raise NumericalError("omega_array and u_array must have the same length.")
        f_array = np.fft.ifft(np.fft.fft(omega_array) * np.fft.fft(u_array)).real
        return f_array[self.grid_x - 1 : 2 * self.grid_x - 1] * self.h / 3.0

    def interpolate(self, values: np.ndarray, x: float = 0.0) -> float:
        return float(np.interp(x, self.grid, values))
