"""
Shared quadrature math utilities for grid setup and convolution.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

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
        num_std_devs: float = 10.0,
        *,
        align_log: Optional[float] = None,
        fft_padding_factor: int = 1,
        fft_filter_alpha: float = 0.0,
        fft_filter_power: int = 8,
    ) -> None:
        self.grid_x = int(grid_x)
        self.spot = float(spot)
        self.maturity = float(maturity)
        self.vol_max = float(vol_max)
        self.num_std_devs = float(num_std_devs)
        self.align_log = float(align_log) if align_log is not None else None
        self.fft_padding_factor = int(fft_padding_factor)
        self.fft_filter_alpha = float(fft_filter_alpha)
        self.fft_filter_power = int(fft_filter_power)

        if self.grid_x < 3:
            raise ValidationError("grid_points must be at least 3.")
        if self.spot <= 0.0:
            raise ValidationError(f"spot must be positive, got {self.spot}.")
        if self.maturity <= 0.0:
            raise ValidationError(f"maturity must be positive, got {self.maturity}.")
        if self.vol_max <= 0.0:
            raise ValidationError(f"volatility must be positive, got {self.vol_max}.")
        if self.num_std_devs <= 0.0:
            raise ValidationError(
                f"num_std_devs must be positive, got {self.num_std_devs}."
            )
        if self.num_std_devs < 3.0:
            raise ValidationError(
                f"num_std_devs should be at least 3, got {self.num_std_devs}."
            )

        if self.fft_padding_factor < 1:
            raise ValidationError(
                f"fft_padding_factor must be >= 1, got {self.fft_padding_factor}."
            )
        if self.fft_filter_alpha < 0.0:
            raise ValidationError(
                f"fft_filter_alpha must be non-negative, got {self.fft_filter_alpha}."
            )
        if self.fft_filter_power < 1:
            raise ValidationError(
                f"fft_filter_power must be >= 1, got {self.fft_filter_power}."
            )

        self.constant_c = math.exp(self._factor_c())
        log_c = math.log(self.constant_c)
        self.h = 2.0 * log_c / (self.grid_x - 1)
        grid_shift = 0.0
        if (
            self.align_log is not None
            and math.isfinite(self.align_log)
            and -log_c <= self.align_log <= log_c
        ):
            idx = int(round((self.align_log + log_c) / self.h))
            idx = max(0, min(idx, self.grid_x - 1))
            grid_shift = self.align_log - (-log_c + idx * self.h)

        self.grid = np.linspace(-log_c, log_c, self.grid_x) + grid_shift
        self.z_grid = -2.0 * log_c + np.arange(2 * self.grid_x - 1) * self.h
        self._weights: np.ndarray | None = None
        self._fft_filter_cache: dict[int, np.ndarray] = {}
        self._omega_fft_cache: dict[tuple[int, bytes], np.ndarray] = {}

    def _factor_c(self) -> float:
        return (
            self.num_std_devs * self.vol_max * math.sqrt(self.maturity)
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

    def simpson_weights_many(
        self, values: np.ndarray, p_lr: int, p_ur: int, p0: int
    ) -> np.ndarray:
        values = np.asarray(values)
        if values.ndim != 2:
            raise NumericalError("values must be a 2D array.")
        if values.shape[1] != self.grid_x:
            raise NumericalError(
                f"values second dimension must be grid_x={self.grid_x}, "
                f"got {values.shape[1]}."
            )

        u_array = np.zeros((values.shape[0], 2 * self.grid_x - 1), dtype=values.dtype)
        u_array[:, p_lr] = values[:, p_lr]
        u_array[:, p_ur + p0] = values[:, p_ur + p0]
        u_array[:, p_lr + 1 : p_ur + p0 : 2] = (
            4.0 * values[:, p_lr + 1 : p_ur + p0 : 2]
        )
        u_array[:, p_lr + 2 : p_ur + p0 - 1 : 2] = (
            2.0 * values[:, p_lr + 2 : p_ur + p0 - 1 : 2]
        )
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
        base_len = len(u_array)
        fft_len = base_len if self.fft_padding_factor <= 1 else int(self.fft_padding_factor * base_len)
        omega_fft = self._get_omega_fft(omega_array, fft_len)
        u_fft = np.fft.fft(u_array, n=fft_len)
        product = omega_fft * u_fft
        fft_filter = self._get_fft_filter(fft_len)
        if fft_filter is not None:
            product *= fft_filter
        f_array = np.fft.ifft(product).real
        return f_array[self.grid_x - 1 : 2 * self.grid_x - 1] * self.h / 3.0

    def convolution_fft_many(
        self, omega_array: np.ndarray, u_array: np.ndarray
    ) -> np.ndarray:
        omega_array = np.asarray(omega_array).ravel()
        u_array = np.asarray(u_array)
        if u_array.ndim != 2:
            raise NumericalError("u_array must be a 2D array.")
        if u_array.shape[1] != len(omega_array):
            raise NumericalError(
                "omega_array length must match u_array second dimension."
            )

        base_len = len(omega_array)
        fft_len = (
            base_len
            if self.fft_padding_factor <= 1
            else int(self.fft_padding_factor * base_len)
        )
        omega_fft = self._get_omega_fft(omega_array, fft_len)
        u_fft = np.fft.fft(u_array, n=fft_len, axis=1)
        product = u_fft * omega_fft.reshape(1, -1)
        fft_filter = self._get_fft_filter(fft_len)
        if fft_filter is not None:
            product *= fft_filter.reshape(1, -1)
        f_array = np.fft.ifft(product, axis=1).real
        return f_array[:, self.grid_x - 1 : 2 * self.grid_x - 1] * self.h / 3.0

    def _get_omega_fft(self, omega_array: np.ndarray, fft_len: int) -> np.ndarray:
        omega_array = np.ascontiguousarray(omega_array, dtype=float)
        key = (int(fft_len), omega_array.tobytes())
        cached = self._omega_fft_cache.get(key)
        if cached is not None:
            return cached
        omega_fft = np.fft.fft(omega_array, n=fft_len)
        self._omega_fft_cache[key] = omega_fft
        return omega_fft

    def _get_fft_filter(self, length: int) -> Optional[np.ndarray]:
        if self.fft_filter_alpha <= 0.0:
            return None
        cached = self._fft_filter_cache.get(length)
        if cached is not None:
            return cached
        freq = np.fft.fftfreq(length)
        max_freq = np.max(np.abs(freq))
        if max_freq <= 0.0:
            return None
        scaled = np.abs(freq) / max_freq
        filt = np.exp(-self.fft_filter_alpha * (scaled ** self.fft_filter_power))
        self._fft_filter_cache[length] = filt
        return filt

    def interpolate(self, values: np.ndarray, x: float = 0.0) -> float:
        return float(np.interp(x, self.grid, values))
