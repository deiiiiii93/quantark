# -*- coding: utf-8 -*-
"""
Created on Thu May 30 2024

@author: yaofuxin
@description: A simple implementation of quadrature method for pricing options with
early-exercise features. Based on paper "A SIMPLE AND EFFICIENT NUMERICAL METHOD FOR
PRICING DISCRETELY MONITORED EARLY-EXERCISE OPTIONS".
"""

import numpy as np
from scipy.stats import norm
from scipy.fftpack import fft, ifft
from numba import jit
import pyfftw


@jit(nopython=True)
def calculate_integral_simpson_numba(values, grid_x, p_lr, p_ur, p0):
    U_array = np.zeros(2 * grid_x - 1)
    U_array[p_lr] = values[p_lr]
    U_array[p_ur + p0] = values[p_ur + p0]
    for i in range(p_lr + 1, p_ur + p0, 2):
        U_array[i] = 4 * values[i]
    for i in range(p_lr + 2, p_ur + p0 - 1, 2):
        U_array[i] = 2 * values[i]
    return U_array


class QuadratureOptionPricer:
    """使用数值积分方法对离散观察期权定价的类。"""

    def __init__(
        self,
        grid_x: int,
        grid_t: int,
        maturity: float,
        spot: float,
        r: float,
        q: float,
        vol: float,
        integ_method: str = "simpson",
    ):
        """初始化定价器。

        Args:
            grid_x: x方向网格点数
            grid_t: t方向网格点数
            maturity: 到期时间
            spot: 标的资产现价
            r: 无风险利率
            q: 分红率
            vol: 波动率
        """
        self.grid_x = grid_x if grid_x > 0 else 1000
        self.grid_t = grid_t if grid_t > 0 else int(244 * maturity)
        self.maturity = maturity
        self.spot = spot
        self.r = r
        self.q = q
        self.vol = vol

        integ_method_dict = {
            "simpson": self._calculate_integral_simpson,
            "simpson_numba": self._calculate_integral_simpson_numba,
        }
        self._process_integrate = integ_method_dict.get(
            integ_method.lower(), self._calculate_integral_simpson
        )

        # 初始化网格和参数
        self.dt = maturity / self.grid_t
        self.tau = 0.5 * vol**2 * self.dt
        self.alpha = (r - q - 0.5 * vol**2) / vol**2
        self.beta = (r - q - 0.5 * vol**2) ** 2 / vol**4 + 2 * r / vol**2

        # 设置积分边界
        self.constant_c = np.exp(self._factor_c())
        self.grid = np.linspace(
            -np.log(self.constant_c), np.log(self.constant_c), self.grid_x
        )
        self.h = 2 * np.log(self.constant_c) / (self.grid_x - 1)

    def _factor_c(self) -> float:
        """计算积分边界的系数C。参考论文公式(8.2)。"""
        vol_max = self.vol if np.isscalar(self.vol) else np.max(self.vol)
        return (
            10 * vol_max * np.sqrt(self.maturity)
            + (1 + 0.5 * vol_max**2) * self.maturity
        )

    def _factor_value_at_m(
        self, spot: float, strike: float, epsilon: int, type: str
    ) -> float:
        """计算二元期权的价值。参考论文引理3.1。"""
        # 处理特殊情况
        if strike <= 0:
            if type == "a":
                return (
                    spot * np.exp(-2 * self.q * self.tau / self.vol**2)
                    if epsilon > 0
                    else 0
                )
            else:  # type == "b"
                return (
                    np.exp(-2 * self.r * self.tau / self.vol**2) if epsilon > 0 else 0
                )

        # 正常情况的计算
        d1 = (
            np.log(spot / strike)
            + (self.r - self.q + 0.5 * self.vol**2) * self.tau * 2 / self.vol**2
        )
        d1 = d1 / np.sqrt(2 * self.tau)

        if type == "a":  # 资产二元期权
            return (
                spot
                * np.exp(-2 * self.q * self.tau / self.vol**2)
                * norm.cdf(epsilon * d1)
            )
        elif type == "b":  # 现金二元期权
            d2 = d1 - np.sqrt(2 * self.tau)
            return np.exp(-2 * self.r * self.tau / self.vol**2) * norm.cdf(epsilon * d2)
        else:
            raise ValueError("type must be 'a' or 'b'")

    def _option_value_at_previous_m(
        self,
        spot: np.ndarray | float,
        strike_upper: float,
        strike_lower: float,
        factors: dict,
        m: int = -1,
    ) -> np.ndarray | float:
        """计算前一个时间点的期权价值。参考论文命题3.3。

        Args:
            spot: 标的资产现价(可以是数组或标量)
            strike_upper: 上边界价格
            strike_lower: 下边界价格
            factors: 系数字典
            m: 时间索引,默认为-1
        """
        # 计算6个二元期权的价值
        v1 = self._factor_value_at_m(spot, strike_lower, -1, "a")
        v2 = self._factor_value_at_m(spot, strike_lower, -1, "b")
        v3 = self._factor_value_at_m(spot, strike_lower, 1, "a")
        v4 = self._factor_value_at_m(spot, strike_upper, 1, "a")
        v5 = self._factor_value_at_m(spot, strike_lower, 1, "b")
        v6 = self._factor_value_at_m(spot, strike_upper, 1, "b")

        # 返回二元期权的线性组合
        return (
            factors["asset1"][m] * v1
            + factors["cash1"][m] * v2
            + factors["asset2"][m] * (v3 - v4)
            + factors["cash2"][m] * (v5 - v6)
            + factors["asset3"][m] * v4
            + factors["cash3"][m] * v6
        )

    def _omega(self, x: np.ndarray) -> np.ndarray:
        """计算权重函数ω。参考论文第4节。"""
        return np.exp(-(x**2) / self.tau / 4 - self.alpha * x)

    def _calculate_integral_simpson(
        self, values: np.ndarray, p_lr: int, p_ur: int, p0: int
    ) -> np.ndarray:
        """使用Simpson规则计算积分。参考论文公式(4.4)。"""
        U_array = np.zeros(2 * self.grid_x - 1)
        U_array[p_lr] = values[p_lr]
        U_array[p_ur + p0] = values[p_ur + p0]
        U_array[p_lr + 1 : p_ur + p0 : 2] = 4 * values[p_lr + 1 : p_ur + p0 : 2]
        U_array[p_lr + 2 : p_ur + p0 - 1 : 2] = 2 * values[p_lr + 2 : p_ur + p0 - 1 : 2]
        return U_array

    def _calculate_integral_simpson_numba(
        self, values: np.ndarray, p_lr: int, p_ur: int, p0: int
    ) -> np.ndarray:
        """使用Numba加速的Simpson积分计算"""
        return calculate_integral_simpson_numba(values, self.grid_x, p_lr, p_ur, p0)

    def _calculate_convolution_fft(
        self, omega_array: np.ndarray, U_array: np.ndarray
    ) -> np.ndarray:
        """使用FFT计算卷积。参考论文命题4.1。"""
        # 确保输入数组是一维的
        omega_array = np.asarray(omega_array).ravel()
        U_array = np.asarray(U_array).ravel()

        # 确保两个数组长度相同
        if len(omega_array) != len(U_array):
            raise ValueError("omega_array and U_array must have the same length")

        # 执行FFT计算
        F_array = ifft(fft(omega_array) * fft(U_array)).real

        # 确保返回正确的切片
        start_idx = self.grid_x - 1
        end_idx = min(2 * self.grid_x - 1, len(F_array))

        return F_array[start_idx:end_idx] * self.h / 3

    def _calculate_convolution_fft_2(
        self, omega_array: np.ndarray, U_array: np.ndarray
    ) -> np.ndarray:
        """使用FFT计算卷积。参考论文命题4.1。"""
        # 确保输入数组是一维的
        omega_array = np.asarray(omega_array).ravel()
        U_array = np.asarray(U_array).ravel()

        # 确保两个数组长度相同
        if len(omega_array) != len(U_array):
            raise ValueError("omega_array and U_array must have the same length")

        # 使用 pyFFTW 进行 FFT 计算
        fft_omega = pyfftw.interfaces.numpy_fft.fft(omega_array)
        fft_U = pyfftw.interfaces.numpy_fft.fft(U_array)
        F_array = pyfftw.interfaces.numpy_fft.ifft(fft_omega * fft_U).real

        # 确保返回正确的切片
        start_idx = self.grid_x - 1
        end_idx = min(2 * self.grid_x - 1, len(F_array))

        return F_array[start_idx:end_idx] * self.h / 3

    def price(self, ko_data: dict) -> float:
        """计算期权价值。"""
        # 初始化时间网格
        grid_t = np.linspace(0, self.maturity, self.grid_t + 1)
        ko_idx = np.searchsorted(grid_t, ko_data["ko_dates_u"])
        ki_idx = np.searchsorted(grid_t, ko_data["ko_dates_l"])

        # 设置边界条件
        self.ko = np.ones(self.grid_t + 1) * self.spot * self.constant_c
        self.ki = np.ones(self.grid_t + 1) * self.spot / self.constant_c
        self.ko[ko_idx] = ko_data["ko_prices_u"]
        self.ki[ki_idx] = ko_data["ko_prices_l"]

        # 计算积分边界
        integral_bound_upper = np.minimum(self.ko, self.spot * self.constant_c)
        integral_bound_lower = np.maximum(self.ki, self.spot / self.constant_c)
        integral_bound_upper = np.log(integral_bound_upper / self.spot)
        integral_bound_lower = np.log(integral_bound_lower / self.spot)

        # 计算M-1时刻的值
        values_at_M_n1 = self._calculate_values_at_M_minus_1(
            integral_bound_upper, integral_bound_lower, ko_data["factors"]
        )

        # 向后递归计算M-2到1时刻的值
        result = self._backward_recursion(
            values_at_M_n1,
            integral_bound_upper,
            integral_bound_lower,
            ko_idx,
            ki_idx,
            ko_data["factors"],
        )

        return result

    def _calculate_values_at_M_minus_1(
        self, bound_upper: np.ndarray, bound_lower: np.ndarray, factors: dict
    ) -> tuple:
        """计算M-1时刻的期权价值。"""
        # 计算积分边界点
        bound_ur_M_n1 = bound_upper[-2]  # M-1时刻的上边界
        bound_lr_M_n1 = bound_lower[-2]  # M-1时刻的下边界

        # 找到边界对应的网格点索引
        p_lr_M_n1 = np.argmax(self.grid >= bound_lr_M_n1)
        p_ur_M_n1 = np.argmin(self.grid < bound_ur_M_n1) - 1
        p0_M_n1 = (p_ur_M_n1 - p_lr_M_n1) % 2

        # 计算额外的积分点
        xee_lr_M_n1 = 0.5 * (self.grid[p_lr_M_n1] + bound_lr_M_n1)
        xee_ur_M_n1 = 0.5 * (self.grid[p_ur_M_n1 + p0_M_n1] + bound_ur_M_n1)
        x_M_n1 = np.array([bound_lr_M_n1, bound_ur_M_n1, xee_lr_M_n1, xee_ur_M_n1])

        # 计算这些点的期权价值
        spot_array_M_n1 = self.spot * np.exp(x_M_n1)
        values_at_M_n1 = self._option_value_at_previous_m(
            spot_array_M_n1, self.ko[-2], self.ki[-2], factors, -1
        )

        values = (
            values_at_M_n1,
            p_lr_M_n1,
            p_ur_M_n1,
            p0_M_n1,
            x_M_n1,
            bound_lr_M_n1,
            bound_ur_M_n1,
            xee_lr_M_n1,
            xee_ur_M_n1,
        )

        return values

    def _backward_recursion(
        self,
        M_minus_1_data: tuple,
        bound_upper: np.ndarray,
        bound_lower: np.ndarray,
        ko_idx: np.ndarray,
        ki_idx: np.ndarray,
        factors: dict,
    ) -> float:
        """向后递归计算期权价值。参考论文第4节。"""
        (
            values_at_M_n1,
            p_lr_m,
            p_ur_m,
            p0_m,
            x_m,
            bound_lr_m,
            bound_ur_m,
            xee_lr_m,
            xee_ur_m,
        ) = M_minus_1_data
        v_bound_lr_m, v_bound_ur_m, v_xee_lr_m, v_xee_ur_m = values_at_M_n1

        # 初始化网格上的期权价值
        spot_array = self.spot * np.exp(self.grid)
        y_array = self._option_value_at_previous_m(
            spot_array, self.ko[-1], self.ki[-1], factors, -1
        )

        # 计算Simpson权重
        U_array = self._process_integrate(y_array, p_lr_m, p_ur_m, p0_m)

        # 计算omega数组
        z = np.array(
            [
                -2 * np.log(self.constant_c) + (i - 1) * self.h
                for i in range(1, 2 * self.grid_x)
            ]
        )
        omega_array = self._omega(z)

        # 向后递归
        for m in range(self.grid_t - 1, 1, -1):
            on_barrier = False
            if m in ko_idx or m + 1 in ko_idx or m - 1 in ko_idx:
                on_barrier = True
            if m in ki_idx or m + 1 in ki_idx or m - 1 in ki_idx:
                on_barrier = True
                # 处理敲出观察日期
            values = self._calculate_values_in_process(
                m,
                U_array,
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

            # 更新所有值用于下次迭代
            y_array = values["y_array"]
            U_array = values["U_array"]
            v_bound_lr_m, v_bound_ur_m, v_xee_lr_m, v_xee_ur_m = values["values"]
            p_lr_m, p_ur_m, p0_m = values["points"]
            bound_lr_m, bound_ur_m, xee_lr_m, xee_ur_m = values["boundaries"]
        # 计算最终结果
        return self._calculate_final_value(
            y_array,
            U_array,
            factors,
            values["values"],
            values["points"],
            values["boundaries"],
        )

    def _calculate_values_in_process(
        self,
        m: int,
        U_array: np.ndarray,
        omega_array: np.ndarray,
        y_array: np.ndarray,
        bound_upper: np.ndarray,
        bound_lower: np.ndarray,
        factors: dict,
        prev_values: tuple,
        prev_points: tuple,
        prev_boundaries: tuple,
        on_barrier: bool = False,
    ) -> dict:
        """计算敲出观察日期的期权价值。"""
        # 解包前一次迭代的值
        bound_lr_m, bound_ur_m, xee_lr_m, xee_ur_m = prev_boundaries
        v_bound_lr_m, v_bound_ur_m, v_xee_lr_m, v_xee_ur_m = prev_values
        p_lr_m, p_ur_m, p0_m = prev_points

        if on_barrier:
            # 更新边界条件和网格点
            bound_lr_m_n1 = bound_lower[m - 1]
            bound_ur_m_n1 = bound_upper[m - 1]
            p_lr_m_n1 = np.argmax(self.grid >= bound_lr_m_n1)
            p_ur_m_n1 = np.argmin(self.grid < bound_ur_m_n1) - 1
            p0_m_n1 = (p_ur_m_n1 - p_lr_m_n1) % 2
            xee_lr_m_n1 = 0.5 * (self.grid[p_lr_m_n1] + bound_lr_m_n1)
            xee_ur_m_n1 = 0.5 * (self.grid[p_ur_m_n1 + p0_m_n1] + bound_ur_m_n1)
            x_m_n1 = np.array([bound_lr_m_n1, bound_ur_m_n1, xee_lr_m_n1, xee_ur_m_n1])
        else:
            bound_lr_m_n1 = bound_lr_m
            bound_ur_m_n1 = bound_ur_m
            p_lr_m_n1 = p_lr_m
            p_ur_m_n1 = p_ur_m
            p0_m_n1 = p0_m
            xee_lr_m_n1 = xee_lr_m
            xee_ur_m_n1 = xee_ur_m
            x_m_n1 = np.array([bound_lr_m, bound_ur_m, xee_lr_m, xee_ur_m])

        # 计算y1, y2, y3，使用前一次迭代的边界值
        y1 = self._calculate_convolution_fft(omega_array, U_array)  # 使用FFT计算卷积

        y2 = (self.grid[p_lr_m] - bound_lr_m) / 6
        y2 = y2 * (
            self._omega(self.grid - bound_lr_m) * v_bound_lr_m
            + 4 * self._omega(self.grid - xee_lr_m) * v_xee_lr_m
            + self._omega(self.grid - self.grid[p_lr_m]) * y_array[p_lr_m]
        )

        y3 = (bound_ur_m - self.grid[p_ur_m + p0_m]) / 6
        y3 = y3 * (
            self._omega(self.grid - bound_ur_m) * v_bound_ur_m
            + 4 * self._omega(self.grid - xee_ur_m) * v_xee_ur_m
            + self._omega(self.grid - self.grid[p_ur_m + p0_m]) * y_array[p_ur_m + p0_m]
        )

        # 计算边界值
        a = []
        for s in x_m_n1:
            tmp = (
                self._omega(
                    s - self.grid[p_lr_m : p_ur_m + p0_m + 1],
                )
                * U_array[p_lr_m : p_ur_m + p0_m + 1]
            )
            a.append(np.sum(tmp) * self.h / 3)
        a = np.array(a)

        b = (self.grid[p_lr_m] - bound_lr_m) / 6
        b = b * (
            self._omega(x_m_n1 - bound_lr_m) * v_bound_lr_m
            + 4 * self._omega(x_m_n1 - xee_lr_m) * v_xee_lr_m
            + self._omega(x_m_n1 - self.grid[p_lr_m]) * y_array[p_lr_m]
        )

        c = (bound_ur_m - self.grid[p_ur_m + p0_m]) / 6
        c = c * (
            self._omega(x_m_n1 - bound_ur_m) * v_bound_ur_m
            + 4 * self._omega(x_m_n1 - xee_ur_m) * v_xee_ur_m
            + self._omega(x_m_n1 - self.grid[p_ur_m + p0_m]) * y_array[p_ur_m + p0_m]
        )

        # 更新边界值
        spot_array_m_n1 = self.spot * np.exp(x_m_n1)
        v_bound_lr_m, v_bound_ur_m, v_xee_lr_m, v_xee_ur_m = (
            (a + b + c) * np.exp(-self.beta * self.tau) / np.sqrt(np.pi * self.tau) / 2
        ) + (
            factors["asset3"][m]
            * self._factor_value_at_m(spot_array_m_n1, self.ko[m], 1, "a")
            + factors["cash3"][m]
            * self._factor_value_at_m(spot_array_m_n1, self.ko[m], 1, "b")
            + factors["asset1"][m]
            * self._factor_value_at_m(spot_array_m_n1, self.ki[m], -1, "a")
            + factors["cash1"][m]
            * self._factor_value_at_m(spot_array_m_n1, self.ki[m], -1, "b")
        )

        # 计算新的期权价值
        y_array_new = (
            (y1 + y2 + y3)
            * np.exp(-self.beta * self.tau)
            / np.sqrt(np.pi * self.tau)
            / 2
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

        # 更新Simpson权重
        U_array_new = self._process_integrate(
            y_array_new, p_lr_m_n1, p_ur_m_n1, p0_m_n1
        )

        return {
            "y_array": y_array_new,
            "U_array": U_array_new,
            "values": (v_bound_lr_m, v_bound_ur_m, v_xee_lr_m, v_xee_ur_m),
            "points": (p_lr_m_n1, p_ur_m_n1, p0_m_n1),
            "boundaries": (bound_lr_m_n1, bound_ur_m_n1, xee_lr_m_n1, xee_ur_m_n1),
            "x": x_m_n1,
        }

    def _calculate_final_value(
        self,
        y_array: np.ndarray,
        U_array: np.ndarray,
        factors: dict,
        final_values: tuple,
        final_points: tuple,
        final_boundaries: tuple,
    ) -> float:
        """计算最终的期权价值。"""

        bound_lr_m, bound_ur_m, xee_lr_m, xee_ur_m = final_boundaries
        v_bound_lr_m, v_bound_ur_m, v_xee_lr_m, v_xee_ur_m = final_values
        p_lr_m, p_ur_m, p0_m = final_points

        y1 = np.sum(
            self._omega(0 - self.grid[p_lr_m : p_ur_m + p0_m + 1])
            * U_array[p_lr_m : p_ur_m + p0_m + 1]
            * self.h
            / 3
        )

        y2 = (self.grid[p_lr_m] - bound_lr_m) / 6
        y2 = y2 * (
            self._omega(0 - bound_lr_m) * v_bound_lr_m
            + 4 * self._omega(0 - xee_lr_m) * v_xee_lr_m
            + self._omega(0 - self.grid[p_lr_m]) * y_array[p_lr_m]
        )

        y3 = (bound_ur_m - self.grid[p_ur_m + p0_m]) / 6
        y3 = y3 * (
            self._omega(0 - bound_ur_m) * v_bound_ur_m
            + 4 * self._omega(0 - xee_ur_m) * v_xee_ur_m
            + self._omega(0 - self.grid[p_ur_m + p0_m]) * y_array[p_ur_m + p0_m]
        )

        final_value = (
            (y1 + y2 + y3)
            * np.exp(-self.beta * self.tau)
            / np.sqrt(np.pi * self.tau)
            / 2
        )

        # 添加边界条件的贡献
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

        return final_value


# 添加便捷的定价函数
def price_one_touch(
    grid_x: int,
    grid_t: int,
    maturity: float,
    ko_prices: list,
    ko_dates: list,
    ko_cal_dates: list,
    ko_return: float,
    is_annualized: bool,
    is_call: bool,
    spot: float,
    r: float,
    q: float,
    vol: float,
) -> float:
    """定价单触碰期权。"""
    pricer = QuadratureOptionPricer(grid_x, grid_t, maturity, spot, r, q, vol)

    # 将输入转换为numpy数组
    ko_dates = np.array(ko_dates)
    ko_cal_dates = np.array(ko_cal_dates)
    ko_prices = np.array(ko_prices)

    # 准备敲出数据
    ko_data = {
        "ko_prices_u": ko_prices if is_call else [float("inf")] * len(ko_dates),
        "ko_prices_l": 0 if is_call else ko_prices,
        "ko_dates_u": ko_dates,
        "ko_dates_l": [],
        "factors": {
            "asset1": np.zeros(grid_t + 1),  # 使用numpy数组
            "asset2": np.zeros(grid_t + 1),
            "asset3": np.zeros(grid_t + 1),
            "cash1": np.zeros(grid_t + 1),
            "cash2": np.zeros(grid_t + 1),
            "cash3": np.zeros(grid_t + 1),
        },
    }

    # 设置观察日期的因子
    grid_t_array = np.linspace(0, maturity, grid_t + 1)
    ko_idx = np.searchsorted(grid_t_array, ko_dates)

    discount_factors = np.exp(-r * ko_dates) if is_annualized else np.exp(-r * maturity)
    ko_return = ko_return * discount_factors

    if is_call:
        ko_data["factors"]["cash3"][ko_idx] = (
            ko_return if not is_annualized else ko_return * ko_cal_dates
        )
    else:
        ko_data["factors"]["cash1"][ko_idx] = (
            ko_return if not is_annualized else ko_return * ko_cal_dates
        )

    return pricer.price(ko_data)


def price_double_touch(
    grid_x: int,
    grid_t: int,
    maturity: float,
    ko_prices_u: np.ndarray,
    ko_prices_l: np.ndarray,
    ko_dates_u: np.ndarray,
    ko_dates_l: np.ndarray,
    touch_type: str,
    rebate: float,
    spot: float,
    r: float,
    q: float,
    vol: float,
) -> float:
    """计算双触碰期权价值。

    Args:
        grid_x: x方向网格点数
        grid_t: t方向网格点数
        maturity: 到期时间（年）
        ko_prices_u: 上边界价格数组
        ko_prices_l: 下边界价格数组
        ko_dates_u: 上边界观察日期数组（年化）
        ko_dates_l: 下边界观察日期数组（年化）
        payoff_asset: 资产支付系数
        payoff_cash: 现金支付金额
        spot: 标的价格
        r: 无风险利率
        q: 分红率
        vol: 波动率

    Returns:
        float: 期权价值
    """
    pricer = QuadratureOptionPricer(grid_x, grid_t, maturity, spot, r, q, vol)
    touch_data = {
        "ko_prices_u": ko_prices_u,
        "ko_prices_l": ko_prices_l,
        "ko_dates_u": ko_dates_u,
        "ko_dates_l": ko_dates_l,
        "factors": {
            "asset1": np.zeros(grid_t + 1),
            "asset2": np.zeros(grid_t + 1),
            "asset3": np.zeros(grid_t + 1),
            "cash1": np.zeros(grid_t + 1),
            "cash2": np.zeros(grid_t + 1),
            "cash3": np.zeros(grid_t + 1),
        },
    }
    discount_factors = np.exp(-r * maturity)
    rebate = rebate * discount_factors

    if touch_type == "no_touch":
        touch_data["factors"]["cash2"][-1] = rebate
    elif touch_type == "touch":
        touch_data["factors"]["cash1"][-1] = rebate
        touch_data["factors"]["cash3"][-1] = rebate

    return pricer.price(touch_data)


def price_single_barrier(
    grid_x: int,
    grid_t: int,
    maturity: float,
    ko_prices: np.ndarray,
    ko_dates: np.ndarray,
    call_or_put: str,
    strike: float,
    spot: float,
    r: float,
    q: float,
    vol: float,
    ko_direction: str = "up",  # 可选值 "up" 或 "down"
) -> float:
    """计算单障碍期权价值。

    对于看涨期权：
       - 上敲：采用 ko_prices 作为上侧障碍（ko_prices_u），下侧障碍设为0；
       - 下敲：采用 ko_prices 作为下侧障碍（ko_prices_l），上侧障碍取无穷大。
    对于看跌期权：
       - 一般情况下 put 期权多为下敲，但在 autocallable（雪球）产品中也常见上敲 put，
         此时若 ko_direction 为 "up"，则将上侧障碍设置为 ko_prices， 下侧障碍设为0；
       - 若为下敲 put，则 ko_prices 作为下侧障碍，将上侧障碍设为无穷大。

    Args:
        grid_x: x方向网格点数
        grid_t: t方向网格点数
        maturity: 到期时间（年）
        ko_prices: 障碍价格数组
        ko_dates: 观察障碍的日期数组（年化）
        call_or_put: "call" 或 "put"
        strike: 行权价格
        spot: 标的价格
        r: 无风险利率
        q: 分红率
        vol: 波动率
        ko_direction: 障碍类型，"up" 表示上敲，"down" 表示下敲

    Returns:
        float: 期权价值
    """
    pricer = QuadratureOptionPricer(grid_x, grid_t, maturity, spot, r, q, vol)
    # 初始化 factors
    factors = {
        "asset1": np.zeros(grid_t + 1),
        "asset2": np.zeros(grid_t + 1),
        "asset3": np.zeros(grid_t + 1),
        "cash1": np.zeros(grid_t + 1),
        "cash2": np.zeros(grid_t + 1),
        "cash3": np.zeros(grid_t + 1),
    }

    if call_or_put == "call":
        if ko_direction == "up":
            barrier_data = {
                "ko_prices_u": np.array(ko_prices, copy=True),  # 创建一个独立的副本
                "ko_prices_l": 0,  # 下侧障碍为0
                "ko_dates_u": ko_dates,
                "ko_dates_l": np.array([]),
                "factors": factors,
            }
            barrier_data["ko_prices_u"][-1] = strike
        elif ko_direction == "down":
            barrier_data = {
                "ko_prices_u": np.inf,
                "ko_prices_l": np.array(ko_prices, copy=True),  # 下侧障碍
                "ko_dates_u": np.array([]),
                "ko_dates_l": ko_dates,
                "factors": factors,
            }
            barrier_data["ko_prices_l"][-1] = strike
        else:
            raise ValueError("invalid ko_direction value")
        payoff_asset = 1  # 到期 payoff 为 S - strike
        payoff_cash = -strike
    elif call_or_put == "put":
        if ko_direction == "up":
            # 对于上敲 put（常见于雪球产品）：当标的价格上破障碍时敲出
            barrier_data = {
                "ko_prices_u": np.array(ko_prices, copy=True),  # 上侧障碍为实际障碍价
                "ko_prices_l": 0,  # 下侧不设障碍
                "ko_dates_u": ko_dates,
                "ko_dates_l": np.array([]),
                "factors": factors,
            }
            barrier_data["ko_prices_u"][-1] = strike
        elif ko_direction == "down":
            # 对于下敲 put，障碍在下侧：
            barrier_data = {
                "ko_prices_u": np.inf,  # 上侧设为无穷大
                "ko_prices_l": np.array(ko_prices, copy=True),  # 下侧障碍为实际障碍价
                "ko_dates_u": np.array([]),
                "ko_dates_l": ko_dates,
                "factors": factors,
            }
            barrier_data["ko_prices_l"][-1] = strike
        else:
            raise ValueError("invalid ko_direction value")
        payoff_asset = -1  # 到期 payoff 为 strike - S
        payoff_cash = strike
    else:
        raise ValueError(f"invalid call_or_put value {call_or_put}")

    discount_factors = np.exp(-r * maturity)
    payoff_asset = payoff_asset * discount_factors
    payoff_cash = payoff_cash * discount_factors

    # 在最后一期设置 payoff（线性组合代表最终权利金，例如 S - strike 或 strike - S）
    barrier_data["factors"]["asset2"][-1] = payoff_asset
    barrier_data["factors"]["cash2"][-1] = payoff_cash

    return pricer.price(barrier_data)


def price_double_barrier(
    grid_x: int,
    grid_t: int,
    maturity: float,
    ko_prices_u: np.ndarray,
    ko_prices_l: float | np.ndarray,
    ko_dates_u: np.ndarray,
    ko_dates_l: np.ndarray,
    call_or_put: str,
    strike: float,
    spot: float,
    r: float,
    q: float,
    vol: float,
) -> float:
    """计算单障碍期权价值。

    Args:
        grid_x: x方向网格点数
        grid_t: t方向网格点数
        maturity: 到期时间（年）
        ko_prices_u: 上边界价格数组
        ko_prices_l: 下边界价格（单一值）
        ko_dates_u: 上边界观察日期数组（年化）
        ko_dates_l: 下边界观察日期数组（年化）
        call_or_put: 看涨/看跌
        strike: 行权价格
        spot: 标的价格
        r: 无风险利率
        q: 分红率
        vol: 波动率

    Returns:
        float: 期权价值
    """
    pricer = QuadratureOptionPricer(grid_x, grid_t, maturity, spot, r, q, vol)
    barrier_data = {
        "ko_prices_u": (
            np.array(ko_prices_u, copy=True)
            if not np.isscalar(ko_prices_u)
            else ko_prices_u
        ),
        "ko_prices_l": (
            np.array(ko_prices_l, copy=True)
            if not np.isscalar(ko_prices_l)
            else ko_prices_l
        ),
        "ko_dates_u": ko_dates_u,
        "ko_dates_l": ko_dates_l,
        "factors": {
            "asset1": np.zeros(grid_t + 1),
            "asset2": np.zeros(grid_t + 1),
            "asset3": np.zeros(grid_t + 1),
            "cash1": np.zeros(grid_t + 1),
            "cash2": np.zeros(grid_t + 1),
            "cash3": np.zeros(grid_t + 1),
        },
    }

    if call_or_put == "call":
        payoff_asset = 1
        payoff_cash = -strike
        barrier_data["ko_prices_l"][-1] = strike
    elif call_or_put == "put":
        payoff_asset = -1
        payoff_cash = strike
        barrier_data["ko_prices_u"][-1] = strike
    else:
        raise ValueError(f"invalid call_or_put value {call_or_put} input")

    discount_factors = np.exp(-r * maturity)
    payoff_asset = payoff_asset * discount_factors
    payoff_cash = payoff_cash * discount_factors

    barrier_data["factors"]["asset2"][-1] = payoff_asset
    barrier_data["factors"]["cash2"][-1] = payoff_cash

    return pricer.price(barrier_data)


def price_kiko(
    grid_x: int,
    grid_t: int,
    maturity: float,
    ko_prices_u: np.ndarray,
    ko_dates_u: np.ndarray,
    ki_prices_l: float | np.ndarray,
    ki_dates_l: np.ndarray,
    call_or_put: str,
    strike: float,
    spot: float,
    r: float,
    q: float,
    vol: float,
) -> float:
    """Price KIKO (knock-in knock-out) option via clean wrapper.

    Computes value of a European option with payoff at maturity, subject to:
      - Knock-out at the upper barrier schedule (ko_prices_u, ko_dates_u)
      - Knock-in at the lower barrier schedule (ki_prices_l, ki_dates_l)

    For a down-and-in up-and-out structure, the payoff is realized only if
    the lower barrier is touched at some observation before maturity and the
    upper barrier is never touched before maturity. This equals:

        KIKO = UpAndOut(option) - DoubleKnockOut(option)

    where DoubleKnockOut has both the same upper and lower barriers.
    """
    # Up-and-out component
    up_out = price_single_barrier(
        grid_x,
        grid_t,
        maturity,
        ko_prices_u,
        ko_dates_u,
        call_or_put,
        strike,
        spot,
        r,
        q,
        vol,
        ko_direction="up",
    )

    # Double knock-out component
    dko = price_double_barrier(
        grid_x,
        grid_t,
        maturity,
        ko_prices_u,
        ki_prices_l,
        ko_dates_u,
        ki_dates_l,
        call_or_put,
        strike,
        spot,
        r,
        q,
        vol,
    )

    return up_out - dko


def price_euro_digital(
    grid_x: int,
    barrier: float,
    maturity: float,
    payoff: float,
    spot: float,
    r: float,
    q: float,
    vol: float,
    barrier_type: str = "up",
) -> float:
    """使用解析解公式定价欧式数字期权。

    Args:
        grid_x: 价格网格点数（为了保持接口一致，实际未使用）
        barrier: 障碍价格
        maturity: 到期时间（年化）
        payoff: 支付金额
        spot: 标的资产现价
        r: 无风险利率（年化）
        q: 股息率（年化）
        vol: 波动率（年化）
        barrier_type: 障碍类型，"up"表示向上敲入，"down"表示向下敲入

    Returns:
        float: 期权价值
    """
    if maturity <= 0:
        # 到期时根据标的价格是否超过障碍价格决定是否支付
        if barrier_type == "up":
            return payoff if spot >= barrier else 0
        else:
            return payoff if spot <= barrier else 0

    # 计算d2
    fwd = spot * np.exp((r - q) * maturity)
    d2 = (np.log(fwd / barrier) - 0.5 * vol * vol * maturity) / (
        vol * np.sqrt(maturity)
    )

    # 计算折现因子
    df = np.exp(-r * maturity)

    # 根据障碍类型计算期权价值
    if barrier_type == "up":
        return payoff * df * norm.cdf(d2)
    else:
        return payoff * df * norm.cdf(-d2)


if __name__ == "__main__":
    # 测试代码
    import time

    # 设置参数
    maturity_in_days = 241
    maturity = 241 / 244
    time_resolution = 12 * 4
    grid_resolution = 1000

    print("grid resolution:", grid_resolution, "x", time_resolution)

    # 设置敲出价格和日期
    ko_prices = [1.03] * 12
    ko_dates = np.array([22, 37, 58, 79, 96, 118, 140, 161, 182, 198, 220, 241]) / 244
    ko_cal_dates = (
        np.array([34, 63, 92, 125, 153, 184, 216, 245, 278, 307, 337, 366]) / 365
    )

    # 市场参数
    spot = 1.00
    r = 0.03
    q = 0.02
    vol = 0.2
    ntl = 1000000

    # 计时开始
    st = time.perf_counter()

    # 定价单触碰期权
    v1 = price_one_touch(
        grid_resolution,
        time_resolution,
        maturity,
        ko_prices,
        ko_dates,
        ko_cal_dates,
        0.1,
        True,
        True,
        spot,
        r,
        q,
        vol,
    )
    print(f"one touch price: {round(v1*ntl,4)}")
    print(f"time consumption: {round(time.perf_counter() - st, 2)} s")
