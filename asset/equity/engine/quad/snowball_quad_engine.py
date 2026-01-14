"""
Quadrature pricing engine for Snowball (autocallable) options.

Implements a direct two-state (knocked-in / not-knocked-in) quadrature recursion.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
from scipy.special import erfc

from asset.equity.engine.base_engine import BaseEngine
from asset.equity.engine.quad.quad_math import QuadratureMath
from asset.equity.param import QuadParams
from asset.equity.product.base_equity_product import BaseEquityProduct
from asset.equity.product.option.snowball_option import SnowballOption
from priceenv import PricingEnvironment
from util.enum import ObservationType
from util.enum.engine_enums import EngineType
from util.exceptions import PricingError, ValidationError
from util.numerical import (
    Tolerance,
    is_close,
    is_zero,
    safe_exp,
    safe_log,
    validate_non_negative,
    validate_positive,
)


class SnowballQuadEngine(BaseEngine):
    """
    Quadrature pricing engine for Snowball (autocallable) options.

    Uses a single backward quadrature recursion for two regimes:
    - V_in: knock-in already occurred
    - V_out: knock-in not yet occurred
    """

    engine_type = EngineType.QUADRATURE

    def __init__(self, params: Optional[QuadParams] = None) -> None:
        if params is None:
            params = QuadParams()
        if not isinstance(params, QuadParams):
            raise ValidationError(
                f"params must be QuadParams instance, got {type(params).__name__}"
            )
        super().__init__(params)

    def price(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        if not isinstance(product, SnowballOption):
            raise PricingError(
                f"SnowballQuadEngine only supports SnowballOption, got {type(product).__name__}"
            )
        if pricing_env is None:
            raise PricingError("PricingEnvironment is required for SnowballQuadEngine.")

        self._validate_product(product)

        spot = pricing_env.spot
        maturity = product.get_maturity(pricing_env)
        validate_positive(spot, "spot")
        validate_positive(maturity, "maturity", allow_zero=True)
        if is_zero(maturity, tol=Tolerance.ZERO):
            return product.get_payoff(spot, pricing_env)

        rate = pricing_env.get_rate(maturity)
        div = pricing_env.get_div_yield(maturity)
        vol = pricing_env.get_vol(product.strike, maturity)
        validate_positive(vol, "volatility")
        validate_non_negative(div, "dividend_yield")
        if vol > 5.0:
            raise ValidationError(f"Volatility too high for quadrature stability: {vol}")

        ko_records = product.resolve_ko_observations(pricing_env)
        if not ko_records:
            raise PricingError("KO observation schedule is empty for SnowballQuadEngine.")

        ki_continuous = product.has_ki_barrier and (
            product.barrier_config.ki_continuous
            or product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
        )
        ki_records: Sequence = []
        if product.has_ki_barrier and not ki_continuous:
            ki_records = product.resolve_ki_observations(pricing_env)
            if not ki_records:
                raise PricingError("KI observation schedule is empty for SnowballQuadEngine.")

        times = self._merge_times(
            [rec.observation_time for rec in ko_records],
            [rec.observation_time for rec in ki_records],
            maturity,
        )
        if not times:
            raise PricingError("Observation time grid is empty for SnowballQuadEngine.")

        math_utils = QuadratureMath(
            grid_x=self.params.grid_points,
            spot=spot,
            maturity=maturity,
            vol_max=vol,
        )
        grid = math_utils.grid
        spot_grid = spot * np.exp(grid)
        dt = self._build_dt(times)
        tau = 0.5 * vol * vol * dt
        if np.any(tau[1:] <= 0.0):
            raise ValidationError("time step too small for quadrature solver.")

        alpha = (rate - div - 0.5 * vol * vol) / (vol * vol)
        beta = (rate - div - 0.5 * vol * vol) ** 2 / (vol**4) + 2.0 * rate / (
            vol * vol
        )

        v_in = np.array(
            [
                product.get_maturity_payoff_v1(spot_value, pricing_env=pricing_env)
                for spot_value in spot_grid
            ],
            dtype=float,
        )
        v_out = np.array(
            [
                product.get_maturity_payoff_v0(spot_value, pricing_env=pricing_env)
                for spot_value in spot_grid
            ],
            dtype=float,
        )

        log_ki_barrier = None
        if product.has_ki_barrier:
            if ki_continuous:
                if product.barrier_config.ki_barrier is None:
                    raise PricingError("KI barrier configuration is missing.")
                if isinstance(product.barrier_config.ki_barrier, list):
                    raise PricingError("Continuous KI requires scalar ki_barrier.")
                log_ki_barrier = safe_log(product.barrier_config.ki_barrier / spot)

        full_p_lr, full_p_ur, full_p0 = 0, len(grid) - 1, (len(grid) - 1) % 2
        omega_grid = math_utils.z_grid

        for step_index in range(len(times), 0, -1):
            obs_time = times[step_index - 1]
            ko_record = self._match_record(obs_time, ko_records)
            ko_mask = None
            if ko_record is not None:
                ko_mask = (
                    spot_grid <= ko_record.barrier
                    if product.is_reverse
                    else spot_grid >= ko_record.barrier
                )
                discount = self._ko_discount(
                    rate, obs_time, ko_record.settlement_time
                )
                ko_value = ko_record.payoff * discount
                v_in[ko_mask] = ko_value
                v_out[ko_mask] = ko_value

            if ki_continuous and log_ki_barrier is not None:
                ki_mask = (
                    spot_grid >= product.barrier_config.ki_barrier
                    if product.is_reverse
                    else spot_grid <= product.barrier_config.ki_barrier
                )
                v_out[ki_mask] = v_in[ki_mask]
            elif ki_records:
                ki_record = self._match_record(obs_time, ki_records)
                if ki_record is not None:
                    ki_mask = (
                        spot_grid >= ki_record.barrier
                        if product.is_reverse
                        else spot_grid <= ki_record.barrier
                    )
                    if ko_mask is not None:
                        ki_mask = ki_mask & ~ko_mask
                    v_out[ki_mask] = v_in[ki_mask]

            tau_step = float(tau[step_index])
            prefactor = math.exp(-beta * tau_step) / math.sqrt(math.pi * tau_step) / 2.0
            omega_array = np.exp(-(omega_grid**2) / (4.0 * tau_step) - alpha * omega_grid)

            v_in = self._diffuse_fft(
                v_in,
                math_utils,
                omega_array,
                prefactor,
                full_p_lr,
                full_p_ur,
                full_p0,
                alpha,
                beta,
                tau_step,
            )

            if ki_continuous:
                v_out = self._diffuse_with_bridge(
                    v_out,
                    v_in,
                    math_utils,
                    omega_array,
                    prefactor,
                    full_p_lr,
                    full_p_ur,
                    full_p0,
                    log_ki_barrier,
                    alpha,
                    beta,
                    vol,
                    dt[step_index],
                    tau_step,
                    product.is_reverse,
                )
            else:
                v_out = self._diffuse_fft(
                    v_out,
                    math_utils,
                    omega_array,
                    prefactor,
                    full_p_lr,
                    full_p_ur,
                    full_p0,
                    alpha,
                    beta,
                    tau_step,
                )

        return math_utils.interpolate(v_out, x=0.0)

    def _validate_product(self, product: SnowballOption) -> None:
        if product.barrier_config.ko_observation_type != ObservationType.DISCRETE:
            raise PricingError("SnowballQuadEngine requires discrete KO monitoring.")
        if product.barrier_config.disable_ko_after_ki:
            raise PricingError("SnowballQuadEngine does not support disable_ko_after_ki.")
        if product.airbag_config.airbag_barrier is not None:
            raise PricingError("SnowballQuadEngine does not support airbag features.")
        if product.payoff_config.call_rebate_enabled:
            raise PricingError("SnowballQuadEngine does not support call-style rebates.")

    def _diffuse_fft(
        self,
        values: np.ndarray,
        math_utils: QuadratureMath,
        omega_array: np.ndarray,
        prefactor: float,
        p_lr: int,
        p_ur: int,
        p0: int,
        alpha: float,
        beta: float,
        tau_step: float,
    ) -> np.ndarray:
        u_array = math_utils.simpson_weights(values, p_lr, p_ur, p0)
        conv = math_utils.convolution_fft(omega_array, u_array)
        base = prefactor * conv
        return base + self._tail_correction(values, math_utils, alpha, beta, tau_step)

    def _diffuse_with_bridge(
        self,
        v_out: np.ndarray,
        v_in: np.ndarray,
        math_utils: QuadratureMath,
        omega_array: np.ndarray,
        prefactor: float,
        p_lr: int,
        p_ur: int,
        p0: int,
        log_barrier: float,
        alpha: float,
        beta: float,
        vol: float,
        dt: float,
        tau_step: float,
        is_reverse: bool,
    ) -> np.ndarray:
        grid = math_utils.grid
        weights = math_utils.simpson_weight_vector()
        base = self._diffuse_fft(
            v_out,
            math_utils,
            omega_array,
            prefactor,
            p_lr,
            p_ur,
            p0,
            alpha,
            beta,
            tau_step,
        )

        delta = v_in - v_out
        correction = np.zeros_like(base)
        band = self._bridge_band(tau_step, math_utils.h, grid.size)
        denom = vol * vol * dt

        for i in range(grid.size):
            j0 = max(0, i - band)
            j1 = min(grid.size, i + band + 1)
            y = grid[j0:j1]
            z = grid[i] - y
            omega = np.exp(-(z**2) / (4.0 * tau_step) - alpha * z)
            w = weights[j0:j1]

            if is_reverse:
                d0 = log_barrier - grid[i]
                d1 = log_barrier - y
            else:
                d0 = grid[i] - log_barrier
                d1 = y - log_barrier

            safe = (d0 > 0.0) & (d1 > 0.0)
            exponent = np.where(safe, -2.0 * d0 * d1 / denom, 0.0)
            exponent = np.clip(exponent, -745.0, 0.0)
            p_hit = np.where(safe, np.exp(exponent), 1.0)
            correction[i] = prefactor * np.sum(omega * w * p_hit * delta[j0:j1])

        return base + correction

    def _tail_correction(
        self,
        values: np.ndarray,
        math_utils: QuadratureMath,
        alpha: float,
        beta: float,
        tau_step: float,
    ) -> np.ndarray:
        grid = math_utils.grid
        x_min = grid[0]
        x_max = grid[-1]
        sqrt_tau = math.sqrt(tau_step)
        u_left = (grid - x_min + 2.0 * tau_step * alpha) / (2.0 * sqrt_tau)
        u_right = (grid - x_max + 2.0 * tau_step * alpha) / (2.0 * sqrt_tau)
        tail_scale = 0.5 * math.exp(tau_step * (alpha * alpha - beta))
        return (
            values[0] * tail_scale * erfc(u_left)
            + values[-1] * tail_scale * erfc(-u_right)
        )

    def _bridge_band(self, tau_step: float, h: float, n: int) -> int:
        std = math.sqrt(2.0 * tau_step)
        band = int(math.ceil(self.params.num_std_devs * std / h))
        return max(1, min(band, n - 1))

    def _ko_discount(
        self,
        rate: float,
        obs_time: float,
        settlement_time: Optional[float],
    ) -> float:
        if settlement_time is None:
            return 1.0
        delay = max(settlement_time - obs_time, 0.0)
        if is_zero(delay, tol=Tolerance.ZERO):
            return 1.0
        return safe_exp(-rate * delay)

    def _merge_times(
        self, ko_times: Sequence[float], ki_times: Sequence[float], maturity: float
    ) -> list[float]:
        merged = []
        for t in sorted(list(ko_times) + list(ki_times)):
            if not merged or not is_close(t, merged[-1], abs_tol=Tolerance.PRECISION):
                merged.append(t)
        if not merged or not is_close(merged[-1], maturity, abs_tol=Tolerance.PRECISION):
            merged.append(maturity)
        return merged

    def _build_dt(self, times: Sequence[float]) -> np.ndarray:
        times_full = np.concatenate(([0.0], np.asarray(times, dtype=float)))
        dt = np.diff(times_full)
        if np.any(dt <= Tolerance.ZERO):
            raise ValidationError("observation_times must be strictly increasing.")
        return np.concatenate(([0.0], dt))

    def _match_record(self, target_time: float, records: Sequence) -> Optional[object]:
        for rec in records:
            if is_close(target_time, rec.observation_time, abs_tol=Tolerance.PRECISION):
                return rec
        return None

    def __repr__(self) -> str:
        return "SnowballQuadEngine()"
