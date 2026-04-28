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
from asset.equity.engine.event_stats import AutocallableEventStats
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
        if not np.isfinite(div):
            raise ValidationError(f"Dividend yield must be finite, got {div}")
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

        align_log = self._select_alignment_log(spot, product)
        fft_padding_factor = self._resolve_fft_padding_factor()
        fft_filter_alpha, fft_filter_power = self._resolve_fft_filter()
        math_utils = QuadratureMath(
            grid_x=self.params.grid_points,
            spot=spot,
            maturity=maturity,
            vol_max=vol,
            num_std_devs=self.params.num_std_devs,
            align_log=align_log,
            fft_padding_factor=fft_padding_factor,
            fft_filter_alpha=fft_filter_alpha,
            fft_filter_power=fft_filter_power,
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

        disable_ko_after_ki = product.barrier_config.disable_ko_after_ki

        smoothing_width = self._resolve_event_smoothing_width(math_utils, product)

        for step_index in range(len(times), 0, -1):
            obs_time = times[step_index - 1]
            ko_record = self._match_record(obs_time, ko_records)
            ko_weight = None
            if ko_record is not None:
                discount = self._ko_discount(
                    rate, obs_time, ko_record.settlement_time
                )
                ko_value = ko_record.payoff * discount
                ko_weight = self._smooth_step_weight(
                    grid,
                    ko_record.barrier,
                    spot,
                    smoothing_width,
                    trigger_is_down=product.is_reverse,
                )
                if ko_weight is None:
                    ko_mask = (
                        spot_grid <= ko_record.barrier
                        if product.is_reverse
                        else spot_grid >= ko_record.barrier
                    )
                    ko_weight = ko_mask.astype(float)

                # KO always applies to the not-yet-KI surface; KI surface only if enabled.
                v_out = ko_weight * ko_value + (1.0 - ko_weight) * v_out
                if not disable_ko_after_ki:
                    v_in = ko_weight * ko_value + (1.0 - ko_weight) * v_in

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
                    ki_weight = self._smooth_step_weight(
                        grid,
                        ki_record.barrier,
                        spot,
                        smoothing_width,
                        trigger_is_down=not product.is_reverse,
                    )
                    if ki_weight is None:
                        ki_mask = (
                            spot_grid >= ki_record.barrier
                            if product.is_reverse
                            else spot_grid <= ki_record.barrier
                        )
                        ki_weight = ki_mask.astype(float)
                    if ko_weight is not None and not disable_ko_after_ki:
                        ki_weight = ki_weight * (1.0 - ko_weight)
                    v_out = (1.0 - ki_weight) * v_out + ki_weight * v_in

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

        value_surface = (
            v_in if getattr(product, "_otc_lifecycle_knocked_in", False) else v_out
        )
        return math_utils.interpolate(value_surface, x=0.0)

    def calculate_event_stats(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> Optional[AutocallableEventStats]:
        """
        Provide per-observation KO probabilities and expected discounted cashflows.

        This implementation runs a single quadrature recursion for all KO observations
        by propagating stacked indicator surfaces. It is typically much faster than MC
        for risk-neutral event stats.
        """
        if not isinstance(product, SnowballOption):
            return None
        if pricing_env is None:
            raise PricingError("PricingEnvironment is required for SnowballQuadEngine.")

        self._validate_product(product)

        spot = pricing_env.spot
        maturity = product.get_maturity(pricing_env)
        validate_positive(spot, "spot")
        validate_positive(maturity, "maturity", allow_zero=True)
        if is_zero(maturity, tol=Tolerance.ZERO):
            return None

        rate = pricing_env.get_rate(maturity)
        div = pricing_env.get_div_yield(maturity)
        vol = pricing_env.get_vol(product.strike, maturity)
        validate_positive(vol, "volatility")
        if not np.isfinite(div):
            raise ValidationError(f"Dividend yield must be finite, got {div}")
        if vol > 5.0:
            raise ValidationError(f"Volatility too high for quadrature stability: {vol}")

        ko_records = product.resolve_ko_observations(pricing_env)
        if not ko_records:
            return None

        ki_continuous = product.has_ki_barrier and (
            product.barrier_config.ki_continuous
            or product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
        )
        ki_records = []
        if product.has_ki_barrier and not ki_continuous:
            ki_records = product.resolve_ki_observations(pricing_env)

        times = self._merge_times(
            [rec.observation_time for rec in ko_records],
            [rec.observation_time for rec in ki_records],
            maturity,
        )
        dt = self._build_dt(times)

        align_log = self._select_alignment_log(spot, product)
        fft_padding_factor = self._resolve_fft_padding_factor()
        fft_filter_alpha, fft_filter_power = self._resolve_fft_filter()
        math_utils = QuadratureMath(
            grid_x=self.params.grid_points,
            spot=spot,
            maturity=maturity,
            vol_max=vol,
            num_std_devs=self.params.num_std_devs,
            align_log=align_log,
            fft_padding_factor=fft_padding_factor,
            fft_filter_alpha=fft_filter_alpha,
            fft_filter_power=fft_filter_power,
        )
        grid = math_utils.grid
        spot_grid = spot * np.exp(grid)

        tau = 0.5 * vol * vol * dt
        alpha = (rate - div - 0.5 * vol * vol) / (vol * vol)
        beta = (rate - div - 0.5 * vol * vol) ** 2 / (vol**4) + 2.0 * rate / (
            vol * vol
        )
        full_p_lr, full_p_ur, full_p0 = 0, len(grid) - 1, (len(grid) - 1) % 2
        omega_grid = math_utils.z_grid

        disable_ko_after_ki = product.barrier_config.disable_ko_after_ki

        # --- KO indicator recursion (stacked over KO observations) ---
        n_ko = len(ko_records)
        v_in = np.zeros((n_ko, grid.size), dtype=float)
        v_out = np.zeros((n_ko, grid.size), dtype=float)

        for step_index in range(len(times), 0, -1):
            obs_time = times[step_index - 1]

            ko_mask = None
            ko_index = None
            ko_record = None
            for idx, rec in enumerate(ko_records):
                if is_close(obs_time, rec.observation_time, abs_tol=Tolerance.PRECISION):
                    ko_index = idx
                    ko_record = rec
                    break

            if ko_record is not None:
                ko_mask = (
                    spot_grid <= ko_record.barrier
                    if product.is_reverse
                    else spot_grid >= ko_record.barrier
                )
                discount_delay = self._ko_discount(
                    rate, obs_time, ko_record.settlement_time
                )
                # KO always applies to not-yet-KI surface; KI surface only if enabled.
                v_out[:, ko_mask] = 0.0
                v_out[int(ko_index), ko_mask] = float(discount_delay)
                if not disable_ko_after_ki:
                    v_in[:, ko_mask] = 0.0
                    v_in[int(ko_index), ko_mask] = float(discount_delay)

            if ki_continuous:
                # KI transition handled via Brownian bridge in diffusion step.
                pass
            elif ki_records:
                ki_record = self._match_record(obs_time, ki_records)
                if ki_record is not None:
                    ki_mask = (
                        spot_grid >= ki_record.barrier
                        if product.is_reverse
                        else spot_grid <= ki_record.barrier
                    )
                    if ko_mask is not None and not disable_ko_after_ki:
                        ki_mask = ki_mask & ~ko_mask
                    v_out[:, ki_mask] = v_in[:, ki_mask]

            tau_step = float(tau[step_index])
            prefactor = math.exp(-beta * tau_step) / math.sqrt(math.pi * tau_step) / 2.0
            omega_array = np.exp(
                -(omega_grid**2) / (4.0 * tau_step) - alpha * omega_grid
            )

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
                if product.barrier_config.ki_barrier is None:
                    raise PricingError("KI barrier configuration is missing.")
                if isinstance(product.barrier_config.ki_barrier, list):
                    raise PricingError("Continuous KI requires scalar ki_barrier.")
                log_ki_barrier = safe_log(product.barrier_config.ki_barrier / spot)
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

        ed_unit = np.array(
            [math_utils.interpolate(v_out[i], x=0.0) for i in range(n_ko)], dtype=float
        )
        ko_times = np.array([rec.observation_time for rec in ko_records], dtype=float)
        ko_prob = np.zeros(n_ko, dtype=float)
        ed_ko_cf = np.zeros(n_ko, dtype=float)
        survival_prob = np.ones(n_ko, dtype=float)

        cumulative = 0.0
        for i, rec in enumerate(ko_records):
            df_total = math.exp(-rate * float(rec.observation_time)) * float(
                self._ko_discount(rate, float(rec.observation_time), rec.settlement_time)
            )
            if df_total > 0:
                ko_prob[i] = float(ed_unit[i] / df_total)
            payoff = float(rec.payoff) if rec.payoff is not None else 0.0
            ed_ko_cf[i] = float(ed_unit[i] * payoff)
            cumulative += ko_prob[i]
            survival_prob[i] = max(0.0, 1.0 - cumulative)

        pv = float(self.price(product, pricing_env))
        expected_discounted_maturity_cf = float(pv - float(np.sum(ed_ko_cf)))

        # KI probability (no-KO): propagate terminal indicator on KI surface with KO absorbing to 0.
        ki_probability = 0.0
        if product.has_ki_barrier:
            v_in_ki = np.ones(grid.size, dtype=float)
            v_out_ki = np.zeros(grid.size, dtype=float)
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
                    v_out_ki[ko_mask] = 0.0
                    if not disable_ko_after_ki:
                        v_in_ki[ko_mask] = 0.0

                if not ki_continuous and ki_records:
                    ki_record = self._match_record(obs_time, ki_records)
                    if ki_record is not None:
                        ki_mask = (
                            spot_grid >= ki_record.barrier
                            if product.is_reverse
                            else spot_grid <= ki_record.barrier
                        )
                        if ko_mask is not None and not disable_ko_after_ki:
                            ki_mask = ki_mask & ~ko_mask
                        v_out_ki[ki_mask] = v_in_ki[ki_mask]

                tau_step = float(tau[step_index])
                prefactor = (
                    math.exp(-beta * tau_step) / math.sqrt(math.pi * tau_step) / 2.0
                )
                omega_array = np.exp(
                    -(omega_grid**2) / (4.0 * tau_step) - alpha * omega_grid
                )
                v_in_ki = self._diffuse_fft(
                    v_in_ki,
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
                    if product.barrier_config.ki_barrier is None:
                        raise PricingError("KI barrier configuration is missing.")
                    if isinstance(product.barrier_config.ki_barrier, list):
                        raise PricingError("Continuous KI requires scalar ki_barrier.")
                    log_ki_barrier = safe_log(product.barrier_config.ki_barrier / spot)
                    v_out_ki = self._diffuse_with_bridge(
                        v_out_ki,
                        v_in_ki,
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
                    v_out_ki = self._diffuse_fft(
                        v_out_ki,
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

            df_T = math.exp(-rate * maturity)
            pv_ki_no_ko = float(math_utils.interpolate(v_out_ki, x=0.0))
            if df_T > 0:
                ki_probability = float(pv_ki_no_ko / df_T)

        return AutocallableEventStats(
            pv=pv,
            ko_times=ko_times,
            ko_probability=ko_prob,
            survival_probability=survival_prob,
            expected_discounted_ko_cashflow=ed_ko_cf,
            ki_probability=ki_probability,
            expected_discounted_maturity_cashflow=expected_discounted_maturity_cf,
            reconciliation_error=0.0,
        )

    def _validate_product(self, product: SnowballOption) -> None:
        if product.barrier_config.ko_observation_type != ObservationType.DISCRETE:
            raise PricingError("SnowballQuadEngine requires discrete KO monitoring.")
        # Airbag and call-rebate features are handled via product payoff functions.

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
        if values.ndim == 1:
            u_array = math_utils.simpson_weights(values, p_lr, p_ur, p0)
            conv = math_utils.convolution_fft(omega_array, u_array)
            base = prefactor * conv
            return base + self._tail_correction(values, math_utils, alpha, beta, tau_step)

        out = np.zeros_like(values, dtype=float)
        for i in range(values.shape[0]):
            out[i] = self._diffuse_fft(
                values[i],
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
        return out

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
        correction = np.zeros_like(base, dtype=float)
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
            kernel = omega * w * p_hit
            if delta.ndim == 1:
                correction[i] = prefactor * float(np.sum(kernel * delta[j0:j1]))
            else:
                correction[:, i] = prefactor * (delta[:, j0:j1] @ kernel)

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
        if values.ndim == 1:
            return (
                values[0] * tail_scale * erfc(u_left)
                + values[-1] * tail_scale * erfc(-u_right)
            )

        left = values[:, 0].reshape(-1, 1)
        right = values[:, -1].reshape(-1, 1)
        return left * tail_scale * erfc(u_left) + right * tail_scale * erfc(-u_right)

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

    def _resolve_fft_padding_factor(self) -> int:
        factor = getattr(self.params, "fft_padding_factor", None)
        if factor is None or int(factor) <= 0:
            return 2
        return int(factor)

    def _resolve_fft_filter(self) -> tuple[float, int]:
        alpha = getattr(self.params, "fft_filter_alpha", None)
        power = getattr(self.params, "fft_filter_power", None)

        if alpha is None:
            alpha = 12.0
        if power is None:
            power = 8

        return float(alpha), int(power)

    def _resolve_align_priority(self) -> str:
        priority = getattr(self.params, "align_priority", None)
        if priority is None:
            return "auto"
        return str(priority).lower()

    def _select_alignment_log(
        self, spot: float, product: SnowballOption
    ) -> Optional[float]:
        ko_candidates: list[float] = []
        ki_candidates: list[float] = []

        ko_barrier = product.barrier_config.ko_barrier
        if isinstance(ko_barrier, list):
            ko_candidates.extend([float(b) for b in ko_barrier if b > 0])
        elif ko_barrier is not None and ko_barrier > 0:
            ko_candidates.append(float(ko_barrier))

        if product.has_ki_barrier:
            ki_barrier = product.barrier_config.ki_barrier
            if isinstance(ki_barrier, list):
                ki_candidates.extend([float(b) for b in ki_barrier if b > 0])
            elif ki_barrier is not None and ki_barrier > 0:
                ki_candidates.append(float(ki_barrier))

        def to_logs(candidates: list[float]) -> list[float]:
            logs = []
            for b in candidates:
                try:
                    logs.append(safe_log(b / spot))
                except Exception:
                    continue
            return logs

        ko_logs = to_logs(ko_candidates)
        ki_logs = to_logs(ki_candidates)

        if not ko_logs and not ki_logs:
            return None

        def closest(logs: list[float]) -> Optional[float]:
            if not logs:
                return None
            idx = int(np.argmin(np.abs(np.asarray(logs))))
            return float(logs[idx])

        priority = self._resolve_align_priority()
        if priority == "ko":
            return closest(ko_logs) or closest(ki_logs)
        if priority == "ki":
            return closest(ki_logs) or closest(ko_logs)
        if priority == "coupon":
            return closest(ko_logs) or closest(ki_logs)

        # auto: prefer KO near spot for reverse, else closest overall
        if product.is_reverse:
            ko_near = closest(ko_logs)
            if ko_near is not None and abs(ko_near) <= 0.05:
                return ko_near
            return closest(ko_logs + ki_logs)

        return closest(ko_logs + ki_logs)

    def _resolve_event_smoothing_width(
        self, math_utils: QuadratureMath, product: SnowballOption
    ) -> float:
        mode = getattr(self.params, "event_smoothing_mode", "fixed")
        cells = getattr(self.params, "event_smoothing_cells", 0)
        kernel_width = getattr(self.params, "event_smoothing_log_width", 0.002)

        try:
            cells = int(cells)
        except (TypeError, ValueError):
            cells = 0

        if str(mode).lower() == "reverse_aware" and product.is_reverse:
            cells = 0
        elif str(mode).lower() == "auto":
            h = float(math_utils.h)
            cells = max(1, int(0.5 + float(kernel_width) / h))

        if cells <= 0:
            return 0.0
        return float(cells) * float(math_utils.h)

    def _smooth_step_weight(
        self,
        grid: np.ndarray,
        barrier: float,
        spot: float,
        width: float,
        *,
        trigger_is_down: bool,
    ) -> Optional[np.ndarray]:
        if width <= 0.0:
            return None
        if barrier is None or barrier <= 0.0 or spot <= 0.0:
            return None
        barrier_log = safe_log(barrier / spot)
        return self._smooth_step_weight_log(
            grid, barrier_log, width, trigger_is_down=trigger_is_down
        )

    def _smooth_step_weight_log(
        self,
        grid: np.ndarray,
        barrier_log: float,
        width: float,
        *,
        trigger_is_down: bool,
    ) -> Optional[np.ndarray]:
        if width <= 0.0:
            return None
        x = grid - float(barrier_log)
        kernel = str(getattr(self.params, "event_smoothing_kernel", "cosine")).lower()
        if kernel == "tanh":
            base = 0.5 * (1.0 + np.tanh(x / width))
        else:
            t = np.clip((x + width) / (2.0 * width), 0.0, 1.0)
            base = 0.5 - 0.5 * np.cos(np.pi * t)
        if trigger_is_down:
            return 1.0 - base
        return base

    def __repr__(self) -> str:
        return "SnowballQuadEngine()"
