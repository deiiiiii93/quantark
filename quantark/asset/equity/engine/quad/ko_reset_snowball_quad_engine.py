"""
Quadrature pricing engine for KO-reset Snowball options.

Uses two value surfaces (V_out/V_in) with separate KO schedules:
- V_out uses pre-KI KO observations
- V_in uses post-KI KO observations
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np

from quantark.asset.equity.engine.event_stats import KOResetEventStats
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.engine.quad.quad_math import QuadratureMath
from quantark.asset.equity.param import QuadParams
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option.ko_reset_snowball_option import (
    KnockOutResetSnowballOption,
)
from quantark.asset.equity.product.option.observation_schedule import ResolvedObservationRecord
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import CouponPayType, ObservationType, PostKOScheduleMode
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import PricingError, ValidationError
from quantark.util.numerical import (
    Tolerance,
    is_close,
    is_zero,
    safe_log,
    validate_non_negative,
    validate_positive,
)


class KOResetSnowballQuadEngine(SnowballQuadEngine):
    """
    Quadrature pricing engine for KO-reset Snowball options.
    """

    engine_type = EngineType.QUADRATURE

    def __init__(self, params: Optional[QuadParams] = None) -> None:
        super().__init__(params=params)

    def price(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        if not isinstance(product, KnockOutResetSnowballOption):
            raise PricingError(
                "KOResetSnowballQuadEngine only supports KnockOutResetSnowballOption, "
                f"got {type(product).__name__}"
            )
        if pricing_env is None:
            raise PricingError("PricingEnvironment is required for KOResetSnowballQuadEngine.")

        self._validate_product(product)

        # Opt-in backward-grid capture for the CVA exposure layer (see SnowballQuadEngine).
        # Reset BEFORE any early return so a terminated bumped re-price cannot leave a
        # previous trade's surfaces behind for the exposure layer to read as if live.
        record_grids = getattr(self, "record_backward_grids", False)
        if record_grids:
            self._backward_grids = {}

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
        if vol > 5.0:
            raise ValidationError(f"Volatility too high for quadrature stability: {vol}")

        pre_ko_records = self._resolve_ko_records(
            product, pricing_env, product.barrier_config
        )
        pre_ko_records = [
            rec
            for rec in pre_ko_records
            if rec.observation_time <= maturity
            or is_close(rec.observation_time, maturity, abs_tol=Tolerance.PRECISION)
        ]
        if not pre_ko_records:
            raise PricingError("Pre-KO observation schedule is empty for KOResetSnowballQuadEngine.")

        post_ko_records = self._resolve_ko_records(
            product, pricing_env, product.post_barrier_config
        )
        post_ko_records = [
            rec
            for rec in post_ko_records
            if rec.observation_time <= maturity
            or is_close(rec.observation_time, maturity, abs_tol=Tolerance.PRECISION)
        ]
        if not post_ko_records:
            raise PricingError("Post-KO observation schedule is empty for KOResetSnowballQuadEngine.")

        ki_continuous = product.has_ki_barrier and (
            product.barrier_config.ki_continuous
            or product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
        )
        ki_records: Sequence = []
        if product.has_ki_barrier and not ki_continuous:
            ki_records = product.resolve_ki_observations(pricing_env)
            if not ki_records:
                raise PricingError("KI observation schedule is empty for KOResetSnowballQuadEngine.")

        times = self._merge_times(
            [rec.observation_time for rec in pre_ko_records]
            + [rec.observation_time for rec in post_ko_records],
            [rec.observation_time for rec in ki_records],
            maturity,
        )
        if not times:
            raise PricingError("Observation time grid is empty for KOResetSnowballQuadEngine.")

        # When recording for CVA exposure, diffuse through (and record at) delayed KO
        # settlement times from BOTH the pre-KI and post-KI schedules (no-event nodes).
        if record_grids:
            times = self._insert_settlement_times(
                times, list(pre_ko_records) + list(post_ko_records), maturity)

        align_log = self._select_alignment_log(spot, product)
        fft_padding_factor = self._resolve_fft_padding_factor()
        fft_filter_alpha, fft_filter_power = self._resolve_fft_filter()
        grid_points = self._resolve_grid_points(
            maturity, vol, [rec.observation_time for rec in ki_records]
        )
        math_utils = QuadratureMath(
            grid_x=grid_points,
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

            pre_ko_weight = None
            pre_ko_record = self._match_record(obs_time, pre_ko_records)
            if pre_ko_record is not None:
                discount = self._ko_discount(
                    rate, obs_time, pre_ko_record.settlement_time
                )
                ko_value = pre_ko_record.payoff * discount
                ko_weight = self._smooth_step_weight(
                    grid,
                    pre_ko_record.barrier,
                    spot,
                    smoothing_width,
                    trigger_is_down=product.is_reverse,
                )
                if ko_weight is None:
                    ko_mask = (
                        spot_grid <= pre_ko_record.barrier
                        if product.is_reverse
                        else spot_grid >= pre_ko_record.barrier
                    )
                    ko_weight = ko_mask.astype(float)

                pre_ko_weight = ko_weight
                v_out = ko_weight * ko_value + (1.0 - ko_weight) * v_out

            post_ko_record = self._match_record(obs_time, post_ko_records)
            if post_ko_record is not None and not disable_ko_after_ki:
                discount = self._ko_discount(
                    rate, obs_time, post_ko_record.settlement_time
                )
                ko_value = post_ko_record.payoff * discount
                ko_weight = self._smooth_step_weight(
                    grid,
                    post_ko_record.barrier,
                    spot,
                    smoothing_width,
                    trigger_is_down=product.is_reverse,
                )
                if ko_weight is None:
                    ko_mask = (
                        spot_grid <= post_ko_record.barrier
                        if product.is_reverse
                        else spot_grid >= post_ko_record.barrier
                    )
                    ko_weight = ko_mask.astype(float)

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
                    # Resolution-aware smoothed KI transition (same O(h)
                    # bias fix as SnowballQuadEngine), narrowed by the
                    # pre-KO weight on simultaneous observations.
                    v_out = self._blend_ki_transition(
                        v_out,
                        v_in,
                        grid,
                        spot_grid,
                        ki_record.barrier,
                        spot,
                        smoothing_width,
                        product.is_reverse,
                        ko_weight=pre_ko_weight,
                    )

            # Post-event continuation surfaces at obs_time (v_out = pre-KI/not-yet-KI,
            # v_in = post-KI/knocked-in), before diffusing back to the previous step.
            if record_grids:
                self._backward_grids[float(obs_time)] = (
                    spot_grid.copy(), v_in.copy(), v_out.copy())

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

            if ki_continuous and log_ki_barrier is not None:
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

        if record_grids:
            self._backward_grids[0.0] = (spot_grid.copy(), v_in.copy(), v_out.copy())

        self._last_spot_greeks_grid = (spot_grid.copy(), v_out.copy())
        return math_utils.interpolate(v_out, x=0.0)

    def calculate_event_stats(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> Optional[KOResetEventStats]:
        if not isinstance(product, KnockOutResetSnowballOption):
            return None
        if pricing_env is None:
            raise PricingError("PricingEnvironment is required for KOResetSnowballQuadEngine.")
        return self._compute_event_stats(product, pricing_env)

    def _compute_event_stats(
        self,
        product: KnockOutResetSnowballOption,
        pricing_env: PricingEnvironment,
    ) -> Optional[KOResetEventStats]:
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
        if vol > 5.0:
            raise ValidationError(f"Volatility too high for quadrature stability: {vol}")

        pre_ko_records = self._resolve_ko_records(
            product, pricing_env, product.barrier_config
        )
        pre_ko_records = [
            rec
            for rec in pre_ko_records
            if rec.observation_time <= maturity
            or is_close(rec.observation_time, maturity, abs_tol=Tolerance.PRECISION)
        ]
        if not pre_ko_records:
            return None

        post_ko_records = self._resolve_ko_records(
            product, pricing_env, product.post_barrier_config
        )
        post_ko_records = [
            rec
            for rec in post_ko_records
            if rec.observation_time <= maturity
            or is_close(rec.observation_time, maturity, abs_tol=Tolerance.PRECISION)
        ]
        if not post_ko_records:
            return None

        ki_continuous = product.has_ki_barrier and (
            product.barrier_config.ki_continuous
            or product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
        )
        ki_records: Sequence = []
        if product.has_ki_barrier and not ki_continuous:
            ki_records = product.resolve_ki_observations(pricing_env)
            if not ki_records:
                raise PricingError("KI observation schedule is empty for KOResetSnowballQuadEngine.")

        times = self._merge_times(
            [rec.observation_time for rec in pre_ko_records]
            + [rec.observation_time for rec in post_ko_records],
            [rec.observation_time for rec in ki_records],
            maturity,
        )
        if not times:
            return None

        align_log = self._select_alignment_log(spot, product)
        fft_padding_factor = self._resolve_fft_padding_factor()
        fft_filter_alpha, fft_filter_power = self._resolve_fft_filter()
        grid_points = self._resolve_grid_points(
            maturity, vol, [rec.observation_time for rec in ki_records]
        )
        math_utils = QuadratureMath(
            grid_x=grid_points,
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

        n_pre = len(pre_ko_records)
        n_post = len(post_ko_records)
        post_offset = n_pre
        ki_col = n_pre + n_post
        ki_ever_col = ki_col + 1
        n_rows = ki_ever_col + 1

        v_in = np.zeros((n_rows, grid.size), dtype=float)
        v_out = np.zeros((n_rows, grid.size), dtype=float)
        v_in[ki_col] = 1.0
        v_in[ki_ever_col] = 1.0

        log_ki_barrier = None
        if product.has_ki_barrier and ki_continuous:
            if product.barrier_config.ki_barrier is None:
                raise PricingError("KI barrier configuration is missing.")
            if isinstance(product.barrier_config.ki_barrier, list):
                raise PricingError("Continuous KI requires scalar ki_barrier.")
            log_ki_barrier = safe_log(product.barrier_config.ki_barrier / spot)

        knocked_in_at_valuation = self._is_knocked_in_at_valuation(
            product,
            spot,
            pricing_env,
            ki_continuous=ki_continuous,
            ki_records=ki_records,
        )
        full_p_lr, full_p_ur, full_p0 = 0, len(grid) - 1, (len(grid) - 1) % 2
        omega_grid = math_utils.z_grid
        disable_ko_after_ki = product.barrier_config.disable_ko_after_ki
        smoothing_width = self._resolve_event_smoothing_width(math_utils, product)

        for step_index in range(len(times), 0, -1):
            obs_time = times[step_index - 1]

            pre_ko_weight = None
            pre_ko_record = self._match_record(obs_time, pre_ko_records)
            if pre_ko_record is not None:
                pre_idx = pre_ko_records.index(pre_ko_record)
                ko_weight = self._smooth_step_weight(
                    grid,
                    pre_ko_record.barrier,
                    spot,
                    smoothing_width,
                    trigger_is_down=product.is_reverse,
                )
                if ko_weight is None:
                    ko_mask = (
                        spot_grid <= pre_ko_record.barrier
                        if product.is_reverse
                        else spot_grid >= pre_ko_record.barrier
                    )
                    ko_weight = ko_mask.astype(float)

                pre_ko_weight = ko_weight
                ever_before = v_out[ki_ever_col].copy()
                v_out *= 1.0 - ko_weight
                v_out[pre_idx] += ko_weight * float(
                    self._ko_discount(rate, obs_time, pre_ko_record.settlement_time)
                )
                v_out[ki_ever_col] = ever_before

            post_ko_record = self._match_record(obs_time, post_ko_records)
            if post_ko_record is not None and not disable_ko_after_ki:
                post_idx = post_ko_records.index(post_ko_record)
                ko_weight = self._smooth_step_weight(
                    grid,
                    post_ko_record.barrier,
                    spot,
                    smoothing_width,
                    trigger_is_down=product.is_reverse,
                )
                if ko_weight is None:
                    ko_mask = (
                        spot_grid <= post_ko_record.barrier
                        if product.is_reverse
                        else spot_grid >= post_ko_record.barrier
                    )
                    ko_weight = ko_mask.astype(float)

                ever_before = v_in[ki_ever_col].copy()
                v_in *= 1.0 - ko_weight
                v_in[post_offset + post_idx] += ko_weight * float(
                    self._ko_discount(rate, obs_time, post_ko_record.settlement_time)
                )
                v_in[ki_ever_col] = ever_before

            if ki_continuous and log_ki_barrier is not None:
                ki_mask = (
                    spot_grid >= product.barrier_config.ki_barrier
                    if product.is_reverse
                    else spot_grid <= product.barrier_config.ki_barrier
                )
                v_out[:, ki_mask] = v_in[:, ki_mask]
            elif ki_records:
                ki_record = self._match_record(obs_time, ki_records)
                if ki_record is not None:
                    # Smoothed discrete-KI transition, consistent with the
                    # smoothed KO absorption above and with the price() path;
                    # narrowed by the pre-KO weight so KO keeps precedence at
                    # a simultaneous observation (reduces to the hard mask at
                    # width 0, where KO and KI regions are disjoint).
                    v_out = self._blend_ki_transition(
                        v_out,
                        v_in,
                        grid,
                        spot_grid,
                        ki_record.barrier,
                        spot,
                        smoothing_width,
                        product.is_reverse,
                        ko_weight=pre_ko_weight,
                    )

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

            if ki_continuous and log_ki_barrier is not None:
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

        initial_surface = v_in if knocked_in_at_valuation else v_out
        pre_unit = np.array(
            [math_utils.interpolate(initial_surface[i], x=0.0) for i in range(n_pre)],
            dtype=float,
        )
        post_unit = np.array(
            [
                math_utils.interpolate(initial_surface[post_offset + i], x=0.0)
                for i in range(n_post)
            ],
            dtype=float,
        )

        pre_times = np.array([rec.observation_time for rec in pre_ko_records], dtype=float)
        post_times = np.array([rec.observation_time for rec in post_ko_records], dtype=float)
        pre_probability = np.zeros(n_pre, dtype=float)
        post_probability = np.zeros(n_post, dtype=float)
        expected_discounted_pre_ko_cashflow = np.zeros(n_pre, dtype=float)
        expected_discounted_post_ko_cashflow = 0.0

        for i, rec in enumerate(pre_ko_records):
            df_total = math.exp(-rate * float(rec.observation_time)) * float(
                self._ko_discount(rate, float(rec.observation_time), rec.settlement_time)
            )
            if df_total > 0:
                pre_probability[i] = float(pre_unit[i] / df_total)
            payoff = float(rec.payoff) if rec.payoff is not None else 0.0
            expected_discounted_pre_ko_cashflow[i] = float(pre_unit[i] * payoff)

        for i, rec in enumerate(post_ko_records):
            df_total = math.exp(-rate * float(rec.observation_time)) * float(
                self._ko_discount(rate, float(rec.observation_time), rec.settlement_time)
            )
            if df_total > 0:
                post_probability[i] = float(post_unit[i] / df_total)
            payoff = float(rec.payoff) if rec.payoff is not None else 0.0
            expected_discounted_post_ko_cashflow += float(post_unit[i] * payoff)

        df_maturity = math.exp(-rate * maturity)
        ki_survive = 0.0
        ki_ever = 0.0
        if df_maturity > 0.0:
            ki_survive = float(
                math_utils.interpolate(initial_surface[ki_col], x=0.0) / df_maturity
            )
            ki_ever = float(
                math_utils.interpolate(initial_surface[ki_ever_col], x=0.0)
                / df_maturity
            )

        survival_probability = np.ones(n_pre, dtype=float)
        cumulative_pre_ko = 0.0
        for i in range(n_pre):
            cumulative_pre_ko += pre_probability[i]
            survival_probability[i] = max(0.0, 1.0 - cumulative_pre_ko)

        pv = float(self.price(product, pricing_env))
        expected_discounted_maturity_cashflow = float(
            pv
            - float(np.sum(expected_discounted_pre_ko_cashflow))
            - expected_discounted_post_ko_cashflow
        )

        return KOResetEventStats(
            pv=pv,
            ko_times=pre_times,
            ko_probability=pre_probability,
            survival_probability=survival_probability,
            expected_discounted_ko_cashflow=expected_discounted_pre_ko_cashflow,
            ki_probability=ki_ever,
            expected_discounted_maturity_cashflow=expected_discounted_maturity_cashflow,
            reconciliation_error=0.0,
            ki_ever_probability=ki_ever,
            ki_survive_knocked_in_probability=ki_survive,
            pre_ko_times=pre_times,
            pre_ko_probability=pre_probability,
            post_ko_times=post_times,
            post_ko_probability=post_probability,
            pre_ko_probability_total=float(np.sum(pre_probability)),
            post_ko_probability_total=float(np.sum(post_probability)),
            expected_discounted_post_ko_cashflow=float(
                expected_discounted_post_ko_cashflow
            ),
        )

    def _validate_product(self, product: KnockOutResetSnowballOption) -> None:
        if product.barrier_config.ko_observation_type != ObservationType.DISCRETE:
            raise PricingError("KOResetSnowballQuadEngine requires discrete KO monitoring.")
        if product.post_barrier_config.ko_observation_type != ObservationType.DISCRETE:
            raise PricingError("Post-KO monitoring must be discrete for quadrature engine.")
        if product.post_ko_mode != PostKOScheduleMode.ABSOLUTE:
            raise ValidationError(
                "KOResetSnowballQuadEngine only supports PostKOScheduleMode.ABSOLUTE"
            )
        if product.post_barrier_config.ko_observation_schedule is None:
            raise ValidationError("Post-KO observation schedule is required.")

    def _resolve_ko_records(
        self,
        product: KnockOutResetSnowballOption,
        pricing_env: PricingEnvironment,
        config,
    ) -> List[ResolvedObservationRecord]:
        resolved_schedule, rates, schedule_records = product._resolve_ko_schedule(
            config, pricing_env
        )
        principal_component = (
            product.initial_price * product.contract_multiplier
            if product.payoff_config.include_principal
            else 0.0
        )
        maturity_time: Optional[float] = None
        ko_records: List[ResolvedObservationRecord] = []
        for idx, rec in enumerate(resolved_schedule):
            rate = rates[idx]
            schedule_record = schedule_records[idx]
            accrual_factor = product.compute_ko_accrual_factor(
                rec.observation_time, schedule_record, pricing_env
            )
            coupon_payoff = (
                product.initial_price
                * product.contract_multiplier
                * float(rate)
                * float(accrual_factor)
            )
            payoff = principal_component + coupon_payoff

            settlement_time = rec.settlement_time
            if product.accrual_config.coupon_pay_type == CouponPayType.EXPIRY:
                maturity_time = (
                    maturity_time
                    if maturity_time is not None
                    else product.get_maturity(pricing_env)
                )
                settlement_time = maturity_time

            ko_records.append(
                ResolvedObservationRecord(
                    observation_time=rec.observation_time,
                    barrier=rec.barrier,
                    payoff=payoff,
                    settlement_time=settlement_time,
                )
            )
        return ko_records

    def __repr__(self) -> str:
        return "KOResetSnowballQuadEngine()"
