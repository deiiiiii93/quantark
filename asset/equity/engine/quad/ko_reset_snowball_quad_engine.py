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

from asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from asset.equity.engine.quad.quad_math import QuadratureMath
from asset.equity.param import QuadParams
from asset.equity.product.base_equity_product import BaseEquityProduct
from asset.equity.product.option.ko_reset_snowball_option import (
    KnockOutResetSnowballOption,
)
from asset.equity.product.option.observation_schedule import ResolvedObservationRecord
from priceenv import PricingEnvironment
from util.enum import CouponPayType, ObservationType, PostKOScheduleMode
from util.enum.engine_enums import EngineType
from util.exceptions import PricingError, ValidationError
from util.numerical import (
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
                    ki_mask = (
                        spot_grid >= ki_record.barrier
                        if product.is_reverse
                        else spot_grid <= ki_record.barrier
                    )
                    v_out[ki_mask] = v_in[ki_mask]

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

        return math_utils.interpolate(v_out, x=0.0)

    def calculate_event_stats(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> Optional[object]:
        return None

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
