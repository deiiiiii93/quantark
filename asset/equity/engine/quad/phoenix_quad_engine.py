"""
Quadrature pricing engine for Phoenix (autocallable) options.

Implements a two-state (knocked-in / not-knocked-in) quadrature recursion
with coupon jumps at observation times.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from asset.equity.engine.quad.quad_math import QuadratureMath
from asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from asset.equity.param import MCParams, QuadParams
from asset.equity.product.base_equity_product import BaseEquityProduct
from asset.equity.product.option.phoenix_option import PhoenixOption
from priceenv import PricingEnvironment
from util.enum import CouponPayType, ObservationType
from util.exceptions import PricingError, ValidationError
from util.numerical import Tolerance, is_close, is_zero, safe_log


class PhoenixQuadEngine(SnowballQuadEngine):
    """
    Quadrature pricing engine for Phoenix options with coupon jumps.
    """

    def __init__(self, params: Optional[QuadParams] = None):
        super().__init__(params=params)

    def _validate_product(self, product: PhoenixOption) -> None:
        if product.barrier_config.ko_observation_type != ObservationType.DISCRETE:
            raise PricingError("PhoenixQuadEngine requires discrete KO monitoring.")

    def price(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        if not isinstance(product, PhoenixOption):
            raise PricingError(
                f"PhoenixQuadEngine only supports PhoenixOption, got {type(product).__name__}"
            )
        if pricing_env is None:
            raise PricingError("PricingEnvironment is required for PhoenixQuadEngine.")

        self._validate_product(product)

        spot = pricing_env.spot
        maturity = product.get_maturity(pricing_env)
        if is_zero(maturity, tol=Tolerance.ZERO):
            return product.get_payoff(spot, pricing_env=pricing_env)

        rate = pricing_env.get_rate(maturity)
        div = pricing_env.get_div_yield(maturity)
        vol = pricing_env.get_vol(product.strike, maturity)
        if vol <= 0:
            raise ValidationError(f"Volatility must be positive, got {vol}")
        if div < 0:
            raise ValidationError(f"Dividend yield must be non-negative, got {div}")
        if vol > 5.0:
            raise ValidationError(f"Volatility too high for quadrature stability: {vol}")

        ko_records = product.resolve_ko_observations(pricing_env)
        if not ko_records:
            raise PricingError("KO observation schedule is empty for PhoenixQuadEngine.")

        ki_continuous = product.has_ki_barrier and (
            product.barrier_config.ki_continuous
            or product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
        )
        ki_records: Sequence = []
        if product.has_ki_barrier and not ki_continuous:
            ki_records = product.resolve_ki_observations(pricing_env)
            if not ki_records:
                raise PricingError("KI observation schedule is empty for PhoenixQuadEngine.")

        times = self._merge_times(
            [rec.observation_time for rec in ko_records],
            [rec.observation_time for rec in ki_records],
            maturity,
        )
        if not times:
            raise PricingError("Observation time grid is empty for PhoenixQuadEngine.")

        coupon_barrier = product.coupon_config.coupon_barrier
        if isinstance(coupon_barrier, list):
            if len(coupon_barrier) != len(ko_records):
                raise ValidationError(
                    "Coupon barrier schedule length does not match KO observations."
                )
            coupon_barriers = np.array(coupon_barrier, dtype=float)
        else:
            coupon_barriers = np.full(len(ko_records), float(coupon_barrier))

        align_log = self._select_alignment_log(
            spot,
            ko_records,
            coupon_barriers=coupon_barriers,
            ki_records=ki_records,
            product=product,
        )
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
        if product.has_ki_barrier and ki_continuous:
            if product.barrier_config.ki_barrier is None:
                raise PricingError("KI barrier configuration is missing.")
            if isinstance(product.barrier_config.ki_barrier, list):
                raise PricingError("Continuous KI requires scalar ki_barrier.")
            log_ki_barrier = safe_log(product.barrier_config.ki_barrier / spot)

        ko_times = np.array([rec.observation_time for rec in ko_records], dtype=float)
        period_year_fractions = np.array(
            product.get_coupon_period_year_fractions(ko_times.tolist()),
            dtype=float,
        )
        coupon_amounts = np.array(
            [
                product.get_coupon_payoff(i, year_fraction=period_year_fractions[i])
                for i in range(len(ko_records))
            ],
            dtype=float,
        )
        coupon_cumulative = np.concatenate(([0.0], np.cumsum(coupon_amounts)))

        use_memory = product.has_memory_coupon
        if use_memory and len(ko_records) > 50:
            raise ValidationError(
                f"Too many observations ({len(ko_records)}) for Memory Phoenix Quad engine. "
                "Limit is 50 to prevent performance degradation. Use MC engine instead."
            )

        # Initialize value function vectors.
        # v_in_list[k] = Value(S, t, memory=k)
        # At maturity, we initialize for the maximum possible accumulated coupons.
        # Max accumulation at maturity is N (if we missed all N-1 previous and miss the last one?
        # Actually, at maturity, we just pay what is due.
        # Let's track surfaces based on "number of missed coupons entering the next period".
        # At T (step N), we construct N+1 surfaces for having missed 0, 1, ..., N coupons.
        num_obs = len(ko_records)
        v_in_list = []
        v_out_list = []

        def accumulated_before(obs_idx: int, missed: int) -> float:
            if missed <= 0 or obs_idx <= 0:
                return 0.0
            start = max(obs_idx - missed, 0)
            return float(coupon_cumulative[obs_idx] - coupon_cumulative[start])

        # Terminal condition for each memory state (base payoff; coupons added via jumps)
        for k in range(num_obs + 1):
            v_in_k = np.array(
                [
                    product.get_maturity_payoff_v1(spot_value, pricing_env=pricing_env)
                    for spot_value in spot_grid
                ],
                dtype=float,
            )
            # V1 usually doesn't pay coupons? Checked logic: get_maturity_payoff_v1 doesn't take accumulated.
            # So accumulated is ignored for V1.

            v_out_k = np.array(
                [
                    product.get_maturity_payoff_v0(
                        spot_value,
                        accumulated_coupons=0.0,
                        pricing_env=pricing_env,
                    )
                    for spot_value in spot_grid
                ],
                dtype=float,
            )
            v_in_list.append(v_in_k)
            v_out_list.append(v_out_k)

        full_p_lr, full_p_ur, full_p0 = 0, len(grid) - 1, (len(grid) - 1) % 2
        omega_grid = math_utils.z_grid
        disable_ko_after_ki = product.barrier_config.disable_ko_after_ki
        smoothing_width = self._resolve_event_smoothing_width(math_utils, product)

        for step_index in range(len(times), 0, -1):
            obs_time = times[step_index - 1]

            # Identify if this time step is a KO/Coupon observation
            ko_weight = None
            ko_index = None
            ko_record = None
            for idx, rec in enumerate(ko_records):
                if is_close(obs_time, rec.observation_time, abs_tol=Tolerance.PRECISION):
                    ko_index = idx
                    ko_record = rec
                    break

            # If this is an observation $i$ (0-based index `ko_index`),
            # we are moving from $t_{i+1}$ (where we have valid surfaces) to $t_i$.
            # The surfaces coming from diffusion are $U[k]$ for $k \in \{0, \dots, i+1\}$ (missed 0 to i+1 coupons).
            # We need to construct surfaces $W[k]$ for $k \in \{0, \dots, i\}$ (missed 0 to i coupons).

            if ko_record is not None and ko_index is not None:
                barrier_val = coupon_barriers[ko_index]
                coupon_amt = float(coupon_amounts[ko_index])
                
                # Determine Pay/Miss weights (smoothed if enabled)
                pay_weight = self._smooth_step_weight(
                    grid,
                    barrier_val,
                    spot,
                    smoothing_width,
                    trigger_is_down=product.is_reverse,
                )
                if pay_weight is None:
                    if product.is_reverse:
                        pay_mask = spot_grid <= barrier_val
                    else:
                        pay_mask = spot_grid >= barrier_val
                    pay_weight = pay_mask.astype(float)

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

                ko_discount = self._ko_discount(
                    rate, obs_time, ko_record.settlement_time
                )
                base_ko_payoff = float(ko_record.payoff or 0.0)
                
                coupon_discount = 1.0
                if product.coupon_config.coupon_pay_type == CouponPayType.EXPIRY:
                    coupon_discount = self._ko_discount(rate, obs_time, maturity)

                new_v_in_list = []
                new_v_out_list = []

                # Max possible missed coupons entering this step is `ko_index`.
                # If use_memory is False, we only have state 0.
                max_k = ko_index if use_memory else 0

                for k in range(max_k + 1):
                    # We are in state "missed k coupons" just before observation.
                    
                    # Branch 1: Coupon Condition Met (Pay)
                    # Payoff = Current + (k * accumulated)
                    # Transition -> State 0 (reset memory)
                    accumulated_pay = (
                        accumulated_before(ko_index, k) if use_memory else 0.0
                    )
                    total_pay = (coupon_amt + accumulated_pay) * coupon_discount
                    
                    # Continuation value if paid: comes from state 0 of the next step
                    # Wait, diffused surfaces `v_out_list` are indexed by *future* memory state.
                    # If we pay now, the future starts with 0 memory.
                    # So we use `v_out_list[0]`.
                    val_pay_out = v_out_list[0] + total_pay
                    val_pay_in = v_in_list[0] + total_pay

                    # Branch 2: Coupon Condition Missed
                    # Payoff = 0
                    # Transition -> State k+1 (accumulate)
                    # Use `v_out_list[k+1]`
                    # If memory is off, we stay in state 0 (or technically state 0 again for next step).
                    next_k_miss = k + 1 if use_memory else 0
                    val_miss_out = v_out_list[next_k_miss]
                    val_miss_in = v_in_list[next_k_miss]

                    # Combine
                    # W[k] = PayMask * (Payoff + U[0]) + MissMask * U[k+1]
                    combined_out = pay_weight * val_pay_out + (1.0 - pay_weight) * val_miss_out
                    combined_in = pay_weight * val_pay_in + (1.0 - pay_weight) * val_miss_in

                    # Apply KO logic (KO overrides everything if hit)
                    # KO Payoff = BaseKO + (Current + k*Acc)
                    # Note: KO usually implies coupon payment if coupon condition met?
                    # Product logic `get_ko_payoff` adds `current_coupon` if triggered.
                    # It also adds `accumulated_coupons`.
                    # So:
                    ko_pay_val = (base_ko_payoff + total_pay) * ko_discount
                    # If miss coupon barrier but hit KO barrier? 
                    # Usually KO barrier >= Coupon barrier (for standard).
                    # If Spot > KO, then Spot > Coupon. So PayMask is True.
                    # If reverse: Spot < KO <= Coupon. So PayMask is True.
                    # So if KO hit, we typically pay the coupon too.
                    # But we should be careful if KO barrier is tighter than coupon barrier (rare).
                    # `get_ko_payoff` handles this check. 
                    # Here we assume if KO hit, we pay.
                    # Ideally we check masks again.
                    
                    # Detailed KO payoff logic:
                    # If KO hit:
                    #   If Coupon hit: Pay Base + Current + Acc.
                    #   If Coupon miss: Pay Base + Acc? Or Base only?
                    #   Phoenix `get_ko_payoff` adds `accumulated_coupons` unconditionally?
                    #   Let's check `get_ko_payoff`.
                    #   It adds `accumulated_coupons` unconditionally.
                    #   It adds `current_coupon` IF `is_coupon_triggered`.
                    
                    ko_val_grid = np.full_like(combined_out, base_ko_payoff * ko_discount)
                    # Add accumulated unconditionally
                    ko_val_grid += (accumulated_pay * ko_discount)
                    # Add current coupon proportionally to pay-weight
                    ko_val_grid += pay_weight * (coupon_amt * ko_discount)

                    combined_out = ko_weight * ko_val_grid + (1.0 - ko_weight) * combined_out
                    if not disable_ko_after_ki:
                        combined_in = ko_weight * ko_val_grid + (1.0 - ko_weight) * combined_in

                    new_v_in_list.append(combined_in)
                    new_v_out_list.append(combined_out)

                v_in_list = new_v_in_list
                v_out_list = new_v_out_list

            # KI Logic (Discrete/Continuous) applied to ALL surfaces
            if ki_continuous and log_ki_barrier is not None:
                ki_mask = (
                    spot_grid >= product.barrier_config.ki_barrier
                    if product.is_reverse
                    else spot_grid <= product.barrier_config.ki_barrier
                )
                for i in range(len(v_out_list)):
                    v_out_list[i][ki_mask] = v_in_list[i][ki_mask]
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
                    for i in range(len(v_out_list)):
                        v_out_list[i] = (1.0 - ki_weight) * v_out_list[i] + ki_weight * v_in_list[i]

            # Diffusion Step
            tau_step = float(tau[step_index])
            prefactor = math.exp(-beta * tau_step) / math.sqrt(math.pi * tau_step) / 2.0
            omega_array = np.exp(-(omega_grid**2) / (4.0 * tau_step) - alpha * omega_grid)

            # Diffuse all surfaces
            for i in range(len(v_in_list)):
                v_in_list[i] = self._diffuse_fft(
                    v_in_list[i],
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
                    v_out_list[i] = self._diffuse_with_bridge(
                        v_out_list[i],
                        v_in_list[i],
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
                    v_out_list[i] = self._diffuse_fft(
                        v_out_list[i],
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

        # Final result is value at t=0 with 0 accumulated coupons
        return math_utils.interpolate(v_out_list[0], x=0.0)

    def calculate_event_stats(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ):
        if not isinstance(product, PhoenixOption):
            return None

        from asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine

        mc_engine = PhoenixMCEngine(params=MCParams())
        return mc_engine.calculate_event_stats(product, pricing_env)

    def __repr__(self):
        return "PhoenixQuadEngine()"

    def _select_alignment_log(
        self,
        spot: float,
        ko_records: Sequence,
        coupon_barriers: Optional[np.ndarray],
        ki_records: Sequence,
        product: PhoenixOption,
    ) -> Optional[float]:
        ko_candidates: list[float] = []
        for rec in ko_records:
            if rec.barrier is not None and rec.barrier > 0:
                ko_candidates.append(float(rec.barrier))

        coupon_candidates: list[float] = []
        if coupon_barriers is not None:
            coupon_candidates.extend([float(b) for b in coupon_barriers if b > 0])

        ki_candidates: list[float] = []
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
        coupon_logs = to_logs(coupon_candidates)
        ki_logs = to_logs(ki_candidates)

        if not ko_logs and not coupon_logs and not ki_logs:
            return None

        def closest(logs: list[float]) -> Optional[float]:
            if not logs:
                return None
            idx = int(np.argmin(np.abs(np.asarray(logs))))
            return float(logs[idx])

        priority = self._resolve_align_priority()
        if priority == "ko":
            return closest(ko_logs) or closest(coupon_logs) or closest(ki_logs)
        if priority == "coupon":
            return closest(coupon_logs) or closest(ko_logs) or closest(ki_logs)
        if priority == "ki":
            return closest(ki_logs) or closest(ko_logs) or closest(coupon_logs)

        # auto: reverse prefers KO near spot, else coupon, else KI
        if product.is_reverse:
            ko_near = closest(ko_logs)
            if ko_near is not None and abs(ko_near) <= 0.05:
                return ko_near
            return closest(coupon_logs) or closest(ki_logs) or closest(ko_logs)

        return closest(ko_logs + coupon_logs + ki_logs)

    def _resolve_align_priority(self) -> str:
        priority = getattr(self.params, "align_priority", None)
        if priority is None:
            return "auto"
        return str(priority).lower()

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

    def _resolve_event_smoothing_width(
        self, math_utils: QuadratureMath, product: PhoenixOption
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
