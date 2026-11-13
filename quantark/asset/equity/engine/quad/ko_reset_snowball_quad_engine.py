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

from quantark.asset.equity.engine.event_stats import EventStatistics
from quantark.asset.equity.engine.settlement_support import (
    resolve_terminal_timing,
)
from quantark.asset.equity.engine.quad.quad_adapters import (
    resolve_structured_payment_timings,
)
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.priceenv.term_sampling import make_df_fn
from quantark.asset.equity.engine.quad.quad_math import QuadratureMath
from quantark.asset.equity.param import QuadParams
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option.ko_reset_snowball_option import (
    KnockOutResetSnowballOption,
)
from quantark.asset.equity.product.option.observation_schedule import ResolvedObservationRecord
from quantark.asset.equity.settlement import CashflowKind, SettlementRequest, SettlementResolver
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import CouponPayType, ObservationType, PostKOScheduleMode
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import PricingError, ValidationError
from quantark.util.numerical import (
    Tolerance,
    is_close,
    safe_log,
    validate_non_negative,
    validate_positive,
)


class KOResetSnowballQuadEngine(SnowballQuadEngine):
    """
    Quadrature pricing engine for KO-reset Snowball options.
    """

    engine_type = EngineType.QUADRATURE

    # This engine reads its price off its OWN surfaces, so the
    # transition readout is not implemented here yet; refuse it rather
    # than report a legacy-readout price under another name.
    supported_readouts = ("legacy_linear",)

    def __init__(self, params: Optional[QuadParams] = None) -> None:
        super().__init__(params=params)

    def price(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        return self._price_once(product, pricing_env)

    def _folded_solve(self, product, pricing_env, fold) -> float:
        """``price`` never auto-converges on this engine, so neither does the folded solve."""
        return self._price_once(product, pricing_env, fold=fold)

    def _bind_ko_reset_fold(self, fold, product, pricing_env, maturity: float):
        """Both schedules' in-life records bound to the overlay: ``({id: amount}, pre, post)``."""
        pre, post = (
            self._fold_in_life(self._resolve_ko_records(product, pricing_env, config), maturity)
            for config in (product.barrier_config, product.post_barrier_config)
        )
        amounts = self._bind_fold_ko(
            fold, [("pre", rec) for rec in pre] + [("post", rec) for rec in post]
        )
        return amounts, pre, post

    def _price_once(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        *,
        fold=None,
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

        # The contract's two rules (see KnockOutResetSnowballOption): a knock-in carried from an earlier
        # day has replaced the first schedule by the second for good, so the price is read off the
        # knocked-in surface; and a contract past its first schedule without one has matured.
        carried_knock_in = bool(getattr(product, "_otc_lifecycle_knocked_in", False))
        product.require_alive(pricing_env, carried_knock_in)

        spot = pricing_env.spot
        maturity = product.get_maturity(pricing_env)
        validate_positive(spot, "spot")
        validate_positive(maturity, "maturity", allow_zero=True)
        if maturity <= 0.0:
            self._refuse_expired_fold(fold)
        # Observations AT the valuation instant are decided on the known spot, on the schedule the
        # knock-in state puts in force and with the knock-out winning a tie; the recursion below
        # starts after them (``_merge_times`` drops time zero). This engine used to skip them, and at
        # zero time to maturity returned the never-knocked-in payoff whatever was pending and
        # whatever knock-in the contract carried.
        decision = product.decide_observations_at_valuation(
            spot,
            pricing_env,
            knocked_in=carried_knock_in,
        )
        if decision.knocked_out:
            record = decision.ko_record
            settlement = record.settlement_time
            if product.accrual_config.coupon_pay_type == CouponPayType.EXPIRY:
                settlement = resolve_terminal_timing(product, pricing_env).payment_time
            discount = (
                pricing_env.get_discount_factor(settlement)
                if settlement is not None and settlement > 0.0
                else 1.0
            )
            if fold is not None:
                amounts, pre, post = self._bind_ko_reset_fold(fold, product, pricing_env, maturity)
                decided = self._match_record(
                    record.observation_time, pre if decision.ko_regime == "pre" else post
                )
                return (
                    float(record.payoff) + self._folded_amount(amounts, decided)
                ) * float(discount)
            return float(record.payoff) * float(discount)
        self._require_resolvable_time(maturity)
        if maturity <= 0.0:
            # the redemption is cash at its payment time, which a settlement lag puts after this instant
            redemption = product.get_payoff(spot, pricing_env, knocked_in=decision.knocked_in)
            return float(redemption) * float(resolve_terminal_timing(product, pricing_env).payment_df)
        discrete_ki = not (
            product.barrier_config.ki_continuous
            or product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
        )
        if discrete_ki:
            # a knock-in observed now replaces the first schedule from here on; a continuously
            # monitored level below the spot stays with the surfaces, which already carry it
            carried_knock_in = decision.knocked_in

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
        if not pre_ko_records and not carried_knock_in:
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
        # An empty schedule 2 (aged past its last date, spec 2026-10-07) is legal: a knock-in
        # then leaves no knock-out before maturity. An empty knock-in side folds away too.
        all_ko_records = [*pre_ko_records, *post_ko_records]
        payment_timings = resolve_structured_payment_timings(
            product,
            pricing_env,
            all_ko_records,
        )
        event_delay_by_record = {
            id(record): float(delay_df)
            for record, delay_df in zip(
                all_ko_records,
                payment_timings.event_delay_dfs,
            )
        }
        fold_ko: Optional[dict] = None
        if fold is not None:
            fold_ko = self._bind_fold_ko(
                fold,
                [("pre", rec) for rec in self._fold_in_life(pre_ko_records, maturity)]
                + [("post", rec) for rec in self._fold_in_life(post_ko_records, maturity)],
            )

        ki_continuous = product.has_ki_barrier and (
            product.barrier_config.ki_continuous
            or product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
        )
        ki_records: Sequence = []
        if product.has_ki_barrier and not ki_continuous and not carried_knock_in:
            ki_records = product.resolve_ki_observations(pricing_env)

        times = self._merge_times(
            [rec.observation_time for rec in pre_ko_records]
            + [rec.observation_time for rec in post_ko_records],
            [rec.observation_time for rec in ki_records],
            maturity,
        )
        if ki_continuous:
            from quantark.asset.equity.engine.continuous_ki_support import continuous_time_grid
            times, self._continuous_ki_model = continuous_time_grid(pricing_env, times, self.params)
            times = self._merge_times(times, (), maturity)
        if not times:
            raise PricingError("Observation time grid is empty for KOResetSnowballQuadEngine.")

        # When recording for CVA exposure, diffuse through (and record at) delayed KO
        # settlement times from BOTH the pre-KI and post-KI schedules (no-event nodes).
        if record_grids:
            times = self._insert_settlement_times(
                times, list(pre_ko_records) + list(post_ko_records), maturity)

        align_log = self._align_to_the_level_in_force(spot, product, carried_knock_in)
        barrier_logs = self._barrier_logs(
            spot,
            [record.barrier for record in pre_ko_records],
            [record.barrier for record in post_ko_records],
            [record.barrier for record in ki_records],
            product.barrier_config.ki_barrier,
        )
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
            barrier_logs=barrier_logs,
            cell_stretch=self._resolve_align_cell_stretch(),
            integration_rule=self.params.integration_rule,
            fft_padding_factor=fft_padding_factor,
            fft_filter_alpha=fft_filter_alpha,
            fft_filter_power=fft_filter_power,
        )
        grid = math_utils.grid
        spot_grid = spot * np.exp(grid)
        dt = self._build_dt(times)
        rate_vec, div_vec, vol_vec = self._term_step_params(
            pricing_env, product.strike, times, rate, div, vol
        )
        df_local = make_df_fn(pricing_env)
        tau = 0.5 * vol_vec * vol_vec * dt
        if np.any(tau[1:] <= 0.0):
            raise ValidationError("time step too small for quadrature solver.")

        alpha_vec = (rate_vec - div_vec - 0.5 * vol_vec * vol_vec) / (
            vol_vec * vol_vec
        )
        beta_vec = (rate_vec - div_vec - 0.5 * vol_vec * vol_vec) ** 2 / (
            vol_vec**4
        ) + 2.0 * rate_vec / (vol_vec * vol_vec)

        terminal = payment_timings.terminal
        pre_maturity = float(product.get_pre_maturity_time(pricing_env))
        v_in_cash = np.array(
            [
                product.get_maturity_payoff_v1(spot_value, pricing_env=pricing_env)
                for spot_value in spot_grid
            ],
            dtype=float,
        )
        if fold is not None:
            v_in_cash = v_in_cash + fold.bind_terminal(
                "ki", terminal.determination_time, terminal.payment_time
            )
        v_in = v_in_cash * terminal.delay_df
        # The not-yet-knocked-in contract matures on the PRE-KI schedule; only a
        # knocked-in one runs on to `maturity`. Seeding v_out at the end of the
        # grid would price it through a period it can never reach, so it is
        # seeded when the sweep reaches pre_maturity and is identically zero
        # above -- which is its true value there, the contract having already
        # terminated in that state.
        v_out_terminal = np.array(
            [
                product.get_maturity_payoff_v0(spot_value, pricing_env=pricing_env)
                for spot_value in spot_grid
            ],
            dtype=float,
        ) * payment_timings.terminal.delay_df
        if fold is not None:
            # Folded cash of the alive branch is paid at the first schedule's maturity plus the
            # terminal settlement lag (the fold structure's slot), discounted from there to the
            # pre-maturity by its own exact ratio; the product's own redemption keeps its discount.
            no_ki_payment = pre_maturity + (terminal.payment_time - terminal.determination_time)
            no_ki_cash = fold.bind_terminal("no_ki", pre_maturity, no_ki_payment)
            v_out_terminal = v_out_terminal + no_ki_cash * (
                pricing_env.get_discount_factor(no_ki_payment)
                / pricing_env.get_discount_factor(pre_maturity)
            )
        v_out = np.zeros_like(v_out_terminal)
        v_out_seeded = False

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

            out_live = obs_time < pre_maturity or is_close(
                obs_time, pre_maturity, abs_tol=Tolerance.PRECISION
            )
            if out_live and not v_out_seeded:
                v_out = v_out_terminal.copy()
                v_out_seeded = True

            pre_record = self._match_record(obs_time, pre_ko_records)
            post_record = self._match_record(obs_time, post_ko_records)

            def ko_cash(record):
                if record is None:
                    return 0.0
                extra = 0.0 if fold_ko is None else fold_ko[id(record)]
                return (record.payoff + extra) * event_delay_by_record[id(record)]

            v_out, v_in = self._apply_reset_observation(
                product, grid, spot_grid, spot, smoothing_width, v_out, v_in,
                obs_time, out_live, pre_ko_records, post_ko_records, ki_records,
                ki_continuous, log_ki_barrier, ko_cash(pre_record), ko_cash(post_record),
            )

            # Post-event continuation surfaces at obs_time (v_out = pre-KI/not-yet-KI,
            # v_in = post-KI/knocked-in), before diffusing back to the previous step.
            if record_grids:
                self._backward_grids[float(obs_time)] = (
                    spot_grid.copy(), v_in.copy(), v_out.copy())

            tau_step = float(tau[step_index])
            alpha = float(alpha_vec[step_index])
            beta = float(beta_vec[step_index])
            vol_step = float(vol_vec[step_index])
            prefactor = math.exp(-beta * tau_step) / math.sqrt(math.pi * tau_step) / 2.0
            omega_array = np.exp(
                -(omega_grid**2) / (4.0 * tau_step) - alpha * omega_grid
            )

            # The bridge correction below samples BOTH surfaces at this step's
            # slice, so retain the knocked-in surface before it is rolled back.
            v_in_at_obs = v_in
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

            if not out_live:
                # v_out is identically zero above pre_maturity: in that
                # state the contract has already matured.
                continue

            if ki_continuous and log_ki_barrier is not None:
                v_out = self._diffuse_with_bridge(
                    v_out,
                    v_in_at_obs,
                    math_utils,
                    omega_array,
                    prefactor,
                    full_p_lr,
                    full_p_ur,
                    full_p0,
                    log_ki_barrier,
                    alpha,
                    beta,
                    vol_step,
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

        if not v_out_seeded and not carried_knock_in:
            # The first schedule ends at the valuation instant, which has no step of its own (``_merge_times``
            # drops time zero), so v_out was never seeded and read 0. Its last observation is decided above; a
            # knock-in now (continuously monitored, the level at the spot) runs on the second schedule, and a
            # contract not knocked in has matured here on the known spot, its redemption determined now and paid
            # as Monte Carlo pays it (review 2026-09-21 R9).
            if decision.knocked_in:
                carried_knock_in = True
            else:
                timing = SettlementResolver.resolve_contingent(
                    product,
                    SettlementRequest(
                        kind=CashflowKind.TERMINAL,
                        determination_time=pre_maturity,
                        cashflow_id="ko_reset_v0_terminal",
                    ),
                    pricing_env,
                )
                redemption = product.get_maturity_payoff_v0(spot, pricing_env=pricing_env)
                if fold is not None:
                    redemption = float(redemption) + fold.bind_terminal(
                        "no_ki", pre_maturity, timing.payment_time
                    )
                return float(redemption) * float(timing.payment_df)

        # This engine always read the not-knocked-in surface, so a knocked-in contract was priced as if
        # its first schedule were still in force.
        value_surface = v_in if carried_knock_in else v_out
        self._last_spot_greeks_grid = (spot_grid.copy(), value_surface.copy())
        return math_utils.interpolate(value_surface, x=0.0)

    def _apply_reset_observation(
        self, product, grid, spot_grid, spot, smoothing_width, v_out, v_in,
        obs_time, out_live, pre_ko_records, post_ko_records, ki_records,
        ki_continuous, log_ki_barrier, pre_value, post_value,
    ):
        """Shared price/statistics event ordering, including overlapping barriers."""
        disable_ko_after_ki = product.barrier_config.disable_ko_after_ki
        pre_ko_weight = None
        pre_ko_last = False
        pre_ko_record = self._match_record(obs_time, pre_ko_records)
        if pre_ko_record is not None:
            ko_value = pre_value
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
            pre_ko_value = ko_value
            # The first-schedule knock-out wins a tie with a knock-in OBSERVED at the same
            # instant. The knocked-in surface never carries that knock-out (it carries the second
            # schedule's), so a knock-in copy made after it would overwrite it wherever both
            # levels are breached: next to a discrete knock-in it is applied last, below. (A
            # continuously monitored knock-in breached at a node was touched before the
            # observation. The two events write disjoint nodes unless the first-schedule level
            # is at or below the knock-in level.)
            pre_ko_last = bool(
                out_live
                and not ki_continuous
                and self._match_record(obs_time, ki_records) is not None
            )
            if not pre_ko_last:
                v_out = ko_weight * ko_value + (1.0 - ko_weight) * v_out

        post_ko_record = self._match_record(obs_time, post_ko_records)
        if post_ko_record is not None and not disable_ko_after_ki:
            ko_value = post_value
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

        if out_live and ki_continuous and log_ki_barrier is not None:
            ki_mask = (
                spot_grid >= product.barrier_config.ki_barrier
                if product.is_reverse
                else spot_grid <= product.barrier_config.ki_barrier
            )
            v_out[..., ki_mask] = v_in[..., ki_mask]
        elif out_live and ki_records:
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
                    ko_weight=None if pre_ko_last else pre_ko_weight,
                )

        if pre_ko_last:
            v_out = pre_ko_weight * pre_ko_value + (1.0 - pre_ko_weight) * v_out

        return v_out, v_in

    def _align_to_the_level_in_force(self, spot, product, carried_knock_in: bool):
        """The lattice is pinned to the knock-out level in force: the second schedule's once knocked in.

        Pinned to the first schedule's level, a knocked-in contract converged at first order in the
        spacing (5e-3 at 2001 nodes, against 1e-4 for the same claim written as a knocked-in snowball).
        """
        if not carried_knock_in:
            return self._select_alignment_log(spot, product)
        levels = product.post_barrier_config.ko_barrier
        return self._select_alignment_log(
            spot,
            product,
            ko_candidates_override=levels if isinstance(levels, list) else [levels],
        )

    def calculate_event_stats(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        *,
        streams: Optional[frozenset] = None,
    ) -> Optional[EventStatistics]:
        """``EventStatistics`` of the price's own KO-reset recursion (unified event statistics spec T17).

        Columns: per knock-out record of both schedules its discounted unit, the maturity unit and cash (the first
        schedule's end, never knocked in), the knocked-in settlement unit and cash. The knock-in path statistics come
        from the forward adjoint pass, run only when ``streams`` asks for the knock-in stream (spec D17);
        ``event_stats_mode='forward_density'`` is ignored (this recursion has no forward-density mode).
        """
        if not isinstance(product, KnockOutResetSnowballOption):
            return None
        if pricing_env is None:
            raise PricingError("PricingEnvironment is required for KOResetSnowballQuadEngine.")
        return self._compute_event_stats(product, pricing_env, streams=streams)

    _statistics_family = "ko_reset"

    def _compute_event_stats(
        self,
        product: KnockOutResetSnowballOption,
        pricing_env: PricingEnvironment,
        *,
        streams: Optional[frozenset] = None,
        pv: Optional[float] = None,
    ) -> Optional[EventStatistics]:
        from quantark.asset.equity.engine.pde.core.statistics import event_statistics as es
        from quantark.asset.equity.engine.quad import v1_statistics
        from quantark.asset.equity.engine.quad.v1_adjoint import EventStep, MarchLog

        self._validate_product(product)
        want_ki = es.want_knock_in_path(streams)
        carried_knock_in = bool(getattr(product, "_otc_lifecycle_knocked_in", False))
        product.require_alive(pricing_env, carried_knock_in)

        spot = pricing_env.spot
        maturity = product.get_maturity(pricing_env)
        validate_positive(spot, "spot")
        validate_positive(maturity, "maturity", allow_zero=True)
        self._require_resolvable_time(maturity)
        if maturity <= 0.0:
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
        post_ko_records = self._resolve_ko_records(
            product, pricing_env, product.post_barrier_config
        )
        post_ko_records = [
            rec
            for rec in post_ko_records
            if rec.observation_time <= maturity
            or is_close(rec.observation_time, maturity, abs_tol=Tolerance.PRECISION)
        ]
        # A knock-out at the valuation instant is decided on the known spot, as in ``price``.
        decision = product.decide_observations_at_valuation(
            spot, pricing_env, knocked_in=carried_knock_in
        )
        if decision.knocked_out:
            return self._decided_ko_event_stats(
                product, pricing_env, pre_ko_records, post_ko_records, decision, streams=streams
            )
        if not (
            product.barrier_config.ki_continuous
            or product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
        ):
            carried_knock_in = decision.knocked_in
        # An ended schedule is legal: past the first, only a carried knock-in is alive (``require_alive``);
        # an aged-out second leaves a knock-in no knock-out before maturity. The recursion reads either
        # as no records on that side.
        if not pre_ko_records and not carried_knock_in:
            return None

        ki_continuous = product.has_ki_barrier and (
            product.barrier_config.ki_continuous
            or product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
        )
        ki_records: Sequence = []
        if product.has_ki_barrier and not ki_continuous and not carried_knock_in:
            ki_records = product.resolve_ki_observations(pricing_env)

        times = self._merge_times(
            [rec.observation_time for rec in pre_ko_records]
            + [rec.observation_time for rec in post_ko_records],
            [rec.observation_time for rec in ki_records],
            maturity,
        )
        if ki_continuous:
            from quantark.asset.equity.engine.continuous_ki_support import continuous_time_grid
            times, self._continuous_ki_model = continuous_time_grid(pricing_env, times, self.params)
            times = self._merge_times(times, (), maturity)
        if not times:
            return None

        align_log = self._align_to_the_level_in_force(spot, product, carried_knock_in)
        barrier_logs = self._barrier_logs(
            spot,
            [record.barrier for record in pre_ko_records],
            [record.barrier for record in post_ko_records],
            [record.barrier for record in ki_records],
            product.barrier_config.ki_barrier,
        )
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
            barrier_logs=barrier_logs,
            cell_stretch=self._resolve_align_cell_stretch(),
            integration_rule=self.params.integration_rule,
            fft_padding_factor=fft_padding_factor,
            fft_filter_alpha=fft_filter_alpha,
            fft_filter_power=fft_filter_power,
        )
        grid = math_utils.grid
        spot_grid = spot * np.exp(grid)
        dt = self._build_dt(times)
        rate_vec, div_vec, vol_vec = self._term_step_params(
            pricing_env, product.strike, times, rate, div, vol
        )
        df_local = make_df_fn(pricing_env)
        tau = 0.5 * vol_vec * vol_vec * dt
        if np.any(tau[1:] <= 0.0):
            raise ValidationError("time step too small for quadrature solver.")

        alpha_vec = (rate_vec - div_vec - 0.5 * vol_vec * vol_vec) / (
            vol_vec * vol_vec
        )
        beta_vec = (rate_vec - div_vec - 0.5 * vol_vec * vol_vec) ** 2 / (
            vol_vec**4
        ) + 2.0 * rate_vec / (vol_vec * vol_vec)

        # Columns: knock-out units of both schedules, then the maturity unit and cash (the first schedule's end:
        # v_out is seeded there, as ``price`` seeds it) and the knocked-in settlement unit and cash.
        n_pre = len(pre_ko_records)
        n_post = len(post_ko_records)
        post_offset = n_pre
        m_unit = n_pre + n_post
        m_cash, kis_unit, kis_cash = m_unit + 1, m_unit + 2, m_unit + 3
        n_rows = m_unit + 4

        all_ko_records = [*pre_ko_records, *post_ko_records]
        timings = resolve_structured_payment_timings(product, pricing_env, all_ko_records)
        terminal = timings.terminal
        v_in_cash = np.array(
            [product.get_maturity_payoff_v1(s, pricing_env=pricing_env) for s in spot_grid], dtype=float
        )
        v_out_cash = np.array(
            [product.get_maturity_payoff_v0(s, pricing_env=pricing_env) for s in spot_grid], dtype=float
        )
        terminal_in_cash = v_in_cash * terminal.delay_df
        terminal_out_cash = v_out_cash * terminal.delay_df

        v_in = np.zeros((n_rows, grid.size), dtype=float)
        v_out = np.zeros((n_rows, grid.size), dtype=float)
        v_in[kis_unit] = 1.0
        v_in[kis_cash] = terminal_in_cash
        v_out_seeded = False

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
        smoothing_width = self._resolve_event_smoothing_width(math_utils, product)
        pre_maturity = float(product.get_pre_maturity_time(pricing_env))
        log = MarchLog(engine=self, math_utils=math_utils, readout_in=bool(knocked_in_at_valuation),
                       terminal_out={"maturity_unit": np.ones(grid.size), "maturity_cash": terminal_out_cash},
                       terminal_in={"ki_settlement_unit": np.ones(grid.size), "ki_settlement_cash": terminal_in_cash})

        for step_index in range(len(times), 0, -1):
            obs_time = times[step_index - 1]

            out_live = obs_time < pre_maturity or is_close(
                obs_time, pre_maturity, abs_tol=Tolerance.PRECISION
            )
            seed = out_live and not v_out_seeded
            if seed:
                v_out[m_unit] = 1.0
                v_out[m_cash] = terminal_out_cash
                v_out_seeded = True

            pre_record = self._match_record(obs_time, pre_ko_records)
            post_record = self._match_record(obs_time, post_ko_records)
            pre_value = np.zeros_like(v_out)
            post_value = np.zeros_like(v_in)
            sources = []
            units = {}
            pre_unit = post_unit = 0.0
            if pre_record is not None:
                pre_unit = self._ko_discount(rate, obs_time, pre_record.settlement_time, df_fn=df_local)
                pre_value[pre_ko_records.index(pre_record)] = pre_unit
                sources.append(("ko", pre_ko_records.index(pre_record)))
                units[sources[-1]] = float(pre_unit)
            if post_record is not None:
                post_unit = self._ko_discount(rate, obs_time, post_record.settlement_time, df_fn=df_local)
                post_value[post_offset + post_ko_records.index(post_record)] = post_unit
                if not product.barrier_config.disable_ko_after_ki:
                    sources.append(("ko", post_offset + post_ko_records.index(post_record)))
                    units[sources[-1]] = float(post_unit)
            v_out, v_in = self._apply_reset_observation(
                product, grid, spot_grid, spot, smoothing_width, v_out, v_in,
                obs_time, out_live, pre_ko_records, post_ko_records, ki_records,
                ki_continuous, log_ki_barrier, pre_value, post_value,
            )

            tau_step = float(tau[step_index])
            alpha = float(alpha_vec[step_index])
            beta = float(beta_vec[step_index])
            vol_step = float(vol_vec[step_index])
            prefactor = math.exp(-beta * tau_step) / math.sqrt(math.pi * tau_step) / 2.0
            omega_array = np.exp(
                -(omega_grid**2) / (4.0 * tau_step) - alpha * omega_grid
            )
            log.steps.append(EventStep(
                time=float(obs_time),
                event=self._reset_statistics_event(
                    product, grid, spot_grid, spot, smoothing_width, obs_time, out_live, pre_ko_records,
                    post_ko_records, ki_records, ki_continuous, log_ki_barrier,
                    pre_unit=float(pre_unit), post_unit=float(post_unit),
                    pre_key=sources[0] if pre_record is not None else None,
                ),
                sources=tuple(sources),
                source_units=units,
                diffusion=(tau_step, alpha, beta, prefactor, omega_array),
                bridge=((log_ki_barrier, vol_step, float(dt[step_index]), bool(product.is_reverse))
                        if ki_continuous and log_ki_barrier is not None else None),
                out_diffused=bool(out_live),
                seed_out=bool(seed),
            ))

            # The bridge correction below samples BOTH surfaces at this step's
            # slice, so retain the knocked-in surface before it is rolled back.
            v_in_at_obs = v_in
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

            if not out_live:
                # v_out is identically zero above pre_maturity: in that
                # state the contract has already matured.
                continue

            if ki_continuous and log_ki_barrier is not None:
                v_out = self._diffuse_with_bridge(
                    v_out,
                    v_in_at_obs,
                    math_utils,
                    omega_array,
                    prefactor,
                    full_p_lr,
                    full_p_ur,
                    full_p0,
                    log_ki_barrier,
                    alpha,
                    beta,
                    vol_step,
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

        if pv is None:
            pv = float(self.price(product, pricing_env))
        if not v_out_seeded and not knocked_in_at_valuation:
            # the first schedule ended at the valuation instant (no step of its own): ``price`` decides the
            # maturity on the known spot, so the whole price is the maturity row, certain
            return self._matured_at_valuation_statistics(
                product, pricing_env, pre_ko_records, post_ko_records, pv=pv, pre_maturity=pre_maturity,
                streams=streams,
            )

        initial_surface = v_in if knocked_in_at_valuation else v_out
        values_at_spot = np.array(
            [math_utils.interpolate(initial_surface[i], x=0.0) for i in range(n_rows)], dtype=float
        )
        unit = values_at_spot[:m_unit]
        log.columns = {("ko", i): float(unit[i]) for i in range(m_unit)
                       if not (i >= post_offset and product.barrier_config.disable_ko_after_ki)}
        log.columns.update({
            ("terminal", "maturity_unit"): float(values_at_spot[m_unit]),
            ("terminal", "maturity_cash"): float(values_at_spot[m_cash]),
            ("terminal", "ki_settlement_unit"): float(values_at_spot[kis_unit]),
            ("terminal", "ki_settlement_cash"): float(values_at_spot[kis_cash]),
        })
        self._last_v1_march = log

        observations, maturity_row, ki_settlement = self._reset_rows(
            product, pricing_env, pre_ko_records, post_ko_records, unit, terminal,
            maturity_unit=float(values_at_spot[m_unit]), maturity_cash=float(values_at_spot[m_cash]),
            ki_unit=float(values_at_spot[kis_unit]), ki_cash=float(values_at_spot[kis_cash]),
            pre_maturity=pre_maturity, maturity=float(maturity),
        )
        rule = es.knock_in_rule(product, knocked_out_now=False, knocked_in_now=bool(knocked_in_at_valuation))
        product_ki_records = (product.resolve_ki_observations(pricing_env)
                              if product.has_ki_barrier and not ki_continuous else [])
        return v1_statistics.statistics(
            product, pricing_env, family=self._statistics_family, pv=pv, observations=observations, coupons=None,
            maturity=maturity_row, ki_settlement=ki_settlement, rule=rule, want_knock_in=want_ki,
            forward=lambda: v1_statistics.knock_in_table(
                product, pricing_env, log, horizon=pre_maturity, continuous=bool(ki_continuous),
                ki_records=product_ki_records,
            ),
            unavailable={},
        )

    def _reset_statistics_event(self, product, grid, spot_grid, spot, width, obs_time, out_live, pre_ko_records,
                                post_ko_records, ki_records, ki_continuous, log_ki_barrier, *, pre_unit, post_unit,
                                pre_key):
        """The observation's event map on the two note states, for the adjoint pass: ``_apply_reset_observation``
        with every knock-out source zero (``source=None``) or one schedule's source alone."""

        def event(v_out, v_in, source):
            v_out = np.array(v_out, dtype=float)
            v_in = np.array(v_in, dtype=float)
            pre_value = np.zeros_like(v_out)
            post_value = np.zeros_like(v_in)
            if source is not None:
                if source == pre_key:
                    pre_value[...] = pre_unit
                else:
                    post_value[...] = post_unit
            return self._apply_reset_observation(
                product, grid, spot_grid, spot, width, v_out, v_in, obs_time, out_live, pre_ko_records,
                post_ko_records, ki_records, ki_continuous, log_ki_barrier, pre_value, post_value,
            )

        return event

    def _reset_rows(self, product, pricing_env, pre_ko_records, post_ko_records, unit, terminal, *, maturity_unit,
                    maturity_cash, ki_unit, ki_cash, pre_maturity, maturity):
        """``(observations, maturity, ki_settlement)``: both schedules' knock-out rows in time order (the first
        schedule's record first at a tie), the maturity row at the first schedule's end (its mass leaves the survival
        there, F5) and the knocked-in settlement at maturity."""
        from quantark.asset.equity.engine.pde.core.statistics import event_statistics as es

        records = [*pre_ko_records, *post_ko_records]
        tags = ["pre"] * len(pre_ko_records) + ["post"] * len(post_ko_records)
        times = np.array([float(r.observation_time) for r in records], dtype=float)
        order = (np.lexsort((np.array([0 if t == "pre" else 1 for t in tags]), times)) if records
                 else np.zeros(0, dtype=int))
        observations = es.observation_table(pricing_env, records, unit, schedule=tags, order=order)
        delay = float(terminal.payment_time) - float(terminal.determination_time)
        last_pre = pre_ko_records[-1] if pre_ko_records else None
        pre_date = (last_pre.observation_date if last_pre is not None
                    and is_close(float(last_pre.observation_time), pre_maturity) else None)
        df_pre = float(pricing_env.get_discount_factor(pre_maturity)) if pre_maturity > 0.0 else 1.0
        maturity_row = es.outcome_row(
            pricing_env, time=pre_maturity, payment_time=pre_maturity + delay, date=pre_date,
            payment_date=pre_date if is_close(delay, 0.0) else None,
            probability=maturity_unit / df_pre, discounted=maturity_cash,
        )
        ki_settlement = es.outcome_row(
            pricing_env, time=maturity, payment_time=float(terminal.payment_time), date=terminal.determination_date,
            payment_date=terminal.payment_date,
            probability=ki_unit / float(pricing_env.get_discount_factor(maturity)), discounted=ki_cash,
        )
        if observations.size and maturity_row.time < float(observations.times[-1]) - Tolerance.ZERO:
            from dataclasses import replace

            # F5: the first schedule's maturity mass leaves the survival at its own determination
            observations = replace(observations, survival=observations.survival - np.where(
                observations.times >= maturity_row.time, maturity_row.probability, 0.0))
        return observations, maturity_row, ki_settlement

    def _decided_ko_event_stats(
        self,
        product: KnockOutResetSnowballOption,
        pricing_env: PricingEnvironment,
        pre_ko_records: Sequence,
        post_ko_records: Sequence,
        decision,
        *,
        streams=None,
    ) -> EventStatistics:
        """A knock-out decided at valuation: certain, on its record of the schedule in force (DESIGN §4.5).

        ``price`` decides an observation at the valuation instant on the known spot: the decided record carries the
        whole price, nothing is left to mature. No knock-in path statistics but a carried knock-in (the knock-out
        wins a tie with a knock-in observed now)."""
        from quantark.asset.equity.engine.pde.core.statistics import event_statistics as es
        from quantark.asset.equity.engine.quad import v1_statistics

        records = pre_ko_records if decision.ko_regime == "pre" else post_ko_records
        decided = self._match_record(decision.ko_record.observation_time, records)
        if decided is None:
            raise PricingError("The knock-out decided at valuation has no record on the engine's schedule.")
        pv = float(self.price(product, pricing_env))
        all_records = [*pre_ko_records, *post_ko_records]
        unit = self._decided_ko_unit(pricing_env, all_records, decided)
        terminal = resolve_terminal_timing(product, pricing_env)
        pre_maturity = float(product.get_pre_maturity_time(pricing_env))
        observations, maturity_row, ki_settlement = self._reset_rows(
            product, pricing_env, pre_ko_records, post_ko_records, unit, terminal, maturity_unit=0.0,
            maturity_cash=0.0, ki_unit=0.0, ki_cash=0.0, pre_maturity=max(pre_maturity, 0.0),
            maturity=float(product.get_maturity(pricing_env)),
        )
        carried = bool(getattr(product, "_otc_lifecycle_knocked_in", False))
        rule = es.knock_in_rule(product, knocked_out_now=True, knocked_in_now=carried)
        return v1_statistics.statistics(
            product, pricing_env, family=self._statistics_family, pv=pv, observations=observations, coupons=None,
            maturity=maturity_row, ki_settlement=ki_settlement, rule=rule,
            want_knock_in=es.want_knock_in_path(streams), forward=None, unavailable={},
        )

    def _matured_at_valuation_statistics(self, product, pricing_env, pre_ko_records, post_ko_records, *, pv,
                                         pre_maturity, streams):
        """The first schedule ended at the valuation instant without a knock-in: the maturity is decided on the
        known spot (``price``'s rule), certain; the whole price is its cash."""
        from quantark.asset.equity.engine.pde.core.statistics import event_statistics as es
        from quantark.asset.equity.engine.quad import v1_statistics
        from quantark.asset.equity.settlement import CashflowKind, SettlementRequest, SettlementResolver

        timing = SettlementResolver.resolve_contingent(
            product,
            SettlementRequest(kind=CashflowKind.TERMINAL, determination_time=pre_maturity,
                              cashflow_id="ko_reset_v0_terminal"),
            pricing_env,
        )
        all_records = [*pre_ko_records, *post_ko_records]
        terminal = resolve_terminal_timing(product, pricing_env)
        observations, _, ki_settlement = self._reset_rows(
            product, pricing_env, pre_ko_records, post_ko_records, np.zeros(len(all_records)), terminal,
            maturity_unit=0.0, maturity_cash=0.0, ki_unit=0.0, ki_cash=0.0, pre_maturity=max(pre_maturity, 0.0),
            maturity=float(product.get_maturity(pricing_env)),
        )
        maturity_row = es.outcome_row(
            pricing_env, time=max(pre_maturity, 0.0), payment_time=float(timing.payment_time),
            date=None, payment_date=timing.payment_date, probability=1.0, discounted=pv,
        )
        rule = es.knock_in_rule(product, knocked_out_now=True, knocked_in_now=False)
        return v1_statistics.statistics(
            product, pricing_env, family=self._statistics_family, pv=pv, observations=observations, coupons=None,
            maturity=maturity_row, ki_settlement=ki_settlement, rule=rule,
            want_knock_in=es.want_knock_in_path(streams), forward=None, unavailable={},
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
        terminal_payment_time: Optional[float] = None
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
                terminal_payment_time = (
                    terminal_payment_time
                    if terminal_payment_time is not None
                    else resolve_terminal_timing(
                        product, pricing_env
                    ).payment_time
                )
                settlement_time = terminal_payment_time

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
