"""
Quadrature pricing engine for Snowball (autocallable) options.

Implements a direct two-state (knocked-in / not-knocked-in) quadrature recursion.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
from scipy.special import erfc

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.engine.event_stats import AutocallableEventStats
from quantark.asset.equity.engine.quad.quad_math import QuadratureMath
from quantark.asset.equity.param import QuadParams
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType
from quantark.util.enum.engine_enums import EngineType, KnockInMonitoringMode
from quantark.util.exceptions import NumericalError, PricingError, ValidationError
from quantark.util.numerical import (
    Tolerance,
    is_close,
    is_greater_than,
    is_less_than,
    is_zero,
    safe_exp,
    safe_log,
    validate_positive,
)

# Broadie-Glasserman-Kou (1997) discrete-monitoring barrier correction:
# beta = -zeta(1/2) / sqrt(2*pi). A barrier monitored discretely at spacing
# dt is approximated by a continuous barrier shifted by exp(±beta*vol*sqrt(dt)).
_BGK_BETA = 0.5825971579390107

# Validity gates for KnockInMonitoringMode.BGK_APPROXIMATION. The shift assumes
# regular monitoring of a constant barrier over the full pricing horizon at a
# single volatility; outside these bounds the engine refuses to approximate.
# Regularity is measured against the median spacing: most intervals must lie
# inside a band around it (business-day daily schedules in calendar time put
# ~80% of intervals exactly at the median, the rest are weekend gaps), and no
# interval may exceed the holiday-cluster cap. Alternating and mixed-frequency
# schedules fail the band; monthly-then-daily mixes fail the cap.
_BGK_REGULAR_BAND = (0.5, 1.5)
_BGK_MIN_REGULAR_FRACTION = 0.70
_BGK_MAX_SPACING_RATIO = 9.0  # max spacing vs median spacing
# The schedule must start within this many median spacings of valuation and
# end within this many median spacings of maturity: continuous monitoring
# covers the gaps, so a wider unmonitored window materially overstates the
# knock-in probability. 3x tolerates a weekend at either boundary.
_BGK_MAX_EDGE_GAP_RATIO = 3.0
# Maximum relative deviation of the surface vol sampled along the schedule
# (at the KI barrier) from the single pricing vol used for the shift.
_BGK_MAX_VOL_DISPERSION = 0.25


class SnowballQuadEngine(BaseEngine):
    """
    Quadrature pricing engine for Snowball (autocallable) options.

    Uses a single backward quadrature recursion for two regimes:
    - V_in: knock-in already occurred
    - V_out: knock-in not yet occurred
    """

    engine_type = EngineType.QUADRATURE
    supports_spot_greeks_grid = True

    def __init__(self, params: Optional[QuadParams] = None) -> None:
        if params is None:
            params = QuadParams()
        if not isinstance(params, QuadParams):
            raise ValidationError(
                f"params must be QuadParams instance, got {type(params).__name__}"
            )
        super().__init__(params)
        # Opt-in capture of the per-observation backward-induction value surfaces
        # (the v_in / v_out continuation grids on the inception-anchored spot grid).
        # Off by default so normal pricing pays no memory cost; the CVA exposure
        # layer enables it to read a per-node x per-state value surface straight off
        # the backward sweep, with no re-pricing. Maps observation_time ->
        # (spot_grid, v_in, v_out), plus key 0.0 for the valuation-date surfaces.
        self.record_backward_grids = False
        self._backward_grids: dict = {}

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

        # Reset the opt-in backward-grid capture up front, BEFORE any early return
        # (zero maturity / immediate KO). Otherwise a terminated bumped re-price would
        # leave the previous trade's surfaces in _backward_grids for the CVA exposure
        # layer to read as if live — a silent corruption of bumped sensitivities.
        record_grids = getattr(self, "record_backward_grids", False)
        if record_grids:
            self._backward_grids = {}

        spot = pricing_env.spot
        maturity = product.get_maturity(pricing_env)
        validate_positive(spot, "spot")
        validate_positive(maturity, "maturity", allow_zero=True)
        if is_zero(maturity, tol=Tolerance.ZERO):
            knocked_in = bool(getattr(product, "_otc_lifecycle_knocked_in", False))
            return product.get_payoff(spot, pricing_env, knocked_in=knocked_in)

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
        ki_barrier_continuous: Optional[float] = None
        if ki_continuous:
            ki_barrier_continuous = self._continuous_ki_barrier_value(product)
        if product.has_ki_barrier and not ki_continuous:
            ki_records = product.resolve_ki_observations(pricing_env)
            if not ki_records:
                raise PricingError("KI observation schedule is empty for SnowballQuadEngine.")

        ko_record_0 = self._match_record(0.0, ko_records)
        if self._ko_record_triggers_at_spot(product, spot, ko_record_0):
            return self._get_immediate_ko_payoff(ko_record_0, pricing_env)

        # Valuation-time KI state follows the contractual monitoring semantics,
        # so it must be determined before any BGK schedule conversion.
        knocked_in_at_valuation = self._is_knocked_in_at_valuation(
            product,
            spot,
            pricing_env,
            ki_continuous=ki_continuous,
            ki_records=ki_records,
        )

        if ki_records and (
            self._ki_monitoring_mode() is KnockInMonitoringMode.BGK_APPROXIMATION
        ):
            ki_barrier_continuous = self._bgk_equivalent_continuous_ki_barrier(
                product, ki_records, pricing_env, vol, maturity
            )
            ki_continuous = True
            ki_records = []

        times = self._merge_times(
            [rec.observation_time for rec in ko_records],
            [rec.observation_time for rec in ki_records],
            maturity,
        )
        if not times:
            raise PricingError("Observation time grid is empty for SnowballQuadEngine.")

        # When the CVA exposure layer is recording surfaces, also diffuse through (and
        # record at) delayed KO-settlement times so a still-alive path can be valued at
        # the settlement node. These are pure recording nodes (no KO/KI event matches
        # them), so the recursion records the bare continuation surface there. Gated on
        # record_grids: normal pricing keeps its exact observation-only time grid.
        if record_grids:
            times = self._insert_settlement_times(times, ko_records, maturity)

        align_log = self._select_alignment_log(
            spot, product, ki_barrier_override=ki_barrier_continuous
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
            log_ki_barrier = safe_log(ki_barrier_continuous / spot)

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
                    spot_grid >= ki_barrier_continuous
                    if product.is_reverse
                    else spot_grid <= ki_barrier_continuous
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

            # Post-event continuation surfaces AT obs_time (before diffusing back to
            # the previous step): v_out = value of a not-yet-knocked-in contract,
            # v_in = value once knock-in has occurred, both as a function of spot on
            # spot_grid. These are exactly the per-(t, state) value slices the CVA
            # exposure layer needs; the diffusion below rolls them to the prior step.
            if record_grids:
                self._backward_grids[float(obs_time)] = (
                    spot_grid.copy(), v_in.copy(), v_out.copy())

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

        # After the full backward sweep, v_in/v_out are the valuation-date (t=0)
        # surfaces; record them so the exposure grid has a t0 slice for both states.
        if record_grids:
            self._backward_grids[0.0] = (
                spot_grid.copy(), v_in.copy(), v_out.copy())

        value_surface = v_in if knocked_in_at_valuation else v_out
        self._last_spot_greeks_grid = (spot_grid.copy(), value_surface.copy())
        return math_utils.interpolate(value_surface, x=0.0)

    def calculate_event_stats(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> Optional[AutocallableEventStats]:
        """Provide per-observation KO probabilities and expected discounted cashflows.

        Runs a single quadrature recursion over stacked indicator surfaces; usually
        much faster than MC for risk-neutral event stats.
        """
        if not isinstance(product, self._event_stats_product_type()):
            return None
        if pricing_env is None:
            raise PricingError("PricingEnvironment is required for SnowballQuadEngine.")
        return self._compute_event_stats(product, pricing_env)

    def _event_stats_product_type(self) -> type:
        """Product type accepted by ``calculate_event_stats`` (overridable)."""
        return SnowballOption

    def _make_event_stats(self, **fields) -> AutocallableEventStats:
        """Construct the event-stats dataclass (overridable by subclasses)."""
        return AutocallableEventStats(**fields)

    def _event_stats_alignment_log(
        self, spot: float, product, ki_barrier_override: Optional[float] = None
    ) -> Optional[float]:
        """Grid-alignment log used by event stats.

        Calls the Snowball implementation explicitly so a subclass that overrides
        the price-path ``_select_alignment_log`` (e.g. PhoenixQuadEngine, with a
        different signature) does not intercept the event-stats recursion.
        """
        return SnowballQuadEngine._select_alignment_log(
            self, spot, product, ki_barrier_override=ki_barrier_override
        )

    # --- Extra indicator-row hooks (overridden by Phoenix for coupons) ---

    def _n_extra_quad_rows(self, n_ko: int) -> int:
        """Extra stacked indicator rows beyond the ``n_ko`` KO rows."""
        return 0

    def _set_extra_quad_indicators(
        self, v_in, v_out, spot_grid, n_ko, ko_index, ko_record, obs_time,
        rate, product, disable_ko_after_ki,
    ) -> None:
        """Set extra indicator rows at a KO observation (no-op for Snowball)."""
        return None

    def _extract_extra_quad_stats(
        self, initial_surface, math_utils, n_ko, ko_records, rate
    ) -> dict:
        """Extra event-stats fields from the extra rows (none for Snowball)."""
        return {}

    def _compute_event_stats(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> Optional[AutocallableEventStats]:
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
        ki_barrier_continuous: Optional[float] = None
        if ki_continuous:
            ki_barrier_continuous = self._continuous_ki_barrier_value(product)
        if product.has_ki_barrier and not ki_continuous:
            ki_records = product.resolve_ki_observations(pricing_env)

        ko_record_0 = self._match_record(0.0, ko_records)
        # Valuation-time KI state follows the contractual monitoring semantics,
        # so it must be determined before any BGK schedule conversion.
        knocked_in_at_valuation = self._is_knocked_in_at_valuation(
            product,
            spot,
            pricing_env,
            ki_continuous=ki_continuous,
            ki_records=ki_records,
        )

        if self._ko_record_triggers_at_spot(product, spot, ko_record_0):
            return self._build_immediate_ko_event_stats(
                product,
                pricing_env,
                ko_records,
                ko_record_0,
                knocked_in_at_valuation,
            )

        if ki_records and (
            self._ki_monitoring_mode() is KnockInMonitoringMode.BGK_APPROXIMATION
        ):
            ki_barrier_continuous = self._bgk_equivalent_continuous_ki_barrier(
                product, ki_records, pricing_env, vol, maturity
            )
            ki_continuous = True
            ki_records = []

        times = self._merge_times(
            [rec.observation_time for rec in ko_records],
            [rec.observation_time for rec in ki_records],
            maturity,
        )
        dt = self._build_dt(times)

        align_log = self._event_stats_alignment_log(
            spot, product, ki_barrier_override=ki_barrier_continuous
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
        # Rows [0..n_ko-1] are KO indicators; subclasses may append extra rows
        # (e.g. Phoenix coupon-trigger rows) that ride the same diffusion/jumps.
        n_ko = len(ko_records)
        n_rows = n_ko + self._n_extra_quad_rows(n_ko)
        v_in = np.zeros((n_rows, grid.size), dtype=float)
        v_out = np.zeros((n_rows, grid.size), dtype=float)

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
                # Extra rows (coupon) are set AFTER the KO zeroing so a coupon at a
                # simultaneous KO is retained; future-coupon rows stay zeroed by KO.
                self._set_extra_quad_indicators(
                    v_in, v_out, spot_grid, n_ko, int(ko_index), ko_record,
                    obs_time, rate, product, disable_ko_after_ki,
                )

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
                log_ki_barrier = safe_log(ki_barrier_continuous / spot)
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
        ed_unit = np.array(
            [math_utils.interpolate(initial_surface[i], x=0.0) for i in range(n_ko)],
            dtype=float,
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
        ki_times = np.array([], dtype=float)
        ki_event_probability = np.array([], dtype=float)
        ki_survival_probability = np.array([], dtype=float)
        if knocked_in_at_valuation:
            ki_probability = 1.0
            ki_times = np.array([0.0], dtype=float)
            ki_event_probability = np.array([1.0], dtype=float)
            ki_survival_probability = np.array([0.0], dtype=float)
        elif product.has_ki_barrier:
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
                    log_ki_barrier = safe_log(ki_barrier_continuous / spot)
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

        extra_fields = self._extract_extra_quad_stats(
            initial_surface, math_utils, n_ko, ko_records, rate
        )

        return self._make_event_stats(
            pv=pv,
            ko_times=ko_times,
            ko_probability=ko_prob,
            survival_probability=survival_prob,
            expected_discounted_ko_cashflow=ed_ko_cf,
            ki_probability=ki_probability,
            expected_discounted_maturity_cashflow=expected_discounted_maturity_cf,
            reconciliation_error=0.0,
            ki_times=ki_times,
            ki_event_probability=ki_event_probability,
            ki_survival_probability=ki_survival_probability,
            **extra_fields,
        )

    def _validate_product(self, product: SnowballOption) -> None:
        if product.barrier_config.ko_observation_type != ObservationType.DISCRETE:
            raise PricingError("SnowballQuadEngine requires discrete KO monitoring.")
        # Airbag and call-rebate features are handled via product payoff functions.

    def _resolve_grid_points(
        self, maturity: float, vol: float, times: Sequence[float]
    ) -> int:
        """Refine the spatial grid until the shortest diffusion step is resolved."""
        requested = int(self.params.grid_points)
        min_cells = float(getattr(self.params, "min_diffusion_stddev_cells", 0.0) or 0.0)
        if min_cells <= 0.0 or not times:
            return requested

        times_full = np.concatenate(([0.0], np.asarray(times, dtype=float)))
        positive_dt = np.diff(times_full)
        positive_dt = positive_dt[positive_dt > Tolerance.ZERO]
        if positive_dt.size == 0:
            return requested

        min_diffusion_stddev = vol * math.sqrt(float(np.min(positive_dt)))
        log_half_width = (
            float(self.params.num_std_devs) * vol * math.sqrt(maturity)
            + (1.0 + 0.5 * vol * vol) * maturity
        )
        required = int(
            math.ceil(1.0 + 2.0 * log_half_width * min_cells / min_diffusion_stddev)
        )
        if required % 2 == 0:
            required += 1
        max_adaptive = int(getattr(self.params, "max_adaptive_grid_points", 5001))
        if required > max_adaptive and requested < required:
            raise NumericalError(
                "Observation intervals require at least "
                f"{required} quadrature grid points to resolve the shortest diffusion "
                f"step, exceeding max_adaptive_grid_points={max_adaptive}. Increase "
                "max_adaptive_grid_points or use a different pricing engine."
            )
        return max(requested, required)

    def _ki_monitoring_mode(self) -> KnockInMonitoringMode:
        mode = getattr(
            self.params, "ki_monitoring_mode", KnockInMonitoringMode.EXACT_DISCRETE
        )
        if isinstance(mode, str):
            mode = KnockInMonitoringMode(mode.lower())
        return mode

    def _continuous_ki_barrier_value(self, product: SnowballOption) -> float:
        """Validated scalar KI barrier for continuous-monitoring pricing."""
        ki_barrier = product.barrier_config.ki_barrier
        if ki_barrier is None:
            raise PricingError("KI barrier configuration is missing.")
        if isinstance(ki_barrier, list):
            raise PricingError("Continuous KI requires scalar ki_barrier.")
        return float(ki_barrier)

    def _bgk_equivalent_continuous_ki_barrier(
        self,
        product: SnowballOption,
        ki_records: Sequence,
        pricing_env: PricingEnvironment,
        vol: float,
        maturity: float,
    ) -> float:
        """
        Continuous-equivalent KI barrier for a regular discrete schedule.

        A down-in barrier B monitored at spacing dt is approximated by a
        continuous barrier at B * exp(-beta * vol * sqrt(dt)); an up-in
        (reverse) barrier shifts by the reciprocal factor
        (Broadie-Glasserman-Kou 1997). The shift is first-order accurate and
        only sound for approximately regular schedules with a constant
        resolved barrier covering the full pricing horizon under stable
        volatility; this method validates those conditions and raises
        ValidationError when the schedule does not qualify.
        """
        barrier = self._bgk_constant_resolved_barrier(ki_records)

        obs_times = np.unique(
            np.asarray([rec.observation_time for rec in ki_records], dtype=float)
        )
        spacings = np.diff(obs_times)
        spacings = spacings[spacings > Tolerance.ZERO]
        if spacings.size == 0:
            raise ValidationError(
                "BGK_APPROXIMATION requires at least two distinct KI observation "
                "dates; use KnockInMonitoringMode.EXACT_DISCRETE."
            )

        min_count = int(getattr(self.params, "bgk_min_ki_observations", 100))
        if obs_times.size < min_count:
            raise ValidationError(
                "BGK_APPROXIMATION is a performance mode for dense KI schedules: "
                f"got {obs_times.size} observation dates, fewer than "
                f"bgk_min_ki_observations={min_count}; use "
                "KnockInMonitoringMode.EXACT_DISCRETE or lower "
                "bgk_min_ki_observations."
            )

        # Documented boundary values (e.g. an exact 3-day weekend gap at 3x a
        # 1-day median) must pass despite last-ulp float noise, so every gate
        # comparison is widened by Tolerance.PRECISION.
        median_dt = float(np.median(spacings))
        band_lo, band_hi = _BGK_REGULAR_BAND
        in_band_fraction = float(
            np.mean(
                (spacings >= band_lo * median_dt - Tolerance.PRECISION)
                & (spacings <= band_hi * median_dt + Tolerance.PRECISION)
            )
        )
        max_dt = float(np.max(spacings))
        max_spacing_cap = _BGK_MAX_SPACING_RATIO * median_dt
        if is_less_than(
            in_band_fraction,
            _BGK_MIN_REGULAR_FRACTION,
            abs_tol=Tolerance.PRECISION,
        ) or is_greater_than(
            max_dt, max_spacing_cap, abs_tol=Tolerance.PRECISION
        ):
            raise ValidationError(
                "BGK_APPROXIMATION requires approximately regular KI observation "
                f"intervals: {in_band_fraction:.0%} of intervals lie within "
                f"[{band_lo:g}, {band_hi:g}]x the median spacing {median_dt:.6g} "
                f"(need >= {_BGK_MIN_REGULAR_FRACTION:.0%}) and the max spacing "
                f"{max_dt:.6g} must not exceed {_BGK_MAX_SPACING_RATIO:g}x the "
                "median; use KnockInMonitoringMode.EXACT_DISCRETE."
            )

        first_gap = float(obs_times[0])
        last_gap = float(maturity) - float(obs_times[-1])
        edge_tol = _BGK_MAX_EDGE_GAP_RATIO * median_dt
        first_exceeds = is_greater_than(
            first_gap, edge_tol, abs_tol=Tolerance.PRECISION
        )
        last_exceeds = is_greater_than(
            last_gap, edge_tol, abs_tol=Tolerance.PRECISION
        )
        if first_exceeds or last_exceeds:
            raise ValidationError(
                "BGK_APPROXIMATION requires KI monitoring to cover the full "
                f"pricing horizon: schedule spans [{float(obs_times[0]):.6g}, "
                f"{float(obs_times[-1]):.6g}] of [0, {float(maturity):.6g}]; use "
                "KnockInMonitoringMode.EXACT_DISCRETE."
            )

        self._validate_bgk_vol_stability(pricing_env, barrier, obs_times, vol)

        mean_dt = float(np.mean(spacings))
        shift = _BGK_BETA * vol * math.sqrt(mean_dt)
        return barrier * safe_exp(shift if product.is_reverse else -shift)

    def _bgk_constant_resolved_barrier(self, ki_records: Sequence) -> float:
        """Resolved KI barrier, validated constant across all observations."""
        barriers = [rec.barrier for rec in ki_records]
        if any(b is None for b in barriers):
            raise ValidationError(
                "BGK_APPROXIMATION requires every KI observation to resolve a "
                "barrier level; use KnockInMonitoringMode.EXACT_DISCRETE."
            )
        reference = float(barriers[0])
        for b in barriers[1:]:
            if not is_close(float(b), reference, abs_tol=Tolerance.PRECISION):
                raise ValidationError(
                    "BGK_APPROXIMATION requires a constant resolved KI barrier, "
                    f"got per-observation levels {reference:.6g} and {float(b):.6g}; "
                    "use KnockInMonitoringMode.EXACT_DISCRETE."
                )
        return reference

    def _validate_bgk_vol_stability(
        self,
        pricing_env: PricingEnvironment,
        barrier: float,
        obs_times: np.ndarray,
        vol: float,
    ) -> None:
        """Require the surface vol along the schedule to stay near the pricing vol."""
        sample_times = obs_times
        if sample_times.size > 16:
            indices = np.linspace(0, sample_times.size - 1, 16).astype(int)
            sample_times = sample_times[indices]
        for t in sample_times:
            sampled = float(pricing_env.get_vol(barrier, float(t)))
            if not np.isfinite(sampled) or sampled <= 0.0:
                raise ValidationError(
                    f"BGK_APPROXIMATION sampled an invalid volatility {sampled} at "
                    f"t={float(t):.6g}; use KnockInMonitoringMode.EXACT_DISCRETE."
                )
            if abs(sampled - vol) > _BGK_MAX_VOL_DISPERSION * vol:
                raise ValidationError(
                    "BGK_APPROXIMATION requires stable volatility over the KI "
                    f"schedule: vol {sampled:.6g} at t={float(t):.6g} deviates more "
                    f"than {_BGK_MAX_VOL_DISPERSION:.0%} from pricing vol "
                    f"{vol:.6g}; use KnockInMonitoringMode.EXACT_DISCRETE."
                )

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

        u_array = math_utils.simpson_weights_many(values, p_lr, p_ur, p0)
        conv = math_utils.convolution_fft_many(omega_array, u_array)
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
        correction = np.zeros_like(base, dtype=float)
        band = self._bridge_band(tau_step, math_utils.h, grid.size)
        denom = vol * vol * dt

        for offset in range(-band, band + 1):
            if offset >= 0:
                source = slice(0, grid.size - offset)
                target = slice(offset, grid.size)
            else:
                source = slice(-offset, grid.size)
                target = slice(0, grid.size + offset)

            source_grid = grid[source]
            target_grid = grid[target]
            z = float(offset) * math_utils.h
            omega = math.exp(-(z**2) / (4.0 * tau_step) - alpha * z)
            w = weights[source]

            if is_reverse:
                d0 = log_barrier - target_grid
                d1 = log_barrier - source_grid
            else:
                d0 = target_grid - log_barrier
                d1 = source_grid - log_barrier

            safe = (d0 > 0.0) & (d1 > 0.0)
            exponent = np.where(safe, -2.0 * d0 * d1 / denom, 0.0)
            exponent = np.clip(exponent, -745.0, 0.0)
            p_hit = np.where(safe, np.exp(exponent), 1.0)
            kernel = omega * w * p_hit
            if delta.ndim == 1:
                correction[target] += prefactor * kernel * delta[source]
            else:
                correction[:, target] += prefactor * kernel.reshape(1, -1) * delta[
                    :, source
                ]

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

    def _insert_settlement_times(
        self, times: Sequence[float], ko_records: Sequence, maturity: float
    ) -> list[float]:
        """Merge delayed KO-settlement times (within (obs, maturity]) into ``times``.

        Used only when recording backward grids for CVA exposure: a KO that settles
        after its observation creates a pending receivable whose default window ends at
        settlement, so the exposure grid needs a node there, and a still-alive path
        needs a continuation surface at that node. Settlements beyond maturity are not
        added here (no diffusion exists past the terminal node); the exposure builder
        rejects those rather than approximating.
        """
        merged = list(times)
        for rec in ko_records:
            st = rec.settlement_time
            if st is None:
                continue
            st = float(st)
            if st <= float(rec.observation_time) + Tolerance.PRECISION:
                continue  # immediate settlement: already an observation node
            if is_greater_than(st, maturity, abs_tol=Tolerance.PRECISION):
                continue  # post-maturity tail: handled (rejected) by the exposure layer
            st = min(st, float(maturity))
            if not any(is_close(st, t, abs_tol=Tolerance.PRECISION) for t in merged):
                merged.append(st)
        return sorted(merged)

    def _merge_times(
        self, ko_times: Sequence[float], ki_times: Sequence[float], maturity: float
    ) -> list[float]:
        merged = []
        for t in sorted(list(ko_times) + list(ki_times)):
            t = float(t)
            if t <= Tolerance.ZERO:
                continue
            if is_greater_than(t, maturity, abs_tol=Tolerance.PRECISION):
                continue
            if is_close(t, maturity, abs_tol=Tolerance.PRECISION):
                t = float(maturity)
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

    def _ko_record_triggers_at_spot(
        self,
        product: SnowballOption,
        spot: float,
        ko_record: Optional[object],
    ) -> bool:
        """Return True when a KO record is triggered at the valuation spot."""
        if ko_record is None:
            return False
        if ko_record.barrier is None:
            raise ValidationError("KO observation at valuation requires a barrier level.")
        if product.is_reverse:
            return spot <= float(ko_record.barrier)
        return spot >= float(ko_record.barrier)

    def _get_immediate_ko_payoff(
        self,
        ko_record: object,
        pricing_env: PricingEnvironment,
    ) -> float:
        """Return discounted KO payoff for a KO triggered at valuation."""
        payoff = float(ko_record.payoff) if ko_record.payoff is not None else 0.0
        settlement_time = ko_record.settlement_time
        if settlement_time is not None and float(settlement_time) > 0.0:
            payoff *= float(pricing_env.get_discount_factor(float(settlement_time)))
        return payoff

    def _is_already_knocked_in(self, product: SnowballOption, spot: float) -> bool:
        """Check whether valuation spot is already in the KI region."""
        if not product.has_ki_barrier:
            return False

        ki_barrier = product.barrier_config.ki_barrier
        if isinstance(ki_barrier, list):
            ki_barrier = ki_barrier[0]

        if product.is_reverse:
            return spot >= float(ki_barrier)
        return spot <= float(ki_barrier)

    def _is_knocked_in_at_valuation(
        self,
        product: SnowballOption,
        spot: float,
        pricing_env: PricingEnvironment,
        ki_continuous: bool,
        ki_records: Sequence = (),
    ) -> bool:
        """Determine KI state at valuation before the Quad recursion starts."""
        if getattr(product, "_otc_lifecycle_knocked_in", False):
            return True
        if not product.has_ki_barrier:
            return False
        if ki_continuous:
            return self._is_already_knocked_in(product, spot)

        if not ki_records:
            ki_records = product.resolve_ki_observations(pricing_env)
        ki_record_0 = self._match_record(0.0, ki_records)
        if ki_record_0 is None:
            return False
        if ki_record_0.barrier is None:
            raise ValidationError("KI observation at valuation requires a barrier level.")
        if product.is_reverse:
            return spot >= float(ki_record_0.barrier)
        return spot <= float(ki_record_0.barrier)

    def _build_immediate_ko_event_stats(
        self,
        product: SnowballOption,
        pricing_env: PricingEnvironment,
        ko_records: Sequence,
        ko_record_0: object,
        knocked_in_at_valuation: bool,
    ) -> AutocallableEventStats:
        """Build deterministic event stats when KO is triggered at valuation."""
        ko_times = np.array([float(rec.observation_time) for rec in ko_records], dtype=float)
        ko_probability = np.zeros(len(ko_records), dtype=float)
        expected_discounted_ko_cashflow = np.zeros(len(ko_records), dtype=float)
        ko_index = 0
        for idx, rec in enumerate(ko_records):
            if rec is ko_record_0 or is_close(
                rec.observation_time,
                ko_record_0.observation_time,
                abs_tol=Tolerance.PRECISION,
            ):
                ko_index = idx
                break

        pv = self._get_immediate_ko_payoff(ko_record_0, pricing_env)
        ko_probability[ko_index] = 1.0
        expected_discounted_ko_cashflow[ko_index] = pv

        survival_probability = np.ones(len(ko_records), dtype=float)
        cumulative = 0.0
        for idx in range(len(ko_records)):
            cumulative += ko_probability[idx]
            survival_probability[idx] = max(0.0, 1.0 - cumulative)

        ki_times = np.array([], dtype=float)
        ki_event_probability = np.array([], dtype=float)
        ki_survival_probability = np.array([], dtype=float)
        ki_probability = 0.0
        if knocked_in_at_valuation:
            ki_probability = 1.0
            ki_times = np.array([0.0], dtype=float)
            ki_event_probability = np.array([1.0], dtype=float)
            ki_survival_probability = np.array([0.0], dtype=float)

        return self._make_event_stats(
            pv=pv,
            ko_times=ko_times,
            ko_probability=ko_probability,
            survival_probability=survival_probability,
            expected_discounted_ko_cashflow=expected_discounted_ko_cashflow,
            ki_probability=ki_probability,
            expected_discounted_maturity_cashflow=0.0,
            reconciliation_error=0.0,
            ki_times=ki_times,
            ki_event_probability=ki_event_probability,
            ki_survival_probability=ki_survival_probability,
        )

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
        self,
        spot: float,
        product: SnowballOption,
        ki_barrier_override: Optional[float] = None,
    ) -> Optional[float]:
        ko_candidates: list[float] = []
        ki_candidates: list[float] = []

        ko_barrier = product.barrier_config.ko_barrier
        if isinstance(ko_barrier, list):
            ko_candidates.extend([float(b) for b in ko_barrier if b > 0])
        elif ko_barrier is not None and ko_barrier > 0:
            ko_candidates.append(float(ko_barrier))

        if ki_barrier_override is not None:
            # The model's operative KI barrier (e.g. BGK-shifted) takes
            # precedence over the contractual level for grid alignment.
            if ki_barrier_override > 0:
                ki_candidates.append(float(ki_barrier_override))
        elif product.has_ki_barrier:
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
