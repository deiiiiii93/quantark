"""
Quadrature pricing engine for Snowball (autocallable) options.

Implements a direct two-state (knocked-in / not-knocked-in) quadrature recursion.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import numpy as np
from scipy.special import erfc

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.engine.settlement_support import validate_settlement_capability
from quantark.asset.equity.engine.capabilities import SettlementSupport
from quantark.asset.equity.engine.event_stats import EventStatistics
from quantark.asset.equity.engine.pde.grid.events import (
    breach_fractions,
    project_event_values,
)
from quantark.asset.equity.engine.quad.quad_math import QuadratureMath
from quantark.asset.equity.engine.quad.quad_adapters import (
    resolve_structured_payment_timings,
)
from quantark.asset.equity.engine.quad.term_inputs import build_quad_term_params
from quantark.priceenv.term_sampling import make_df_fn
from quantark.asset.equity.param import QuadParams
from quantark.asset.equity.param.engine_params import QUAD_READOUT_MODES
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType
from quantark.util.enum.engine_enums import (
    EngineType,
    EventProjectionMode,
    KnockInMonitoringMode,
)
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

if TYPE_CHECKING:
    from quantark.cashleg.folding import FoldOverlay

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
    settlement_support = SettlementSupport.EVENT_AND_TERMINAL
    supports_spot_greeks_grid = True
    supports_cash_leg_folding = True

    #: Per-solve artefacts a folded solve must not leave behind (they describe the plain price).
    _FOLD_TRANSIENT_ATTRS: Tuple[str, ...] = (
        "_last_spot_greeks_grid",
        "_backward_grids",
        "_last_ignored_ko_observation_indices",
        "_last_convergence_info",
    )

    def fold_drilldown_refusal(self, product, pricing_env) -> Optional[str]:
        """``auto_converge`` refines the grid per solve: a folded and a plain solve may settle on
        different grids, so per-leg differences would mix them (ship spec D11c)."""
        if self.params.auto_converge:
            return "auto_converge settles each solve on its own grid"
        return None

    #: Readout modes this engine actually implements. A subclass with its own
    #: readout path narrows this rather than silently reporting a
    #: legacy-readout price as though it were something else.
    supported_readouts: Tuple[str, ...] = QUAD_READOUT_MODES

    def __init__(self, params: Optional[QuadParams] = None) -> None:
        if params is None:
            params = QuadParams()
        if not isinstance(params, QuadParams):
            raise ValidationError(
                f"params must be QuadParams instance, got {type(params).__name__}"
            )
        if params.readout not in self.supported_readouts:
            raise NotImplementedError(
                f"{type(self).__name__} does not implement "
                f"readout={params.readout!r}; it supports "
                f"{self.supported_readouts}."
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
        # Bridge-kernel cache: the continuous-KI bridge correction rebuilds an
        # identical per-offset kernel stack (omega * w * p_hit) on every call,
        # and one price_with_events walks the same time steps up to three times
        # (stacked KO-indicator recursion, KI-probability recursion, internal
        # price pass). The kernel is row-independent and fully determined by
        # the cache key (grid geometry, band, tau_step, alpha, vol^2*dt,
        # barrier, direction), so reuse is bit-identical. Bounded by
        # _BRIDGE_KERNEL_CACHE_MAX_ENTRIES (clear-all on overflow): a flat env
        # needs only a handful of entries; a term-structure env may produce one
        # per step, where eviction degrades gracefully to the uncached cost.
        self._bridge_kernel_cache: dict = {}
        # Mass-conservation diagnostic of the last forward-density event-stats
        # run (terminal density mass + absorbed KO mass); observational only.
        self._last_forward_mass_diagnostic: Optional[float] = None

    def price(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        if self.params.auto_converge:
            return self._price_with_convergence(product, pricing_env)
        return self._price_once(product, pricing_env)

    def __getstate__(self):
        """Pickle without the last statistics march's adjoint log (``_last_v1_march``): a per-solve artefact whose
        event maps are closures over the solve (a process pool ships engines between workers)."""
        state = dict(self.__dict__)
        state.pop("_last_v1_march", None)
        return state

    def price_folded(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        overlay: "FoldOverlay",
        *,
        lifecycle_state=None,
    ) -> float:
        """Per-unit NPV with ``overlay``'s cash folded into the product's own payoff (ship spec D4).

        Knock-out cash is added to each knock-out record's payoff and maturity cash to each
        branch's redemption, before the engine's own settlement-delay factors, on the same
        lattice; the valuation-instant short-circuits pay the decided record's cash too. An
        empty overlay is ``price()`` bitwise. The solve leaves the plain solve's artefacts
        (spot-greeks grid, recorded backward grids) untouched.
        """
        if overlay.is_empty:
            return self.price(product, pricing_env, lifecycle_state=lifecycle_state)
        validate_settlement_capability(self, product, lifecycle_state)
        saved = {
            name: self.__dict__[name]
            for name in self._FOLD_TRANSIENT_ATTRS
            if name in self.__dict__
        }
        try:
            return float(self._folded_solve(product, pricing_env, overlay))
        finally:
            for name in self._FOLD_TRANSIENT_ATTRS:
                self.__dict__.pop(name, None)
            self.__dict__.update(saved)

    def _folded_solve(self, product, pricing_env, fold: "FoldOverlay") -> float:
        """The solve ``price`` runs, with ``fold`` threaded through."""
        if self.params.auto_converge:
            return self._price_with_convergence(product, pricing_env, fold=fold)
        return self._price_once(product, pricing_env, fold=fold)

    @staticmethod
    def _refuse_expired_fold(fold) -> None:
        """Expired, the fold structure is trivial and leg overlays are empty: nothing can land."""
        if fold is not None:
            from quantark.execution.errors import CapabilityError

            raise CapabilityError(
                "an expired autocallable has no knock-out or maturity event left to carry "
                "folded cash; its cash legs are constants on the event route"
            )

    @staticmethod
    def _fold_in_life(records: Sequence, maturity: float) -> list:
        """The knock-out records a solve can reach: from the valuation instant to maturity."""
        return [
            rec
            for rec in records
            if rec.observation_time is not None
            and not is_less_than(float(rec.observation_time), 0.0, rel_tol=0.0, abs_tol=0.0)
            and not is_greater_than(
                float(rec.observation_time), maturity, rel_tol=0.0, abs_tol=Tolerance.PRECISION
            )
        ]

    @staticmethod
    def _bind_fold_ko(fold, tagged_records: Sequence[tuple]) -> dict:
        """``{id(record): amount}`` for ``(schedule, record)`` pairs, bound one to one by the overlay."""
        amounts = fold.bind_ko(
            [
                (
                    tag,
                    float(rec.observation_time),
                    float(
                        rec.settlement_time
                        if rec.settlement_time is not None
                        else rec.observation_time
                    ),
                )
                for tag, rec in tagged_records
            ]
        )
        return {
            id(rec): float(amount)
            for (_, rec), amount in zip(tagged_records, amounts)
        }

    @staticmethod
    def _folded_amount(fold_ko: dict, record) -> float:
        """The bound amount of a record a short-circuit decided; one outside the bound set refuses."""
        if record is None or id(record) not in fold_ko:
            raise ValidationError(
                "the knock-out decided at the valuation instant is not one of the records the "
                "folded cash was bound to"
            )
        return fold_ko[id(record)]

    def _price_once(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        *,
        fold=None,
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
        self._require_resolvable_time(maturity)
        if maturity <= 0.0:
            self._refuse_expired_fold(fold)
            knocked_in = bool(getattr(product, "_otc_lifecycle_knocked_in", False))
            if type(product) is SnowballOption:
                # The observations still pending at this instant are decided on the known spot, as the
                # backward induction decides them when more of the claim lies ahead. Going straight to
                # the carried state's payoff priced a claim that knocks in or out at the last
                # observation as if it had survived.
                decision = product.decide_observations_at_valuation(
                    spot, pricing_env, knocked_in=knocked_in
                )
                if decision.knocked_out:
                    record = decision.ko_record
                    settlement = record.settlement_time
                    discount = (
                        pricing_env.get_discount_factor(settlement)
                        if settlement is not None and settlement > 0.0
                        else 1.0
                    )
                    return float(record.payoff) * float(discount)
                knocked_in = decision.knocked_in
            from quantark.asset.equity.engine.settlement_support import resolve_terminal_timing

            # the redemption is cash at its payment time, which a settlement lag puts after this instant
            redemption = product.get_payoff(spot, pricing_env, knocked_in=knocked_in)
            return float(redemption) * float(resolve_terminal_timing(product, pricing_env).payment_df)

        rate = pricing_env.get_rate(maturity)
        div = pricing_env.get_div_yield(maturity)
        vol = pricing_env.get_vol(product.strike, maturity)
        validate_positive(vol, "volatility")
        if not np.isfinite(div):
            raise ValidationError(f"Dividend yield must be finite, got {div}")
        if vol > 5.0:
            raise ValidationError(f"Volatility too high for quadrature stability: {vol}")

        # An empty side (spec 2026-10-07 theta boundary: aged past its last date) folds away:
        # no knock-out records = no autocall before maturity; no knock-in records = no knock-in.
        ko_records = product.resolve_ko_observations(pricing_env)
        payment_timings = resolve_structured_payment_timings(
            product, pricing_env, ko_records
        )
        event_delay_by_record = {
            id(record): float(delay_df)
            for record, delay_df in zip(
                ko_records, payment_timings.event_delay_dfs
            )
        }

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

        fold_ko: Optional[dict] = None
        if fold is not None:
            fold_ko = self._bind_fold_ko(
                fold, [("main", rec) for rec in self._fold_in_life(ko_records, maturity)]
            )

        ko_record_0 = self._match_record(0.0, ko_records)
        if self._ko_record_triggers_at_spot(product, spot, ko_record_0):
            if fold_ko is not None:
                return self._get_immediate_ko_payoff(
                    ko_record_0, pricing_env, extra_cash=self._folded_amount(fold_ko, ko_record_0)
                )
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
        if ki_continuous:
            from quantark.asset.equity.engine.continuous_ki_support import continuous_time_grid
            times, self._continuous_ki_model = continuous_time_grid(pricing_env, times, self.params)
            times = self._merge_times(times, (), maturity)
        if not times:
            raise PricingError("Observation time grid is empty for SnowballQuadEngine.")

        # When the CVA exposure layer is recording surfaces, also diffuse through (and
        # record at) delayed KO-settlement times so a still-alive path can be valued at
        # the settlement node. These are pure recording nodes (no KO/KI event matches
        # them), so the recursion records the bare continuation surface there. Gated on
        # record_grids: normal pricing keeps its exact observation-only time grid.
        if record_grids:
            times = self._insert_settlement_times(times, ko_records, maturity)

        rate_vec, div_vec, vol_vec = self._term_step_params(
            pricing_env, product.strike, times, rate, div, vol
        )
        vol_max_val = float(np.max(vol_vec))
        df_local = make_df_fn(pricing_env)
        dt = self._build_dt(times)
        reachable_ko = self._reachable_ko_observations(
            ko_records,
            times,
            spot,
            rate_vec,
            div_vec,
            vol_vec,
            dt,
            is_reverse=product.is_reverse,
        )
        reachable_by_record = {
            id(record): active
            for record, active in zip(ko_records, reachable_ko)
        }
        self._last_ignored_ko_observation_indices = tuple(
            index for index, active in enumerate(reachable_ko) if not active
        )

        align_log = self._select_alignment_log(
            spot,
            product,
            ki_barrier_override=ki_barrier_continuous,
            ko_candidates_override=[
                float(record.barrier)
                for record, active in zip(ko_records, reachable_ko)
                if active and record.barrier is not None
            ],
        )
        barrier_logs = self._barrier_logs(
            spot,
            [record.barrier for record in ko_records],
            ki_barrier_continuous
            if ki_barrier_continuous is not None
            else [record.barrier for record in ki_records],
        )
        fft_padding_factor = self._resolve_fft_padding_factor()
        fft_filter_alpha, fft_filter_power = self._resolve_fft_filter()
        grid_points = self._resolve_grid_points(
            maturity, vol_max_val, [rec.observation_time for rec in ki_records]
        )
        math_utils = QuadratureMath(
            grid_x=grid_points,
            spot=spot,
            maturity=maturity,
            vol_max=vol_max_val,
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
        tau = 0.5 * vol_vec * vol_vec * dt
        if np.any(tau[1:] <= 0.0):
            raise ValidationError("time step too small for quadrature solver.")

        alpha_vec = (rate_vec - div_vec - 0.5 * vol_vec * vol_vec) / (
            vol_vec * vol_vec
        )
        beta_vec = (rate_vec - div_vec - 0.5 * vol_vec * vol_vec) ** 2 / (
            vol_vec**4
        ) + 2.0 * rate_vec / (vol_vec * vol_vec)

        v_in_cash = np.array(
            [
                product.get_maturity_payoff_v1(spot_value, pricing_env=pricing_env)
                for spot_value in spot_grid
            ],
            dtype=float,
        )
        v_out_cash = np.array(
            [
                product.get_maturity_payoff_v0(spot_value, pricing_env=pricing_env)
                for spot_value in spot_grid
            ],
            dtype=float,
        )
        if fold is not None:
            terminal = payment_timings.terminal
            v_in_cash = v_in_cash + fold.bind_terminal(
                "ki", terminal.determination_time, terminal.payment_time
            )
            v_out_cash = v_out_cash + fold.bind_terminal(
                "no_ki", terminal.determination_time, terminal.payment_time
            )
        v_in = v_in_cash * payment_timings.terminal.delay_df
        v_out = v_out_cash * payment_timings.terminal.delay_df

        log_ki_barrier = None
        if product.has_ki_barrier and ki_continuous:
            log_ki_barrier = safe_log(ki_barrier_continuous / spot)

        full_p_lr, full_p_ur, full_p0 = 0, len(grid) - 1, (len(grid) - 1) % 2
        omega_grid = math_utils.z_grid

        transition_readout = self.params.readout == "transition"
        if transition_readout and ki_continuous:
            # The bridged transition is not a plain convolution, so it has no
            # pointwise form here yet. Refuse rather than quietly reading the
            # continuous-KI price off a different rule than the one asked for.
            raise NotImplementedError(
                "readout='transition' does not support continuous knock-in "
                "monitoring: the Brownian-bridge transition has no pointwise "
                "evaluation in this engine. Use readout='legacy_linear'."
            )
        final_transition: Optional[tuple] = None

        disable_ko_after_ki = product.barrier_config.disable_ko_after_ki

        smoothing_width = self._resolve_event_smoothing_width(math_utils, product)

        for step_index in range(len(times), 0, -1):
            obs_time = times[step_index - 1]
            ko_record = self._match_record(obs_time, ko_records)
            ko_weight = None
            if ko_record is not None:
                ko_is_reachable = bool(reachable_by_record[id(ko_record)])
                discount = event_delay_by_record[id(ko_record)]
                if fold_ko is None:
                    ko_value = ko_record.payoff * discount
                else:
                    ko_value = (ko_record.payoff + fold_ko[id(ko_record)]) * discount
                ko_weight = (
                    self._event_weight(
                        grid,
                        ko_record.barrier,
                        spot,
                        smoothing_width,
                        trigger_is_down=product.is_reverse,
                    )
                    if ko_is_reachable
                    else np.zeros_like(grid)
                )
            ki_record = self._match_record(obs_time, ki_records) if ki_records else None
            v_out, v_in = self._apply_observation(
                product, grid, spot_grid, spot, smoothing_width, v_out, v_in,
                ko_record=ko_record if ko_record is not None and ko_is_reachable else None,
                ko_value=ko_value if ko_record is not None else 0.0,
                ko_weight=ko_weight, ki_record=ki_record,
                ki_barrier_continuous=ki_barrier_continuous if ki_continuous else None,
            )

            # Post-event continuation surfaces AT obs_time (before diffusing back to
            # the previous step): v_out = value of a not-yet-knocked-in contract,
            # v_in = value once knock-in has occurred, both as a function of spot on
            # spot_grid. These are exactly the per-(t, state) value slices the CVA
            # exposure layer needs; the diffusion below rolls them to the prior step.
            if record_grids:
                self._backward_grids[float(obs_time)] = (
                    spot_grid.copy(), v_in.copy(), v_out.copy())

            tau_step = float(tau[step_index])
            alpha = float(alpha_vec[step_index])
            beta = float(beta_vec[step_index])
            vol_step = float(vol_vec[step_index])
            prefactor = math.exp(-beta * tau_step) / math.sqrt(math.pi * tau_step) / 2.0
            omega_array = np.exp(-(omega_grid**2) / (4.0 * tau_step) - alpha * omega_grid)

            if step_index == 1 and transition_readout:
                # The last transition carries the surfaces to the valuation
                # date; retain its INPUT so the readout can evaluate it at the
                # spot instead of interpolating its output between nodes.
                final_transition = (v_in, v_out, prefactor, alpha, beta, tau_step)

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

            if ki_continuous:
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

        # After the full backward sweep, v_in/v_out are the valuation-date (t=0)
        # surfaces; record them so the exposure grid has a t0 slice for both states.
        if record_grids:
            self._backward_grids[0.0] = (
                spot_grid.copy(), v_in.copy(), v_out.copy())

        value_surface = v_in if knocked_in_at_valuation else v_out
        self._last_spot_greeks_grid = (spot_grid.copy(), value_surface.copy())
        if final_transition is not None:
            pre_in, pre_out, prefactor, alpha, beta, tau_step = final_transition
            return float(
                self._readout_at_spot(
                    pre_in if knocked_in_at_valuation else pre_out,
                    math_utils,
                    prefactor,
                    full_p_lr,
                    full_p_ur,
                    full_p0,
                    alpha,
                    beta,
                    tau_step,
                )
            )
        return math_utils.interpolate(value_surface, x=0.0)

    def _price_with_convergence(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        *,
        fold=None,
    ) -> float:
        """Refine nested odd grids until consecutive prices meet tolerance.

        The returned value is always the finest engine price, not a scalar
        extrapolation, so ``_last_spot_greeks_grid`` and optional backward
        surfaces remain consistent with the reported PV.
        """
        requested = int(self.params.grid_points)
        cap = int(self.params.max_convergence_grid_points)
        estimates = []
        final_engine = None

        while requested <= cap:
            candidate_params = replace(
                self.params,
                grid_points=requested,
                auto_converge=False,
            )
            candidate = self.__class__(params=candidate_params)
            candidate.record_backward_grids = self.record_backward_grids
            if fold is None:
                value = float(candidate.price(product, pricing_env))
            else:
                value = float(candidate._price_once(product, pricing_env, fold=fold))
            grid_artifact = getattr(candidate, "_last_spot_greeks_grid", None)
            used = (
                int(grid_artifact[0].size)
                if grid_artifact is not None
                else requested
            )
            estimates.append((used, value))
            final_engine = candidate

            if len(estimates) >= 2:
                previous = estimates[-2][1]
                error = abs(value - previous)
                tolerance = float(self.params.convergence_abs_tol) + float(
                    self.params.convergence_rel_tol
                ) * max(1.0, abs(value), abs(previous))
                if error <= tolerance:
                    self._adopt_converged_engine(candidate)
                    self._last_convergence_info = {
                        "converged": True,
                        "grid_points": used,
                        "error_estimate": error,
                        "tolerance": tolerance,
                        "estimates": tuple(estimates),
                    }
                    return value

            next_requested = 2 * (used - 1) + 1
            if next_requested <= requested:
                next_requested = 2 * (requested - 1) + 1
            if next_requested > cap:
                break
            requested = next_requested

        if final_engine is not None:
            self._adopt_converged_engine(final_engine)
        last_error = (
            abs(estimates[-1][1] - estimates[-2][1])
            if len(estimates) >= 2
            else float("inf")
        )
        self._last_convergence_info = {
            "converged": False,
            "grid_points": estimates[-1][0] if estimates else requested,
            "error_estimate": last_error,
            "tolerance": (
                float(self.params.convergence_abs_tol)
                + float(self.params.convergence_rel_tol)
                * max(1.0, *(abs(v) for _, v in estimates[-2:]))
                if estimates
                else float("nan")
            ),
            "estimates": tuple(estimates),
        }
        raise NumericalError(
            "QUAD grid convergence was not reached by "
            f"max_convergence_grid_points={cap}; last successive PV difference "
            f"was {last_error:.6g}. Increase the cap or relax "
            "convergence_rel_tol/convergence_abs_tol."
        )

    def _adopt_converged_engine(self, candidate: "SnowballQuadEngine") -> None:
        """Copy numerical artifacts from the accepted finest-grid solve."""
        if hasattr(candidate, "_last_spot_greeks_grid"):
            self._last_spot_greeks_grid = tuple(
                np.array(values, copy=True)
                for values in candidate._last_spot_greeks_grid
            )
        elif hasattr(self, "_last_spot_greeks_grid"):
            del self._last_spot_greeks_grid
        self._backward_grids = candidate._backward_grids
        if hasattr(candidate, "_last_ignored_ko_observation_indices"):
            self._last_ignored_ko_observation_indices = (
                candidate._last_ignored_ko_observation_indices
            )

    def calculate_event_stats(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        *,
        streams: Optional[frozenset] = None,
    ) -> Optional[EventStatistics]:
        """``EventStatistics`` of the price's own recursion (unified event statistics spec T17).

        One stacked recursion marches a column per termination outcome through the price's operators: per
        knock-out record its discounted unit, the maturity unit and cash (never knocked in), the knocked-in
        settlement unit and cash (and the Phoenix coupon columns). P(knock in before any knock-out), its expected
        time and the per-date knock-in table come from the forward adjoint pass of that recursion
        (``v1_adjoint.py``), run only when ``streams`` asks for the knock-in stream (spec D17); every other column
        is never pruned. ``pv`` is ``price()`` (spec D23). ``None`` for an expired note or one with no knock-out
        record left.
        """
        if not isinstance(product, self._event_stats_product_type()):
            return None
        if pricing_env is None:
            raise PricingError("PricingEnvironment is required for SnowballQuadEngine.")
        if self.params.auto_converge:
            return self._event_stats_on_converged_grid(product, pricing_env, streams=streams)
        return self._compute_event_stats(product, pricing_env, streams=streams)

    def _event_stats_on_converged_grid(self, product, pricing_env, *, streams=None):
        """Use the accepted price grid for every statistical column; ``pv`` is the converged price itself."""
        pv = float(self.price(product, pricing_env))
        params = replace(
            self.params, auto_converge=False,
            grid_points=self._last_convergence_info["grid_points"],
        )
        candidate = self.__class__(params=params)
        candidate.record_backward_grids = self.record_backward_grids
        stats = candidate._compute_event_stats(product, pricing_env, streams=streams, pv=pv)
        self._last_v1_march = getattr(candidate, "_last_v1_march", None)
        return stats

    def price_with_events(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        emit_distribution: bool = True,
        streams: Optional[frozenset] = None,
        *,
        lifecycle_state=None,
    ):
        """Base behaviour with ``streams`` forwarded to the event-stats pass
        [§11.1] — the base implementation drops it, which forfeits the QUAD
        pruning above."""
        from quantark.cashleg.event_distribution import EventDistribution, PricingResult
        from quantark.asset.equity.engine.settlement_support import (
            validate_settlement_capability,
        )

        validate_settlement_capability(self, product, lifecycle_state)

        if emit_distribution:
            stats = self.calculate_event_stats(product, pricing_env, streams=streams)
            if stats is not None:
                return PricingResult(
                    npv=float(stats.pv),
                    event_distribution=EventDistribution.from_statistics(stats),
                )

        if lifecycle_state is None:
            npv = self.price(product, pricing_env)
        else:
            npv = self.price(product, pricing_env, lifecycle_state=lifecycle_state)
        return PricingResult(
            npv=npv,
            event_distribution=EventDistribution.trivial(product.get_maturity(pricing_env)),
        )

    def _event_stats_product_type(self) -> type:
        """Product type accepted by ``calculate_event_stats`` (overridable)."""
        return SnowballOption

    #: the statistics' product family (``EventStatistics.product_family``)
    _statistics_family = "snowball"

    def _apply_observation(
        self, product, grid, spot_grid, spot, width, v_out, v_in, *,
        ko_record=None, ko_value=0.0, ko_weight=None, ki_record=None,
        ki_barrier_continuous=None, coupon_index=None, n_ko=0,
        coupon_record=None, observation_index=None,
    ):
        """Apply the value's event order and projection to any number of columns."""
        disable = product.barrier_config.disable_ko_after_ki
        ko_last = ko_record is not None and ki_record is not None and disable

        def knock_out(surface):
            if self._use_cell_average_events():
                return self._project_quad_event(
                    grid, ko_record.barrier, spot, v_survive=surface,
                    v_breach=ko_value, breach_up=not product.is_reverse,
                )
            return ko_weight * ko_value + (1.0 - ko_weight) * surface

        if ko_record is not None:
            if not ko_last:
                v_out = knock_out(v_out)
            if not disable:
                v_in = knock_out(v_in)
        if ki_barrier_continuous is not None:
            mask = (spot_grid >= ki_barrier_continuous if product.is_reverse
                    else spot_grid <= ki_barrier_continuous)
            v_out[..., mask] = v_in[..., mask]
        elif ki_record is not None:
            v_out = self._blend_ki_transition(
                v_out, v_in, grid, spot_grid, ki_record.barrier, spot,
                width, product.is_reverse,
                ko_weight=ko_weight if not disable else None,
            )
        if ko_last:
            v_out = knock_out(v_out)
        return v_out, v_in

    def _event_stats_barrier_logs(self, spot, product, ko_records, ki_records, continuous_barrier):
        return self._barrier_logs(
            spot, [r.barrier for r in ko_records],
            continuous_barrier if continuous_barrier is not None else [r.barrier for r in ki_records],
        )

    def _event_stats_alignment_log(
        self,
        spot: float,
        product,
        ki_barrier_override: Optional[float] = None,
        ko_candidates_override: Optional[Sequence[float]] = None,
        pricing_env=None,
    ) -> Optional[float]:
        """Grid-alignment log used by event stats.

        Calls the Snowball implementation explicitly so a subclass that overrides
        the price-path ``_select_alignment_log`` (e.g. PhoenixQuadEngine, with a
        different signature) does not intercept the event-stats recursion.
        """
        return SnowballQuadEngine._select_alignment_log(
            self,
            spot,
            product,
            ki_barrier_override=ki_barrier_override,
            ko_candidates_override=ko_candidates_override,
        )

    # --- Extra indicator-row hooks (overridden by Phoenix for coupons) ---

    def _n_extra_quad_rows(self, n_ko: int) -> int:
        """Extra stacked indicator rows beyond the ``n_ko`` KO rows."""
        return 0

    def _initialize_extra_quad_rows(self, v_out, v_in, n_ko: int) -> None:
        """Terminal values of the extra rows, in place (zero unless a subclass sets them)."""

    def _coupon_columns(self, values_at_spot, n_ko, ko_records, product, pricing_env, df_fn) -> Optional[dict]:
        """The coupon fields read from the extra rows at the spot (``None``: no coupons, Snowball)."""
        return None

    def _terminal_cash_surfaces(self, product, pricing_env, spot_grid):
        """``(knocked-in, not knocked-in)`` maturity redemptions on the lattice, as ``price`` seeds them."""
        v_in_cash = np.array(
            [product.get_maturity_payoff_v1(s, pricing_env=pricing_env) for s in spot_grid], dtype=float
        )
        v_out_cash = np.array(
            [product.get_maturity_payoff_v0(s, pricing_env=pricing_env) for s in spot_grid], dtype=float
        )
        return v_in_cash, v_out_cash

    def _compute_event_stats(
        self,
        product: BaseEquityProduct,
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
        product_ki_records = list(ki_records)

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
            return self._decided_ko_statistics(
                product, pricing_env, ko_records, ko_record_0, streams=streams
            )

        knock_in_refusal = None
        if ki_records and (
            self._ki_monitoring_mode() is KnockInMonitoringMode.BGK_APPROXIMATION
        ):
            ki_barrier_continuous = self._bgk_equivalent_continuous_ki_barrier(
                product, ki_records, pricing_env, vol, maturity
            )
            ki_continuous = True
            ki_records = []
            knock_in_refusal = v1_statistics.BGK_REASON

        times = self._merge_times(
            [rec.observation_time for rec in ko_records],
            [rec.observation_time for rec in ki_records],
            maturity,
        )
        if ki_continuous:
            from quantark.asset.equity.engine.continuous_ki_support import continuous_time_grid
            times, self._continuous_ki_model = continuous_time_grid(pricing_env, times, self.params)
            times = self._merge_times(times, (), maturity)
        dt = self._build_dt(times)

        rate_vec, div_vec, vol_vec = self._term_step_params(
            pricing_env, product.strike, times, rate, div, vol
        )
        vol_max_val = float(np.max(vol_vec))
        df_local = make_df_fn(pricing_env)
        reachable_ko = self._reachable_ko_observations(
            ko_records,
            times,
            spot,
            rate_vec,
            div_vec,
            vol_vec,
            dt,
            is_reverse=product.is_reverse,
        )

        align_log = self._event_stats_alignment_log(
            spot,
            product,
            ki_barrier_override=ki_barrier_continuous,
            ko_candidates_override=[
                float(record.barrier)
                for record, active in zip(ko_records, reachable_ko)
                if active and record.barrier is not None
            ],
            pricing_env=pricing_env,
        )
        barrier_logs = self._event_stats_barrier_logs(
            spot, product, ko_records, ki_records, ki_barrier_continuous
        )
        fft_padding_factor = self._resolve_fft_padding_factor()
        fft_filter_alpha, fft_filter_power = self._resolve_fft_filter()
        grid_points = self._resolve_grid_points(
            maturity, vol_max_val, [rec.observation_time for rec in ki_records]
        )
        math_utils = QuadratureMath(
            grid_x=grid_points,
            spot=spot,
            maturity=maturity,
            vol_max=vol_max_val,
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
        smoothing_width = self._resolve_event_smoothing_width(math_utils, product)

        tau = 0.5 * vol_vec * vol_vec * dt
        alpha_vec = (rate_vec - div_vec - 0.5 * vol_vec * vol_vec) / (
            vol_vec * vol_vec
        )
        beta_vec = (rate_vec - div_vec - 0.5 * vol_vec * vol_vec) ** 2 / (
            vol_vec**4
        ) + 2.0 * rate_vec / (vol_vec * vol_vec)
        full_p_lr, full_p_ur, full_p0 = 0, len(grid) - 1, (len(grid) - 1) % 2
        omega_grid = math_utils.z_grid

        disable_ko_after_ki = product.barrier_config.disable_ko_after_ki
        timings = resolve_structured_payment_timings(product, pricing_env, ko_records)
        terminal = timings.terminal
        v_in_cash, v_out_cash = self._terminal_cash_surfaces(product, pricing_env, spot_grid)
        if pv is None:
            pv = float(self.price(product, pricing_env))

        if str(self.params.event_stats_mode) == "forward_density":
            q = self._forward_event_quantities(
                product=product, pricing_env=pricing_env,
                math_utils=math_utils, times=times, tau=tau,
                alpha_vec=alpha_vec, vol_vec=vol_vec, dt=dt,
                ko_records=ko_records, ki_records=ki_records,
                ki_continuous=ki_continuous,
                ki_barrier_continuous=ki_barrier_continuous,
                reachable_ko=reachable_ko, spot=spot, spot_grid=spot_grid,
                smoothing_width=smoothing_width,
                disable_ko_after_ki=disable_ko_after_ki,
                knocked_in_at_valuation=knocked_in_at_valuation,
                want_ki=True, want_coupon=True,
                df_local=df_local, rate=rate,
            )
            # Diagnostic only (spec section 8): terminal mass + absorbed KO
            # mass; drifts from 1.0 by edge leakage. Never used numerically.
            self._last_forward_mass_diagnostic = float(q["mass_diagnostic"])
            return self._forward_density_statistics(
                product, pricing_env, q, ko_records=ko_records, terminal=terminal, pv=pv,
                knocked_in_at_valuation=knocked_in_at_valuation, streams=streams,
                terminal_cash=(
                    float(df_local(maturity)) * float(terminal.delay_df)
                    * self._density_integral(math_utils, q["terminal_out"] * v_out_cash),
                    float(df_local(maturity)) * float(terminal.delay_df)
                    * self._density_integral(math_utils, q["terminal_in"] * v_in_cash),
                ),
            )

        # --- The statistics recursion: one column per outcome, stacked ---
        # Rows [0..n_ko-1] are knock-out units (one paid at the record's settlement); subclasses append extra rows
        # (Phoenix coupon columns) that ride the same diffusion/jumps; then the maturity unit and cash (never
        # knocked in) and the knocked-in settlement unit and cash.
        n_ko = len(ko_records)
        n_main = n_ko + self._n_extra_quad_rows(n_ko)
        m_unit, m_cash, kis_unit, kis_cash = n_main, n_main + 1, n_main + 2, n_main + 3
        n_rows = n_main + 4
        v_in = np.zeros((n_rows, grid.size), dtype=float)
        v_out = np.zeros((n_rows, grid.size), dtype=float)
        self._initialize_extra_quad_rows(v_out, v_in, n_ko)
        terminal_out_cash = v_out_cash * terminal.delay_df
        terminal_in_cash = v_in_cash * terminal.delay_df
        v_out[m_unit] = 1.0
        v_out[m_cash] = terminal_out_cash
        v_in[kis_unit] = 1.0
        v_in[kis_cash] = terminal_in_cash

        # The readout mode governs the event decomposition too, so its parts
        # keep reconciling with the price under either rule.
        transition_readout = self.params.readout == "transition"
        if transition_readout and ki_continuous:
            raise NotImplementedError(
                "readout='transition' does not support continuous knock-in "
                "monitoring: the Brownian-bridge transition has no pointwise "
                "evaluation in this engine. Use readout='legacy_linear'."
            )
        final_transition: Optional[dict] = None
        log = MarchLog(engine=self, math_utils=math_utils, readout_in=bool(knocked_in_at_valuation),
                       terminal_out={"maturity_unit": np.ones(grid.size), "maturity_cash": terminal_out_cash},
                       terminal_in={"ki_settlement_unit": np.ones(grid.size), "ki_settlement_cash": terminal_in_cash})
        log_ki_barrier = (safe_log(ki_barrier_continuous / spot) if ki_continuous else None)

        for step_index in range(len(times), 0, -1):
            obs_time = times[step_index - 1]

            ko_w = None
            ko_index = None
            ko_record = None
            for idx, rec in enumerate(ko_records):
                if is_close(obs_time, rec.observation_time, abs_tol=Tolerance.PRECISION):
                    ko_index = idx
                    ko_record = rec
                    break

            ko_is_reachable = ko_record is not None and bool(reachable_ko[int(ko_index)])
            ko_breach = np.zeros_like(v_out)
            discount_delay = 1.0
            if ko_record is not None:
                discount_delay = self._ko_discount(
                    rate, obs_time, ko_record.settlement_time, df_fn=df_local
                )
                ko_breach[int(ko_index)] = float(discount_delay)
                ko_w = self._event_weight(
                    grid, ko_record.barrier, spot, smoothing_width,
                    trigger_is_down=product.is_reverse,
                ) if ko_is_reachable else np.zeros_like(grid)
            ki_record = self._match_record(obs_time, ki_records) if ki_records else None
            v_out, v_in = self._apply_observation(
                product, grid, spot_grid, spot, smoothing_width, v_out, v_in,
                ko_record=ko_record if ko_is_reachable else None,
                ko_value=ko_breach, ko_weight=ko_w, ki_record=ki_record,
                ki_barrier_continuous=ki_barrier_continuous if ki_continuous else None,
                coupon_index=ko_index, n_ko=n_ko,
                coupon_record=ko_record, observation_index=ko_index,
            )

            tau_step = float(tau[step_index])
            alpha = float(alpha_vec[step_index])
            beta = float(beta_vec[step_index])
            vol_step = float(vol_vec[step_index])
            prefactor = math.exp(-beta * tau_step) / math.sqrt(math.pi * tau_step) / 2.0
            omega_array = np.exp(
                -(omega_grid**2) / (4.0 * tau_step) - alpha * omega_grid
            )

            # The adjoint log: this step's event map (the same call with every source zero, or one knock-out
            # source alone) and its diffusion back to the previous step.
            event = None
            if ko_record is not None or ki_record is not None or ki_continuous:
                event = self._statistics_event(
                    product, grid, spot_grid, spot, smoothing_width,
                    ko_record=ko_record if ko_is_reachable else None, ko_weight=ko_w,
                    ki_record=ki_record,
                    ki_barrier_continuous=ki_barrier_continuous if ki_continuous else None,
                    n_ko=n_ko, coupon_record=ko_record, observation_index=ko_index,
                    discount=float(discount_delay),
                )
            log.steps.append(EventStep(
                time=float(obs_time), event=event,
                sources=(("ko", int(ko_index)),) if ko_is_reachable else (),
                source_units={("ko", int(ko_index)): float(discount_delay)} if ko_is_reachable else {},
                diffusion=(tau_step, alpha, beta, prefactor, omega_array),
                bridge=(log_ki_barrier, vol_step, float(dt[step_index]), bool(product.is_reverse))
                if ki_continuous else None,
            ))

            if step_index == 1 and transition_readout:
                # Inputs to the transition that reaches the valuation date, so
                # the readout can evaluate it at the spot.
                final_transition = {
                    "kernel": (prefactor, alpha, beta, tau_step),
                    "in": v_in,
                    "out": v_out,
                }

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

            if ki_continuous:
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

        initial_surface = v_in if knocked_in_at_valuation else v_out
        if final_transition is not None:
            prefactor, alpha, beta, tau_step = final_transition["kernel"]
            log.transition = (prefactor, alpha, beta, tau_step)
            pre = final_transition["in" if knocked_in_at_valuation else "out"]
            values_at_spot = np.asarray(
                self._readout_at_spot(
                    pre, math_utils, prefactor, full_p_lr, full_p_ur, full_p0,
                    alpha, beta, tau_step,
                ),
                dtype=float,
            )
        else:
            values_at_spot = np.array(
                [math_utils.interpolate(initial_surface[i], x=0.0) for i in range(n_rows)],
                dtype=float,
            )
        ed_unit = values_at_spot[:n_ko]
        log.columns = {("ko", i): float(ed_unit[i]) for i in range(n_ko)}
        log.columns.update({
            ("terminal", "maturity_unit"): float(values_at_spot[m_unit]),
            ("terminal", "maturity_cash"): float(values_at_spot[m_cash]),
            ("terminal", "ki_settlement_unit"): float(values_at_spot[kis_unit]),
            ("terminal", "ki_settlement_cash"): float(values_at_spot[kis_cash]),
        })
        self._last_v1_march = log

        df_maturity = float(df_local(maturity))
        timing = dict(time=float(terminal.determination_time), payment_time=float(terminal.payment_time),
                      date=terminal.determination_date, payment_date=terminal.payment_date)
        maturity_row = es.outcome_row(pricing_env, probability=float(values_at_spot[m_unit]) / df_maturity,
                                      discounted=float(values_at_spot[m_cash]), **timing)
        ki_settlement = es.outcome_row(pricing_env, probability=float(values_at_spot[kis_unit]) / df_maturity,
                                       discounted=float(values_at_spot[kis_cash]), **timing)
        coupon_fields = self._coupon_columns(values_at_spot, n_ko, ko_records, product, pricing_env, df_local)
        observations = es.observation_table(
            pricing_env, ko_records, ed_unit,
            is_coupon_obs=None if coupon_fields is None else np.ones(n_ko, dtype=bool),
        )
        unavailable: dict = {}
        coupons, path_dependent_cash = None, None
        if coupon_fields is not None:
            coupons, path_dependent_cash = v1_statistics.coupon_table(
                pricing_env, terminal, observations, streams=streams, unavailable=unavailable, **coupon_fields
            )
        rule = es.knock_in_rule(product, knocked_out_now=False, knocked_in_now=bool(knocked_in_at_valuation))
        return v1_statistics.statistics(
            product, pricing_env, family=self._statistics_family, pv=pv, observations=observations,
            coupons=coupons, maturity=maturity_row, ki_settlement=ki_settlement, rule=rule, want_knock_in=want_ki,
            forward=lambda: v1_statistics.knock_in_table(
                product, pricing_env, log, horizon=float(maturity), continuous=bool(ki_continuous),
                ki_records=product_ki_records,
            ),
            unavailable=unavailable, path_dependent_cash=path_dependent_cash, knock_in_refusal=knock_in_refusal,
        )

    def _statistics_event(self, product, grid, spot_grid, spot, width, *, ko_record, ko_weight, ki_record,
                          ki_barrier_continuous, n_ko, coupon_record, observation_index, discount):
        """The observation's event map on the two note states, for the adjoint pass: ``event(v_out, v_in, source)``
        is ``_apply_observation`` with every knock-out source zero (``source=None``) or with the record's own
        source alone (its discounted unit, as the knock-out column breaches); coupon columns are not touched."""

        def event(v_out, v_in, source):
            v_out = np.array(v_out, dtype=float)
            v_in = np.array(v_in, dtype=float)
            ko_value = np.zeros_like(v_out) if source is None else np.full_like(v_out, discount)
            return self._apply_observation(
                product, grid, spot_grid, spot, width, v_out, v_in,
                ko_record=ko_record, ko_value=ko_value, ko_weight=ko_weight, ki_record=ki_record,
                ki_barrier_continuous=ki_barrier_continuous, coupon_index=None, n_ko=n_ko,
                coupon_record=coupon_record, observation_index=observation_index,
            )

        return event

    def _decided_ko_statistics(self, product, pricing_env, ko_records, ko_record_0, *, streams=None):
        """A knock-out decided at the valuation instant: certain, on its record (DESIGN §4.5); the whole price is
        the record's cash. No knock-in path statistics but a carried knock-in (the knock-out wins a tie)."""
        from quantark.asset.equity.engine.pde.core.statistics import event_statistics as es

        pv = float(self._get_immediate_ko_payoff(ko_record_0, pricing_env))
        observations = es.observation_table(pricing_env, ko_records, self._decided_ko_unit(
            pricing_env, ko_records, ko_record_0))
        return self._decided_statistics(product, pricing_env, observations, pv=pv, streams=streams,
                                        terminal=resolve_structured_payment_timings(
                                            product, pricing_env, ko_records).terminal)

    @staticmethod
    def _decided_ko_unit(pricing_env, ko_records, decided) -> np.ndarray:
        """The discounted knock-out units of a knock-out decided now: one paid at the decided record's settlement
        (its probability exactly 1), zero elsewhere."""
        unit = np.zeros(len(ko_records), dtype=float)
        for i, rec in enumerate(ko_records):
            if rec is decided:
                payment = rec.settlement_time if rec.settlement_time is not None else rec.observation_time
                unit[i] = float(pricing_env.get_discount_factor(float(payment)))
        return unit

    def _decided_statistics(self, product, pricing_env, observations, *, pv, streams, terminal, coupons=None,
                            unavailable=None):
        """The statistics of a note decided at the valuation instant: nothing left to mature."""
        from quantark.asset.equity.engine.pde.core.statistics import event_statistics as es
        from quantark.asset.equity.engine.quad import v1_statistics

        nothing = dict(time=float(terminal.determination_time), payment_time=float(terminal.payment_time),
                       date=terminal.determination_date, payment_date=terminal.payment_date,
                       probability=0.0, discounted=0.0)
        carried = bool(getattr(product, "_otc_lifecycle_knocked_in", False))
        rule = es.knock_in_rule(product, knocked_out_now=True, knocked_in_now=carried)
        return v1_statistics.statistics(
            product, pricing_env, family=self._statistics_family, pv=pv, observations=observations,
            coupons=coupons, maturity=es.outcome_row(pricing_env, **nothing),
            ki_settlement=es.outcome_row(pricing_env, **nothing), rule=rule,
            want_knock_in=es.want_knock_in_path(streams), forward=None,
            unavailable={} if unavailable is None else unavailable,
        )

    def _forward_density_statistics(self, product, pricing_env, q, *, ko_records, terminal, pv,
                                    knocked_in_at_valuation, streams, terminal_cash):
        """``EventStatistics`` of the opt-in forward-density mode (provenance ``quad_v1_forward_density``).

        Its own numbers: knock-out probabilities from the absorbed mass, the maturity and knocked-in settlement from
        the terminal densities (their cash integrated against the redemptions), the Phoenix coupon fields where the
        density alone determines them. It tracks no first knock-in before a knock-out, so the knock-in path
        statistics are unavailable with a reason (its KI-ever counted knock-ins after a knock-out, spec AC9)."""
        from quantark.asset.equity.engine.pde.core.statistics import event_statistics as es
        from quantark.asset.equity.engine.quad import v1_statistics

        unit = np.array([
            float(p) * float(pricing_env.get_discount_factor(
                float(r.settlement_time if r.settlement_time is not None else r.observation_time)))
            for p, r in zip(q["ko_prob"], ko_records)
        ], dtype=float)
        coupon_fields = self._forward_density_coupon_fields(q, ko_records, product, pricing_env)
        observations = es.observation_table(
            pricing_env, ko_records, unit,
            is_coupon_obs=None if coupon_fields is None else np.ones(len(ko_records), dtype=bool),
        )
        timing = dict(time=float(terminal.determination_time), payment_time=float(terminal.payment_time),
                      date=terminal.determination_date, payment_date=terminal.payment_date)
        maturity_row = es.outcome_row(pricing_env, probability=float(q["terminal_out_mass"]),
                                      discounted=float(terminal_cash[0]), **timing)
        ki_settlement = es.outcome_row(pricing_env, probability=float(q["terminal_in_mass"]),
                                       discounted=float(terminal_cash[1]), **timing)
        unavailable: dict = {}
        coupons, path_dependent_cash = None, None
        if coupon_fields is not None:
            coupons, path_dependent_cash = v1_statistics.coupon_table(
                pricing_env, terminal, observations, streams=streams, unavailable=unavailable, **coupon_fields
            )
        rule = es.knock_in_rule(product, knocked_out_now=False, knocked_in_now=bool(knocked_in_at_valuation))
        return v1_statistics.statistics(
            product, pricing_env, family=self._statistics_family, pv=pv, observations=observations,
            coupons=coupons, maturity=maturity_row, ki_settlement=ki_settlement, rule=rule,
            want_knock_in=es.want_knock_in_path(streams), forward=None, unavailable=unavailable,
            provenance=v1_statistics.FORWARD_DENSITY_PROVENANCE, path_dependent_cash=path_dependent_cash,
            knock_in_refusal=v1_statistics.FORWARD_DENSITY_KI_REASON,
        )

    def _forward_density_coupon_fields(self, q, ko_records, product, pricing_env) -> Optional[dict]:
        """The forward-density coupon fields (``None``: no coupons, Snowball)."""
        return None

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

    def _reachable_ko_observations(
        self,
        ko_records: Sequence,
        times: Sequence[float],
        spot: float,
        rate_vec: np.ndarray,
        div_vec: np.ndarray,
        vol_vec: np.ndarray,
        dt: np.ndarray,
        *,
        is_reverse: bool,
    ) -> tuple[bool, ...]:
        """KO reachability at each record's own observation time.

        This is a numerical sentinel policy, not schedule mutation. Every
        observation remains on the time grid; only a KO trigger beyond the
        configured Gaussian tail envelope is suppressed. A barrier already
        triggered at valuation is always active.
        """
        if not self.params.filter_unreachable_barriers:
            return tuple(True for _ in ko_records)

        mu = (
            np.asarray(rate_vec, dtype=float)
            - np.asarray(div_vec, dtype=float)
            - 0.5 * np.asarray(vol_vec, dtype=float) ** 2
        )
        dt_arr = np.asarray(dt, dtype=float)
        cumulative_mean = np.cumsum(mu * dt_arr)
        cumulative_var = np.cumsum(
            np.asarray(vol_vec, dtype=float) ** 2 * dt_arr
        )
        reach_std = float(
            self.params.barrier_reach_stddevs
            if self.params.barrier_reach_stddevs is not None
            else self.params.num_std_devs
        )
        time_arr = np.asarray(times, dtype=float)
        active = []
        for rec in ko_records:
            barrier = getattr(rec, "barrier", None)
            if barrier is None or not np.isfinite(float(barrier)) or barrier <= 0:
                active.append(False)
                continue
            barrier = float(barrier)
            triggered_now = barrier >= spot if is_reverse else barrier <= spot
            if float(rec.observation_time) <= Tolerance.ZERO:
                active.append(triggered_now)
                continue
            if triggered_now:
                active.append(True)
                continue

            matches = np.flatnonzero(
                np.isclose(
                    time_arr,
                    float(rec.observation_time),
                    rtol=0.0,
                    atol=Tolerance.PRECISION,
                )
            )
            if matches.size == 0:
                raise ValidationError(
                    "KO observation time is missing from the QUAD recursion grid: "
                    f"{rec.observation_time}"
                )
            step = int(matches[0]) + 1
            sigma = math.sqrt(max(float(cumulative_var[step]), 0.0))
            lower = float(cumulative_mean[step]) - reach_std * sigma
            upper = float(cumulative_mean[step]) + reach_std * sigma
            barrier_log = safe_log(barrier / float(spot))
            active.append(
                barrier_log >= lower if is_reverse else barrier_log <= upper
            )
        return tuple(active)

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

        # Stacked indicator rows are identically zero until the backward sweep
        # crosses their own observation date; diffusion and the tail correction
        # are linear, so those rows stay exactly zero — skip them. Bit-identical
        # because pocketfft transforms rows independently and 0 * erfc = +0.0.
        active = np.any(values != 0.0, axis=1)
        if not bool(active.all()):
            out = np.zeros_like(values, dtype=float)
            if bool(active.any()):
                out[active] = self._diffuse_fft(
                    values[active], math_utils, omega_array, prefactor,
                    p_lr, p_ur, p0, alpha, beta, tau_step,
                )
            return out

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
        """Roll ``v_out`` back one step under CONTINUOUS knock-in monitoring.

        Between two observation dates a path may touch the KI barrier and
        recover, so the not-yet-knocked-in surface must be blended with the
        knocked-in one using the Brownian-bridge touch probability::

            V_out(x, t_km1) = INT K(x,y) [ (1 - p_hit) V_out(y, t_k)
                                           + p_hit     V_in (y, t_k) ] dy

        which is evaluated here as ``base + INT K p_hit (v_in - v_out)``.

        Contract, and the reason this method takes ``v_in`` as an argument
        rather than reading a rolled surface: **both ``v_out`` and ``v_in``
        must be the slices at ``t_k``**, i.e. neither may have been diffused
        yet. Passing a ``v_in`` that the caller has already rolled back to
        ``t_km1`` subtracts surfaces from two different times, and in the
        ``p_hit -> 1`` limit returns ``K(K(v_in))`` instead of ``K(v_in)``.
        That defect biased PV low in proportion to ``p_hit`` -- worst just
        above the barrier -- and was invisible to grid refinement.
        """
        grid = math_utils.grid
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

        # Indicator rows whose delta is identically zero contribute an exactly
        # zero correction (the kernel is finite and non-negative); reduce the
        # accumulation to the active rows and scatter back.
        active = None
        work = delta
        if delta.ndim == 2:
            row_active = np.any(delta != 0.0, axis=1)
            if not bool(row_active.all()):
                active = row_active
                work = delta[row_active]

        band = self._bridge_band(tau_step, math_utils.h, grid.size)
        denom = vol * vol * dt
        kernels = self._bridge_kernels(
            math_utils, band, tau_step, alpha, denom, log_barrier, is_reverse
        )

        correction_work = np.zeros_like(work, dtype=float)
        if work.size:
            for source, target, kernel in kernels:
                if work.ndim == 1:
                    correction_work[target] += prefactor * kernel * work[source]
                else:
                    correction_work[:, target] += (
                        prefactor * kernel.reshape(1, -1) * work[:, source]
                    )

        if active is not None:
            correction = np.zeros_like(delta, dtype=float)
            if work.size:
                correction[active] = correction_work
        else:
            correction = correction_work
        return base + correction

    # Clear-all bound for _bridge_kernel_cache; each entry holds ~(2*band+1)
    # grid-length arrays (~1 MB at grid 1001 / band ~63).
    _BRIDGE_KERNEL_CACHE_MAX_ENTRIES = 64

    def _bridge_kernels(
        self,
        math_utils: QuadratureMath,
        band: int,
        tau_step: float,
        alpha: float,
        denom: float,
        log_barrier: float,
        is_reverse: bool,
    ) -> list:
        """Per-offset ``(source, target, omega * w * p_hit)`` kernel stack for
        ``_diffuse_with_bridge``, cached on the engine (see __init__).

        Computed exactly as the historical loop body computed them — the cache
        key carries every input the kernel depends on, so a hit is
        bit-identical to a rebuild.
        """
        grid = math_utils.grid
        key = (
            grid.size,
            float(math_utils.h),
            float(grid[0]),
            band,
            float(tau_step),
            float(alpha),
            float(denom),
            float(log_barrier),
            bool(is_reverse),
        )
        cached = self._bridge_kernel_cache.get(key)
        if cached is not None:
            return cached

        weights = math_utils.simpson_weight_vector()
        kernels = []
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
            kernels.append((source, target, omega * w * p_hit))

        if len(self._bridge_kernel_cache) >= self._BRIDGE_KERNEL_CACHE_MAX_ENTRIES:
            self._bridge_kernel_cache.clear()
        self._bridge_kernel_cache[key] = kernels
        return kernels

    # --- Forward-density primitives (event_stats_mode="forward_density") ---

    def _forward_kernel(self, math_utils, tau_step: float, alpha: float):
        """Forward transition kernel: the backward omega with the alpha sign
        flipped (correlation -> convolution) and the discount term dropped
        from beta (densities are undiscounted; discounting happens at
        readout)."""
        omega_fwd = np.exp(
            -(math_utils.z_grid ** 2) / (4.0 * tau_step)
            + alpha * math_utils.z_grid
        )
        prefactor_fwd = (
            math.exp(-(alpha * alpha) * tau_step)
            / math.sqrt(math.pi * tau_step) / 2.0
        )
        return omega_fwd, prefactor_fwd

    def _diffuse_density(
        self, values, math_utils, omega_fwd, prefactor_fwd, p_lr, p_ur, p0
    ):
        """One forward transport step. No tail correction: a density is ~0 at
        the grid edges by construction (num_std_devs-wide grid); mass
        bookkeeping is the caller's diagnostic."""
        if values.ndim == 1:
            u = math_utils.simpson_weights(values, p_lr, p_ur, p0)
            return prefactor_fwd * math_utils.convolution_fft(omega_fwd, u)
        u = math_utils.simpson_weights_many(values, p_lr, p_ur, p0)
        return prefactor_fwd * math_utils.convolution_fft_many(omega_fwd, u)

    @staticmethod
    def _density_integral(math_utils, values) -> float:
        """Simpson integral of a grid function (weights are h-scaled)."""
        return float(np.dot(math_utils.simpson_weight_vector(), values))

    def _forward_seed(self, math_utils, tau1: float, alpha1: float):
        """Exact density after the first interval (analytic first step —
        no discrete delta on the grid; spec section 4.3). var = sigma^2*dt1 =
        2*tau1; mean = m*dt1 = alpha1 * 2*tau1."""
        var = 2.0 * tau1
        mean = var * alpha1
        grid = math_utils.grid
        return np.exp(-((grid - mean) ** 2) / (2.0 * var)) / math.sqrt(
            2.0 * math.pi * var
        )

    def _forward_seed_touch_probability(
        self, math_utils, tau1: float, log_barrier: float, is_reverse: bool
    ):
        """Point-source Brownian-bridge touch probability over the first
        interval (source pinned at x=0 = spot), same formula family as the
        step kernel's p_hit."""
        grid = math_utils.grid
        denom = 2.0 * tau1
        if is_reverse:
            d0 = log_barrier - grid
            d1 = log_barrier - 0.0
        else:
            d0 = grid - log_barrier
            d1 = 0.0 - log_barrier
        safe = (d0 > 0.0) & (d1 > 0.0)
        exponent = np.clip(
            np.where(safe, -2.0 * d0 * d1 / denom, 0.0), -745.0, 0.0
        )
        return np.where(safe, np.exp(exponent), 1.0)

    # --- Extra forward-readout hooks (Phoenix overrides; Snowball no-ops) ---

    def _forward_extra_state(self, n_ko: int):
        return None

    def _forward_extra_readouts_at_obs(
        self, state, ko_index, ko_record, p_alive, math_utils,
        smoothing_width, product,
    ) -> None:
        return None

    def _forward_extra_fields(
        self, state, ko_records, product, pricing_env, rate, df_local
    ) -> dict:
        return {}

    def _forward_event_quantities(
        self, *, product, pricing_env, math_utils, times, tau, alpha_vec,
        vol_vec, dt, ko_records, ki_records, ki_continuous,
        ki_barrier_continuous, reachable_ko, spot, spot_grid,
        smoothing_width, disable_ko_after_ki, knocked_in_at_valuation,
        want_ki, want_coupon, df_local, rate,
    ) -> dict:
        """Forward transition-density march producing the event-distribution
        quantities (spec section 4): two surfaces p_out / p_in (KO-absorbing,
        KI-splitting). No knock-in is tracked after a knock-out (unified event
        statistics T17: the KO-independent "ever" surface is removed)."""
        grid = math_utils.grid
        n_ko = len(ko_records)
        full_p_lr, full_p_ur, full_p0 = 0, grid.size - 1, (grid.size - 1) % 2
        absorbed = np.zeros(n_ko, dtype=float)
        extra_state = self._forward_extra_state(n_ko) if want_coupon else None
        log_ki = (
            safe_log(ki_barrier_continuous / spot) if ki_continuous else None
        )

        # Analytic first step to times[0] (interval index 1).
        tau1, alpha1 = float(tau[1]), float(alpha_vec[1])
        p_free = self._forward_seed(math_utils, tau1, alpha1)
        if knocked_in_at_valuation:
            p_in = p_free
            p_out = np.zeros_like(p_free)
        elif ki_continuous and product.has_ki_barrier:
            touch0 = self._forward_seed_touch_probability(
                math_utils, tau1, log_ki, product.is_reverse
            )
            p_out = p_free * (1.0 - touch0)
            p_in = p_free * touch0
        else:
            p_out = p_free
            p_in = np.zeros_like(p_free)

        for step_index in range(1, len(times) + 1):
            if step_index > 1:
                tau_step = float(tau[step_index])
                alpha = float(alpha_vec[step_index])
                vol_step = float(vol_vec[step_index])
                omega_fwd, prefactor_fwd = self._forward_kernel(
                    math_utils, tau_step, alpha
                )
                rows = [p_out, p_in]
                transported = self._diffuse_density(
                    np.vstack(rows), math_utils, omega_fwd, prefactor_fwd,
                    full_p_lr, full_p_ur, full_p0,
                )
                if ki_continuous and not knocked_in_at_valuation:
                    band = self._bridge_band(tau_step, math_utils.h, grid.size)
                    denom = vol_step * vol_step * float(dt[step_index])
                    kernels = self._bridge_kernels(
                        math_utils, band, tau_step, -alpha, denom,
                        log_ki, product.is_reverse,
                    )
                    touch_rows = np.vstack([p_out])
                    touched = np.zeros(
                        (touch_rows.shape[0], grid.size), dtype=float
                    )
                    for source, target, kernel in kernels:
                        touched[:, target] += (
                            prefactor_fwd * kernel * touch_rows[:, source]
                        )
                    p_out = transported[0] - touched[0]
                    p_in = transported[1] + touched[0]
                else:
                    p_out = transported[0]
                    p_in = transported[1]

            obs_time = times[step_index - 1]
            ko_index, ko_record = None, None
            for idx, rec in enumerate(ko_records):
                if is_close(obs_time, rec.observation_time,
                            abs_tol=Tolerance.PRECISION):
                    ko_index, ko_record = idx, rec
                    break
            if ko_record is not None:
                if extra_state is not None:
                    # Coupons read the alive density BEFORE KO absorption
                    # (a coupon on a simultaneous KO is still paid).
                    self._forward_extra_readouts_at_obs(
                        extra_state, int(ko_index), ko_record, p_out + p_in,
                        math_utils, smoothing_width, product,
                    )
                if bool(reachable_ko[int(ko_index)]):
                    if self._use_cell_average_events():
                        zero = np.zeros((1, grid.size), dtype=float)
                        proj = self._project_quad_event(
                            grid, ko_record.barrier, spot,
                            v_survive=p_out[None, :], v_breach=zero,
                            breach_up=not product.is_reverse,
                        )[0]
                        absorbed_mass = self._density_integral(
                            math_utils, p_out
                        ) - self._density_integral(math_utils, proj)
                        p_out = proj
                        if not disable_ko_after_ki:
                            proj_in = self._project_quad_event(
                                grid, ko_record.barrier, spot,
                                v_survive=p_in[None, :], v_breach=zero,
                                breach_up=not product.is_reverse,
                            )[0]
                            absorbed_mass += self._density_integral(
                                math_utils, p_in
                            ) - self._density_integral(math_utils, proj_in)
                            p_in = proj_in
                    else:
                        ko_w = self._event_weight(
                            grid, ko_record.barrier, spot, smoothing_width,
                            trigger_is_down=product.is_reverse,
                        )
                        absorbed_mass = self._density_integral(
                            math_utils, ko_w * p_out
                        )
                        p_out = p_out * (1.0 - ko_w)
                        if not disable_ko_after_ki:
                            absorbed_mass += self._density_integral(
                                math_utils, ko_w * p_in
                            )
                            p_in = p_in * (1.0 - ko_w)
                    absorbed[int(ko_index)] = absorbed_mass

            if (
                (not ki_continuous)
                and ki_records
                and not knocked_in_at_valuation
            ):
                ki_record = self._match_record(obs_time, ki_records)
                if ki_record is not None:
                    ki_w = self._event_weight(
                        grid, ki_record.barrier, spot, smoothing_width,
                        trigger_is_down=not product.is_reverse,
                    )
                    transfer = ki_w * p_out
                    p_out = p_out - transfer
                    p_in = p_in + transfer

        # --- Readouts ---
        ko_prob = absorbed.copy()
        ed_ko_cf = np.zeros(n_ko, dtype=float)
        survival_prob = np.ones(n_ko, dtype=float)
        cumulative = 0.0
        for i, rec in enumerate(ko_records):
            obs = float(rec.observation_time)
            df_total = float(df_local(obs)) * float(
                self._ko_discount(rate, obs, rec.settlement_time,
                                  df_fn=df_local)
            )
            payoff = float(rec.payoff) if rec.payoff is not None else 0.0
            ed_ko_cf[i] = ko_prob[i] * df_total * payoff
            cumulative += ko_prob[i]
            survival_prob[i] = max(0.0, 1.0 - cumulative)

        extra_fields = (
            self._forward_extra_fields(
                extra_state, ko_records, product, pricing_env, rate, df_local
            )
            if extra_state is not None
            else {}
        )
        terminal_mass = self._density_integral(math_utils, p_out + p_in)
        return {
            "terminal_out": p_out,
            "terminal_in": p_in,
            "terminal_out_mass": self._density_integral(math_utils, p_out),
            "terminal_in_mass": self._density_integral(math_utils, p_in),
            "ko_prob": ko_prob,
            "ed_ko_cf": ed_ko_cf,
            "survival_prob": survival_prob,
            "extra_fields": extra_fields,
            "mass_diagnostic": float(terminal_mass + np.sum(absorbed)),
        }

    def _readout_at_spot(
        self,
        values: np.ndarray,
        math_utils: QuadratureMath,
        prefactor: float,
        p_lr: int,
        p_ur: int,
        p0: int,
        alpha: float,
        beta: float,
        tau_step: float,
    ) -> np.ndarray:
        """Evaluate the final backward transition AT the spot, x = 0.

        ``values`` is the surface BEFORE that transition. The transition is
        an explicit smooth function of the readout coordinate, so the price
        needs no interpolation between nodes: this is one off-grid row of the
        operator the sweep already applies, exactly as
        ``QuadratureCore._calculate_final_value`` evaluates it for the
        product-agnostic path. Returns a scalar for a single surface and one
        value per row for the stacked indicator surfaces.
        """
        grid = math_utils.grid
        h = float(math_utils.h)
        scale = h if math_utils.integration_rule == "trapezoid" else h / 3.0
        if values.ndim == 1:
            weighted = math_utils.simpson_weights(values, p_lr, p_ur, p0)
        else:
            weighted = math_utils.simpson_weights_many(values, p_lr, p_ur, p0)

        # The convolution multiplies omega, u and the spectral filter in
        # frequency, so the filter can be carried by EITHER factor. Moving it
        # onto the weighted values leaves an analytic kernel that can be
        # evaluated off-lattice, which is the whole point of this readout.
        weighted = math_utils.filtered_weights(weighted)

        # omega(x - x_j) at x = 0, over the extended lattice the convolution
        # implies: entry j pairs with the node at grid[0] + j*h.
        offsets = grid[0] + np.arange(weighted.shape[-1], dtype=float) * h
        omega = np.exp(-(offsets * offsets) / (4.0 * tau_step) + alpha * offsets)
        if values.ndim == 1:
            main = prefactor * scale * float(np.dot(weighted, omega))
        else:
            main = prefactor * scale * (weighted @ omega)
        return main + self._tail_at_spot(values, math_utils, alpha, beta, tau_step)

    def _tail_at_spot(
        self,
        values: np.ndarray,
        math_utils: QuadratureMath,
        alpha: float,
        beta: float,
        tau_step: float,
    ) -> np.ndarray:
        """``_tail_correction`` evaluated at x = 0 rather than on the grid."""
        grid = math_utils.grid
        sqrt_tau = math.sqrt(tau_step)
        u_left = (-grid[0] + 2.0 * tau_step * alpha) / (2.0 * sqrt_tau)
        u_right = (-grid[-1] + 2.0 * tau_step * alpha) / (2.0 * sqrt_tau)
        tail_scale = 0.5 * math.exp(tau_step * (alpha * alpha - beta))
        left = values[0] if values.ndim == 1 else values[:, 0]
        right = values[-1] if values.ndim == 1 else values[:, -1]
        return left * tail_scale * erfc(u_left) + right * tail_scale * erfc(-u_right)

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
        df_fn=None,
    ) -> float:
        if settlement_time is None:
            return 1.0
        delay = max(settlement_time - obs_time, 0.0)
        if is_zero(delay, tol=Tolerance.ZERO):
            return 1.0
        if df_fn is not None:
            return float(df_fn(obs_time + delay)) / float(df_fn(obs_time))
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

    @staticmethod
    def _require_resolvable_time(time: float) -> None:
        """Refuse a positive time the lattice cannot resolve.

        Zero is the valuation instant, decided on the known spot. A positive time is still ahead and carries
        variance, but below ``Tolerance.ZERO`` years (about 3 ms) the recursion has no step to put it on: it used to
        drop such an observation or decide it on the spot (review 2026-09-21 R1).
        """
        if 0.0 < time <= Tolerance.ZERO:
            raise ValidationError(
                f"a time {time:.3g} years after valuation is below the quadrature lattice's time resolution "
                f"({Tolerance.ZERO:g} years): it is still ahead of the valuation instant and carries variance"
            )

    def _merge_times(
        self, ko_times: Sequence[float], ki_times: Sequence[float], maturity: float
    ) -> list[float]:
        merged = []
        for t in sorted(list(ko_times) + list(ki_times)):
            t = float(t)
            self._require_resolvable_time(t)
            if t <= 0.0:
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

    def _term_step_params(
        self, pricing_env, ref_strike, times, rate, div, vol
    ):
        """0-prefixed per-step forward (rate, div, vol) arrays for the
        recursion; collapses back to constant arrays built from the exact
        cumulative scalars when the curves are flat (bit-identical to the
        pre-term code path)."""
        tp = build_quad_term_params(pricing_env, ref_strike, times)
        if (
            np.all(np.abs(tp.rate - rate) <= 1e-12)
            and np.all(np.abs(tp.div - div) <= 1e-12)
            and np.all(np.abs(tp.vol - vol) <= 1e-12)
        ):
            n = len(times) + 1
            return (
                np.full(n, float(rate)),
                np.full(n, float(div)),
                np.full(n, float(vol)),
            )

        def z(a):
            return np.concatenate(([a[0]], np.asarray(a, dtype=float)))

        return z(tp.rate), z(tp.div), z(tp.vol)

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
        extra_cash: Optional[float] = None,
    ) -> float:
        """Return discounted KO payoff for a KO triggered at valuation.

        ``extra_cash`` is folded cash paid with the record's payoff (ship spec D4).
        """
        payoff = float(ko_record.payoff) if ko_record.payoff is not None else 0.0
        if extra_cash is not None:
            payoff += float(extra_cash)
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

    def _resolve_align_cell_stretch(self) -> Optional[float]:
        stretch = getattr(self.params, "align_cell_stretch", None)
        return None if stretch is None else float(stretch)

    @staticmethod
    def _barrier_logs(spot: float, *barrier_groups) -> tuple[float, ...]:
        """Every barrier level the lattice should resolve, in log-moneyness.

        Deliberately the CONTRACTUAL set, not the reachability-filtered one
        that picks the alignment target. The cell width derived from these
        must not move when a remote knock-out flips reachable across a spot
        bump, or the cure would reintroduce the discontinuity it is for.
        Separations are differences of these logs, so the spot they are
        quoted against cancels and the result is spot-independent.
        """
        logs: list[float] = []
        for group in barrier_groups:
            if group is None:
                continue
            if isinstance(group, (list, tuple, np.ndarray)):
                values = list(group)
            else:
                values = [group]
            for barrier in values:
                if barrier is None:
                    continue
                try:
                    level = float(barrier)
                except (TypeError, ValueError):
                    continue
                if level <= 0.0:
                    continue
                try:
                    logs.append(safe_log(level / spot))
                except Exception:
                    continue
        return tuple(sorted(set(logs)))

    def _select_alignment_log(
        self,
        spot: float,
        product: SnowballOption,
        ki_barrier_override: Optional[float] = None,
        ko_candidates_override: Optional[Sequence[float]] = None,
    ) -> Optional[float]:
        ko_candidates: list[float] = []
        ki_candidates: list[float] = []

        if ko_candidates_override is not None:
            ko_candidates.extend(
                float(barrier)
                for barrier in ko_candidates_override
                if barrier is not None and barrier > 0
            )
        else:
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

    def _blend_ki_transition(
        self,
        v_out: np.ndarray,
        v_in: np.ndarray,
        grid: np.ndarray,
        spot_grid: np.ndarray,
        ki_barrier: float,
        spot: float,
        width: float,
        is_reverse: bool,
        ko_weight: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Apply the smoothed discrete-KI transition to the not-yet-KI surface.

        Computes the resolution-aware KI weight (hard mask when smoothing is
        disabled), optionally narrows it by a simultaneous KO weight so KO
        keeps precedence at the same observation, and returns the blended
        ``(1-w)*v_out + w*v_in``. Works for both a single value vector and a
        stacked (rows, grid) indicator array (the weight broadcasts over
        rows). Reduces exactly to the old hard-mask copy at width 0.
        """
        if self._use_cell_average_events():
            # In cell-average mode the complete survive/breach branches are
            # integrated over the threshold-straddling dual cell.  KO has
            # already been applied to both branches when it keeps precedence.
            return self._project_quad_event(
                grid,
                ki_barrier,
                spot,
                v_survive=v_out,
                v_breach=v_in,
                breach_up=is_reverse,
            )

        ki_w = self._event_weight(
            grid, ki_barrier, spot, width, trigger_is_down=not is_reverse
        )
        if ko_weight is not None:
            ki_w = ki_w * (1.0 - ko_weight)
        return (1.0 - ki_w) * v_out + ki_w * v_in

    def _use_cell_average_events(self) -> bool:
        """Whether discrete QUAD events use conservative dual-cell projection."""
        return self.params.event_projection == EventProjectionMode.CELL_AVERAGE

    def _event_weight(
        self,
        grid: np.ndarray,
        barrier: float,
        spot: float,
        width: float,
        *,
        trigger_is_down: bool,
    ) -> np.ndarray:
        """Resolution-aware event weight for indicator recursions.

        Cell-average mode returns the exact breached fraction of each dual
        cell. Legacy nodal mode retains the configured smoothing kernel and
        falls back to a hard mask when smoothing is disabled.
        """
        barrier_log = safe_log(float(barrier) / float(spot))
        if self._use_cell_average_events():
            return breach_fractions(
                grid, barrier_log, breach_up=not trigger_is_down
            )
        weight = self._smooth_step_weight(
            grid,
            barrier,
            spot,
            width,
            trigger_is_down=trigger_is_down,
        )
        if weight is not None:
            return weight
        if trigger_is_down:
            return (grid <= barrier_log).astype(float)
        return (grid >= barrier_log).astype(float)

    @staticmethod
    def _project_quad_event(
        grid: np.ndarray,
        barrier: float,
        spot: float,
        *,
        v_survive,
        v_breach,
        breach_up: bool,
    ) -> np.ndarray:
        """Project a one-threshold event with spatial dimension last.

        The shared projection core expects spot on axis zero. QUAD value
        vectors are one-dimensional while event-stat stacks store spot on the
        final axis, so stacked inputs are moved temporarily and restored.
        """
        survive = np.asarray(v_survive, dtype=float)
        breach = np.asarray(v_breach, dtype=float)
        stacked = survive.ndim > 1
        if stacked:
            survive = np.moveaxis(survive, -1, 0)
            breach = np.moveaxis(breach, -1, 0)
        projected = project_event_values(
            grid,
            safe_log(float(barrier) / float(spot)),
            survive,
            breach,
            breach_up,
        )
        return np.moveaxis(projected, 0, -1) if stacked else projected

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
