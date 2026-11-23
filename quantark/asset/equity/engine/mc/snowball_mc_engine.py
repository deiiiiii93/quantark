"""
Monte Carlo pricing engine for Snowball (autocallable) options.

This engine prices snowball options using Monte Carlo simulation with support for:
- Standard and reverse snowball structures
- KO-reset snowball structures (pre/post KO schedules)
- Discrete KO observations with time-varying barriers and rates
- Discrete or continuous KI monitoring
- INSTANT or EXPIRY coupon payment timing
- Vectorized NumPy operations for efficiency
- Optional Dask parallelization for batch processing
"""

import math
import warnings
from contextlib import contextmanager, nullcontext
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from quantark.montecarlo.std_error import sample_std_error
from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.engine.capabilities import SettlementSupport
from quantark.asset.equity.engine.event_stats import EventStatistics
from quantark.asset.equity.engine.settlement_support import first_schedule_payment_time, first_schedule_terminal_timing
from quantark.asset.equity.engine.mc.autocallable_payment_timings import (
    resolve_autocallable_payment_timings,
)
from quantark.asset.equity.engine.mc._event_stats_recorder import (
    STATS_RECORDS_KEY,
    EventStatsRecordingMixin,
    aggregate_records,
    control_corrected_record,
    grouped_sums,
    row_sums,
)
from quantark.asset.equity.engine.mc._snowball_event_statistics import (
    ko_reset_event_statistics,
    snowball_event_statistics,
)
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.process.bsm.qmc_path_generator import GBMPathGenerator
from quantark.montecarlo.conditional_snowball import (
    conditional_standard_snowball_event_record,
    conditional_standard_snowball_moments,
)
from quantark.montecarlo.qmc_rqmc_driver import RQMCRunSpec, run_rqmc
from quantark.asset.equity.process.bsm.qmc_sobol import (
    PseudoRandomNormalGenerator,
    SobolNormalGenerator,
)
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.asset.equity.product.option.ko_reset_snowball_option import (
    KnockOutResetSnowballOption,
)
from quantark.asset.equity.product.option.accrual import knock_in_anchor_for, leg_annualized
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord
from quantark.asset.equity.settlement import (
    CashflowKind,
    SettlementRequest,
    SettlementResolver,
)
from quantark.asset.equity.engine.mc.ki_bridge import (
    continuous_ki_step_count,
    validate_ki_bridge_nodes_per_day,
)
from quantark.asset.equity.engine.mc.term_inputs import build_mc_term_inputs, make_df_fn
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import DayCountConvention
from quantark.util.enum import (
    CouponPayType,
    ObservationType,
    PostKOScheduleMode,
)
from quantark.util.enum.engine_enums import EngineType, MonteCarloMethod
from quantark.util.exceptions import PricingError, ValidationError
from quantark.util.numerical import Tolerance, merge_time_nodes, node_index, safe_log, same_instant

# Optional Dask import
from quantark.asset.equity.engine.mc.autocallable_dask_batch import (
    run_autocallable_dask_batches,
)

try:
    from dask import compute, delayed  # noqa: F401 — availability probe

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False


class _FoldedPayoffMixin:
    """Cash-leg folding for the autocallable MC engines (ship spec 2026-10-21 D4, D5, D8, D11c, D14).

    ``price_folded`` runs the engine's own ``price`` with the overlay held for that one call
    (``_fold``, None on every other call). Each per-path payoff kernel binds it to the records it
    actually pays -- ``(schedule, determination, settlement)`` of every knock-out record and of each
    maturity branch, through ``FoldOverlay.bind_ko`` / ``bind_terminal`` (fail closed) -- and adds
    the cash where the record's own payoff goes, before the path's settlement discount. Every route
    that prices through the kernel (serial, Dask batches, RQMC batches) therefore folds; the
    estimators with their own assembly (LocalVol one-step survival, the Heston conditional moments)
    fold at their own sites.
    """

    supports_cash_leg_folding = True
    #: the overlay of a running ``price_folded``; None on every other call
    _fold = None

    @contextmanager
    def _folding(self, overlay):
        """Hold ``overlay`` for the solves run inside the block; never left behind."""
        self._fold = overlay
        try:
            yield
        finally:
            self._fold = None

    def price_folded(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        overlay,
        *,
        lifecycle_state=None,
    ) -> float:
        """Per-unit NPV with ``overlay``'s cash legs folded into each path's payoff.

        One ``price`` on the same paths: knock-out cash is added to the knock-out record's payoff
        of every path that knocks out there, maturity cash to the payoff of every path ending on
        that branch, both before the path's settlement discount. ``overlay.constant_pv`` is not
        included. An empty overlay is ``price()`` bitwise.
        """
        if overlay.is_empty:
            return self.price(product, pricing_env, lifecycle_state=lifecycle_state)
        if (
            isinstance(product, KnockOutResetSnowballOption)
            and product.post_ko_mode != PostKOScheduleMode.ABSOLUTE
        ):
            self._refuse_rebased_fold()
        with self._folding(overlay):
            return self.price(product, pricing_env, lifecycle_state=lifecycle_state)

    def fold_drilldown_refusal(self, product, pricing_env) -> Optional[str]:
        """Adaptive RQMC and Dask batch routes refuse DRILLDOWN (ship spec D11c)."""
        if self.method == MonteCarloMethod.RANDOMIZED_QUASI:
            return (
                "adaptive RQMC stops on the batch statistics of the payoff it prices, so a "
                "folded and a plain solve may run different path sets; per-leg differences "
                "would mix them"
            )
        if self.use_dask and self.num_batches > 1:
            return (
                "Dask batch pricing reduces separately priced batches; per-leg differences "
                "of folded and plain solves are refused on it"
            )
        return None

    @staticmethod
    def _refuse_rebased_fold():
        from quantark.execution.errors import CapabilityError

        raise CapabilityError(
            "a REBASED KO-reset schedule settles at knock-in-dependent times; cash legs "
            "cannot be folded onto it"
        )

    def _refuse_expired_fold(self) -> None:
        """At zero time to maturity the event structure is trivial: folded cash has no event."""
        if self._fold is not None:
            from quantark.execution.errors import CapabilityError

            raise CapabilityError(
                "an expired product has no knock-out or maturity event left to fold cash onto"
            )

    def _bound_fold(self, records, no_ki_timing, ki_timing):
        """``(KO amounts aligned to records, alive cash, knocked-in cash)``, or None unfolded."""
        fold = self._fold
        if fold is None:
            return None
        return (
            fold.bind_ko(records),
            fold.bind_terminal("no_ki", *no_ki_timing),
            fold.bind_terminal("ki", *ki_timing),
        )

    def _conditional_fold(self, product, pricing_env, maturity: float) -> dict:
        """The conditional-moment estimator's fold: KO cash at each record's settlement (its
        knock-out discount) and branch cash at the maturity it discounts the terminal leg to."""
        if self._fold is None:
            return {}
        profile = product.get_ko_observation_profile(pricing_env)
        settle = np.asarray(profile["settlement_times"], dtype=float)
        records = [
            ("main", float(t), float(s))
            for t, s in zip(profile["observation_times"], settle)
        ]
        ko, no_ki, ki = self._bound_fold(
            records, (float(maturity), float(maturity)), (float(maturity), float(maturity))
        )
        return {"ko_cash": ko, "no_ki_cash": no_ki, "ki_cash": ki}


@dataclass
class SnowballMCResult:
    """Result container for Snowball MC pricing."""

    price: float
    std_error: float
    num_paths: int
    ko_probability: float
    v0_probability: float
    v1_probability: float
    avg_ko_time: Optional[float] = None
    batches_used: Optional[int] = None


def _knock_out_first(ko_times: np.ndarray, ki_times: np.ndarray, ki_continuous: bool) -> np.ndarray:
    """Per path: whether the first knock-out is not preceded by the first knock-in.

    The knock-out wins a tie with a DISCRETE knock-in, which is observed at the same instant. A continuously
    monitored one is detected inside the step that the knock-out observation closes, so an equal grid time means
    the knock-in came first -- except at time zero, where no step precedes: a level already breached at valuation
    is observed at that instant with the observations there, and the knock-out wins as the decision at valuation
    has it. The tolerance only absorbs two spellings of one observation time.
    """
    if ki_continuous:
        return (ko_times < ki_times) | ((ko_times == 0.0) & (ki_times == 0.0))
    return ko_times <= ki_times + 1e-12


def _ki_level_breached(spots: np.ndarray, ki_barrier: float, is_reverse: bool) -> np.ndarray:
    """Per spot: whether it is at or beyond a knock-in level (above it for a reverse product)."""
    return spots >= ki_barrier if is_reverse else spots <= ki_barrier


class SnowballMCEngine(_FoldedPayoffMixin, EventStatsRecordingMixin, BaseEngine):
    """
    Monte Carlo pricing engine for Snowball (autocallable) options.

    Supports three Monte Carlo methods:
    - PSEUDO: Standard Monte Carlo with pseudorandom numbers
    - QUASI: Quasi-Monte Carlo with Sobol sequences
    - RANDOMIZED_QUASI: Randomized QMC with adaptive batching

    Usage:
        # Preferred: Two-level enum pattern
        engine = SnowballMCEngine(
            params=MCParams(num_paths=100000, time_steps=252),
            method=EngineType.MONTE_CARLO(MonteCarloMethod.QUASI)
        )

        # Alternative: Direct method enum
        engine = SnowballMCEngine(
            params=MCParams(num_paths=100000),
            method=MonteCarloMethod.QUASI
        )

        # With optional Dask parallelization
        engine = SnowballMCEngine(
            params=MCParams(num_paths=100000),
            use_dask=True,
            num_batches=8
        )

    The engine creates a GBMPathGenerator internally based on the pricing
    environment and product observation schedule.
    """

    # Request-scoped term-structure context (set at every public pricing
    # entry point from ITS pricing_env; never carried across calls).
    # Private helpers called outside a public entry fall back to their
    # scalar arguments (None ctx) or fail loudly (None _df) rather than
    # silently reusing a previous environment. Engine instances are not
    # safe for concurrent pricing calls (pre-existing engine contract).
    _term_ctx = None
    _df = None
    _payment_timings = None


    DEFAULT_METHOD = MonteCarloMethod.PSEUDO
    settlement_support = SettlementSupport.EVENT_AND_TERMINAL

    def __init__(
        self,
        params: Optional[MCParams] = None,
        method: Union[str, MonteCarloMethod, tuple, None] = None,
        use_dask: bool = False,
        num_batches: int = 4,
        *,
        ki_bridge_nodes_per_day: int = 1,
    ):
        """
        Initialize Snowball Monte Carlo engine.

        Args:
            params: Monte Carlo configuration parameters (MCParams)
            method: Monte Carlo method selection, one of:
                - EngineType.MONTE_CARLO(MonteCarloMethod.XXX) (preferred)
                - MonteCarloMethod.XXX
                - String: "pseudo", "quasi", "randomized_quasi"
                - None: defaults to MonteCarloMethod.PSEUDO
            use_dask: Enable Dask parallel processing (requires Dask installed)
            num_batches: Number of batches for parallel processing
            ki_bridge_nodes_per_day: Continuous-KI monitoring nodes per business
                day (opt-in; default 1 is the historical grid, bit for bit).
                The nodes are contractual: the SDE steps on them and the
                Brownian bridge runs between them, so a larger value shrinks
                the bridge interval -- needed under local/stochastic vol, where
                the one-day bridge is biased (see ``ki_bridge``).

        Raises:
            ValidationError: If method is invalid or params are invalid
        """
        self.ki_bridge_nodes_per_day = validate_ki_bridge_nodes_per_day(
            ki_bridge_nodes_per_day
        )
        if params is None:
            params = MCParams()

        if not isinstance(params, MCParams):
            raise ValidationError(
                f"params must be MCParams instance, got {type(params).__name__}"
            )

        super().__init__(params)

        # Parse method
        if method is None:
            self.method = self.DEFAULT_METHOD
        elif isinstance(method, tuple):
            engine_type, mc_method = method
            if engine_type != EngineType.MONTE_CARLO:
                raise ValidationError(
                    f"Expected EngineType.MONTE_CARLO, got {engine_type}"
                )
            if not isinstance(mc_method, MonteCarloMethod):
                raise ValidationError(
                    f"Expected MonteCarloMethod, got {type(mc_method).__name__}"
                )
            self.method = mc_method
        elif isinstance(method, MonteCarloMethod):
            self.method = method
        elif isinstance(method, str):
            try:
                self.method = MonteCarloMethod[method.upper()]
            except KeyError:
                valid_methods = [m.name for m in MonteCarloMethod]
                raise ValidationError(
                    f"Invalid method string '{method}'. Valid methods: {valid_methods}"
                )
        else:
            raise ValidationError(
                f"Invalid method type {type(method).__name__}. "
                "Expected MonteCarloMethod, tuple, str, or None"
            )

        # Dask configuration
        self.use_dask = use_dask and DASK_AVAILABLE
        if use_dask and not DASK_AVAILABLE:
            warnings.warn(
                "Dask requested but not installed. Falling back to single-threaded NumPy.",
                UserWarning,
            )
        self.num_batches = num_batches

        # Result storage
        self._last_result: Optional[SnowballMCResult] = None

    def price(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        """
        Price a Snowball option using Monte Carlo simulation.

        Args:
            product: Snowball option to price
            pricing_env: Pricing environment with market data

        Returns:
            Option price

        Raises:
            PricingError: If product is not a SnowballOption
            ValidationError: If pricing parameters are invalid
        """
        if not isinstance(product, (SnowballOption, KnockOutResetSnowballOption)):
            raise PricingError(
                "SnowballMCEngine only supports SnowballOption or "
                f"KnockOutResetSnowballOption, got {type(product).__name__}"
            )

        # Extract market data
        S = pricing_env.spot
        T = (
            product.get_max_maturity_time(pricing_env)
            if isinstance(product, KnockOutResetSnowballOption)
            else product.get_maturity(pricing_env)
        )
        r = pricing_env.get_rate(T)
        q = pricing_env.get_div_yield(T)
        sigma = pricing_env.get_vol(product.strike, T)

        self._validate_inputs(S, T, r, q, sigma, product)
        # Term-structure context for this pricing call (see term_inputs.py)
        self._term_ctx = (pricing_env, product.strike)
        self._df = make_df_fn(pricing_env)
        self._prepare_payment_timings(product, pricing_env)


        # Zero time to maturity. Exactly zero: a positive maturity, however short, still carries variance and is
        # simulated (a 1e-10-year cut decided an observation 3 ms ahead on the spot, review 2026-09-21 R1).
        if T <= 0.0:
            self._refuse_expired_fold()
            self._note_stats_expired()
            knocked_in = bool(getattr(product, "_otc_lifecycle_knocked_in", False))
            if type(product) is SnowballOption:
                # The observations still pending at this instant are decided on the known spot, as
                # _build_time_grid decides them (index -1) when more of the claim lies ahead. Going
                # straight to the carried state's payoff priced a claim that knocks in or out at the
                # last observation as if it had survived.
                decision = product.decide_observations_at_valuation(
                    S, pricing_env, knocked_in=knocked_in
                )
                if decision.knocked_out:
                    timings = self._payment_timings
                    instant = (
                        product.accrual_config.coupon_pay_type == CouponPayType.INSTANT
                    )
                    payment_time = (
                        timings.observation_payment_times[decision.ko_index]
                        if instant
                        else timings.terminal.payment_time
                    )
                    return float(decision.ko_record.payoff) * self._df(
                        float(payment_time)
                    ) / self._df(float(T))
                knocked_in = decision.knocked_in
            elif isinstance(product, KnockOutResetSnowballOption):
                product.require_alive(pricing_env, knocked_in)
                # Same decision on the schedule the knock-in state puts in force, as the path loop
                # applies it (a knock-in puts this instant's post-KI observation in force too).
                decision = product.decide_observations_at_valuation(
                    S, pricing_env, knocked_in=knocked_in
                )
                if decision.knocked_out:
                    return self._ko_reset_decided_knock_out(
                        product, pricing_env, decision, float(T)
                    )
                knocked_in = decision.knocked_in
            payoff = product.get_payoff(
                S, pricing_env, knocked_in=knocked_in
            )
            if self._payment_timings is None:
                payment_time = self._resolve_payment_time(
                    product,
                    pricing_env,
                    float(T),
                    CashflowKind.TERMINAL,
                    cashflow_id="ko_reset_near_expiry",
                )
                return payoff * (
                    self._df(payment_time) / self._df(float(T))
                )
            return payoff * self._payment_timings.terminal.delay_df

        # Price using appropriate method
        if isinstance(product, KnockOutResetSnowballOption):
            if self.method == MonteCarloMethod.RANDOMIZED_QUASI:
                result = self._price_ko_reset_rqmc(
                    product, pricing_env, S, T, r, q, sigma
                )
            elif self.use_dask and self.num_batches > 1:
                result = self._price_ko_reset_parallel(
                    product, pricing_env, S, T, r, q, sigma
                )
            else:
                result = self._price_ko_reset_mc_or_qmc(
                    product, pricing_env, S, T, r, q, sigma
                )
        else:
            if self.method == MonteCarloMethod.RANDOMIZED_QUASI:
                result = self._price_rqmc(product, pricing_env, S, T, r, q, sigma)
            elif self.use_dask and self.num_batches > 1:
                result = self._price_parallel(product, pricing_env, S, T, r, q, sigma)
            else:
                result = self._price_mc_or_qmc(product, pricing_env, S, T, r, q, sigma)

        return self._complete_price(product, result)

    def _ko_reset_decided_knock_out(self, product, pricing_env, decision, maturity: float) -> float:
        """A KO-reset knock-out decided at the valuation instant, paid as ``_compute_ko_schedule_payoffs``
        pays it: at the record's own settlement under INSTANT, else with the note it ends -- a first-schedule
        knock-out with the note that ends with schedule 1 (spec D30, D33), a
        second-schedule one with the terminal payment. The first-schedule payment is the grid engines'
        (``first_schedule_payment_time``): reached only at zero time to maturity, where schedule 1 ends at the
        product terminal, it is that terminal's payment."""
        profile = (
            product.get_pre_ko_observation_profile(pricing_env)
            if decision.ko_regime == "pre"
            else product.get_post_ko_observation_profile(pricing_env)
        )
        if product.accrual_config.coupon_pay_type == CouponPayType.INSTANT:
            payment_time = self._resolve_payment_time(
                product,
                pricing_env,
                float(decision.ko_record.observation_time),
                CashflowKind.REDEMPTION,
                record=profile["records"][decision.ko_index],
                cashflow_id=f"ko_reset[{decision.ko_index}]",
            )
        elif decision.ko_regime == "pre":
            payment_time = float(first_schedule_payment_time(product, pricing_env))
        else:
            payment_time = self._resolve_payment_time(
                product,
                pricing_env,
                maturity,
                CashflowKind.TERMINAL,
                cashflow_id="ko_reset_terminal",
            )
        return float(decision.ko_record.payoff) * self._df(float(payment_time)) / self._df(maturity)

    def _complete_price(self, product, result) -> float:
        """price() postamble shared with the session adaptive path."""
        self._last_result = result
        # Negative PV is valid for some structures (e.g., principal-excluded notes)
        # where the payoff is effectively "coupon minus embedded option loss".
        # (A folded solve carries cash legs, which may outweigh the product: the check is the
        # product's own.)
        if result.price < 0 and product.payoff_config.include_principal and self._fold is None:
            raise PricingError(f"Negative price computed: {result.price}")
        return result.price

    def _prepare_payment_timings(self, product, pricing_env) -> None:
        """Resolve standard Snowball KO and terminal payment timing once."""
        if isinstance(product, KnockOutResetSnowballOption):
            self._payment_timings = None
            return
        records = product.resolve_ko_observations(pricing_env)
        # A knock-out redemption pays at its record's settlement whatever the
        # accrual convention — this is the pre-settlement-feature behavior
        # (the profile's settlement_times were record-level, never gated on
        # pay type) and the certified KO-settlement ruling.
        self._payment_timings = resolve_autocallable_payment_timings(
            product,
            pricing_env,
            records,
            event_paid=True,
        )

    @staticmethod
    def _resolve_payment_time(
        product,
        pricing_env,
        determination_time: float,
        kind: CashflowKind,
        *,
        record: Optional[ObservationRecord] = None,
        cashflow_id: str,
    ) -> float:
        timing = SettlementResolver.resolve_contingent(
            product,
            SettlementRequest(
                kind=kind,
                determination_time=float(determination_time),
                explicit_payment_date=(
                    None if record is None else record.settlement_date
                ),
                explicit_payment_time=(
                    None if record is None else record.settlement_time
                ),
                cashflow_id=cashflow_id,
            ),
            pricing_env,
        )
        return float(timing.payment_time)

    @classmethod
    def _resolve_payment_time_array(
        cls,
        product,
        pricing_env,
        determination_times,
        kind: CashflowKind,
        *,
        records: Optional[List[ObservationRecord]] = None,
        record_indices=None,
        cashflow_id: str,
    ) -> np.ndarray:
        """Resolve the small set of unique event nodes, then vectorize."""
        times = np.asarray(determination_times, dtype=float)
        indices = (
            np.full(times.shape, -1, dtype=int)
            if record_indices is None
            else np.asarray(record_indices, dtype=int)
        )
        pairs = np.column_stack((times.ravel(), indices.ravel()))
        unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
        resolved = np.empty(unique.shape[0], dtype=float)
        for index, (time, record_index_raw) in enumerate(unique):
            record_index = int(record_index_raw)
            record = (
                None
                if records is None or record_index < 0
                else records[record_index]
            )
            resolved[index] = cls._resolve_payment_time(
                product,
                pricing_env,
                float(time),
                kind,
                record=record,
                cashflow_id=(
                    f"{cashflow_id}[{record_index}]"
                    if record_index >= 0
                    else cashflow_id
                ),
            )
        return resolved[inverse].reshape(times.shape)

    def _rqmc_streams_per_step(self) -> int:
        """Random streams consumed per time step in RQMC mode (session
        plan metadata; model-family subclasses override)."""
        return 1

    def _rqmc_substep_factor(self) -> int:
        """SDE steps per contractual interval (session plan metadata; the
        sub-observation refinement mixin overrides). Draw dimension scales
        with this factor while path nodes stay contractual."""
        return 1

    def _rqmc_dimension(self, contractual_time_steps: int) -> int:
        return (
            self._rqmc_streams_per_step()
            * int(contractual_time_steps)
            * self._rqmc_substep_factor()
        )

    def _rqmc_randomization_key(self, contractual_time_steps: int):
        return (
            "scrambled_sobol",
            int(self.params.seed),
            self._rqmc_scheme_label(),
            int(contractual_time_steps),
            self._rqmc_substep_factor(),
        )

    def _rqmc_scheme_label(self) -> str:
        """Session plan scheme identifier: class plus model scheme, so two
        configurations with different draw pipelines never share a plan
        fingerprint (code-gate finding 2026-07-16)."""
        scheme = getattr(self, "scheme", None)
        suffix = f"-{scheme.name.lower()}" if hasattr(scheme, "name") else ""
        label = f"{type(self).__qualname__}{suffix}/rqmc-native/v1"
        # opt-in grid refinements name themselves; the default label is unchanged
        if self._ki_bridge_nodes_per_day() != 1:
            label = f"{label}+kib:{self._ki_bridge_nodes_per_day()}"
        return label

    def _ki_bridge_nodes_per_day(self) -> int:
        """Continuous-KI nodes per business day (1 for engines built before the keyword existed)."""
        return getattr(self, "ki_bridge_nodes_per_day", 1)

    def build_rqmc_session_spec(self, product, pricing_env):
        """Execution-framework seam: the price() preamble plus the RQMC run
        description, or None when a direct call would not take the adaptive
        RQMC route (non-RQMC method, near-expiry shortcut) - the session
        then falls back to the native legacy call."""
        if self.method != MonteCarloMethod.RANDOMIZED_QUASI:
            return None
        if not isinstance(product, (SnowballOption, KnockOutResetSnowballOption)):
            raise PricingError(
                "SnowballMCEngine only supports SnowballOption or "
                f"KnockOutResetSnowballOption, got {type(product).__name__}"
            )
        S = pricing_env.spot
        T = (
            product.get_max_maturity_time(pricing_env)
            if isinstance(product, KnockOutResetSnowballOption)
            else product.get_maturity(pricing_env)
        )
        r = pricing_env.get_rate(T)
        q = pricing_env.get_div_yield(T)
        sigma = pricing_env.get_vol(product.strike, T)
        self._validate_inputs(S, T, r, q, sigma, product)
        self._term_ctx = (pricing_env, product.strike)
        self._df = make_df_fn(pricing_env)
        self._prepare_payment_timings(product, pricing_env)
        if T <= 0.0:
            return None
        if isinstance(product, KnockOutResetSnowballOption):
            return self._ko_reset_rqmc_spec(
                product, pricing_env, S, T, r, q, sigma
            )
        return self._rqmc_spec(product, pricing_env, S, T, r, q, sigma)

    def calculate_event_stats(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        *,
        streams: Optional[frozenset] = None,
    ) -> Optional[EventStatistics]:
        """
        Event statistics (``EventStatistics``) on ``price()``'s own paths.

        Runs this engine's own ``price`` under a statistics recorder (unified event statistics spec AC2, D14,
        D23): every payoff-kernel batch of that price deposits its outcome sums, so the statistics describe
        exactly the paths, seeds, bridge seeds, batches and estimator of the price on every route (serial,
        Dask batches, adaptive RQMC, the conditional-moment estimators), and ``pv`` IS the value ``price``
        returned. Every outcome is summed directly over its own paths (the residuals are the type's own);
        knock-in counts only before any knock-out; the continuous knock-in table spreads each step's
        first-crossing mass over the trading days it overlaps (spec D18, labelled ``step_overlap``); REBASED
        KO-reset carries the post-KI offset table and the absolute termination distribution on the
        deterministic candidate axis (spec D13). Standard errors come from independent batch means (spec
        AC21): one batch, or deterministic QMC batches, give none, with the reason in ``unavailable``.

        The Heston-SLV conditional control prices ``native - sampled proxy + exact proxy``; its statistics are
        the same estimator outcome by outcome (each batch's native record minus the proxy paths' record plus
        the proxy's exact conditional outcomes), so the outcomes reconcile to the price at round-off. Such an
        estimate is unbiased but unbounded: one that breaks a probability bound (a rare outcome at a small path
        count) raises ``PricingError`` rather than being clipped.

        ``streams`` is accepted for the engine contract and ignored: every number comes from the same paths,
        so nothing is pruned. At zero time to maturity ``price`` decides the claim on the spot and no path is
        priced: the statistics are ``None`` (the documented expired rule). The batch records stay on
        ``_last_stats_records``.
        """
        if not isinstance(product, (SnowballOption, KnockOutResetSnowballOption)):
            return None
        with self._recording_event_stats() as recorder:
            pv = self.price(product, pricing_env)
        if recorder.expired:
            return None
        if not recorder.records:
            raise PricingError(
                f"{type(self).__name__}.price recorded no priced batch for the event statistics"
            )
        records = tuple(recorder.records)
        totals = aggregate_records(records)
        if isinstance(product, KnockOutResetSnowballOption):
            return ko_reset_event_statistics(self, product, pricing_env, float(pv), totals, records)
        try:
            return snowball_event_statistics(self, product, pricing_env, float(pv), totals, records)
        except ValidationError as exc:
            if not any(record.get("kind") == "control_corrected" for record in records):
                raise
            # A control-variate estimate of one outcome (native - sampled proxy + exact proxy) is unbiased but not
            # bounded: on a rare outcome at a small path count it can fall below zero. It is never clipped.
            raise PricingError(
                "Heston-SLV conditional-control statistics: a control-corrected outcome estimate (native - "
                f"sampled proxy + exact proxy, the price's own estimator) breaks its probability bounds: {exc}. "
                "The estimator's per-outcome noise exceeds that outcome's mass at this path count; more paths "
                "per batch give valid statistics."
            ) from exc

    @staticmethod
    def _resolve_payment_timing(
        product,
        pricing_env,
        determination_time: float,
        kind: CashflowKind,
        *,
        record: Optional[ObservationRecord] = None,
        cashflow_id: str,
    ):
        """The full resolved timing (dates included) of the request ``_resolve_payment_time`` resolves."""
        return SettlementResolver.resolve_contingent(
            product,
            SettlementRequest(
                kind=kind,
                determination_time=float(determination_time),
                explicit_payment_date=(
                    None if record is None else record.settlement_date
                ),
                explicit_payment_time=(
                    None if record is None else record.settlement_time
                ),
                cashflow_id=cashflow_id,
            ),
            pricing_env,
        )

    def _snowball_batch_record(
        self,
        *,
        payoffs: np.ndarray,
        settlement_times: np.ndarray,
        is_ko: np.ndarray,
        is_v0: np.ndarray,
        is_v1: np.ndarray,
        first_ko_idx: np.ndarray,
        ko_times: np.ndarray,
        ki_triggered: np.ndarray,
        first_ki_idx: np.ndarray,
        ki_row_times: np.ndarray,
        ki_path_times: Optional[np.ndarray],
        ki_continuous: bool,
        carried_knock_in: bool,
        breached_at_valuation: bool,
        principal: float,
    ) -> dict:
        """One native batch's outcome sums (read-only on the kernel's finished arrays).

        Knock-in statistics count a knock-in only before any knock-out (DESIGN §4.5): ``ki_first_count`` per
        KI row (discrete) or MC step (continuous), ``ki_time_sum`` their first knock-in times (a level
        breached at valuation, or a carried knock-in, at 0); ``ko_clean_count`` the knock-outs with no knock-in
        before them; ``v1_disc_loss`` the discounted knocked-in downside ``principal - redemption``.
        """
        dfs = np.asarray(self._df(settlement_times), dtype=float)
        discounted = payoffs * dfs
        n_ko = len(ko_times)
        if carried_knock_in:
            ki_before_ko = ki_triggered.copy()
            ki_time = np.zeros(len(payoffs), dtype=float)
        elif ki_path_times is None:
            ki_before_ko = np.zeros_like(ki_triggered)
            ki_time = np.zeros(len(payoffs), dtype=float)
        else:
            ko_path_times = np.where(is_ko, ko_times[np.maximum(first_ko_idx, 0)], np.inf)
            ki_before_ko = ki_triggered & ~(
                is_ko & _knock_out_first(ko_path_times, ki_path_times, ki_continuous)
            )
            ki_time = np.where(ki_before_ko, ki_path_times, 0.0)
        ki_rows = 0 if carried_knock_in else len(ki_row_times)
        loss = np.where(is_v1, (principal - payoffs) * dfs, 0.0)
        return {
            "kind": "native",
            "n": int(len(payoffs)),
            "pv_sum": float(discounted.sum()),
            "ko_count": row_sums(first_ko_idx, n_ko, is_ko),
            "ko_clean_count": row_sums(first_ko_idx, n_ko, is_ko & ~ki_before_ko),
            "ko_disc_cash": row_sums(first_ko_idx, n_ko, is_ko, discounted),
            "ko_cash": row_sums(first_ko_idx, n_ko, is_ko, payoffs),
            "v0_count": float(is_v0.sum()),
            "v1_count": float(is_v1.sum()),
            "v0_disc_cash": float(discounted[is_v0].sum()),
            "v1_disc_cash": float(discounted[is_v1].sum()),
            "v0_cash": float(payoffs[is_v0].sum()),
            "v1_cash": float(payoffs[is_v1].sum()),
            "v1_disc_loss": float(loss.sum()),
            "ki_no_ko_count": float((ki_triggered & ~is_ko).sum()),
            "ki_before_ko_count": float(ki_before_ko.sum()),
            "ki_time_sum": float(ki_time.sum()),
            "ki_first_count": row_sums(first_ki_idx, ki_rows, ki_before_ko),
            "ki_row_times": np.asarray(ki_row_times, dtype=float),
            "ki_breached_at_valuation": bool(breached_at_valuation),
        }

    def _ko_reset_batch_record(
        self,
        *,
        payoffs: np.ndarray,
        settlement_times: np.ndarray,
        is_pre_ko: np.ndarray,
        is_post_ko: np.ndarray,
        is_v0: np.ndarray,
        is_v1: np.ndarray,
        first_pre_idx: np.ndarray,
        first_post_idx: np.ndarray,
        n_pre: int,
        n_post: int,
        ko_time: np.ndarray,
        terminal_determination: np.ndarray,
        ki_triggered: np.ndarray,
        first_ki_idx: np.ndarray,
        ki_row_times: np.ndarray,
        ki_time: np.ndarray,
        carried_knock_in: bool,
        breached_at_valuation: bool,
        rebased: bool,
        principal: float,
    ) -> dict:
        """One KO-reset batch's outcome sums (read-only on the kernel's finished arrays). Second-schedule
        knock-outs count per schedule-2 record (an offset under REBASED). A knock-in counts only before any
        knock-out: knocked in and not knocked out on the first schedule (a second-schedule knock-out follows a
        knock-in). REBASED: the path-dependent outcomes (second-schedule knock-outs, both maturity branches)
        are also grouped by their actual (determination, payment) for the absolute termination distribution."""
        dfs = np.asarray(self._df(settlement_times), dtype=float)
        discounted = payoffs * dfs
        ki_before_ko = ki_triggered & ~is_pre_ko
        first_ki_time = np.zeros(len(payoffs), dtype=float) if carried_knock_in else np.where(
            ki_before_ko, ki_time, 0.0
        )
        loss = np.where(is_v1, (principal - payoffs) * dfs, 0.0)
        record = {
            "kind": "native",
            "n": int(len(payoffs)),
            "pv_sum": float(discounted.sum()),
            "pre_count": row_sums(first_pre_idx, n_pre, is_pre_ko),
            "pre_disc_cash": row_sums(first_pre_idx, n_pre, is_pre_ko, discounted),
            "pre_cash": row_sums(first_pre_idx, n_pre, is_pre_ko, payoffs),
            "post_count": row_sums(first_post_idx, n_post, is_post_ko),
            "post_disc_cash": row_sums(first_post_idx, n_post, is_post_ko, discounted),
            "post_cash": row_sums(first_post_idx, n_post, is_post_ko, payoffs),
            "v0_count": float(is_v0.sum()),
            "v1_count": float(is_v1.sum()),
            "v0_disc_cash": float(discounted[is_v0].sum()),
            "v1_disc_cash": float(discounted[is_v1].sum()),
            "v0_cash": float(payoffs[is_v0].sum()),
            "v1_cash": float(payoffs[is_v1].sum()),
            "v1_disc_loss": float(loss.sum()),
            "ki_no_ko_count": float((ki_triggered & ~(is_pre_ko | is_post_ko)).sum()),
            "ki_before_ko_count": float(ki_before_ko.sum()),
            "ki_time_sum": float(first_ki_time.sum()),
            "ki_first_count": row_sums(first_ki_idx, len(ki_row_times), ki_before_ko),
            "ki_row_times": np.asarray(ki_row_times, dtype=float),
            "ki_breached_at_valuation": bool(breached_at_valuation),
        }
        if rebased:
            ledger = grouped_sums("post", is_post_ko, ko_time, settlement_times, discounted, payoffs)
            ledger.update(grouped_sums("maturity", is_v0, terminal_determination, settlement_times, discounted,
                                       payoffs))
            ledger.update(grouped_sums("ki_settlement", is_v1, terminal_determination, settlement_times,
                                       discounted, payoffs))
            record["ledger"] = ledger
        return record

    def _validate_inputs(
        self,
        S: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        product: BaseEquityProduct,
    ) -> None:
        """Validate pricing inputs."""
        if S <= 0:
            raise ValidationError(f"Spot price must be positive, got {S}")
        if T < 0:
            raise ValidationError(f"Time to maturity must be non-negative, got {T}")
        if sigma <= 0:
            raise ValidationError(f"Volatility must be positive, got {sigma}")
        if not np.isfinite(q):
            raise ValidationError(f"Dividend yield must be finite, got {q}")

        if isinstance(product, SnowballOption):
            # Validate observation schedule exists
            if product.barrier_config.ko_observation_type == ObservationType.DISCRETE:
                if (
                    product.barrier_config.ko_observation_schedule is None
                    and product.barrier_config.ko_observation_dates is None
                ):
                    raise ValidationError(
                        "KO observation schedule or dates required for discrete monitoring"
                    )
        if isinstance(product, KnockOutResetSnowballOption):
            pre_config = product.barrier_config
            if pre_config.ko_observation_type == ObservationType.DISCRETE:
                if (
                    pre_config.ko_observation_schedule is None
                    and pre_config.ko_observation_dates is None
                ):
                    raise ValidationError(
                        "Pre-KI KO observation schedule or dates required for discrete monitoring"
                    )
            post_config = product.post_barrier_config
            if (
                post_config.ko_observation_schedule is None
                and post_config.ko_observation_dates is None
            ):
                raise ValidationError(
                    "Post-KI KO observation schedule or dates required for discrete monitoring"
                )

    def _build_time_grid(
        self, product: SnowballOption, pricing_env: PricingEnvironment, T: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build time grid aligned with observation dates.

        Returns:
            Tuple of:
            - all_times: Sorted unique observation times including maturity
            - dt_array: Time increments between observations
            - ko_indices: Indices into all_times for KO observations
            - ki_indices: Indices into all_times for KI observations (or fine grid)
        """
        # Get KO observation times
        ko_profile = product.get_ko_observation_profile(pricing_env)
        ko_times = ko_profile["observation_times"]

        # Get KI observation times (if applicable)
        ki_times = []
        ki_continuous = (
            product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
            or product.barrier_config.ki_continuous
        )

        if product.has_ki_barrier:
            if ki_continuous:
                # Generate fine grid for continuous monitoring
                num_ki_steps = continuous_ki_step_count(
                    pricing_env.bus_days_in_year, T, self._ki_bridge_nodes_per_day()
                )
                ki_times = list(np.linspace(0, T, num_ki_steps + 1)[1:])
            else:
                ki_profile = product.get_ki_observation_profile(pricing_env)
                ki_times = ki_profile["observation_times"]

        # Recording the realized per-step variance costs a second
        # paths-sized array, so the vol-model engines only do it when a
        # continuously monitored KI barrier will actually run the bridge.
        self._ki_bridge_wanted = bool(product.has_ki_barrier and ki_continuous)

        # An observation exactly at valuation is decided on the known spot: it is
        # no simulation node (a zero step would be rejected) and maps to index -1,
        # i.e. path column 0 under the "+1" readout of the barrier checks. Grids
        # without such a record are unchanged.
        def at_valuation(t):
            return t <= 0.0

        grid_times = [t for t in list(ko_times) + list(ki_times) if not at_valuation(t)]

        # Combine all times and ensure maturity is included: one node per instant (W17a). A knock-out record and a
        # continuous-KI node can be one instant spelled an ulp apart (a dated record's (d - a) / 365, an aged
        # numeric one's d / 365 - a / 365, a linspace node); kept as two nodes, the empty step between them re-dealt
        # every random draw after it. A grid without such a pair is the sorted set of its times, bit for bit.
        # The continuous-KI linspace and the curve knots are nodes the engine adds itself: a contract instant (a KO
        # record, maturity) is never displaced by one a hair after it (contract clock D2). A discrete grid holds only
        # contract times, which arrive canonical: nothing to anchor.
        anchors = [t for t in ko_times if not at_valuation(t)] + [T] if ki_continuous else None
        all_times = merge_time_nodes(grid_times + [T], pin=T, anchors=anchors)

        # Build dt_array
        if ki_continuous:
            from quantark.asset.equity.engine.continuous_ki_support import continuous_time_grid
            all_times, self._continuous_ki_model = continuous_time_grid(pricing_env, all_times, self.params)
            all_times = merge_time_nodes(all_times, pin=T, anchors=anchors)

        times_with_zero = np.concatenate([[0.0], all_times])
        dt_array = np.diff(times_with_zero)

        def indices_for(times):
            return np.array([-1 if at_valuation(t) else node_index(all_times, t) for t in times], dtype=int)

        # Find indices for KO and KI observations
        ko_indices = indices_for(ko_times)
        if ki_continuous:
            # All times except t=0 are KI observation points
            ki_indices = np.arange(len(all_times))
        elif len(ki_times):
            ki_indices = indices_for(ki_times)
        else:
            ki_indices = np.array([], dtype=int)

        return all_times, dt_array, ko_indices, ki_indices

    def _build_time_grid_ko_reset(
        self,
        product: KnockOutResetSnowballOption,
        pricing_env: PricingEnvironment,
        T: float,
    ) -> Dict[str, object]:
        """
        Build time grid for KO-reset snowball options.

        Returns a dictionary with grid arrays and schedule metadata.
        """
        pre_profile = product.get_pre_ko_observation_profile(pricing_env)
        post_profile = product.get_post_ko_observation_profile(pricing_env)

        pre_times = np.array(pre_profile["observation_times"], dtype=float)
        post_times = np.array(post_profile["observation_times"], dtype=float)

        if not product.has_ki_barrier:
            raise ValidationError("KO-reset snowball requires KI barrier configuration.")

        # The contract's two rules (see KnockOutResetSnowballOption). A knock-in carried from an earlier
        # day has already replaced the first schedule by the second, for good: nothing is left to test.
        # Otherwise a knock-in is tested only while the first schedule is live; a record after its end
        # belongs to a contract that has already matured.
        carried_knock_in = bool(getattr(product, "_otc_lifecycle_knocked_in", False))
        product.require_alive(pricing_env, carried_knock_in)
        pre_maturity = float(product.get_pre_maturity_time(pricing_env))

        ki_continuous = not carried_knock_in and (
            product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
            or product.barrier_config.ki_continuous
        )
        ki_times: List[float] = []
        ki_barriers: List[float] = []
        # the source knock-in records aligned with ki_times, and their indices in the priced product's knock-in
        # schedule: each path's knock-in anchor (accrual time spec §5.1)
        ki_records: List[ObservationRecord] = []
        ki_record_indices: List[int] = []
        if carried_knock_in:
            pass
        elif ki_continuous:
            ki_horizon = pre_maturity
            num_ki_steps = continuous_ki_step_count(
                pricing_env.bus_days_in_year, ki_horizon, self._ki_bridge_nodes_per_day()
            )
            ki_times = list(np.linspace(0, ki_horizon, num_ki_steps + 1)[1:])
        else:
            ki_profile = product.get_ki_observation_profile(pricing_env)
            in_horizon = [
                index
                for index, time in enumerate(ki_profile["observation_times"])
                if time <= pre_maturity or same_instant(time, pre_maturity)
            ]
            ki_times = [ki_profile["observation_times"][index] for index in in_horizon]
            ki_barriers = [ki_profile["barriers"][index] for index in in_horizon]
            # the profile holds the records still ahead; a dated contract keeps its passed ones in its schedule
            ki_schedule_records = product.barrier_config.ki_observation_schedule.records
            ki_source_indices = product.active_ki_source_indices(pricing_env)
            ki_record_indices = [ki_source_indices[index] for index in in_horizon]
            ki_records = [ki_schedule_records[index] for index in ki_record_indices]

        all_times_set = set(pre_times) | set(ki_times) | {T}
        # the contract's own instants (contract clock D2): every time below except the continuous-KI linspace and
        # the business-day grid, which the engine adds itself
        contract_set = set(pre_times) | (set() if ki_continuous else set(ki_times)) | {T}

        daily_times = np.array([], dtype=float)
        use_daily_grid = getattr(self.params, "use_business_day_grid", False)
        calendar = getattr(pricing_env, "calendar", None)
        if (
            use_daily_grid
            and pricing_env.day_count_convention == DayCountConvention.BUSINESS_DAYS
            and calendar is not None
        ):
            end_dates: List[datetime] = []
            for schedule in (
                product.barrier_config.ko_observation_schedule,
                product.barrier_config.ki_observation_schedule,
                product.post_barrier_config.ko_observation_schedule,
            ):
                if schedule is None:
                    continue
                for rec in schedule.records:
                    if rec.observation_date is not None:
                        end_dates.append(rec.observation_date)

            if end_dates:
                end_date = max(end_dates)
                # The environment's BUSINESS_DAYS clock counts (valuation, date]: the valuation date is
                # t = 0 whether or not it is a business day. Counting it as day 1 put every daily node one
                # business day ahead of the record times and overran maturity.
                current = pricing_env.valuation_date
                count = 0
                times: List[float] = []
                while current < end_date:
                    current = current + timedelta(days=1)
                    if calendar.is_business_day(current):
                        count += 1
                        times.append(count / float(pricing_env.bus_days_in_year))
                daily_times = np.array(times, dtype=float)
                all_times_set |= set(daily_times)

        post_mode = post_profile["mode"]
        post_indices_by_ki: Optional[List[np.ndarray]] = None
        # REBASED: schedule 2's times for a knock-in on each knock-in record, t_ki + offset (numeric clock) or the
        # generated dates (date clock, ship spec 2026-10-26 Decision 6)
        post_times_by_ki: Optional[List[np.ndarray]] = None
        if post_mode == PostKOScheduleMode.ABSOLUTE:
            all_times_set |= set(post_times)
            contract_set |= set(post_times)
        else:
            # REBASED: treat post_times as offsets from KI time
            if carried_knock_in:
                # TODO: a carried knock-in does not record WHEN it happened, so nothing places the
                # offsets of a REBASED second schedule on the clock.
                raise ValidationError(
                    "A REBASED post-KO schedule needs the knock-in time, which a carried knock-in does not hold."
                )
            if ki_continuous:
                raise ValidationError(
                    "Rebased post-KO schedule requires discrete KI monitoring."
                )
            post_times_by_ki = product.rebased_schedule_two_times(
                pricing_env, ki_times, ki_record_indices, post_times
            )
            for actual_times in post_times_by_ki:
                all_times_set.update(actual_times)
                contract_set.update(actual_times)

        # An observation exactly at valuation is decided on the known spot, as in _build_time_grid: it
        # is no simulation node (a zero step is rejected) and maps to index -1, i.e. path column 0 under
        # the "+1" readout of the barrier checks. This grid used to keep time zero as a node, so a
        # contract observed now could not be priced at all. Grids without such a record are unchanged.
        def at_valuation(t):
            return t <= 0.0

        # One node per instant, as in _build_time_grid (W17a): two spellings of one instant (a record against a
        # continuous-KI node, a REBASED t_ki + offset against another node) are merged into one node.
        # A node the engine adds itself (the continuous-KI linspace, the business-day grid, the curve knots) never
        # displaces a contract instant a hair before it (contract clock D2); without such nodes nothing is anchored.
        anchors = (
            sorted(t for t in contract_set if not at_valuation(t))
            if ki_continuous or daily_times.size
            else None
        )
        all_times = merge_time_nodes(
            [t for t in all_times_set if not at_valuation(t)], pin=T, anchors=anchors
        )
        if ki_continuous:
            from quantark.asset.equity.engine.continuous_ki_support import continuous_time_grid
            all_times, self._continuous_ki_model = continuous_time_grid(pricing_env, all_times, self.params)
            all_times = merge_time_nodes(all_times, pin=T, anchors=anchors)

        times_with_zero = np.concatenate([[0.0], all_times])
        dt_array = np.diff(times_with_zero)

        def indices_for(times):
            return np.array(
                [-1 if at_valuation(t) else node_index(all_times, t) for t in times],
                dtype=int,
            )

        pre_indices = indices_for(pre_times)

        if post_mode == PostKOScheduleMode.ABSOLUTE:
            post_indices = indices_for(post_times)
        else:
            post_indices = np.array([], dtype=int)
            post_indices_by_ki = [indices_for(actual_times) for actual_times in post_times_by_ki]

        if ki_continuous:
            ki_indices = np.array([], dtype=int)
            # the last node of the first schedule; -1 when it ends at valuation: no future interval is monitored,
            # only the known spot (path column 0). Clamped to 0, a later hit switched a claim that had already
            # matured on the first schedule over to the second (review 2026-09-21 R8).
            # (within Tolerance.ZERO: the node standing for pre_maturity may be its merged spelling just past it)
            ki_horizon_idx = int(
                np.searchsorted(all_times, pre_maturity + Tolerance.ZERO, side="right") - 1
            )
        else:
            ki_indices = indices_for(ki_times)
            ki_horizon_idx = None

        # See _build_time_grid: the vol-model engines record the realized
        # per-step variance only when the bridge will actually run.
        self._ki_bridge_wanted = bool(product.has_ki_barrier and ki_continuous)

        return {
            "carried_knock_in": carried_knock_in,
            "ki_barriers": np.array(ki_barriers, dtype=float),
            "ki_records": ki_records,
            "ki_record_indices": ki_record_indices,
            # the not-knocked-in contract matures HERE: its rebate is read off this node
            "pre_maturity_idx": int(indices_for([pre_maturity])[0]) if len(pre_times) else None,
            # ABSOLUTE (a carried knock-in included): the knocked-in contract matures at schedule 2's end, before
            # the grid's last node when schedule 1 runs on past it; its V1 is read off this node (spec D31)
            "post_maturity_idx": (
                int(indices_for([float(product.get_post_maturity_time(pricing_env))])[0])
                if post_mode == PostKOScheduleMode.ABSOLUTE
                else None
            ),
            "all_times": all_times,
            "dt_array": dt_array,
            "pre_times": pre_times,
            "pre_indices": pre_indices,
            "post_times": post_times,
            "post_indices": post_indices,
            "post_indices_by_ki": post_indices_by_ki,
            "post_times_by_ki": post_times_by_ki,
            "ki_times": np.array(ki_times, dtype=float),
            "ki_indices": ki_indices,
            "ki_continuous": ki_continuous,
            "ki_horizon_idx": ki_horizon_idx,
            "pre_profile": pre_profile,
            "post_profile": post_profile,
        }

    def _compute_ko_schedule_payoffs(
        self,
        product: KnockOutResetSnowballOption,
        times: np.ndarray,
        rates: np.ndarray,
        records: List[ObservationRecord],
        pricing_env: PricingEnvironment,
        maturity_time: float,
        *,
        first_schedule: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute KO payoffs and settlement times for a resolved schedule.

        Under EXPIRY pay every knock-out is paid with the note it ends, determined at ``maturity_time``. For schedule 1
        (``first_schedule``) that note ends with schedule 1 (spec D30) and is paid by ``first_schedule_terminal_timing``:
        its explicit payment when booked (spec D33, D34), else the settlement convention after ``maturity_time``.
        """
        if len(times) == 0:
            return np.array([]), np.array([])

        principal_component = (
            product.initial_price * product.contract_multiplier
            if product.payoff_config.include_principal
            else 0.0
        )
        payoffs = np.zeros(len(times), dtype=float)
        for idx, t in enumerate(times):
            # the resolver's precedence: an explicit factor, else 1.0 when not annualized, else the period
            accrual_factor = product.compute_ko_accrual_factor(float(t), records[idx], pricing_env)
            coupon = (
                product.initial_price
                * product.contract_multiplier
                * float(rates[idx])
                * float(accrual_factor)
            )
            payoffs[idx] = principal_component + coupon

        if product.accrual_config.coupon_pay_type == CouponPayType.INSTANT:
            settlement_times = np.array(
                [
                    self._resolve_payment_time(
                        product,
                        pricing_env,
                        float(time),
                        CashflowKind.REDEMPTION,
                        record=records[index],
                        cashflow_id=f"ko_reset[{index}]",
                    )
                    for index, time in enumerate(times)
                ],
                dtype=float,
            )
        elif first_schedule:
            terminal_payment_time = float(first_schedule_terminal_timing(
                product, pricing_env, determination_time=float(maturity_time),
            ).payment_time)
            settlement_times = np.full(
                len(times), terminal_payment_time, dtype=float
            )
        else:
            terminal_payment_time = self._resolve_payment_time(
                product,
                pricing_env,
                float(maturity_time),
                CashflowKind.TERMINAL,
                cashflow_id="ko_reset_terminal",
            )
            settlement_times = np.full(
                len(times), terminal_payment_time, dtype=float
            )

        return payoffs, settlement_times

    def _create_path_generator(
        self,
        S: float,
        r: float,
        q: float,
        sigma: float,
        T: float,
        dt_array: np.ndarray,
        batch_id: Optional[int] = None,
        num_paths: Optional[int] = None,
    ) -> GBMPathGenerator:
        """
        Create a GBMPathGenerator configured for the observation grid.

        Args:
            S: Spot price
            r: Risk-free rate
            q: Dividend yield
            sigma: Volatility
            T: Time to maturity
            dt_array: Non-uniform time increments
            batch_id: Batch identifier for RQMC

        Returns:
            Configured GBMPathGenerator
        """
        params = self.params
        effective_num_paths = params.num_paths if num_paths is None else int(num_paths)
        if effective_num_paths <= 0:
            raise ValidationError(
                f"num_paths must be positive, got {effective_num_paths}"
            )

        if self.method == MonteCarloMethod.PSEUDO:
            seed = params.seed + (batch_id or 0) * 1000
            random_stream = PseudoRandomNormalGenerator(seed=seed)
            is_qmc = False
        elif self.method in (MonteCarloMethod.QUASI, MonteCarloMethod.RANDOMIZED_QUASI):
            random_stream = SobolNormalGenerator(base_seed=params.seed)
            is_qmc = True
        else:
            raise ValidationError(f"Unknown Monte Carlo method: {self.method}")

        # No antithetic variates for barrier options (breaks correlation structure)
        vr_config = None

        term_ctx = getattr(self, "_term_ctx", None)
        if term_ctx is not None:
            env_ctx, ref_strike = term_ctx
            term = build_mc_term_inputs(
                env_ctx, ref_strike=ref_strike, maturity=T,
                time_steps=len(dt_array), dt_array=dt_array,
            )
            vol_in, rrf_in, div_in = term.vol, term.rrf, term.div
        else:
            vol_in, rrf_in, div_in = sigma, r, q

        # The continuous bridge must use the same interval variance as GBM.
        # This is one vector, not another paths-sized allocation.
        self._native_ki_step_variance = np.square(np.asarray(vol_in)) * dt_array

        generator = GBMPathGenerator(
            initial_value=S,
            vol=vol_in,
            rrf=rrf_in,
            div=div_in,
            maturity=T,
            time_steps=len(dt_array),
            num_paths=effective_num_paths,
            model="bsm",
            random_stream=random_stream,
            use_brownian_bridge=is_qmc,
            vr_config=vr_config,
            is_qmc=is_qmc,
            dt_array=dt_array,
        )

        return generator

    def _check_ko_barriers(
        self,
        paths: np.ndarray,
        ko_indices: np.ndarray,
        ko_barriers: np.ndarray,
        is_reverse: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized KO barrier checking.

        Args:
            paths: Simulated paths, shape (num_paths, num_times + 1)
            ko_indices: Indices into paths for KO observations
            ko_barriers: KO barrier levels, shape (num_ko_obs,)
            is_reverse: True for reverse snowball (DOWN barrier)

        Returns:
            Tuple of:
            - ko_triggered: Boolean array (num_paths,) indicating if KO was triggered
            - first_ko_idx: Index of first KO trigger (-1 if never triggered)
        """
        # Extract prices at KO observation times (offset by 1 for t=0)
        ko_prices = paths[:, ko_indices + 1]  # (num_paths, num_ko_obs)

        # Vectorized barrier check
        if is_reverse:
            # Reverse snowball: DOWN barrier (KO if price <= barrier)
            ko_hit = ko_prices <= ko_barriers
        else:
            # Standard snowball: UP barrier (KO if price >= barrier)
            ko_hit = ko_prices >= ko_barriers

        # Find first KO time per path
        ko_triggered = ko_hit.any(axis=1)
        first_ko_idx = np.full(len(paths), -1, dtype=int)

        if ko_triggered.any():
            # argmax returns first True in each row
            first_ko_idx[ko_triggered] = np.argmax(ko_hit[ko_triggered], axis=1)

        return ko_triggered, first_ko_idx

    def _check_ki_barriers(
        self,
        paths: np.ndarray,
        ki_indices: np.ndarray,
        ki_barriers: Union[float, np.ndarray],
        is_reverse: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized KI barrier checking.

        Args:
            paths: Simulated paths, shape (num_paths, num_times + 1)
            ki_indices: Indices into paths for KI observations
            ki_barrier: KI barrier level (single value for now)
            is_reverse: True for reverse snowball (UP barrier for KI)

        Returns:
            Tuple of:
            - ki_triggered: Boolean array (num_paths,) indicating if KI was triggered
            - first_ki_idx: Index of first KI trigger (-1 if never triggered)
        """
        if len(ki_indices) == 0:
            return np.zeros(len(paths), dtype=bool), np.full(len(paths), -1, dtype=int)

        # Extract prices at KI observation times (offset by 1 for t=0)
        ki_prices = paths[:, ki_indices + 1]  # (num_paths, num_ki_obs)
        num_ki_obs_times = ki_prices.shape[1]

        ki_barriers_effective = np.array(ki_barriers)

        # If ki_barriers has a single value (i.e., scalar or [scalar]), broadcast it
        if ki_barriers_effective.shape == () or ki_barriers_effective.shape == (1,):
            ki_barriers_aligned = np.full(
                num_ki_obs_times, ki_barriers_effective.item()
            )
        else:
            # If it's an array with multiple values, it must match the number of observation times
            if ki_barriers_effective.shape[0] != num_ki_obs_times:
                raise ValidationError(
                    f"ki_barriers array (shape {ki_barriers_effective.shape[0]}) "
                    f"does not match number of KI observation times ({num_ki_obs_times})"
                )
            ki_barriers_aligned = ki_barriers_effective

        # Vectorized barrier check
        if is_reverse:
            # Reverse snowball: UP barrier for KI (KI if price >= barrier)
            ki_hit = ki_prices >= ki_barriers_aligned
        else:
            # Standard snowball: DOWN barrier for KI (KI if price <= barrier)
            ki_hit = ki_prices <= ki_barriers_aligned

        # Find first KI time per path
        ki_triggered = ki_hit.any(axis=1)
        first_ki_idx = np.full(len(paths), -1, dtype=int)

        if ki_triggered.any():
            first_ki_idx[ki_triggered] = np.argmax(ki_hit[ki_triggered], axis=1)

        return ki_triggered, first_ki_idx

    def _ki_bridge_step_log_variance(
        self, paths: np.ndarray, horizon_steps: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """Log-variance accumulated over each monitoring interval, per path.

        Shape ``(n_paths, n_steps)``, or ``None`` to use the scalar ``sigma``
        the bridge was handed. The Brownian-bridge crossing probability is
        governed by the quadratic variation of log S over the interval, so a
        constant-vol engine is exactly served by ``sigma**2 * dt`` -- but a
        local-vol, Heston or SLV path accumulates variance those engines
        simulate and this engine never sees. They override.

        ``horizon_steps``: the paths are the leading ``horizon_steps`` steps of
        the simulated grid (a KO-reset's knock-in horizon); ``None`` means the
        whole grid.
        """
        variance = getattr(self, "_native_ki_step_variance", None)
        n_steps = paths.shape[1] - 1
        if variance is None or len(variance) < n_steps:
            return None
        return np.broadcast_to(variance[:n_steps], (len(paths), n_steps))

    def _check_ki_barriers_continuous_with_bridge(
        self,
        paths: np.ndarray,
        all_times: np.ndarray,
        ki_barrier: float,
        sigma: float,
        is_reverse: bool,
        rng_seed: int,
        horizon_steps: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Continuous KI monitoring with Brownian-bridge barrier correction.

        The path is simulated on a discrete grid. If both endpoints of an interval
        are on the non-breached side of the KI barrier, a Brownian-bridge estimate
        is used to sample whether a barrier hit occurred within the interval.
        ``horizon_steps``: ``paths`` are the leading steps of the simulated grid
        (see ``_ki_bridge_step_log_variance``).
        """
        if ki_barrier <= 0:
            raise ValidationError(f"ki_barrier must be positive, got {ki_barrier}")
        if sigma <= 0:
            raise ValidationError(f"volatility must be positive, got {sigma}")

        n_paths = len(paths)
        if n_paths == 0:
            return np.zeros(0, dtype=bool), np.zeros(0, dtype=int)

        ki_triggered = np.zeros(n_paths, dtype=bool)
        first_ki_idx = np.full(n_paths, -1, dtype=int)

        # Immediate breach at valuation (t=0) counts as KI for continuous monitoring.
        already_breached = _ki_level_breached(paths[:, 0], ki_barrier, is_reverse)
        if already_breached.any():
            ki_triggered[already_breached] = True
            first_ki_idx[already_breached] = 0

        all_times = np.asarray(all_times, dtype=float)
        if all_times.ndim != 1:
            raise ValidationError("all_times must be a 1D array of time points")

        n_steps = paths.shape[1] - 1
        if all_times.shape[0] != n_steps:
            raise ValidationError(
                f"all_times length ({all_times.shape[0]}) must match number of steps ({n_steps})"
            )
        if n_steps == 0:
            # the monitoring window ends at valuation: the known spot is all there is to observe
            return ki_triggered, first_ki_idx

        dt = np.empty(n_steps, dtype=float)
        dt[0] = float(all_times[0])
        if n_steps > 1:
            dt[1:] = np.diff(all_times)
        if np.any(dt <= 0.0):
            raise ValidationError("all_times must be strictly increasing and > 0")

        rng = np.random.default_rng(int(rng_seed))
        step_log_variance = self._ki_bridge_step_log_variance(
            paths, horizon_steps=horizon_steps
        )

        # Sample step-wise hit events using the Brownian-bridge probabilities.
        # If a hit occurs in step k, we record the right endpoint index k.
        for k in range(n_steps):
            active = ~ki_triggered
            if not active.any():
                break

            # If the barrier is already breached at the right endpoint, it's a certain hit.
            s1 = paths[:, k + 1]
            if is_reverse:
                breached_at_endpoint = s1 >= ki_barrier
            else:
                breached_at_endpoint = s1 <= ki_barrier

            new_hit = active & breached_at_endpoint
            if new_hit.any():
                ki_triggered[new_hit] = True
                first_ki_idx[new_hit] = k

            active = ~ki_triggered
            if not active.any():
                break

            s0 = paths[:, k]
            s1 = paths[:, k + 1]

            if is_reverse:
                # KI if price >= barrier; non-breached region is below the barrier.
                non_breached = (s0 < ki_barrier) & (s1 < ki_barrier)
            else:
                # KI if price <= barrier; non-breached region is above the barrier.
                non_breached = (s0 > ki_barrier) & (s1 > ki_barrier)

            bridge_candidates = active & non_breached
            if not bridge_candidates.any():
                continue

            idx = np.flatnonzero(bridge_candidates)
            dt_k = float(dt[k])
            # h2 is the log-variance the path ACTUALLY accumulated over this
            # interval. sigma**2 * dt is exact only when the diffusion is the
            # constant `sigma`; engines whose diffusion varies with spot,
            # time or a simulated variance report what they integrated. A
            # step that accumulated no variance cannot have crossed, and the
            # -inf exponent clips to give exactly that.
            h2 = (
                float(sigma * sigma) * dt_k
                if step_log_variance is None
                else step_log_variance[idx, k]
            )

            log_term = safe_log(s0[idx] / ki_barrier) * safe_log(s1[idx] / ki_barrier)
            with np.errstate(divide="ignore"):
                exponent = -2.0 * log_term / h2
            exponent = np.clip(exponent, -745.0, 0.0)
            p = np.exp(exponent)

            u = rng.random(idx.size)
            hit = u < p
            if hit.any():
                hit_paths = idx[hit]
                ki_triggered[hit_paths] = True
                first_ki_idx[hit_paths] = k

        return ki_triggered, first_ki_idx

    def _compute_payoffs(
        self,
        product: SnowballOption,
        pricing_env: PricingEnvironment,
        paths: np.ndarray,
        all_times: np.ndarray,
        ko_indices: np.ndarray,
        ki_indices: np.ndarray,
        r: float,
        T: float,
        sigma: float,
        rng_seed: int,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Compute payoffs for all paths based on their terminal state.

        Args:
            product: SnowballOption product
            pricing_env: Pricing environment
            paths: Simulated paths, shape (num_paths, num_times + 1)
            all_times: All observation times
            ko_indices: Indices for KO observations
            ki_indices: Indices for KI observations
            r: Risk-free rate
            T: Time to maturity

        Returns:
            Tuple of:
            - payoffs: Undiscounted payoffs, shape (num_paths,)
            - settlement_times: Settlement time for each path, shape (num_paths,)
            - stats: Dictionary with probability statistics
        """
        num_paths = len(paths)

        # Get resolved observation profiles
        ko_profile = product.get_ko_observation_profile(pricing_env)
        ko_barriers = np.array(ko_profile["barriers"])
        ko_payoffs_schedule = np.array(ko_profile["payoffs"])
        ko_times = np.array(ko_profile["observation_times"])
        timings = self._payment_timings
        if timings is None:
            self._prepare_payment_timings(product, pricing_env)
            timings = self._payment_timings
        ko_settlement_times = np.array(
            timings.observation_payment_times,
            copy=False,
        )

        # Get KI barriers (can be scalar or array)
        ki_barriers_val = None
        if product.has_ki_barrier:
            ki_profile = product.get_ki_observation_profile(pricing_env)
            ki_barriers_val = np.array(ki_profile["barriers"])

        # Check barriers
        ko_triggered, first_ko_idx = self._check_ko_barriers(
            paths, ko_indices, ko_barriers, product.is_reverse
        )

        ki_triggered = np.zeros(num_paths, dtype=bool)
        first_ki_idx = np.full(num_paths, -1, dtype=int)
        if ki_barriers_val is not None:
            ki_continuous = (
                product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
                or product.barrier_config.ki_continuous
            )
            if ki_continuous:
                if ki_barriers_val.shape not in ((), (1,)):
                    raise ValidationError(
                        "Continuous KI monitoring requires a scalar ki_barrier."
                    )
                ki_barrier_scalar = float(ki_barriers_val.reshape(-1)[0])
                ki_triggered, first_ki_idx = (
                    self._check_ki_barriers_continuous_with_bridge(
                        paths=paths,
                        all_times=all_times,
                        ki_barrier=ki_barrier_scalar,
                        sigma=float(sigma),
                        is_reverse=product.is_reverse,
                        rng_seed=int(rng_seed),
                    )
                )
            else:
                ki_triggered, first_ki_idx = self._check_ki_barriers(
                    paths, ki_indices, ki_barriers_val, product.is_reverse
                )

        already_knocked_in = bool(getattr(product, "_otc_lifecycle_knocked_in", False))
        if already_knocked_in:
            ki_triggered[:] = True
            first_ki_idx[:] = 0

        # Handle disable_ko_after_ki logic
        if product.barrier_config.disable_ko_after_ki and (
            ki_barriers_val is not None or already_knocked_in
        ):
            if already_knocked_in:
                ko_valid = np.zeros_like(ko_triggered)
            else:
                # Get times for comparison
                ko_trigger_times = np.where(
                    first_ko_idx >= 0,
                    ko_times[first_ko_idx],
                    np.inf,
                )
                # index -1 is an observation at valuation (time 0.0)
                node_times = np.concatenate([[0.0], all_times])
                ki_trigger_times = np.where(
                    first_ki_idx >= 0,
                    node_times[ki_indices[first_ki_idx] + 1]
                    if len(ki_indices) > 0
                    else np.inf,
                    np.inf,
                )

                # A knock-out and a knock-in observed at the SAME instant: the knock-out wins, so the
                # knock-in observed with it does not disable it. A continuously monitored knock-in is
                # different: a touch found inside a step happened BEFORE the observation closing it.
                ko_valid = ko_triggered & _knock_out_first(
                    ko_trigger_times, ki_trigger_times, ki_continuous
                )
        else:
            ko_valid = ko_triggered

        # Classify paths into states
        is_ko = ko_valid
        is_v0 = ~is_ko & ~ki_triggered
        is_v1 = ~is_ko & ki_triggered

        # Initialize payoffs and settlement times
        payoffs = np.zeros(num_paths)
        settlement_times = np.full(
            num_paths,
            float(timings.terminal.payment_time),
        )

        # Folded cash legs (price_folded only): each record's cash where its own payoff goes
        fold = None
        if self._fold is not None:
            terminal = (float(timings.terminal.determination_time), float(timings.terminal.payment_time))
            instant = product.accrual_config.coupon_pay_type == CouponPayType.INSTANT
            fold = self._bound_fold(
                [
                    ("main", float(t), float(ko_settlement_times[i]) if instant else terminal[1])
                    for i, t in enumerate(ko_times)
                ],
                terminal,
                terminal,
            )
            ko_payoffs_schedule = ko_payoffs_schedule + fold[0]

        # KO payoffs
        if is_ko.any():
            ko_idx_for_payoff = first_ko_idx[is_ko]
            payoffs[is_ko] = ko_payoffs_schedule[ko_idx_for_payoff]

            # Settlement time depends on coupon pay type
            if product.accrual_config.coupon_pay_type == CouponPayType.INSTANT:
                settlement_times[is_ko] = ko_settlement_times[ko_idx_for_payoff]
            # else: EXPIRY - settlement at maturity (already set)

        # V0 payoffs (never KO, never KI)
        if is_v0.any():
            terminal_spots = paths[is_v0, -1]
            v0_payoffs = np.array(
                [
                    product.get_maturity_payoff_v0(spot, pricing_env)
                    for spot in terminal_spots
                ]
            )
            payoffs[is_v0] = v0_payoffs if fold is None else v0_payoffs + fold[1]

        # V1 payoffs (never KO, KI happened)
        if is_v1.any():
            terminal_spots = paths[is_v1, -1]
            v1_payoffs = np.array(
                [
                    product.get_maturity_payoff_v1(spot, pricing_env)
                    for spot in terminal_spots
                ]
            )
            payoffs[is_v1] = v1_payoffs if fold is None else v1_payoffs + fold[2]

        # Compute statistics
        ko_count = int(is_ko.sum())
        v0_count = int(is_v0.sum())
        v1_count = int(is_v1.sum())

        stats = {
            "ko_probability": float(is_ko.mean()),
            "v0_probability": float(is_v0.mean()),
            "v1_probability": float(is_v1.mean()),
            "ko_count": ko_count,
            "v0_count": v0_count,
            "v1_count": v1_count,
        }

        if is_ko.any():
            ko_times_hit = ko_times[first_ko_idx[is_ko]]
            stats["avg_ko_time"] = float(ko_times_hit.mean())
            stats["ko_time_sum"] = float(ko_times_hit.sum())
            stats["ko_time_count"] = ko_count
        else:
            stats["avg_ko_time"] = None
            stats["ko_time_sum"] = 0.0
            stats["ko_time_count"] = 0

        if self._stats_recording():
            # event statistics of a recorded price (calculate_event_stats): this batch's outcome sums
            has_ki_rows = ki_barriers_val is not None
            breached_now = False
            if not has_ki_rows:
                ki_row_times = np.zeros(0, dtype=float)
                ki_path_times = None
            elif ki_continuous:
                # the MC steps of the window (step k ends at all_times[k]); a level breached at valuation is a
                # knock-in at time 0 (DESIGN §4.5), every path's
                ki_row_times = np.asarray(all_times, dtype=float)
                breached_now = bool(
                    num_paths and _ki_level_breached(paths[:1, 0], ki_barrier_scalar, product.is_reverse)[0]
                )
                ki_path_times = np.where(
                    ki_triggered,
                    0.0 if breached_now else ki_row_times[np.maximum(first_ki_idx, 0)],
                    np.inf,
                )
            else:
                ki_row_times = np.asarray(ki_profile["observation_times"], dtype=float)
                ki_path_times = np.where(
                    ki_triggered, ki_row_times[np.maximum(first_ki_idx, 0)], np.inf
                ) if ki_row_times.size else np.full(num_paths, np.inf, dtype=float)
            self._record_stats_batch(
                self._snowball_batch_record(
                    payoffs=payoffs,
                    settlement_times=settlement_times,
                    is_ko=is_ko,
                    is_v0=is_v0,
                    is_v1=is_v1,
                    first_ko_idx=first_ko_idx,
                    ko_times=np.asarray(ko_times, dtype=float),
                    ki_triggered=ki_triggered,
                    first_ki_idx=first_ki_idx,
                    ki_row_times=ki_row_times,
                    ki_path_times=ki_path_times,
                    ki_continuous=bool(has_ki_rows and ki_continuous),
                    carried_knock_in=already_knocked_in,
                    breached_at_valuation=breached_now,
                    principal=(
                        product.initial_price * product.contract_multiplier
                        if product.payoff_config.include_principal
                        else 0.0
                    ),
                )
            )

        return payoffs, settlement_times, stats

    @staticmethod
    def _knock_in_anchors(grid) -> list:
        """One anchor per knock-in column of the grid, indexed by its record's place in the PRICED product's
        knock-in schedule. That is the contract's original index only for an unaged product: an aged copy has
        dropped its past knock-in records and keeps no count of them (see ``KnockInAnchor.source_index``)."""
        return [knock_in_anchor_for(rec, index) for index, rec in zip(grid["ki_record_indices"], grid["ki_records"])]

    def _compute_payoffs_ko_reset(
        self,
        product: KnockOutResetSnowballOption,
        pricing_env: PricingEnvironment,
        paths: np.ndarray,
        grid: Dict[str, object],
        r: float,
        T: float,
        sigma: float,
        rng_seed: int,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], Dict[str, object]]:
        """
        Compute payoffs for KO-reset snowball paths.
        """
        num_paths = len(paths)
        if num_paths == 0:
            return np.zeros(0), np.zeros(0), {}, {}

        all_times = grid["all_times"]
        pre_times = grid["pre_times"]
        post_times = grid["post_times"]
        pre_indices = grid["pre_indices"]
        post_indices = grid["post_indices"]
        post_indices_by_ki = grid["post_indices_by_ki"]
        post_times_by_ki = grid["post_times_by_ki"]
        ki_times = grid["ki_times"]
        ki_indices = grid["ki_indices"]
        ki_continuous = grid["ki_continuous"]
        ki_horizon_idx = grid["ki_horizon_idx"]

        pre_profile = grid["pre_profile"]
        post_profile = grid["post_profile"]

        pre_barriers = np.array(pre_profile["barriers"], dtype=float)
        pre_rates = np.array(pre_profile["rates"], dtype=float)
        pre_records = pre_profile["records"]

        post_barriers = np.array(post_profile["barriers"], dtype=float)
        post_rates = np.array(post_profile["rates"], dtype=float)
        post_records = post_profile["records"]
        post_mode = post_profile["mode"]

        pre_maturity = product.get_pre_maturity_time(pricing_env)
        post_max_offset = float(max(post_times)) if len(post_times) else 0.0
        post_maturity_abs = (
            product.get_post_maturity_time(pricing_env)
            if post_mode == PostKOScheduleMode.ABSOLUTE
            else None
        )
        # REBASED: row k holds schedule 2's times after a knock-in on ki_times[k]; ``post_last`` is its final
        # record (the largest offset), the knocked-in end
        post_table = (
            np.vstack(post_times_by_ki)
            if post_times_by_ki and len(post_times) > 0
            else None
        )
        post_last = int(np.argmax(post_times)) if len(post_times) else -1

        # KO payoffs for pre schedule
        pre_payoffs, pre_settlement_times = self._compute_ko_schedule_payoffs(
            product,
            pre_times,
            pre_rates,
            pre_records,
            pricing_env,
            pre_maturity,
            first_schedule=True,
        )

        post_payoffs = np.array([], dtype=float)
        post_settlement_times = np.array([], dtype=float)
        if post_mode == PostKOScheduleMode.ABSOLUTE and len(post_times) > 0:
            post_payoffs, post_settlement_times = self._compute_ko_schedule_payoffs(
                product,
                post_times,
                post_rates,
                post_records,
                pricing_env,
                float(post_maturity_abs),
            )

        # KI barriers: the records the grid kept (rule 2), or the contract's level when monitored continuously
        carried_knock_in = bool(grid["carried_knock_in"])
        ki_barriers_val = None
        if product.has_ki_barrier and not carried_knock_in:
            if ki_continuous:
                ki_profile = product.get_ki_observation_profile(pricing_env)
                ki_barriers_val = np.array(ki_profile["barriers"], dtype=float)
            else:
                ki_barriers_val = grid["ki_barriers"]

        # Check KI triggers
        ki_triggered = np.zeros(num_paths, dtype=bool)
        first_ki_idx = np.full(num_paths, -1, dtype=int)
        if ki_barriers_val is not None:
            if ki_continuous:
                if ki_barriers_val.shape not in ((), (1,)):
                    raise ValidationError(
                        "Continuous KI monitoring requires a scalar ki_barrier."
                    )
                ki_barrier_scalar = float(ki_barriers_val.reshape(-1)[0])
                if ki_horizon_idx is None:
                    raise ValidationError("Missing KI horizon index for KO-reset grid.")
                ki_paths = paths[:, : ki_horizon_idx + 2]
                ki_times_slice = all_times[: ki_horizon_idx + 1]
                ki_triggered, first_ki_idx = (
                    self._check_ki_barriers_continuous_with_bridge(
                        paths=ki_paths,
                        all_times=ki_times_slice,
                        ki_barrier=ki_barrier_scalar,
                        sigma=float(sigma),
                        is_reverse=product.is_reverse,
                        rng_seed=int(rng_seed),
                        horizon_steps=ki_horizon_idx + 1,
                    )
                )
            else:
                ki_triggered, first_ki_idx = self._check_ki_barriers(
                    paths, ki_indices, ki_barriers_val, product.is_reverse
                )

        # Map KI trigger time
        ki_time = np.full(num_paths, np.inf, dtype=float)
        if carried_knock_in:
            # rule 1: knocked in before today, so the first schedule is gone on every path and every
            # observation of the second is in force (its time is "after the knock-in" whatever it is)
            ki_triggered[:] = True
            ki_time[:] = -np.inf
        elif ki_continuous:
            valid = first_ki_idx >= 0
            if valid.any():
                ki_time[valid] = all_times[first_ki_idx[valid]]
            # The bridge reports a level already breached at valuation as index 0, which is also a hit inside
            # the first step (at its right end). That knock-in happened NOW: at time zero it puts a second-schedule
            # observation at this instant in force, as the decision at valuation does. Read as the first future
            # node, it left a path knocked in and out now exposed to later prices (review 2026-09-21 R10).
            ki_time[ki_triggered & _ki_level_breached(paths[:, 0], ki_barrier_scalar, product.is_reverse)] = 0.0
        else:
            valid = first_ki_idx >= 0
            if valid.any():
                ki_time[valid] = ki_times[first_ki_idx[valid]]

        # Pre-KO hits
        pre_ko_triggered = np.zeros(num_paths, dtype=bool)
        first_pre_idx = np.full(num_paths, -1, dtype=int)
        if len(pre_indices) > 0:
            pre_ko_triggered, first_pre_idx = self._check_ko_barriers(
                paths, pre_indices, pre_barriers, product.is_reverse
            )

        if len(pre_times) == 0:
            pre_ko_time = np.full(num_paths, np.inf, dtype=float)
        else:
            pre_ko_time = np.where(
                first_pre_idx >= 0, pre_times[first_pre_idx], np.inf
            )
        # A first-schedule knock-out and a knock-in observed at the SAME instant: the knock-out wins, so
        # the knock-in observed with it does not replace the schedule that knock-out belongs to. (A
        # continuously monitored knock-in found inside a step happened BEFORE the observation closing it.)
        pre_valid = pre_ko_triggered & _knock_out_first(pre_ko_time, ki_time, ki_continuous)

        # Post-KO hits
        post_ko_triggered = np.zeros(num_paths, dtype=bool)
        first_post_idx = np.full(num_paths, -1, dtype=int)
        post_ko_time = np.full(num_paths, np.inf, dtype=float)

        post_allowed = ki_triggered & ~pre_valid
        if product.barrier_config.disable_ko_after_ki:
            post_allowed = np.zeros(num_paths, dtype=bool)

        if post_allowed.any() and len(post_times) > 0:
            if post_mode == PostKOScheduleMode.ABSOLUTE:
                post_prices = paths[:, post_indices + 1]
                if product.is_reverse:
                    post_hit = post_prices <= post_barriers
                else:
                    post_hit = post_prices >= post_barriers
                # The second schedule is in force from the observation that knocks the path in, that
                # observation included -- the rule of the PDE and QUAD engines, and of this engine's own
                # REBASED branch, which tests an offset of zero at the knock-in instant. Counting it
                # strictly after the knock-in left a path knocked in and out at one instant alive
                # until the next date. (The tolerance only absorbs two spellings of one time.)
                time_mask = post_times[None, :] >= ki_time[:, None] - 1e-12
                post_hit_filtered = post_hit & time_mask
                post_triggered = post_hit_filtered.any(axis=1)
                post_triggered &= post_allowed
                if post_triggered.any():
                    first_idx_local = np.argmax(post_hit_filtered[post_triggered], axis=1)
                    post_ko_triggered[post_triggered] = True
                    first_post_idx[post_triggered] = first_idx_local
                    post_ko_time[post_triggered] = post_times[first_idx_local]
            else:
                if post_indices_by_ki is None:
                    raise ValidationError("Missing post_indices_by_ki for rebased KO reset.")
                eligible_idx = np.flatnonzero(post_allowed)
                if eligible_idx.size > 0:
                    for ki_idx_val in np.unique(first_ki_idx[eligible_idx]):
                        if ki_idx_val < 0:
                            continue
                        group_mask = post_allowed & (first_ki_idx == ki_idx_val)
                        if not group_mask.any():
                            continue
                        indices = post_indices_by_ki[ki_idx_val]
                        if len(indices) == 0:
                            continue
                        group_paths = paths[group_mask][:, indices + 1]
                        if product.is_reverse:
                            hit_matrix = group_paths <= post_barriers
                        else:
                            hit_matrix = group_paths >= post_barriers
                        triggered = hit_matrix.any(axis=1)
                        if not triggered.any():
                            continue
                        first_local = np.argmax(hit_matrix[triggered], axis=1)
                        group_idx = np.flatnonzero(group_mask)[triggered]
                        post_ko_triggered[group_idx] = True
                        first_post_idx[group_idx] = first_local
                        post_ko_time[group_idx] = post_table[ki_idx_val, first_local]

        is_pre_ko = pre_valid
        is_post_ko = post_ko_triggered & ~is_pre_ko
        is_ko = is_pre_ko | is_post_ko
        is_v0 = ~is_ko & ~ki_triggered
        is_v1 = ~is_ko & ki_triggered

        payoffs = np.zeros(num_paths, dtype=float)
        settlement_times = np.zeros(num_paths, dtype=float)

        # Folded cash legs (price_folded only): bound to the records and branch settlements paid below
        fold = None
        if self._fold is not None:
            if post_mode != PostKOScheduleMode.ABSOLUTE:
                self._refuse_rebased_fold()
            fold = self._bound_fold(
                [("pre", float(t), float(p)) for t, p in zip(pre_times, pre_settlement_times)]
                + [("post", float(t), float(p)) for t, p in zip(post_times, post_settlement_times)],
                (
                    float(pre_maturity),
                    float(first_schedule_terminal_timing(
                        product, pricing_env, determination_time=float(pre_maturity),
                    ).payment_time),
                ),
                (
                    float(post_maturity_abs),
                    self._resolve_payment_time(
                        product, pricing_env, float(post_maturity_abs), CashflowKind.TERMINAL,
                        cashflow_id="ko_reset_v1_terminal",
                    ),
                ),
            )
            n_pre = len(pre_times)
            pre_payoffs = pre_payoffs + fold[0][:n_pre]
            post_payoffs = post_payoffs + fold[0][n_pre:]

        if is_pre_ko.any():
            ko_idx = first_pre_idx[is_pre_ko]
            payoffs[is_pre_ko] = pre_payoffs[ko_idx]
            settlement_times[is_pre_ko] = pre_settlement_times[ko_idx]

        if is_post_ko.any():
            if post_mode == PostKOScheduleMode.ABSOLUTE:
                ko_idx = first_post_idx[is_post_ko]
                payoffs[is_post_ko] = post_payoffs[ko_idx]
                settlement_times[is_post_ko] = post_settlement_times[ko_idx]
            else:
                principal_component = (
                    product.initial_price * product.contract_multiplier
                    if product.payoff_config.include_principal
                    else 0.0
                )
                # each path's knock-out accrual hangs from the knock-in record that fired on it (spec §5.1)
                anchors = self._knock_in_anchors(grid)
                accrual_table = np.array([
                    [float(product.compute_ko_accrual_factor(None, rec, pricing_env, knock_in=anchor))
                     for rec in post_records]
                    for anchor in anchors
                ], dtype=float).reshape(len(anchors), len(post_records))
                idx = np.flatnonzero(is_post_ko)
                local_idx = first_post_idx[idx]
                rates = post_rates[local_idx]
                accrual = accrual_table[first_ki_idx[idx], local_idx]
                coupon = (
                    product.initial_price
                    * product.contract_multiplier
                    * rates
                    * accrual
                )
                payoffs[idx] = principal_component + coupon
                if product.accrual_config.coupon_pay_type == CouponPayType.INSTANT:
                    determination_times = post_table[first_ki_idx[idx], local_idx]
                    settlement_times[idx] = self._resolve_payment_time_array(
                        product,
                        pricing_env,
                        determination_times,
                        CashflowKind.REDEMPTION,
                        records=post_records,
                        record_indices=local_idx,
                        cashflow_id="ko_reset_rebased",
                    )
                else:
                    determination_times = post_table[first_ki_idx[idx], post_last]
                    settlement_times[idx] = self._resolve_payment_time_array(
                        product,
                        pricing_env,
                        determination_times,
                        CashflowKind.TERMINAL,
                        cashflow_id="ko_reset_rebased_terminal",
                    )

        if is_v0.any():
            # rule 2: not knocked in at the end of the first schedule means matured THERE, so a rebate
            # that depends on the spot (call-style) reads the spot of that node, not the grid's last one
            terminal_spots = paths[is_v0, grid["pre_maturity_idx"] + 1]
            v0_payoffs = np.array(
                [
                    product.get_maturity_payoff_v0(spot, pricing_env)
                    for spot in terminal_spots
                ],
                dtype=float,
            )
            payoffs[is_v0] = v0_payoffs if fold is None else v0_payoffs + fold[1]
            # the note that ends with schedule 1: its explicit payment when booked (spec D33, D34), else the delay
            settlement_times[is_v0] = float(first_schedule_terminal_timing(
                product, pricing_env, determination_time=float(pre_maturity),
            ).payment_time)

        if is_v1.any():
            tenors = None
            if post_mode == PostKOScheduleMode.ABSOLUTE:
                # a knocked-in snowball on schedule 2 (a carried knock-in is ABSOLUTE): decided at schedule 2's end,
                # where it settles below -- not the grid's last node, schedule 1's end when that runs on past it
                # (spec D31; the never-knocked-in readout above reads its own schedule's end likewise). Every knock-in
                # meets schedule 2 in force: a knock-in is tested only during schedule 1, which ends no later than
                # schedule 2 (spec D35, refused at construction otherwise), so no path knocks in after schedule 2's end.
                terminal_spots = paths[is_v1, grid["post_maturity_idx"] + 1]
            elif len(post_times) == 0:
                # (a carried knock-in never reaches here: the grid refuses a REBASED one)
                terminal_spots = paths[is_v1, -1]
            else:
                # REBASED: each path matures at its own ki_time + max(offset), before the grid's last node when
                # it knocked in early, so V1 is read there
                v1_paths = np.flatnonzero(is_v1)
                last = int(np.argmax(post_times))
                nodes = np.array(
                    [post_indices_by_ki[int(first_ki_idx[i])][last] for i in v1_paths], dtype=int
                )
                terminal_spots = paths[v1_paths, nodes + 1]
                if leg_annualized(product, "knock_in"):
                    # a REBASED knocked-in tenor hangs from the path's knock-in (accrual spec §4.4, §5.1)
                    anchors = self._knock_in_anchors(grid)
                    by_ki = np.array(
                        [product._resolve_post_contract_tenor(pricing_env, knock_in=a) for a in anchors],
                        dtype=float,
                    )
                    tenors = by_ki[first_ki_idx[v1_paths]]
            v1_payoffs = np.array(
                [
                    product.get_maturity_payoff_v1(
                        spot, pricing_env, contract_tenor=None if tenors is None else float(tenors[n])
                    )
                    for n, spot in enumerate(terminal_spots)
                ],
                dtype=float,
            )
            payoffs[is_v1] = v1_payoffs if fold is None else v1_payoffs + fold[2]
            if post_mode == PostKOScheduleMode.ABSOLUTE:
                settlement_times[is_v1] = self._resolve_payment_time(
                    product,
                    pricing_env,
                    float(post_maturity_abs),
                    CashflowKind.TERMINAL,
                    cashflow_id="ko_reset_v1_terminal",
                )
            else:
                determination_times = (
                    ki_time[is_v1] + post_max_offset
                    if post_table is None
                    else post_table[first_ki_idx[is_v1], post_last]
                )
                settlement_times[is_v1] = self._resolve_payment_time_array(
                    product,
                    pricing_env,
                    determination_times,
                    CashflowKind.TERMINAL,
                    cashflow_id="ko_reset_v1_rebased_terminal",
                )

        ko_time = np.where(is_pre_ko, pre_ko_time, np.where(is_post_ko, post_ko_time, np.inf))

        stats = {
            "ko_probability": float(is_ko.mean()),
            "v0_probability": float(is_v0.mean()),
            "v1_probability": float(is_v1.mean()),
            "ko_count": int(is_ko.sum()),
            "v0_count": int(is_v0.sum()),
            "v1_count": int(is_v1.sum()),
        }

        if is_ko.any():
            stats["avg_ko_time"] = float(np.mean(ko_time[is_ko]))
            stats["ko_time_sum"] = float(np.sum(ko_time[is_ko]))
            stats["ko_time_count"] = int(is_ko.sum())
        else:
            stats["avg_ko_time"] = None
            stats["ko_time_sum"] = 0.0
            stats["ko_time_count"] = 0

        # each path's knocked-in end under REBASED (inf where it never knocked in)
        post_end_time = ki_time + post_max_offset
        if post_mode != PostKOScheduleMode.ABSOLUTE and post_table is not None and not carried_knock_in:
            knocked = first_ki_idx >= 0
            post_end_time = np.full(num_paths, np.inf, dtype=float)
            post_end_time[knocked] = post_table[first_ki_idx[knocked], post_last]

        extra = {
            "is_pre_ko": is_pre_ko,
            "is_post_ko": is_post_ko,
            "post_end_time": post_end_time,
            "first_pre_idx": first_pre_idx,
            "first_post_idx": first_post_idx,
            "ki_triggered": ki_triggered,
            "ki_time": ki_time,
            "ko_time": ko_time,
        }

        if self._stats_recording():
            # event statistics of a recorded price (calculate_event_stats): this batch's outcome sums
            if post_mode == PostKOScheduleMode.ABSOLUTE:
                terminal_determination = np.where(
                    is_v0, float(pre_maturity), float(post_maturity_abs)
                )
            else:
                terminal_determination = np.where(is_v0, float(pre_maturity), post_end_time)
            breached_now = False
            if carried_knock_in or ki_barriers_val is None:
                ki_row_times = np.zeros(0, dtype=float)
            elif ki_continuous:
                ki_row_times = np.asarray(all_times[: ki_horizon_idx + 1], dtype=float)
                breached_now = bool(
                    _ki_level_breached(paths[:1, 0], ki_barrier_scalar, product.is_reverse)[0]
                )
            else:
                ki_row_times = np.asarray(ki_times, dtype=float)
            self._record_stats_batch(
                self._ko_reset_batch_record(
                    payoffs=payoffs,
                    settlement_times=settlement_times,
                    is_pre_ko=is_pre_ko,
                    is_post_ko=is_post_ko,
                    is_v0=is_v0,
                    is_v1=is_v1,
                    first_pre_idx=first_pre_idx,
                    first_post_idx=first_post_idx,
                    n_pre=len(pre_times),
                    n_post=len(post_times),
                    ko_time=ko_time,
                    terminal_determination=terminal_determination,
                    ki_triggered=ki_triggered,
                    first_ki_idx=first_ki_idx,
                    ki_row_times=ki_row_times,
                    ki_time=ki_time,
                    carried_knock_in=carried_knock_in,
                    breached_at_valuation=breached_now,
                    rebased=post_mode != PostKOScheduleMode.ABSOLUTE,
                    principal=(
                        product.initial_price * product.contract_multiplier
                        if product.payoff_config.include_principal
                        else 0.0
                    ),
                )
            )

        return payoffs, settlement_times, stats, extra

    def _antithetic_pairs_in_paths(self) -> bool:
        """True when the simulated paths are the ``[z; -z]`` antithetic layout (spec I3/A9).

        The plain GBM engine draws iid paths whatever ``use_antithetic`` says, so it never pairs; the
        vol-model subclasses override this. The Dask batch route reduces sum / sum-of-squares and cannot
        recover pair means, so it keeps the iid formula.
        """
        return False  # TODO(w16): Dask route (_price_parallel) keeps the iid s.e.; pair sums are not in sum_x/sum_x2

    def _price_mc_or_qmc(
        self,
        product: SnowballOption,
        pricing_env: PricingEnvironment,
        S: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
    ) -> SnowballMCResult:
        """
        Price using normal MC or QMC (non-randomized).
        """
        # Build time grid
        all_times, dt_array, ko_indices, ki_indices = self._build_time_grid(
            product, pricing_env, T
        )

        # Create path generator
        generator = self._create_path_generator(S, r, q, sigma, T, dt_array)

        # Generate paths
        paths, _ = generator.generate_paths(return_aux=False)

        # Compute payoffs
        payoffs, settlement_times, stats = self._compute_payoffs(
            product,
            pricing_env,
            paths,
            all_times,
            ko_indices,
            ki_indices,
            r,
            T,
            sigma,
            rng_seed=int(self.params.seed) + 1337,
        )

        # Discount payoffs
        discount_factors = self._df(settlement_times)
        discounted_payoffs = payoffs * discount_factors

        # Compute price and standard error
        price = float(discounted_payoffs.mean())
        # Antithetic vol-model runs report the pair-mean s.e. (spec I3/A9); plain QMC keeps the iid
        # formula, which is NOT a valid error bar for deterministic QMC (use RQMC).
        std_error = sample_std_error(
            discounted_payoffs, antithetic=self._antithetic_pairs_in_paths()
        )

        return SnowballMCResult(
            price=price,
            std_error=std_error,
            num_paths=len(paths),
            ko_probability=stats["ko_probability"],
            v0_probability=stats["v0_probability"],
            v1_probability=stats["v1_probability"],
            avg_ko_time=stats.get("avg_ko_time"),
        )

    def _price_ko_reset_mc_or_qmc(
        self,
        product: KnockOutResetSnowballOption,
        pricing_env: PricingEnvironment,
        S: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
    ) -> SnowballMCResult:
        """
        Price KO-reset snowball using normal MC or QMC (non-randomized).
        """
        grid = self._build_time_grid_ko_reset(product, pricing_env, T)
        generator = self._create_path_generator(S, r, q, sigma, T, grid["dt_array"])

        paths, _ = generator.generate_paths(return_aux=False)
        payoffs, settlement_times, stats, _ = self._compute_payoffs_ko_reset(
            product,
            pricing_env,
            paths,
            grid,
            r,
            T,
            sigma,
            rng_seed=int(self.params.seed) + 1337,
        )

        discount_factors = self._df(settlement_times)
        discounted_payoffs = payoffs * discount_factors

        price = float(discounted_payoffs.mean())
        if len(payoffs) > 0:
            std_error = sample_std_error(
                discounted_payoffs, antithetic=self._antithetic_pairs_in_paths()
            )
        else:
            std_error = 0.0

        return SnowballMCResult(
            price=price,
            std_error=std_error,
            num_paths=len(paths),
            ko_probability=stats.get("ko_probability", 0.0),
            v0_probability=stats.get("v0_probability", 0.0),
            v1_probability=stats.get("v1_probability", 0.0),
            avg_ko_time=stats.get("avg_ko_time"),
        )

    def _price_single_batch(
        self,
        batch_id: int,
        batch_num_paths: int,
        product: SnowballOption,
        pricing_env: PricingEnvironment,
        S: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        all_times: np.ndarray,
        dt_array: np.ndarray,
        ko_indices: np.ndarray,
        ki_indices: np.ndarray,
    ) -> Dict[str, float]:
        """
        Price a single batch of paths for parallel processing.
        """
        # Create path generator for this batch
        generator = self._create_path_generator(
            S, r, q, sigma, T, dt_array, batch_id=batch_id, num_paths=batch_num_paths
        )

        # Generate paths
        paths, _ = generator.generate_paths(return_aux=False, batch_id=batch_id)

        # Compute payoffs (a recorded price captures this batch's statistics record: the batch may run
        # in another process, so it travels back inside the batch result)
        with self._stats_batch_capture() as captured:
            payoffs, settlement_times, stats = self._compute_payoffs(
                product,
                pricing_env,
                paths,
                all_times,
                ko_indices,
                ki_indices,
                r,
                T,
                sigma,
                rng_seed=int(self.params.seed) + 1337 + int(batch_id) * 1000,
            )

        # Discount payoffs
        discount_factors = self._df(settlement_times)
        discounted_payoffs = payoffs * discount_factors

        discounted_payoffs = np.asarray(discounted_payoffs, dtype=float)
        n = int(discounted_payoffs.size)
        sum_x = float(discounted_payoffs.sum())
        sum_x2 = float(np.square(discounted_payoffs).sum())

        result = {
            "n": n,
            "sum_x": sum_x,
            "sum_x2": sum_x2,
            "ko_count": int(stats.get("ko_count", 0)),
            "v0_count": int(stats.get("v0_count", 0)),
            "v1_count": int(stats.get("v1_count", 0)),
            "ko_time_sum": float(stats.get("ko_time_sum", 0.0)),
            "ko_time_count": int(stats.get("ko_time_count", 0)),
        }
        if captured is not None:
            result[STATS_RECORDS_KEY] = captured
        return result

    def _price_single_batch_ko_reset(
        self,
        batch_id: int,
        batch_num_paths: int,
        product: KnockOutResetSnowballOption,
        pricing_env: PricingEnvironment,
        S: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        grid: Dict[str, object],
    ) -> Dict[str, float]:
        """
        Price a single batch of paths for KO-reset snowball parallel processing.
        """
        generator = self._create_path_generator(
            S, r, q, sigma, T, grid["dt_array"], batch_id=batch_id, num_paths=batch_num_paths
        )

        paths, _ = generator.generate_paths(return_aux=False, batch_id=batch_id)
        with self._stats_batch_capture() as captured:
            payoffs, settlement_times, stats, _ = self._compute_payoffs_ko_reset(
                product,
                pricing_env,
                paths,
                grid,
                r,
                T,
                sigma,
                rng_seed=int(self.params.seed) + 1337 + int(batch_id) * 1000,
            )

        discount_factors = self._df(settlement_times)
        discounted_payoffs = payoffs * discount_factors

        discounted_payoffs = np.asarray(discounted_payoffs, dtype=float)
        n = int(discounted_payoffs.size)
        sum_x = float(discounted_payoffs.sum())
        sum_x2 = float(np.square(discounted_payoffs).sum())

        result = {
            "n": n,
            "sum_x": sum_x,
            "sum_x2": sum_x2,
            "ko_count": int(stats.get("ko_count", 0)),
            "v0_count": int(stats.get("v0_count", 0)),
            "v1_count": int(stats.get("v1_count", 0)),
            "ko_time_sum": float(stats.get("ko_time_sum", 0.0)),
            "ko_time_count": int(stats.get("ko_time_count", 0)),
        }
        if captured is not None:
            result[STATS_RECORDS_KEY] = captured
        return result

    def _price_parallel(
        self,
        product: SnowballOption,
        pricing_env: PricingEnvironment,
        S: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
    ) -> SnowballMCResult:
        """
        Price using Dask parallel batch processing.
        """
        # Build time grid (shared across batches)
        all_times, dt_array, ko_indices, ki_indices = self._build_time_grid(
            product, pricing_env, T
        )

        totals = run_autocallable_dask_batches(
            num_batches=self.num_batches,
            total_paths=int(self.params.num_paths),
            batch_fn=self._price_single_batch,
            batch_kwargs=dict(
                product=product,
                pricing_env=pricing_env,
                S=S,
                T=T,
                r=r,
                q=q,
                sigma=sigma,
                all_times=all_times,
                dt_array=dt_array,
                ko_indices=ko_indices,
                ki_indices=ki_indices,
            ),
            on_batch_results=self._deposit_captured_stats,
        )

        return SnowballMCResult(
            price=totals.price,
            std_error=totals.std_error,
            num_paths=totals.num_paths,
            ko_probability=totals.ko_probability,
            v0_probability=totals.v0_probability,
            v1_probability=totals.v1_probability,
            avg_ko_time=totals.avg_ko_time,
            batches_used=totals.batches_used,
        )

    def _price_ko_reset_parallel(
        self,
        product: KnockOutResetSnowballOption,
        pricing_env: PricingEnvironment,
        S: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
    ) -> SnowballMCResult:
        """
        Price KO-reset snowball using Dask parallel batch processing.
        """
        grid = self._build_time_grid_ko_reset(product, pricing_env, T)

        totals = run_autocallable_dask_batches(
            num_batches=self.num_batches,
            total_paths=int(self.params.num_paths),
            batch_fn=self._price_single_batch_ko_reset,
            batch_kwargs=dict(
                product=product,
                pricing_env=pricing_env,
                S=S,
                T=T,
                r=r,
                q=q,
                sigma=sigma,
                grid=grid,
            ),
            on_batch_results=self._deposit_captured_stats,
        )

        return SnowballMCResult(
            price=totals.price,
            std_error=totals.std_error,
            num_paths=totals.num_paths,
            ko_probability=totals.ko_probability,
            v0_probability=totals.v0_probability,
            v1_probability=totals.v1_probability,
            avg_ko_time=totals.avg_ko_time,
            batches_used=totals.batches_used,
        )

    def _rqmc_spec(
        self,
        product: SnowballOption,
        pricing_env: PricingEnvironment,
        S: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
    ) -> RQMCRunSpec:
        """Build the RQMC run description for a SnowballOption.

        Shared by the direct path (_price_rqmc) and the execution-framework
        session path (build_rqmc_session_spec): ONE construction of the
        grid, pricer, generator, controls, and result assembly.
        """
        # Build time grid
        all_times, dt_array, ko_indices, ki_indices = self._build_time_grid(
            product, pricing_env, T
        )

        def affine_moments(paths, aux):
            if aux is None or aux.get("affine_spot_factor") != "standard_normal":
                return None
            loadings = aux.get("log_spot_factor_loadings")
            if loadings is None:
                raise PricingError(
                    "affine spot-factor metadata is missing log-spot loadings"
                )
            return conditional_standard_snowball_moments(
                product=product,
                pricing_env=pricing_env,
                base_paths=paths,
                log_spot_factor_loadings=loadings,
                ko_indices=ko_indices,
                ki_indices=ki_indices,
                maturity=T,
                discount_factors=self._df,
                **self._conditional_fold(product, pricing_env, T),
            )

        def collapse_conditional_outer(values, aux):
            values = np.asarray(values, dtype=float)
            outer_group_size = int(
                1 if aux is None else aux.get("conditional_outer_group_size", 1)
            )
            if outer_group_size > 1:
                if values.size % outer_group_size:
                    raise PricingError(
                        "bridge-stratified control payoff shape is inconsistent"
                    )
                values = values.reshape(-1, outer_group_size).mean(axis=1)
            return values

        def raw_conditional_control_payoffs(paths, aux):
            if aux is None or not aux.get("heston_conditional_control", False):
                raise PricingError(
                    "Heston conditional-control metadata is unavailable"
                )
            required = (
                "control_base_paths",
                "control_log_spot_factor_loadings",
            )
            missing = [name for name in required if aux.get(name) is None]
            if missing:
                raise PricingError(
                    "Heston conditional-control metadata is incomplete: "
                    + ", ".join(missing)
                )
            return conditional_standard_snowball_moments(
                product=product,
                pricing_env=pricing_env,
                base_paths=aux["control_base_paths"],
                log_spot_factor_loadings=aux[
                    "control_log_spot_factor_loadings"
                ],
                ko_indices=ko_indices,
                ki_indices=ki_indices,
                maturity=T,
                discount_factors=self._df,
                **self._conditional_fold(product, pricing_env, T),
            ).discounted_payoff

        def conditional_control_payoffs(paths, aux):
            return collapse_conditional_outer(
                raw_conditional_control_payoffs(paths, aux), aux
            )

        def pricer_fn(paths, aux):
            """Pricer function for RQMC driver."""
            moments = affine_moments(paths, aux)
            if moments is not None:
                if self._stats_recording():
                    # the exact conditional expectations of the native outcomes on the same paths
                    # and spot-factor loadings the price integrates (recording only)
                    self._record_stats_batch(
                        conditional_standard_snowball_event_record(
                            product=product,
                            pricing_env=pricing_env,
                            base_paths=paths,
                            log_spot_factor_loadings=aux["log_spot_factor_loadings"],
                            ko_indices=ko_indices,
                            ki_indices=ki_indices,
                            maturity=T,
                            discount_factors=self._df,
                        )
                    )
                return collapse_conditional_outer(
                    moments.discounted_payoff, aux
                )
            batch_id = 0
            if aux is not None and "batch_id" in aux:
                batch_id = int(aux["batch_id"])

            def native_discounted(candidate_paths):
                payoffs, settlement_times, _ = self._compute_payoffs(
                    product,
                    pricing_env,
                    candidate_paths,
                    all_times,
                    ko_indices,
                    ki_indices,
                    r,
                    T,
                    sigma,
                    rng_seed=int(self.params.seed) + 1337 + batch_id * 1000,
                )
                return payoffs * self._df(settlement_times)

            control = aux is not None and aux.get("heston_conditional_control", False)
            # a recorded conditional-control batch: the native record is held back and deposited corrected below
            with (self._stats_batch_capture() if control else nullcontext()) as native_records:
                discounted = native_discounted(paths)
            if not control:
                return discounted
            required = (
                "control_paths",
                "control_base_paths",
                "control_log_spot_factor_loadings",
            )
            missing = [name for name in required if aux.get(name) is None]
            if missing:
                raise PricingError(
                    "Heston conditional-control metadata is incomplete: "
                    + ", ".join(missing)
                )
            with self._stats_batch_capture() as control_records:
                sampled_control = native_discounted(aux["control_paths"])
            exact_control = raw_conditional_control_payoffs(paths, aux)
            if native_records and control_records:
                # the price is native - sampled proxy + exact proxy: so is every recorded outcome (T10). The
                # proxy's exact conditional outcomes on the same base rows and loadings the price integrates.
                self._record_stats_batch(
                    control_corrected_record(
                        native_records[0],
                        control_records[0],
                        conditional_standard_snowball_event_record(
                            product=product,
                            pricing_env=pricing_env,
                            base_paths=aux["control_base_paths"],
                            log_spot_factor_loadings=aux["control_log_spot_factor_loadings"],
                            ko_indices=ko_indices,
                            ki_indices=ki_indices,
                            maturity=T,
                            discount_factors=self._df,
                        ),
                    )
                )
            residual = discounted - sampled_control
            group_size = int(aux.get("conditional_group_size", 1))
            if group_size > 1:
                if residual.size != exact_control.size * group_size:
                    raise PricingError(
                        "stratified spot-control payoff shape is inconsistent"
                    )
                residual = residual.reshape(-1, group_size).mean(axis=1)
            return collapse_conditional_outer(residual + exact_control, aux)

        params = self.params
        max_batches = getattr(
            params, "rqmc_max_batches", getattr(params, "max_batches", 32)
        )
        min_batches = getattr(
            params, "rqmc_min_batches", getattr(params, "min_batches", 4)
        )
        if hasattr(params, "resolve_rqmc_target_std"):
            target_std = params.resolve_rqmc_target_std(
                product=product, pricing_env=pricing_env
            )
        else:
            target_std = getattr(params, "target_std", 1e-4)
        if hasattr(params, "resolve_rqmc_paths_per_batch"):
            per_batch_paths = params.resolve_rqmc_paths_per_batch(
                max_batches=max_batches
            )
        else:
            per_batch_paths = params.num_paths

        # Create path generator (will be used with different batch_ids)
        generator = self._create_path_generator(
            S, r, q, sigma, T, dt_array, num_paths=per_batch_paths
        )

        def finalize(result):
            # Run one more batch to get statistics
            paths, aux = generator.generate_paths(return_aux=True, batch_id=0)
            moments = affine_moments(paths, aux)
            if moments is not None:
                ko_probability = float(np.mean(moments.ko_probability))
                ko_time_weight = float(np.sum(moments.ko_time_numerator))
                ko_probability_weight = float(np.sum(moments.ko_probability))
                return SnowballMCResult(
                    price=result.price,
                    std_error=result.std_error,
                    num_paths=result.total_paths,
                    ko_probability=ko_probability,
                    v0_probability=float(np.mean(moments.v0_probability)),
                    v1_probability=float(np.mean(moments.v1_probability)),
                    avg_ko_time=(
                        ko_time_weight / ko_probability_weight
                        if ko_probability_weight > 0.0
                        else None
                    ),
                    batches_used=result.batches_used,
                )
            # a result-assembly rerun of batch 0, not a priced batch: not recorded
            with self._without_stats_recording():
                _, _, stats = self._compute_payoffs(
                    product,
                    pricing_env,
                    paths,
                    all_times,
                    ko_indices,
                    ki_indices,
                    r,
                    T,
                    sigma,
                    rng_seed=int(self.params.seed) + 1337,
                )

            return SnowballMCResult(
                price=result.price,
                std_error=result.std_error,
                num_paths=result.total_paths,
                ko_probability=stats["ko_probability"],
                v0_probability=stats["v0_probability"],
                v1_probability=stats["v1_probability"],
                batches_used=result.batches_used,
            )

        conditional_control_only = bool(
            getattr(self, "rqmc_conditional_control_only", False)
        )
        return RQMCRunSpec(
            pricer_fn=pricer_fn,
            path_generator=generator,
            max_batches=max_batches,
            min_batches=min_batches,
            target_std=target_std,
            paths_per_batch=per_batch_paths,
            time_steps=int(dt_array.size),
            scheme=self._rqmc_scheme_label(),
            finalize=finalize,
            product=product,
            dimension=self._rqmc_dimension(int(dt_array.size)),
            randomization_key=self._rqmc_randomization_key(int(dt_array.size)),
            control_pricer_fn=(
                conditional_control_payoffs
                if getattr(self, "rqmc_heston_conditional_control", False)
                else None
            ),
            path_valuation_multiplier=(
                2 * int(getattr(self, "rqmc_spot_bridge_strata", 1))
                if conditional_control_only
                else (
                    int(getattr(self, "rqmc_spot_strata", 1))
                    * (
                        2
                        if getattr(self, "rqmc_spot_antithetic", False)
                        else 1
                    )
                    * int(getattr(self, "rqmc_spot_bridge_strata", 1))
                    if getattr(self, "rqmc_heston_conditional_control", False)
                    else (
                        int(getattr(self, "rqmc_spot_bridge_strata", 1))
                        if getattr(self, "rqmc_affine_spot_factor", False)
                        else 1
                    )
                )
            ),
            homogeneous_spot_scaling=bool(
                getattr(self, "rqmc_homogeneous_spot_scaling", False)
            ),
            initial_spot=float(S),
        )

    def _price_rqmc(
        self,
        product: SnowballOption,
        pricing_env: PricingEnvironment,
        S: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
    ) -> SnowballMCResult:
        """
        Price using Randomized QMC with adaptive batching.
        """
        spec = self._rqmc_spec(product, pricing_env, S, T, r, q, sigma)
        return spec.finalize(run_rqmc(
            pricer_fn=spec.pricer_fn,
            path_generator=spec.path_generator,
            max_batches=spec.max_batches,
            target_std=spec.target_std,
            min_batches=spec.min_batches,
        ))

    def _ko_reset_rqmc_spec(
        self,
        product: KnockOutResetSnowballOption,
        pricing_env: PricingEnvironment,
        S: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
    ) -> RQMCRunSpec:
        """Build the RQMC run description for a KO-reset snowball (shared
        by the direct and session paths, mirroring _rqmc_spec)."""
        grid = self._build_time_grid_ko_reset(product, pricing_env, T)

        def pricer_fn(paths, aux):
            batch_id = 0
            if aux is not None and "batch_id" in aux:
                batch_id = int(aux["batch_id"])
            payoffs, settlement_times, _stats, _ = self._compute_payoffs_ko_reset(
                product,
                pricing_env,
                paths,
                grid,
                r,
                T,
                sigma,
                rng_seed=int(self.params.seed) + 1337 + batch_id * 1000,
            )
            discount_factors = self._df(settlement_times)
            return payoffs * discount_factors

        params = self.params
        max_batches = getattr(
            params, "rqmc_max_batches", getattr(params, "max_batches", 32)
        )
        min_batches = getattr(
            params, "rqmc_min_batches", getattr(params, "min_batches", 4)
        )
        if hasattr(params, "resolve_rqmc_target_std"):
            target_std = params.resolve_rqmc_target_std(
                product=product, pricing_env=pricing_env
            )
        else:
            target_std = getattr(params, "target_std", 1e-4)
        if hasattr(params, "resolve_rqmc_paths_per_batch"):
            per_batch_paths = params.resolve_rqmc_paths_per_batch(
                max_batches=max_batches
            )
        else:
            per_batch_paths = params.num_paths

        generator = self._create_path_generator(
            S, r, q, sigma, T, grid["dt_array"], num_paths=per_batch_paths
        )

        def finalize(result):
            paths, _ = generator.generate_paths(return_aux=False, batch_id=0)
            # a result-assembly rerun of batch 0, not a priced batch: not recorded
            with self._without_stats_recording():
                _, _, stats, _ = self._compute_payoffs_ko_reset(
                    product,
                    pricing_env,
                    paths,
                    grid,
                    r,
                    T,
                    sigma,
                    rng_seed=int(self.params.seed) + 1337,
                )

            return SnowballMCResult(
                price=result.price,
                std_error=result.std_error,
                num_paths=result.total_paths,
                ko_probability=stats.get("ko_probability", 0.0),
                v0_probability=stats.get("v0_probability", 0.0),
                v1_probability=stats.get("v1_probability", 0.0),
                batches_used=result.batches_used,
            )

        return RQMCRunSpec(
            pricer_fn=pricer_fn,
            path_generator=generator,
            max_batches=max_batches,
            min_batches=min_batches,
            target_std=target_std,
            paths_per_batch=per_batch_paths,
            time_steps=int(grid["dt_array"].size),
            scheme=self._rqmc_scheme_label() + "#ko-reset",
            finalize=finalize,
            product=product,
            dimension=(
                self._rqmc_streams_per_step()
                * int(grid["dt_array"].size)
                * self._rqmc_substep_factor()
            ),
            randomization_key=(
                "scrambled_sobol",
                int(self.params.seed),
                self._rqmc_scheme_label() + "#ko-reset",
                int(grid["dt_array"].size),
                self._rqmc_substep_factor(),
            ),
        )

    def _price_ko_reset_rqmc(
        self,
        product: KnockOutResetSnowballOption,
        pricing_env: PricingEnvironment,
        S: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
    ) -> SnowballMCResult:
        """
        Price KO-reset snowball using Randomized QMC with adaptive batching.
        """
        spec = self._ko_reset_rqmc_spec(product, pricing_env, S, T, r, q, sigma)
        return spec.finalize(run_rqmc(
            pricer_fn=spec.pricer_fn,
            path_generator=spec.path_generator,
            max_batches=spec.max_batches,
            target_std=spec.target_std,
            min_batches=spec.min_batches,
        ))

    def get_last_result(self) -> Optional[SnowballMCResult]:
        """
        Get the full result from the last pricing run.

        Returns:
            SnowballMCResult object, or None if no pricing has been performed
        """
        return self._last_result

    def get_last_std_error(self) -> Optional[float]:
        """
        Get the standard error from the last pricing run.

        Returns:
            Standard error, or None if no pricing has been performed
        """
        if self._last_result is None:
            return None
        return self._last_result.std_error

    def __repr__(self):
        dask_str = f", use_dask={self.use_dask}" if self.use_dask else ""
        return f"SnowballMCEngine(method={self.method.name}{dask_str})"
