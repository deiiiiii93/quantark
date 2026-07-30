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
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.engine.capabilities import SettlementSupport
from quantark.asset.equity.engine.event_stats import (
    AutocallableEventStats,
    KOResetEventStats,
    payment_aware_cashflow_fields,
)
from quantark.asset.equity.engine.mc.autocallable_payment_timings import (
    resolve_autocallable_payment_timings,
)
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.process.bsm.qmc_path_generator import GBMPathGenerator
from quantark.montecarlo.conditional_snowball import (
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
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord
from quantark.asset.equity.settlement import (
    CashflowKind,
    SettlementRequest,
    SettlementResolver,
)
from quantark.asset.equity.engine.mc.term_inputs import build_mc_term_inputs, make_df_fn
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import DayCountConvention, calculate_year_fraction
from quantark.util.enum import (
    CouponPayType,
    ObservationType,
    PostKOScheduleMode,
)
from quantark.util.enum.engine_enums import EngineType, MonteCarloMethod
from quantark.util.exceptions import PricingError, ValidationError
from quantark.util.numerical import safe_log

# Optional Dask import
from quantark.asset.equity.engine.mc.autocallable_dask_batch import (
    run_autocallable_dask_batches,
)

try:
    from dask import compute, delayed  # noqa: F401 — availability probe

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False


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


class SnowballMCEngine(BaseEngine):
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

        Raises:
            ValidationError: If method is invalid or params are invalid
        """
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


        # Handle near-expiry case
        if T < 1e-10:
            knocked_in = bool(getattr(product, "_otc_lifecycle_knocked_in", False))
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

    def _complete_price(self, product, result) -> float:
        """price() postamble shared with the session adaptive path."""
        self._last_result = result
        # Negative PV is valid for some structures (e.g., principal-excluded notes)
        # where the payoff is effectively "coupon minus embedded option loss".
        if result.price < 0 and product.payoff_config.include_principal:
            raise PricingError(f"Negative price computed: {result.price}")
        return result.price

    def _prepare_payment_timings(self, product, pricing_env) -> None:
        """Resolve standard Snowball KO and terminal payment timing once."""
        if isinstance(product, KnockOutResetSnowballOption):
            self._payment_timings = None
            return
        records = product.resolve_ko_observations(pricing_env)
        self._payment_timings = resolve_autocallable_payment_timings(
            product,
            pricing_env,
            records,
            event_paid=(
                product.accrual_config.coupon_pay_type
                == CouponPayType.INSTANT
            ),
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
        return f"{type(self).__qualname__}{suffix}/rqmc-native/v1"

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
        if T < 1e-10:
            return None
        if isinstance(product, KnockOutResetSnowballOption):
            return self._ko_reset_rqmc_spec(
                product, pricing_env, S, T, r, q, sigma
            )
        return self._rqmc_spec(product, pricing_env, S, T, r, q, sigma)

    def calculate_event_stats(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> Optional[AutocallableEventStats]:
        """
        Provide per-observation event probabilities and expected discounted cashflows.

        This is a Monte Carlo implementation for Snowball. QUAD/PDE engines can
        optionally implement the same API later to provide faster risk-neutral
        event stats without Monte Carlo.
        """
        if isinstance(product, KnockOutResetSnowballOption):
            return self._calculate_event_stats_ko_reset(product, pricing_env)
        if not isinstance(product, SnowballOption):
            return None

        S = pricing_env.spot
        T = product.get_maturity(pricing_env)
        r = pricing_env.get_rate(T)
        q = pricing_env.get_div_yield(T)
        sigma = pricing_env.get_vol(product.strike, T)
        # Term-structure context for this entry point (see term_inputs.py)
        self._term_ctx = (pricing_env, product.strike)
        self._df = make_df_fn(pricing_env)
        self._prepare_payment_timings(product, pricing_env)
        self._validate_inputs(S, T, r, q, sigma, product)

        all_times, dt_array, ko_indices, ki_indices = self._build_time_grid(
            product, pricing_env, T
        )
        generator = self._create_path_generator(S, r, q, sigma, T, dt_array)
        paths, _ = generator.generate_paths(return_aux=False)

        ko_profile = product.get_ko_observation_profile(pricing_env)
        ko_times = np.array(ko_profile["observation_times"], dtype=float)
        ko_barriers = np.array(ko_profile["barriers"], dtype=float)
        ko_payoffs = np.array(ko_profile["payoffs"], dtype=float)
        timings = self._payment_timings
        ko_settlement_times = np.asarray(
            timings.observation_payment_times,
            dtype=float,
        )

        ko_triggered, first_ko_idx = self._check_ko_barriers(
            paths, ko_indices, ko_barriers, product.is_reverse
        )

        ki_triggered = np.zeros(len(paths), dtype=bool)
        first_ki_idx = np.full(len(paths), -1, dtype=int)
        ki_event_times = np.array([], dtype=float)
        if product.has_ki_barrier:
            ki_continuous = (
                product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
                or product.barrier_config.ki_continuous
            )
            ki_profile = product.get_ki_observation_profile(pricing_env)
            ki_barriers_val = np.array(ki_profile["barriers"], dtype=float)
            if ki_continuous:
                ki_event_times = np.array(all_times, dtype=float)
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
                        rng_seed=int(self.params.seed) + 1337,
                    )
                )
            else:
                ki_event_times = np.array(ki_profile["observation_times"], dtype=float)
                ki_triggered, first_ki_idx = self._check_ki_barriers(
                    paths, ki_indices, ki_barriers_val, product.is_reverse
                )

        already_knocked_in = bool(getattr(product, "_otc_lifecycle_knocked_in", False))
        if already_knocked_in:
            ki_triggered[:] = True
            first_ki_idx[:] = 0

        if product.barrier_config.disable_ko_after_ki and (
            product.has_ki_barrier or already_knocked_in
        ):
            if already_knocked_in:
                ko_valid = np.zeros_like(ko_triggered)
            else:
                ko_trigger_times = np.where(
                    first_ko_idx >= 0, ko_times[first_ko_idx], np.inf
                )
                if len(ki_event_times) > 0:
                    ki_trigger_times = np.where(
                        first_ki_idx >= 0, ki_event_times[first_ki_idx], np.inf
                    )
                else:
                    ki_trigger_times = np.full(len(paths), np.inf, dtype=float)
                ko_valid = ko_triggered & (ko_trigger_times < ki_trigger_times)
        else:
            ko_valid = ko_triggered

        is_ko = ko_valid
        is_v0 = ~is_ko & ~ki_triggered
        is_v1 = ~is_ko & ki_triggered

        ko_probability = np.zeros(len(ko_times), dtype=float)
        expected_discounted_ko_cashflow = np.zeros(len(ko_times), dtype=float)
        for i in range(len(ko_times)):
            hit_i = is_ko & (first_ko_idx == i)
            p_i = float(np.mean(hit_i))
            ko_probability[i] = p_i
            if p_i > 0.0:
                df = timings.observation_payment_dfs[i]
                expected_discounted_ko_cashflow[i] = p_i * float(ko_payoffs[i]) * df

        survival_probability = np.ones(len(ko_times), dtype=float)
        cumulative_ko = 0.0
        for i in range(len(ko_times)):
            cumulative_ko += ko_probability[i]
            survival_probability[i] = max(0.0, 1.0 - cumulative_ko)

        ki_event_probability = np.array([], dtype=float)
        ki_survival_probability = np.array([], dtype=float)
        if already_knocked_in:
            ki_event_times = np.array([0.0], dtype=float)
            ki_event_probability = np.array([1.0], dtype=float)
            ki_survival_probability = np.array([0.0], dtype=float)
        elif product.has_ki_barrier:
            if ki_event_times.size:
                ki_event_probability = np.zeros(len(ki_event_times), dtype=float)
                for i in range(len(ki_event_times)):
                    ki_event_probability[i] = float(
                        np.mean(ki_triggered & (first_ki_idx == i))
                    )
                ki_survival_probability = np.maximum(
                    0.0, 1.0 - np.cumsum(ki_event_probability)
                )

        maturity_spots = paths[:, -1]
        maturity_df = timings.terminal.payment_df
        maturity_payoff_all = np.zeros(len(paths), dtype=float)
        if is_v0.any():
            maturity_payoff_all[is_v0] = np.array(
                [
                    product.get_maturity_payoff_v0(float(s), pricing_env)
                    for s in maturity_spots[is_v0]
                ],
                dtype=float,
            )
        if is_v1.any():
            maturity_payoff_all[is_v1] = np.array(
                [
                    product.get_maturity_payoff_v1(float(s), pricing_env)
                    for s in maturity_spots[is_v1]
                ],
                dtype=float,
            )
        # Total discounted payoff PV from the same simulation.
        total_discounted = np.zeros(len(paths), dtype=float)
        if is_ko.any():
            payoff = ko_payoffs[first_ko_idx[is_ko]]
            settle = ko_settlement_times[first_ko_idx[is_ko]]
            dfs = np.array(
                [pricing_env.get_discount_factor(float(t)) for t in settle], dtype=float
            )
            total_discounted[is_ko] = payoff * dfs
        total_discounted[~is_ko] = maturity_payoff_all[~is_ko] * maturity_df

        pv = float(np.mean(total_discounted))
        expected_discounted_maturity_cashflow = float(
            pv - np.sum(expected_discounted_ko_cashflow)
        )
        reconciliation_error = 0.0

        records = product.resolve_ko_observations(pricing_env)
        terminal = timings.terminal
        cashflow_fields = payment_aware_cashflow_fields(
            pricing_env,
            determination_times=np.concatenate(
                [ko_times, [terminal.determination_time]]
            ),
            payment_times=np.concatenate(
                [
                    timings.observation_payment_times,
                    [terminal.payment_time],
                ]
            ),
            expected_discounted_cashflows=np.concatenate(
                [
                    expected_discounted_ko_cashflow,
                    [expected_discounted_maturity_cashflow],
                ]
            ),
            determination_dates=[
                *(record.observation_date for record in records),
                terminal.determination_date,
            ],
            payment_dates=[
                *(
                    record.settlement_date
                    if (
                        product.accrual_config.coupon_pay_type
                        == CouponPayType.INSTANT
                    )
                    else terminal.payment_date
                    for record in records
                ),
                terminal.payment_date,
            ],
        )

        # Unambiguous, cross-engine-consistent KI definitions (investigation
        # 2026-07-01): ki_ever = P(KI ever, KO-independent); ki_survive =
        # P(KI ever AND never KO) = settles knocked-in. Legacy ki_probability keeps
        # MC's historical "KI ever" meaning.
        if product.has_ki_barrier or already_knocked_in:
            ki_ever_probability = float(np.mean(ki_triggered))
            ki_survive_knocked_in_probability = float(np.mean(ki_triggered & ~is_ko))
        else:
            ki_ever_probability = 0.0
            ki_survive_knocked_in_probability = 0.0

        return AutocallableEventStats(
            pv=pv,
            ko_times=ko_times,
            ko_probability=ko_probability,
            survival_probability=survival_probability,
            expected_discounted_ko_cashflow=expected_discounted_ko_cashflow,
            ki_probability=ki_ever_probability,
            expected_discounted_maturity_cashflow=expected_discounted_maturity_cashflow,
            reconciliation_error=float(reconciliation_error),
            ki_times=ki_event_times,
            ki_event_probability=ki_event_probability,
            ki_survival_probability=ki_survival_probability,
            ki_ever_probability=ki_ever_probability,
            ki_survive_knocked_in_probability=ki_survive_knocked_in_probability,
            **cashflow_fields,
        )

    def _calculate_event_stats_ko_reset(
        self,
        product: KnockOutResetSnowballOption,
        pricing_env: PricingEnvironment,
    ) -> KOResetEventStats:
        S = pricing_env.spot
        T = product.get_max_maturity_time(pricing_env)
        r = pricing_env.get_rate(T)
        q = pricing_env.get_div_yield(T)
        sigma = pricing_env.get_vol(product.strike, T)
        # Term-structure context for this entry point (see term_inputs.py)
        self._term_ctx = (pricing_env, product.strike)
        self._df = make_df_fn(pricing_env)
        self._prepare_payment_timings(product, pricing_env)
        self._validate_inputs(S, T, r, q, sigma, product)

        grid = self._build_time_grid_ko_reset(product, pricing_env, T)
        generator = self._create_path_generator(S, r, q, sigma, T, grid["dt_array"])
        paths, _ = generator.generate_paths(return_aux=False)

        payoffs, settlement_times, _stats, extra = self._compute_payoffs_ko_reset(
            product,
            pricing_env,
            paths,
            grid,
            r,
            T,
            sigma,
            rng_seed=int(self.params.seed) + 1337,
        )

        is_pre_ko = extra["is_pre_ko"]
        is_post_ko = extra["is_post_ko"]
        is_ko = is_pre_ko | is_post_ko
        ki_triggered = extra["ki_triggered"]
        is_v0 = ~is_ko & ~ki_triggered
        is_v1 = ~is_ko & ki_triggered

        pre_times = grid["pre_times"]
        post_times = grid["post_times"]
        post_mode = grid["post_profile"]["mode"]
        post_maturity_abs = (
            product.get_post_maturity_time(pricing_env)
            if post_mode == PostKOScheduleMode.ABSOLUTE
            else None
        )
        post_max_offset = (
            float(max(post_times)) if len(post_times) else 0.0
        )

        pre_prob = np.zeros(len(pre_times), dtype=float)
        for i in range(len(pre_times)):
            pre_prob[i] = float(np.mean(is_pre_ko & (extra["first_pre_idx"] == i)))

        post_prob = np.zeros(len(post_times), dtype=float)
        for i in range(len(post_times)):
            post_prob[i] = float(np.mean(is_post_ko & (extra["first_post_idx"] == i)))

        pre_prob_total = float(pre_prob.sum())
        post_prob_total = float(post_prob.sum())

        pre_profile = grid["pre_profile"]
        pre_rates = np.array(pre_profile["rates"], dtype=float)
        pre_records = pre_profile["records"]
        pre_maturity = product.get_pre_maturity_time(pricing_env)
        pre_payoffs, pre_settlement_times = self._compute_ko_schedule_payoffs(
            product,
            pre_times,
            pre_rates,
            pre_records,
            pricing_env,
            pre_maturity,
        )

        expected_discounted_ko_cashflow = np.zeros(len(pre_times), dtype=float)
        for i in range(len(pre_times)):
            if pre_prob[i] <= 0.0:
                continue
            df = pricing_env.get_discount_factor(float(pre_settlement_times[i]))
            expected_discounted_ko_cashflow[i] = pre_prob[i] * float(pre_payoffs[i]) * df

        expected_discounted_post_ko_cashflow = 0.0
        if is_post_ko.any():
            dfs = np.array(
                [pricing_env.get_discount_factor(float(t)) for t in settlement_times[is_post_ko]],
                dtype=float,
            )
            expected_discounted_post_ko_cashflow = float(
                np.sum(payoffs[is_post_ko] * dfs) / len(paths)
            )

        dfs_all = np.array(
            [pricing_env.get_discount_factor(float(t)) for t in settlement_times],
            dtype=float,
        )
        maturity_payoff_all = np.zeros(len(paths), dtype=float)
        maturity_payoff_all[~is_ko] = payoffs[~is_ko]
        expected_discounted_maturity_cashflow = float(
            np.mean(maturity_payoff_all * dfs_all)
        )

        total_discounted = payoffs * dfs_all
        pv = float(np.mean(total_discounted))
        reconciliation_error = 0.0

        determination_ledger = list(np.asarray(pre_times, dtype=float))
        payment_ledger = list(np.asarray(pre_settlement_times, dtype=float))
        discounted_ledger = list(expected_discounted_ko_cashflow)

        def _append_grouped_path_cashflows(mask, determination_times):
            indices = np.flatnonzero(mask)
            if indices.size == 0:
                return
            grouped = {}
            for path_index, determination_time in zip(
                indices,
                np.asarray(determination_times, dtype=float),
            ):
                payment_time = float(settlement_times[path_index])
                key = (float(determination_time), payment_time)
                discounted = float(
                    payoffs[path_index]
                    * pricing_env.get_discount_factor(payment_time)
                    / len(paths)
                )
                grouped[key] = grouped.get(key, 0.0) + discounted
            for (determination_time, payment_time), discounted in sorted(
                grouped.items()
            ):
                determination_ledger.append(determination_time)
                payment_ledger.append(payment_time)
                discounted_ledger.append(discounted)

        _append_grouped_path_cashflows(
            is_post_ko,
            extra["ko_time"][is_post_ko],
        )
        if post_mode == PostKOScheduleMode.ABSOLUTE:
            terminal_determination = np.where(
                is_v0,
                float(pre_maturity),
                float(post_maturity_abs),
            )
        else:
            terminal_determination = np.where(
                is_v0,
                float(pre_maturity),
                extra["ki_time"] + post_max_offset,
            )
        _append_grouped_path_cashflows(
            ~is_ko,
            terminal_determination[~is_ko],
        )

        cashflow_fields = payment_aware_cashflow_fields(
            pricing_env,
            determination_times=determination_ledger,
            payment_times=payment_ledger,
            expected_discounted_cashflows=discounted_ledger,
        )
        ledger_sum = float(np.sum(cashflow_fields["expected_discounted_cashflows"]))
        if discounted_ledger:
            adjusted = np.array(
                cashflow_fields["expected_discounted_cashflows"],
                copy=True,
            )
            adjusted[-1] += pv - ledger_sum
            cashflow_fields = payment_aware_cashflow_fields(
                pricing_env,
                determination_times=determination_ledger,
                payment_times=payment_ledger,
                expected_discounted_cashflows=adjusted,
            )

        survival_probability = np.ones(len(pre_times), dtype=float)
        cumulative_ko = 0.0
        for i in range(len(pre_times)):
            cumulative_ko += pre_prob[i]
            survival_probability[i] = max(0.0, 1.0 - cumulative_ko)

        return KOResetEventStats(
            pv=pv,
            ko_times=pre_times,
            ko_probability=pre_prob,
            survival_probability=survival_probability,
            expected_discounted_ko_cashflow=expected_discounted_ko_cashflow,
            ki_probability=float(np.mean(ki_triggered)),
            expected_discounted_maturity_cashflow=expected_discounted_maturity_cashflow,
            reconciliation_error=float(reconciliation_error),
            # ki_ever = P(KI ever, KO-independent); ki_survive = P(KI ever AND
            # never KO'd, pre- or post-reset). Legacy ki_probability stays "KI ever".
            ki_ever_probability=float(np.mean(ki_triggered)),
            ki_survive_knocked_in_probability=float(np.mean(ki_triggered & ~is_ko)),
            pre_ko_times=pre_times,
            pre_ko_probability=pre_prob,
            post_ko_times=post_times,
            post_ko_probability=post_prob,
            pre_ko_probability_total=pre_prob_total,
            post_ko_probability_total=post_prob_total,
            expected_discounted_post_ko_cashflow=float(expected_discounted_post_ko_cashflow),
            **cashflow_fields,
        )

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
                num_ki_steps = int(pricing_env.bus_days_in_year * T) + 1
                ki_times = list(np.linspace(0, T, num_ki_steps + 1)[1:])
            else:
                ki_profile = product.get_ki_observation_profile(pricing_env)
                ki_times = ki_profile["observation_times"]

        # Recording the realized per-step variance costs a second
        # paths-sized array, so the vol-model engines only do it when a
        # continuously monitored KI barrier will actually run the bridge.
        self._ki_bridge_wanted = bool(product.has_ki_barrier and ki_continuous)

        # Combine all times and ensure maturity is included
        all_times_set = set(ko_times) | set(ki_times) | {T}
        all_times = np.array(sorted(all_times_set))

        # Build dt_array
        times_with_zero = np.concatenate([[0.0], all_times])
        dt_array = np.diff(times_with_zero)

        # Find indices for KO and KI observations
        ko_indices = np.searchsorted(all_times, ko_times)
        if ki_continuous:
            # All times except t=0 are KI observation points
            ki_indices = np.arange(len(all_times))
        elif ki_times:
            ki_indices = np.searchsorted(all_times, ki_times)
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

        ki_continuous = (
            product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
            or product.barrier_config.ki_continuous
        )
        ki_times: List[float] = []
        if ki_continuous:
            ki_horizon = product.get_pre_maturity_time(pricing_env)
            num_ki_steps = int(pricing_env.bus_days_in_year * ki_horizon) + 1
            ki_times = list(np.linspace(0, ki_horizon, num_ki_steps + 1)[1:])
        else:
            ki_profile = product.get_ki_observation_profile(pricing_env)
            ki_times = ki_profile["observation_times"]

        all_times_set = set(pre_times) | set(ki_times) | {T}

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
                current = pricing_env.valuation_date
                count = 1 if calendar.is_business_day(current) else 0
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
        if post_mode == PostKOScheduleMode.ABSOLUTE:
            all_times_set |= set(post_times)
        else:
            # REBASED: treat post_times as offsets from KI time
            if ki_continuous:
                raise ValidationError(
                    "Rebased post-KO schedule requires discrete KI monitoring."
                )
            for t_ki in ki_times:
                for offset in post_times:
                    all_times_set.add(t_ki + offset)

        all_times = np.array(sorted(all_times_set), dtype=float)
        times_with_zero = np.concatenate([[0.0], all_times])
        dt_array = np.diff(times_with_zero)

        pre_indices = np.searchsorted(all_times, pre_times)

        if post_mode == PostKOScheduleMode.ABSOLUTE:
            post_indices = np.searchsorted(all_times, post_times)
        else:
            post_indices = np.array([], dtype=int)
            post_indices_by_ki = []
            for t_ki in ki_times:
                actual_times = t_ki + post_times
                post_indices_by_ki.append(np.searchsorted(all_times, actual_times))

        if ki_continuous:
            ki_indices = np.array([], dtype=int)
            ki_horizon_idx = max(
                0, int(np.searchsorted(all_times, product.get_pre_maturity_time(pricing_env), side="right") - 1)
            )
        else:
            ki_indices = np.searchsorted(all_times, np.array(ki_times, dtype=float))
            ki_horizon_idx = None

        # See _build_time_grid: the vol-model engines record the realized
        # per-step variance only when the bridge will actually run.
        self._ki_bridge_wanted = bool(product.has_ki_barrier and ki_continuous)

        return {
            "all_times": all_times,
            "dt_array": dt_array,
            "pre_times": pre_times,
            "pre_indices": pre_indices,
            "post_times": post_times,
            "post_indices": post_indices,
            "post_indices_by_ki": post_indices_by_ki,
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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute KO payoffs and settlement times for a resolved schedule.
        """
        if len(times) == 0:
            return np.array([]), np.array([])

        principal_component = (
            product.initial_price * product.contract_multiplier
            if product.payoff_config.include_principal
            else 0.0
        )
        annualized_ko = product._effective_annualized_flag(
            product.accrual_config.is_annualized_ko
        )

        payoffs = np.zeros(len(times), dtype=float)
        for idx, t in enumerate(times):
            if annualized_ko:
                accrual_factor = product.compute_ko_accrual_factor(
                    float(t), records[idx], pricing_env
                )
            else:
                accrual_factor = 1.0
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
        self, paths: np.ndarray
    ) -> Optional[np.ndarray]:
        """Log-variance accumulated over each monitoring interval, per path.

        Shape ``(n_paths, n_steps)``, or ``None`` to use the scalar ``sigma``
        the bridge was handed. The Brownian-bridge crossing probability is
        governed by the quadratic variation of log S over the interval, so a
        constant-vol engine is exactly served by ``sigma**2 * dt`` -- but a
        local-vol, Heston or SLV path accumulates variance those engines
        simulate and this engine never sees. They override.
        """
        return None

    def _check_ki_barriers_continuous_with_bridge(
        self,
        paths: np.ndarray,
        all_times: np.ndarray,
        ki_barrier: float,
        sigma: float,
        is_reverse: bool,
        rng_seed: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Continuous KI monitoring with Brownian-bridge barrier correction.

        The path is simulated on a discrete grid. If both endpoints of an interval
        are on the non-breached side of the KI barrier, a Brownian-bridge estimate
        is used to sample whether a barrier hit occurred within the interval.
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
        spot0 = paths[:, 0]
        if is_reverse:
            already_breached = spot0 >= ki_barrier
        else:
            already_breached = spot0 <= ki_barrier
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

        dt = np.empty(n_steps, dtype=float)
        dt[0] = float(all_times[0])
        if n_steps > 1:
            dt[1:] = np.diff(all_times)
        if np.any(dt <= 0.0):
            raise ValidationError("all_times must be strictly increasing and > 0")

        rng = np.random.default_rng(int(rng_seed))
        step_log_variance = self._ki_bridge_step_log_variance(paths)

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
                ki_trigger_times = np.where(
                    first_ki_idx >= 0,
                    all_times[ki_indices[first_ki_idx]]
                    if len(ki_indices) > 0
                    else np.inf,
                    np.inf,
                )

                # KO is only valid if it happens before KI
                ko_before_ki = ko_trigger_times < ki_trigger_times
                ko_valid = ko_triggered & ko_before_ki
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
            payoffs[is_v0] = v0_payoffs

        # V1 payoffs (never KO, KI happened)
        if is_v1.any():
            terminal_spots = paths[is_v1, -1]
            v1_payoffs = np.array(
                [
                    product.get_maturity_payoff_v1(spot, pricing_env)
                    for spot in terminal_spots
                ]
            )
            payoffs[is_v1] = v1_payoffs

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

        return payoffs, settlement_times, stats

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

        # KO payoffs for pre schedule
        pre_payoffs, pre_settlement_times = self._compute_ko_schedule_payoffs(
            product,
            pre_times,
            pre_rates,
            pre_records,
            pricing_env,
            pre_maturity,
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

        # KI barriers
        ki_barriers_val = None
        if product.has_ki_barrier:
            ki_profile = product.get_ki_observation_profile(pricing_env)
            ki_barriers_val = np.array(ki_profile["barriers"], dtype=float)

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
                    )
                )
            else:
                ki_triggered, first_ki_idx = self._check_ki_barriers(
                    paths, ki_indices, ki_barriers_val, product.is_reverse
                )

        # Map KI trigger time
        ki_time = np.full(num_paths, np.inf, dtype=float)
        if ki_continuous:
            valid = first_ki_idx >= 0
            if valid.any():
                ki_time[valid] = all_times[first_ki_idx[valid]]
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
        pre_valid = pre_ko_triggered & (pre_ko_time < ki_time)

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
                time_mask = post_times[None, :] > ki_time[:, None]
                post_hit_filtered = post_hit & time_mask
                post_triggered = post_hit_filtered.any(axis=1)
                post_triggered &= post_allowed
                if post_triggered.any():
                    first_idx_local = np.argmax(post_hit_filtered[post_triggered], axis=1)
                    post_ko_triggered[post_triggered] = True
                    first_post_idx[post_triggered] = first_idx_local
                    post_ko_time[post_triggered] = post_times[first_idx_local]
            else:
                post_offsets = post_times
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
                        post_ko_time[group_idx] = (
                            ki_time[group_idx] + post_offsets[first_local]
                        )

        is_pre_ko = pre_valid
        is_post_ko = post_ko_triggered & ~is_pre_ko
        is_ko = is_pre_ko | is_post_ko
        is_v0 = ~is_ko & ~ki_triggered
        is_v1 = ~is_ko & ki_triggered

        payoffs = np.zeros(num_paths, dtype=float)
        settlement_times = np.zeros(num_paths, dtype=float)

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
                annualized_ko = product._effective_annualized_flag(
                    product.accrual_config.is_annualized_ko
                )
                if annualized_ko and product.initial_date is not None:
                    if pricing_env is None:
                        raise ValidationError(
                            "PricingEnvironment required to resolve KO accrual without observation_date."
                        )
                    if pricing_env.valuation_date < product.initial_date:
                        raise ValidationError(
                            "valuation_date must be on or after initial_date to resolve KO accrual."
                        )
                    if pricing_env.valuation_date == product.initial_date:
                        initial_to_valuation = 0.0
                    else:
                        initial_to_valuation = calculate_year_fraction(
                            product.initial_date,
                            pricing_env.valuation_date,
                            product.annualization_day_count,
                            pricing_env.bus_days_in_year,
                            calendar=getattr(pricing_env, "calendar", None),
                        )
                else:
                    initial_to_valuation = 0.0

                idx = np.flatnonzero(is_post_ko)
                local_idx = first_post_idx[idx]
                offsets = post_times[local_idx]
                rates = post_rates[local_idx]
                if annualized_ko:
                    accrual = initial_to_valuation + ki_time[idx] + offsets
                else:
                    accrual = 1.0
                coupon = (
                    product.initial_price
                    * product.contract_multiplier
                    * rates
                    * accrual
                )
                payoffs[idx] = principal_component + coupon
                if product.accrual_config.coupon_pay_type == CouponPayType.INSTANT:
                    determination_times = ki_time[idx] + offsets
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
                    determination_times = ki_time[idx] + post_max_offset
                    settlement_times[idx] = self._resolve_payment_time_array(
                        product,
                        pricing_env,
                        determination_times,
                        CashflowKind.TERMINAL,
                        cashflow_id="ko_reset_rebased_terminal",
                    )

        if is_v0.any():
            terminal_spots = paths[is_v0, -1]
            v0_payoffs = np.array(
                [
                    product.get_maturity_payoff_v0(spot, pricing_env)
                    for spot in terminal_spots
                ],
                dtype=float,
            )
            payoffs[is_v0] = v0_payoffs
            settlement_times[is_v0] = self._resolve_payment_time(
                product,
                pricing_env,
                float(pre_maturity),
                CashflowKind.TERMINAL,
                cashflow_id="ko_reset_v0_terminal",
            )

        if is_v1.any():
            terminal_spots = paths[is_v1, -1]
            v1_payoffs = np.array(
                [
                    product.get_maturity_payoff_v1(spot, pricing_env)
                    for spot in terminal_spots
                ],
                dtype=float,
            )
            payoffs[is_v1] = v1_payoffs
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

        extra = {
            "is_pre_ko": is_pre_ko,
            "is_post_ko": is_post_ko,
            "first_pre_idx": first_pre_idx,
            "first_post_idx": first_post_idx,
            "ki_triggered": ki_triggered,
            "ki_time": ki_time,
            "ko_time": ko_time,
        }

        return payoffs, settlement_times, stats, extra

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
        std_payoff = float(discounted_payoffs.std(ddof=1))
        std_error = std_payoff / math.sqrt(len(payoffs))

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
        std_payoff = float(discounted_payoffs.std(ddof=1))
        std_error = std_payoff / math.sqrt(len(payoffs)) if len(payoffs) > 0 else 0.0

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
            rng_seed=int(self.params.seed) + 1337 + int(batch_id) * 1000,
        )

        # Discount payoffs
        discount_factors = self._df(settlement_times)
        discounted_payoffs = payoffs * discount_factors

        discounted_payoffs = np.asarray(discounted_payoffs, dtype=float)
        n = int(discounted_payoffs.size)
        sum_x = float(discounted_payoffs.sum())
        sum_x2 = float(np.square(discounted_payoffs).sum())

        return {
            "n": n,
            "sum_x": sum_x,
            "sum_x2": sum_x2,
            "ko_count": int(stats.get("ko_count", 0)),
            "v0_count": int(stats.get("v0_count", 0)),
            "v1_count": int(stats.get("v1_count", 0)),
            "ko_time_sum": float(stats.get("ko_time_sum", 0.0)),
            "ko_time_count": int(stats.get("ko_time_count", 0)),
        }

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

        return {
            "n": n,
            "sum_x": sum_x,
            "sum_x2": sum_x2,
            "ko_count": int(stats.get("ko_count", 0)),
            "v0_count": int(stats.get("v0_count", 0)),
            "v1_count": int(stats.get("v1_count", 0)),
            "ko_time_sum": float(stats.get("ko_time_sum", 0.0)),
            "ko_time_count": int(stats.get("ko_time_count", 0)),
        }

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
            ).discounted_payoff

        def conditional_control_payoffs(paths, aux):
            return collapse_conditional_outer(
                raw_conditional_control_payoffs(paths, aux), aux
            )

        def pricer_fn(paths, aux):
            """Pricer function for RQMC driver."""
            moments = affine_moments(paths, aux)
            if moments is not None:
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

            discounted = native_discounted(paths)
            if aux is None or not aux.get("heston_conditional_control", False):
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
            sampled_control = native_discounted(aux["control_paths"])
            exact_control = raw_conditional_control_payoffs(paths, aux)
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
