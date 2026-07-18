"""
Monte Carlo pricing engine for Phoenix (autocallable) options.

This engine prices Phoenix options using Monte Carlo simulation with support for:
- Standard and reverse Phoenix structures
- Discrete KO observations with time-varying barriers and rates
- Discrete or continuous KI monitoring
- Periodic coupon payments with optional memory feature
- INSTANT or EXPIRY coupon payment timing
- Optional Dask parallelization for batch processing
"""

import math
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.engine.event_stats import PhoenixEventStats
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.process.bsm.qmc_path_generator import GBMPathGenerator
from quantark.montecarlo.qmc_rqmc_driver import RQMCRunSpec, run_rqmc
from quantark.asset.equity.process.bsm.qmc_sobol import (
    PseudoRandomNormalGenerator,
    SobolNormalGenerator,
)
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.engine.mc.term_inputs import build_mc_term_inputs, make_df_fn
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import CouponPayType, ObservationType
from quantark.util.enum.engine_enums import EngineType, MonteCarloMethod
from quantark.util.exceptions import PricingError, ValidationError
from quantark.util.numerical import is_zero, safe_log

try:
    from dask import compute, delayed

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False


@dataclass
class PhoenixMCResult:
    """Result container for Phoenix MC pricing."""

    price: float
    std_error: float
    num_paths: int
    ko_probability: float
    v0_probability: float
    v1_probability: float
    avg_ko_time: Optional[float] = None
    batches_used: Optional[int] = None
    coupon_probabilities: Optional[np.ndarray] = None
    expected_discounted_coupon_cashflow: Optional[np.ndarray] = None


class PhoenixMCEngine(BaseEngine):
    """
    Monte Carlo pricing engine for Phoenix options.

    Supports three Monte Carlo methods:
    - PSEUDO: Standard Monte Carlo with pseudorandom numbers
    - QUASI: Quasi-Monte Carlo with Sobol sequences
    - RANDOMIZED_QUASI: Randomized QMC with adaptive batching
    """

    # Request-scoped term-structure context (set at every public pricing
    # entry point from ITS pricing_env; never carried across calls).
    # Private helpers called outside a public entry fall back to their
    # scalar arguments (None ctx) or fail loudly (None _df) rather than
    # silently reusing a previous environment. Engine instances are not
    # safe for concurrent pricing calls (pre-existing engine contract).
    _term_ctx = None
    _df = None


    DEFAULT_METHOD = MonteCarloMethod.PSEUDO

    def __init__(
        self,
        params: Optional[MCParams] = None,
        method: Union[str, MonteCarloMethod, tuple, None] = None,
        use_dask: bool = False,
        num_batches: int = 4,
    ):
        if params is None:
            params = MCParams()

        if not isinstance(params, MCParams):
            raise ValidationError(
                f"params must be MCParams instance, got {type(params).__name__}"
            )

        super().__init__(params)

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

        self.use_dask = use_dask and DASK_AVAILABLE
        if use_dask and not DASK_AVAILABLE:
            warnings.warn(
                "Dask requested but not installed. Falling back to single-threaded NumPy.",
                UserWarning,
            )
        self.num_batches = num_batches

        self._last_result: Optional[PhoenixMCResult] = None

    def price(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        if not isinstance(product, PhoenixOption):
            raise PricingError(
                f"PhoenixMCEngine only supports PhoenixOption, got {type(product).__name__}"
            )

        S = pricing_env.spot
        T = product.get_maturity(pricing_env)
        r = pricing_env.get_rate(T)
        q = pricing_env.get_div_yield(T)
        sigma = pricing_env.get_vol(product.strike, T)

        self._validate_inputs(S, T, r, q, sigma, product)
        # Term-structure context for this pricing call (see term_inputs.py)
        self._term_ctx = (pricing_env, product.strike)
        self._df = make_df_fn(pricing_env)


        if T < 1e-10:
            return product.get_payoff(S, pricing_env=pricing_env)

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
        if result.price < 0 and product.payoff_config.include_principal:
            raise PricingError(f"Negative price computed: {result.price}")
        return result.price

    def _rqmc_streams_per_step(self) -> int:
        """Random streams consumed per time step in RQMC mode (session
        plan metadata; model-family subclasses override)."""
        return 1

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
        if not isinstance(product, PhoenixOption):
            raise PricingError(
                f"PhoenixMCEngine only supports PhoenixOption, got {type(product).__name__}"
            )
        S = pricing_env.spot
        T = product.get_maturity(pricing_env)
        r = pricing_env.get_rate(T)
        q = pricing_env.get_div_yield(T)
        sigma = pricing_env.get_vol(product.strike, T)
        self._validate_inputs(S, T, r, q, sigma, product)
        self._term_ctx = (pricing_env, product.strike)
        self._df = make_df_fn(pricing_env)
        if T < 1e-10:
            return None
        return self._rqmc_spec(product, pricing_env, S, T, r, q, sigma)

    def calculate_event_stats(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> Optional[PhoenixEventStats]:
        if not isinstance(product, PhoenixOption):
            return None

        S = pricing_env.spot
        T = product.get_maturity(pricing_env)
        r = pricing_env.get_rate(T)
        q = pricing_env.get_div_yield(T)
        sigma = pricing_env.get_vol(product.strike, T)
        # Term-structure context for this entry point (see term_inputs.py)
        self._term_ctx = (pricing_env, product.strike)
        self._df = make_df_fn(pricing_env)
        self._validate_inputs(S, T, r, q, sigma, product)

        all_times, dt_array, ko_indices, ki_indices = self._build_time_grid(
            product, pricing_env, T
        )
        ki_continuous = product.has_ki_barrier and (
            product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
            or product.barrier_config.ki_continuous
        )
        ki_event_times = np.array([], dtype=float)
        if product.has_ki_barrier:
            if ki_continuous:
                ki_event_times = np.array(all_times, dtype=float)
            else:
                ki_profile = product.get_ki_observation_profile(pricing_env)
                ki_event_times = np.array(
                    ki_profile["observation_times"], dtype=float
                )
        generator = self._create_path_generator(S, r, q, sigma, T, dt_array)
        paths, _ = generator.generate_paths(return_aux=False)

        (
            payoffs,
            settlement_times,
            stats,
            coupon_probabilities,
            coupon_cashflows,
            instant_coupon_discounted,
            ko_times,
            ko_payoffs,
            ko_settlement_times,
        ) = self._compute_payoffs(
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
            collect_coupon_stats=True,
        )

        discount_factors = self._df(settlement_times)
        discounted_payoffs = payoffs * discount_factors + instant_coupon_discounted
        pv = float(discounted_payoffs.mean())

        ko_probability = np.zeros(len(ko_times), dtype=float)
        expected_discounted_ko_cashflow = np.zeros(len(ko_times), dtype=float)
        for i in range(len(ko_times)):
            hit_i = stats["is_ko"] & (stats["first_ko_idx"] == i)
            p_i = float(np.mean(hit_i))
            ko_probability[i] = p_i
            if p_i > 0.0:
                df = self._df(float(ko_settlement_times[i]))
                expected_discounted_ko_cashflow[i] = p_i * float(ko_payoffs[i]) * df

        survival_probability = np.ones(len(ko_times), dtype=float)
        cumulative_ko = 0.0
        for i in range(len(ko_times)):
            cumulative_ko += ko_probability[i]
            survival_probability[i] = max(0.0, 1.0 - cumulative_ko)

        maturity_df = self._df(float(T))
        maturity_payoff_all = np.zeros(len(paths), dtype=float)
        is_ko = stats["is_ko"]
        is_v0 = stats["is_v0"]
        is_v1 = stats["is_v1"]
        if product.coupon_config.coupon_pay_type == CouponPayType.EXPIRY:
            maturity_payoff_all[~is_ko] = payoffs[~is_ko]
        else:
            if is_v0.any():
                maturity_payoff_all[is_v0] = np.array(
                    [
                        product.get_maturity_payoff_v0(
                            float(s),
                            pricing_env=pricing_env,
                        )
                        for s in paths[is_v0, -1]
                    ],
                    dtype=float,
                )
            if is_v1.any():
                maturity_payoff_all[is_v1] = np.array(
                    [
                        product.get_maturity_payoff_v1(
                            float(s),
                            pricing_env=pricing_env,
                        )
                        for s in paths[is_v1, -1]
                    ],
                    dtype=float,
                )
        expected_discounted_maturity_cashflow = float(
            np.mean(maturity_payoff_all * maturity_df)
        )

        coupon_cf_total = float(np.sum(coupon_cashflows))
        if product.coupon_config.coupon_pay_type == CouponPayType.EXPIRY:
            coupon_cf_total = 0.0
        pv_cashflows = float(
            np.sum(expected_discounted_ko_cashflow)
            + expected_discounted_maturity_cashflow
            + coupon_cf_total
        )
        reconciliation_error = pv - pv_cashflows
        ki_event_probability = np.array([], dtype=float)
        ki_survival_probability = np.array([], dtype=float)
        if product.has_ki_barrier:
            if bool(getattr(product, "_otc_lifecycle_knocked_in", False)):
                ki_event_times = np.array([0.0], dtype=float)
                ki_event_probability = np.array([1.0], dtype=float)
                ki_survival_probability = np.array([0.0], dtype=float)
            elif ki_event_times.size and "first_ki_idx" in stats:
                first_ki_idx = stats["first_ki_idx"]
                ki_triggered = stats["ki_triggered"]
                ki_event_probability = np.zeros(len(ki_event_times), dtype=float)
                for i in range(len(ki_event_times)):
                    ki_event_probability[i] = float(
                        np.mean(ki_triggered & (first_ki_idx == i))
                    )
                ki_survival_probability = np.maximum(
                    0.0, 1.0 - np.cumsum(ki_event_probability)
                )

        # Unambiguous, cross-engine-consistent KI definitions (investigation
        # 2026-07-01). `ki_ever` counts any path that touches the KI barrier;
        # `ki_survive` restricts to paths that also reach maturity without KO
        # (i.e. the note settles knocked-in). The legacy `ki_probability` keeps
        # MC's historical "KI ever" meaning.
        if product.has_ki_barrier:
            ki_triggered = np.asarray(stats["ki_triggered"], dtype=bool)
            ki_ever_probability = float(np.mean(ki_triggered))
            ki_survive_knocked_in_probability = float(
                np.mean(ki_triggered & ~is_ko)
            )
        else:
            ki_ever_probability = 0.0
            ki_survive_knocked_in_probability = 0.0

        return PhoenixEventStats(
            pv=pv,
            ko_times=np.array(ko_times, dtype=float),
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
            coupon_probability=coupon_probabilities,
            expected_discounted_coupon_cashflow=coupon_cashflows,
        )

    def _validate_inputs(
        self,
        S: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        product: PhoenixOption,
    ) -> None:
        if S <= 0:
            raise ValidationError(f"Spot price must be positive, got {S}")
        if T < 0:
            raise ValidationError(f"Time to maturity must be non-negative, got {T}")
        if sigma <= 0:
            raise ValidationError(f"Volatility must be positive, got {sigma}")
        if not np.isfinite(q):
            raise ValidationError(f"Dividend yield must be finite, got {q}")

        if product.barrier_config.ko_observation_type == ObservationType.DISCRETE:
            if (
                product.barrier_config.ko_observation_schedule is None
                and product.barrier_config.ko_observation_dates is None
            ):
                raise ValidationError(
                    "KO observation schedule or dates required for discrete monitoring"
                )

    def _build_time_grid(
        self, product: PhoenixOption, pricing_env: PricingEnvironment, T: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ko_profile = product.get_ko_observation_profile(pricing_env)
        ko_times = np.array(ko_profile["observation_times"], dtype=float)

        ki_times = np.array([], dtype=float)
        ki_continuous = (
            product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
            or product.barrier_config.ki_continuous
        )

        if product.has_ki_barrier:
            if ki_continuous:
                num_ki_steps = int(pricing_env.bus_days_in_year * T) + 1
                ki_times = np.linspace(0, T, num_ki_steps + 1)[1:]
            else:
                ki_profile = product.get_ki_observation_profile(pricing_env)
                ki_times = np.array(ki_profile["observation_times"], dtype=float)

        ko_grid_times = [t for t in ko_times if t > 0 and not is_zero(t)]
        ki_grid_times = [t for t in ki_times if t > 0 and not is_zero(t)]

        all_times_set = set(ko_grid_times) | set(ki_grid_times) | {T}
        all_times = np.array(sorted(all_times_set), dtype=float)

        times_with_zero = np.concatenate([[0.0], all_times])
        dt_array = np.diff(times_with_zero)

        def path_index_for_time(time_val: float) -> int:
            if time_val <= 0.0 or is_zero(time_val):
                return 0
            return int(np.searchsorted(all_times, time_val)) + 1

        ko_indices = np.array([path_index_for_time(t) for t in ko_times], dtype=int)
        if not ki_continuous and ki_times.size > 0:
            ki_indices = np.array(
                [path_index_for_time(t) for t in ki_times], dtype=int
            )
        else:
            ki_indices = np.array([], dtype=int)

        return all_times, dt_array, ko_indices, ki_indices

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
            vr_config=None,
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
        ko_prices = paths[:, ko_indices]

        if is_reverse:
            ko_hit = ko_prices <= ko_barriers
        else:
            ko_hit = ko_prices >= ko_barriers

        ko_triggered = ko_hit.any(axis=1)
        first_ko_idx = np.full(len(paths), -1, dtype=int)

        if ko_triggered.any():
            first_ko_idx[ko_triggered] = np.argmax(ko_hit[ko_triggered], axis=1)

        return ko_triggered, first_ko_idx

    def _check_ki_barriers(
        self,
        paths: np.ndarray,
        ki_indices: np.ndarray,
        ki_barriers: Union[float, np.ndarray],
        is_reverse: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(ki_indices) == 0:
            return np.zeros(len(paths), dtype=bool), np.full(len(paths), -1, dtype=int)

        ki_prices = paths[:, ki_indices]
        num_ki_obs_times = ki_prices.shape[1]

        ki_barriers_effective = np.array(ki_barriers)

        if ki_barriers_effective.shape == () or ki_barriers_effective.shape == (1,):
            ki_barriers_aligned = np.full(
                num_ki_obs_times, ki_barriers_effective.item()
            )
        else:
            if ki_barriers_effective.shape[0] != num_ki_obs_times:
                raise ValidationError(
                    f"ki_barriers array (shape {ki_barriers_effective.shape[0]}) "
                    f"does not match number of KI observation times ({num_ki_obs_times})"
                )
            ki_barriers_aligned = ki_barriers_effective

        if is_reverse:
            ki_hit = ki_prices >= ki_barriers_aligned
        else:
            ki_hit = ki_prices <= ki_barriers_aligned

        ki_triggered = ki_hit.any(axis=1)
        first_ki_idx = np.full(len(paths), -1, dtype=int)

        if ki_triggered.any():
            first_ki_idx[ki_triggered] = np.argmax(ki_hit[ki_triggered], axis=1)

        return ki_triggered, first_ki_idx

    def _check_ki_barriers_continuous_with_bridge(
        self,
        paths: np.ndarray,
        all_times: np.ndarray,
        ki_barrier: float,
        sigma: float,
        is_reverse: bool,
        rng_seed: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if ki_barrier <= 0:
            raise ValidationError(f"ki_barrier must be positive, got {ki_barrier}")
        if sigma <= 0:
            raise ValidationError(f"volatility must be positive, got {sigma}")

        n_paths = len(paths)
        if n_paths == 0:
            return np.zeros(0, dtype=bool), np.zeros(0, dtype=int)

        ki_triggered = np.zeros(n_paths, dtype=bool)
        first_ki_idx = np.full(n_paths, -1, dtype=int)

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

        for k in range(n_steps):
            active = ~ki_triggered
            if not active.any():
                break

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
                non_breached = (s0 < ki_barrier) & (s1 < ki_barrier)
            else:
                non_breached = (s0 > ki_barrier) & (s1 > ki_barrier)

            bridge_candidates = active & non_breached
            if not bridge_candidates.any():
                continue

            idx = np.flatnonzero(bridge_candidates)
            dt_k = float(dt[k])
            h2 = float(sigma * sigma) * dt_k

            log_term = safe_log(s0[idx] / ki_barrier) * safe_log(s1[idx] / ki_barrier)
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

    @staticmethod
    def _coupon_trigger_mask(
        product: PhoenixOption, spots: np.ndarray, obs_idx: int
    ) -> np.ndarray:
        """Vectorized coupon trigger check using product logic."""
        return np.array(
            [product.is_coupon_triggered(float(s), obs_idx) for s in spots],
            dtype=bool,
        )

    def _compute_payoffs(
        self,
        product: PhoenixOption,
        pricing_env: PricingEnvironment,
        paths: np.ndarray,
        all_times: np.ndarray,
        ko_indices: np.ndarray,
        ki_indices: np.ndarray,
        r: float,
        T: float,
        sigma: float,
        rng_seed: int,
        collect_coupon_stats: bool = False,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        Dict[str, np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        num_paths = len(paths)

        ko_profile = product.get_ko_observation_profile(pricing_env)
        ko_times = np.array(ko_profile["observation_times"], dtype=float)
        ko_barriers = np.array(ko_profile["barriers"], dtype=float)
        ko_payoffs_schedule = np.array(ko_profile["payoffs"], dtype=float)
        ko_settlement_times = np.array(ko_profile["settlement_times"], dtype=float)

        num_obs = len(ko_times)
        coupon_barrier = product.coupon_config.coupon_barrier
        if isinstance(coupon_barrier, list):
            if len(coupon_barrier) != num_obs:
                raise ValidationError(
                    "Coupon barrier schedule length does not match KO observations."
                )
            coupon_barriers = np.array(coupon_barrier, dtype=float)
        else:
            coupon_barriers = np.full(num_obs, float(coupon_barrier))

        period_year_fractions = np.array(
            product.get_coupon_period_year_fractions(ko_times.tolist()),
            dtype=float,
        )
        coupon_amounts = np.array(
            [
                product.get_coupon_payoff(i, year_fraction=period_year_fractions[i])
                for i in range(num_obs)
            ],
            dtype=float,
        )

        ko_triggered, first_ko_idx = self._check_ko_barriers(
            paths, ko_indices, ko_barriers, product.is_reverse
        )

        ki_triggered = np.zeros(num_paths, dtype=bool)
        first_ki_idx = np.full(num_paths, -1, dtype=int)
        ki_times = np.array([], dtype=float)
        if product.has_ki_barrier:
            ki_continuous = (
                product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
                or product.barrier_config.ki_continuous
            )
            if ki_continuous:
                ki_barrier_val = product.barrier_config.ki_barrier
                if isinstance(ki_barrier_val, list):
                    raise ValidationError(
                        "Continuous KI monitoring requires a scalar ki_barrier."
                    )
                ki_barrier_scalar = float(ki_barrier_val)
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
                ki_profile = product.get_ki_observation_profile(pricing_env)
                ki_times = np.array(ki_profile["observation_times"], dtype=float)
                ki_barriers_val = np.array(ki_profile["barriers"], dtype=float)
                ki_triggered, first_ki_idx = self._check_ki_barriers(
                    paths, ki_indices, ki_barriers_val, product.is_reverse
                )

        already_knocked_in = bool(getattr(product, "_otc_lifecycle_knocked_in", False))
        if already_knocked_in and product.has_ki_barrier:
            ki_triggered[:] = True
            first_ki_idx[:] = 0

        if product.barrier_config.disable_ko_after_ki and product.has_ki_barrier:
            if already_knocked_in:
                ko_valid = np.zeros_like(ko_triggered)
            else:
                ko_trigger_times = np.where(
                    first_ko_idx >= 0, ko_times[first_ko_idx], np.inf
                )
                if product.has_ki_barrier:
                    if ki_continuous:
                        ki_obs_times = all_times
                    else:
                        ki_obs_times = ki_times
                    if ki_obs_times.size > 0:
                        ki_trigger_times = np.where(
                            first_ki_idx >= 0, ki_obs_times[first_ki_idx], np.inf
                        )
                    else:
                        ki_trigger_times = np.full(num_paths, np.inf, dtype=float)
                else:
                    ki_trigger_times = np.full(num_paths, np.inf, dtype=float)
                ko_valid = ko_triggered & (ko_trigger_times < ki_trigger_times)
        else:
            ko_valid = ko_triggered

        is_ko = ko_valid
        is_v0 = ~is_ko & ~ki_triggered
        is_v1 = ~is_ko & ki_triggered

        payoffs = np.zeros(num_paths, dtype=float)
        settlement_times = np.full(num_paths, T, dtype=float)
        instant_coupon_discounted = np.zeros(num_paths, dtype=float)

        coupon_probabilities = np.zeros(num_obs, dtype=float)
        coupon_cashflows = np.zeros(num_obs, dtype=float)

        accrued = np.zeros(num_paths, dtype=float)
        expiry_coupon = np.zeros(num_paths, dtype=float)

        for obs_idx in range(num_obs):
            spot_obs = paths[:, ko_indices[obs_idx]]
            ko_at_obs = is_ko & (first_ko_idx == obs_idx)
            alive_before = (~is_ko) | (first_ko_idx >= obs_idx)

            coupon_hit = self._coupon_trigger_mask(product, spot_obs, obs_idx)
            coupon_hit = coupon_hit & alive_before

            current_coupon = coupon_amounts[obs_idx]
            if product.coupon_config.memory_coupon:
                coupon_to_pay = current_coupon + accrued
            else:
                coupon_to_pay = np.full(num_paths, current_coupon, dtype=float)

            if collect_coupon_stats:
                coupon_probabilities[obs_idx] = float(np.mean(coupon_hit))
                if product.coupon_config.coupon_pay_type == CouponPayType.INSTANT:
                    df_obs = self._df(float(ko_times[obs_idx]))
                    coupon_cashflows[obs_idx] = float(
                        np.mean(coupon_to_pay * coupon_hit) * df_obs
                    )
                else:
                    df_T = self._df(float(T))
                    coupon_cashflows[obs_idx] = float(
                        np.mean(coupon_to_pay * coupon_hit) * df_T
                    )

            non_ko_coupon = coupon_hit & ~ko_at_obs
            if non_ko_coupon.any():
                if product.coupon_config.coupon_pay_type == CouponPayType.INSTANT:
                    df_obs = self._df(float(ko_times[obs_idx]))
                    instant_coupon_discounted[non_ko_coupon] += (
                        coupon_to_pay[non_ko_coupon] * df_obs
                    )
                else:
                    expiry_coupon[non_ko_coupon] += coupon_to_pay[non_ko_coupon]

                if product.coupon_config.memory_coupon:
                    accrued[non_ko_coupon] = 0.0

            if product.coupon_config.memory_coupon:
                missed = alive_before & ~coupon_hit & ~ko_at_obs
                if missed.any():
                    accrued[missed] += current_coupon

            if ko_at_obs.any():
                ko_coupon_hit = self._coupon_trigger_mask(product, spot_obs, obs_idx)

                current_coupon_pay = current_coupon
                if product.coupon_config.memory_coupon:
                    current_coupon_pay = current_coupon + accrued
                ko_coupon = np.where(ko_coupon_hit, current_coupon_pay, 0.0)

                payoffs[ko_at_obs] = ko_payoffs_schedule[obs_idx] + ko_coupon[ko_at_obs]

                if product.coupon_config.coupon_pay_type == CouponPayType.INSTANT:
                    settlement_times[ko_at_obs] = ko_settlement_times[obs_idx]

                if product.coupon_config.memory_coupon:
                    accrued[ko_at_obs] = 0.0

        if (~is_ko).any():
            maturity_spots = paths[~is_ko, -1]
            maturity_payoff = np.zeros(maturity_spots.size, dtype=float)
            if is_v0.any():
                maturity_payoff[is_v0[~is_ko]] = np.array(
                    [
                        product.get_maturity_payoff_v0(
                            float(s),
                            pricing_env=pricing_env,
                        )
                        for s in maturity_spots[is_v0[~is_ko]]
                    ],
                    dtype=float,
                )
            if is_v1.any():
                maturity_payoff[is_v1[~is_ko]] = np.array(
                    [
                        product.get_maturity_payoff_v1(
                            float(s),
                            pricing_env=pricing_env,
                        )
                        for s in maturity_spots[is_v1[~is_ko]]
                    ],
                    dtype=float,
                )
            if product.coupon_config.coupon_pay_type == CouponPayType.EXPIRY:
                maturity_payoff += expiry_coupon[~is_ko]
            payoffs[~is_ko] += maturity_payoff

        stats = {
            "ko_probability": float(is_ko.mean()),
            "v0_probability": float(is_v0.mean()),
            "v1_probability": float(is_v1.mean()),
            "ko_count": int(is_ko.sum()),
            "v0_count": int(is_v0.sum()),
            "v1_count": int(is_v1.sum()),
            "avg_ko_time": None,
            "ko_time_sum": 0.0,
            "ko_time_count": 0,
            "is_ko": is_ko,
            "is_v0": is_v0,
            "is_v1": is_v1,
            "first_ko_idx": first_ko_idx,
            "ki_triggered": ki_triggered,
            "first_ki_idx": first_ki_idx,
        }

        if is_ko.any():
            ko_times_hit = ko_times[first_ko_idx[is_ko]]
            stats["avg_ko_time"] = float(ko_times_hit.mean())
            stats["ko_time_sum"] = float(ko_times_hit.sum())
            stats["ko_time_count"] = int(is_ko.sum())

        return (
            payoffs,
            settlement_times,
            stats,
            coupon_probabilities,
            coupon_cashflows,
            instant_coupon_discounted,
            ko_times,
            ko_payoffs_schedule,
            ko_settlement_times,
        )

    def _price_mc_or_qmc(
        self,
        product: PhoenixOption,
        pricing_env: PricingEnvironment,
        S: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
    ) -> PhoenixMCResult:
        all_times, dt_array, ko_indices, ki_indices = self._build_time_grid(
            product, pricing_env, T
        )
        generator = self._create_path_generator(S, r, q, sigma, T, dt_array)
        paths, _ = generator.generate_paths(return_aux=False)

        (
            payoffs,
            settlement_times,
            stats,
            coupon_probabilities,
            coupon_cashflows,
            instant_coupon_discounted,
            _,
            _,
            _,
        ) = self._compute_payoffs(
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
            collect_coupon_stats=True,
        )

        discount_factors = self._df(settlement_times)
        discounted_payoffs = payoffs * discount_factors + instant_coupon_discounted

        price = float(discounted_payoffs.mean())
        std_payoff = float(discounted_payoffs.std(ddof=1))
        std_error = std_payoff / math.sqrt(len(payoffs))

        return PhoenixMCResult(
            price=price,
            std_error=std_error,
            num_paths=len(paths),
            ko_probability=stats["ko_probability"],
            v0_probability=stats["v0_probability"],
            v1_probability=stats["v1_probability"],
            avg_ko_time=stats.get("avg_ko_time"),
            coupon_probabilities=coupon_probabilities,
            expected_discounted_coupon_cashflow=coupon_cashflows,
        )

    def _price_single_batch(
        self,
        batch_id: int,
        batch_num_paths: int,
        product: PhoenixOption,
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
        generator = self._create_path_generator(
            S, r, q, sigma, T, dt_array, batch_id=batch_id, num_paths=batch_num_paths
        )

        paths, _ = generator.generate_paths(return_aux=False, batch_id=batch_id)

        (
            payoffs,
            settlement_times,
            stats,
            _,
            _,
            instant_coupon_discounted,
            _,
            _,
            _,
        ) = self._compute_payoffs(
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

        discount_factors = self._df(settlement_times)
        discounted_payoffs = payoffs * discount_factors + instant_coupon_discounted

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
        product: PhoenixOption,
        pricing_env: PricingEnvironment,
        S: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
    ) -> PhoenixMCResult:
        all_times, dt_array, ko_indices, ki_indices = self._build_time_grid(
            product, pricing_env, T
        )

        if self.num_batches <= 0:
            raise ValidationError(
                f"num_batches must be positive, got {self.num_batches}"
            )

        total_paths_target = int(self.params.num_paths)
        base = total_paths_target // self.num_batches
        remainder = total_paths_target % self.num_batches
        batch_sizes = [
            (base + 1 if i < remainder else base) for i in range(self.num_batches)
        ]

        batch_results = []
        batches_used = 0
        for batch_id, batch_num_paths in enumerate(batch_sizes):
            if batch_num_paths <= 0:
                continue
            batches_used += 1
            batch_results.append(
                delayed(self._price_single_batch)(
                    batch_id=batch_id,
                    batch_num_paths=batch_num_paths,
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
                )
            )

        results = compute(*batch_results)

        total_n = 0
        total_sum_x = 0.0
        total_sum_x2 = 0.0
        total_ko_count = 0
        total_v0_count = 0
        total_v1_count = 0
        total_ko_time_sum = 0.0
        total_ko_time_count = 0

        for res in results:
            total_n += int(res["n"])
            total_sum_x += float(res["sum_x"])
            total_sum_x2 += float(res["sum_x2"])
            total_ko_count += int(res.get("ko_count", 0))
            total_v0_count += int(res.get("v0_count", 0))
            total_v1_count += int(res.get("v1_count", 0))
            total_ko_time_sum += float(res.get("ko_time_sum", 0.0))
            total_ko_time_count += int(res.get("ko_time_count", 0))

        if total_n <= 0:
            raise PricingError("Dask parallel pricing produced zero simulated paths")

        price = total_sum_x / total_n

        if total_n > 1:
            sample_var = (total_sum_x2 - (total_sum_x * total_sum_x) / total_n) / (
                total_n - 1
            )
            sample_var = max(sample_var, 0.0)
            std_error = math.sqrt(sample_var) / math.sqrt(total_n)
        else:
            std_error = 0.0

        ko_probability = float(total_ko_count / total_n)
        v0_probability = float(total_v0_count / total_n)
        v1_probability = float(total_v1_count / total_n)

        if total_ko_time_count > 0:
            avg_ko_time = float(total_ko_time_sum / total_ko_time_count)
        else:
            avg_ko_time = None

        return PhoenixMCResult(
            price=float(price),
            std_error=float(std_error),
            num_paths=int(total_n),
            ko_probability=ko_probability,
            v0_probability=v0_probability,
            v1_probability=v1_probability,
            avg_ko_time=avg_ko_time,
            batches_used=batches_used,
        )

    def _rqmc_spec(
        self,
        product: PhoenixOption,
        pricing_env: PricingEnvironment,
        S: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
    ) -> RQMCRunSpec:
        """Build the RQMC run description for a PhoenixOption.

        Shared by the direct path (_price_rqmc) and the execution-framework
        session path (build_rqmc_session_spec): ONE construction of the
        grid, pricer, generator, controls, and result assembly.
        """
        all_times, dt_array, ko_indices, ki_indices = self._build_time_grid(
            product, pricing_env, T
        )
        generator = self._create_path_generator(S, r, q, sigma, T, dt_array)

        def pricer_fn(paths, aux):
            batch_id = 0
            if aux is not None and "batch_id" in aux:
                batch_id = int(aux["batch_id"])
            (
                payoffs,
                settlement_times,
                _,
                _,
                _,
                instant_coupon_discounted,
                _,
                _,
                _,
            ) = self._compute_payoffs(
                product,
                pricing_env,
                paths,
                all_times,
                ko_indices,
                ki_indices,
                r,
                T,
                sigma,
                rng_seed=int(self.params.seed) + 1337 + batch_id * 1000,
            )
            discount_factors = self._df(settlement_times)
            return payoffs * discount_factors + instant_coupon_discounted

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
            S, r, q, sigma, T, dt_array, num_paths=per_batch_paths
        )

        def finalize(result):
            paths, _ = generator.generate_paths(return_aux=False, batch_id=0)
            _, _, stats, _, _, _, _, _, _ = self._compute_payoffs(
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

            return PhoenixMCResult(
                price=result.price,
                std_error=result.std_error,
                num_paths=result.total_paths,
                ko_probability=stats["ko_probability"],
                v0_probability=stats["v0_probability"],
                v1_probability=stats["v1_probability"],
                batches_used=result.batches_used,
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
            dimension=self._rqmc_streams_per_step() * int(dt_array.size),
        )

    def _price_rqmc(
        self,
        product: PhoenixOption,
        pricing_env: PricingEnvironment,
        S: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
    ) -> PhoenixMCResult:
        spec = self._rqmc_spec(product, pricing_env, S, T, r, q, sigma)
        return spec.finalize(run_rqmc(
            pricer_fn=spec.pricer_fn,
            path_generator=spec.path_generator,
            max_batches=spec.max_batches,
            target_std=spec.target_std,
            min_batches=spec.min_batches,
        ))

    def get_last_result(self) -> Optional[PhoenixMCResult]:
        return self._last_result

    def get_last_std_error(self) -> Optional[float]:
        if self._last_result is None:
            return None
        return self._last_result.std_error

    def __repr__(self):
        dask_str = f", use_dask={self.use_dask}" if self.use_dask else ""
        return f"PhoenixMCEngine(method={self.method.name}{dask_str})"
