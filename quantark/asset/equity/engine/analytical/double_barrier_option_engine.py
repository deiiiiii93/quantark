"""
Analytical pricing engine for double-barrier options.

Implements the Ikeda & Kuintomo (1992) infinite-series formula for double
knock-out options with continuous monitoring. Supports:
- Continuous observation (direct closed-form)
- Discrete observation (barrier-shift approximation)
- Expiry-only observation (truncated-domain vanilla payoff)

Knock-in options are priced via parity: knock-in = vanilla - knock-out.
"""

import math
from typing import Optional

from scipy import stats

from quantark.asset.equity.engine.barrier_schedule_rules import (
    forward_double_barrier_levels,
    forward_knock_out_cash,
    refuse_knock_in_record_payoff,
    refuse_mixed_forward_levels,
    valuation_double_barrier_terms,
)
from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.engine.capabilities import SettlementSupport
from quantark.asset.equity.engine.settlement_support import (
    pending_receivable_pv,
    resolve_terminal_timing,
    terminal_lifecycle_pv,
    validate_settlement_capability,
)
from quantark.asset.equity.engine.monitoring_ended import (
    knock_in_carried,
    remaining_contract_applies,
    remaining_replication,
)
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option import DoubleBarrierOption, EuropeanVanillaOption
from quantark.asset.equity.param import EngineParams
from quantark.priceenv import PricingEnvironment
from quantark.util.barrier_shift import apply_barrier_shift
from quantark.util.enum import ObservationType
from quantark.util.enum.engine_enums import DiscreteBarrierApproximation, EngineType
from quantark.util.exceptions import ValidationError, PricingError
from quantark.util.numerical import (
    is_zero,
    safe_log,
    safe_exp,
    safe_sqrt,
    safe_power,
    validate_positive,
    validate_non_negative,
)

from .black_scholes_engine import BlackScholesEngine
from .double_no_touch import double_no_touch
from .first_observation import (
    first_future_observation,
    observed_at_valuation,
    projects_through_first_observation,
    resolve_current_records,
    through_first_observation,
    validate_discrete_barrier_approximation,
)


class DoubleBarrierOptionAnalyticalEngine(BaseEngine):
    """
    Closed-form pricing engine for double-barrier options.

    Uses the Ikeda & Kuintomo (1992) infinite-series solution for
    continuous monitoring. Discrete monitoring is approximated via
    Broadie-Glasserman-Kou barrier shift; only a record at the valuation
    instant decides now, and between observations
    ``discrete_barrier_approximation`` decides the value (``BGK``: the shifted
    closed form at today's spot inside or outside the corridor;
    ``FIRST_OBSERVATION_BGK``: through the next observation). Expiry
    observation prices the truncated-domain vanilla payoff.

    Edge Cases Handled:
    -------------------
    1. T = 0: Returns intrinsic payoff or rebate immediately.
    2. sigma = 0: Returns discounted deterministic payoff.
    3. Spot outside barriers (KO), continuously or at a record on the valuation instant: Returns rebate
       immediately.
    4. Spot outside barriers (KI), likewise: Returns vanilla price (already knocked in).
    4b. DISCRETE, ``BGK``, spot outside the SHIFTED corridor with a rebate: ``ValidationError``
        (``TODO(bgk-beyond)``): the double no-touch probability is defined only inside its corridor.
    5. Strike outside [L, U]: Raises ValidationError (Ikeda-Kuintomo limitation).
    """

    engine_type = EngineType.ANALYTICAL
    # params-only engine: instance state == construction arguments (patch spec 2026-09-03 §7)
    MODEL_FINGERPRINT_ATTRS = ('params', 'discrete_barrier_approximation', '_bs_engine')
    settlement_support = SettlementSupport.EVENT_AND_TERMINAL
    supports_lifecycle_state = True

    MIN_VOL = 1e-12
    MAX_VOL = 5.0
    MAX_MATURITY = 100.0
    DEFAULT_MAX_TERMS = 10
    SERIES_TOLERANCE = 1e-15

    def __init__(
        self,
        params: Optional[EngineParams] = None,
        *,
        discrete_barrier_approximation: DiscreteBarrierApproximation = DiscreteBarrierApproximation.BGK,
    ):
        """
        Initialize the double-barrier analytical engine.

        Args:
            params: Engine configuration parameters.
            discrete_barrier_approximation: The value of a DISCRETE schedule between observations (see
                ``DiscreteBarrierApproximation``).
        """
        super().__init__(params)
        self.discrete_barrier_approximation = validate_discrete_barrier_approximation(discrete_barrier_approximation)
        self._bs_engine = BlackScholesEngine(params)

    def price(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        *,
        lifecycle_state=None,
    ) -> float:
        validate_settlement_capability(self, product, lifecycle_state)
        refuse_knock_in_record_payoff(product)      # one rebate per knock-in, as on every route (W18 Q3, R19)
        fixed_pv = terminal_lifecycle_pv(lifecycle_state, pricing_env)
        if fixed_pv is not None:
            return fixed_pv
        if remaining_contract_applies(product, lifecycle_state) and product.get_maturity(pricing_env) > 0.0:
            # no observation left (spec 2026-10-07 theta boundary 3.3): the remaining contract, by replication
            return (remaining_replication(product, knock_in_carried(product, lifecycle_state)).price(pricing_env)
                    + pending_receivable_pv(lifecycle_state, pricing_env))
        timing = resolve_terminal_timing(product, pricing_env)
        return (
            self._price_immediate(product, pricing_env, knocked_in=knock_in_carried(product, lifecycle_state))
            * timing.delay_df
            + pending_receivable_pv(lifecycle_state, pricing_env)
        )

    def _price_immediate(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment, knocked_in: bool = False
    ) -> float:
        """
        Price a double-barrier option.

        Args:
            product: DoubleBarrierOption to price.
            pricing_env: Pricing environment with market data.
            knocked_in: A knock-in carried as triggered (read at zero maturity; before it the remaining-contract
                replication in ``price`` prices the carried state).

        Returns:
            Option price.

        Raises:
            PricingError: If product is not a DoubleBarrierOption.
            ValidationError: If inputs are invalid.
            NumericalError: If numerical issues occur during calculation.
        """
        if not isinstance(product, DoubleBarrierOption):
            raise PricingError(
                f"DoubleBarrierOptionAnalyticalEngine only supports DoubleBarrierOption, "
                f"got {type(product).__name__}"
            )

        S = pricing_env.spot
        K = product.strike
        T = product.get_maturity(pricing_env)
        r = pricing_env.get_rate(T)
        q = pricing_env.get_div_yield(T)
        sigma = pricing_env.get_vol(K, T)
        multiplier = product.contract_multiplier
        obs_type = product.observation_type
        discrete = obs_type == ObservationType.DISCRETE
        if discrete:
            # W18c (spec 2026-10-13 I4, A8): a DISCRETE schedule is breached now only by a record at the valuation
            # instant, at its own levels; forward, the closed form monitors the corridor every remaining record
            # shares (None: the levels differ -- no closed form, refused before any forward value)
            hit_now = self._breached_at_valuation(product, pricing_env, S)
            forward = forward_double_barrier_levels(product, pricing_env)
        else:
            hit_now = product.is_barrier_hit(S)
            forward = (product.upper_barrier, product.lower_barrier)
        # mixed remaining levels price no forward value: the product's corridor then feeds only the input checks
        U, L = forward if forward is not None else (product.upper_barrier, product.lower_barrier)

        self._validate_inputs(S, K, T, r, q, sigma, L, U)
        # the cash of a future hit: a DISCRETE knock-out pays the hit record's payoff (W18c follow-up); a knock-in's
        # rebate is paid when the corridor is never left
        rebate = forward_knock_out_cash(product, pricing_env) if discrete and product.is_knock_out else product.rebate

        # Edge case: zero maturity
        if is_zero(T):
            return self._price_zero_maturity(product, pricing_env, S, multiplier, knocked_in=knocked_in)

        if discrete and forward is None and (not hit_now or is_zero(sigma)):
            # a forward value needs one corridor; so does the zero-volatility shortcut, which reads none of the records
            refuse_mixed_forward_levels(product.observation_schedule, pricing_env, default_payoff=product.rebate)

        # Edge case: zero volatility
        if is_zero(sigma):
            return self._price_zero_vol(product, S, T, r, q, multiplier, L, U, rebate)

        # EXPIRY: nothing is observed before maturity, and the closed form holds from any spot (spec 3C §5.8)
        if obs_type == ObservationType.EXPIRY:
            return self._price_expiry(product, pricing_env, S, K, T, r, q, sigma, L, U, multiplier)

        # No observation due now (A15): FIRST_OBSERVATION_BGK values the contract through the next observation
        if discrete and not hit_now and projects_through_first_observation(
            self.discrete_barrier_approximation, S >= U or S <= L, lambda: self._resolved_records(product, pricing_env)
        ):
            return self._price_beyond_discrete(product, pricing_env, S, K, T, r, q, sigma, L, U, multiplier, rebate)

        # Spot already outside barriers: continuously, or at an observation at the valuation instant
        if hit_now:
            if product.is_knock_out:
                # the deciding record's payoff on a DISCRETE schedule (W18c), the rebate otherwise -- paid at the
                # terminal payment time as every forward leg pays it (T17 concern 3): DF(T) here, the delay to the
                # payment date in ``price``
                cash = valuation_double_barrier_terms(product, pricing_env)[2] if discrete else product.rebate
                return cash * float(safe_exp(-r * T)) * multiplier
            # Knock-in: already activated, price as vanilla
            vanilla = self._create_vanilla(product)
            return self._bs_engine.price(vanilla, pricing_env) * multiplier

        if obs_type == ObservationType.DISCRETE:
            return self._price_discrete(product, pricing_env, S, K, T, r, q, sigma, L, U, multiplier, rebate)

        if obs_type == ObservationType.CONTINUOUS:
            return self._price_continuous(product, pricing_env, S, K, T, r, q, sigma, L, U, multiplier)

        raise PricingError(f"Unsupported observation type: {obs_type}")

    def _resolved_records(self, product: DoubleBarrierOption, pricing_env: PricingEnvironment):
        return resolve_current_records(
            product.observation_schedule,
            pricing_env,
            default_upper=product.upper_barrier,
            default_lower=product.lower_barrier,
            default_payoff=product.rebate,
            require_double=True,
            product=product,
        )

    def _breached_at_valuation(
        self, product: DoubleBarrierOption, pricing_env: PricingEnvironment, spot: float
    ) -> bool:
        """Whether an observation at the valuation instant is breached by the spot, at its own levels."""
        return any(
            spot <= rec.lower_barrier or spot >= rec.upper_barrier
            for rec in observed_at_valuation(self._resolved_records(product, pricing_env))
        )

    def _price_zero_maturity(
        self,
        product: DoubleBarrierOption,
        pricing_env: PricingEnvironment,
        spot: float,
        multiplier: float,
        knocked_in: bool = False,
    ) -> float:
        """Return price at maturity: a knock-in carried as triggered (``knocked_in``, ruling 2026-10-10) pays the
        payoff; otherwise a discrete schedule is breached only by its records at the valuation instant, each at its
        own levels (a knock-out hit there pays that record's payoff, W18c), and continuous and expiry monitoring
        observe the spot."""
        if product.observation_type == ObservationType.DISCRETE:
            hit = self._breached_at_valuation(product, pricing_env, spot)
        else:
            hit = product.is_barrier_hit(spot)
        if product.is_knock_out:
            if hit:
                if product.observation_type == ObservationType.DISCRETE:
                    return valuation_double_barrier_terms(product, pricing_env)[2] * multiplier
                return product.rebate * multiplier
            return product.get_payoff(spot)
        # Knock-in: carried as triggered, or hit now
        if knocked_in or hit:
            return product.get_payoff(spot)
        return product.rebate * multiplier

    def _price_beyond_discrete(
        self,
        product: DoubleBarrierOption,
        pricing_env: PricingEnvironment,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        L: float,
        U: float,
        multiplier: float,
        rebate: float,
    ) -> float:
        """The value through the first future observation (``first_observation``), no observation due now
        (``FIRST_OBSERVATION_BGK``: the spot inside or outside the corridor).

        A breach at the first future observation ``t1`` pays the knock-out's rebate at maturity or knocks the
        option in (Black-Scholes from ``t1``); otherwise the continuation is this engine's shifted formula from
        ``t1`` (no observation left: no barrier). Values exclude the terminal payment delay ``price`` applies.
        """
        schedule = product.observation_schedule
        if schedule is None or not schedule.records:
            raise PricingError("Discrete barrier monitoring requires ObservationSchedule.")
        schedule.assert_analytical_ready(default_payoff=product.rebate, pricing_env=pricing_env)
        freq = schedule.ensure_regular_frequency(schedule.times)
        first = first_future_observation(self._resolved_records(product, pricing_env), schedule, single=False)
        tau = T - first.time
        lower_shifted = apply_barrier_shift(barrier=L, is_up_barrier=False, volatility=sigma, observation_interval=freq)
        upper_shifted = apply_barrier_shift(barrier=U, is_up_barrier=True, volatility=sigma, observation_interval=freq)
        is_call = product.is_call()
        rebate_df = float(safe_exp(-r * tau))

        def vanilla(s: float) -> float:
            if is_zero(tau):
                return max(s - K, 0.0) if is_call else max(K - s, 0.0)
            sqrt_tau = math.sqrt(tau)
            d1 = (math.log(s / K) + (r - q + 0.5 * sigma * sigma) * tau) / (sigma * sqrt_tau)
            d2 = d1 - sigma * sqrt_tau
            if is_call:
                return self._bs_engine._price_call(s, K, tau, r, q, d1, d2)
            return self._bs_engine._price_put(s, K, tau, r, q, d1, d2)

        if product.is_knock_out:
            def hit_value(s: float) -> float:
                return rebate * rebate_df * multiplier
        else:
            def hit_value(s: float) -> float:
                return vanilla(s) * multiplier

        def continuation(s: float) -> float:
            if first.remaining == 0:
                return (vanilla(s) if product.is_knock_out else rebate * rebate_df) * multiplier
            return self._price_continuous(
                product=product, pricing_env=pricing_env, S=s, K=K, T=tau, r=r, q=q, sigma=sigma,
                L=lower_shifted, U=upper_shifted, multiplier=multiplier, vanilla_price=vanilla(s), rebate=rebate,
            )

        return through_first_observation(
            spot=S, first=first, rate=r, div=q, vol=sigma,
            hit_value=hit_value, continuation=continuation, remaining_time=tau,
            hit_is_constant=product.is_knock_out,
        )

    def _price_zero_vol(
        self,
        product: DoubleBarrierOption,
        spot: float,
        T: float,
        r: float,
        q: float,
        multiplier: float,
        L: float,
        U: float,
        rebate: float,
    ) -> float:
        """Price when volatility is zero (deterministic path), on the corridor ``L``..``U`` (a DISCRETE schedule's
        remaining records' levels, W18c), a hit paying ``rebate`` (their payoff on a knock-out)."""
        forward = float(spot * safe_exp((r - q) * T))
        is_outside = forward >= U or forward <= L
        discount = float(safe_exp(-r * T))

        if product.is_knock_out:
            if is_outside:
                return rebate * multiplier
            payoff = product.get_payoff(forward) / multiplier  # get_payoff already applies multiplier
            return payoff * discount

        # Knock-in
        if is_outside:
            payoff = product.get_payoff(forward) / multiplier
            return payoff * discount
        return rebate * multiplier

    def _price_continuous(
        self,
        product: DoubleBarrierOption,
        pricing_env: PricingEnvironment,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        L: float,
        U: float,
        multiplier: float,
        delta1: float = 0.0,
        delta2: float = 0.0,
        vanilla_price: Optional[float] = None,
        rebate: Optional[float] = None,
    ) -> float:
        """Price using Ikeda & Kuintomo continuous formula (``vanilla_price``: the knock-in parity's vanilla,
        read from ``pricing_env`` when not given; ``rebate``: default ``product.rebate``, a DISCRETE knock-out's
        remaining records' payoff, W18c)."""
        if rebate is None:
            rebate = product.rebate
        b = r - q
        ko_price = self._price_knock_out_ikeda_kuintomo(
            S=S,
            K=K,
            L=L,
            U=U,
            T=T,
            r=r,
            b=b,
            sigma=sigma,
            is_call=product.is_call(),
            delta1=delta1,
            delta2=delta2,
        )

        if product.is_knock_out:
            price = max(ko_price, 0.0)
            if not is_zero(rebate):
                # the rebate is paid only if a barrier is hit (spec 3C decision 28)
                p_inside = double_no_touch(S, T, r, q, sigma, L, U)
                price += rebate * float(safe_exp(-r * T)) * (1.0 - p_inside)
            return price * multiplier

        # Knock-in via parity
        if vanilla_price is None:
            vanilla_price = self._bs_engine.price(self._create_vanilla(product), pricing_env)
        price = max(vanilla_price - max(ko_price, 0.0), 0.0)
        if not is_zero(rebate):
            # a knock-in's rebate is paid if the corridor is never left
            price += rebate * float(safe_exp(-r * T)) * double_no_touch(S, T, r, q, sigma, L, U)
        return float(price * multiplier)

    def _price_discrete(
        self,
        product: DoubleBarrierOption,
        pricing_env: PricingEnvironment,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        L: float,
        U: float,
        multiplier: float,
        rebate: Optional[float] = None,
    ) -> float:
        """Price discrete observation using barrier shift (``rebate``: a knock-out's remaining records' payoff), at
        today's spot ``S`` on either side of the corridor (``BGK``: nothing is decided between observations).

        Outside the shifted corridor the Ikeda-Kunitomo series is evaluated as it stands (its continuation, clamped
        at 0 as everywhere). The rebate needs the double no-touch probability, which is defined only inside its
        corridor (its "outside: 0" branch would decide the hit now): refused there.
        """
        schedule = product.observation_schedule
        if schedule is None or not schedule.records:
            raise PricingError(
                "Discrete barrier monitoring requires ObservationSchedule."
            )
        schedule.assert_analytical_ready(default_payoff=product.rebate, pricing_env=pricing_env)
        freq = schedule.ensure_regular_frequency(schedule.times)

        lower_shifted = apply_barrier_shift(
            barrier=L,
            is_up_barrier=False,
            volatility=sigma,
            observation_interval=freq,
        )
        upper_shifted = apply_barrier_shift(
            barrier=U,
            is_up_barrier=True,
            volatility=sigma,
            observation_interval=freq,
        )
        if not lower_shifted < S < upper_shifted and not is_zero(rebate if rebate is not None else product.rebate):
            # TODO(bgk-beyond): a no-touch probability from a spot outside the shifted corridor (the image series'
            # continuation and its error bound) -- not implemented; price on QUAD / PDE / MC or FIRST_OBSERVATION_BGK
            raise ValidationError(
                f"TODO(bgk-beyond): spot {S} is outside the BGK-shifted corridor ({lower_shifted}, {upper_shifted}); "
                "the double no-touch probability of the rebate is not defined there"
            )

        return self._price_continuous(
            product=product,
            pricing_env=pricing_env,
            S=S,
            K=K,
            T=T,
            r=r,
            q=q,
            sigma=sigma,
            L=lower_shifted,
            U=upper_shifted,
            multiplier=multiplier,
            rebate=rebate,
        )

    def _price_expiry(
        self,
        product: DoubleBarrierOption,
        pricing_env: PricingEnvironment,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        L: float,
        U: float,
        multiplier: float,
    ) -> float:
        """Price expiry-observed double barrier using truncated-domain vanilla."""
        vanilla = self._create_vanilla(product)
        vanilla_price = self._bs_engine.price(vanilla, pricing_env)

        sqrt_t = safe_sqrt(T)
        drift_adj = (r - q - 0.5 * sigma * sigma) * T
        d2_u = (safe_log(S / U) + drift_adj) / (sigma * sqrt_t)
        d2_l = (safe_log(S / L) + drift_adj) / (sigma * sqrt_t)

        discount = float(safe_exp(-r * T))
        prob_outside = float(stats.norm.cdf(float(d2_u)) + stats.norm.cdf(-float(d2_l)))
        prob_inside = float(stats.norm.cdf(float(d2_l)) - stats.norm.cdf(float(d2_u)))

        if product.is_call():
            # Truncated call payoff = Call(K) - Call(U) - (U-K) * digital(U)
            d1_u = float(d2_u) + sigma * float(sqrt_t)
            call_u = self._bs_engine._price_call(S, U, T, r, q, d1_u, float(d2_u))
            truncated = max(vanilla_price - call_u - (U - K) * discount * stats.norm.cdf(float(d2_u)), 0.0)
        else:
            # Truncated put payoff = Put(K) - Put(L) - (K-L) * digital(L)
            d1_l = float(d2_l) + sigma * float(sqrt_t)
            put_l = self._bs_engine._price_put(S, L, T, r, q, d1_l, float(d2_l))
            truncated = max(vanilla_price - put_l - (K - L) * discount * stats.norm.cdf(-float(d2_l)), 0.0)

        if product.is_knock_out:
            price = truncated + product.rebate * discount * prob_outside
        else:
            price = max(vanilla_price - truncated, 0.0) + product.rebate * discount * prob_inside

        return float(price * multiplier)

    def _price_knock_out_ikeda_kuintomo(
        self,
        S: float,
        K: float,
        L: float,
        U: float,
        T: float,
        r: float,
        b: float,
        sigma: float,
        is_call: bool,
        delta1: float = 0.0,
        delta2: float = 0.0,
        max_terms: int = DEFAULT_MAX_TERMS,
    ) -> float:
        """
        Ikeda & Kuintomo (1992) infinite-series formula for double knock-out.

        Returns the unadjusted theoretical price (per unit, no multiplier).
        """
        F = float(U * float(safe_exp(delta1 * T)))
        E = float(L * float(safe_exp(delta2 * T)))

        if is_call:
            arg1 = K  # for d1, d3
            arg2 = F  # for d2, d4
        else:
            arg1 = E  # for y1, y3
            arg2 = K  # for y2, y4

        sqrt_t = float(safe_sqrt(T))
        denom = sigma * sqrt_t
        drift = float((b + 0.5 * sigma * sigma) * T)
        sig2 = sigma * sigma

        asset_sum = 0.0
        strike_sum = 0.0

        for n in range(-max_terms, max_terms + 1):
            U_pow = math.pow(U, n)
            L_pow = math.pow(L, n)
            U2n = U_pow * U_pow
            L2n = L_pow * L_pow
            L2n2 = L2n * L * L

            mu1 = 2.0 * (b - delta2 - n * (delta1 - delta2)) / sig2 + 1.0
            mu2 = 2.0 * n * (delta1 - delta2) / sig2
            mu3 = 2.0 * (b - delta2 + n * (delta1 - delta2)) / sig2 + 1.0

            w1 = float(safe_power((U_pow / L_pow), mu1) * safe_power((L / S), mu2))
            w2 = float(safe_power((math.pow(L, n + 1) / (U_pow * S)), mu3))

            d_a = float((float(safe_log(S * U2n / (arg1 * L2n))) + drift) / denom)
            d_b = float((float(safe_log(S * U2n / (arg2 * L2n))) + drift) / denom)
            d_c = float((float(safe_log(L2n2 / (arg1 * S * U2n))) + drift) / denom)
            d_d = float((float(safe_log(L2n2 / (arg2 * S * U2n))) + drift) / denom)

            cdf_ab = stats.norm.cdf(d_a) - stats.norm.cdf(d_b)
            cdf_cd = stats.norm.cdf(d_c) - stats.norm.cdf(d_d)

            term_asset = 0.0
            if math.isfinite(w1) and math.isfinite(w2):
                term_asset = w1 * cdf_ab - w2 * cdf_cd
            elif math.isfinite(w1):
                term_asset = w1 * cdf_ab
            elif math.isfinite(w2):
                term_asset = -w2 * cdf_cd
            asset_sum += term_asset

            w1_strike = float(safe_power((U_pow / L_pow), mu1 - 2.0) * safe_power((L / S), mu2))
            w2_strike = float(safe_power((math.pow(L, n + 1) / (U_pow * S)), mu3 - 2.0))

            cdf_ab_strike = stats.norm.cdf(d_a - denom) - stats.norm.cdf(d_b - denom)
            cdf_cd_strike = stats.norm.cdf(d_c - denom) - stats.norm.cdf(d_d - denom)

            term_strike = 0.0
            if math.isfinite(w1_strike) and math.isfinite(w2_strike):
                term_strike = w1_strike * cdf_ab_strike - w2_strike * cdf_cd_strike
            elif math.isfinite(w1_strike):
                term_strike = w1_strike * cdf_ab_strike
            elif math.isfinite(w2_strike):
                term_strike = -w2_strike * cdf_cd_strike
            strike_sum += term_strike

        df_carry = float(safe_exp((b - r) * T))
        df_riskfree = float(safe_exp(-r * T))

        if is_call:
            price = S * df_carry * asset_sum - K * df_riskfree * strike_sum
        else:
            price = K * df_riskfree * strike_sum - S * df_carry * asset_sum

        return float(price)

    def _create_vanilla(self, product: DoubleBarrierOption) -> EuropeanVanillaOption:
        """Create a vanilla option with same terms as the double barrier."""
        return EuropeanVanillaOption(
            strike=product.strike,
            option_type=product.option_type,
            maturity=product.maturity,
            exercise_date=product.exercise_date,
        )

    def _validate_inputs(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        L: float,
        U: float,
    ) -> None:
        """Comprehensive input validation."""
        validate_positive(S, "spot")
        validate_positive(K, "strike")
        validate_non_negative(T, "maturity")
        validate_positive(sigma, "volatility")
        validate_positive(L, "lower_barrier")
        validate_positive(U, "upper_barrier")

        if L >= U:
            raise ValidationError(
                f"Lower barrier ({L}) must be less than upper barrier ({U})"
            )

        if not (L < K < U):
            raise ValidationError(
                f"Strike ({K}) must be strictly between lower ({L}) and upper ({U}) "
                f"barriers for the Ikeda-Kuintomo formula."
            )

        if sigma < self.MIN_VOL or sigma > self.MAX_VOL:
            raise ValidationError(
                f"Volatility {sigma} outside supported range "
                f"[{self.MIN_VOL}, {self.MAX_VOL}]"
            )

        if T > self.MAX_MATURITY:
            raise ValidationError(
                f"Maturity {T} years exceeds maximum {self.MAX_MATURITY}"
            )

        if abs(r) > 1.0:
            raise ValidationError(f"Risk-free rate outside reasonable bounds: {r}")


    def static_replication(self, product, pricing_env):
        """An ended schedule's remaining contract (spec 2026-10-07 theta boundary 3.3), else none."""
        if remaining_contract_applies(product):
            return remaining_replication(product, knock_in_carried(product))
        return super().static_replication(product, pricing_env)

    def __repr__(self):
        if self.discrete_barrier_approximation is DiscreteBarrierApproximation.BGK:
            return "DoubleBarrierOptionAnalyticalEngine()"
        return ("DoubleBarrierOptionAnalyticalEngine("
                f"discrete_barrier_approximation={self.discrete_barrier_approximation.name})")
