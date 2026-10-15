"""
Analytical pricing engine for double sharkfin options.

The payoff is decomposed into a no-rebate double knock-out option, a
double-touch cash leg for the knock-out rebate, and a double-no-touch cash leg
for the no-hit rebate.
"""

import math
from dataclasses import replace
from typing import Optional

import numpy as np
from scipy import stats

from quantark.asset.equity.engine.barrier_schedule_rules import (
    forward_double_barrier_levels,
    forward_knock_out_cash,
    refuse_mixed_forward_levels,
)
from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.engine.capabilities import SettlementSupport
from quantark.asset.equity.engine.settlement_support import (
    constant_hit_lag_year_fraction,
    pending_receivable_pv,
    resolve_terminal_timing,
    terminal_lifecycle_pv,
    validate_settlement_capability,
)
from quantark.asset.equity.engine.monitoring_ended import monitoring_ended, remaining_replication
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option import DoubleBarrierOption, DoubleSharkfinOption
from quantark.asset.equity.product.option.observation_schedule import ObservationSchedule
from quantark.asset.equity.settlement import (
    CashflowKind,
    SettlementRequest,
    SettlementResolver,
)
from quantark.asset.equity.param import EngineParams
from quantark.priceenv import PricingEnvironment
from quantark.util.barrier_shift import apply_barrier_shift
from quantark.util.enum import DoubleBarrierType, ObservationType
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import NumericalError, PricingError, ValidationError
from quantark.util.numerical import (
    Tolerance,
    is_zero,
    safe_exp,
    safe_log,
    safe_sqrt,
    validate_non_negative,
    validate_positive,
)

from .double_barrier_option_engine import DoubleBarrierOptionAnalyticalEngine
from .double_no_touch import double_no_touch
from .first_observation import (
    first_future_observation,
    has_future_observation,
    observed_at_valuation,
    resolve_current_records,
    through_first_observation,
)


class DoubleSharkfinOptionAnalyticalEngine(BaseEngine):
    """
    Semi-closed-form analytical engine for DoubleSharkfinOption.

    Expiry-only monitoring is priced from truncated lognormal probabilities.
    Continuous monitoring uses the Ikeda-Kunitomo double-barrier option engine
    for the participation leg and the error-controlled double no-touch
    probability (``double_no_touch``) for the cash survival legs. Discrete monitoring applies a BGK barrier shift.
    """

    engine_type = EngineType.ANALYTICAL
    # params-only engine: instance state == construction arguments (patch spec 2026-09-03 §7)
    MODEL_FINGERPRINT_ATTRS = ('params', '_double_barrier_engine')
    settlement_support = SettlementSupport.EVENT_AND_TERMINAL
    supports_lifecycle_state = True

    MIN_MATURITY = 1e-10
    MIN_VOL = 0.001
    MAX_VOL = 5.0
    MAX_MATURITY = 50.0
    HIT_LOW_ORDER = 8
    HIT_HIGH_ORDER = 12
    HIT_EDGE_SD = 9.0          # below t_min neither barrier is reached within 9 sd: survival 1 to 4.5e-19

    def __init__(self, params: Optional[EngineParams] = None):
        """
        Initialize double sharkfin analytical engine.

        Args:
            params: Engine configuration parameters. (``max_terms`` was removed in 0.10.0 and ``quad_points``
                after it: the paid-at-hit integral controls its own error.)
        """
        super().__init__(params)
        self._double_barrier_engine = DoubleBarrierOptionAnalyticalEngine(params)

    def price(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        *,
        lifecycle_state=None,
    ) -> float:
        """
        Price a double sharkfin option analytically.

        Args:
            product: DoubleSharkfinOption instance.
            pricing_env: Market data environment.

        Returns:
            Present value scaled by product.contract_multiplier.
        """
        if not isinstance(product, DoubleSharkfinOption):
            raise PricingError(
                "DoubleSharkfinOptionAnalyticalEngine only supports "
                f"DoubleSharkfinOption, got {type(product).__name__}"
            )
        validate_settlement_capability(self, product, lifecycle_state)
        fixed_pv = terminal_lifecycle_pv(lifecycle_state, pricing_env)
        if fixed_pv is not None:
            return fixed_pv
        if monitoring_ended(product) and product.get_maturity(pricing_env) > 0.0:
            # no observation left (spec 2026-10-07 theta boundary 3.3): the remaining contract, by replication
            knocked_in = bool(getattr(lifecycle_state, "knocked_in", False))
            return (remaining_replication(product, knocked_in).price(pricing_env)
                    + pending_receivable_pv(lifecycle_state, pricing_env))
        terminal_timing = resolve_terminal_timing(product, pricing_env)
        pending_pv = pending_receivable_pv(lifecycle_state, pricing_env)

        spot = pricing_env.spot
        maturity = product.get_maturity(pricing_env)
        rate = pricing_env.get_rate(maturity)
        div = pricing_env.get_div_yield(maturity)
        vol = pricing_env.get_vol(product.strike, maturity)
        discrete = product.observation_type == ObservationType.DISCRETE
        resolved = None
        if discrete:
            # W18c (spec 2026-10-13 I4, A8): breached now only by a record at the valuation instant, at its own
            # levels; forward, the corridor every remaining record shares (None: the levels differ -- no closed
            # form, refused before any forward value)
            resolved = self._resolved_records(product, pricing_env)    # once per price
            hit_record = self._breached_at_valuation(product, pricing_env, spot, resolved)
            forward = forward_double_barrier_levels(product, pricing_env)
        else:
            hit_record = None
            forward = (product.upper_barrier, product.lower_barrier)
        # mixed remaining levels price no forward value: the product's corridor then feeds only the input checks
        upper, lower = forward if forward is not None else (product.upper_barrier, product.lower_barrier)

        self._validate_inputs(
            spot=spot,
            strike=product.strike,
            lower_barrier=lower,
            upper_barrier=upper,
            maturity=maturity,
            rate=rate,
            div=div,
            vol=vol,
            participation_rate=product.participation_rate,
            knock_out_rebate=product.knock_out_rebate,
            no_hit_rebate=product.no_hit_rebate,
            contract_multiplier=product.contract_multiplier,
        )

        if maturity < self.MIN_MATURITY:
            return (
                self._price_at_maturity(product, pricing_env, spot, terminal_timing.payment_df, resolved)
                + pending_pv
            )

        if product.observation_type not in (
            ObservationType.EXPIRY,
            ObservationType.CONTINUOUS,
            ObservationType.DISCRETE,
        ):
            raise PricingError(
                f"Unsupported observation type: {product.observation_type}"
            )

        if discrete and hit_record is None:
            if forward is None:
                refuse_mixed_forward_levels(
                    product.observation_schedule, pricing_env, default_payoff=product.knock_out_rebate,
                    business_days_in_year=product.business_days_in_year)
            if spot >= upper or spot <= lower or has_future_observation(resolved):
                # No observation deciding now, inside or outside the corridor: the next observation decides
                return (
                    self._price_through_first_observation(
                        product, pricing_env, spot, maturity, rate, div, vol, terminal_timing, upper, lower, resolved
                    )
                    + pending_pv
                )

        if hit_record is not None or (
            not discrete
            and product.observation_type != ObservationType.EXPIRY
            and product.is_barrier_hit(spot)
        ):
            # the deciding record's payoff on a DISCRETE schedule (W18c), the knock-out rebate otherwise
            return (
                self._price_already_hit(
                    product, pricing_env, terminal_timing.payment_df,
                    cash=hit_record.payoff if hit_record is not None else None,
                )
                + pending_pv
            )

        # Continuous hit-paid cash leg with a CONSTANT settlement lag: exact
        # exp(-rate * lag) scaling; hit-date-dependent lags raise here (patch
        # spec 2026-09-03 §5). The discrete leg discounts each node at its own
        # resolved settlement time and needs no factor.
        hit_lag = (
            constant_hit_lag_year_fraction(product, pricing_env, maturity)
            if (
                product.observation_type == ObservationType.CONTINUOUS
                and product.pay_at_hit
                and product.knock_out_rebate > 0.0
            )
            else 0.0
        )

        option_leg = self._price_no_rebate_knock_out(product, pricing_env)
        survival_prob = self._survival_probability(
            product, maturity, rate, div, vol, pricing_env, upper, lower
        )
        knock_out_cash = self._price_knock_out_cash_leg(
            product=product,
            maturity=maturity,
            rate=rate,
            div=div,
            vol=vol,
            pricing_env=pricing_env,
            survival_prob=survival_prob,
            hit_lag=hit_lag,
            upper=upper,
            lower=lower,
        )
        no_hit_cash = (
            product.no_hit_rebate
            * terminal_timing.payment_df
            * survival_prob
        )

        value = product.participation_rate * option_leg + knock_out_cash + no_hit_cash
        return (
            max(float(value), 0.0) * product.contract_multiplier
            + pending_pv
        )

    def _resolved_records(self, product: DoubleSharkfinOption, pricing_env: PricingEnvironment):
        return resolve_current_records(
            product.observation_schedule,
            pricing_env,
            default_upper=product.upper_barrier,
            default_lower=product.lower_barrier,
            default_payoff=product.knock_out_rebate,
            require_double=True,
            product=product,
        )

    def _breached_at_valuation(
        self, product: DoubleSharkfinOption, pricing_env: PricingEnvironment, spot: float, resolved=None
    ):
        """The first observation at the valuation instant the spot breaches at its own levels, or None
        (``resolved``: the records already resolved for this price)."""
        if resolved is None:
            resolved = self._resolved_records(product, pricing_env)
        for rec in observed_at_valuation(resolved):
            if spot <= rec.lower_barrier or spot >= rec.upper_barrier:
                return rec
        return None

    def _price_at_maturity(
        self,
        product: DoubleSharkfinOption,
        pricing_env: PricingEnvironment,
        spot: float,
        terminal_payment_df: float,
        resolved=None,
    ) -> float:
        """The value at maturity: a discrete schedule is breached only by its records at the valuation instant,
        each at its own levels and paying its own cash (hit-paid: at the record's payment time); continuous and
        expiry monitoring observe the spot. ``resolved``: the records already resolved for this price."""
        if product.observation_type == ObservationType.DISCRETE:
            record = self._breached_at_valuation(product, pricing_env, spot, resolved)
            if record is None:
                return product.get_no_hit_payoff(spot) * terminal_payment_df
            cash_df = (
                pricing_env.get_discount_factor(record.settlement_time)
                if product.pay_at_hit
                else terminal_payment_df
            )
            return record.payoff * cash_df * product.contract_multiplier
        if not product.is_barrier_hit(spot):
            return product.get_no_hit_payoff(spot) * terminal_payment_df
        if product.observation_type == ObservationType.EXPIRY:
            return product.get_barrier_payoff() * terminal_payment_df
        return self._price_already_hit(product, pricing_env, terminal_payment_df)

    def _price_through_first_observation(
        self,
        product: DoubleSharkfinOption,
        pricing_env: PricingEnvironment,
        spot: float,
        maturity: float,
        rate: float,
        div: float,
        vol: float,
        terminal_timing,
        upper: float,
        lower: float,
        resolved=None,
    ) -> float:
        """The value through the first future observation (``first_observation``), no observation deciding now,
        the spot inside or outside the corridor.

        A breach at the first future observation ``t1`` pays the knock-out cash (at the record's payment time when
        hit-paid, else at the terminal payment time); otherwise the continuation is this engine's shifted value
        from ``t1``: the participation leg (Ikeda-Kunitomo on the shifted corridor), the knock-out cash on the
        remaining observations and the no-hit cash. Under the engine's flat rate throughout. ``upper`` / ``lower``:
        the corridor the remaining records share (W18c). ``resolved``: the records already resolved for this price.
        The closures take an array of spots at ``t1``; the survivals to the later observations are one array call.
        """
        schedule = product.observation_schedule
        lower_shifted, upper_shifted = self._shift_discrete_barriers(product, vol, pricing_env, upper, lower)
        if resolved is None:
            resolved = self._resolved_records(product, pricing_env)
        first = first_future_observation(resolved, schedule, single=False)
        t1 = first.time
        tau = maturity - t1
        later = [rec for rec in resolved if rec.observation_time > t1]
        terminal_df = float(safe_exp(-rate * tau)) * terminal_timing.delay_df
        multiplier = product.contract_multiplier
        participation = product.participation_rate
        is_call = product.is_call()
        # a later hit pays the hit record's payoff (W18c follow-up)
        ko_cash, no_hit_cash = forward_knock_out_cash(product, pricing_env), product.no_hit_rebate
        db = self._double_barrier_engine

        def cash_df(settlement_time: float) -> float:
            return float(safe_exp(-rate * (settlement_time - t1)))

        def hit_value(s: np.ndarray) -> float:
            paid = cash_df(first.settlement_time) if product.pay_at_hit else terminal_df
            return first.payoff * paid * multiplier

        def survival(s: np.ndarray, horizon) -> np.ndarray:
            return double_no_touch(s, horizon, rate, div, vol, lower_shifted, upper_shifted)

        horizons = np.array([min(rec.observation_time, maturity) - t1 for rec in later])

        def continuation(s: np.ndarray) -> np.ndarray:
            if not later:
                return self._unmonitored_value(product, s, tau, rate, div, vol, terminal_timing.delay_df)
            option_leg = np.maximum(
                db._price_knock_out_ikeda_kuintomo(
                    S=s, K=product.strike, L=lower_shifted, U=upper_shifted, T=tau,
                    r=rate, b=rate - div, sigma=vol, is_call=is_call,
                ),
                0.0,
            ) * terminal_timing.delay_df
            survive = survival(s, tau)
            value = participation * option_leg + no_hit_cash * terminal_df * survive
            if ko_cash > 0.0:
                if product.pay_at_hit:
                    survivals = survival(s[:, None], horizons[None, :])    # spot x later observation
                    previous = 1.0
                    for j, rec in enumerate(later):
                        current = survivals[:, j]
                        value += ko_cash * np.maximum(previous - current, 0.0) * cash_df(rec.settlement_time)
                        previous = np.minimum(np.maximum(current, 0.0), previous)
                else:
                    value += ko_cash * terminal_df * (1.0 - survive)
            return np.maximum(value, 0.0) * multiplier

        return through_first_observation(
            spot=spot, first=first, rate=rate, div=div, vol=vol,
            hit_value=hit_value, continuation=continuation, remaining_time=tau,
            hit_is_constant=True,
            kinks=(product.strike, product.lower_barrier, product.upper_barrier),    # the no-hit payoff's
        )

    def _unmonitored_value(
        self,
        product: DoubleSharkfinOption,
        spot: float,
        tau: float,
        rate: float,
        div: float,
        vol: float,
        delay_df: float,
    ) -> float:
        """The no-hit payoff with no observation left: a call spread K..U (put spread L..K) plus the no-hit cash.
        ``spot`` a float or an array of spots."""
        array = isinstance(spot, np.ndarray)
        if is_zero(tau):
            if array:
                return np.array([product.get_no_hit_payoff(s) for s in spot.tolist()]) * delay_df
            return product.get_no_hit_payoff(spot) * delay_df
        bs = self._double_barrier_engine._bs_engine
        sqrt_tau = float(safe_sqrt(tau))

        def vanilla(strike: float, call: bool) -> float:
            log_moneyness = safe_log(spot / strike) if array else float(safe_log(spot / strike))
            d1 = (log_moneyness + (rate - div + 0.5 * vol * vol) * tau) / (vol * sqrt_tau)
            d2 = d1 - vol * sqrt_tau
            return bs._price_call(spot, strike, tau, rate, div, d1, d2) if call else bs._price_put(
                spot, strike, tau, rate, div, d1, d2)

        if product.is_call():
            spread = vanilla(product.strike, True) - vanilla(product.upper_barrier, True)
        else:
            spread = vanilla(product.strike, False) - vanilla(product.lower_barrier, False)
        value = product.participation_rate * spread + product.no_hit_rebate * float(safe_exp(-rate * tau))
        return value * delay_df * product.contract_multiplier

    def _price_already_hit(
        self,
        product: DoubleSharkfinOption,
        pricing_env: PricingEnvironment,
        terminal_payment_df: float,
        cash: Optional[float] = None,
    ) -> float:
        """Return the value when a monitored product is already hit: ``cash`` (the deciding record's payoff on a
        DISCRETE schedule, W18c), by default the knock-out rebate."""
        if cash is None:
            cash = product.knock_out_rebate
        if product.pay_at_hit:
            timing = SettlementResolver.resolve_contingent(
                product,
                SettlementRequest(
                    kind=CashflowKind.HIT,
                    determination_date=pricing_env.valuation_date,
                    determination_time=0.0,
                    cashflow_id="double-sharkfin:already-hit",
                ),
                pricing_env,
            )
            payment_df = timing.payment_df
        else:
            payment_df = terminal_payment_df
        return (
            cash
            * payment_df
            * product.contract_multiplier
        )

    def _price_no_rebate_knock_out(
        self, product: DoubleSharkfinOption, pricing_env: PricingEnvironment
    ) -> float:
        """Value the capped participation leg as a double knock-out option: no cash on a hit, so its records pay
        nothing (the knock-out cash leg pays it; a record paying it here would count it twice, W18c)."""
        schedule = product.observation_schedule
        if schedule is not None:
            schedule = ObservationSchedule(
                records=[replace(rec, payoff=0.0) for rec in schedule.records],
                aggregation_mode=schedule.aggregation_mode,
                frequency=schedule.frequency,
                monitoring_ended=schedule.monitoring_ended,
            )
        double_barrier = DoubleBarrierOption(
            strike=product.strike,
            option_type=product.option_type,
            upper_barrier=product.upper_barrier,
            lower_barrier=product.lower_barrier,
            barrier_type=DoubleBarrierType.KNOCK_OUT,
            maturity=product.maturity,
            exercise_date=product.exercise_date,
            settlement_date=product.settlement_date,
            settlement_convention=product.settlement_convention,
            rebate=0.0,
            observation_type=product.observation_type,
            observation_dates=product.observation_dates,
            observation_schedule=schedule,
            contract_multiplier=1.0,
        )
        return self._double_barrier_engine.price(double_barrier, pricing_env)

    def _survival_probability(
        self,
        product: DoubleSharkfinOption,
        maturity: float,
        rate: float,
        div: float,
        vol: float,
        pricing_env: PricingEnvironment,
        upper: Optional[float] = None,
        lower: Optional[float] = None,
    ) -> float:
        """Probability that neither double sharkfin barrier is hit (spec 3C §5.8: the no-touch probability
        carries its own error bound and is not clipped; a ``NumericalError`` from it propagates). DISCRETE: on the
        corridor ``upper`` / ``lower`` the remaining records share (W18c; default the product's)."""
        if upper is None or lower is None:
            upper, lower = product.upper_barrier, product.lower_barrier
        if product.observation_type == ObservationType.EXPIRY:
            prob = self._prob_inside_at_expiry(
                spot=pricing_env.spot,
                maturity=maturity,
                rate=rate,
                div=div,
                vol=vol,
                lower_barrier=product.lower_barrier,
                upper_barrier=product.upper_barrier,
            )
            return min(max(float(prob), 0.0), 1.0)
        if product.observation_type == ObservationType.CONTINUOUS:
            return double_no_touch(pricing_env.spot, maturity, rate, div, vol,
                                   product.lower_barrier, product.upper_barrier)
        if product.observation_type == ObservationType.DISCRETE:
            lower_barrier, upper_barrier = self._shift_discrete_barriers(
                product, vol, pricing_env, upper, lower
            )
            return double_no_touch(pricing_env.spot, maturity, rate, div, vol, lower_barrier, upper_barrier)
        raise PricingError(
            f"Unsupported observation type: {product.observation_type}"
        )

    def _price_knock_out_cash_leg(
        self,
        product: DoubleSharkfinOption,
        maturity: float,
        rate: float,
        div: float,
        vol: float,
        pricing_env: PricingEnvironment,
        survival_prob: float,
        hit_lag: float = 0.0,
        upper: Optional[float] = None,
        lower: Optional[float] = None,
    ) -> float:
        """Value the fixed cash leg paid when either barrier is hit (DISCRETE: on the corridor ``upper`` /
        ``lower`` the remaining records share, W18c).

        ``hit_lag`` is the constant year-fraction settlement delay of the
        continuous hit-paid leg (0.0 = pays at the hit; the unlagged path is
        bitwise unchanged). A DISCRETE hit pays the hit record's payoff (``forward_knock_out_cash``, W18c).
        """
        cash = forward_knock_out_cash(product, pricing_env)
        if cash <= 0.0:
            return 0.0

        if product.observation_type == ObservationType.EXPIRY or not product.pay_at_hit:
            return (
                cash
                * resolve_terminal_timing(product, pricing_env).payment_df
                * (1.0 - survival_prob)
            )

        if product.observation_type == ObservationType.DISCRETE:
            hit_discount_factor = self._discrete_hit_discount_factor(
                product=product,
                maturity=maturity,
                rate=rate,
                div=div,
                vol=vol,
                spot=pricing_env.spot,
                pricing_env=pricing_env,
                upper=upper,
                lower=lower,
            )
        else:
            hit_discount_factor = self._continuous_hit_discount_factor(
                spot=pricing_env.spot,
                maturity=maturity,
                rate=rate,
                div=div,
                vol=vol,
                lower_barrier=product.lower_barrier,
                upper_barrier=product.upper_barrier,
                survival_at_maturity=survival_prob,
            )
            if hit_lag != 0.0:
                # E[e^{-r(tau+L)} 1{tau<=T}] = e^{-rL} E[e^{-r tau} 1{tau<=T}]
                hit_discount_factor *= safe_exp(-rate * hit_lag)

        return cash * max(float(hit_discount_factor), 0.0)

    def _discrete_hit_discount_factor(
        self,
        product: DoubleSharkfinOption,
        maturity: float,
        rate: float,
        div: float,
        vol: float,
        spot: float,
        pricing_env: PricingEnvironment,
        upper: float,
        lower: float,
    ) -> float:
        """Approximate discounted first-hit probability on a discrete grid."""
        lower_barrier, upper_barrier = self._shift_discrete_barriers(product, vol, pricing_env, upper, lower)
        schedule = product.observation_schedule
        if schedule is None or not schedule.records:
            raise PricingError(
                "Discrete double sharkfin monitoring requires observation times."
            )
        resolved = schedule.resolve(
            pricing_env,
            default_upper=product.upper_barrier,
            default_lower=product.lower_barrier,
            default_payoff=product.knock_out_rebate,
            require_double=True,
            product=product,
        )

        previous_survival = 1.0
        hit_discount_factor = 0.0
        for record in resolved:
            current_time = min(float(record.observation_time), maturity)
            current_survival = double_no_touch(spot, current_time, rate, div, vol, lower_barrier, upper_barrier)
            first_hit_prob = max(previous_survival - current_survival, 0.0)
            hit_discount_factor += (
                first_hit_prob
                * pricing_env.get_discount_factor(record.settlement_time)
            )
            previous_survival = min(max(current_survival, 0.0), previous_survival)

        return hit_discount_factor

    def _continuous_hit_discount_factor(
        self,
        spot: float,
        maturity: float,
        rate: float,
        div: float,
        vol: float,
        lower_barrier: float,
        upper_barrier: float,
        survival_at_maturity: float,
    ) -> float:
        """
        Compute E[exp(-r tau) 1_{tau<=T}] from the survival curve.
        """
        if is_zero(rate):
            return 1.0 - survival_at_maturity

        integral = self._discounted_survival_integral(
            spot, maturity, rate, div, vol, lower_barrier, upper_barrier
        )

        hit_factor = (
            1.0
            - safe_exp(-rate * maturity) * survival_at_maturity
            - rate * integral
        )
        return max(float(hit_factor), 0.0)

    def _discounted_survival_integral(
        self,
        spot: float,
        maturity: float,
        rate: float,
        div: float,
        vol: float,
        lower_barrier: float,
        upper_barrier: float,
    ) -> float:
        """``int_0^T exp(-r t) P(t) dt``, ``P`` the double no-touch probability, with its error checked.

        Next to a barrier ``P`` falls from 1 on the time scale ``d^2 / vol^2`` (``d`` the log distance to the
        nearer barrier), far below ``T``: a rule spread over ``(0, T)`` misses it. Below ``t_min``, where neither
        barrier is within ``HIT_EDGE_SD`` drift-adjusted deviations, ``P`` is 1 to 4.5e-19 and the piece is exact;
        above it the panels double in length up to ``T``, so each sees ``P`` smooth. Two orders per panel; their
        difference beyond ``Tolerance.RELATIVE`` (relative to the integral, floor ``T``) raises ``NumericalError``.
        """
        d = min(float(safe_log(upper_barrier / spot)), float(safe_log(spot / lower_barrier)))
        drift = abs(rate - div - 0.5 * vol * vol)
        t_min = min(maturity, (d / (self.HIT_EDGE_SD * vol)) ** 2)
        while t_min > 0.0 and (d - drift * t_min) < self.HIT_EDGE_SD * vol * math.sqrt(t_min):
            t_min *= 0.25
        head = t_min if is_zero(rate) else -math.expm1(-rate * t_min) / rate
        edges = [t_min]
        while edges[-1] < maturity:
            edges.append(min(maturity, 2.0 * edges[-1]))
        values = []
        for order in (self.HIT_LOW_ORDER, self.HIT_HIGH_ORDER):
            nodes, weights = np.polynomial.legendre.leggauss(order)
            total = head
            for a, b in zip(edges[:-1], edges[1:]):
                half, mid = 0.5 * (b - a), 0.5 * (b + a)
                for x, w in zip(mid + half * nodes, half * weights):
                    t = float(x)
                    total += float(w) * float(safe_exp(-rate * t)) * double_no_touch(
                        spot, t, rate, div, vol, lower_barrier, upper_barrier
                    )
            values.append(total)
        low, high = values
        if abs(high - low) > Tolerance.RELATIVE * max(maturity, abs(high)):
            raise NumericalError(
                "double sharkfin paid-at-hit integral did not resolve: orders "
                f"{self.HIT_LOW_ORDER} and {self.HIT_HIGH_ORDER} differ by {abs(high - low):.3e}"
            )
        return high

    def _shift_discrete_barriers(
        self,
        product: DoubleSharkfinOption,
        vol: float,
        pricing_env: PricingEnvironment,
        upper: float,
        lower: float,
    ) -> tuple[float, float]:
        """Apply BGK shifts to the discrete corridor ``lower`` / ``upper`` (the remaining records' levels, W18c)."""
        schedule = product.observation_schedule
        if schedule is None or not schedule.records:
            raise PricingError(
                "Discrete double sharkfin monitoring requires ObservationSchedule."
            )
        schedule.assert_analytical_ready(
            default_payoff=product.knock_out_rebate,
            business_days_in_year=product.business_days_in_year,
            pricing_env=pricing_env,
        )
        frequency = schedule.ensure_regular_frequency(
            schedule.times,
            business_days_in_year=product.business_days_in_year,
        )
        lower_barrier = apply_barrier_shift(
            barrier=lower,
            is_up_barrier=False,
            volatility=vol,
            observation_interval=frequency,
        )
        upper_barrier = apply_barrier_shift(
            barrier=upper,
            is_up_barrier=True,
            volatility=vol,
            observation_interval=frequency,
        )
        if not (lower_barrier < product.strike < upper_barrier):
            raise PricingError(
                "Discrete barrier shift produced barriers that do not contain "
                f"strike: lower={lower_barrier}, strike={product.strike}, "
                f"upper={upper_barrier}"
            )
        return lower_barrier, upper_barrier

    def _prob_inside_at_expiry(
        self,
        spot: float,
        maturity: float,
        rate: float,
        div: float,
        vol: float,
        lower_barrier: float,
        upper_barrier: float,
    ) -> float:
        """Risk-neutral probability that terminal spot is inside (L, U)."""
        sqrt_t = safe_sqrt(maturity)
        drift_adj = (rate - div - 0.5 * vol * vol) * maturity
        d2_upper = (safe_log(spot / upper_barrier) + drift_adj) / (vol * sqrt_t)
        d2_lower = (safe_log(spot / lower_barrier) + drift_adj) / (vol * sqrt_t)
        return float(stats.norm.cdf(float(d2_lower)) - stats.norm.cdf(float(d2_upper)))

    def _validate_inputs(
        self,
        spot: float,
        strike: float,
        lower_barrier: float,
        upper_barrier: float,
        maturity: float,
        rate: float,
        div: float,
        vol: float,
        participation_rate: float,
        knock_out_rebate: float,
        no_hit_rebate: float,
        contract_multiplier: float,
    ) -> None:
        """Validate market and product inputs for analytical pricing."""
        validate_positive(spot, "spot")
        validate_positive(strike, "strike")
        validate_positive(lower_barrier, "lower_barrier")
        validate_positive(upper_barrier, "upper_barrier")
        validate_non_negative(maturity, "maturity")
        validate_positive(vol, "volatility")
        validate_non_negative(participation_rate, "participation_rate")
        validate_non_negative(knock_out_rebate, "knock_out_rebate")
        validate_non_negative(no_hit_rebate, "no_hit_rebate")
        validate_positive(contract_multiplier, "contract_multiplier")

        if lower_barrier >= upper_barrier:
            raise ValidationError(
                f"Lower barrier ({lower_barrier}) must be less than upper barrier "
                f"({upper_barrier})"
            )
        if not (lower_barrier < strike < upper_barrier):
            raise ValidationError(
                f"Strike ({strike}) must be strictly between lower "
                f"({lower_barrier}) and upper ({upper_barrier}) barriers."
            )
        if vol < self.MIN_VOL or vol > self.MAX_VOL:
            raise ValidationError(
                f"Volatility {vol} outside supported range "
                f"[{self.MIN_VOL}, {self.MAX_VOL}]"
            )
        if maturity > self.MAX_MATURITY:
            raise ValidationError(
                f"Maturity too long for analytical double sharkfin pricing: {maturity}"
            )
        if abs(rate) > 1.0:
            raise ValidationError(f"Risk-free rate outside reasonable bounds: {rate}")

    def static_replication(self, product, pricing_env):
        """An ended schedule's remaining contract (spec 2026-10-07 theta boundary 3.3), else none."""
        if monitoring_ended(product):
            return remaining_replication(product)
        return super().static_replication(product, pricing_env)

    def __repr__(self):
        return "DoubleSharkfinOptionAnalyticalEngine()"
