"""Opt-in autocallable wrappers around the shared QUAD V2 core."""

from copy import deepcopy
from dataclasses import asdict
import numpy as np

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.engine.capabilities import SettlementSupport
from quantark.asset.equity.engine.settlement_support import (
    pending_receivable_pv,
    terminal_lifecycle_pv,
    validate_settlement_capability,
)
from quantark.asset.equity.param.quad_v2_params import QuadV2Params
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.ko_reset_snowball_option import (
    KnockOutResetSnowballOption,
)
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import ValidationError
from .adapters import compile_contract
from .context import prepare_compiled, interval_moments
from .contract import Action, Event, CompiledContract


class AutocallableQuadEngineV2(BaseEngine):
    engine_type = EngineType.QUADRATURE
    settlement_support = SettlementSupport.EVENT_AND_TERMINAL
    supports_lifecycle_state = True
    supports_spot_greeks_grid = True
    product_type = None

    def __init__(self, params=None):
        if params is None:
            params = QuadV2Params()
        if not isinstance(params, QuadV2Params):
            raise ValidationError("V2 requires QuadV2Params")
        super().__init__(deepcopy(params))
        self._prepared = None
        self._prepared_key = None

    def _compile(self, product, pricing_env, components, event_phase, lifecycle_state, extra_ko_cash=(), extra_terminal_cash=0.0):
        if event_phase not in {"before", "after"}:
            raise ValidationError("event_phase must be before or after")
        if self.product_type is not None and type(product) is not self.product_type:
            raise ValidationError(
                f"{type(self).__name__} requires {self.product_type.__name__}"
            )
        validate_settlement_capability(self, product, lifecycle_state)
        if terminal_lifecycle_pv(lifecycle_state, pricing_env) is not None:
            channels = (
                ("total", "ko", "coupon", "terminal") if components else ("total",)
            )
            zero = (0.0,) * len(channels)
            return CompiledContract(
                (Event(0.0, (), ((Action(cash=zero, asset=zero),),)),),
                ("terminated",),
                0,
                channels,
                float(product.initial_price),
            )
        return compile_contract(
            product,
            pricing_env,
            self.params,
            components=components,
            event_phase=event_phase,
            lifecycle_state=lifecycle_state,
            extra_ko_cash=extra_ko_cash,
            extra_terminal_cash=extra_terminal_cash,
        )

    def prepare(
        self,
        product,
        pricing_env,
        spot_levels=None,
        *,
        components=False,
        event_phase="before",
        lifecycle_state=None,
        extra_ko_cash=(),
        extra_terminal_cash=0.0,
    ):
        """Compile and solve once for the explicitly declared spot range.

        Gaussian term inputs are sampled at the contractual strike. Strike-
        dependent volatility is rejected, rather than labelled local volatility.
        ``extra_ko_cash``/``extra_terminal_cash`` fold path-independent
        contingent cash (e.g. cash legs) into the KO records and terminal
        payoff; the compiled contract carries them, so the cache key covers
        them.
        """
        self.params.__post_init__()
        env = deepcopy(pricing_env)
        product = deepcopy(product)
        lifecycle_state = deepcopy(lifecycle_state)
        contract = self._compile(
            product, env, components, event_phase, lifecycle_state,
            extra_ko_cash=extra_ko_cash,
            extra_terminal_cash=extra_terminal_cash,
        )
        self._validate_market(product, env, contract)
        if spot_levels is None:
            bump = self.params.get_effective_bump_config()
            width = max(0.02, 2 * bump.spot_bump, 2 * (bump.gamma_spot_bump or 0.0))
            spot_levels = [env.spot * (1 - width), env.spot * (1 + width)]
        pending = pending_receivable_pv(lifecycle_state, env)
        context = prepare_compiled(
            contract, env, product.strike, self.params, spot_levels, pending_pv=pending
        )
        self._prepared = context
        self._prepared_key = self._key(contract, env, product.strike, pending)
        return context

    @staticmethod
    def _validate_market(product, env, contract):
        from quantark.param.vol.vol_surface import (
            FlatVolSurface,
            TermStructureVolSurface,
            ParallelShiftVolSurface,
        )
        from quantark.param.vol.trading_clock_surface import TradingClockVolSurface

        def deterministic(surface):
            if type(surface) in {FlatVolSurface, TermStructureVolSurface}:
                return True
            if type(surface) is ParallelShiftVolSurface:
                return deterministic(surface.base)
            if type(surface) is TradingClockVolSurface:
                return deterministic(surface.inner)
            return getattr(surface, "quad_v2_deterministic_variance", False) is True

        if any(e.time > 0 for e in contract.events) and not deterministic(
            env.vol_surface
        ):
            raise NotImplementedError(
                "QUAD V2 requires a declared deterministic volatility term structure; smile/local-vol models are unsupported"
            )
        for t in {e.time for e in contract.events if e.time > 0}:
            vols = [env.get_vol(product.strike * f, t) for f in (0.9, 1.0, 1.1)]
            if not np.all(np.isfinite(vols)) or min(vols) < 0:
                raise ValidationError("QUAD V2 requires finite nonnegative volatility")
            if np.ptp(vols) > 1e-12 * max(1.0, max(vols)):
                raise NotImplementedError(
                    "QUAD V2 requires strike-independent deterministic volatility"
                )

    def _key(self, contract, env, strike, pending):
        moments = interval_moments(env, strike, [e.time for e in contract.events])
        return (
            contract,
            tuple(a.tobytes() for a in moments),
            repr(asdict(self.params)),
            pending,
            contract.metadata.get("continuous_model"),
        )

    def _context(
        self,
        product,
        pricing_env,
        *,
        components=False,
        event_phase="before",
        lifecycle_state=None,
    ):
        contract = self._compile(
            product, pricing_env, components, event_phase, lifecycle_state
        )
        self._validate_market(product, pricing_env, contract)
        pending = pending_receivable_pv(lifecycle_state, pricing_env)
        key = self._key(contract, pricing_env, product.strike, pending)
        cached = self._prepared
        if (
            cached is not None
            and key == self._prepared_key
            and cached.spot_range[0] <= pricing_env.spot <= cached.spot_range[1]
        ):
            return cached
        return self.prepare(
            product,
            pricing_env,
            components=components,
            event_phase=event_phase,
            lifecycle_state=lifecycle_state,
        )

    def price(
        self, product, pricing_env, *, lifecycle_state=None, event_phase="before"
    ):
        return float(
            self._context(
                product,
                pricing_env,
                lifecycle_state=lifecycle_state,
                event_phase=event_phase,
            ).evaluate([pricing_env.spot])["price"][0]
        )

    def calculate_point_greeks(
        self, product, pricing_env, *, lifecycle_state=None, event_phase="before"
    ):
        result = self._context(
            product,
            pricing_env,
            lifecycle_state=lifecycle_state,
            event_phase=event_phase,
        ).evaluate([pricing_env.spot])
        return {name: float(result[name][0]) for name in ("price", "delta", "gamma")}

    def calculate_greeks(
        self, product, pricing_env, *, lifecycle_state=None, event_phase="before"
    ):
        """Preserve the existing desk finite-bump convention using one context."""
        context = self._context(
            product,
            pricing_env,
            lifecycle_state=lifecycle_state,
            event_phase=event_phase,
        )
        bumps = self.params.get_effective_bump_config()
        b, g, s = (
            bumps.spot_bump,
            bumps.gamma_spot_bump or bumps.spot_bump,
            pricing_env.spot,
        )
        if (
            s * (1 - max(b, g)) < context.spot_range[0]
            or s * (1 + max(b, g)) > context.spot_range[1]
        ):
            context = self.prepare(
                product,
                pricing_env,
                lifecycle_state=lifecycle_state,
                event_phase=event_phase,
            )
        values = context.evaluate(
            [s, s * (1 + b), s * (1 - b), s * (1 + g), s * (1 - g)]
        )["price"]
        return {
            "price": float(values[0]),
            "delta": float((values[1] - values[2]) / (2 * s * b)),
            "gamma": float((values[3] - 2 * values[0] + values[4]) / (s * g) ** 2),
        }

    def calculate_spot_greeks_curve(
        self,
        product,
        pricing_env,
        spot_levels,
        *,
        lifecycle_state=None,
        event_phase="before",
    ):
        spots = np.asarray(spot_levels, dtype=float).reshape(-1)
        if len(spots) == 0:
            return []
        context = self.prepare(
            product,
            pricing_env,
            spots,
            lifecycle_state=lifecycle_state,
            event_phase=event_phase,
        )
        result = context.evaluate(spots)
        return [
            dict(
                spot=float(s),
                price=float(result["price"][i]),
                delta=float(result["delta"][i]),
                gamma=float(result["gamma"][i]),
                calculation_mode="quad_v2_point",
            )
            for i, s in enumerate(spots)
        ]

    def price_components(
        self, product, pricing_env, *, lifecycle_state=None, event_phase="before"
    ):
        result = self._context(
            product,
            pricing_env,
            components=True,
            lifecycle_state=lifecycle_state,
            event_phase=event_phase,
        ).evaluate([pricing_env.spot])
        components = {
            name: float(value[0]) for name, value in result["components"].items()
        }
        return {
            "price": float(result["price"][0]),
            **components,
            "reconciliation_error": float(result["price"][0])
            - sum(components.values()),
        }

    def calculate_event_stats(
        self,
        product,
        pricing_env,
        *,
        lifecycle_state=None,
        streams=None,
        event_phase="before",
    ):
        """Independently propagated event probabilities and cashflow ledger.

        Phoenix coupon PVs are attributed to the earning observation, including
        memory released on KO. Deferred coupons are discounted to the actual
        termination settlement; the ledger groups them by payment event.
        """
        if terminal_lifecycle_pv(lifecycle_state, pricing_env) is not None:
            raise NotImplementedError(
                "terminated V2 contracts use the realized lifecycle ledger, not a prospective event distribution"
            )
        context = self._context(
            product,
            pricing_env,
            components="events",
            lifecycle_state=lifecycle_state,
            event_phase=event_phase,
        )
        result = context.evaluate([pricing_env.spot])
        return self._event_stats_from_result(
            product, pricing_env, context, result, 0, lifecycle_state
        )

    def calculate_event_stats_at_spots(
        self,
        product,
        pricing_env,
        spot_levels,
        *,
        lifecycle_state=None,
        streams=None,
        event_phase="before",
    ):
        """Per-spot event stats from ONE prepared context.

        Same discretization and diagnostics as ``calculate_event_stats``; the
        event surfaces are evaluated at every requested spot without
        re-solving, so finite-difference spot greeks of distribution-derived
        quantities (cash-leg PVs) cost one prepare instead of one per bump.
        Returns one stats object per requested spot, in order.
        """
        spots = [float(s) for s in np.asarray(spot_levels, dtype=float).reshape(-1)]
        if not spots:
            return []
        if terminal_lifecycle_pv(lifecycle_state, pricing_env) is not None:
            raise NotImplementedError(
                "terminated V2 contracts use the realized lifecycle ledger, not a prospective event distribution"
            )
        context = self._context(
            product,
            pricing_env,
            components="events",
            lifecycle_state=lifecycle_state,
            event_phase=event_phase,
        )
        result = context.evaluate(spots)
        return [
            self._event_stats_from_result(
                product, pricing_env, context, result, i, lifecycle_state
            )
            for i in range(len(spots))
        ]

    def _event_stats_from_result(
        self, product, pricing_env, context, result, index, lifecycle_state
    ):
        from quantark.asset.equity.engine.event_stats import (
            AutocallableEventStats,
            KOResetEventStats,
            PhoenixEventStats,
            payment_aware_cashflow_fields,
        )

        values = {k: float(v[index]) for k, v in result["components"].items()}
        metadata = context.diagnostics
        records = metadata["ko_metadata"]
        times = np.array(sorted({r[1] for r in records}))
        probability = np.array(
            [
                sum(values[f"ko_probability:{r[0]}"] for r in records if r[1] == t)
                for t in times
            ]
        )
        cash = np.array(
            [
                sum(values[f"ko_cash:{r[0]}"] for r in records if r[1] == t)
                for t in times
            ]
        )
        term = values["terminal_alive_cash"] + values["terminal_ki_cash"]
        determination = [r[1] for r in records]
        payment = [r[2] for r in records]
        flows = [values[f"ko_cash:{r[0]}"] for r in records]
        phoenix = isinstance(product, PhoenixOption)
        if phoenix:
            determination.extend(r[1] for r in records)
            payment.extend(r[2] for r in records)
            flows.extend(values[f"coupon_payment:{r[0]}"] for r in records)
        if lifecycle_state is not None and lifecycle_state.valuation_point is not None:
            from quantark.asset.equity.settlement import SettlementResolver

            point = lifecycle_state.valuation_point
            for cashflow in lifecycle_state.ledger.pending(point):
                timing = SettlementResolver.resolve_pending(
                    cashflow, pricing_env, valuation_point=point
                )
                determination.append(timing.determination_time)
                payment.append(timing.payment_time)
                flows.append(cashflow.amount * timing.payment_df)
        # Terminal entries remain last, after pending receivables, for the
        # EventDistribution conversion's terminal settlement readout.
        determination.extend([metadata["pre_maturity"], metadata["maturity"]])
        payment.extend(
            [
                metadata["pre_maturity"] + metadata["terminal_delay"],
                metadata["maturity"] + metadata["terminal_delay"],
            ]
        )
        flows.extend(
            values[f"terminal_{suffix}_cash"]
            + (values[f"terminal_{suffix}_coupon"] if phoenix else 0.0)
            for suffix in ("alive", "ki")
        )
        fields = dict(
            pv=float(result["price"][index]),
            ko_times=times,
            ko_probability=probability,
            survival_probability=1 - np.cumsum(probability),
            expected_discounted_ko_cashflow=cash,
            ki_probability=values["terminal_ki_probability"],
            expected_discounted_maturity_cashflow=term,
            reconciliation_error=float(result["price"][index]) - sum(flows),
            ki_ever_probability=(
                None
                if metadata["continuous"]
                else 1.0
                if metadata["initial_ki"]
                else values["ki_ever"]
            ),
            ki_survive_knocked_in_probability=values["terminal_ki_probability"],
            **payment_aware_cashflow_fields(
                pricing_env,
                determination_times=determination,
                payment_times=payment,
                expected_discounted_cashflows=flows,
            ),
        )
        if phoenix:
            return PhoenixEventStats(
                **fields,
                coupon_probability=np.array(
                    [values[f"coupon_probability:{r[0]}"] for r in records]
                ),
                expected_discounted_coupon_cashflow=np.array(
                    [values[f"coupon_cash:{r[0]}"] for r in records]
                ),
                coupon_payment_is_path_dependent=metadata["expiry_coupons"],
            )
        if isinstance(product, KnockOutResetSnowballOption):
            pre = [r for r in records if r[3] == "pre"]
            post = [r for r in records if r[3] == "post"]
            fields.update(
                pre_ko_times=np.array([r[1] for r in pre]),
                post_ko_times=np.array([r[1] for r in post]),
                pre_ko_probability=np.array(
                    [values[f"ko_probability:{r[0]}"] for r in pre]
                ),
                post_ko_probability=np.array(
                    [values[f"ko_probability:{r[0]}"] for r in post]
                ),
                pre_ko_probability_total=sum(
                    values[f"ko_probability:{r[0]}"] for r in pre
                ),
                post_ko_probability_total=sum(
                    values[f"ko_probability:{r[0]}"] for r in post
                ),
                expected_discounted_post_ko_cashflow=sum(
                    values[f"ko_cash:{r[0]}"] for r in post
                ),
            )
            return KOResetEventStats(**fields)
        return AutocallableEventStats(**fields)

    def price_with_events(
        self,
        product,
        pricing_env,
        emit_distribution=True,
        streams=None,
        *,
        lifecycle_state=None,
        event_phase="before",
    ):
        from quantark.cashleg.event_distribution import EventDistribution, PricingResult

        if emit_distribution:
            stats = self.calculate_event_stats(
                product,
                pricing_env,
                lifecycle_state=lifecycle_state,
                streams=streams,
                event_phase=event_phase,
            )
            return PricingResult(
                npv=stats.pv,
                event_distribution=EventDistribution.from_autocallable_stats(stats),
            )
        return PricingResult(
            npv=self.price(
                product,
                pricing_env,
                lifecycle_state=lifecycle_state,
                event_phase=event_phase,
            ),
            event_distribution=EventDistribution.trivial(
                product.get_maturity(pricing_env)
            ),
        )

    def calculate_scenarios(self, scenarios, *, event_phase="before"):
        """Price explicit (product, environment[, lifecycle]) scenarios.

        Time scenarios supply an already aged contract and resolved lifecycle.
        Each changed market is compiled and fingerprinted before any reuse.
        """
        result = []
        for scenario in scenarios:
            product, env, *state = scenario
            result.append(
                self.calculate_point_greeks(
                    product,
                    env,
                    lifecycle_state=state[0] if state else None,
                    event_phase=event_phase,
                )
            )
        return result


class SnowballQuadEngineV2(AutocallableQuadEngineV2):
    product_type = SnowballOption


class PhoenixQuadEngineV2(AutocallableQuadEngineV2):
    product_type = PhoenixOption


class KOResetSnowballQuadEngineV2(AutocallableQuadEngineV2):
    product_type = KnockOutResetSnowballOption
