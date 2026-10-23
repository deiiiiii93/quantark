"""Opt-in autocallable wrappers around the shared QUAD V2 core."""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
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
from quantark.execution.errors import CapabilityError
from quantark.util.exceptions import NumericalError, ValidationError
from .adapters import compile_contract
from .context import prepare_compiled, interval_moments
from .contract import Action, Event, CompiledContract


def _canonical(value):
    """An equality-comparable form of compiled-contract metadata (arrays by bytes, mappings by sorted items)."""
    if isinstance(value, np.ndarray):
        return ("ndarray", value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, (dict, MappingProxyType)):
        return tuple(sorted((key, _canonical(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_canonical(item) for item in value)
    return value


@dataclass
class _SpotBatch:
    """One ``spot_batch`` session: prepared contexts by session key, most recently used last."""

    lo: float
    hi: float
    max_contexts: int
    contexts: list = field(default_factory=list)
    seen: list = field(default_factory=list)
    failed: list = field(default_factory=list)

    def lookup(self, key, spot):
        for i, (k, context) in enumerate(self.contexts):
            if k == key and context.spot_range[0] <= spot <= context.spot_range[1]:
                self.contexts.append(self.contexts.pop(i))
                return context
        return None

    def store(self, key, context):
        self.contexts = [(k, c) for k, c in self.contexts if not k == key]
        self.contexts.append((key, context))
        del self.contexts[: max(0, len(self.contexts) - self.max_contexts)]
        if not any(k == key for k in self.seen):
            self.seen.append(key)

    def widens(self, key):
        """A key already used once whose wide prepare has not failed (callers are inside the range)."""
        return any(k == key for k in self.seen) and not any(k == key for k in self.failed)


class AutocallableQuadEngineV2(BaseEngine):
    engine_type = EngineType.QUADRATURE
    settlement_support = SettlementSupport.EVENT_AND_TERMINAL
    supports_lifecycle_state = True
    supports_spot_greeks_grid = True
    supports_cash_leg_folding = True
    supports_spot_batch = True
    product_type = None

    def __init__(self, params=None):
        if params is None:
            params = QuadV2Params()
        if not isinstance(params, QuadV2Params):
            raise ValidationError("V2 requires QuadV2Params")
        super().__init__(deepcopy(params))
        self._prepared = None
        self._prepared_key = None
        self._batch = None

    def __getstate__(self):
        """Copies and pickles start cold, as a fresh engine.

        The prepared context is a cache (and its read-only diagnostics mapping does not pickle), so a
        copy never inherits the original's call history.
        """
        state = self.__dict__.copy()
        state["_prepared"] = None
        state["_prepared_key"] = None
        state["_batch"] = None
        return state

    def spot_width(self, *extra_spot_bumps: float) -> float:
        """Relative half-width of the spot range a default context covers around one spot: at least 2%, and
        twice every spot bump (the engine's own and any ``extra_spot_bumps`` a caller will apply)."""
        bump = self.params.get_effective_bump_config()
        bumps = [bump.spot_bump, bump.gamma_spot_bump or 0.0, *extra_spot_bumps]
        return max([0.02] + [2 * b for b in bumps])

    @contextmanager
    def spot_batch(self, spot_range: "tuple[float, float]", *, max_contexts: int = 32):
        """Scope a session that reuses prepared contexts across calls differing only by spot.

        Inside the session a call at a spot in ``spot_range`` reuses a context when the engine's cache key,
        the compiled contract's metadata and the event phase are all equal and its spot lies in that
        context's range; a call outside ``spot_range`` takes the scalar single-slot path. The first
        use of a key prepares exactly the scalar context (the default width around its spot); a later use
        at a spot outside it, inside ``spot_range``, prepares one context over the whole declared range.
        A wide prepare over the resource limit (``NumericalError``) falls back to the scalar context and
        is not retried for that key. On exit the single-slot cache is restored and the contexts dropped.
        Inside a session ``_prepared`` is not the last context evaluated (hits and wide contexts live in the
        session), so code that reads it for diagnostics must run outside one. Sessions do not nest, and an
        engine in a session must not be shared across threads.
        """
        try:
            lo, hi = (float(x) for x in spot_range)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"spot_batch needs a (lo, hi) pair of numbers, got {spot_range!r}") from exc
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 0 or lo > hi:
            raise ValidationError(f"spot_batch needs a finite range 0 < lo <= hi, got {spot_range!r}")
        if isinstance(max_contexts, bool) or not isinstance(max_contexts, int) or max_contexts < 1:
            raise ValidationError(f"max_contexts must be a positive int, got {max_contexts!r}")
        if getattr(self, "_batch", None) is not None:
            raise ValidationError("spot_batch sessions do not nest")
        saved = (self._prepared, self._prepared_key)
        self._batch = _SpotBatch(lo, hi, max_contexts)
        try:
            yield self
        finally:
            self._batch = None
            self._prepared, self._prepared_key = saved

    def _batch_context(
        self, batch, key, contract, product, pricing_env, pending, *, components, event_phase,
        lifecycle_state, fold,
    ):
        """The session's context for ``key`` at this spot (see ``spot_batch``)."""
        spot = pricing_env.spot
        context = batch.lookup(key, spot)
        if context is not None:
            return context
        if batch.widens(key):
            try:
                context = prepare_compiled(
                    contract, pricing_env, product.strike, self.params, [batch.lo, batch.hi],
                    pending_pv=pending,
                )
            except NumericalError:
                batch.failed.append(key)
        if context is None:
            context = self.prepare(
                product, pricing_env, components=components, event_phase=event_phase,
                lifecycle_state=lifecycle_state, fold=fold,
            )
        batch.store(key, context)
        return context

    def _compile(self, product, pricing_env, components, event_phase, lifecycle_state, fold=None):
        if event_phase not in {"before", "after"}:
            raise ValidationError("event_phase must be before or after")
        if self.product_type is not None and type(product) is not self.product_type:
            raise ValidationError(
                f"{type(self).__name__} requires {self.product_type.__name__}"
            )
        validate_settlement_capability(self, product, lifecycle_state)
        if terminal_lifecycle_pv(lifecycle_state, pricing_env) is not None:
            if fold is not None and not fold.is_empty:
                raise CapabilityError(
                    "a terminated V2 contract has no prospective event distribution; "
                    "its cash legs cannot be folded"
                )
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
            fold=fold,
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
        fold=None,
    ):
        """Compile and solve once for the explicitly declared spot range.

        Gaussian term inputs are sampled at the contractual strike. Strike-
        dependent volatility is rejected, rather than labelled local volatility.
        ``fold`` is a ``FoldOverlay`` of cash legs bound into the KO records and
        each maturity branch's terminal payoff (see ``compile_contract``); the
        compiled contract carries the bound cash and the cache key names the
        overlay, so a later unfolded or differently folded call never reuses it.
        """
        self.params.__post_init__()
        env = deepcopy(pricing_env)
        product = deepcopy(product)
        lifecycle_state = deepcopy(lifecycle_state)
        fold = self._active_fold(fold)
        contract = self._compile(
            product, env, components, event_phase, lifecycle_state, fold=fold
        )
        self._validate_market(product, env, contract)
        if spot_levels is None:
            width = self.spot_width()
            spot_levels = [env.spot * (1 - width), env.spot * (1 + width)]
        pending = pending_receivable_pv(lifecycle_state, env)
        context = prepare_compiled(
            contract, env, product.strike, self.params, spot_levels, pending_pv=pending
        )
        self._prepared = context
        self._prepared_key = self._key(contract, env, product.strike, pending, fold)
        return context

    @staticmethod
    def _active_fold(fold):
        """``None`` for an absent or empty overlay: the unfolded contract and key."""
        return None if fold is None or fold.is_empty else fold

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

    def _key(self, contract, env, strike, pending, fold=None):
        moments = interval_moments(env, strike, [e.time for e in contract.events])
        return (
            contract,
            tuple(a.tobytes() for a in moments),
            repr(asdict(self.params)),
            pending,
            contract.metadata.get("continuous_model"),
            fold,
        )

    def _context(
        self,
        product,
        pricing_env,
        *,
        components=False,
        event_phase="before",
        lifecycle_state=None,
        fold=None,
    ):
        fold = self._active_fold(fold)
        contract = self._compile(
            product, pricing_env, components, event_phase, lifecycle_state, fold=fold
        )
        self._validate_market(product, pricing_env, contract)
        pending = pending_receivable_pv(lifecycle_state, pricing_env)
        key = self._key(contract, pricing_env, product.strike, pending, fold)
        batch = getattr(self, "_batch", None)
        if batch is not None and batch.lo <= pricing_env.spot <= batch.hi:
            return self._batch_context(
                batch, (key, _canonical(contract.metadata), event_phase), contract, product,
                pricing_env, pending, components=components, event_phase=event_phase,
                lifecycle_state=lifecycle_state, fold=fold,
            )
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
            fold=fold,
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

    def price_folded(
        self,
        product,
        pricing_env,
        overlay,
        *,
        lifecycle_state=None,
        event_phase="before",
    ):
        """Per-unit NPV with ``overlay``'s cash legs folded into the contract: one context.

        An empty overlay returns ``price()`` bitwise. A non-empty one refuses
        ``event_phase="after"`` (the fold structure includes observations at the
        valuation instant, which that phase drops) and a terminated lifecycle (no
        prospective distribution). ``overlay.constant_pv`` is not included.
        """
        if overlay.is_empty:
            return self.price(
                product,
                pricing_env,
                lifecycle_state=lifecycle_state,
                event_phase=event_phase,
            )
        if event_phase == "after":
            raise CapabilityError(
                "QUAD V2 folds cash legs only with event_phase='before': the folded "
                "event structure includes observations at the valuation instant"
            )
        return float(
            self._context(
                product,
                pricing_env,
                lifecycle_state=lifecycle_state,
                event_phase=event_phase,
                fold=overlay,
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
                # each knock-out's settlement and the two maturity outcomes (spec 3B §5.4, F6)
                pre_ko_payment_times=np.array([r[2] for r in pre], dtype=float),
                post_ko_payment_times=np.array([r[2] for r in post], dtype=float),
                expected_discounted_pre_maturity_cashflow=values["terminal_alive_cash"],
                expected_discounted_post_maturity_cashflow=values["terminal_ki_cash"],
                pre_maturity_probability=values["terminal_alive_probability"],
                post_maturity_probability=values["terminal_ki_probability"],
                pre_maturity_time=metadata["pre_maturity"],
                post_maturity_time=metadata["maturity"],
                pre_maturity_payment_time=metadata["pre_maturity"] + metadata["terminal_delay"],
                post_maturity_payment_time=metadata["maturity"] + metadata["terminal_delay"],
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
