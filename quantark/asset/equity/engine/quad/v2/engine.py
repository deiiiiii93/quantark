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
                # the wide prepare keeps today's resource-limit refusal (no local intervals): its fallback, the
                # scalar context, is ``price()`` bitwise (sub-resolution run S4, S2 concern 1)
                context = prepare_compiled(
                    contract, pricing_env, product.strike, self.params, [batch.lo, batch.hi],
                    pending_pv=pending, local_intervals=False,
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
        """``EventStatistics`` of the contract (unified event statistics spec §Contracts; ``quad/v2/statistics.py``).

        Every termination outcome is its own backward channel of the price's recursion, read at the spot; the
        knock-in path statistics come from the forward adjoint pass of the same recursion (``quad/v2/adjoint.py``),
        run only when ``streams`` asks for the knock-in stream (``None``: everything). ``pv`` is ``price()``'s value
        (pending receivables included). A terminated lifecycle state returns its receivables alone, or ``None``
        when nothing is left unpaid (spec AC17).
        """
        if terminal_lifecycle_pv(lifecycle_state, pricing_env) is not None:
            return self._terminated_statistics(product, pricing_env, lifecycle_state, event_phase)
        context = self._context(
            product,
            pricing_env,
            components="events",
            lifecycle_state=lifecycle_state,
            event_phase=event_phase,
        )
        price = None
        if context._march.pointwise_recursion:
            price = type(self)(self.params).price(
                product, pricing_env, lifecycle_state=lifecycle_state, event_phase=event_phase
            )
        return self._statistics_at(
            product, pricing_env, context, pricing_env.spot, lifecycle_state, streams, price=price
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
        """Per-spot event statistics from ONE prepared context.

        Same discretization and diagnostics as ``calculate_event_stats``; each requested spot is read from the one
        context without re-solving (its own forward pass for the knock-in fields), so finite-difference spot greeks of
        distribution-derived quantities (cash-leg PVs) cost one prepare instead of one per bump. A context prepared
        over a wider spot range than the default is a different mesh, so its ``pv`` is that context's price there.
        Returns one statistics object per requested spot, in order.
        """
        spots = [float(s) for s in np.asarray(spot_levels, dtype=float).reshape(-1)]
        if not spots:
            return []
        if terminal_lifecycle_pv(lifecycle_state, pricing_env) is not None:
            stats = self._terminated_statistics(product, pricing_env, lifecycle_state, event_phase)
            return [stats for _ in spots]
        cached = self._prepared
        width = self.spot_width()
        outside_default = min(spots) < pricing_env.spot * (1-width) or max(spots) > pricing_env.spot * (1+width)
        if outside_default and (cached is None or min(spots) < cached.spot_range[0] or max(spots) > cached.spot_range[1]):
            context = self.prepare(
                product, pricing_env, [min(spots), max(spots)], components="events",
                lifecycle_state=lifecycle_state, event_phase=event_phase,
            )
        else:
            context = self._context(
                product,
                pricing_env,
                components="events",
                lifecycle_state=lifecycle_state,
                event_phase=event_phase,
            )
        if min(spots) < context.spot_range[0] or max(spots) > context.spot_range[1]:
            context = self.prepare(
                product, pricing_env, [min(spots), max(spots)], components="events",
                lifecycle_state=lifecycle_state, event_phase=event_phase,
            )
        # D23 at the env spot, as ``calculate_event_stats``: where the recursion read a grid function pointwise its
        # price channel need not be price()'s bitwise, so pv there is price() itself (other spots: the context's own
        # price channel, as documented above)
        price = None
        if context._march.pointwise_recursion and any(spot == float(pricing_env.spot) for spot in spots):
            price = type(self)(self.params).price(
                product, pricing_env, lifecycle_state=lifecycle_state, event_phase=event_phase
            )
        return [
            self._statistics_at(product, pricing_env, context, spot, lifecycle_state, streams,
                                price=price if spot == float(pricing_env.spot) else None)
            for spot in spots
        ]

    def _provenance(self):
        from quantark.asset.equity.engine.event_stats import StatisticsProvenance

        name = type(self).__name__
        return StatisticsProvenance(statistics_model=name, price_model=name)

    def _statistics_at(self, product, pricing_env, context, spot, lifecycle_state, streams, price=None):
        """The statistics of ``context`` read at ``spot``; ``price``: the price path's value there, or ``None`` to
        read it from the context's price channel."""
        from .context import price_channel
        from .statistics import v2_event_statistics

        result = context.evaluate([spot])
        values = {name: float(value[0]) for name, value in result["components"].items()}
        # D23: the price channel read alone -- the one-channel arithmetic of price()'s own readout on the recursion's
        # own channel 0 (every grid operation of the recursion is per channel); the caller passes price() itself
        # where the recursion read a grid function pointwise (``MarchLog.pointwise_recursion``)
        pv = float(price_channel(context).evaluate([spot])["price"][0]) if price is None else float(price)
        return v2_event_statistics(
            product, pricing_env, context, values, pv=pv, spot=float(spot),
            lifecycle_state=lifecycle_state, streams=streams, provenance=self._provenance(),
        )

    def _terminated_statistics(self, product, pricing_env, lifecycle_state, event_phase):
        """A knocked-out or matured lifecycle state: receivables only, or ``None`` when settled (spec AC17)."""
        from .statistics import terminated_statistics

        validate_settlement_capability(self, product, lifecycle_state)
        if self.product_type is not None and type(product) is not self.product_type:
            raise ValidationError(f"{type(self).__name__} requires {self.product_type.__name__}")
        knocked_in = bool(getattr(lifecycle_state, "knocked_in",
                                  getattr(product, "_otc_lifecycle_knocked_in", False)))
        pv = self.price(product, pricing_env, lifecycle_state=lifecycle_state, event_phase=event_phase)
        return terminated_statistics(product, pricing_env, lifecycle_state, pv=pv, provenance=self._provenance(),
                                     knocked_in=knocked_in)

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
        """NPV (``price()``) and the event distribution (``EventDistribution.from_statistics``); a terminated
        lifecycle state has no event left (``EventDistribution.ended``, spec AC17). ``emit_distribution=False``
        gets the maturity-only distribution without statistics."""
        from quantark.cashleg.event_distribution import EventDistribution, PricingResult, knock_in_was_carried

        if emit_distribution:
            stats = self.calculate_event_stats(
                product,
                pricing_env,
                lifecycle_state=lifecycle_state,
                streams=streams,
                event_phase=event_phase,
            )
            if stats is None:
                return PricingResult(
                    npv=self.price(product, pricing_env, lifecycle_state=lifecycle_state, event_phase=event_phase),
                    event_distribution=EventDistribution.ended(provenance=self._provenance()),
                )
            return PricingResult(npv=stats.pv, event_distribution=EventDistribution.from_statistics(
                stats, knock_in_carried=knock_in_was_carried(product, lifecycle_state)))
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
