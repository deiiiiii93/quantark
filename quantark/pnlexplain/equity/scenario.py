"""ScenarioCache: state(S) for a SET of applied factors + memoised values (spec §6)."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, FrozenSet, Iterable, Optional, Tuple

from quantark.asset.equity.riskmeasures.greeks.bump_envs import resolve_bump_engine
from quantark.param.vol.sticky import shocked_surface
from quantark.pnlexplain.base import MARKET_FACTORS, Factor, ValueBreakdown
from quantark.pnlexplain.config import PnLExplainConfig
from quantark.pnlexplain.equity.factor_diff import FactorMoves
from quantark.pnlexplain.equity.lifecycle import LifecycleTransition
from quantark.pnlexplain.equity.snapshot import ValuationSnapshot, value
from quantark.util.enum.greek_conventions import GreekConvention


class ScenarioCache:
    """All scenario values of one explain call come from this one object."""

    def __init__(self, snap0: ValuationSnapshot, snap1: ValuationSnapshot,
                 transition: LifecycleTransition, moves: FactorMoves, config: PnLExplainConfig):
        self.snap0, self.snap1, self.transition, self.moves, self.config = snap0, snap1, transition, moves, config
        self.effective: FrozenSet[Factor] = frozenset(
            f for f in MARKET_FACTORS if f in moves.changed and f in moves.coordinate.applicable
        )
        # One bump context PER ENGINE OBJECT, every one of them resolved at the
        # t0 product and market, and reused for every state that engine values,
        # including the t1 endpoint (spec §5.1 "one scenario cache", §6).
        # A replacement (MODEL) engine gets the same treatment: Shapley prices
        # {MODEL} without TIME, i.e. the new engine at the t0 state, so its
        # context must be frozen there too, not at the t1 market. Re-resolving
        # any endpoint at its own product/market would hand it a different
        # frozen PDE grid or MC seed context, and the event / model rows would
        # then carry re-meshing / seed noise instead of the event. The price of
        # that exactness is that pv_t1 can differ from the production engine's
        # fresh-grid MTM: the recorders report that as `gap_states`.
        contexts: Dict[int, Any] = {}

        def context(engine: Any) -> Any:
            key = id(engine)
            if key not in contexts:
                contexts[key] = resolve_bump_engine(snap0.product, snap0.pricing_env, engine)
            return contexts[key]

        self.bump_engine_t0 = context(snap0.engine)
        self.bump_engine_alive = context(transition.engine_alive_t1)
        self.bump_engine_t1 = context(snap1.engine)
        self._memo: Dict[FrozenSet[Factor], ValueBreakdown] = {}
        self._t1: Optional[ValueBreakdown] = None

    def normalize(self, applied: Iterable[Factor]) -> FrozenSet[Factor]:
        return frozenset(applied) & self.effective

    def build_state(self, applied: Iterable[Factor]) -> Tuple[Any, Any, Any, Any]:
        """(product, engine, env, valuation_point) for the state after applying `applied`."""
        s = self.normalize(applied)
        e0, e1 = self.snap0.pricing_env, self.snap1.pricing_env
        env = deepcopy(e0)
        product, engine, point = self.snap0.product, self.bump_engine_t0, self.snap0.point
        if Factor.TIME in s:
            env.valuation_date = self.snap1.date
            product, point = self.transition.product_alive_t1, self.snap1.point
        if Factor.SPOT in s:
            env.spot_quote = deepcopy(e1.spot_quote)
        if Factor.VOL in s:
            env.vol_surface = e1.vol_surface
        elif Factor.SPOT in s and self.config.spot_convention is GreekConvention.STICKY_MONEYNESS:
            env.vol_surface = shocked_surface(e0.vol_surface, float(e0.spot), float(e1.spot),
                                              GreekConvention.STICKY_MONEYNESS)
        if Factor.RATE in s:
            env.rate_curve = e1.rate_curve
        if Factor.DIVIDEND in s:
            env.div_yield = e1.div_yield
        if Factor.BASIS in s:
            env.basis_yield = e1.basis_yield
        if Factor.MODEL in s:
            engine = self.bump_engine_alive
        return product, engine, env, point

    def value_for(self, applied: Iterable[Factor]) -> ValueBreakdown:
        key = self.normalize(applied)
        if key not in self._memo:
            product, engine, env, point = self.build_state(key)
            self._memo[key] = value(self.snap0, engine=engine, product=product,
                                    pricing_env=env, valuation_point=point)
        return self._memo[key]

    def all_market(self) -> ValueBreakdown:
        return self.value_for(MARKET_FACTORS)

    def value_t1(self) -> ValueBreakdown:
        if self._t1 is None:
            self._t1 = value(self.snap1, engine=self.bump_engine_t1)
        return self._t1

    def time_pure(self) -> float:
        return self.value_for((Factor.TIME,)).total - self.value_for(()).total
