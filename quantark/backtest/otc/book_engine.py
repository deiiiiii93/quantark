"""Per-underlying multi-product net-delta hedging backtest (autocallable lifecycle aware)."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import pandas as pd

from quantark.util.exceptions import ValidationError
from .config import AutocallableEngineConfig, FuturesRollPolicy, SurfaceGridConfig
from .market import AutocallableMarketDataSet, ImpliedBasisYield, SignedDividendYield
from .state import AutocallableDeltaHedgeStrategy, AutocallableLifecycleState, FuturesHedgePosition
from quantark.backtest.transaction_costs import TransactionCostModel, ZeroCostModel
from .engine_factory import create_pricing_engine, create_surface_engine, create_event_stats_engine
from ._replay import ProductReplay


@dataclass
class BookProduct:
    product: Any
    quantity: float
    position_id: int
    has_lifecycle: bool
    initial_price: Optional[float] = None

    def __post_init__(self):
        if self.product is None:
            raise ValidationError("BookProduct.product is required")
        if self.quantity == 0:
            raise ValidationError("BookProduct.quantity must be non-zero")


@dataclass
class HedgeSpec:
    kind: str = "futures"
    multiplier: float = 1.0
    roll_policy: Optional[FuturesRollPolicy] = None

    def __post_init__(self):
        if self.kind not in ("futures", "spot"):
            raise ValidationError(f"HedgeSpec.kind must be futures|spot, got {self.kind}")
        if self.kind == "futures" and self.roll_policy is None:
            self.roll_policy = FuturesRollPolicy()


@dataclass
class BookAutocallableBacktestConfig:
    products: list[BookProduct]
    market_data: AutocallableMarketDataSet
    hedge: HedgeSpec = field(default_factory=HedgeSpec)
    engine_config: AutocallableEngineConfig = field(default_factory=AutocallableEngineConfig)
    strategy: Any = None
    transaction_cost_model: TransactionCostModel = field(default_factory=ZeroCostModel)
    underlying: str = "equity_index"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    fixed_dividend_yield: Optional[float] = None
    delta_bump_size: Optional[float] = None
    gamma_bump_size: Optional[float] = None
    surface_config: SurfaceGridConfig = field(default_factory=SurfaceGridConfig)
    calculate_surfaces: bool = False
    calculate_event_probabilities: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.products:
            raise ValidationError("BookAutocallableBacktestConfig.products must be non-empty")
        if self.market_data is None:
            raise ValidationError("market_data is required")
        if self.strategy is None:
            self.strategy = AutocallableDeltaHedgeStrategy()


class BookAutocallableBacktestEngine:
    """
    Per-underlying multi-product net-delta hedging backtest engine.

    Mirrors :class:`AutocallableBacktestEngine` but loops over a *book* of
    products that share a single underlying (hence a single per-day
    ``PricingEnvironment``, which is product-independent) and a single futures
    (or spot) hedge.  Per-product work (lifecycle, pricing, greeks, event
    probabilities) runs once per product per day; hedging and daily recording
    run once per day on the aggregated net position.

    A book of a single product is byte-identical to the equivalent single
    :class:`AutocallableBacktestEngine` run.
    """

    def __init__(self, config: BookAutocallableBacktestConfig):
        self.config = config
        self.strategy = config.strategy
        self.hedge = config.hedge
        self.hedge_position = FuturesHedgePosition()

        self._states: list[dict[str, Any]] = []
        self._greeks: list[dict[str, Any]] = []
        self._rebalances: list[dict[str, Any]] = []
        self._trades: list[dict[str, Any]] = []
        self._actions: list[dict[str, Any]] = []
        self._surfaces: list[dict[str, Any]] = []
        self._daily_event_summary: list[dict[str, Any]] = []
        self._event_probabilities: list[dict[str, Any]] = []

        self._initial_book_value: Optional[float] = None
        self._transaction_costs: float = 0.0
        self._start_date: Optional[pd.Timestamp] = None

        self._replays: list[ProductReplay] = []
        self._quantities: list[float] = []
        for bp in config.products:
            lifecycle = AutocallableLifecycleState()
            pricing_engine = create_pricing_engine(bp.product, config.engine_config)
            surface_engine = create_surface_engine(bp.product, config.engine_config)
            event_stats_engine = create_event_stats_engine(
                bp.product, config.engine_config
            )
            replay = ProductReplay(
                product=bp.product,
                product_quantity=bp.quantity,
                lifecycle=lifecycle,
                pricing_engine=pricing_engine,
                surface_engine=surface_engine,
                event_stats_engine=event_stats_engine,
                engine_config=config.engine_config,
                market_data=config.market_data,
                start_date=None,
                underlying=config.underlying,
                fixed_dividend_yield=config.fixed_dividend_yield,
                delta_bump_size=config.delta_bump_size,
                gamma_bump_size=config.gamma_bump_size,
                surface_config=config.surface_config,
                actions_sink=self._actions,
                event_prob_sink=self._event_probabilities,
                daily_event_sink=self._daily_event_summary,
                surfaces_sink=self._surfaces,
            )
            self._replays.append(replay)
            self._quantities.append(float(bp.quantity))

    def run(self) -> "BookBacktestResults":
        dates = self._backtest_dates()
        if len(dates) == 0:
            raise ValidationError("No common market-data dates for backtest")
        self._start_date = pd.Timestamp(dates[0]).normalize()
        for replay in self._replays:
            replay.start_date = self._start_date

        current_contract: Optional[str] = None
        for date in dates:
            date = pd.Timestamp(date).normalize()
            market = self.config.market_data.get_market_row(date)

            if self.hedge.kind == "futures":
                futures_slice = self.config.market_data.get_futures_slice(date)
                selected = self.hedge.roll_policy.select_contract(
                    futures_slice, date, current_contract
                )
                if current_contract != str(selected["contract"]):
                    self._roll_contract(
                        date, selected, futures_slice, current_contract
                    )
                    current_contract = str(selected["contract"])
            else:
                selected = self._spot_selected(date, market)

            multiplier = float(selected["multiplier"])

            # env is product-independent: build it once from any replay.
            env, basis_yield, implied_q, futures_ttm = self._replays[0].build_env(
                date, market, selected
            )
            if self.hedge.kind == "spot":
                # For spot hedges, build_env synthesises a 100-year future
                # (futures_price == spot), which collapses implied_q ≈ rate.
                # That is economically wrong: override with an explicit dividend
                # yield so the pricer receives a dividend, not the risk-free rate.
                implied_q = (
                    float(self.config.fixed_dividend_yield)
                    if self.config.fixed_dividend_yield is not None
                    else 0.0
                )
                basis_yield = 0.0
                env.div_yield = SignedDividendYield(implied_q)
                env.basis_yield = ImpliedBasisYield(0.0)
            pricing_q = self._replays[0].pricing_dividend_yield(implied_q)

            # Initial book value: priced BEFORE lifecycle on the first day,
            # mirroring the single engine's pre-lifecycle initial value.
            if self._initial_book_value is None:
                initial_book_value = 0.0
                for bp, replay in zip(self.config.products, self._replays):
                    if bp.initial_price is not None:
                        initial_price = float(bp.initial_price)
                    else:
                        product = replay.product_for_date(date, env)
                        initial_price = float(replay.pricing_engine.price(product, env))
                    initial_book_value += float(bp.quantity) * initial_price
                self._initial_book_value = initial_book_value

            net_position_delta = 0.0
            net_position_gamma = 0.0
            book_product_mtm = 0.0
            book_cashflows = 0.0

            for quantity, replay in zip(self._quantities, self._replays):
                lifecycle_product = replay.product_for_lifecycle()
                replay.apply_lifecycle_events(
                    date, lifecycle_product, env, market["spot"]
                )
                replay.settle_maturity_if_due(
                    date, lifecycle_product, env, market["spot"]
                )

                if replay.lifecycle.alive:
                    product = replay.product_for_date(date, env)
                    price = float(replay.pricing_engine.price(product, env))
                    greeks = replay.calculate_greeks(product, env, price)
                    if self.config.calculate_event_probabilities:
                        replay.record_event_probabilities(date, product, env)
                    if self.config.calculate_surfaces:
                        replay.record_surfaces(
                            date,
                            product,
                            env,
                            market["spot"],
                            replay.pricing_dividend_yield(implied_q),
                        )
                    net_position_delta += float(greeks.get("delta", 0.0)) * quantity
                    net_position_gamma += float(greeks.get("gamma", 0.0)) * quantity
                    book_product_mtm += quantity * price

                book_cashflows += replay.lifecycle.realized_cashflows

            any_alive = any(replay.lifecycle.alive for replay in self._replays)

            pre_hedge_contracts = self.hedge_position.quantity
            self._rebalance(
                date, selected, net_position_delta, multiplier, any_alive
            )
            self._record_day(
                date=date,
                selected=selected,
                market=market,
                basis_yield=basis_yield,
                implied_q=implied_q,
                pricing_q=pricing_q,
                futures_ttm=futures_ttm,
                net_position_delta=net_position_delta,
                net_position_gamma=net_position_gamma,
                book_product_mtm=book_product_mtm,
                book_cashflows=book_cashflows,
                pre_hedge_contracts=pre_hedge_contracts,
                multiplier=multiplier,
                any_alive=any_alive,
            )

        return BookBacktestResults(
            config=self.config,
            states=self._states,
            greeks=self._greeks,
            rebalances=self._rebalances,
            trades=self._trades,
            actions=self._actions,
            daily_event_summary=self._daily_event_summary,
            event_probabilities=self._event_probabilities,
            surfaces=self._surfaces,
            products_meta=[
                {
                    "position_id": bp.position_id,
                    "underlying": self.config.underlying,
                    "has_lifecycle": bp.has_lifecycle,
                    "quantity": bp.quantity,
                }
                for bp in self.config.products
            ],
        )

    def _backtest_dates(self) -> pd.DatetimeIndex:
        dates = self.config.market_data.dates
        if self.config.start_date is not None:
            dates = dates[dates >= pd.Timestamp(self.config.start_date).normalize()]
        if self.config.end_date is not None:
            dates = dates[dates <= pd.Timestamp(self.config.end_date).normalize()]
        return dates

    def _spot_selected(self, date: pd.Timestamp, market: dict[str, float]) -> dict[str, Any]:
        """Synthesize a contract-like row for spot hedging."""
        return {
            "contract": "SPOT",
            "futures_price": float(market["spot"]),
            "multiplier": float(self.hedge.multiplier),
            "expiry_date": pd.Timestamp(date).normalize() + pd.Timedelta(days=36500),
        }

    def _rebalance(
        self,
        date: pd.Timestamp,
        selected,
        net_position_delta: float,
        multiplier: float,
        any_alive: bool,
    ) -> None:
        is_spot = self.hedge.kind == "spot"
        target = 0.0
        reason = "inside_band"
        trade_type = "hedge_rebalance"
        if any_alive:
            if is_spot:
                # Spot mode: always target full delta-neutral; strategy.hedge_ratio
                # and target_delta are not applied — only delta_threshold (band) is.
                target = -float(net_position_delta)
            else:
                target = self.strategy.target_contracts(
                    product_delta=float(net_position_delta),
                    product_quantity=1.0,
                    futures_multiplier=multiplier,
                )
            should_rebalance = self.strategy.should_rebalance(
                self.hedge_position.quantity, target
            )
            if should_rebalance:
                reason = "delta_rebalance"
        else:
            should_rebalance = abs(self.hedge_position.quantity) > 1e-12
            if should_rebalance:
                reason = "product_terminated"
                trade_type = "hedge_close"

        trade_contracts = target - self.hedge_position.quantity
        if should_rebalance and abs(trade_contracts) > 1e-12:
            self._execute_hedge_trade(
                date=date,
                selected=selected,
                quantity_delta=trade_contracts,
                trade_type=trade_type,
                reason=reason,
                instrument_type="spot" if is_spot else "futures",
            )

        self._rebalances.append(
            {
                "date": date,
                "active_contract": str(selected["contract"]),
                "current_contracts": self.hedge_position.quantity,
                "target_contracts": target,
                "trade_contracts": trade_contracts if should_rebalance else 0.0,
                "should_rebalance": should_rebalance,
                "threshold_status": (
                    "outside_band" if should_rebalance else "inside_band"
                ),
                "no_trade_reason": None if should_rebalance else "inside_band",
                "reason": reason,
            }
        )

    def _roll_contract(
        self,
        date: pd.Timestamp,
        selected,
        futures_slice: pd.DataFrame,
        current_contract: Optional[str],
    ) -> None:
        if self.hedge.kind == "spot":
            return
        if abs(self.hedge_position.quantity) < 1e-12:
            return
        old_contract = self.hedge_position.contract or current_contract
        if old_contract is None:
            return
        old_rows = futures_slice[futures_slice["contract"] == old_contract]
        close_reason = "futures_roll"
        if old_rows.empty:
            old = selected.copy()
            old["contract"] = old_contract
            close_reason = "futures_roll_missing_old_contract"
        else:
            old = old_rows.iloc[0]
        qty = self.hedge_position.quantity
        self._execute_hedge_trade(
            date=date,
            selected=old,
            quantity_delta=-qty,
            trade_type="roll_close",
            reason=close_reason,
            instrument_type="futures",
        )
        self._execute_hedge_trade(
            date=date,
            selected=selected,
            quantity_delta=qty,
            trade_type="roll_open",
            reason=close_reason,
            instrument_type="futures",
        )

    def _execute_hedge_trade(
        self,
        *,
        date: pd.Timestamp,
        selected,
        quantity_delta: float,
        trade_type: str,
        reason: str,
        instrument_type: str = "futures",
    ) -> None:
        price = float(selected["futures_price"])
        multiplier = float(selected["multiplier"])
        contract = str(selected["contract"])
        notional = abs(float(quantity_delta) * price * multiplier)
        cost = self.config.transaction_cost_model.calculate_cost(
            quantity=float(quantity_delta),
            price=price,
            notional=notional,
            instrument_type=instrument_type,
            trade_type=trade_type,
        )
        self.hedge_position.trade(quantity_delta, price, contract, multiplier)
        self._transaction_costs += float(cost)
        self._trades.append(
            {
                "date": date,
                "trade_type": trade_type,
                "instrument_type": instrument_type,
                "contract": contract,
                "quantity": float(quantity_delta),
                "price": price,
                "multiplier": multiplier,
                "notional": notional,
                "transaction_cost": float(cost),
                "reason": reason,
            }
        )

    def _record_day(
        self,
        *,
        date: pd.Timestamp,
        selected,
        market: dict[str, float],
        basis_yield: float,
        implied_q: float,
        pricing_q: float,
        futures_ttm: float,
        net_position_delta: float,
        net_position_gamma: float,
        book_product_mtm: float,
        book_cashflows: float,
        pre_hedge_contracts: float,
        multiplier: float,
        any_alive: bool,
    ) -> None:
        futures_price = float(selected["futures_price"])
        spot = float(market["spot"])
        hedge_mtm = self.hedge_position.mark_to_market(futures_price)
        # book_product_mtm already only sums alive products.
        product_mtm = book_product_mtm
        product_pnl = (
            product_mtm + book_cashflows - float(self._initial_book_value or 0.0)
        )
        total_pnl = product_pnl + hedge_mtm - self._transaction_costs
        cash = book_cashflows - self._transaction_costs
        portfolio_value = product_mtm + hedge_mtm + cash
        product_position_delta = float(net_position_delta)
        product_position_gamma = float(net_position_gamma)
        pre_hedge_futures_delta = float(pre_hedge_contracts) * multiplier
        post_hedge_futures_delta = self.hedge_position.quantity * multiplier
        pre_hedge_delta = product_position_delta + pre_hedge_futures_delta
        post_hedge_delta = product_position_delta + post_hedge_futures_delta
        pre_hedge_gamma = product_position_gamma
        post_hedge_gamma = product_position_gamma
        one_percent_spot_move = spot * 0.01
        pre_hedge_delta_cash_1pct = pre_hedge_delta * one_percent_spot_move
        post_hedge_delta_cash_1pct = post_hedge_delta * one_percent_spot_move
        pre_hedge_gamma_cash_1pct = pre_hedge_gamma * spot**2 / 100.0
        post_hedge_gamma_cash_1pct = post_hedge_gamma * spot**2 / 100.0

        # Book-level aggregates: alive=ANY product alive (hedge-control field);
        # knocked_out/matured=ALL products in that terminal state (book is only
        # fully exited when every leg has terminated).
        knocked_in = any(r.lifecycle.knocked_in for r in self._replays)
        knocked_out = all(r.lifecycle.knocked_out for r in self._replays)
        matured = all(r.lifecycle.matured for r in self._replays)

        self._states.append(
            {
                "date": date,
                "portfolio_value": portfolio_value,
                "product_mtm": product_mtm,
                "hedge_mtm": hedge_mtm,
                "cash": cash,
                "cashflows": book_cashflows,
                "transaction_costs": self._transaction_costs,
                "product_pnl": product_pnl,
                "hedge_pnl": hedge_mtm,
                "total_pnl": total_pnl,
                "spot": market["spot"],
                "volatility": market["volatility"],
                "rate": market["rate"],
                "basis_yield": basis_yield,
                "implied_q": implied_q,
                "pricing_q": pricing_q,
                "active_contract": str(selected["contract"]),
                "futures_price": futures_price,
                "futures_ttm": futures_ttm,
                "futures_multiplier": multiplier,
                "futures_contracts": self.hedge_position.quantity,
                "alive": any_alive,
                "knocked_in": knocked_in,
                "knocked_out": knocked_out,
                "matured": matured,
            }
        )
        self._greeks.append(
            {
                "date": date,
                "delta": product_position_delta,
                "gamma": product_position_gamma,
                "product_position_delta": product_position_delta,
                "product_position_gamma": product_position_gamma,
                "pre_hedge_contracts": float(pre_hedge_contracts),
                "post_hedge_contracts": self.hedge_position.quantity,
                "futures_multiplier": multiplier,
                "pre_hedge_futures_delta": pre_hedge_futures_delta,
                "post_hedge_futures_delta": post_hedge_futures_delta,
                "pre_hedge_delta": pre_hedge_delta,
                "post_hedge_delta": post_hedge_delta,
                "pre_hedge_gamma": pre_hedge_gamma,
                "post_hedge_gamma": post_hedge_gamma,
                "pre_hedge_delta_cash_1pct": pre_hedge_delta_cash_1pct,
                "post_hedge_delta_cash_1pct": post_hedge_delta_cash_1pct,
                "pre_hedge_gamma_cash_1pct": pre_hedge_gamma_cash_1pct,
                "post_hedge_gamma_cash_1pct": post_hedge_gamma_cash_1pct,
                "delta_cash_1pct": post_hedge_delta_cash_1pct,
                "gamma_cash_1pct": post_hedge_gamma_cash_1pct,
            }
        )


class BookBacktestResults:
    def __init__(self, *, config, states, greeks, rebalances, trades, actions,
                 daily_event_summary, event_probabilities, surfaces, products_meta):
        self.config = config
        self._states = states
        self._greeks = greeks
        self._rebalances = rebalances
        self._trades = trades
        self._actions = actions
        self._daily_event_summary = daily_event_summary
        self._event_probabilities = event_probabilities
        self._surfaces = surfaces
        self._products_meta = products_meta

    @staticmethod
    def _frame(rows, index=None):
        df = pd.DataFrame(rows)
        if index and not df.empty:
            df = df.set_index(index)
        return df

    def states_df(self): return self._frame(self._states)
    def greeks_df(self): return self._frame(self._greeks)
    def rebalances_df(self): return self._frame(self._rebalances)
    def trades_df(self): return self._frame(self._trades)
    def actions_df(self): return self._frame(self._actions)
    def daily_event_summary_df(self): return self._frame(self._daily_event_summary)
    def event_probability_df(self): return self._frame(self._event_probabilities)
    def surfaces_df(self): return self._frame(self._surfaces)

    def get_summary(self):
        states = self.states_df()
        if states.empty:
            return {"num_days": 0, "num_trades": len(self._trades), "total_pnl": 0.0,
                    "num_products": len(self._products_meta), "num_lifecycle_events": len(self._actions)}
        return {
            "num_days": int(len(states)),
            "start_date": str(states["date"].iloc[0]),
            "end_date": str(states["date"].iloc[-1]),
            "initial_portfolio_value": float(states["portfolio_value"].iloc[0]),
            "final_portfolio_value": float(states["portfolio_value"].iloc[-1]),
            "total_pnl": float(states["total_pnl"].iloc[-1]),
            "product_pnl": float(states["product_pnl"].iloc[-1]),
            "hedge_pnl": float(states["hedge_pnl"].iloc[-1]),
            "transaction_costs": float(states["transaction_costs"].iloc[-1]),
            "num_trades": int(len(self._trades)),
            "num_products": len(self._products_meta),
            "num_lifecycle_events": len(self._actions),
        }
