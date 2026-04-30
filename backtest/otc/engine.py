"""
Daily OTC autocallable futures-hedging backtest engine.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd

from asset.equity.product.option.phoenix_option import PhoenixOption
from asset.equity.product.option.snowball_option import SnowballOption
from asset.equity.engine.base_engine import BaseEngine
from param import FlatRateCurve, FlatVolSurface, SpotQuote
from priceenv import PricingEnvironment
from util.exceptions import ValidationError

from .config import AutocallableBacktestConfig
from .engine_factory import (
    create_event_stats_engine,
    create_mc_event_stats_engine,
    create_pricing_engine,
    create_surface_engine,
)
from .market import (
    ImpliedBasisYield,
    SignedDividendYield,
    derive_implied_dividend_yield,
)
from .results import AutocallableBacktestResults
from .state import (
    AutocallableDeltaHedgeStrategy,
    AutocallableLifecycleState,
    FuturesHedgePosition,
)


class AutocallableBacktestEngine:
    """
    Historical replay engine for Snowball/Phoenix delta hedging with futures.
    """

    def __init__(self, config: AutocallableBacktestConfig):
        self.config = config
        self.lifecycle = AutocallableLifecycleState()
        self.strategy = config.strategy or AutocallableDeltaHedgeStrategy()
        self.pricing_engine = create_pricing_engine(
            config.product, config.engine_config
        )
        self.surface_engine = create_surface_engine(
            config.product, config.engine_config
        )
        self.event_stats_engine = create_event_stats_engine(
            config.product, config.engine_config
        )
        self.hedge_position = FuturesHedgePosition()

        self._states: list[dict[str, Any]] = []
        self._greeks: list[dict[str, Any]] = []
        self._rebalances: list[dict[str, Any]] = []
        self._trades: list[dict[str, Any]] = []
        self._actions: list[dict[str, Any]] = []
        self._surfaces: list[dict[str, Any]] = []
        self._daily_event_summary: list[dict[str, Any]] = []
        self._event_probabilities: list[dict[str, Any]] = []

        self._initial_product_value: Optional[float] = None
        self._transaction_costs: float = 0.0
        self._start_date: Optional[pd.Timestamp] = None

    def run(self) -> AutocallableBacktestResults:
        dates = self._backtest_dates()
        if len(dates) == 0:
            raise ValidationError("No common market-data dates for backtest")
        self._start_date = pd.Timestamp(dates[0]).normalize()

        current_contract: Optional[str] = None
        for date in dates:
            date = pd.Timestamp(date).normalize()
            market = self.config.market_data.get_market_row(date)
            futures_slice = self.config.market_data.get_futures_slice(date)
            selected = self.config.roll_policy.select_contract(
                futures_slice, date, current_contract
            )
            if current_contract != str(selected["contract"]):
                self._roll_contract(date, selected, futures_slice, current_contract)
                current_contract = str(selected["contract"])

            env, basis_yield, implied_q, futures_ttm = self._build_env(
                date, market, selected
            )
            product = self._product_for_date(date, env)

            if self._initial_product_value is None:
                initial_price = (
                    float(self.config.initial_product_price)
                    if self.config.initial_product_price is not None
                    else float(self.pricing_engine.price(product, env))
                )
                self._initial_product_value = (
                    self.config.product_quantity * initial_price
                )

            lifecycle_product = self._product_for_lifecycle()
            self._apply_lifecycle_events(date, lifecycle_product, env, market["spot"])
            self._settle_maturity_if_due(date, lifecycle_product, env, market["spot"])
            product = self._product_for_date(date, env)

            price = 0.0
            greeks = {"price": 0.0, "delta": 0.0, "gamma": 0.0}
            if self.lifecycle.alive:
                price = float(self.pricing_engine.price(product, env))
                greeks = self._calculate_greeks(product, env, price)

            if self.config.calculate_event_probabilities and self.lifecycle.alive:
                self._record_event_probabilities(date, product, env)

            if self.config.calculate_surfaces and self.lifecycle.alive:
                self._record_surfaces(
                    date,
                    product,
                    env,
                    market["spot"],
                    self._pricing_dividend_yield(implied_q),
                )

            pre_hedge_contracts = self.hedge_position.quantity
            self._rebalance(date, selected, greeks)
            self._record_day(
                date=date,
                selected=selected,
                market=market,
                basis_yield=basis_yield,
                implied_q=implied_q,
                pricing_q=self._pricing_dividend_yield(implied_q),
                futures_ttm=futures_ttm,
                price=price,
                greeks=greeks,
                pre_hedge_contracts=pre_hedge_contracts,
            )

        return AutocallableBacktestResults(
            config=self.config,
            states=self._states,
            greeks=self._greeks,
            rebalances=self._rebalances,
            trades=self._trades,
            actions=self._actions,
            surfaces=self._surfaces,
            daily_event_summary=self._daily_event_summary,
            event_probabilities=self._event_probabilities,
        )

    def _backtest_dates(self) -> pd.DatetimeIndex:
        dates = self.config.market_data.dates
        if self.config.start_date is not None:
            dates = dates[dates >= pd.Timestamp(self.config.start_date).normalize()]
        if self.config.end_date is not None:
            dates = dates[dates <= pd.Timestamp(self.config.end_date).normalize()]
        return dates

    def _product_for_lifecycle(self):
        product = deepcopy(self.config.product)
        setattr(product, "_otc_lifecycle_knocked_in", self.lifecycle.knocked_in)
        return product

    def _product_for_date(self, date: pd.Timestamp, pricing_env: PricingEnvironment):
        product = deepcopy(self.config.product)
        setattr(product, "_otc_lifecycle_knocked_in", self.lifecycle.knocked_in)
        if (
            getattr(product, "exercise_date", None) is None
            and getattr(product, "maturity", None) is not None
            and self._start_date is not None
        ):
            elapsed = max(0.0, (date - self._start_date).days / 365.0)
            product.maturity = max(float(product.maturity) - elapsed, 1e-8)
        elif self._start_date is not None:
            elapsed = max(0.0, (date - self._start_date).days / 365.0)
        else:
            elapsed = 0.0
        barrier_config = getattr(product, "barrier_config", None)
        if barrier_config is not None and hasattr(barrier_config, "time_shift"):
            shifted_config, dropped_all = barrier_config.time_shift(
                elapsed,
                date.to_pydatetime(),
                pricing_env,
            )
            if shifted_config is not None and not dropped_all:
                product.barrier_config = shifted_config
        return product

    def _build_env(self, date: pd.Timestamp, market: dict[str, float], selected):
        expiry = pd.Timestamp(selected["expiry_date"]).normalize()
        futures_ttm = (expiry - date).days / 365.0
        basis_yield, implied_q = derive_implied_dividend_yield(
            rate=market["rate"],
            spot=market["spot"],
            futures_price=float(selected["futures_price"]),
            time_to_maturity=futures_ttm,
        )
        pricing_q = self._pricing_dividend_yield(implied_q)
        env = PricingEnvironment(
            spot_quote=SpotQuote(spot=market["spot"], asset_name=self.config.underlying),
            vol_surface=FlatVolSurface(volatility=market["volatility"]),
            rate_curve=FlatRateCurve(rate=market["rate"]),
            div_yield=SignedDividendYield(pricing_q),
            basis_yield=ImpliedBasisYield(basis_yield),
            valuation_date=date.to_pydatetime(),
        )
        return env, basis_yield, implied_q, futures_ttm

    def _pricing_dividend_yield(self, implied_q: float) -> float:
        if self.config.fixed_dividend_yield is not None:
            return float(self.config.fixed_dividend_yield)
        return float(implied_q)

    def _calculate_greeks(
        self, product: Any, env: PricingEnvironment, price: float
    ) -> dict[str, float]:
        params = getattr(self.pricing_engine, "params", None)
        uses_base_greeks = (
            isinstance(self.pricing_engine, BaseEngine)
            and type(self.pricing_engine).calculate_greeks is BaseEngine.calculate_greeks
        )
        bump = float(getattr(params, "bump_size", 1e-4)) if params is not None else 0.0
        if uses_base_greeks and bump > 0.0 and np.isfinite(price):
            try:
                env_up = deepcopy(env)
                env_up.spot_quote.spot *= 1.0 + bump
                price_up = float(self.pricing_engine.price(product, env_up))

                env_down = deepcopy(env)
                env_down.spot_quote.spot *= 1.0 - bump
                price_down = float(self.pricing_engine.price(product, env_down))

                spot_bump = float(env.spot) * bump
                delta = (price_up - price_down) / (2.0 * spot_bump)
                gamma = (price_up - 2.0 * float(price) + price_down) / (spot_bump**2)
                return {"price": float(price), "delta": delta, "gamma": gamma}
            except Exception:
                pass
        try:
            greeks = dict(self.pricing_engine.calculate_greeks(product, env))
        except Exception:
            greeks = {"price": price, "delta": 0.0, "gamma": 0.0}
        greeks.setdefault("price", price)
        greeks.setdefault("delta", 0.0)
        greeks.setdefault("gamma", 0.0)
        return greeks

    def _rebalance(self, date: pd.Timestamp, selected, greeks: dict[str, float]) -> None:
        target = 0.0
        reason = "inside_band"
        trade_type = "hedge_rebalance"
        if self.lifecycle.alive:
            target = self.strategy.target_contracts(
                product_delta=float(greeks.get("delta", 0.0)),
                product_quantity=self.config.product_quantity,
                futures_multiplier=float(selected["multiplier"]),
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
            self._execute_futures_trade(
                date=date,
                selected=selected,
                quantity_delta=trade_contracts,
                trade_type=trade_type,
                reason=reason,
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
        self._execute_futures_trade(
            date=date,
            selected=old,
            quantity_delta=-qty,
            trade_type="roll_close",
            reason=close_reason,
        )
        self._execute_futures_trade(
            date=date,
            selected=selected,
            quantity_delta=qty,
            trade_type="roll_open",
            reason=close_reason,
        )

    def _execute_futures_trade(
        self,
        *,
        date: pd.Timestamp,
        selected,
        quantity_delta: float,
        trade_type: str,
        reason: str,
    ) -> None:
        price = float(selected["futures_price"])
        multiplier = float(selected["multiplier"])
        contract = str(selected["contract"])
        notional = abs(float(quantity_delta) * price * multiplier)
        cost = self.config.transaction_cost_model.calculate_cost(
            quantity=float(quantity_delta),
            price=price,
            notional=notional,
            instrument_type="futures",
            trade_type=trade_type,
        )
        self.hedge_position.trade(quantity_delta, price, contract, multiplier)
        self._transaction_costs += float(cost)
        self._trades.append(
            {
                "date": date,
                "trade_type": trade_type,
                "instrument_type": "futures",
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
        price: float,
        greeks: dict[str, float],
        pre_hedge_contracts: float,
    ) -> None:
        futures_price = float(selected["futures_price"])
        spot = float(market["spot"])
        multiplier = float(selected["multiplier"])
        hedge_mtm = self.hedge_position.mark_to_market(futures_price)
        product_mtm = (
            self.config.product_quantity * price if self.lifecycle.alive else 0.0
        )
        product_pnl = (
            product_mtm
            + self.lifecycle.realized_cashflows
            - float(self._initial_product_value or 0.0)
        )
        total_pnl = product_pnl + hedge_mtm - self._transaction_costs
        cash = self.lifecycle.realized_cashflows - self._transaction_costs
        portfolio_value = product_mtm + hedge_mtm + cash
        product_delta = float(greeks.get("delta", 0.0))
        product_gamma = float(greeks.get("gamma", 0.0))
        product_position_delta = product_delta * self.config.product_quantity
        product_position_gamma = product_gamma * self.config.product_quantity
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

        self._states.append(
            {
                "date": date,
                "portfolio_value": portfolio_value,
                "product_mtm": product_mtm,
                "hedge_mtm": hedge_mtm,
                "cash": cash,
                "cashflows": self.lifecycle.realized_cashflows,
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
                "alive": self.lifecycle.alive,
                "knocked_in": self.lifecycle.knocked_in,
                "knocked_out": self.lifecycle.knocked_out,
                "matured": self.lifecycle.matured,
            }
        )
        self._greeks.append(
            {
                "date": date,
                "price": float(greeks.get("price", price)),
                "delta": float(greeks.get("delta", 0.0)),
                "gamma": float(greeks.get("gamma", 0.0)),
                "product_delta": product_delta,
                "product_gamma": product_gamma,
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
                "vega": float(greeks.get("vega", np.nan)),
                "theta": float(greeks.get("theta", np.nan)),
                "rho": float(greeks.get("rho", np.nan)),
                "dividend_sensitivity": float(
                    greeks.get("dividend_sensitivity", np.nan)
                ),
                "basis_sensitivity": float(greeks.get("basis_sensitivity", np.nan)),
            }
        )

    def _record_surfaces(
        self,
        date: pd.Timestamp,
        product: Any,
        env: PricingEnvironment,
        spot: float,
        q_center: float,
    ) -> None:
        spec = self.config.surface_config
        spot_grid = np.linspace(
            spot * (1.0 - spec.spot_width),
            spot * (1.0 + spec.spot_width),
            spec.spot_nodes,
        )
        if spec.q_nodes == 1:
            q_grid = np.array([q_center], dtype=float)
        else:
            q_lower = max(0.0, q_center - spec.q_width)
            q_upper = max(0.0, q_center + spec.q_width)
            q_grid = np.linspace(
                q_lower, q_upper, spec.q_nodes
            )

        for s in spot_grid:
            for q in q_grid:
                surf_env = deepcopy(env)
                surf_env.spot_quote = SpotQuote(spot=float(s), asset_name=self.config.underlying)
                surf_env.div_yield = SignedDividendYield(float(q))
                try:
                    greeks = self.surface_engine.calculate_greeks(product, surf_env)
                    delta = float(greeks.get("delta", np.nan))
                    gamma = float(greeks.get("gamma", np.nan))
                    spot_node = float(s)
                    one_percent_node_move = spot_node * 0.01
                    row = {
                        "date": date,
                        "surface_type": "spot_q",
                        "spot_node": float(s),
                        "q_node": float(q),
                        "price": float(greeks.get("price", np.nan)),
                        "delta": delta,
                        "gamma": gamma,
                        "delta_cash_1pct": delta * one_percent_node_move,
                        "gamma_cash_1pct": gamma * spot_node**2 / 100.0,
                    }
                except Exception:
                    row = {
                        "date": date,
                        "surface_type": "spot_q",
                        "spot_node": float(s),
                        "q_node": float(q),
                        "price": np.nan,
                        "delta": np.nan,
                        "gamma": np.nan,
                        "delta_cash_1pct": np.nan,
                        "gamma_cash_1pct": np.nan,
                    }
                self._surfaces.append(row)

    def _record_event_probabilities(
        self, date: pd.Timestamp, product: Any, env: PricingEnvironment
    ) -> None:
        stats = self._calculate_event_stats(product, env)
        if stats is None:
            return

        ko_probs = np.asarray(getattr(stats, "ko_probability", []), dtype=float)
        ko_times = np.asarray(getattr(stats, "ko_times", []), dtype=float)
        survival = np.asarray(getattr(stats, "survival_probability", []), dtype=float)
        ed_ko_cf = np.asarray(
            getattr(stats, "expected_discounted_ko_cashflow", []), dtype=float
        )

        next_ko_prob = float(ko_probs[0]) if ko_probs.size else 0.0
        next_ko_date = (
            self._date_from_time(date, float(ko_times[0])) if ko_times.size else pd.NaT
        )
        ki_prob_scalar = float(getattr(stats, "ki_probability", 0.0))
        self._daily_event_summary.append(
            {
                "date": date,
                "next_ko_date": next_ko_date,
                "next_ko_probability": next_ko_prob,
                "total_remaining_ko_probability": float(np.nansum(ko_probs)),
                "ki_probability_to_maturity": ki_prob_scalar,
                "survival_probability": float(survival[-1]) if survival.size else np.nan,
                "expected_discounted_ko_cashflow": float(np.nansum(ed_ko_cf)),
                "expected_discounted_maturity_cashflow": float(
                    getattr(stats, "expected_discounted_maturity_cashflow", np.nan)
                ),
                "pv": float(getattr(stats, "pv", np.nan)),
            }
        )

        previous_survival = 1.0
        for i, probability in enumerate(ko_probs):
            conditional = probability / previous_survival if previous_survival > 0 else np.nan
            event_date = self._date_from_time(date, float(ko_times[i]))
            self._event_probabilities.append(
                {
                    "date": date,
                    "event_date": event_date,
                    "event_type": "KO",
                    "event_probability": float(probability),
                    "conditional_probability": float(conditional),
                    "survival_probability": float(survival[i]) if i < survival.size else np.nan,
                    "expected_discounted_cashflow": float(ed_ko_cf[i]) if i < ed_ko_cf.size else np.nan,
                }
            )
            if i < survival.size:
                previous_survival = float(survival[i])

        ki_times = np.asarray(getattr(stats, "ki_times", []), dtype=float)
        ki_event_prob = np.asarray(
            getattr(stats, "ki_event_probability", []), dtype=float
        )
        ki_survival = np.asarray(
            getattr(stats, "ki_survival_probability", []), dtype=float
        )
        if ki_times.size == 0 and ki_prob_scalar > 0:
            ki_times = np.array([product.get_maturity(env)], dtype=float)
            ki_event_prob = np.array([ki_prob_scalar], dtype=float)
            ki_survival = np.array([np.nan], dtype=float)

        for i, probability in enumerate(ki_event_prob):
            event_date = self._date_from_time(date, float(ki_times[i]))
            self._event_probabilities.append(
                {
                    "date": date,
                    "event_date": event_date,
                    "event_type": "KI",
                    "event_probability": float(probability),
                    "conditional_probability": np.nan,
                    "survival_probability": float(ki_survival[i]) if i < ki_survival.size else np.nan,
                    "expected_discounted_cashflow": np.nan,
                }
            )

    def _calculate_event_stats(self, product: Any, env: PricingEnvironment):
        try:
            stats = self.event_stats_engine.calculate_event_stats(product, env)
            if stats is not None:
                return stats
        except Exception:
            pass
        try:
            fallback = create_mc_event_stats_engine(product, self.config.engine_config)
            return fallback.calculate_event_stats(product, env)
        except Exception:
            return None

    def _lifecycle_snapshot(self) -> dict[str, Any]:
        return {
            "alive": self.lifecycle.alive,
            "knocked_in": self.lifecycle.knocked_in,
            "knocked_out": self.lifecycle.knocked_out,
            "matured": self.lifecycle.matured,
        }

    def _append_action(
        self,
        *,
        before_state: dict[str, Any],
        action_type: str,
        date: pd.Timestamp,
        observation_index: Optional[int],
        spot: float,
        barrier: Optional[float],
        cashflow: float,
        **extra: Any,
    ) -> None:
        after_state = self._lifecycle_snapshot()
        row = {
            "date": date,
            "action_type": action_type,
            "observation_index": observation_index,
            "spot": spot,
            "barrier": barrier,
            "cashflow": cashflow,
            "alive_before": before_state["alive"],
            "knocked_in_before": before_state["knocked_in"],
            "knocked_out_before": before_state["knocked_out"],
            "matured_before": before_state["matured"],
            "alive_after": after_state["alive"],
            "knocked_in_after": after_state["knocked_in"],
            "knocked_out_after": after_state["knocked_out"],
            "matured_after": after_state["matured"],
        }
        row.update(extra)
        self._actions.append(row)

    def _maturity_market_date(self, product: Any, env: PricingEnvironment) -> pd.Timestamp:
        explicit = (
            getattr(product, "maturity_date", None)
            or getattr(product, "exercise_date", None)
        )
        if explicit is not None:
            return self._next_available_market_date(pd.Timestamp(explicit).normalize())

        maturity = float(product.get_maturity(env))
        base_date = pd.Timestamp(getattr(product, "initial_date", None) or self._start_date)
        maturity_date = (
            base_date + timedelta(days=int(round(maturity * 365.0)))
        ).normalize()
        return self._next_available_market_date(maturity_date)

    def _settle_maturity_if_due(
        self, date: pd.Timestamp, product: Any, env: PricingEnvironment, spot: float
    ) -> None:
        if not self.lifecycle.alive:
            return
        if date < self._maturity_market_date(product, env):
            return

        before = self._lifecycle_snapshot()
        payoff = float(
            product.get_payoff(
                spot,
                env,
                knocked_in=self.lifecycle.knocked_in,
            )
        )
        cashflow = self.config.product_quantity * payoff
        if self.lifecycle.mark_maturity(date.to_pydatetime(), cashflow):
            self._append_action(
                before_state=before,
                action_type="MATURITY",
                date=date,
                observation_index=None,
                spot=spot,
                barrier=None,
                cashflow=cashflow,
                payoff=payoff,
            )

    def _apply_lifecycle_events(
        self, date: pd.Timestamp, product: Any, env: PricingEnvironment, spot: float
    ) -> None:
        if not self.lifecycle.alive:
            return
        ko_records = self._scheduled_records(product, env, "ko")
        ko_disabled_after_ki = bool(
            self.lifecycle.knocked_in
            and getattr(product.barrier_config, "disable_ko_after_ki", False)
        )
        if not ko_disabled_after_ki:
            for idx, rec in enumerate(ko_records):
                if idx in self.lifecycle.observed_ko_indices:
                    continue
                if date < rec["date"]:
                    continue
                self.lifecycle.observed_ko_indices.add(idx)
                if self._barrier_hit(spot, rec["barrier"], product.is_reverse, is_ko=True):
                    before = self._lifecycle_snapshot()
                    cashflow = self.config.product_quantity * float(rec.get("payoff", 0.0))
                    if self.lifecycle.mark_ko(date.to_pydatetime(), cashflow):
                        self._append_action(
                            before_state=before,
                            action_type="KO",
                            date=date,
                            observation_index=idx,
                            spot=spot,
                            barrier=rec["barrier"],
                            cashflow=cashflow,
                            payoff=float(rec.get("payoff", 0.0)),
                        )
                    return

        ki_records = self._scheduled_records(product, env, "ki")
        ki_observation_type = getattr(product.barrier_config, "ki_observation_type", None)
        ki_continuous = getattr(product, "has_ki_barrier", False) and (
            product.barrier_config.ki_continuous
            or getattr(ki_observation_type, "name", None) == "CONTINUOUS"
        )
        if ki_continuous:
            barrier = product.barrier_config.ki_barrier
            if isinstance(barrier, list):
                barrier = barrier[0]
            if self._barrier_hit(spot, float(barrier), product.is_reverse, is_ko=False):
                before = self._lifecycle_snapshot()
                if self.lifecycle.mark_ki(date.to_pydatetime()):
                    self._append_action(
                        before_state=before,
                        action_type="KI",
                        date=date,
                        observation_index=None,
                        spot=spot,
                        barrier=float(barrier),
                        cashflow=0.0,
                        monitoring="daily_close",
                    )
        else:
            for idx, rec in enumerate(ki_records):
                if idx in self.lifecycle.observed_ki_indices:
                    continue
                if date < rec["date"]:
                    continue
                self.lifecycle.observed_ki_indices.add(idx)
                if self._barrier_hit(spot, rec["barrier"], product.is_reverse, is_ko=False):
                    before = self._lifecycle_snapshot()
                    if self.lifecycle.mark_ki(date.to_pydatetime()):
                        self._append_action(
                            before_state=before,
                            action_type="KI",
                            date=date,
                            observation_index=idx,
                            spot=spot,
                            barrier=rec["barrier"],
                            cashflow=0.0,
                        )

        if isinstance(product, PhoenixOption):
            for idx, rec in enumerate(ko_records):
                if idx in self.lifecycle.observed_coupon_indices:
                    continue
                if date < rec["date"]:
                    continue
                self.lifecycle.observed_coupon_indices.add(idx)
                if product.is_coupon_triggered(spot, idx):
                    before = self._lifecycle_snapshot()
                    coupon = self.config.product_quantity * product.get_coupon_payoff(idx)
                    self.lifecycle.add_cashflow(coupon)
                    self.lifecycle.coupon_memory_count = 0
                    self._append_action(
                        before_state=before,
                        action_type="COUPON",
                        date=date,
                        observation_index=idx,
                        spot=spot,
                        barrier=product.get_coupon_barrier_at(idx),
                        cashflow=coupon,
                    )
                elif product.has_memory_coupon:
                    self.lifecycle.coupon_memory_count += 1

    def _schedule_resolution_env(
        self, product: Any, env: PricingEnvironment
    ) -> PricingEnvironment:
        """Resolve lifecycle schedules from the contract issue date."""
        base_date = getattr(product, "initial_date", None) or self._start_date
        if base_date is None:
            return env

        schedule_env = deepcopy(env)
        schedule_env.valuation_date = pd.Timestamp(base_date).to_pydatetime()
        return schedule_env

    def _scheduled_records(
        self, product: Any, env: PricingEnvironment, kind: str
    ) -> list[dict[str, Any]]:
        schedule_env = self._schedule_resolution_env(product, env)
        if kind == "ko":
            profile = product.get_ko_observation_profile(schedule_env)
            schedule = getattr(product.barrier_config, "ko_observation_schedule", None)
        else:
            if not getattr(product, "has_ki_barrier", False):
                return []
            profile = product.get_ki_observation_profile(schedule_env)
            schedule = getattr(product.barrier_config, "ki_observation_schedule", None)

        times = list(profile.get("observation_times", []))
        barriers = list(profile.get("barriers", []))
        payoffs = list(profile.get("payoffs", [0.0] * len(times)))
        schedule_dates = []
        if schedule is not None:
            for rec in schedule.records:
                schedule_dates.append(getattr(rec, "observation_date", None))

        records = []
        base_date = pd.Timestamp(getattr(product, "initial_date", None) or self._start_date)
        for idx, obs_time in enumerate(times):
            if idx < len(schedule_dates) and schedule_dates[idx] is not None:
                obs_date = pd.Timestamp(schedule_dates[idx]).normalize()
            else:
                obs_date = (base_date + timedelta(days=int(round(float(obs_time) * 365)))).normalize()
            obs_date = self._next_available_market_date(obs_date)
            records.append(
                {
                    "date": obs_date,
                    "time": float(obs_time),
                    "barrier": float(barriers[idx]) if idx < len(barriers) and barriers[idx] is not None else None,
                    "payoff": float(payoffs[idx]) if idx < len(payoffs) and payoffs[idx] is not None else 0.0,
                }
            )
        return records

    def _next_available_market_date(self, date: pd.Timestamp) -> pd.Timestamp:
        dates = self.config.market_data.dates
        eligible = dates[dates >= pd.Timestamp(date).normalize()]
        if len(eligible) == 0:
            return pd.Timestamp(date).normalize()
        return pd.Timestamp(eligible[0]).normalize()

    @staticmethod
    def _barrier_hit(
        spot: float, barrier: Optional[float], is_reverse: bool, is_ko: bool
    ) -> bool:
        if barrier is None:
            return False
        if is_ko:
            return spot <= barrier if is_reverse else spot >= barrier
        return spot >= barrier if is_reverse else spot <= barrier

    @staticmethod
    def _date_from_time(date: pd.Timestamp, time_years: float) -> pd.Timestamp:
        return pd.Timestamp(date).normalize() + timedelta(
            days=int(round(float(time_years) * 365))
        )
