"""
FX backtest engine.

Steps an FX book through a market path (per-pair spot / vol / two-rate levels),
reprices each day, and neutralises spot delta with a linear FX spot hedge,
tracking hedge P&L and transaction costs.
"""
from __future__ import annotations

import dataclasses
from copy import deepcopy
from typing import Dict

from quantark.asset.fx.riskmeasures.fx_greeks_calculator import FxGreeksCalculator
from quantark.backtest.fx.config import FXBacktestConfig
from quantark.backtest.fx.results import FXBacktestResults
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.portfolio.fx import FXPortfolio
from quantark.util.exceptions import ValidationError


class FXBacktestEngine:
    """Simulate an FX hedging strategy over a market path."""

    def __init__(self, config: FXBacktestConfig):
        self.config = config
        self._calc = FxGreeksCalculator()

    def run(self) -> FXBacktestResults:
        cfg = self.config
        portfolio = self._clone_portfolio(cfg.portfolio)
        pairs = sorted({p.underlying for p in portfolio.positions.values()})
        path = cfg.market_path
        strategy = cfg.strategy
        strategy.reset()
        cost_model = cfg.transaction_cost_model

        for pair in pairs:
            if f"{pair}_spot" not in path.columns:
                raise ValidationError(f"market_path missing required column '{pair}_spot'")

        # Apply first row, establish baseline.
        first_ts = path.index[0]
        self._apply_row(portfolio, pairs, path.iloc[0], first_ts)
        initial_value = portfolio.get_portfolio_value()

        hedge: Dict[str, float] = {pair: 0.0 for pair in pairs}
        prev_spot: Dict[str, float] = {
            pair: portfolio.pricing_environments[pair].spot_quote.spot for pair in pairs
        }

        cum_hedge_pnl = 0.0
        cum_costs = 0.0
        num_hedges = 0
        rows = []

        for ts, row in path.iterrows():
            self._apply_row(portfolio, pairs, row, ts)

            # Mark the existing hedge to the new spot before rebalancing.
            for pair in pairs:
                spot = portfolio.pricing_environments[pair].spot_quote.spot
                cum_hedge_pnl += hedge[pair] * (spot - prev_spot[pair])
                prev_spot[pair] = spot

            value = portfolio.get_portfolio_value()
            portfolio_pnl = value - initial_value

            delta_pre_total = 0.0
            delta_post_total = 0.0
            hedged_today = False
            costs_today = 0.0

            for pair in pairs:
                spot = portfolio.pricing_environments[pair].spot_quote.spot
                pair_delta = 0.0
                if cfg.calculate_greeks:
                    pair_delta = portfolio.get_greeks_by_underlying(pair, self._calc)["delta"]
                net_delta = pair_delta + hedge[pair]
                delta_pre_total += net_delta

                greeks = {"delta": net_delta}
                market = {"spot": spot, "timestamp": ts}
                strategy.on_step(ts, greeks, market)
                if strategy.should_hedge(ts, greeks, market):
                    size = strategy.calculate_hedge_size(ts, greeks, market)
                    hedge[pair] += size
                    cost = cost_model.calculate_cost(
                        quantity=abs(size), price=spot, notional=abs(size) * spot,
                        instrument_type="spot", trade_type="hedge")
                    costs_today += cost
                    num_hedges += 1
                    hedged_today = True
                    strategy.on_hedge_executed(ts, size, spot)

                delta_post_total += pair_delta + hedge[pair]

            cum_costs += costs_today
            net_pnl = portfolio_pnl + cum_hedge_pnl - cum_costs

            rows.append({
                "timestamp": ts,
                "portfolio_value": value,
                "portfolio_pnl": portfolio_pnl,
                "hedge_pnl": cum_hedge_pnl,
                "transaction_costs": cum_costs,
                "net_pnl": net_pnl,
                "delta_pre": delta_pre_total,
                "delta_post": delta_post_total,
                "hedged": hedged_today,
                "hedge_notional": sum(hedge.values()),
            })

        return FXBacktestResults(
            rows=rows,
            initial_value=initial_value,
            final_value=portfolio.get_portfolio_value(),
            num_hedges=num_hedges,
            total_transaction_costs=cum_costs,
            total_hedge_pnl=cum_hedge_pnl,
            config_summary=cfg.get_summary(),
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _clone_portfolio(portfolio: FXPortfolio) -> FXPortfolio:
        cloned = FXPortfolio(
            portfolio_name=portfolio.portfolio_name + "_backtest",
            pricing_environments={
                pair: deepcopy(env) for pair, env in portfolio.pricing_environments.items()
            },
            creation_date=portfolio.creation_date,
        )
        cloned.positions = deepcopy(portfolio.positions)
        return cloned

    @staticmethod
    def _apply_row(portfolio: FXPortfolio, pairs, row, timestamp) -> None:
        for pair in pairs:
            env = portfolio.pricing_environments[pair]
            changes: Dict[str, object] = {"valuation_date": timestamp}
            spot_col = f"{pair}_spot"
            if spot_col in row and row[spot_col] == row[spot_col]:  # not NaN
                changes["spot_quote"] = SpotQuote(
                    spot=float(row[spot_col]),
                    timestamp=env.spot_quote.timestamp,
                    asset_name=env.spot_quote.asset_name,
                )
            vol_col = f"{pair}_vol"
            if env.vol_surface is not None and vol_col in row and row[vol_col] == row[vol_col]:
                changes["vol_surface"] = FlatVolSurface(volatility=float(row[vol_col]))
            dom_col = f"{pair}_dom_rate"
            if dom_col in row and row[dom_col] == row[dom_col]:
                changes["domestic_curve"] = FlatRateCurve(rate=float(row[dom_col]))
            for_col = f"{pair}_for_rate"
            if for_col in row and row[for_col] == row[for_col]:
                changes["foreign_curve"] = FlatRateCurve(rate=float(row[for_col]))
            portfolio.pricing_environments[pair] = dataclasses.replace(env, **changes)

    def __repr__(self) -> str:
        return f"FXBacktestEngine(strategy={self.config.strategy.name})"
