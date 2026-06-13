"""
Credit backtest engine.

Steps a credit book through a market path (per-entity hazard / rate levels),
reprices each day, and neutralises CS01 with a linear credit-spread hedge,
tracking hedge P&L and transaction costs. The hedge holds an offsetting CS01
amount; its P&L over a spread move is ``hedge_cs01 * Δspread_in_bps``.
"""
from __future__ import annotations

import dataclasses
from copy import deepcopy
from typing import Dict

from quantark.asset.credit.riskmeasures import CreditGreeksCalculator
from quantark.backtest.credit.config import CreditBacktestConfig
from quantark.backtest.credit.results import CreditBacktestResults
from quantark.param import FlatRateCurve
from quantark.param.credit import FlatHazardCurve
from quantark.portfolio.credit import CreditPortfolio
from quantark.util.exceptions import ValidationError

# A CS01 is quoted per 1bp; spread moves are converted to bps with this.
_BPS = 1e-4


class CreditBacktestEngine:
    """Simulate a credit hedging strategy over a market path."""

    def __init__(self, config: CreditBacktestConfig):
        self.config = config
        self._calc = CreditGreeksCalculator()

    def run(self) -> CreditBacktestResults:
        cfg = self.config
        portfolio = self._clone_portfolio(cfg.portfolio)
        entities = sorted({p.reference_entity for p in portfolio.positions.values()})
        path = cfg.market_path
        strategy = cfg.strategy
        strategy.reset()
        cost_model = cfg.transaction_cost_model

        for entity in entities:
            if f"{entity}_hazard" not in path.columns:
                raise ValidationError(
                    f"market_path missing required column '{entity}_hazard'"
                )

        first_ts = path.index[0]
        self._apply_row(portfolio, entities, path.iloc[0], first_ts)
        initial_value = portfolio.get_portfolio_value()

        hedge: Dict[str, float] = {e: 0.0 for e in entities}  # accumulated hedge CS01
        prev_hazard: Dict[str, float] = {
            e: portfolio.pricing_environments[e].hazard_curve.get_hazard_rate(1.0)
            for e in entities
        }

        cum_hedge_pnl = 0.0
        cum_costs = 0.0
        num_hedges = 0
        rows = []

        for ts, row in path.iterrows():
            self._apply_row(portfolio, entities, row, ts)

            # Mark the existing hedge to the new spread before rebalancing.
            for entity in entities:
                hazard = portfolio.pricing_environments[entity].hazard_curve.get_hazard_rate(1.0)
                cum_hedge_pnl += hedge[entity] * (hazard - prev_hazard[entity]) / _BPS
                prev_hazard[entity] = hazard

            value = portfolio.get_portfolio_value()
            portfolio_pnl = value - initial_value

            cs01_pre_total = 0.0
            cs01_post_total = 0.0
            hedged_today = False
            costs_today = 0.0

            for entity in entities:
                hazard = portfolio.pricing_environments[entity].hazard_curve.get_hazard_rate(1.0)
                entity_cs01 = 0.0
                if cfg.calculate_greeks:
                    entity_cs01 = portfolio.get_greeks_by_underlying(entity, self._calc)["cs01"]
                net_cs01 = entity_cs01 + hedge[entity]
                cs01_pre_total += net_cs01

                greeks = {"cs01": net_cs01}
                market = {"spread": hazard, "timestamp": ts}
                strategy.on_step(ts, greeks, market)
                if strategy.should_hedge(ts, greeks, market):
                    size = strategy.calculate_hedge_size(ts, greeks, market)
                    hedge[entity] += size
                    cost = cost_model.calculate_cost(
                        quantity=abs(size), price=hazard, notional=abs(size),
                        instrument_type="cds", trade_type="hedge")
                    costs_today += cost
                    num_hedges += 1
                    hedged_today = True
                    strategy.on_hedge_executed(ts, size, hazard)

                cs01_post_total += entity_cs01 + hedge[entity]

            cum_costs += costs_today
            net_pnl = portfolio_pnl + cum_hedge_pnl - cum_costs

            rows.append({
                "timestamp": ts,
                "portfolio_value": value,
                "portfolio_pnl": portfolio_pnl,
                "hedge_pnl": cum_hedge_pnl,
                "transaction_costs": cum_costs,
                "net_pnl": net_pnl,
                "cs01_pre": cs01_pre_total,
                "cs01_post": cs01_post_total,
                "hedged": hedged_today,
                "hedge_cs01": sum(hedge.values()),
            })

        return CreditBacktestResults(
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
    def _clone_portfolio(portfolio: CreditPortfolio) -> CreditPortfolio:
        cloned = CreditPortfolio(
            portfolio_name=portfolio.portfolio_name + "_backtest",
            pricing_environments={
                e: deepcopy(env) for e, env in portfolio.pricing_environments.items()
            },
            creation_date=portfolio.creation_date,
        )
        cloned.positions = deepcopy(portfolio.positions)
        return cloned

    @staticmethod
    def _apply_row(portfolio: CreditPortfolio, entities, row, timestamp) -> None:
        for entity in entities:
            env = portfolio.pricing_environments[entity]
            changes: Dict[str, object] = {"valuation_date": timestamp}
            hazard_col = f"{entity}_hazard"
            if hazard_col in row and row[hazard_col] == row[hazard_col]:  # not NaN
                changes["hazard_curve"] = FlatHazardCurve(hazard_rate=float(row[hazard_col]))
            rate_col = f"{entity}_rate"
            if rate_col in row and row[rate_col] == row[rate_col]:
                changes["discount_curve"] = FlatRateCurve(rate=float(row[rate_col]))
            portfolio.pricing_environments[entity] = dataclasses.replace(env, **changes)

    def __repr__(self) -> str:
        return f"CreditBacktestEngine(strategy={self.config.strategy.name})"
