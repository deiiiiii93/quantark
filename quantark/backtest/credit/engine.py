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

from quantark.asset.credit.engine.schedule import contractual_coupon_dates
from quantark.asset.credit.product.cds import CDS
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

        hedge: Dict[str, float] = {e: 0.0 for e in entities}  # accumulated spread CS01
        prev_hazard: Dict[str, float] = {
            e: portfolio.pricing_environments[e].hazard_curve.get_hazard_rate(1.0)
            for e in entities
        }

        cum_hedge_pnl = 0.0
        cum_costs = 0.0
        num_hedges = 0
        rows = []

        # Total-return cash ledger: as the valuation date advances the engine
        # reprices each seasoned CDS (roll-down) and settled coupons leave the
        # PV. Booking them as realized cash keeps reported portfolio P&L a
        # carry-preserving total return. Un-dated CDS book nothing (unchanged).
        realized_cash = 0.0
        prev_ts = first_ts

        for ts, row in path.iterrows():
            self._apply_row(portfolio, entities, row, ts)
            for pos in portfolio.positions.values():
                realized_cash += self._realized_coupon_cash(pos, prev_ts, ts)
            prev_ts = ts

            # Mark the existing hedge to the new spread before rebalancing. The
            # hedge holds a spread CS01, so its P&L is driven by the spread move
            # Δs = Δlambda (1 - R), not the raw hazard move.
            for entity in entities:
                hazard = portfolio.pricing_environments[entity].hazard_curve.get_hazard_rate(1.0)
                spread_move_bps = (
                    (hazard - prev_hazard[entity]) * (1.0 - cfg.hedge_recovery) / _BPS
                )
                cum_hedge_pnl += hedge[entity] * spread_move_bps
                prev_hazard[entity] = hazard

            # Total-return value = remaining mark-to-market + cash booked so far.
            value = portfolio.get_portfolio_value() + realized_cash
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
                    # `size` is a hedge CS01 ($ per 1bp). Convert it to the
                    # underlying CDS notional via CS01 = notional * RPV01 * 1bp,
                    # so the cost models see a real trade notional (and unit
                    # price) rather than a dollar-CS01 / hazard-rate mix.
                    hedge_notional = self._cs01_to_notional(portfolio, entity, size)
                    cost = cost_model.calculate_cost(
                        quantity=hedge_notional, price=1.0, notional=hedge_notional,
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
            final_value=portfolio.get_portfolio_value() + realized_cash,
            num_hedges=num_hedges,
            total_transaction_costs=cum_costs,
            total_hedge_pnl=cum_hedge_pnl,
            config_summary=cfg.get_summary(),
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _realized_coupon_cash(position, start, end) -> float:
        """
        Premium cash settled in ``(start, end]`` for one seasoned CDS position.

        Protection buyers pay the running coupon (cash outflow), sellers receive
        it. With no default modelled on the deterministic path the name is taken
        to survive, so each contractual coupon is booked in full at face when its
        payment date is crossed. Un-dated CDS book nothing.
        """
        product = position.product
        if not isinstance(product, CDS) or not product.is_dated:
            return 0.0
        cash = 0.0
        for pay_date, accrual in contractual_coupon_dates(
            product.effective_date, product.maturity_date, product.payment_freq
        ):
            if start < pay_date <= end:
                coupon = product.notional * product.coupon_spread * accrual
                cash += -product.side_sign * position.quantity * coupon
        return cash

    @staticmethod
    def _cs01_to_notional(
        portfolio: CreditPortfolio, entity: str, cs01_amount: float
    ) -> float:
        """
        Convert a hedge CS01 ($ per 1bp) into a CDS notional for cost models.

        Uses the credit-triangle identity ``CS01 = notional * RPV01 * 1bp``,
        where the risky annuity (RPV01) per unit notional is recovered from a
        representative unit-notional CDS on ``entity``. The present value is
        linear in the running coupon (``PV = side_sign * (protection -
        coupon * RPV01)``), so differencing two coupon levels isolates RPV01
        using only the engine's guaranteed ``price`` method.
        """
        positions = [
            p for p in portfolio.positions.values() if p.reference_entity == entity
        ]
        if not positions:
            return abs(cs01_amount)
        reference = positions[0]
        env = portfolio.pricing_environments[entity]
        unit = dataclasses.replace(reference.product, notional=1.0)
        pv_zero = reference.engine.price(
            dataclasses.replace(unit, coupon_spread=0.0), env
        )
        pv_one = reference.engine.price(
            dataclasses.replace(unit, coupon_spread=0.01), env
        )
        rpv01 = abs(pv_zero - pv_one) / 0.01  # risky annuity per unit notional
        if rpv01 <= 0:
            return abs(cs01_amount)
        return abs(cs01_amount) / (rpv01 * _BPS)

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
