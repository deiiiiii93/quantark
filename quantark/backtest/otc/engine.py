"""
Daily OTC autocallable futures-hedging backtest engine.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np
import pandas as pd

from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import ValidationError

from ._replay import ProductReplay
from .config import AutocallableBacktestConfig
from .engine_factory import (
    create_event_stats_engine,
    create_pricing_engine,
    create_surface_engine,
    create_vol_model_engine,
)
from .results import AutocallableBacktestResults
from .state import (
    AutocallableDeltaHedgeStrategy,
    AutocallableLifecycleState,
    FuturesHedgePosition,
)
from quantark.volmodels.calibration import VolModelCalibrator


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
        self._calibration_records: list[dict[str, Any]] = []

        self._initial_product_value: Optional[float] = None
        self._transaction_costs: float = 0.0
        self._start_date: Optional[pd.Timestamp] = None

        # Per-day vol-model calibration (Task 2.3/2.4).  The calibrator is
        # keyed by surface artifact sha and shared across days; each priced
        # day swaps self.pricing_engine for a fresh vol-model engine wired
        # to that day's calibrated model.
        self._calibrator: Optional[VolModelCalibrator] = None
        if config.engine_config.vol_model != "bsm":
            if getattr(config.market_data, "surface_history", None) is None:
                raise ValidationError(
                    "vol_model != 'bsm' requires market_data.surface_history "
                    "(per-day vol-model calibration is keyed by surface artifact)"
                )
            self._calibrator = VolModelCalibrator(
                config.engine_config.vol_model_calibration
            )

        self._replay = ProductReplay(
            product=config.product,
            product_quantity=config.product_quantity,
            has_lifecycle=True,
            lifecycle=self.lifecycle,
            pricing_engine=self.pricing_engine,
            surface_engine=self.surface_engine,
            event_stats_engine=self.event_stats_engine,
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

    def run(self) -> AutocallableBacktestResults:
        dates = self._backtest_dates()
        if len(dates) == 0:
            raise ValidationError("No common market-data dates for backtest")
        self._start_date = pd.Timestamp(dates[0]).normalize()
        self._replay.start_date = self._start_date

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

            env, basis_yield, implied_q, futures_ttm = self._replay.build_env(
                date, market, selected
            )
            # Vol-model variants: calibrate once per surface artifact and
            # swap in the day's vol-model engine before ANY pricing (initial
            # price, base price, bumped greeks) so the whole day is
            # model-consistent.  Skipped once the product is dead.
            day_calibration_record: Optional[dict[str, Any]] = None
            if self._calibrator is not None and (
                self.lifecycle.alive or self._initial_product_value is None
            ):
                day_calibration_record = self._calibrate_day(date)
            product = self._replay.product_for_date(date, env)

            if self._initial_product_value is None:
                initial_price = (
                    float(self.config.initial_product_price)
                    if self.config.initial_product_price is not None
                    else float(self.pricing_engine.price(product, env))
                )
                self._initial_product_value = (
                    self.config.product_quantity * initial_price
                )

            lifecycle_product = self._replay.product_for_lifecycle()
            self._replay.apply_lifecycle_events(
                date, lifecycle_product, env, market["spot"]
            )
            self._replay.settle_maturity_if_due(
                date, lifecycle_product, env, market["spot"]
            )
            product = self._replay.product_for_date(date, env)

            price = 0.0
            greeks = {"price": 0.0, "delta": 0.0, "gamma": 0.0}
            pricing_started = time.perf_counter()
            if self.lifecycle.alive:
                price = float(self.pricing_engine.price(product, env))
                greeks = self._calculate_greeks(product, env, price)
            if day_calibration_record is not None:
                day_calibration_record["pricing_seconds"] = (
                    time.perf_counter() - pricing_started
                )
                self._calibration_records.append(day_calibration_record)

            if self.config.calculate_event_probabilities and self.lifecycle.alive:
                self._replay.record_event_probabilities(date, product, env)

            if self.config.calculate_surfaces and self.lifecycle.alive:
                self._replay.record_surfaces(
                    date,
                    product,
                    env,
                    market["spot"],
                    self._replay.pricing_dividend_yield(implied_q),
                )

            pre_hedge_contracts = self.hedge_position.quantity
            self._rebalance(date, selected, greeks)
            self._record_day(
                date=date,
                selected=selected,
                market=market,
                basis_yield=basis_yield,
                implied_q=implied_q,
                pricing_q=self._replay.pricing_dividend_yield(implied_q),
                futures_ttm=futures_ttm,
                price=price,
                greeks=greeks,
                pre_hedge_contracts=pre_hedge_contracts,
            )

        results = AutocallableBacktestResults(
            config=self.config,
            states=self._states,
            greeks=self._greeks,
            rebalances=self._rebalances,
            trades=self._trades,
            actions=self._actions,
            surfaces=self._surfaces,
            daily_event_summary=self._daily_event_summary,
            event_probabilities=self._event_probabilities,
            calibration_records=self._calibration_records,
        )
        records_path = getattr(
            self.config.engine_config.vol_model_calibration, "records_path", None
        )
        if records_path:
            results.export_calibration_records(records_path)
        return results

    def _calibrate_day(self, date: pd.Timestamp) -> dict[str, Any]:
        """Calibrate the day's vol model and swap in its pricing engine.

        The calibrator caches by surface artifact sha, so carry-forward days
        (and fleet runs sharing the surface) reuse the stored calibration.
        A fresh engine is constructed per day from the frozen calibrated
        model — never a mutation of a shared engine — and drives both the
        base price and the bumped-greek reprices of the day.  Any failure
        propagates (fail-closed; no flat-vol fallback).
        """
        engine_config = self.config.engine_config
        artifact = self.config.market_data.surface_history.surface_for(date)
        calibrated = self._calibrator.calibrate(engine_config.vol_model, artifact)
        self.pricing_engine = create_vol_model_engine(
            vol_model=engine_config.vol_model,
            solver=engine_config.vol_model_solver,
            calibrated=calibrated,
            pde_params=engine_config.pde_params,
            mc_params=engine_config.mc_params,
            mc_method=engine_config.resolve_vol_model_mc_method(),
            engine_options=engine_config.vol_model_engine_options,
        )
        # Sync the replay immediately so no code path can see yesterday's
        # engine (``_calculate_greeks`` also syncs lazily; do not rely on it).
        self._replay.pricing_engine = self.pricing_engine
        record = dict(calibrated.record)
        record["date"] = pd.Timestamp(date).date().isoformat()
        return record

    def _backtest_dates(self) -> pd.DatetimeIndex:
        dates = self.config.market_data.dates
        if self.config.start_date is not None:
            dates = dates[dates >= pd.Timestamp(self.config.start_date).normalize()]
        if self.config.end_date is not None:
            dates = dates[dates <= pd.Timestamp(self.config.end_date).normalize()]
        return dates

    def _calculate_greeks(
        self, product: Any, env: PricingEnvironment, price: float
    ) -> dict[str, float]:
        """Delegate to the per-product replay.

        Kept as a thin engine-level wrapper so callers that swap
        ``self.pricing_engine`` after construction (the replay shares the same
        engine instance) still drive greeks through the overridden engine.
        """
        self._replay.pricing_engine = self.pricing_engine
        return self._replay.calculate_greeks(product, env, price)

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

        state_row = {
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
        # Surface mode only: record which IV-surface artifact (post
        # carry-forward) priced this day.  Scalar mode adds no columns.
        surface_provenance = getattr(self._replay, "last_surface_provenance", None)
        if surface_provenance:
            state_row.update(surface_provenance)
        self._states.append(state_row)
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
