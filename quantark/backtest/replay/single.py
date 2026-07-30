"""
Single-product autocallable backtest — a book-of-one wrapper.

``AutocallableBacktestEngine`` is the public single-product API. Since the
consolidation it no longer carries its own daily loop: it builds a
``ReplayBacktestConfig`` with one ``ReplayProduct``, runs the unified
``ReplayBacktestEngine``, and presents the results through the legacy
property-based ``AutocallableBacktestResults`` with the frozen single-product
row schema (golden-gated byte identity with the pre-consolidation engine).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import AutocallableBacktestConfig, ReplayBacktestConfig, ReplayProduct
from .engine import ReplayBacktestEngine
from .results import AutocallableBacktestResults

__all__ = ["AutocallableBacktestConfig", "AutocallableBacktestEngine"]

# Per-product sensitivity fields read off the greeks dict (NaN when the
# engine does not provide them) — frozen single-product row layout.
_GREEK_SENSITIVITY_FIELDS = (
    "vega",
    "theta",
    "rho",
    "dividend_sensitivity",
    "basis_sensitivity",
)


class AutocallableBacktestEngine:
    """
    Historical replay engine for Snowball/Phoenix delta hedging with futures.

    Thin wrapper over :class:`ReplayBacktestEngine` (book of one product).
    """

    def __init__(self, config: AutocallableBacktestConfig):
        self.config = config
        self._book_config = ReplayBacktestConfig(
            products=[
                ReplayProduct(
                    product=config.product,
                    quantity=config.product_quantity,
                    position_id=0,
                    has_lifecycle=True,
                    initial_price=config.initial_product_price,
                )
            ],
            market_data=config.market_data,
            engine_config=config.engine_config,
            strategy=config.strategy,
            transaction_cost_model=config.transaction_cost_model,
            underlying=config.underlying,
            start_date=config.start_date,
            end_date=config.end_date,
            fixed_dividend_yield=config.fixed_dividend_yield,
            delta_bump_size=config.delta_bump_size,
            gamma_bump_size=config.gamma_bump_size,
            surface_config=config.surface_config,
            calculate_surfaces=config.calculate_surfaces,
            calculate_event_probabilities=config.calculate_event_probabilities,
            metadata=config.metadata,
        )
        # Honor the single config's explicit futures roll policy.
        self._book_config.hedge.roll_policy = config.roll_policy
        self._inner = ReplayBacktestEngine(self._book_config)

    # ------------------------------------------------------------------
    # Legacy surface: tests override the pricing engine before run().
    # ------------------------------------------------------------------

    @property
    def pricing_engine(self):
        return self._inner._pricing_engines[0]

    @pricing_engine.setter
    def pricing_engine(self, engine) -> None:
        self._inner._pricing_engines[0] = engine

    @property
    def lifecycle(self):
        return self._inner._replays[0].lifecycle

    @property
    def strategy(self):
        return self._inner.strategy

    @property
    def hedge_position(self):
        return self._inner.hedge_position

    def _calculate_greeks(self, product, env, price=None):
        """Legacy passthrough (tests exercise per-day greeks directly).

        ``price`` is accepted for signature compatibility; the engine
        recomputes its own base price (identical for deterministic engines).
        """
        del price
        return self._inner._replays[0].calculate_greeks(
            product, env, engine=self.pricing_engine
        )

    # ------------------------------------------------------------------

    def run(self) -> AutocallableBacktestResults:
        self._inner.run()
        inner = self._inner

        states = [
            self._single_state_row(book_row, daily)
            for book_row, daily in zip(inner._states, inner._product_daily)
        ]
        greeks = [
            self._single_greek_row(book_row, daily)
            for book_row, daily in zip(inner._greeks, inner._product_daily)
        ]

        results = AutocallableBacktestResults(
            config=self.config,
            states=states,
            greeks=greeks,
            rebalances=inner._rebalances,
            trades=inner._trades,
            actions=inner._actions,
            surfaces=inner._surfaces,
            daily_event_summary=inner._daily_event_summary,
            event_probabilities=inner._event_probabilities,
            calibration_records=inner._calibration_records,
        )
        records_path = getattr(
            self.config.engine_config.vol_model_calibration, "records_path", None
        )
        if records_path:
            results.export_calibration_records(records_path)
        return results

    @staticmethod
    def _single_state_row(
        book_row: dict[str, Any], daily: dict[str, Any]
    ) -> dict[str, Any]:
        row = dict(book_row)
        provenance = daily.get("provenance")
        if provenance:
            row.update(provenance)
        return row

    @staticmethod
    def _single_greek_row(
        book_row: dict[str, Any], daily: dict[str, Any]
    ) -> dict[str, Any]:
        g = daily["greeks"]
        delta = float(g.get("delta", 0.0))
        gamma = float(g.get("gamma", 0.0))
        row = {
            "date": book_row["date"],
            "price": float(g.get("price", daily["price"])),
            "delta": delta,
            "gamma": gamma,
            "product_delta": delta,
            "product_gamma": gamma,
            "product_position_delta": book_row["product_position_delta"],
            "product_position_gamma": book_row["product_position_gamma"],
            "pre_hedge_contracts": book_row["pre_hedge_contracts"],
            "post_hedge_contracts": book_row["post_hedge_contracts"],
            "futures_multiplier": book_row["futures_multiplier"],
            "pre_hedge_futures_delta": book_row["pre_hedge_futures_delta"],
            "post_hedge_futures_delta": book_row["post_hedge_futures_delta"],
            "pre_hedge_delta": book_row["pre_hedge_delta"],
            "post_hedge_delta": book_row["post_hedge_delta"],
            "pre_hedge_gamma": book_row["pre_hedge_gamma"],
            "post_hedge_gamma": book_row["post_hedge_gamma"],
            "pre_hedge_delta_cash_1pct": book_row["pre_hedge_delta_cash_1pct"],
            "post_hedge_delta_cash_1pct": book_row["post_hedge_delta_cash_1pct"],
            "pre_hedge_gamma_cash_1pct": book_row["pre_hedge_gamma_cash_1pct"],
            "post_hedge_gamma_cash_1pct": book_row["post_hedge_gamma_cash_1pct"],
            "delta_cash_1pct": book_row["delta_cash_1pct"],
            "gamma_cash_1pct": book_row["gamma_cash_1pct"],
        }
        for field in _GREEK_SENSITIVITY_FIELDS:
            row[field] = float(g.get(field, np.nan))
        return row
