"""
Per-product replay helpers for OTC autocallable backtests.

``ProductReplay`` encapsulates per-product daily replay: pricing-environment
construction, Greek calculation, and surface/event-probability recording.
Lifecycle event detection is fully DELEGATED to
``quantark.asset.equity.lifecycle.AutocallableLifecycleTracker``; this class
acts as the adapter that converts the ``LifecycleEvent`` objects returned by
the tracker into the engine's action-row sink format.

The book/hedge-level engine constructs one ``ProductReplay`` per product and
delegates the per-product steps to it, while keeping the futures-hedge and
book-level accounting for itself.  The engine passes in its own output lists as
``*_sink`` arguments so recorded rows are unchanged.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.lifecycle import AutocallableLifecycleTracker
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.numerical import is_close

from .engine_factory import create_mc_event_stats_engine
from .market import (
    ImpliedBasisYield,
    SignedDividendYield,
    derive_implied_dividend_yield,
)


class ProductReplay:
    """
    Per-product daily replay logic for a single autocallable product.

    Holds the inputs the moved methods previously read off the engine/config,
    a shared ``AutocallableLifecycleState`` (the same instance the engine uses,
    so lifecycle mutations are visible to both), and output sink lists.

    Two-phase initialisation note
    ------------------------------
    ``start_date`` may be supplied at construction time or assigned later via
    the ``start_date`` property (``AutocallableBacktestEngine`` assigns it at
    the top of ``run()``).  It MUST be set before any observation method is
    called — the tracker needs it to resolve schedule dates.
    """

    def __init__(
        self,
        *,
        product: Any,
        product_quantity: float,
        has_lifecycle: bool,
        lifecycle: Any,
        pricing_engine: BaseEngine,
        surface_engine: BaseEngine,
        event_stats_engine: BaseEngine,
        engine_config: Any,
        market_data: Any,
        start_date: Optional[pd.Timestamp],
        underlying: str,
        actions_sink: list[dict[str, Any]],
        event_prob_sink: list[dict[str, Any]],
        daily_event_sink: list[dict[str, Any]],
        surfaces_sink: list[dict[str, Any]],
        fixed_dividend_yield: Optional[float] = None,
        delta_bump_size: Optional[float] = None,
        gamma_bump_size: Optional[float] = None,
        surface_config: Any = None,
    ) -> None:
        self.product = product
        self.product_quantity = product_quantity
        self.has_lifecycle = has_lifecycle
        self.lifecycle = lifecycle
        self.pricing_engine = pricing_engine
        self.surface_engine = surface_engine
        self.event_stats_engine = event_stats_engine
        self.engine_config = engine_config
        self.market_data = market_data
        self.underlying = underlying
        self.fixed_dividend_yield = fixed_dividend_yield
        self.delta_bump_size = delta_bump_size
        self.gamma_bump_size = gamma_bump_size
        self.surface_config = surface_config

        self.actions_sink = actions_sink
        self.event_prob_sink = event_prob_sink
        self.daily_event_sink = daily_event_sink
        self.surfaces_sink = surfaces_sink

        # date_resolver captures self; do not replace self.market_data
        # post-construction or the resolver will keep using the old one.
        self._tracker = AutocallableLifecycleTracker(
            product=product,
            quantity=product_quantity,
            lifecycle=lifecycle,
            start_date=start_date,
            date_resolver=self._next_available_market_date,
            has_lifecycle=has_lifecycle,
        )

    @property
    def start_date(self) -> Optional[pd.Timestamp]:
        return self._tracker.start_date

    @start_date.setter
    def start_date(self, value: Optional[pd.Timestamp]) -> None:
        self._tracker.start_date = value

    def product_for_lifecycle(self):
        return self._tracker.product_for_lifecycle()

    def product_for_date(self, date: pd.Timestamp, pricing_env: PricingEnvironment):
        return self._tracker.product_for_pricing(date, pricing_env)

    def build_env(self, date: pd.Timestamp, market: dict[str, float], selected):
        expiry = pd.Timestamp(selected["expiry_date"]).normalize()
        futures_ttm = (expiry - date).days / 365.0
        basis_yield, implied_q = derive_implied_dividend_yield(
            rate=market["rate"],
            spot=market["spot"],
            futures_price=float(selected["futures_price"]),
            time_to_maturity=futures_ttm,
        )
        pricing_q = self.pricing_dividend_yield(implied_q)
        env = PricingEnvironment(
            spot_quote=SpotQuote(spot=market["spot"], asset_name=self.underlying),
            vol_surface=FlatVolSurface(volatility=market["volatility"]),
            rate_curve=FlatRateCurve(rate=market["rate"]),
            div_yield=SignedDividendYield(pricing_q),
            basis_yield=ImpliedBasisYield(basis_yield),
            valuation_date=date.to_pydatetime(),
        )
        return env, basis_yield, implied_q, futures_ttm

    def pricing_dividend_yield(self, implied_q: float) -> float:
        if self.fixed_dividend_yield is not None:
            return float(self.fixed_dividend_yield)
        return float(implied_q)

    def calculate_greeks(
        self, product: Any, env: PricingEnvironment, price: float
    ) -> dict[str, float]:
        params = getattr(self.pricing_engine, "params", None)
        uses_base_greeks = (
            isinstance(self.pricing_engine, BaseEngine)
            and type(self.pricing_engine).calculate_greeks is BaseEngine.calculate_greeks
        )
        engine_bump = 0.0
        if params is not None:
            get_cfg = getattr(params, "get_effective_bump_config", None)
            if callable(get_cfg):
                engine_bump = float(get_cfg().spot_bump)
            else:
                legacy = getattr(params, "bump_size", None)
                engine_bump = float(legacy) if legacy is not None else 0.0
        delta_bump = (
            float(self.delta_bump_size)
            if self.delta_bump_size is not None
            else engine_bump
        )
        gamma_bump = (
            float(self.gamma_bump_size)
            if self.gamma_bump_size is not None
            else engine_bump
        )
        if (
            uses_base_greeks
            and delta_bump > 0.0
            and gamma_bump > 0.0
            and np.isfinite(price)
        ):
            try:
                spot = float(env.spot)
                env_up = deepcopy(env)
                env_up.spot_quote.spot *= 1.0 + delta_bump
                delta_price_up = float(self.pricing_engine.price(product, env_up))

                env_down = deepcopy(env)
                env_down.spot_quote.spot *= 1.0 - delta_bump
                delta_price_down = float(self.pricing_engine.price(product, env_down))

                delta_spot_bump = spot * delta_bump
                delta = (delta_price_up - delta_price_down) / (
                    2.0 * delta_spot_bump
                )

                if is_close(delta_bump, gamma_bump, rel_tol=1e-5, abs_tol=1e-8):
                    gamma_price_up = delta_price_up
                    gamma_price_down = delta_price_down
                else:
                    env_gamma_up = deepcopy(env)
                    env_gamma_up.spot_quote.spot *= 1.0 + gamma_bump
                    gamma_price_up = float(
                        self.pricing_engine.price(product, env_gamma_up)
                    )

                    env_gamma_down = deepcopy(env)
                    env_gamma_down.spot_quote.spot *= 1.0 - gamma_bump
                    gamma_price_down = float(
                        self.pricing_engine.price(product, env_gamma_down)
                    )

                gamma_spot_bump = spot * gamma_bump
                gamma = (gamma_price_up - 2.0 * float(price) + gamma_price_down) / (
                    gamma_spot_bump**2
                )
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

    def record_surfaces(
        self,
        date: pd.Timestamp,
        product: Any,
        env: PricingEnvironment,
        spot: float,
        q_center: float,
    ) -> None:
        if self.surface_config is None:
            return
        spec = self.surface_config
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
                surf_env.spot_quote = SpotQuote(spot=float(s), asset_name=self.underlying)
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
                self.surfaces_sink.append(row)

    def record_event_probabilities(
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
        self.daily_event_sink.append(
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
            self.event_prob_sink.append(
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
            self.event_prob_sink.append(
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
            fallback = create_mc_event_stats_engine(product, self.engine_config)
            return fallback.calculate_event_stats(product, env)
        except Exception:
            return None

    def settle_maturity_if_due(
        self, date: pd.Timestamp, product: Any, env: PricingEnvironment, spot: float
    ) -> None:
        event = self._tracker.settle_maturity_if_due(date, product, env, spot)
        if event is not None:
            self.actions_sink.append(self._event_to_action_row(event))

    def apply_lifecycle_events(
        self, date: pd.Timestamp, product: Any, env: PricingEnvironment, spot: float
    ) -> None:
        for event in self._tracker.observe(date, product, env, spot):
            self.actions_sink.append(self._event_to_action_row(event))

    def _event_to_action_row(self, event) -> dict[str, Any]:
        row = {
            "date": event.date,
            "action_type": event.event_type.value,
            "observation_index": event.observation_index,
            "spot": event.spot,
            "barrier": event.barrier,
            "cashflow": event.cashflow,
            "alive_before": event.state_before["alive"],
            "knocked_in_before": event.state_before["knocked_in"],
            "knocked_out_before": event.state_before["knocked_out"],
            "matured_before": event.state_before["matured"],
            "alive_after": event.state_after["alive"],
            "knocked_in_after": event.state_after["knocked_in"],
            "knocked_out_after": event.state_after["knocked_out"],
            "matured_after": event.state_after["matured"],
        }
        row.update(event.metadata)
        return row

    def _next_available_market_date(self, date: pd.Timestamp) -> pd.Timestamp:
        dates = self.market_data.dates
        eligible = dates[dates >= pd.Timestamp(date).normalize()]
        if len(eligible) == 0:
            return pd.Timestamp(date).normalize()
        return pd.Timestamp(eligible[0]).normalize()

    @staticmethod
    def _date_from_time(date: pd.Timestamp, time_years: float) -> pd.Timestamp:
        return pd.Timestamp(date).normalize() + timedelta(
            days=int(round(float(time_years) * 365))
        )
