"""
Per-product replay helpers for OTC autocallable backtests.

``ProductReplay`` encapsulates the single-product daily logic that operates on
one product and its lifecycle state. The book/hedge-level engine constructs one
``ProductReplay`` per product and delegates the per-product steps to it, while
keeping the futures-hedge and book-level accounting for itself.

This is a behavior-preserving extraction of methods that previously lived on
``AutocallableBacktestEngine``; the engine passes in its own output lists as
``*_sink`` arguments so recorded rows are unchanged.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd

from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment

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
    ``start_date`` is ``None`` at construction time.  It MUST be populated
    before any replay method is called.  ``AutocallableBacktestEngine`` sets
    ``self._replay.start_date`` to the first backtest date at the top of
    ``run()``, before the daily loop begins.  Any standalone caller is
    responsible for the same assignment.
    """

    def __init__(
        self,
        *,
        product: Any,
        product_quantity: float,
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
        self.lifecycle = lifecycle
        self.pricing_engine = pricing_engine
        self.surface_engine = surface_engine
        self.event_stats_engine = event_stats_engine
        self.engine_config = engine_config
        self.market_data = market_data
        self.start_date = start_date
        self.underlying = underlying
        self.fixed_dividend_yield = fixed_dividend_yield
        self.delta_bump_size = delta_bump_size
        self.gamma_bump_size = gamma_bump_size
        self.surface_config = surface_config

        self.actions_sink = actions_sink
        self.event_prob_sink = event_prob_sink
        self.daily_event_sink = daily_event_sink
        self.surfaces_sink = surfaces_sink

    def product_for_lifecycle(self):
        product = deepcopy(self.product)
        setattr(product, "_otc_lifecycle_knocked_in", self.lifecycle.knocked_in)
        return product

    def product_for_date(self, date: pd.Timestamp, pricing_env: PricingEnvironment):
        product = deepcopy(self.product)
        setattr(product, "_otc_lifecycle_knocked_in", self.lifecycle.knocked_in)
        if (
            getattr(product, "exercise_date", None) is None
            and getattr(product, "maturity", None) is not None
            and self.start_date is not None
        ):
            elapsed = max(0.0, (date - self.start_date).days / 365.0)
            product.maturity = max(float(product.maturity) - elapsed, 1e-8)
        elif self.start_date is not None:
            elapsed = max(0.0, (date - self.start_date).days / 365.0)
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
        engine_bump = (
            float(getattr(params, "bump_size", 1e-4)) if params is not None else 0.0
        )
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

                if np.isclose(delta_bump, gamma_bump):
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
        self.actions_sink.append(row)

    def _maturity_market_date(self, product: Any, env: PricingEnvironment) -> pd.Timestamp:
        explicit = (
            getattr(product, "maturity_date", None)
            or getattr(product, "exercise_date", None)
        )
        if explicit is not None:
            return self._next_available_market_date(pd.Timestamp(explicit).normalize())

        maturity = float(product.get_maturity(env))
        base_date = pd.Timestamp(getattr(product, "initial_date", None) or self.start_date)
        maturity_date = (
            base_date + timedelta(days=int(round(maturity * 365.0)))
        ).normalize()
        return self._next_available_market_date(maturity_date)

    def settle_maturity_if_due(
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
        cashflow = self.product_quantity * payoff
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

    def apply_lifecycle_events(
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
                    cashflow = self.product_quantity * float(rec.get("payoff", 0.0))
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
                    coupon = self.product_quantity * product.get_coupon_payoff(idx)
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
        base_date = getattr(product, "initial_date", None) or self.start_date
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
        base_date = pd.Timestamp(getattr(product, "initial_date", None) or self.start_date)
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
        dates = self.market_data.dates
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
