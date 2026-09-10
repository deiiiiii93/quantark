"""Per-underlying multi-product net-delta hedging backtest (autocallable lifecycle aware)."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import math
import time

import pandas as pd

from quantark.util.exceptions import ValidationError
from .config import (
    AutocallableEngineConfig,
    HedgeSpec,
    ReplayBacktestConfig,
    ReplayProduct,
    SurfaceGridConfig,
)
from .market import AutocallableMarketDataSet, ImpliedBasisYield, SignedDividendYield
from .strategy_state import AutocallableDeltaHedgeStrategy, AutocallableLifecycleState, FuturesHedgePosition
from quantark.backtest.futures_ledger import FuturesHedgeBook
from quantark.backtest.strategy.futures_bucket_strategy import (
    FuturesBucketHedgeStrategy,
)
from quantark.backtest.strategy.futures_delta_strategy import (
    ProportionalFuturesDeltaHedgeStrategy,
)
from quantark.backtest.transaction_costs import TransactionCostModel, ZeroCostModel
from .engine_factory import (
    create_event_stats_engine,
    create_pricing_engine,
    create_surface_engine,
    create_vol_model_engine,
)
from quantark.volmodels.calibration import VolModelCalibrator
from .product_replay import ProductReplay
from .results import BookBacktestResults


@dataclass(frozen=True)
class _PlannedTrade:
    """One validated leg trade, priced and costed BEFORE the ledger moves."""

    date: Any
    contract: str
    quantity_delta: float
    price: float
    multiplier: float
    notional: float
    cost: float
    trade_type: str
    reason: str

    def to_row(self) -> dict:
        return {
            "date": self.date,
            "trade_type": self.trade_type,
            "instrument_type": "futures",
            "contract": self.contract,
            "quantity": self.quantity_delta,
            "price": self.price,
            "multiplier": self.multiplier,
            "notional": self.notional,
            "transaction_cost": self.cost,
            "reason": self.reason,
        }


class ReplayBacktestEngine:
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

    def __init__(self, config: ReplayBacktestConfig):
        self.config = config
        self.strategy = config.strategy
        self.hedge = config.hedge
        # One book holds every leg.  Single-contract strategies keep at most
        # one open leg in it, so their arithmetic and goldens are unchanged.
        self.hedge_book = FuturesHedgeBook()
        self._is_bucket = isinstance(config.strategy, FuturesBucketHedgeStrategy)
        self._carry_recording = getattr(config, "carry_recording", None)
        self._hedge_infeasibility: Optional[dict[str, Any]] = None

        self._states: list[dict[str, Any]] = []
        self._greeks: list[dict[str, Any]] = []
        self._rebalances: list[dict[str, Any]] = []
        self._trades: list[dict[str, Any]] = []
        self._actions: list[dict[str, Any]] = []
        self._surfaces: list[dict[str, Any]] = []
        self._daily_event_summary: list[dict[str, Any]] = []
        self._event_probabilities: list[dict[str, Any]] = []
        # Carry risk rows: empty unless the run resolved recording on.
        self._hedge_legs: list[dict[str, Any]] = []
        self._hedge_attribution: list[dict[str, Any]] = []
        self._hedge_stresses: list[dict[str, Any]] = []

        self._initial_book_value: Optional[float] = None
        self._transaction_costs: float = 0.0
        self._start_date: Optional[pd.Timestamp] = None

        self._calibration_records: list[dict[str, Any]] = []
        # Per-product per-day rows (position_id-ordered within each day);
        # the single-product wrapper builds its legacy frames from these.
        self._product_daily: list[dict[str, Any]] = []
        # Per-day vol-model calibration (mirrors the single engine): the
        # calibrator is keyed by surface artifact sha; each priced day swaps
        # every replay's pricing engine for a fresh vol-model engine wired to
        # that day's calibrated model.
        self._calibrator = None
        if config.engine_config.vol_model != "bsm":
            if getattr(config.market_data, "surface_history", None) is None:
                raise ValidationError(
                    "vol_model != 'bsm' requires market_data.surface_history "
                    "(per-day vol-model calibration is keyed by surface artifact)"
                )
            self._calibrator = VolModelCalibrator(
                config.engine_config.vol_model_calibration
            )

        # None = the flat carry channel follows the hedge contract.
        self._dividend_roll_policy = getattr(config, "dividend_roll_policy", None)

        self._replays: list[ProductReplay] = []
        self._quantities: list[float] = []
        # Pricing engines are resolved by the ENGINE and passed explicitly to
        # every pricing call; vol-model days swap this list wholesale.
        self._pricing_engines: list[Any] = []
        for bp in config.products:
            lifecycle = AutocallableLifecycleState()
            self._pricing_engines.append(
                create_pricing_engine(
                    bp.product,
                    config.engine_config,
                    delta_bump_size=config.delta_bump_size,
                    gamma_bump_size=config.gamma_bump_size,
                )
            )
            surface_engine = create_surface_engine(bp.product, config.engine_config)
            event_stats_engine = create_event_stats_engine(
                bp.product, config.engine_config
            )
            replay = ProductReplay(
                product=bp.product,
                product_quantity=bp.quantity,
                has_lifecycle=bp.has_lifecycle,
                lifecycle=lifecycle,
                surface_engine=surface_engine,
                event_stats_engine=event_stats_engine,
                engine_config=config.engine_config,
                market_data=config.market_data,
                start_date=None,
                underlying=config.underlying,
                position_id=bp.position_id,
                fixed_dividend_yield=config.fixed_dividend_yield,
                surface_config=config.surface_config,
                actions_sink=self._actions,
                event_prob_sink=self._event_probabilities,
                daily_event_sink=self._daily_event_summary,
                surfaces_sink=self._surfaces,
            )
            self._replays.append(replay)
            self._quantities.append(float(bp.quantity))

        # Optional daily PnL explain (quantark.pnlexplain); None changes nothing
        self._explain_recorder = None
        self._explain_frames = None
        if getattr(config, "pnl_explain", None) is not None:
            from quantark.pnlexplain.equity.recorder import ReplayPnLExplainRecorder
            self._explain_recorder = ReplayPnLExplainRecorder(config.pnl_explain)
            for replay in self._replays:
                replay.record_events = True

    @property
    def hedge_position(self) -> FuturesHedgePosition:
        """The legacy one-leg VIEW of the hedge book.

        It raises for a multi-leg book on purpose: a bucket path that reached
        for this would silently report one contract as if it were the hedge.
        Bucket code uses ``hedge_book`` and selected-leg quantities directly.
        """
        return self.hedge_book.single_leg

    def _selected_contracts(self, selected) -> float:
        """Quantity held in the SELECTED reference contract.

        This is what the legacy scalar columns have always meant.  Under a
        bucket strategy it stays the selected leg, not the whole hedge, and
        the new risk frame is the authoritative joint measurement.
        """
        if selected is None:
            return 0.0
        return self.hedge_book.quantity(str(selected["contract"]))

    def _chain_rows(self, futures_slice) -> dict:
        """Today's whole chain as ``contract -> row``."""
        return {
            str(row["contract"]): row for _, row in futures_slice.iterrows()
        }

    def _hedge_marks(self, selected, chain_rows) -> dict:
        """Marks for every OPEN leg.

        Bucket mode requires each leg's own quote.  The single-leg path keeps
        the historical behaviour of marking the open leg at the selected
        contract's price, which is the same number because the roll has
        already moved the position onto that contract.
        """
        open_legs = self.hedge_book.contracts()
        if not open_legs:
            return {}
        if self._is_bucket:
            missing = [c for c in open_legs if c not in chain_rows]
            if missing:
                raise ValidationError(
                    f"missing futures mark for held leg(s) {missing}; a stale "
                    "price is never substituted"
                )
            return {c: float(chain_rows[c]["futures_price"]) for c in open_legs}
        price = float(selected["futures_price"])
        return {c: price for c in open_legs}

    def run(self) -> "BookBacktestResults":
        dates = self._backtest_dates()
        if len(dates) == 0:
            raise ValidationError("No common market-data dates for backtest")
        self._start_date = pd.Timestamp(dates[0]).normalize()
        for replay in self._replays:
            replay.start_date = self._start_date

        current_contract: Optional[str] = None
        # The carry leg rolls on its OWN policy when the run decouples it.
        current_dividend_contract: Optional[str] = None
        self._days_replayed = 0
        self._days_in_contract = int(len(dates))
        self._terminated_all_settled = False
        for date in dates:
            date = pd.Timestamp(date).normalize()
            # PnL explain: every trade from here on (rolls included) belongs to this day
            trades_before = len(self._trades)
            market = self.config.market_data.get_market_row(date)

            if self.hedge.kind == "futures":
                futures_slice = self.config.market_data.get_futures_slice(date)
                selected = self.hedge.roll_policy.select_contract(
                    futures_slice, date, current_contract
                )
                if current_contract != str(selected["contract"]):
                    # A bucket hedge holds several legs and closes a retired
                    # one through the validated rebalance plan.  The legacy
                    # roll trades BEFORE pricing and has a missing-old-quote
                    # fallback, so selecting a reference contract must not
                    # drag a multi-leg book through it.
                    if not self._is_bucket:
                        self._roll_contract(
                            date, selected, futures_slice, current_contract
                        )
                    current_contract = str(selected["contract"])
                if self._dividend_roll_policy is not None:
                    dividend_row = self._dividend_roll_policy.select_contract(
                        futures_slice, date, current_dividend_contract
                    )
                    current_dividend_contract = str(dividend_row["contract"])
                else:
                    dividend_row = None
            else:
                selected = self._spot_selected(date, market)
                dividend_row = None

            multiplier = float(selected["multiplier"])

            # env is product-independent: build it once from any replay.
            #
            # A bucket run whose whole book is already dead still has to
            # settle known cash and discount a pending receivable, and that
            # needs spot, rates and the date -- not live carry quotes.  On
            # such a day, and only then, a chain with no eligible contract
            # falls back to the flat carry channel.  A day with a live
            # product always requires the proper curve; substituting a zero
            # initial PV to evade missing data is never acceptable.
            already_dead = (
                self._is_bucket
                and self._initial_book_value is not None
                and not any(r.lifecycle.alive for r in self._replays)
            )
            env, basis_yield, implied_q, futures_ttm = self._replays[0].build_env(
                date,
                market,
                selected,
                dividend_row,
                allow_flat_carry_fallback=already_dead,
            )
            if self.hedge.kind == "spot":
                # For spot hedges, build_env synthesises a 100-year future
                # (futures_price == spot), which collapses implied_q ≈ rate.
                # That is economically wrong: override with an explicit dividend
                # yield so the pricer receives a dividend, not the risk-free rate.
                # A term dividend source never touched that synthetic future,
                # so it keeps its curve; only the recorded scalar is reset.
                implied_q = (
                    float(self.config.fixed_dividend_yield)
                    if self.config.fixed_dividend_yield is not None
                    else 0.0
                )
                basis_yield = 0.0
                if not self._replays[0].uses_term_dividend_source():
                    env.div_yield = SignedDividendYield(implied_q)
                env.basis_yield = ImpliedBasisYield(0.0)
            pricing_q = self._replays[0].recorded_pricing_q(
                env, date, market, implied_q
            )

            # Vol-model variants: calibrate once per surface artifact and swap
            # every replay's engine before ANY pricing of the day (initial
            # price, base price, bumped greeks) so the whole day is
            # model-consistent. Skipped once all products are dead.
            day_calibration_record = None
            if self._calibrator is not None and (
                any(r.lifecycle.alive for r in self._replays)
                or self._initial_book_value is None
            ):
                day_calibration_record = self._calibrate_day(date)
            pricing_started = time.perf_counter()

            # PnL explain: capture today's alive contracts and engines BEFORE lifecycle events
            if self._explain_recorder is not None:
                self._explain_recorder.begin_day(self, date, env)

            # Initial book value: priced BEFORE lifecycle on the first day,
            # mirroring the single engine's pre-lifecycle initial value.
            if self._initial_book_value is None:
                initial_book_value = 0.0
                for bp, replay, engine in zip(
                    self.config.products, self._replays, self._pricing_engines
                ):
                    if bp.initial_price is not None:
                        initial_price = float(bp.initial_price)
                    else:
                        product = replay.product_for_date(date, env)
                        initial_price = float(engine.price(product, env))
                    initial_book_value += float(bp.quantity) * initial_price
                self._initial_book_value = initial_book_value

            net_position_delta = 0.0
            net_position_gamma = 0.0
            book_product_mtm = 0.0
            book_cashflows = 0.0
            # Each surviving product's own resolved engine, priced product and
            # unit delta, so the carry measurement uses the SAME calibrated
            # engine as the day's pricing rather than the factory default.
            alive_specs: list[tuple] = []

            for bp, quantity, replay, engine in zip(
                self.config.products, self._quantities, self._replays,
                self._pricing_engines,
            ):
                lifecycle_product = replay.product_for_lifecycle()
                replay.apply_lifecycle_events(
                    date, lifecycle_product, env, market["spot"]
                )
                replay.settle_maturity_if_due(
                    date, lifecycle_product, env, market["spot"]
                )
                replay.settle_pending_if_due(date)

                price = 0.0
                greeks: dict[str, float] = {"price": 0.0, "delta": 0.0, "gamma": 0.0}
                if replay.lifecycle.alive:
                    product = replay.product_for_date(date, env)
                    price = float(engine.price(product, env))
                    greeks = replay.calculate_greeks(product, env, engine=engine)
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
                    alive_specs.append(
                        (
                            quantity,
                            replay,
                            engine,
                            product,
                            float(greeks.get("delta", 0.0)),
                            price,
                        )
                    )

                provenance = getattr(replay, "last_surface_provenance", None)
                self._product_daily.append(
                    {
                        "date": date,
                        "position_id": bp.position_id,
                        "price": price,
                        "greeks": dict(greeks),
                        "alive": replay.lifecycle.alive,
                        "knocked_in": replay.lifecycle.knocked_in,
                        "knocked_out": replay.lifecycle.knocked_out,
                        "matured": replay.lifecycle.matured,
                        "realized_cashflows": replay.lifecycle.realized_cashflows,
                        "provenance": dict(provenance) if provenance else None,
                    }
                )

                book_cashflows += replay.lifecycle.realized_cashflows

            any_alive = any(replay.lifecycle.alive for replay in self._replays)
            book_receivable_pv = 0.0
            for replay in self._replays:
                book_receivable_pv += replay.pending_receivable_pv(env, date)

            if day_calibration_record is not None:
                day_calibration_record["pricing_seconds"] = (
                    time.perf_counter() - pricing_started
                )
                self._calibration_records.append(day_calibration_record)

            chain_rows = (
                self._chain_rows(futures_slice)
                if self.hedge.kind == "futures"
                else {}
            )
            # Carried holdings, captured BEFORE any rebalance: the day's P&L
            # belongs to what the book actually held overnight.
            carried_holdings = dict(self.hedge_book.holdings())
            pre_hedge_contracts = self._selected_contracts(selected)
            day_risk = None
            hedge_targets = None
            if self._is_bucket:
                context = self._replays[0].last_carry_context
                if any_alive:
                    day_risk = self._measure_book_carry_risk(
                        date, env, context, alive_specs
                    )
                hedge_targets = self._rebalance_buckets(
                    date=date,
                    selected=selected,
                    chain_rows=chain_rows,
                    risk=day_risk,
                    any_alive=any_alive,
                    carried=carried_holdings,
                )
            else:
                self._rebalance(
                    date,
                    selected,
                    net_position_delta,
                    multiplier,
                    any_alive,
                    spot=float(market["spot"]),
                )
            self._record_day(
                date=date,
                selected=selected,
                chain_rows=chain_rows,
                carried_holdings=carried_holdings,
                dividend_row=dividend_row,
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
                receivable_pv=book_receivable_pv,
            )
            # PnL explain: close the day against the recorded state
            if self._explain_recorder is not None:
                self._explain_recorder.end_day(
                    self, date, env, market, selected, self._trades[trades_before:], self._states[-1],
                )
            self._days_replayed += 1
            if self.config.terminate_on_lifecycle_end and all(
                r.lifecycle.settled for r in self._replays
            ):
                # Terminal cash has landed for every product: the replay ends
                # on the later of observation and settlement (study spec §6).
                self._terminated_all_settled = True
                break

        self._explain_frames = (
            self._explain_recorder.frames() if self._explain_recorder is not None else None
        )
        return BookBacktestResults(
            config=self.config,
            explain_frames=self._explain_frames,
            calibration_records=self._calibration_records,
            run_info=self._run_info(),
            states=self._states,
            greeks=self._greeks,
            rebalances=self._rebalances,
            trades=self._trades,
            actions=self._actions,
            daily_event_summary=self._daily_event_summary,
            event_probabilities=self._event_probabilities,
            surfaces=self._surfaces,
            hedge_legs=self._hedge_legs,
            hedge_attribution=self._hedge_attribution,
            hedge_stresses=self._hedge_stresses,
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

    def _run_info(self) -> dict[str, Any]:
        """Termination provenance for the summary (study spec §6).

        Outcome labels (ko / ki_maturity / maturity) apply only when every
        leg actually settled inside the data window; a run that exhausted
        market data with an open receivable is "data_end" — anything else
        would hide truncated settlement exposure.
        """
        all_settled = all(r.lifecycle.settled for r in self._replays)
        if not all_settled:
            reason = "data_end"
        elif any(r.lifecycle.knocked_out for r in self._replays):
            reason = "ko"
        elif all(r.lifecycle.matured for r in self._replays) and any(
            r.lifecycle.knocked_in for r in self._replays
        ):
            reason = "ki_maturity"
        elif all(r.lifecycle.matured for r in self._replays):
            reason = "maturity"
        else:
            reason = "data_end"
        return {
            "termination_reason": reason,
            "days_replayed": int(self._days_replayed),
            "days_in_contract": int(self._days_in_contract),
            "all_settled": bool(all_settled),
            "outstanding_receivable": float(
                sum(
                    r.lifecycle.pending_settlement_cashflow
                    for r in self._replays
                )
            ),
        }

    def _calibrate_day(self, date: pd.Timestamp) -> dict[str, Any]:
        """Calibrate the day's vol model and swap in per-replay day engines.

        One calibration per day (the surface artifact is shared across the
        book); each replay gets a fresh engine built from the same frozen
        CalibratedVolModel — engine construction is cheap, calibration is the
        cached expensive step. Any failure propagates (fail-closed).
        """
        engine_config = self.config.engine_config
        artifact = self.config.market_data.surface_history.surface_for(date)
        calibrated = self._calibrator.calibrate(engine_config.vol_model, artifact)
        self._pricing_engines = [
            create_vol_model_engine(
                vol_model=engine_config.vol_model,
                solver=engine_config.vol_model_solver,
                calibrated=calibrated,
                pde_params=engine_config.pde_params,
                mc_params=engine_config.mc_params,
                mc_method=engine_config.resolve_vol_model_mc_method(),
                engine_options=engine_config.vol_model_engine_options,
                delta_bump_size=self.config.delta_bump_size,
                gamma_bump_size=self.config.gamma_bump_size,
            )
            for _ in self._replays
        ]
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
        spot: float = 0.0,
    ) -> None:
        is_spot = self.hedge.kind == "spot"
        # Guarded single-contract path: at most one leg is ever open here.
        current_contracts = self.hedge_position.quantity
        target = 0.0
        reason = "inside_band"
        trade_type = "hedge_rebalance"
        if any_alive:
            if is_spot:
                # Spot mode: always target full delta-neutral; strategy.hedge_ratio
                # and target_delta are not applied — only delta_threshold (band) is.
                target = -float(net_position_delta)
            elif isinstance(self.strategy, ProportionalFuturesDeltaHedgeStrategy):
                # The S/F control needs both prices; the legacy strategy's
                # call signature below is deliberately left alone.
                target = self.strategy.target_contracts(
                    product_delta=float(net_position_delta),
                    product_quantity=1.0,
                    futures_multiplier=multiplier,
                    spot=float(spot),
                    futures_price=float(selected["futures_price"]),
                )
            else:
                target = self.strategy.target_contracts(
                    product_delta=float(net_position_delta),
                    product_quantity=1.0,
                    futures_multiplier=multiplier,
                )
            should_rebalance = self.strategy.should_rebalance(
                current_contracts, target
            )
            if should_rebalance:
                reason = "delta_rebalance"
        else:
            should_rebalance = abs(current_contracts) > 1e-12
            if should_rebalance:
                reason = "product_terminated"
                trade_type = "hedge_close"

        trade_contracts = target - current_contracts
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
                # Historical quirk, preserved: the decision above uses the
                # PRE-trade quantity, but this column has always recorded the
                # POST-trade one, because the row was built after the trade.
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

    # ------------------------------------------------------------------
    # Multi-leg bucket path
    # ------------------------------------------------------------------

    def _measure_book_carry_risk(self, date, env, context, alive_specs):
        """Aggregate the book's signed carry risk on today's coordinates.

        The context is passed EXPLICITLY to every product.  The shared
        environment is built through replay zero alone, so reading
        ``last_carry_context`` off each replay would leave every product but
        the first without buckets.
        """
        from .carry_risk import aggregate_book_risk

        if context is None:
            raise ValidationError(
                f"bucket hedge needs a futures carry context on {date.date()}; "
                "the day's dividend source produced none"
            )
        if not alive_specs:
            raise ValidationError("no live product to measure carry risk for")
        settings = self._carry_recording.settings
        points = float(settings.futures_bump_points)
        entries = []
        for quantity, replay, engine, product, unit_delta, price in alive_specs:
            entries.append(
                (
                    quantity,
                    replay.measure_carry_risk(
                        product,
                        env,
                        context=context,
                        engine=engine,
                        delta_q=unit_delta,
                        points=points,
                        base_price=price,
                    ),
                )
            )
        return aggregate_book_risk(entries)

    def _bucket_targets(self, date, risk, carried, any_alive):
        """Today's leg plan, or all-zero targets once the book is dead."""
        if not any_alive or risk is None:
            # No feasibility question to ask and no pricer to invoke: a dead
            # book closes whatever it holds, even with one eligible node.
            return None, {c: 0.0 for c in carried}
        try:
            targets = self.strategy.target_legs(
                buckets=risk.buckets,
                delta_q=risk.delta_q,
                spot=risk.spot,
                held_contracts=carried,
            )
        except ValidationError as error:
            self._hedge_infeasibility = {
                "date": str(date),
                "objective": getattr(self.strategy, "objective", None),
                "correction_pair": getattr(self.strategy, "correction_pair", None),
                "eligible_contracts": list(risk.contracts),
                "cause": str(error),
            }
            raise ValidationError(
                f"bucket hedge infeasible on {date.date()}: "
                f"objective={getattr(self.strategy, 'objective', None)!r}, "
                f"eligible={list(risk.contracts)}: {error}"
            ) from error
        return targets, dict(targets.rounded)

    def _rebalance_buckets(
        self, *, date, selected, chain_rows, risk, any_alive, carried
    ):
        """Plan the WHOLE rebalance, validate it, then commit it at once.

        Nothing touches the ledger, the cost accumulator or the trade log
        until every mark, multiplier and transaction cost for every leg has
        been produced.  A cost model that raises on the second leg leaves no
        first-leg trade behind.
        """
        targets, planned = self._bucket_targets(date, risk, carried, any_alive)
        eligible = set(risk.contracts) if risk is not None else set()
        contracts = sorted(set(planned) | set(carried))

        def order_key(contract):
            row = chain_rows.get(contract)
            expiry = row["expiry_date"] if row is not None else pd.Timestamp.max
            # Retired closes first, then eligible rebalances; both by expiry.
            return (0 if contract not in eligible else 1, expiry, contract)

        plan: list[_PlannedTrade] = []
        decisions: list[dict[str, Any]] = []
        for contract in sorted(contracts, key=order_key):
            current = float(carried.get(contract, 0.0))
            target = float(planned.get(contract, 0.0))
            delta = target - current
            retired = contract not in eligible
            reason = "delta_rebalance"
            trade_type = "hedge_rebalance"
            if not any_alive:
                # A dead book has no eligible universe at all, so this test
                # comes FIRST: every leg closes because the product ended,
                # not because it left a curve nobody is looking at.
                reason = "product_terminated"
                trade_type = "hedge_close"
            elif retired:
                reason = "bucket_leg_retired"
                trade_type = "hedge_close"
            should = abs(delta) > 1e-12
            if should and not retired and any_alive:
                # A retired close and a terminated book both bypass the band.
                should = self.strategy.should_rebalance(current, target)
            if should and abs(delta) > 1e-12:
                row = chain_rows.get(contract)
                if row is None:
                    raise ValidationError(
                        f"no tradable mark for {contract!r} on {date.date()}; "
                        "a bucket rebalance never substitutes a stale price"
                    )
                price = float(row["futures_price"])
                multiplier = float(row["multiplier"])
                if not (price > 0.0 and multiplier > 0.0):
                    raise ValidationError(
                        f"invalid mark for {contract!r} on {date.date()}: "
                        f"price={price}, multiplier={multiplier}"
                    )
                notional = abs(delta * price * multiplier)
                cost = float(
                    self.config.transaction_cost_model.calculate_cost(
                        quantity=delta,
                        price=price,
                        notional=notional,
                        instrument_type="futures",
                        trade_type=trade_type,
                    )
                )
                if not math.isfinite(cost):
                    raise ValidationError(
                        f"transaction cost for {contract!r} is not finite: {cost}"
                    )
                plan.append(
                    _PlannedTrade(
                        date=date,
                        contract=contract,
                        quantity_delta=delta,
                        price=price,
                        multiplier=multiplier,
                        notional=notional,
                        cost=cost,
                        trade_type=trade_type,
                        reason=reason,
                    )
                )
            decisions.append(
                {
                    "date": date,
                    "active_contract": contract,
                    "current_contracts": current,
                    "target_contracts": target,
                    "trade_contracts": delta if should else 0.0,
                    "should_rebalance": bool(should),
                    "threshold_status": (
                        "outside_band" if should else "inside_band"
                    ),
                    "no_trade_reason": None if should else "inside_band",
                    "reason": reason,
                }
            )
        self._commit_bucket_trades(plan)
        # A skipped trade still leaves a decision row, so the no-trade error
        # stays measurable against the target the policy actually planned.
        self._rebalances.extend(decisions)
        return targets

    def _commit_bucket_trades(self, planned_trades) -> None:
        trial = self.hedge_book.copy()
        next_costs = self._transaction_costs
        rows = []
        for trade in planned_trades:
            trial.trade(
                trade.contract, trade.quantity_delta, trade.price, trade.multiplier
            )
            next_costs += trade.cost
            rows.append(trade.to_row())
        self.hedge_book = trial
        self._transaction_costs = next_costs
        self._trades.extend(rows)

    def _roll_contract(
        self,
        date: pd.Timestamp,
        selected,
        futures_slice: pd.DataFrame,
        current_contract: Optional[str],
    ) -> None:
        if self.hedge.kind == "spot":
            return
        position = self.hedge_position
        if abs(position.quantity) < 1e-12:
            return
        old_contract = position.contract or current_contract
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
        qty = position.quantity
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
        self.hedge_book.trade(contract, quantity_delta, price, multiplier)
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
        chain_rows: Optional[dict] = None,
        carried_holdings: Optional[dict] = None,
        dividend_row=None,
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
        receivable_pv: float = 0.0,
    ) -> None:
        futures_price = float(selected["futures_price"])
        spot = float(market["spot"])
        chain_rows = chain_rows or {}
        # Bucket mode marks every open leg at its OWN quote; one selected
        # price must never be used to mark several contracts.
        hedge_mtm = self.hedge_book.mark_to_market(
            self._hedge_marks(selected, chain_rows)
        )
        selected_contracts = self._selected_contracts(selected)
        # book_product_mtm already only sums alive products.
        product_mtm = book_product_mtm
        # A pending terminal receivable (delayed KO settlement) is carried at
        # its discounted value in BOTH portfolio value and marked P&L — else
        # P&L would drop at observation and jump at settlement (phantom P&L).
        # Zero under T+0 settlement (golden-invariant). Identity preserved:
        # total_pnl == portfolio_value - initial_book_value.
        product_pnl = (
            product_mtm
            + receivable_pv
            + book_cashflows
            - float(self._initial_book_value or 0.0)
        )
        total_pnl = product_pnl + hedge_mtm - self._transaction_costs
        cash = book_cashflows - self._transaction_costs
        portfolio_value = product_mtm + hedge_mtm + cash + receivable_pv
        product_position_delta = float(net_position_delta)
        product_position_gamma = float(net_position_gamma)
        if self._is_bucket:
            carried = carried_holdings or {}
            pre_hedge_futures_delta = sum(
                float(carried.get(c, 0.0))
                * float(chain_rows[c]["multiplier"])
                * float(chain_rows[c]["futures_price"])
                / spot
                for c in carried
                if c in chain_rows
            )
        else:
            pre_hedge_futures_delta = float(pre_hedge_contracts) * multiplier
        if self._is_bucket:
            # The complete book's currency spot sensitivity, sum h_i m_i F_i/S.
            # It cannot be reconstructed from one selected multiplier and
            # count; the hedge_legs frame is the authoritative measurement.
            post_hedge_futures_delta = self.hedge_book.spot_delta(
                self._hedge_marks(selected, chain_rows), spot
            )
        else:
            # Unchanged single-contract approximation for the legacy rows.
            post_hedge_futures_delta = selected_contracts * multiplier
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
                "futures_contracts": selected_contracts,
                "alive": any_alive,
                "knocked_in": knocked_in,
                "knocked_out": knocked_out,
                "matured": matured,
                # Settlement decomposition (appended so the schema prefix is
                # stable): product_mtm keeps its historical contingent-only
                # meaning; the receivable rides in product_pnl/portfolio_value.
                "contingent_product_mtm": book_product_mtm,
                "pending_receivable_pv": receivable_pv,
                "paid_cash": book_cashflows,
                # Appended last, matching STATE_COLUMNS: the contract the flat
                # carry was inverted from (== active_contract by default).
                "dividend_contract": str(
                    (dividend_row if dividend_row is not None else selected)["contract"]
                ),
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
                "post_hedge_contracts": selected_contracts,
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


# Compatible alias (canonical name above).
BookAutocallableBacktestEngine = ReplayBacktestEngine
