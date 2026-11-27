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

import logging

import dataclasses
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any, Dict, NamedTuple, Optional

import numpy as np
import pandas as pd

from quantark.asset.equity.engine.event_stats import checked_event_statistics
from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.engine.event_stats import EventStatistics, StatisticsRefused
from quantark.asset.equity.lifecycle import DCNLifecycleState, ValuationPoint, tracker_for
from quantark.asset.equity.market import IndexFuturesQuote
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div.forward_carry_curve import ForwardCarryCurve
from quantark.priceenv import CurveQuotingBasis, PricingEnvironment
from quantark.priceenv.clock_validation import is_clock_wrapped
from quantark.util.calendar import BusinessDayConvention, DayCountConvention
from quantark.util.exceptions import PricingError, ValidationError
from quantark.util.fixings import freeze_missing_fixings
from quantark.util.numerical import is_close, is_zero, same_instant
from quantark.util.run_fixings import participates, step_fixings, update_live

from .carry_context import SUPPORTED_EXTRAPOLATIONS, CarryCurveContext
from .dividend_source import term_dividend_yield
from .engine_factory import create_mc_event_stats_engine
from .market import (
    ClockImpliedBasisYield,
    ClockSignedDividendYield,
    ImpliedBasisYield,
    SignedDividendYield,
    derive_implied_dividend_yield,
)

logger = logging.getLogger(__name__)


def _available(stats: EventStatistics, path: str) -> Any:
    """The statistic at ``path`` read through ``EventStatistics.require``, or ``None`` when it is flagged unavailable."""
    if stats.flag(path) is not None:
        return None
    return stats.require(path)


def _first_instant_probability(ko_times: np.ndarray, ko_probs: np.ndarray) -> float:
    """P(knock out at the first knock-out instant): the sum over every knock-out row ON the first row's instant.

    Records on one env instant (e.g. a US BUSINESS_DAYS holiday collapsed onto the day before) keep one statistics row
    each (0.16.0 DCN/dated D8/D14); the instant's knock-out probability is their sum. A single row is read as is.
    """
    first = float(ko_times[0])
    rows = [i for i, t in enumerate(ko_times) if same_instant(t, first)]
    if len(rows) == 1:
        return float(ko_probs[0])
    return float(np.sum(ko_probs[rows]))


def _env_with(
    env: PricingEnvironment,
    *,
    spot: Optional[float] = None,
    div_yield: Any = None,
    underlying: str = "",
) -> PricingEnvironment:
    """Shallow env variant: shares the (immutable-by-convention) vol surface,
    rate curve, and basis yield; replaces only spot/dividend. Replaces the
    deepcopy-per-node pattern, which cloned entire smile-grid surfaces.

    ``dataclasses.replace`` keeps every other field -- the whole clock included (run 7 D4): on a business-day env the
    shared wrappers stay as they are (same anchor and horizon) and a native replacement dividend is wrapped under the
    env's ``CALENDAR_YEAR`` declaration. On a calendar-day env it is the same objects as before.
    """
    return dataclasses.replace(
        env,
        spot_quote=(
            SpotQuote(spot=float(spot), asset_name=underlying)
            if spot is not None
            else env.spot_quote
        ),
        div_yield=div_yield if div_yield is not None else env.div_yield,
    )


class _EnvClock(NamedTuple):
    """The clock keywords every replay day env is built with (run 7 D4): passed one by one, never splatted."""

    day_count_convention: DayCountConvention
    bus_days_in_year: int
    calendar: Any
    holiday_time_adjustment: BusinessDayConvention
    rate_curve_quoting: Optional[CurveQuotingBasis]
    div_yield_quoting: Optional[CurveQuotingBasis]
    basis_yield_quoting: Optional[CurveQuotingBasis]
    trading_clock_horizon: Optional[datetime]


#: ``trading_clock=None``: exactly ``PricingEnvironment``'s own defaults, so the day env is today's calendar-day env.
_CALENDAR_DAY_CLOCK = _EnvClock(**{
    f.name: f.default for f in dataclasses.fields(PricingEnvironment) if f.name in _EnvClock._fields
})


def _naive_day(value) -> Optional[datetime]:
    if value is None or not hasattr(value, "year"):
        return None
    return datetime(value.year, value.month, value.day)


def _product_contract_dates(product: Any, calendar: Any = None) -> list:
    """Every contract date of ``product`` the replay's engines may resolve: exercise / maturity / settlement dates and
    every schedule record's dates (both barrier configs, a plain observation schedule); a DCN's schedule and payment
    dates; and the payment date the product's day-based settlement convention (T+n) derives from the latest of them
    (on the convention's calendar, else ``calendar``)."""
    dates = [getattr(product, name, None) for name in ("exercise_date", "maturity_date", "settlement_date")]
    schedules = []
    for config_name in ("barrier_config", "post_barrier_config"):
        config = getattr(product, config_name, None)
        for schedule_name in ("ko_observation_schedule", "ki_observation_schedule"):
            schedules.append(getattr(config, schedule_name, None))
    schedules.append(getattr(product, "observation_schedule", None))
    records = [r for sch in schedules for r in (getattr(sch, "records", None) or ())]
    records += list(getattr(product, "observation_records", None) or ())
    for record in records:
        dates += [getattr(record, name, None)
                  for name in ("observation_date", "settlement_date", "maturity_date")]
    schedule = getattr(product, "schedule", None)
    if schedule is not None and callable(getattr(product, "payment_dates", None)):
        # a DCN: its monthly rows, daily knock-in dates and every payment date
        dates += [m.observation_date for m in schedule.monthly] + list(schedule.daily_ki_dates)
        paid = product.payment_dates()
        dates += list(paid.coupon_payment_dates) + list(paid.ko_payment_dates) + [paid.loss_payment_date]
    out = [d for d in (_naive_day(x) for x in dates) if d is not None]
    convention = getattr(product, "settlement_convention", None)
    if out and convention is not None and float(convention.lag) > 0.0:
        from quantark.asset.equity.settlement import SettlementLagUnit, payment_date_by_convention

        if convention.lag_unit in (SettlementLagUnit.BUSINESS_DAYS, SettlementLagUnit.CALENDAR_DAYS):
            # the latest determination pays last: a day-based lag rolls forward monotonically
            out.append(_naive_day(payment_date_by_convention(convention, max(out), convention.calendar or calendar)))
    return out


def trading_clock_horizon(market_data: Any, products, calendar: Any = None) -> datetime:
    """The replay's ``trading_clock_horizon`` (run 7 D4, revised): max(the last market date, the latest contract date
    of every product) + 366 calendar days -- strictly after every date an engine resolves (settlement lags, holiday
    rolls, pending receivables). A date past it still raises in the env's clock machinery, never silently."""
    candidates = [_naive_day(pd.Timestamp(d)) for d in market_data.dates]
    for product in products:
        candidates += _product_contract_dates(product, calendar)
    if not candidates:
        raise ValidationError("trading_clock_horizon needs at least one market date")
    return max(candidates) + timedelta(days=366)


def surface_tail_carry_yield(
    *,
    spot: float,
    forward_nodes: list[tuple[float, float]],
    artifact,
    rate_curve,
):
    """Dividend view of the IM chain in log-forward space with an option-forward tail.

    ``forward_nodes`` are the eligible contracts' ``(T_i, F_i)``; they become
    ``B(T_i) = ln(F_i / spot)`` nodes of a ``ForwardCarryCurve`` (piecewise
    linear in B, i.e. piecewise-constant forward carry between contracts:
    the calendar-spread slope).  Beyond the last contract the artifact's
    parity-forward pillars supply the forward carry, level matched at the
    join (``ForwardCarryCurve.extended_with``), then the last option segment's
    carry continues flat.  Shared by the replay engine and the standalone
    study so the two can never drift.
    """
    im = ForwardCarryCurve.from_forward_nodes(spot, forward_nodes)
    # the pillars come back as q_j = r0 - ln(F_j/s0)/T_j for the flat r0 we
    # pass, so (r0 - q_j) * T_j is exactly ln(F_j/s0) whatever r0 is: the
    # option curve is pure forward carry, independent of the rate channel
    r0 = float(rate_curve.get_rate(1.0))
    times, yields = artifact.implied_q_pillars(rate=r0)
    option = ForwardCarryCurve(
        [(float(t), (r0 - float(q)) * float(t)) for t, q in zip(times, yields)]
    )
    return im.extended_with(option).to_dividend_yield(rate_curve)


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
        surface_config: Any = None,
        position_id: Optional[Any] = None,
    ) -> None:
        if callable(getattr(product, "with_history_settled", None)):
            # TODO(tarn-replay): replay a product holding run state - call run_fixings.settle_history after
            # step_fixings and book its FIXING / PAYMENT cash - before admitting FX TARN / TARF here
            raise ValidationError(
                f"ProductReplay cannot replay {type(product).__name__}: it holds run state (FX TARN / TARF) that "
                "run_fixings.settle_history banks and pays, which ProductReplay does not call; run it in the FX "
                "backtest or FX dynamic scenario")
        clock = getattr(engine_config, "trading_clock", None)
        if clock is not None:
            from quantark.asset.clock import ProductClock, product_clock
            from quantark.asset.equity.product.option.dcn_option import DCNOption

            # a DCN is date-only by its own clock rule (it carries no exercise_date for product_clock to read)
            if not isinstance(product, DCNOption) and product_clock(product) is ProductClock.NUMERIC:
                # TODO(run 7 D7): a numeric (float-time) schedule on the business-day clock -- its times ignore the
                # env clock, so the day would mix ACT/365 product times with trading-time curves.
                raise ValidationError(
                    f"{type(product).__name__} has a numeric (float-time) schedule; engine_config.trading_clock "
                    "prices on a business-day clock that a float schedule ignores (mixed semantics). Book the "
                    "contract on dates, or replay it with trading_clock=None"
                )
        self.product = product
        self.product_quantity = product_quantity
        self.has_lifecycle = has_lifecycle
        self.lifecycle = lifecycle
        self.position_id = position_id
        self.surface_engine = surface_engine
        self.event_stats_engine = event_stats_engine
        self.engine_config = engine_config
        self.market_data = market_data
        self.underlying = underlying
        self.fixed_dividend_yield = fixed_dividend_yield
        self.surface_config = surface_config

        self.actions_sink = actions_sink
        self.event_prob_sink = event_prob_sink
        self.daily_event_sink = daily_event_sink
        self.surfaces_sink = surfaces_sink

        # PnL explain support: the recorder-owning engine flips record_events
        # on and the recorder resets events_today at the start of every engine
        # date; an explain-off run never touches events_today.
        self.record_events: bool = False
        self.events_today: list = []

        # Provenance of the IV-surface artifact used by the most recent
        # surface-mode build_env call (None in scalar mode).  The engine
        # folds this into the per-day state row when present.
        self.last_surface_provenance: Optional[dict[str, Any]] = None

        # The carry scenario source of the most recent build_env call, for the
        # two supported actual-futures conventions only (None otherwise).  The
        # engine passes ONE day context explicitly to every product's risk
        # call; it must not read this attribute off another replay, because a
        # shared environment is built through the first replay alone.
        self.last_carry_context: Optional[CarryCurveContext] = None

        # date_resolver captures self; do not replace self.market_data
        # post-construction or the resolver will keep using the old one.
        # The tracker the product's rules need (a KO-reset has its own); with ``lifecycle=None`` it builds
        # the starting state from the product's carried attributes (spec 2026-10-08 section 4).
        self._tracker = tracker_for(
            product,
            quantity=product_quantity,
            lifecycle=lifecycle,
            start_date=start_date,
            date_resolver=self._next_available_market_date,
            has_lifecycle=has_lifecycle,
        )
        self.lifecycle = self._tracker.lifecycle
        # The missing past fixings decided at the run's first valuation ({record index: outcome}); None until then
        self._run_fills: Optional[dict] = None
        self._last: Optional[tuple] = None  # (date, spot) of the last date product_for_date first saw
        self._previous: Optional[tuple] = None  # (date, spot) of the run instant before ``_last`` (W6 gap rule)
        self._live: Dict[int, tuple] = {}  # index -> (date, spot) first live (Ruling 4)
        # Business-day clock (run 7 D4): the horizon every day env of the run shares. This product's own; a book
        # engine replaces it with the book-wide one (one horizon per run).
        self.trading_clock_horizon: Optional[datetime] = (
            trading_clock_horizon(market_data, [product], clock.calendar) if clock is not None else None
        )

    @property
    def start_date(self) -> Optional[pd.Timestamp]:
        return self._tracker.start_date

    @start_date.setter
    def start_date(self, value: Optional[pd.Timestamp]) -> None:
        self._tracker.start_date = value

    @property
    def tracker(self):
        """The product's lifecycle tracker (read-only; the engine's cash-leg books read its clock and state)."""
        return self._tracker

    def retire_if_matured_before_run(self, env: PricingEnvironment) -> bool:
        """The tracker's ``retire_if_matured_before_run``: a contract matured before the run is dead from its
        start, with no cash and a zero value (user ruling 2026-10-09)."""
        return self._tracker.retire_if_matured_before_run(env)

    def product_for_lifecycle(self):
        return self._tracker.product_for_lifecycle()

    def product_for_date(self, date: pd.Timestamp, pricing_env: PricingEnvironment, *, initial: bool = False):
        """The pricing copy for ``date``; ``initial`` before the run's first close is observed.

        The first call is the run's first valuation: a fixing already missing is decided once at its spot (W5b,
        R5). Each later date seen for the first time records the fixings that became history since the last date at
        that date's spot (spec 2026-10-09 run fixings §3.6, R3/R4); every pricing copy carries every decision.
        """
        product = self._tracker.product_for_pricing(date, pricing_env, initial=initial)
        day = pd.Timestamp(date).normalize()
        if self._run_fills is None:
            fills_of = getattr(product, "missing_fixing_fills", None)
            self._run_fills = dict(fills_of(pricing_env)) if fills_of is not None else {}
            self._last = (day, float(pricing_env.spot))
            frozen = freeze_missing_fixings(product, pricing_env) if self._run_fills else product
            if participates(frozen):
                self._live = update_live(frozen.open_fixings(pricing_env), day, pricing_env.spot)
            return frozen
        if participates(product) and day != self._last[0]:
            seen = product.with_fixings_recorded(self._run_fills) if self._run_fills else product
            labels = seen.open_fixings(pricing_env)
            product, new = step_fixings(product, pricing_env, prev_date=self._last[0], prev_spot=self._last[1],
                                        run_fills=self._run_fills,
                                        live_spots={i: s for i, (_d, s) in self._live.items()})
            self._run_fills.update(new)
            self._live = update_live(labels, day, pricing_env.spot, before=self._live, exclude=new)
            self._previous = self._last
            self._last = (day, float(pricing_env.spot))
            return product
        if self._run_fills:
            product = product.with_fixings_recorded(self._run_fills)
        return product

    # ------------------------------------------------------------------
    # Business-day clock (run 7, .ship/2026-11-26-replay-business-day-clock)
    # ------------------------------------------------------------------

    def _trading_clock(self):
        """The run's ``TradingClockConfig``, or ``None`` for the calendar-day replay."""
        return getattr(self.engine_config, "trading_clock", None)

    def _env_clock(self, *, on_calendar: bool = False) -> _EnvClock:
        """The clock keywords of a day env: the run's business-day clock, or (no clock, or ``on_calendar`` for a
        calendar-tenor probe) ``PricingEnvironment``'s calendar-day defaults."""
        clock = self._trading_clock()
        if clock is None or on_calendar:
            return _CALENDAR_DAY_CLOCK
        quoted = CurveQuotingBasis.CALENDAR_YEAR  # replay's cash market data is calendar-quoted (D1)
        return _EnvClock(
            day_count_convention=DayCountConvention.BUSINESS_DAYS,
            bus_days_in_year=clock.bus_days_in_year,
            calendar=clock.calendar,
            holiday_time_adjustment=clock.holiday_time_adjustment,
            rate_curve_quoting=quoted,
            div_yield_quoting=quoted,
            basis_yield_quoting=quoted,
            trading_clock_horizon=self.trading_clock_horizon,
        )

    def flat_dividend(self, q: float, *, on_calendar: bool = False):
        """A flat signed dividend for a day env: ``SignedDividendYield`` on the calendar-day path (bitwise),
        ``ClockSignedDividendYield`` (a wrappable ``DividendYield``) on the business-day clock (D3)."""
        if self._trading_clock() is None or on_calendar:
            return SignedDividendYield(q)
        return ClockSignedDividendYield(q)

    def flat_basis(self, b: float, *, on_calendar: bool = False):
        """A flat basis for a day env (see ``flat_dividend``)."""
        if self._trading_clock() is None or on_calendar:
            return ImpliedBasisYield(b)
        return ClockImpliedBasisYield(b)

    def _clock_vol(self, vol_surface: Any, date: pd.Timestamp) -> Any:
        """The day's vol on the env clock (D2): a calendar-quoted vol is presented on the trading axis through a
        business-date knot map; ``vol_quoting=TRADING_YEAR`` (or no clock) keeps it as is."""
        clock = self._trading_clock()
        if clock is None or clock.vol_quoting is CurveQuotingBasis.TRADING_YEAR:
            return vol_surface
        from quantark.param.vol import CalendarQuotedVolSurface, DateKnotTimeMap
        from quantark.util.calendar.trading_clock import TradingClock

        time_map = DateKnotTimeMap(
            TradingClock(clock.calendar, clock.bus_days_in_year),
            pd.Timestamp(date).normalize().to_pydatetime(),
            self.trading_clock_horizon,
            clock.holiday_time_adjustment,
        )
        return CalendarQuotedVolSurface(vol_surface, time_map)

    def _require_clock_business_day(self, date: pd.Timestamp) -> None:
        clock = self._trading_clock()
        if clock is not None and not clock.calendar.is_business_day(pd.Timestamp(date).to_pydatetime()):
            # TODO(run 7): a replay day that is a holiday of the clock's calendar has no trading time of its own;
            # refused rather than guessed (a FOLLOWING anchor cannot even carry the calendar-quoted vol).
            raise ValidationError(
                f"replay day {pd.Timestamp(date).date()} is not a business day of the trading clock's calendar "
                f"{getattr(clock.calendar, 'name', clock.calendar)!r}; drop it from the market data or use a "
                "calendar on which it trades"
            )

    def build_env(
        self,
        date: pd.Timestamp,
        market: dict[str, float],
        selected,
        dividend_row=None,
        allow_flat_carry_fallback: bool = False,
    ):
        """The day's pricing environment.

        ``selected`` is the HEDGE contract: the returned ``basis_yield`` and
        ``futures_ttm`` describe it, because they are the state row's futures
        columns.  ``dividend_row`` is the contract the flat carry channel is
        inverted from; ``None`` means the hedge contract, which is the
        historical behaviour.  When the two differ the day costs a second
        inversion, and the returned ``implied_q`` is the dividend row's.
        """
        # Yesterday's carry source must not survive into a day that has no
        # eligible chain or a different dividend source.
        self.last_carry_context = None
        self._require_clock_business_day(date)
        expiry = pd.Timestamp(selected["expiry_date"]).normalize()
        futures_ttm = (expiry - date).days / 365.0
        basis_yield, hedge_implied_q = derive_implied_dividend_yield(
            rate=market["rate"],
            spot=market["spot"],
            futures_price=float(selected["futures_price"]),
            time_to_maturity=futures_ttm,
        )
        if dividend_row is None:
            implied_q = hedge_implied_q
        else:
            div_expiry = pd.Timestamp(dividend_row["expiry_date"]).normalize()
            _, implied_q = derive_implied_dividend_yield(
                rate=market["rate"],
                spot=market["spot"],
                futures_price=float(dividend_row["futures_price"]),
                time_to_maturity=(div_expiry - date).days / 365.0,
            )
        pricing_q = self.pricing_dividend_yield(implied_q)
        vol_surface, div_yield = self._vol_and_dividend(date, market, pricing_q)
        rate_curve = FlatRateCurve(rate=market["rate"])
        if self._dividend_source() in ("futures_curve", "surface_forwards"):
            try:
                div_yield = self._term_dividend(date, market, rate_curve)
            except ValidationError:
                # Only a caller that knows the whole book is already dead may
                # ask for this: settling known cash and discounting a pending
                # receivable need spot, rates and the date, not live carry.
                if not allow_flat_carry_fallback:
                    raise
                self.last_carry_context = None
        clock = self._env_clock()
        env = PricingEnvironment(
            spot_quote=SpotQuote(spot=market["spot"], asset_name=self.underlying),
            vol_surface=self._clock_vol(vol_surface, date),
            rate_curve=rate_curve,
            div_yield=div_yield,
            basis_yield=self.flat_basis(basis_yield),
            valuation_date=date.to_pydatetime(),
            day_count_convention=clock.day_count_convention,
            bus_days_in_year=clock.bus_days_in_year,
            calendar=clock.calendar,
            holiday_time_adjustment=clock.holiday_time_adjustment,
            rate_curve_quoting=clock.rate_curve_quoting,
            div_yield_quoting=clock.div_yield_quoting,
            basis_yield_quoting=clock.basis_yield_quoting,
            trading_clock_horizon=clock.trading_clock_horizon,
        )
        return env, basis_yield, implied_q, futures_ttm

    def _dividend_source(self) -> Optional[str]:
        return getattr(self.engine_config, "dividend_source", None)

    def uses_term_dividend_source(self) -> bool:
        return self._dividend_source() in ("futures_curve", "surface_forwards")

    def _term_dividend(self, date: pd.Timestamp, market: dict[str, float], rate_curve):
        """The day's term-structured dividend object for ``dividend_source``.

        ``futures_curve``: every contract in the day's chain with at least
        ``futures_curve_min_tenor_days`` calendar days to expiry becomes an
        ``IndexFuturesQuote`` (ACT/365 year fractions, the same day count the
        active-contract channel uses); ``IndexFuturesCurve`` inverts them
        with continuous compounding, signed (strictly increasing maturities,
        fail-closed).  ``flat_q`` holds the endpoint zero yield beyond the
        last node; ``flat_forward_carry`` continues the last segment's
        forward carry through the ``ForwardCarryCurve`` view.  One eligible
        contract is the one-node limit of both conventions: a flat
        continuous signed yield at that node.  None fails closed.

        ``surface_forwards``: the admitted artifact's parity forward pillars,
        recorded with the same provenance fields surface vol mode writes.
        """
        source = self._dividend_source()
        if source == "futures_curve":
            min_days = int(getattr(self.engine_config, "futures_curve_min_tenor_days", 7))
            if min_days < 1:
                raise ValidationError("futures_curve_min_tenor_days must be at least 1")
            chain = self.market_data.get_futures_slice(date)
            days = (pd.to_datetime(chain["expiry_date"]) - date).dt.days
            chain = chain[days >= min_days].sort_values(["expiry_date", "contract"])
            quotes = [
                IndexFuturesQuote(
                    contract=str(row["contract"]),
                    maturity=(pd.Timestamp(row["expiry_date"]) - date).days / 365.0,
                    price=float(row["futures_price"]),
                    multiplier=float(row["multiplier"]),
                    expiry_date=pd.Timestamp(row["expiry_date"]).to_pydatetime(),
                )
                for _, row in chain.iterrows()
            ]
            if not quotes:
                raise ValidationError(
                    f"dividend_source='futures_curve' found no contract with at "
                    f"least {min_days} days to expiry on {date.date()}"
                )
            extrapolation = getattr(
                self.engine_config, "futures_curve_extrapolation", "flat_q"
            )
            if extrapolation in SUPPORTED_EXTRAPOLATIONS:
                # Every risk scenario for the day is a transformation of this
                # context, so the sampler and the pricer share one builder.
                context = CarryCurveContext(
                    quotes=tuple(quotes),
                    spot=float(market["spot"]),
                    rate_curve=rate_curve,
                    extrapolation=extrapolation,
                    underlying=self.underlying or "index",
                    valuation_date=date,
                )
                self.last_carry_context = context
                return context.dividend()
            artifact = (
                self._surface_artifact(date, "surface_forward_carry")
                if extrapolation == "surface_forward_carry"
                else None
            )
            return term_dividend_yield(
                quotes,
                spot=float(market["spot"]),
                rate_curve=rate_curve,
                extrapolation=extrapolation,
                underlying=self.underlying or "index",
                artifact=artifact,
            )
        if source == "surface_forwards":
            artifact = self._surface_artifact(date, "surface_forwards")
            return artifact.term_structure_dividend_yield(rate=market["rate"])
        raise ValidationError(f"Unknown dividend_source: {source!r}")

    def _surface_artifact(self, date: pd.Timestamp, needed_by: str):
        """The day's admitted IV-surface artifact, recording its provenance."""
        history = getattr(self.market_data, "surface_history", None)
        if history is None:
            raise ValidationError(
                f"{needed_by!r} requires market_data.surface_history"
            )
        artifact = history.surface_for(date)
        self.last_surface_provenance = {
            "surface_date": artifact.trade_date.isoformat(),
            "surface_sha": artifact.sha256,
            "surface_extrapolation": artifact.extrapolation_policy.get(
                "beyond_last_listed_expiry"
            ),
            "surface_max_listed_T": artifact.max_listed_T,
        }
        return artifact

    def recorded_pricing_q(
        self,
        env: PricingEnvironment,
        date: pd.Timestamp,
        market: dict[str, float],
        implied_q: float,
    ) -> float:
        """Scalar ``pricing_q`` for the state row.

        Legacy channels record the flat yield the pricer received.  A term
        source has no single number, so the row carries the zero yield to the
        product's remaining maturity — the carry that actually discounts the
        terminal leg — sampled off the very object the engines price with.

        The engine calls this once per day on its first replay, so in a
        multi-product book the column follows the first product's maturity
        (each product's PRICING samples its own).  The remaining maturity is
        read through the tracker's time-decayed product copy: one small
        deepcopy per day, negligible next to a pricing call.
        """
        if not self.uses_term_dividend_source():
            return self.pricing_dividend_yield(implied_q)
        if self._trading_clock() is not None:
            # D5: the column keeps its calendar meaning -- the calendar-quoted curve (the env wrapper's inner) read
            # at the CALENDAR tenor to the product's maturity, the same number the calendar-day replay records
            remaining = self._remaining_maturity_years(date, market, on_calendar=True)
            div = env.div_yield
            quoted = div.inner if is_clock_wrapped(div) else div
            return float(quoted.get_yield(remaining))
        remaining = self._remaining_maturity_years(date, market)
        return float(env.div_yield.get_yield(remaining))

    def _vol_source(self) -> str:
        return getattr(self.engine_config, "vol_source", "scalar")

    def _surface_vol_mode(self) -> str:
        return getattr(self.engine_config, "surface_vol_mode", "flat_atm_remaining")

    def _vol_and_dividend(self, date: pd.Timestamp, market: dict[str, float], pricing_q: float):
        """
        Build the day's vol surface and dividend curve.

        Scalar mode (the default) is byte-for-byte the historical behavior:
        ``FlatVolSurface(market["volatility"])`` plus a flat signed implied-q.

        Surface mode reprices against the admitted IV-surface artifact for
        ``date`` (carry-forward over excluded dates per the manifest gap
        policy) and requires ``market_data.surface_history``.  The vol object
        depends on ``surface_vol_mode``:

        - ``flat_atm_remaining``: the artifact's ATM term structure sampled
          at the product's remaining maturity (valuation date -> final
          maturity), wrapped in a ``FlatVolSurface``; refreshed daily.
        - ``term_structure``: the artifact's ATM pillar term structure.
        - ``full_grid``: the artifact's full strike x maturity smile grid.

        Curves: the rate curve stays on the existing flat rate channel; the
        dividend curve is derived from the artifact's parity forward pillars,
        ``q(T_i) = rate - ln(F_i / s0) / T_i`` (the flat rate is kept fixed
        while carry comes from the option-implied forwards), as a
        ``TermStructureDividendYield`` — internally consistent with the
        smile.  ``fixed_dividend_yield`` is a scalar-mode knob and does not
        override this term structure.  Extrapolation beyond the artifact's
        ``max_listed_T`` relies on each class's native edge behavior: the vol
        surfaces clamp flat (matching the artifact's ``flat_total_variance``
        policy) and the dividend leg clamps to the endpoint yield (flat-q).
        """
        vol_source = self._vol_source()
        if vol_source == "scalar":
            self.last_surface_provenance = None
            return FlatVolSurface(volatility=market["volatility"]), self.flat_dividend(
                pricing_q
            )
        if vol_source != "surface":
            raise ValidationError(f"Unknown vol_source: {vol_source!r}")

        history = getattr(self.market_data, "surface_history", None)
        if history is None:
            raise ValidationError(
                "vol_source='surface' requires market_data.surface_history; "
                "attach a VolSurfaceHistory or use vol_source='scalar'"
            )
        artifact = history.surface_for(date)
        mode = self._surface_vol_mode()
        if mode == "term_structure":
            vol_surface = artifact.term_structure_vol_surface()
        elif mode == "full_grid":
            vol_surface = artifact.grid_vol_surface()
        elif mode == "flat_atm_remaining":
            atm_term_structure = artifact.term_structure_vol_surface()
            # D6: the ATM lookup is on the artifact's own (calendar) axis; on the business-day clock that tenor is
            # a separate calendar computation (the flat calendar vol is then wrapped onto the env clock)
            remaining = self._remaining_maturity_years(
                date, market, on_calendar=self._trading_clock() is not None
            )
            vol_surface = FlatVolSurface(
                volatility=float(atm_term_structure.get_vol(0.0, remaining, 0.0))
            )
        else:
            raise ValidationError(f"Unknown surface_vol_mode: {mode!r}")
        div_yield = artifact.term_structure_dividend_yield(rate=market["rate"])
        self.last_surface_provenance = {
            "surface_date": artifact.trade_date.isoformat(),
            "surface_sha": artifact.sha256,
            "surface_extrapolation": artifact.extrapolation_policy.get(
                "beyond_last_listed_expiry"
            ),
            "surface_max_listed_T": artifact.max_listed_T,
        }
        return vol_surface, div_yield

    def _remaining_maturity_years(
        self, date: pd.Timestamp, market: dict[str, float], *, on_calendar: bool = False
    ) -> float:
        """
        Product remaining maturity (valuation date -> final maturity) in years.

        Uses the lifecycle tracker's time-decayed pricing product so the ATM
        sample point matches the maturity the pricing engines actually use
        for this date.  A throwaway probe env carries the valuation date and
        day-count convention; its vol/dividend contents are irrelevant to the
        maturity calculation.

        The probe is on the day env's clock (``build_env``'s clock keywords). ``on_calendar`` measures on the
        calendar-day clock instead: the ACT/365 tenor a calendar quote (an IV-surface ATM pillar, a calendar-quoted
        dividend curve) is read at on the business-day clock (run 7 D5 / D6).
        """
        clock = self._env_clock(on_calendar=on_calendar)
        probe_env = PricingEnvironment(
            spot_quote=SpotQuote(spot=market["spot"], asset_name=self.underlying),
            vol_surface=FlatVolSurface(volatility=market["volatility"]),
            rate_curve=FlatRateCurve(rate=market["rate"]),
            div_yield=self.flat_dividend(0.0, on_calendar=on_calendar),
            basis_yield=self.flat_basis(0.0, on_calendar=on_calendar),
            valuation_date=date.to_pydatetime(),
            day_count_convention=clock.day_count_convention,
            bus_days_in_year=clock.bus_days_in_year,
            calendar=clock.calendar,
            holiday_time_adjustment=clock.holiday_time_adjustment,
            rate_curve_quoting=clock.rate_curve_quoting,
            div_yield_quoting=clock.div_yield_quoting,
            basis_yield_quoting=clock.basis_yield_quoting,
            trading_clock_horizon=clock.trading_clock_horizon,
        )
        remaining_of = getattr(self._tracker, "remaining_maturity", None)
        if remaining_of is not None:
            # the DCN tracker answers without a pricing copy: this probe runs before the day's observation, when a
            # skipped schedule date is still undecided (A7'); measured on the probe environment's clock
            remaining = float(remaining_of(date, probe_env))
            return 1e-8 if is_zero(remaining) or remaining < 0.0 else remaining
        product = self._tracker.product_for_pricing(date, probe_env)
        try:
            remaining = float(product.get_maturity(probe_env))
        except ValidationError:
            # Valuation date on/after a date-based final maturity: sample the
            # shortest pillar (the product settles today; its price is unused).
            return 1e-8
        if is_zero(remaining) or remaining < 0.0:
            # ON a date-based final maturity get_maturity is 0.0 (2026-10-09; it
            # used to raise), and a numeric run-fixings product is no longer
            # floored at 1e-8 (0.0 on its maturity day, negative past it): the
            # same pillar, so the readout is unchanged.
            return 1e-8
        return remaining

    def pricing_dividend_yield(self, implied_q: float) -> float:
        if self.fixed_dividend_yield is not None:
            return float(self.fixed_dividend_yield)
        return float(implied_q)

    def calculate_greeks(
        self, product: Any, env: PricingEnvironment, *, engine: BaseEngine
    ) -> dict[str, float]:
        """Greeks via the engine's own method — native where the engine
        overrides it, engine-side BumpConfig bumping otherwise.

        Fail-closed: an engine failure propagates. The pre-consolidation
        silent ``delta=0.0`` fallback is deliberately gone — a zero-delta day
        from a failed price manufactured phantom unwind trades.
        """
        return dict(engine.calculate_greeks(product, env))

    # ------------------------------------------------------------------
    # Carry risk adapter
    # ------------------------------------------------------------------

    def carry_price_callback(
        self, product: Any, env: PricingEnvironment, *, engine: BaseEngine, position: Any = None,
        quantity: float = 1.0,
    ):
        """``price_at(spot, dividend)`` over one fixed product/engine state.

        Every carry scenario prices through this one closure, so the vol
        surface, rate curve, basis yield, valuation date, product barriers and
        lifecycle snapshot are shared by object identity across the whole bump
        set.  It deliberately does not touch the engine factory, the daily
        calibration or the lifecycle tracker: a bump that re-derived any of
        those would measure the rebuild, not the risk.
        """
        # The day's contract carries every fixing the run has recorded (product_for_date); this freeze finds nothing
        # strictly past, and a fixing dated today stays live under the carry bumps (D1).
        product = freeze_missing_fixings(product, env, warn=False)

        if position is not None:
            # a trade with cash legs: its trade value per unit of quantity (spec 2026-10-23 D9), so the
            # aggregation by quantity gives the whole trade
            import copy as _copy

            trade = _copy.copy(position)
            trade.product, trade.engine = product, engine

            def price_trade(spot: float, dividend: Any) -> float:
                scenario = _env_with(env, spot=float(spot), div_yield=dividend, underlying=self.underlying)
                return float(trade.get_trade_value(scenario)) / float(quantity)

            return price_trade

        def price_at(spot: float, dividend: Any) -> float:
            return float(
                engine.price(
                    product,
                    _env_with(
                        env,
                        spot=float(spot),
                        div_yield=dividend,
                        underlying=self.underlying,
                    ),
                )
            )

        return price_at

    def measure_carry_risk(
        self,
        product: Any,
        env: PricingEnvironment,
        *,
        context,
        engine: BaseEngine,
        delta_q: float,
        points: float,
        base_price: Optional[float] = None,
        position: Any = None,
        quantity: float = 1.0,
    ):
        """This product's UNIT-position carry buckets on ``context``.

        ``context`` is passed in explicitly: a shared environment is built
        through the first replay alone, so reading ``last_carry_context`` off
        another replay would leave every later product without buckets.  The
        engine aggregates the returned unit Greeks by position quantity.
        """
        # Local import: keeps ``carry_risk`` free to grow a dependency on this
        # module's env helpers without closing a cycle.
        from .carry_risk import measure_product_carry_risk

        return measure_product_carry_risk(
            self.carry_price_callback(product, env, engine=engine, position=position, quantity=quantity),
            context,
            delta_q=delta_q,
            points=points,
            base_price=base_price,
        )

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
                surf_env = _env_with(
                    env,
                    spot=float(s),
                    div_yield=self.flat_dividend(float(q)),
                    underlying=self.underlying,
                )
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
                    logger.warning(
                        "surface node failed (spot=%s, q=%s)", s, q, exc_info=True
                    )
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
        """One daily summary row and the KO / KI event rows of ``product`` on ``date``.

        The statistics are read as ``EventStatistics`` (spec D21 mapping): ``ki_probability_to_maturity`` is
        ``ki_settlement.probability`` (matured knocked in); ``expected_discounted_maturity_cashflow`` is
        ``maturity.discounted_cash + ki_settlement.discounted_cash``; KO rows are the observation table's KO
        rows, KI rows the per-date KI table's. Every statistic is read through ``EventStatistics.require``: one
        the producer lists in ``unavailable`` is recorded as NaN (an unavailable KO or KI table records no rows
        of that kind), never as the flagged number or a stand-in. ``pv`` is the engine's price, never flagged.
        """
        produced, stats_engine = self._calculate_event_stats(product, env)
        stats = checked_event_statistics(produced)

        ko_rows = _available(stats, "observations.is_ko_obs")
        if ko_rows is None:
            ko_times = ko_probs = survival = np.array([], dtype=float)
            ed_ko_cf: Optional[np.ndarray] = None
        else:
            ko_times = stats.require("observations.times")[ko_rows]
            ko_probs = stats.require("observations.probability")[ko_rows]
            survival = stats.require("observations.survival")[ko_rows]
            cash = _available(stats, "observations.discounted_cash")
            ed_ko_cf = None if cash is None else cash[ko_rows]
        maturity_cash = _available(stats, "maturity.discounted_cash")
        ki_settlement_cash = _available(stats, "ki_settlement.discounted_cash")
        ki_to_maturity = _available(stats, "ki_settlement.probability")

        ko_unknown = ko_rows is None
        self.daily_event_sink.append(
            {
                "date": date,
                "next_ko_date": (
                    self._event_date(date, float(ko_times[0]), env) if ko_times.size else pd.NaT
                ),
                "next_ko_probability": (
                    np.nan if ko_unknown else _first_instant_probability(ko_times, ko_probs)
                    if ko_probs.size else 0.0
                ),
                "total_remaining_ko_probability": (
                    np.nan if ko_unknown else float(np.sum(ko_probs))
                ),
                "ki_probability_to_maturity": (
                    np.nan if ki_to_maturity is None else float(ki_to_maturity)
                ),
                "survival_probability": float(survival[-1]) if survival.size else np.nan,
                "expected_discounted_ko_cashflow": (
                    np.nan if ed_ko_cf is None else float(np.sum(ed_ko_cf))
                ),
                "expected_discounted_maturity_cashflow": (
                    np.nan if maturity_cash is None or ki_settlement_cash is None
                    else float(maturity_cash + ki_settlement_cash)
                ),
                "pv": float(stats.pv),
                "event_stats_engine": stats_engine,
            }
        )

        previous_survival = 1.0
        for i, probability in enumerate(ko_probs):
            conditional = probability / previous_survival if previous_survival > 0 else np.nan
            self.event_prob_sink.append(
                {
                    "date": date,
                    "event_date": self._event_date(date, float(ko_times[i]), env),
                    "event_type": "KO",
                    "event_probability": float(probability),
                    "conditional_probability": float(conditional),
                    "survival_probability": float(survival[i]),
                    "expected_discounted_cashflow": np.nan if ed_ko_cf is None else float(ed_ko_cf[i]),
                    "event_stats_engine": stats_engine,
                }
            )
            previous_survival = float(survival[i])

        ki_table = _available(stats, "ki_table")
        if ki_table is None:
            return
        for i, probability in enumerate(ki_table.probability):
            self.event_prob_sink.append(
                {
                    "date": date,
                    "event_date": self._event_date(date, float(ki_table.times[i]), env),
                    "event_type": "KI",
                    "event_probability": float(probability),
                    "conditional_probability": np.nan,
                    "survival_probability": float(ki_table.survival[i]),
                    "expected_discounted_cashflow": np.nan,
                    "event_stats_engine": stats_engine,
                }
            )

    def _calculate_event_stats(self, product: Any, env: PricingEnvironment):
        """Event stats from the primary engine, fail-closed.

        A ``None`` return is the engine layer's "unsupported" signal; when
        event probabilities were requested that is a failure, not a silent
        no-op. ``event_stats_fallback="mc"`` opts into the MC fallback, which
        is logged and recorded in the rows' ``event_stats_engine`` column.
        """
        mode = getattr(self.engine_config, "event_stats_fallback", "none")
        primary_failure: Optional[BaseException] = None
        stats = None
        try:
            stats = self.event_stats_engine.calculate_event_stats(product, env)
        except StatisticsRefused:
            # a 2D solver's refusal is a configuration decision (spec 3B decision 8): the MC fallback
            # would silently swap the model, so it never masks one
            raise
        except Exception as exc:
            if mode != "mc":
                raise
            if self._trading_clock() is not None and isinstance(exc, ValidationError):
                # run 7 D7: on the business-day clock a refusal is a clock decision (an unsupported surface, a
                # date past the horizon, ...): the MC fallback would silently swap the model, so it never masks one
                raise
            primary_failure = exc
        if stats is not None:
            return stats, "primary"
        if mode != "mc":
            raise PricingError(
                "event-stats engine "
                f"{type(self.event_stats_engine).__name__} returned no stats "
                "while event probabilities were requested; set "
                "event_stats_fallback='mc' to opt into the MC fallback"
            )
        logger.warning(
            "primary event-stats engine %s unavailable (%s); using MC fallback",
            type(self.event_stats_engine).__name__,
            primary_failure if primary_failure is not None else "returned None",
        )
        fallback = create_mc_event_stats_engine(product, self.engine_config)
        stats = fallback.calculate_event_stats(product, env)
        if stats is None:
            raise PricingError(
                "MC event-stats fallback returned no stats (fail-closed)"
            )
        return stats, "mc_fallback"

    def settle_pending_if_due(self, date: pd.Timestamp) -> float:
        """Post a pending terminal receivable once its settlement date is
        reached; returns the amount posted (0.0 otherwise). A DCN's ledger
        (A4) posts every flow whose payment date has been reached."""
        lifecycle = self.lifecycle
        if isinstance(lifecycle, DCNLifecycleState):
            return lifecycle.advance_to(pd.Timestamp(date).normalize().to_pydatetime())
        if lifecycle.settled or lifecycle.settlement_date is None:
            return 0.0
        if pd.Timestamp(date).normalize() >= pd.Timestamp(
            lifecycle.settlement_date
        ).normalize():
            return lifecycle.settle()
        return 0.0

    def pending_receivable_pv(
        self, env: PricingEnvironment, date: pd.Timestamp
    ) -> float:
        """PV of determined-but-unpaid cashflows as of the replay ``date``.

        Date-based flows use the replay day-count convention
        ``(settlement - date).days / 365`` — the exact arithmetic the book
        engine has always used, so existing runs are bit-identical. Numeric
        (time-based) lifecycles have no dates to subtract; they price the
        ledger's pending flows at the current valuation point instead.
        """
        lifecycle = self.lifecycle
        if isinstance(lifecycle, DCNLifecycleState):
            # A4: a DCN carries several dated receivables (coupons, knock-out, loss leg) in its ledger
            point = ValuationPoint(date=pd.Timestamp(date).normalize().to_pydatetime())
            return float(lifecycle.ledger.pending_pv(point, env))
        point = lifecycle.valuation_point
        if point is None or point.date is not None:
            pending = float(lifecycle.pending_settlement_cashflow)
            if pending != 0.0 and lifecycle.settlement_date is not None:
                if self._trading_clock() is not None:
                    # D5: the settlement time from the env's own resolver (business-day clock)
                    from quantark.priceenv.date_axis import env_time

                    settle = pd.Timestamp(lifecycle.settlement_date).normalize().to_pydatetime()
                    tau = max(env_time(env, settle), 0.0)
                    return pending * float(env.get_discount_factor(tau))
                tau_settle = max(
                    (
                        pd.Timestamp(lifecycle.settlement_date).normalize()
                        - pd.Timestamp(date).normalize()
                    ).days
                    / 365.0,
                    0.0,
                )
                return pending * float(env.get_discount_factor(tau_settle))
            return 0.0
        return float(lifecycle.ledger.pending_pv(point, env))

    def paid_cash(self) -> float:
        """Cumulative contractual cash paid through the current replay point."""
        point = self.lifecycle.valuation_point
        if point is None:
            return 0.0
        return float(self.lifecycle.ledger.paid_total(point))

    @property
    def has_run_fixings(self) -> bool:
        """True when the replayed contract takes part in run fixings (an Asian or a range accrual with records):
        on each date the engine steps its fixings first and settles a maturity on that run-fixed copy (spec
        2026-10-13 I7 / A3)."""
        return participates(self.product)

    def settle_maturity_if_due(
        self,
        date: pd.Timestamp,
        product: Any,
        env: PricingEnvironment,
        spot: float,
        *,
        settled_product: Any = None,
        previous_spot: Optional[float] = None,
    ) -> None:
        """The tracker's ``settle_maturity_if_due``; a run-fixings participant passes ``settled_product``, its
        run-fixed copy for ``date`` (``product_for_date``), which its maturity settles from, and ``previous_spot``,
        the spot of the run instant before ``date`` (``settle_run_fixings_maturity``)."""
        if settled_product is None:
            event = self._tracker.settle_maturity_if_due(date, product, env, spot)
        else:
            event = self._tracker.settle_maturity_if_due(
                date, product, env, spot, settled_product=settled_product, previous_spot=previous_spot
            )
        if event is not None:
            if self.record_events:
                self.events_today.append(event)
            self.actions_sink.append(self._event_to_action_row(event))

    def settle_run_fixings_maturity(
        self, date: pd.Timestamp, product: Any, env: PricingEnvironment, spot: float
    ) -> Any:
        """A run-fixings participant's day (spec 2026-10-13 I7 / A3): record ``date``'s fixings first
        (``product_for_date``), then settle a maturity due on that run-fixed copy - on the maturity date at
        ``spot``; on a later date (the run's clock skipped the maturity date) at the spot of the run instant before
        ``date`` (W6 gap rule). Returns the run-fixed copy, the day's pricing copy."""
        day = pd.Timestamp(date).normalize()
        run_fixed = self.product_for_date(date, env)
        previous = self._previous if self._last is not None and self._last[0] == day else None
        self.settle_maturity_if_due(
            date, product, env, spot,
            settled_product=run_fixed,
            previous_spot=None if previous is None else previous[1],
        )
        return run_fixed

    def apply_lifecycle_events(
        self, date: pd.Timestamp, product: Any, env: PricingEnvironment, spot: float
    ) -> None:
        # events_today is reset once per engine date by its owner (the explain
        # recorder's begin_day), never here: observation and settlement both
        # append to the same day's buffer in whichever order the engine runs them.
        for event in self._tracker.observe(date, product, env, spot):
            if self.record_events:
                self.events_today.append(event)
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
        # Cashflow identity: determination vs payment, typed (dates stay None
        # for numeric products — never fabricated). Emitted for every action
        # row so the frame's column order is deterministic.
        cashflow = getattr(event, "realized_cashflow", None)
        cashflow_id = cashflow.cashflow_id if cashflow is not None else None
        if cashflow_id is not None and self.position_id is not None:
            cashflow_id = f"{self.position_id}:{cashflow_id}"
        row.update(
            {
                "position_id": self.position_id,
                "cashflow_id": cashflow_id,
                "determination_date": (
                    cashflow.determination_date if cashflow is not None else None
                ),
                "determination_time": (
                    cashflow.determination_time if cashflow is not None else None
                ),
                "payment_date": (
                    cashflow.payment_date if cashflow is not None else None
                ),
                "payment_time": (
                    cashflow.payment_time if cashflow is not None else None
                ),
            }
        )
        row.update(event.metadata)
        return row

    def _next_available_market_date(self, date: pd.Timestamp) -> pd.Timestamp:
        dates = self.market_data.dates
        eligible = dates[dates >= pd.Timestamp(date).normalize()]
        if len(eligible) == 0:
            return pd.Timestamp(date).normalize()
        return pd.Timestamp(eligible[0]).normalize()

    def _event_date(self, date: pd.Timestamp, time_years: float, env: PricingEnvironment) -> pd.Timestamp:
        """The date of an engine event time: ``_date_from_time`` (ACT/365) on the calendar-day replay; on the
        business-day clock the business date whose env time is that instant (D5)."""
        if self._trading_clock() is None:
            return self._date_from_time(date, time_years)
        return self._business_date_from_time(env, time_years)

    @staticmethod
    def _business_date_from_time(env: PricingEnvironment, time_years: float) -> pd.Timestamp:
        """The business date d of ``env.calendar`` with ``env_time(env, d)`` the same instant as ``time_years``: the
        n-th business day after the (rolled) valuation date, n = the integer business-day count the engines' axis
        is built from. A time that matches no business date is refused."""
        from quantark.priceenv.date_axis import env_time

        cal = env.calendar
        t = float(time_years)
        n = int(round(t * env.bus_days_in_year))
        v = env.valuation_date
        anchor = datetime(v.year, v.month, v.day)
        if not cal.is_business_day(anchor):
            anchor = cal.adjust_date(anchor, env.holiday_time_adjustment)
        day = cal.add_business_days(anchor, n) if n > 0 else anchor
        if n < 0 or not same_instant(env_time(env, day), t):
            raise ValidationError(
                f"event time {t!r} on the business-day clock (D={env.bus_days_in_year}) is no business date of "
                f"{getattr(cal, 'name', cal)!r} after {anchor.date()}"
            )
        return pd.Timestamp(day)

    @staticmethod
    def _date_from_time(date: pd.Timestamp, time_years: float) -> pd.Timestamp:
        return pd.Timestamp(date).normalize() + timedelta(
            days=int(round(float(time_years) * 365))
        )
