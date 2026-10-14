"""
Lifecycle tracking for the vanilla barrier product family.

Covers single/double barrier options, single/double sharkfins, and one-touch /
double one-touch products. CONTINUOUS monitoring is daily-close: each
``observe`` call checks the day's closing spot. Products whose
``observation_type`` is EXPIRY are only settled at expiry, never intra-path.

DISCRETE monitoring is decided only on the schedule's observation dates (A15,
user decision 2026-10-14: "knock out should only happen at the observation
date, otherwise price it"): each record once, on its date at that day's close,
against the record's own level and paying the record's payoff (W18c). A record
date the run never visits is decided at the spot of the run instant before it,
flagged ``gap=True`` (A7'(c), the signed W6 rule, as ``lifecycle/dcn.py``); a
record before the run is history (signed rule 3: no barrier state is assumed,
not hit), "before the run" being before the FIRST RUN INSTANT (ruling R31) --
the first ``observe``, which for a restored checkpoint (the intraday replay) is
the first instant after it; a first run instant after the expiry retires the
contract as matured before the run (W1: the warning, no cash, valued at zero); a numeric record falls on
``start_date + round(t * 365)`` days, the run clock of every tracker. At expiry
only a record on the expiry date hits.

Knock-in semantics: a knocked-in single/double barrier option *is* a European
option, so after KI ``product_for_pricing`` returns the exact European
equivalent (and ``engine_for_pricing`` returns a ``BlackScholesEngine``).

Terminal events register the undiscounted contractual amount; the lifecycle
ledger handles discounting from determination to payment.
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Dict, List, Optional

import pandas as pd

from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.engine.settlement_support import (
    resolve_terminal_timing,
)
from quantark.asset.equity.product.option import (
    BarrierOption,
    EuropeanVanillaOption,
)
from quantark.asset.equity.product.option.double_barrier_option import (
    DoubleBarrierOption,
)
from quantark.asset.equity.product.option.double_one_touch_option import (
    DoubleOneTouchOption,
)
from quantark.asset.equity.product.option.double_sharkfin_option import (
    DoubleSharkfinOption,
)
from quantark.asset.equity.product.option.one_touch_option import OneTouchOption
from quantark.asset.equity.product.option.single_sharkfin_option import (
    SingleSharkfinOption,
)
from quantark.priceenv import PricingEnvironment
from quantark.asset.equity.settlement import (
    CashflowKind,
    SettlementLagUnit,
    SettlementRequest,
    SettlementResolver,
)
from quantark.util.enum import ObservationType
from quantark.util.numerical import is_zero

from .events import LifecycleEvent, LifecycleEventType
from .cashflows import RealizedCashflow, ValuationPoint
from .state import BarrierLifecycleState

TRACKED_BARRIER_PRODUCTS = (
    BarrierOption,
    DoubleBarrierOption,
    SingleSharkfinOption,
    DoubleSharkfinOption,
    OneTouchOption,
    DoubleOneTouchOption,
)


class BarrierLifecycleTracker:
    """Tracks realized barrier events for one vanilla barrier-family position.

    Supports ``BarrierOption``, ``DoubleBarrierOption``, ``SingleSharkfinOption``,
    ``DoubleSharkfinOption``, ``OneTouchOption``, and ``DoubleOneTouchOption``.

    Call ``observe(date, env, spot)`` once per run day with the closing spot.
    CONTINUOUS monitoring checks every close; DISCRETE monitoring decides only
    the schedule's records (module docstring, A15); products with
    ``observation_type == ObservationType.EXPIRY`` are never checked
    intra-path; they are settled only when expiry is detected.

    After a knock-in event on a ``BarrierOption`` or ``DoubleBarrierOption``,
    ``product_for_pricing`` returns the exact European vanilla equivalent (with
    decayed maturity) and ``engine_for_pricing`` returns a ``BlackScholesEngine``.
    """

    def __init__(self, *, product, quantity: float, start_date):
        """
        Initialise the tracker.

        Args:
            product: One of the ``TRACKED_BARRIER_PRODUCTS``.
            quantity: Number of contracts (may be fractional).
            start_date: Trade inception date (datetime or pandas Timestamp).
        """
        self.product = product
        self.quantity = float(quantity)
        self.start_date = pd.Timestamp(start_date).normalize()
        self.state = BarrierLifecycleState()
        self._bs_engine: Optional[BlackScholesEngine] = None
        self._observation_env: Optional[PricingEnvironment] = None
        self._last: Optional[tuple] = None      # (day, spot) of the last observed run instant (DISCRETE gap rule)

    # ------------------------------------------------------------------
    # Time helpers
    # ------------------------------------------------------------------

    def _elapsed_years(self, date: pd.Timestamp) -> float:
        """Return calendar years elapsed since start_date (floor at 0)."""
        date = pd.Timestamp(date).normalize()
        days = (date - self.start_date).days
        return max(0.0, days / 365.0)

    def _remaining_maturity(self, date: pd.Timestamp) -> float:
        """Return remaining maturity in years from *date*.

        When the product carries an ``exercise_date`` attribute (a datetime),
        the remaining life is computed as the calendar-day difference between
        that date and *date* divided by 365.  The float ``maturity`` field is
        not consulted in that case.
        """
        exercise_date = getattr(self.product, "exercise_date", None)
        if exercise_date is not None:
            date_ts = pd.Timestamp(date).normalize()
            exp_ts = pd.Timestamp(exercise_date).normalize()
            return (exp_ts - date_ts).days / 365.0

        mat = getattr(self.product, "maturity", None)
        if mat is None:
            return 0.0
        return float(mat) - self._elapsed_years(date)

    def _expiry_due(self, date: pd.Timestamp) -> bool:
        """Return True if *date* is on or past the product's expiry.

        When ``exercise_date`` is set on the product, expiry is triggered on
        the calendar day that equals or exceeds that date (day-level
        comparison).  The float-maturity arithmetic path is not consulted.
        """
        exercise_date = getattr(self.product, "exercise_date", None)
        if exercise_date is not None:
            date_ts = pd.Timestamp(date).normalize()
            exp_ts = pd.Timestamp(exercise_date).normalize()
            return date_ts >= exp_ts

        remaining = self._remaining_maturity(date)
        return remaining < 0.0 or is_zero(remaining)

    def _monitored_intra_path(self) -> bool:
        """Return True when the product monitors intra-path (not EXPIRY-only)."""
        obs = getattr(self.product, "observation_type", None)
        if obs is None:
            return True
        return getattr(obs, "name", None) != "EXPIRY"

    # ------------------------------------------------------------------
    # Public pricing-product accessors
    # ------------------------------------------------------------------

    def product_for_pricing(self, date, env: PricingEnvironment):
        """Return the product to pass to a pricing engine on *date*.

        If the position has knocked in (``BarrierOption`` or
        ``DoubleBarrierOption``), returns the exact European vanilla equivalent
        with decayed maturity. Otherwise, returns a deep-copy of the original
        product with decayed maturity.

        Args:
            date: Current observation date.
            env: Pricing environment (unused here, kept for interface symmetry).

        Returns:
            A (possibly mutated) copy of the product, or the European replacement.
        """
        if self.state.knocked_in:
            repl = self._replacement_product(date)
            if repl is not None:
                return repl

        # Generic copy with decayed maturity
        copy = deepcopy(self.product)
        if (
            getattr(copy, "exercise_date", None) is None
            and getattr(copy, "maturity", None) is not None
        ):
            # 1e-8 floor matches the maturity floor used by AutocallableLifecycleTracker
            remaining = max(1e-8, self._remaining_maturity(date))
            copy.maturity = remaining
        return copy

    def engine_for_pricing(self) -> Optional[BlackScholesEngine]:
        """Return a ``BlackScholesEngine`` when knocked in, else ``None``.

        After knock-in the position reprices as a plain European option, so the
        Black-Scholes analytical engine is the appropriate pricer.

        Returns:
            A cached ``BlackScholesEngine`` instance when ``state.knocked_in``
            is True, otherwise ``None``.
        """
        if not self.state.knocked_in:
            return None
        if self._bs_engine is None:
            self._bs_engine = BlackScholesEngine()
        return self._bs_engine

    def _replacement_product(self, date) -> Optional[EuropeanVanillaOption]:
        """Build the European vanilla replacement for a knocked-in barrier option.

        The replacement has the same strike, option type, and contract
        multiplier as the original. For ``BarrierOption`` the contract
        multiplier is used as-is (``get_payoff`` does not apply
        ``participation_rate``). For ``DoubleBarrierOption`` likewise.

        Args:
            date: Current observation date (used to compute decayed maturity).

        Returns:
            ``EuropeanVanillaOption`` replacement, or ``None`` if the product
            is not a knock-in barrier type that supports substitution.
        """
        # 1e-8 floor matches the maturity floor used by AutocallableLifecycleTracker
        remaining = max(1e-8, self._remaining_maturity(date))

        if isinstance(self.product, BarrierOption):
            return EuropeanVanillaOption(
                strike=float(self.product.strike),
                option_type=self.product.option_type,
                maturity=remaining,
                contract_multiplier=float(self.product.contract_multiplier),
            )

        if isinstance(self.product, DoubleBarrierOption):
            return EuropeanVanillaOption(
                strike=float(self.product.strike),
                option_type=self.product.option_type,
                maturity=remaining,
                contract_multiplier=float(self.product.contract_multiplier),
            )

        return None

    # ------------------------------------------------------------------
    # Main observe entry point
    # ------------------------------------------------------------------

    def observe(
        self,
        date,
        env: PricingEnvironment,
        spot: float,
    ) -> List[LifecycleEvent]:
        """Check the product lifecycle against today's closing spot.

        Returns a list of ``LifecycleEvent`` objects generated by this
        observation (empty when nothing happened). The position is terminal
        after any ``KNOCK_OUT`` or ``EXPIRY`` event.

        Args:
            date: Observation date (datetime or pandas Timestamp).
            env: Pricing environment (provides discount rate via ``get_rate``).
            spot: Closing underlying price for this observation.

        Returns:
            List of ``LifecycleEvent`` objects (may be empty).
        """
        date = pd.Timestamp(date).normalize()
        self._observation_env = env
        self.state.valuation_point = self._valuation_point(date)
        if not self.state.alive:
            return []

        if self._last is None and self._expired_before(date):
            # R31 applied to matured-before-run: the first run instant is after the expiry, so every date is history
            # and the contract matured before the run (W1: no cash booked, valued at zero)
            self._retire_matured(date)
            self._last = (date, float(spot))
            return []
        if self._discretely_monitored():
            events = self._observe_records(date, spot)
            self._last = (date, float(spot))
            if not self.state.alive:
                return events
            # nothing is decided between records: at expiry only a record on that date could hit (decided above)
            return events + self._observe_family(date, env, spot, hit=False)
        return self._observe_family(date, env, spot)

    def _expired_before(self, date: pd.Timestamp) -> bool:
        """The expiry is strictly before ``date``."""
        remaining = self._remaining_maturity(date)
        return remaining < 0.0 and not is_zero(remaining)

    def _retire_matured(self, date: pd.Timestamp) -> None:
        """Matured before the run (W1 rule, ruling R31): warn, mark expired and dead, book nothing."""
        from .autocallable import MaturedBeforeRunWarning

        warnings.warn(
            MaturedBeforeRunWarning(
                f"{type(self.product).__name__} expired before the first run instant {date.date()}: it is not settled "
                "in the run, books no cash and is valued at zero"),
            stacklevel=3,
        )
        self.state.expired = True
        self.state.alive = False

    def _observe_family(self, date, env, spot, hit: Optional[bool] = None) -> List[LifecycleEvent]:
        """The family's daily-close observation; ``hit`` overrides the spot's barrier test (False: DISCRETE between
        records, which only settles expiry)."""
        if isinstance(self.product, (BarrierOption,)):
            return self._observe_single_barrier(date, env, spot, hit)
        elif isinstance(self.product, DoubleBarrierOption):
            return self._observe_double_barrier(date, env, spot, hit)
        elif isinstance(self.product, (SingleSharkfinOption, DoubleSharkfinOption)):
            return self._observe_sharkfin(date, env, spot, hit)
        elif isinstance(self.product, (OneTouchOption, DoubleOneTouchOption)):
            return self._observe_one_touch(date, env, spot, hit)

        return []

    # ------------------------------------------------------------------
    # DISCRETE monitoring: the schedule's records (A15)
    # ------------------------------------------------------------------

    def _discretely_monitored(self) -> bool:
        product = self.product
        return (getattr(product, "observation_type", None) == ObservationType.DISCRETE
                and getattr(product, "observation_schedule", None) is not None)

    def _record_days(self) -> List[tuple]:
        """(day, record) of every schedule record in date order: a dated record on its date, a numeric one on
        ``start_date + round(t * 365)`` days (the run clock every tracker uses)."""
        out = []
        for position, rec in enumerate(self.product.observation_schedule.records):
            if rec.observation_date is not None:
                day = pd.Timestamp(rec.observation_date).normalize()
            else:
                day = self.start_date + pd.Timedelta(days=int(round(float(rec.observation_time) * 365.0)))
            out.append((day, position, rec))
        return [(day, rec) for day, _, rec in sorted(out, key=lambda item: (item[0], item[1]))]

    def _observe_records(self, date: pd.Timestamp, spot: float) -> List[LifecycleEvent]:
        """Decide every record dated after the last run instant and on or before ``date``: on ``date`` at its close,
        a record the run skipped at the previous instant's spot (``gap=True``)."""
        lower = self._last[0] if self._last is not None else None
        events: List[LifecycleEvent] = []
        gap_decided = False
        for day, rec in self._record_days():
            if day > date or (lower is not None and day <= lower):
                continue
            if day < self.start_date:
                continue                                # before the run (signed rule 3): history, no state assumed
            if day < date:
                if self._last is None:
                    # before the first run instant (ruling R31; a restored checkpoint's covered records too): history,
                    # no barrier state assumed, not hit (signed rule 3)
                    continue
                observed, gap = self._last[1], True
            else:
                observed, gap = float(spot), False
            event = self._decide_record(day, rec, observed, gap=gap, decided_on=date)
            if event is not None:
                events.append(event)
                gap_decided = gap_decided or gap
            if not self.state.alive:
                break
        if gap_decided:
            self.state.valuation_point = self._valuation_point(date)
        return events

    def _decide_record(self, day, rec, spot: float, *, gap: bool, decided_on) -> Optional[LifecycleEvent]:
        """One record at ``spot``: its own level(s) and payoff (W18c)."""
        product = self.product
        metadata = {"monitoring": "discrete", "gap": gap, "decided_on": pd.Timestamp(decided_on),
                    "observation_date": pd.Timestamp(day)}
        if isinstance(product, (DoubleBarrierOption, DoubleSharkfinOption, DoubleOneTouchOption)):
            upper = float(rec.upper_barrier if rec.upper_barrier is not None else product.upper_barrier)
            lower = float(rec.lower_barrier if rec.lower_barrier is not None else product.lower_barrier)
            if not (spot >= upper or spot <= lower):
                return None
            level = upper if spot >= upper else lower
        else:
            level = float(rec.barrier if rec.barrier is not None else product.barrier)
            is_up = product.is_call() if isinstance(product, SingleSharkfinOption) else product.is_up_barrier
            if not (spot >= level if is_up else spot <= level):
                return None
        if isinstance(product, (BarrierOption, DoubleBarrierOption)) and product.is_knock_in:
            if self.state.knocked_in:
                return None
            return self._knock_in_event(day, spot, barrier=level, metadata=metadata)
        return self._terminal_event(LifecycleEventType.KNOCK_OUT, day, spot, barrier=level,
                                    payoff=self._record_hit_payoff(rec), metadata=metadata)

    def _record_hit_payoff(self, rec) -> float:
        """The per-contract cash a hit of ``rec`` fixes: its payoff (unset: the product's knock-out cash), with the
        contract multiplier where the product's own hit cash carries it."""
        product = self.product
        if isinstance(product, (OneTouchOption, DoubleOneTouchOption)):
            pays_on_touch = product.is_one_touch if isinstance(product, OneTouchOption) else product.is_double_one_touch
            if not pays_on_touch:
                return 0.0                                  # a touched no-touch pays nothing
            # a one-touch pays the touching record's payoff (unset: its rebate), as ``get_payoff(touched=True)``
            # scales it: no contract multiplier
            return float(rec.payoff) if rec.payoff is not None else float(product.rebate)
        if isinstance(product, (SingleSharkfinOption, DoubleSharkfinOption)):
            cash = rec.payoff if rec.payoff is not None else product.knock_out_rebate
        else:
            cash = rec.payoff if rec.payoff is not None else product.rebate
        return float(cash) * product.contract_multiplier

    # ------------------------------------------------------------------
    # Per-product-family observation helpers
    # ------------------------------------------------------------------

    def _observe_single_barrier(
        self, date: pd.Timestamp, env: PricingEnvironment, spot: float, hit: Optional[bool] = None
    ) -> List[LifecycleEvent]:
        """Lifecycle observation for ``BarrierOption``."""
        product: BarrierOption = self.product
        barrier_hit = product.is_barrier_hit(spot) if hit is None else hit

        if product.is_knock_out:
            # Intra-path KO check
            if barrier_hit and self._monitored_intra_path():
                payoff = product.rebate * product.contract_multiplier
                return [
                    self._terminal_event(
                        LifecycleEventType.KNOCK_OUT,
                        date,
                        spot,
                        barrier=product.barrier,
                        payoff=payoff,
                    )
                ]

            # Expiry check
            if self._expiry_due(date):
                if barrier_hit:
                    # Spot hit at expiry-only monitoring → KO at expiry
                    payoff = product.rebate * product.contract_multiplier
                else:
                    # No barrier hit; pay vanilla payoff
                    payoff = product.get_payoff(spot)
                return [
                    self._terminal_event(
                        LifecycleEventType.EXPIRY,
                        date,
                        spot,
                        barrier=product.barrier,
                        payoff=payoff,
                        expired=True,
                    )
                ]

        else:  # is_knock_in
            if barrier_hit and not self.state.knocked_in and self._monitored_intra_path():
                return [self._knock_in_event(date, spot, barrier=product.barrier)]

            if self._expiry_due(date):
                if self.state.knocked_in:
                    # Already knocked in: settle as European
                    repl = self._replacement_product(date)
                    if repl is not None:
                        payoff = repl.get_payoff(spot)
                    else:
                        payoff = product.get_payoff(spot)
                elif barrier_hit:
                    # EXPIRY-only monitoring: barrier hit at expiry → European payoff
                    payoff = product.get_payoff(spot)
                else:
                    # Never knocked in → rebate
                    payoff = product.rebate * product.contract_multiplier
                return [
                    self._terminal_event(
                        LifecycleEventType.EXPIRY,
                        date,
                        spot,
                        barrier=product.barrier,
                        payoff=payoff,
                        expired=True,
                    )
                ]

        return []

    def _observe_double_barrier(
        self, date: pd.Timestamp, env: PricingEnvironment, spot: float, hit: Optional[bool] = None
    ) -> List[LifecycleEvent]:
        """Lifecycle observation for ``DoubleBarrierOption``."""
        product: DoubleBarrierOption = self.product
        barrier_hit = product.is_barrier_hit(spot) if hit is None else hit

        if product.is_knock_out:
            if barrier_hit and self._monitored_intra_path():
                # DoubleBarrierOption has no pay_at_hit; its fixed rebate
                # remains pending until the terminal payment date.
                payoff = product.rebate * product.contract_multiplier
                barrier = product.upper_barrier if spot >= product.upper_barrier else product.lower_barrier
                return [
                    self._terminal_event(
                        LifecycleEventType.KNOCK_OUT,
                        date,
                        spot,
                        barrier=barrier,
                        payoff=payoff,
                    )
                ]

            if self._expiry_due(date):
                if barrier_hit:
                    payoff = product.rebate * product.contract_multiplier
                else:
                    payoff = product.get_payoff(spot)
                barrier = None
                if barrier_hit:
                    barrier = product.upper_barrier if spot >= product.upper_barrier else product.lower_barrier
                return [
                    self._terminal_event(
                        LifecycleEventType.EXPIRY,
                        date,
                        spot,
                        barrier=barrier,
                        payoff=payoff,
                        expired=True,
                    )
                ]

        else:  # is_knock_in
            if barrier_hit and not self.state.knocked_in and self._monitored_intra_path():
                hit_barrier = product.upper_barrier if spot >= product.upper_barrier else product.lower_barrier
                return [self._knock_in_event(date, spot, barrier=hit_barrier)]

            if self._expiry_due(date):
                if self.state.knocked_in:
                    repl = self._replacement_product(date)
                    if repl is not None:
                        payoff = repl.get_payoff(spot)
                    else:
                        payoff = product.get_payoff(spot)
                elif barrier_hit:
                    payoff = product.get_payoff(spot)
                else:
                    payoff = product.rebate * product.contract_multiplier
                return [
                    self._terminal_event(
                        LifecycleEventType.EXPIRY,
                        date,
                        spot,
                        barrier=None,
                        payoff=payoff,
                        expired=True,
                    )
                ]

        return []

    def _observe_sharkfin(
        self, date: pd.Timestamp, env: PricingEnvironment, spot: float, hit: Optional[bool] = None
    ) -> List[LifecycleEvent]:
        """Lifecycle observation for ``SingleSharkfinOption`` and ``DoubleSharkfinOption``.

        Sharkfins are always knock-out structures. Barrier hit fixes the
        barrier payoff; the ledger retains it until hit or terminal payment.
        Expiry without a hit fixes the no-hit payoff.
        """
        product = self.product
        barrier_hit = product.is_barrier_hit(spot) if hit is None else hit

        if barrier_hit and self._monitored_intra_path():
            raw_payoff = product.get_barrier_payoff()
            payoff = raw_payoff
            if isinstance(product, SingleSharkfinOption):
                barrier = product.barrier
            else:
                barrier = product.upper_barrier if spot >= product.upper_barrier else product.lower_barrier
            return [
                self._terminal_event(
                    LifecycleEventType.KNOCK_OUT,
                    date,
                    spot,
                    barrier=barrier,
                    payoff=payoff,
                )
            ]

        if self._expiry_due(date):
            payoff = product.get_payoff(spot, barrier_hit=False)
            barrier = None
            return [
                self._terminal_event(
                    LifecycleEventType.EXPIRY,
                    date,
                    spot,
                    barrier=barrier,
                    payoff=payoff,
                    expired=True,
                )
            ]

        return []

    def _observe_one_touch(
        self, date: pd.Timestamp, env: PricingEnvironment, spot: float, hit: Optional[bool] = None
    ) -> List[LifecycleEvent]:
        """Lifecycle observation for ``OneTouchOption`` and ``DoubleOneTouchOption``.

        One-touch: barrier hit → KNOCK_OUT (terminal). Payment is scheduled at
        hit or expiry per ``payment_at_hit``. No-touch pays at expiry.

        For double variants the same logic applies with two barriers.
        """
        product = self.product
        barrier_hit = product.is_barrier_hit(spot) if hit is None else hit

        if barrier_hit and self._monitored_intra_path():
            raw_payoff = product.get_payoff(spot, touched=True)
            payoff = raw_payoff

            if isinstance(product, OneTouchOption):
                barrier = product.barrier
            else:
                barrier = product.upper_barrier if spot >= product.upper_barrier else product.lower_barrier

            return [
                self._terminal_event(
                    LifecycleEventType.KNOCK_OUT,
                    date,
                    spot,
                    barrier=barrier,
                    payoff=payoff,
                )
            ]

        if self._expiry_due(date):
            payoff = product.get_payoff(spot, touched=barrier_hit)
            return [
                self._terminal_event(
                    LifecycleEventType.EXPIRY,
                    date,
                    spot,
                    barrier=None,
                    payoff=payoff,
                    expired=True,
                )
            ]

        return []

    # ------------------------------------------------------------------
    # State snapshot & event constructors
    # ------------------------------------------------------------------

    def _snapshot(self) -> Dict[str, bool]:
        """Return a dict snapshot of the current lifecycle state."""
        s = self.state
        return {
            "alive": s.alive,
            "knocked_in": s.knocked_in,
            "knocked_out": s.knocked_out,
            "expired": s.expired,
        }

    def _knock_in_event(
        self,
        date: pd.Timestamp,
        spot: float,
        *,
        barrier: Optional[float],
        metadata: Optional[Dict] = None,
    ) -> LifecycleEvent:
        """Record a knock-in and return the corresponding event.

        The position remains alive after knock-in; pricing switches to the
        European replacement.

        Args:
            date: Knock-in date.
            spot: Spot at knock-in.
            barrier: Barrier level that was breached.

        Returns:
            A non-terminal ``LifecycleEvent`` of type ``KNOCK_IN``.
        """
        before = self._snapshot()
        self.state.knocked_in = True
        self.state.hit_date = date.to_pydatetime() if hasattr(date, "to_pydatetime") else date
        after = self._snapshot()
        return LifecycleEvent(
            event_type=LifecycleEventType.KNOCK_IN,
            date=date,
            spot=spot,
            barrier=barrier,
            payoff=0.0,
            cashflow=0.0,
            terminates_position=False,
            state_before=before,
            state_after=after,
            metadata=metadata if metadata is not None else {"monitoring": "daily_close"},
        )

    def _terminal_event(
        self,
        event_type: LifecycleEventType,
        date: pd.Timestamp,
        spot: float,
        *,
        barrier: Optional[float],
        payoff: float,
        expired: bool = False,
        metadata: Optional[Dict] = None,
    ) -> LifecycleEvent:
        """Record a terminal event (KO or expiry) and update state.

        Args:
            event_type: ``KNOCK_OUT`` or ``EXPIRY``.
            date: Determination date.
            spot: Spot at settlement.
            barrier: Barrier level (or None for expiry-no-hit).
            payoff: Per-contract payoff amount.
            expired: True when the event is an expiry settlement (vs. mid-path KO).

        Returns:
            A terminal ``LifecycleEvent`` with ``terminates_position=True``.
        """
        before = self._snapshot()

        cashflow = self.quantity * payoff
        timestamp = (
            date.to_pydatetime()
            if hasattr(date, "to_pydatetime")
            else date
        )
        realized = self._realized_cashflow(
            event_type,
            cashflow,
            timestamp,
            expired=expired,
        )
        valuation_point = (
            ValuationPoint(date=timestamp)
            if realized.payment_date is not None
            else ValuationPoint(time=self._elapsed_years(pd.Timestamp(date)))
        )
        self.state.valuation_point = valuation_point
        self.state.ledger.register(realized)

        if expired:
            self.state.expired = True
        else:
            self.state.knocked_out = True
            self.state.hit_date = date.to_pydatetime() if hasattr(date, "to_pydatetime") else date
        self.state.alive = False

        after = self._snapshot()

        return LifecycleEvent(
            event_type=event_type,
            date=date,
            spot=spot,
            barrier=barrier,
            payoff=payoff,
            cashflow=cashflow,
            realized_cashflow=realized,
            terminates_position=True,
            state_before=before,
            state_after=after,
            metadata=metadata if metadata is not None else {"monitoring": "daily_close"},
        )

    def _valuation_point(self, date: pd.Timestamp) -> ValuationPoint:
        if self.state.ledger.cashflows:
            first = self.state.ledger.cashflows[0]
            if first.payment_time is not None:
                return ValuationPoint(time=self._elapsed_years(date))
        convention = getattr(self.product, "settlement_convention", None)
        if (
            convention is not None
            and convention.lag_unit is SettlementLagUnit.YEAR_FRACTION
        ):
            return ValuationPoint(time=self._elapsed_years(date))
        return ValuationPoint(date=pd.Timestamp(date).to_pydatetime())

    def _schedule_resolution_env(
        self, env: PricingEnvironment
    ) -> PricingEnvironment:
        schedule_env = deepcopy(env)
        schedule_env.valuation_date = self.start_date.to_pydatetime()
        return schedule_env

    def _realized_cashflow(
        self,
        event_type: LifecycleEventType,
        amount: float,
        timestamp,
        *,
        expired: bool,
    ) -> RealizedCashflow:
        if self._observation_env is None:
            raise RuntimeError("barrier lifecycle observation environment is missing")

        pays_at_hit = self._pays_at_hit()
        if expired or not pays_at_hit:
            timing = resolve_terminal_timing(
                self.product,
                self._schedule_resolution_env(self._observation_env),
            )
        else:
            timing = SettlementResolver.resolve_contingent(
                self.product,
                SettlementRequest(
                    kind=CashflowKind.HIT,
                    determination_date=timestamp,
                    cashflow_id=(
                        f"{event_type.value.lower()}:{timestamp.isoformat()}"
                    ),
                ),
                self._observation_env,
            )

        cashflow_id = f"{event_type.value.lower()}:{timestamp.isoformat()}"
        if not expired and not pays_at_hit:
            if timing.payment_date is not None:
                return RealizedCashflow(
                    cashflow_id=cashflow_id,
                    event_type=event_type,
                    amount=amount,
                    determination_date=timestamp,
                    payment_date=timing.payment_date,
                )
            return RealizedCashflow(
                cashflow_id=cashflow_id,
                event_type=event_type,
                amount=amount,
                determination_time=self._elapsed_years(
                    pd.Timestamp(timestamp)
                ),
                payment_time=timing.payment_time,
            )

        if (
            timing.determination_date is not None
            and timing.payment_date is not None
        ):
            return RealizedCashflow(
                cashflow_id=cashflow_id,
                event_type=event_type,
                amount=amount,
                determination_date=timing.determination_date,
                payment_date=timing.payment_date,
            )

        delay = float(timing.payment_time) - float(timing.determination_time)
        determination_time = self._elapsed_years(pd.Timestamp(timestamp))
        return RealizedCashflow(
            cashflow_id=cashflow_id,
            event_type=event_type,
            amount=amount,
            determination_time=determination_time,
            payment_time=determination_time + delay,
        )

    def _pays_at_hit(self) -> bool:
        product = self.product
        if isinstance(product, (OneTouchOption, DoubleOneTouchOption)):
            return bool(getattr(product, "payment_at_hit", True))
        return bool(getattr(product, "pay_at_hit", False))
