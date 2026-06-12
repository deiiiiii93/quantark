"""
Lifecycle tracking for the vanilla barrier product family.

Covers single/double barrier options, single/double sharkfins, and one-touch /
double one-touch products. Monitoring is daily-close: each ``observe`` call
checks the day's closing spot. Products whose ``observation_type`` is EXPIRY
are only settled at expiry, never intra-path.

Knock-in semantics: a knocked-in single/double barrier option *is* a European
option, so after KI ``product_for_pricing`` returns the exact European
equivalent (and ``engine_for_pricing`` returns a ``BlackScholesEngine``).

All payoff arithmetic uses ``quantark.util.numerical`` — never raw float
comparisons or bare math operations.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional

import pandas as pd

from quantark.asset.equity.engine.analytical import BlackScholesEngine
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
from quantark.util.numerical import is_zero, safe_exp

from .events import LifecycleEvent, LifecycleEventType
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

    Monitoring is daily-close: call ``observe(date, env, spot)`` once per
    business day with the closing spot. Products with ``observation_type ==
    ObservationType.EXPIRY`` are never checked intra-path; they are settled only
    when expiry is detected.

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
        if not self.state.alive:
            return []

        date = pd.Timestamp(date).normalize()

        if isinstance(self.product, (BarrierOption,)):
            return self._observe_single_barrier(date, env, spot)
        elif isinstance(self.product, DoubleBarrierOption):
            return self._observe_double_barrier(date, env, spot)
        elif isinstance(self.product, (SingleSharkfinOption, DoubleSharkfinOption)):
            return self._observe_sharkfin(date, env, spot)
        elif isinstance(self.product, (OneTouchOption, DoubleOneTouchOption)):
            return self._observe_one_touch(date, env, spot)

        return []

    # ------------------------------------------------------------------
    # Per-product-family observation helpers
    # ------------------------------------------------------------------

    def _observe_single_barrier(
        self, date: pd.Timestamp, env: PricingEnvironment, spot: float
    ) -> List[LifecycleEvent]:
        """Lifecycle observation for ``BarrierOption``."""
        product: BarrierOption = self.product
        barrier_hit = product.is_barrier_hit(spot)

        if product.is_knock_out:
            # Intra-path KO check
            if barrier_hit and self._monitored_intra_path():
                payoff = self._rebate_payoff(
                    rebate=product.rebate,
                    pay_at_hit=product.pay_at_hit,
                    date=date,
                    env=env,
                    contract_multiplier=product.contract_multiplier,
                )
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
        self, date: pd.Timestamp, env: PricingEnvironment, spot: float
    ) -> List[LifecycleEvent]:
        """Lifecycle observation for ``DoubleBarrierOption``."""
        product: DoubleBarrierOption = self.product
        barrier_hit = product.is_barrier_hit(spot)

        if product.is_knock_out:
            if barrier_hit and self._monitored_intra_path():
                # DoubleBarrierOption has no pay_at_hit; treat as pay-at-expiry
                # (PV-discount the rebate).
                payoff = self._rebate_payoff(
                    rebate=product.rebate,
                    pay_at_hit=False,
                    date=date,
                    env=env,
                    contract_multiplier=product.contract_multiplier,
                )
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
        self, date: pd.Timestamp, env: PricingEnvironment, spot: float
    ) -> List[LifecycleEvent]:
        """Lifecycle observation for ``SingleSharkfinOption`` and ``DoubleSharkfinOption``.

        Sharkfins are always knock-out structures. Barrier hit → knock-out with
        the barrier payoff (PV-discounted when ``pay_at_hit=False``). Expiry
        without a hit → no-hit payoff.
        """
        product = self.product
        barrier_hit = product.is_barrier_hit(spot)

        if barrier_hit and self._monitored_intra_path():
            raw_payoff = product.get_barrier_payoff()
            if product.pay_at_hit:
                payoff = raw_payoff
            else:
                payoff = self._discount_to_expiry(raw_payoff, date, env)
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
        self, date: pd.Timestamp, env: PricingEnvironment, spot: float
    ) -> List[LifecycleEvent]:
        """Lifecycle observation for ``OneTouchOption`` and ``DoubleOneTouchOption``.

        One-touch: barrier hit → KNOCK_OUT (terminal). Payment at hit or
        discounted to expiry per ``payment_at_hit``. No-touch: barrier never
        touched → pay rebate at expiry.

        For double variants the same logic applies with two barriers.
        """
        product = self.product
        barrier_hit = product.is_barrier_hit(spot)

        if barrier_hit and self._monitored_intra_path():
            raw_payoff = product.get_payoff(spot, touched=True)
            payment_at_hit = getattr(product, "payment_at_hit", True)
            if payment_at_hit or is_zero(raw_payoff):
                payoff = raw_payoff
            else:
                payoff = self._discount_to_expiry(raw_payoff, date, env)

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
    # Payoff / discount helpers
    # ------------------------------------------------------------------

    def _rebate_payoff(
        self,
        rebate: float,
        pay_at_hit: bool,
        date: pd.Timestamp,
        env: PricingEnvironment,
        *,
        contract_multiplier: float,
    ) -> float:
        """Compute the per-contract rebate payoff, optionally PV-discounted.

        Args:
            rebate: Raw rebate amount (before multiplier).
            pay_at_hit: If True, return undiscounted amount; if False, PV-discount.
            date: Hit date.
            env: Pricing environment.
            contract_multiplier: Contract multiplier to apply to the rebate.

        Returns:
            Rebate payoff per contract (scaled by contract_multiplier).
        """
        amount = rebate * contract_multiplier
        if pay_at_hit:
            return amount
        return self._discount_to_expiry(amount, date, env)

    def _discount_to_expiry(
        self,
        amount: float,
        date: pd.Timestamp,
        env: PricingEnvironment,
    ) -> float:
        """PV-discount *amount* from *date* to the current remaining maturity.

        When remaining maturity is zero (or negligibly small), no discounting
        is applied.

        Args:
            amount: Amount to discount.
            date: Current observation date.
            env: Pricing environment (provides ``get_rate``).

        Returns:
            Discounted present value of *amount*.
        """
        remaining = self._remaining_maturity(date)
        if is_zero(remaining) or remaining <= 0.0:
            return amount
        rate = env.get_rate(remaining)
        return float(amount * safe_exp(-rate * remaining))

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
            metadata={"monitoring": "daily_close"},
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
    ) -> LifecycleEvent:
        """Record a terminal event (KO or expiry) and update state.

        Args:
            event_type: ``KNOCK_OUT`` or ``EXPIRY``.
            date: Settlement date.
            spot: Spot at settlement.
            barrier: Barrier level (or None for expiry-no-hit).
            payoff: Per-contract payoff amount.
            expired: True when the event is an expiry settlement (vs. mid-path KO).

        Returns:
            A terminal ``LifecycleEvent`` with ``terminates_position=True``.
        """
        before = self._snapshot()

        cashflow = self.quantity * payoff
        self.state.realized_cashflows += cashflow

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
            terminates_position=True,
            state_before=before,
            state_after=after,
            metadata={"monitoring": "daily_close"},
        )
