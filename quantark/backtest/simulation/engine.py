"""The vectorised daily loop (spec 4.3, 8).

One pass over the calendar, every per-path quantity an array.  The order of
operations inside a day, the accounting identity and the hedge rule are
``ReplayBacktestEngine``'s; the conformance oracle checks that claim on a
real path.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from quantark.asset.equity.lifecycle import AutocallableLifecycleTracker
from quantark.asset.equity.lifecycle.state import AutocallableLifecycleState
from quantark.backtest.replay.market import ImpliedBasisYield, derive_implied_dividend_yield
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import ValidationError

from .carry import day_chain
from .config import EnsembleConfig
from .dividends import dividend_yield_for_day
from .hedge import (
    POSITION_EPS,
    VectorHedgeLedger,
    day_active_contract,
    should_rebalance_vector,
    target_contracts_vector,
)
from .lifecycle import LifecycleRecord, VectorLifecycle, receivable_pv
from .paths.market_path import MarketPath
from .pricing.base import DayStates, row_keys
from .pricing.cache import StateCache
from .pricing.repricing import RepricingPricer

FLOAT_COLUMNS = (
    "portfolio_value", "product_mtm", "hedge_mtm", "cash", "cashflows", "transaction_costs",
    "product_pnl", "hedge_pnl", "total_pnl", "spot", "volatility", "rate", "pricing_q",
    "implied_q", "basis_yield", "futures_price", "futures_contracts", "pre_hedge_contracts",
    "delta", "gamma", "pending_receivable_pv",
)
BOOL_COLUMNS = ("alive", "knocked_in", "knocked_out", "matured", "settled")

TRADE_COLUMNS = [
    "path", "day", "date", "trade_type", "contract", "quantity", "price",
    "multiplier", "notional", "transaction_cost", "reason",
]


def empty_states(day_index: int, date) -> DayStates:
    """A ``DayStates`` with no paths, for reports that need the shape only."""
    nothing = np.array([])
    return DayStates(
        day_index=day_index, date=date, path_index=np.array([], dtype=int), spot=nothing,
        vol=nothing, rate=nothing, q_T=nothing, div_yield=(), basis_yield=nothing,
        env_key=np.array([], dtype=np.int64), knocked_in=np.array([], dtype=bool),
    )


class StateCube:
    """The replay's state columns that have a per-path meaning (spec 4.3)."""

    def __init__(self, dates: pd.DatetimeIndex, n_paths: int) -> None:
        self.dates = pd.DatetimeIndex(dates)
        self.n_paths = int(n_paths)
        self.active_contract: List[str] = []
        shape = (self.n_paths, len(self.dates))
        for name in FLOAT_COLUMNS:
            setattr(self, name, np.zeros(shape))
        for name in BOOL_COLUMNS:
            setattr(self, name, np.zeros(shape, dtype=bool))

    def frame(self, i: int, last_day: int) -> pd.DataFrame:
        """One path's state rows up to and including ``last_day``."""
        end = int(last_day) + 1
        data: Dict[str, Any] = {
            "date": self.dates[:end],
            "active_contract": self.active_contract[:end],
        }
        for name in FLOAT_COLUMNS + BOOL_COLUMNS:
            data[name] = getattr(self, name)[i, :end]
        return pd.DataFrame(data)

    def freeze_from(self, day: int) -> None:
        """Repeat day ``day``'s values over the rest of the calendar.

        The loop stops once every path has settled; a settled path's columns
        repeat its terminal values, which is the per-path reading of the
        replay's ``terminate_on_lifecycle_end``.
        """
        for name in FLOAT_COLUMNS + BOOL_COLUMNS:
            column = getattr(self, name)
            column[:, day + 1:] = column[:, day: day + 1]
        if self.active_contract:
            self.active_contract += [self.active_contract[-1]] * (
                len(self.dates) - len(self.active_contract)
            )


@dataclass
class EnsembleResults:
    """The run's cube, event logs and manifest (distributions land in plan 4)."""

    cube: StateCube
    trades: List[Dict[str, Any]]
    events: List[LifecycleRecord]
    manifest: Dict[str, Any]
    last_day: np.ndarray
    initial_book_value: np.ndarray

    @property
    def n_paths(self) -> int:
        return int(self.cube.n_paths)

    def path_states(self, i: int) -> pd.DataFrame:
        """One path's daily state rows, in the replay's schema, to its last day."""
        return self.cube.frame(i, int(self.last_day[i]))

    def path_trades(self, i: int) -> pd.DataFrame:
        rows = [t for t in self.trades if t["path"] == i]
        return pd.DataFrame(rows, columns=TRADE_COLUMNS)

    def path_events(self, i: int) -> pd.DataFrame:
        rows = [e.__dict__ for e in self.events if e.path == i]
        return pd.DataFrame(
            rows,
            columns=["product", "path", "day", "event", "index", "spot", "barrier", "cashflow"],
        )


class EnsembleBacktestEngine:
    """Runs a book of snowballs over a batch of simulated paths."""

    def __init__(self, config: EnsembleConfig) -> None:
        self.config = config
        self.quantities = config.quantities

    def run(self, paths: MarketPath) -> EnsembleResults:
        """Walk the calendar once over every path in ``paths``."""
        cfg = self.config
        started = time.perf_counter()
        n_paths, n_days = paths.n_paths, paths.n_days
        dates = paths.dates
        multiplier = float(cfg.hedge.multiplier)

        cache = StateCache(cfg.pricing.cache)
        pricers = [
            RepricingPricer(
                bp.product, engine_config=cfg.engine_config, start_date=dates[0],
                underlying=cfg.underlying, cache=cache,
                delta_bump_size=cfg.delta_bump_size, gamma_bump_size=cfg.gamma_bump_size,
            )
            for bp in cfg.products
        ]
        schedule_env = self._schedule_env(dates[0])
        schedules = [
            self._tracker(bp, dates[0]).resolve_calendar_schedule(dates, schedule_env)
            for bp in cfg.products
        ]
        self._check_calendar(schedules, n_days)

        lifecycle = VectorLifecycle(schedules, self.quantities, n_paths=n_paths)
        ledger = VectorHedgeLedger(n_paths)
        cube = StateCube(dates, n_paths)
        trades: List[Dict[str, Any]] = []
        events: List[LifecycleRecord] = []
        costs = np.zeros(n_paths)
        initial_book_value: Optional[np.ndarray] = None
        settled_day = np.full(n_paths, -1, dtype=np.int64)
        current_contract: Optional[str] = None
        dividend_builds = 0
        last_executed = n_days - 1

        for d in range(n_days):
            day = pd.Timestamp(dates[d])
            spot, vol, rate = paths.spot[:, d], paths.atm_vol[:, d], paths.rate[:, d]

            chain = day_chain(paths, d, multiplier=multiplier)
            code, column = day_active_contract(chain, cfg.hedge.roll_policy, current_contract)
            futures_price = chain.prices[:, column]
            if current_contract is not None and code != current_contract:
                self._roll(ledger, chain, code, column, current_contract, d, day, trades, costs)
            current_contract = code
            cube.active_contract.append(code)

            env_key, div_yield, basis, implied_q, builds = self._day_market(
                paths, d, chain, column, code, rate
            )
            dividend_builds += builds
            q_T = self._pricing_q(pricers[0], day, div_yield, implied_q)

            if initial_book_value is None:
                initial_book_value = self._initial_book_value(
                    pricers, day, d, spot, vol, rate, q_T, div_yield, basis, env_key
                )

            events += lifecycle.step(
                d, spot,
                lambda p, idx, ki, day=day, spot=spot, vol=vol, rate=rate,
                       div_yield=div_yield, basis=basis: self._payoffs(
                    pricers[p], day, idx, ki, spot, vol, rate, div_yield, basis
                ),
            )

            product_mtm, net_delta, net_gamma = self._price_book(
                pricers, lifecycle, day, d, spot, vol, rate, q_T, div_yield, basis, env_key
            )
            receivable = self._receivable_pv(lifecycle, rate, d, schedules)
            cashflows = lifecycle.realized.sum(axis=0)
            flags = lifecycle.book_flags()

            pre_hedge = ledger.quantity.copy()
            self._rebalance(ledger, net_delta, flags["alive"], futures_price, code,
                            multiplier, d, day, trades, costs)

            hedge_mtm = ledger.mark_to_market(futures_price)
            product_pnl = product_mtm + receivable + cashflows - initial_book_value
            total_pnl = product_pnl + hedge_mtm - costs
            cash = cashflows - costs
            portfolio_value = product_mtm + hedge_mtm + cash + receivable

            self._record(
                cube, d, spot=spot, vol=vol, rate=rate, q_T=q_T, implied_q=implied_q, basis=basis,
                futures_price=futures_price, pre_hedge=pre_hedge, contracts=ledger.quantity,
                product_mtm=product_mtm, receivable=receivable, cashflows=cashflows, costs=costs,
                product_pnl=product_pnl, hedge_mtm=hedge_mtm, total_pnl=total_pnl, cash=cash,
                portfolio_value=portfolio_value, delta=net_delta, gamma=net_gamma, flags=flags,
            )

            newly = flags["settled"] & (settled_day < 0)
            settled_day[newly] = d
            last_executed = d
            if flags["settled"].all():
                cube.freeze_from(d)
                break

        last_day = np.where(settled_day >= 0, settled_day, last_executed)
        manifest = self._manifest(paths, pricers, cache, dividend_builds, lifecycle, started,
                                  days_run=last_executed + 1)
        return EnsembleResults(cube=cube, trades=trades, events=events, manifest=manifest,
                               last_day=last_day, initial_book_value=initial_book_value)

    # -- setup ---------------------------------------------------------

    def _tracker(self, bp, start_date) -> AutocallableLifecycleTracker:
        return AutocallableLifecycleTracker(
            product=bp.product, quantity=float(bp.quantity), has_lifecycle=True,
            lifecycle=AutocallableLifecycleState(), start_date=start_date,
        )

    def _schedule_env(self, day) -> PricingEnvironment:
        """A throwaway environment carrying only the date, for schedule work."""
        return PricingEnvironment(
            spot_quote=SpotQuote(spot=1.0, asset_name=self.config.underlying),
            vol_surface=FlatVolSurface(volatility=0.2), rate_curve=FlatRateCurve(rate=0.0),
            valuation_date=pd.Timestamp(day).to_pydatetime(),
        )

    def _check_calendar(self, schedules, n_days: int) -> None:
        short = [s for s in schedules if s.terminal_settlement_day < 0]
        if short and not self.config.allow_data_end:
            raise ValidationError(
                f"the path calendar of {n_days} days ends before {len(short)} product(s) "
                "settle; extend the paths or set allow_data_end=True"
            )

    # -- the day's market ----------------------------------------------

    def _day_market(self, paths, d, chain, column, code, rate):
        """Dividend object, basis and legacy scalar yield per distinct state.

        Mathematically none of the three depends on spot: the chain is
        ``F_j = S exp(B_j)`` and every consumer takes the ratio ``F_j / S``.
        Bitwise they do -- ``calculate_basis_yield`` evaluates ``(F - S) / S``
        and the curve inverts ``ln(F_j / S)``, neither of which reduces to
        ``exp(B_j) - 1`` exactly -- and a one-ulp difference in ``q`` moves a
        PDE price by ~1e-11, which the oracle sees.  So the key that
        identifies the environment carries the spot too: one build serves
        every path with the same ``(rate, spot, carry row)`` (all of them on
        day 0; each on its own afterwards), never a path with another spot.
        """
        n_paths = paths.n_paths
        env_key = row_keys(np.column_stack([rate, paths.spot[:, d], paths.carry[:, d, :]]))
        div_yield: List[Any] = [None] * n_paths
        basis = np.zeros(n_paths)
        implied_q = np.zeros(n_paths)
        tenor = float(chain.tenors[column])
        if tenor <= 0.0:
            # The roll policy only ever selects a contract with expiry after
            # the day, so this cannot happen; fail closed rather than divide.
            raise ValidationError(
                f"active contract {code} has no time to expiry on {chain.date.date()}"
            )
        builds = 0
        for first in np.unique(env_key, return_index=True)[1]:
            i = int(first)
            same = env_key == env_key[i]
            div = dividend_yield_for_day(
                chain, i, spot=float(paths.spot[i, d]), rate=float(rate[i]),
                engine_config=self.config.engine_config, active_contract=code,
                curve_tenors=paths.tenor_grid, curve_carry=paths.carry[i, d, :],
            )
            for j in np.flatnonzero(same):
                div_yield[int(j)] = div
            # The replay's own arithmetic (``ProductReplay.build_env``), so
            # the recorded basis and implied_q agree with it bit for bit.
            b, q = derive_implied_dividend_yield(
                rate=float(rate[i]), spot=float(paths.spot[i, d]),
                futures_price=float(chain.prices[i, column]), time_to_maturity=tenor,
            )
            basis[same] = b
            implied_q[same] = q
            builds += 1
        return env_key, tuple(div_yield), basis, implied_q, builds

    def _pricing_q(self, pricer, day, div_yield, implied_q) -> np.ndarray:
        """The scalar the cube records, per ``ProductReplay.recorded_pricing_q``.

        A legacy channel records the flat yield the pricer received; a term
        source has no single number, so the row carries the zero yield at
        the first product's remaining maturity.
        """
        source = getattr(self.config.engine_config, "dividend_source", None)
        if source not in ("futures_curve", "surface_forwards"):
            return implied_q.copy()
        probe = self._schedule_env(day)
        try:
            remaining = float(pricer.aged_product(day, knocked_in=False).get_maturity(probe))
        except ValidationError:
            remaining = 1e-8      # on/after a date-based maturity: the replay's shortest pillar
        return np.array([float(div.get_yield(remaining)) for div in div_yield])

    # -- pricing -------------------------------------------------------

    def _states_for(self, day, d, idx, knocked_in, spot, vol, rate, q_T, div_yield, basis, env_key):
        return DayStates(
            day_index=d, date=day, path_index=idx, spot=spot[idx], vol=vol[idx], rate=rate[idx],
            q_T=q_T[idx], div_yield=tuple(div_yield[int(i)] for i in idx),
            basis_yield=basis[idx], env_key=env_key[idx], knocked_in=knocked_in,
        )

    def _initial_book_value(self, pricers, day, d, spot, vol, rate, q_T, div_yield, basis, env_key):
        """Day 0's book value, priced BEFORE any lifecycle event, like the replay.

        A ``ReplayProduct`` that states an ``initial_price`` uses it (that is
        how a paired study books the traded price); otherwise the day-0 PV
        stands in, with no knock-in, because nothing has happened yet.
        """
        total = np.zeros(spot.size)
        everyone = np.arange(spot.size)
        for bp, quantity, pricer in zip(self.config.products, self.quantities, pricers):
            if bp.initial_price is not None:
                total += float(quantity) * float(bp.initial_price)
                continue
            states = self._states_for(day, d, everyone, np.zeros(spot.size, dtype=bool),
                                      spot, vol, rate, q_T, div_yield, basis, env_key)
            pv, _, _ = pricer.price_day(states)
            total += float(quantity) * pv
        return total

    def _price_book(self, pricers, lifecycle, day, d, spot, vol, rate, q_T, div_yield, basis, env_key):
        """Mark every alive product-path; a dead one contributes nothing."""
        n_paths = spot.size
        product_mtm = np.zeros(n_paths)
        net_delta = np.zeros(n_paths)
        net_gamma = np.zeros(n_paths)
        for p, pricer in enumerate(pricers):
            idx = np.flatnonzero(lifecycle.alive[p])
            if idx.size == 0:
                continue
            states = self._states_for(day, d, idx, lifecycle.knocked_in[p][idx], spot, vol, rate,
                                      q_T, div_yield, basis, env_key)
            pv, delta, gamma = pricer.price_day(states)
            quantity = float(self.quantities[p])
            product_mtm[idx] += quantity * pv
            net_delta[idx] += quantity * delta
            net_gamma[idx] += quantity * gamma
        return product_mtm, net_delta, net_gamma

    def _payoffs(self, pricer, day, idx, knocked_in, spot, vol, rate, div_yield, basis):
        """``product.get_payoff`` per path, each in its own environment."""
        product = pricer.aged_product(day, knocked_in=False)
        out = np.empty(idx.size)
        for n, i in enumerate(idx):
            i = int(i)
            env = PricingEnvironment(
                spot_quote=SpotQuote(spot=float(spot[i]), asset_name=self.config.underlying),
                vol_surface=FlatVolSurface(volatility=float(vol[i])),
                rate_curve=FlatRateCurve(rate=float(rate[i])),
                div_yield=div_yield[i],
                basis_yield=ImpliedBasisYield(float(basis[i])),
                valuation_date=pd.Timestamp(day).to_pydatetime(),
            )
            out[n] = float(product.get_payoff(float(spot[i]), env, knocked_in=bool(knocked_in[n])))
        return out

    def _receivable_pv(self, lifecycle, rate, d, schedules) -> np.ndarray:
        """``ProductReplay.pending_receivable_pv`` per path, both clocks (see ``receivable_pv``)."""
        return receivable_pv(lifecycle, rate, d)

    def _rebalance(self, ledger, net_delta, any_alive, futures_price, code, multiplier,
                   d, day, trades, costs) -> None:
        """Target the book delta while alive; close out once it is dead."""
        strategy = self.config.strategy
        target = np.where(any_alive, target_contracts_vector(strategy, net_delta, multiplier), 0.0)
        should = np.where(
            any_alive,
            should_rebalance_vector(strategy, ledger.quantity, target),
            np.abs(ledger.quantity) > POSITION_EPS,
        )
        trade_size = np.where(should, target - ledger.quantity, 0.0)
        trade_size[np.abs(trade_size) <= POSITION_EPS] = 0.0
        if not np.any(trade_size):
            return
        trade_type = np.where(any_alive, "hedge_rebalance", "hedge_close")
        reason = np.where(any_alive, "delta_rebalance", "product_terminated")
        self._execute(ledger, trade_size, futures_price, code, multiplier, d, day,
                      trade_type, reason, trades, costs)

    def _roll(self, ledger, chain, code, column, old_contract, d, day, trades, costs) -> None:
        """Close the old lane and reopen in the new contract, both legs costed."""
        qty = ledger.quantity.copy()
        if not np.any(np.abs(qty) > POSITION_EPS):
            return
        if old_contract in chain.contracts:
            old_price = chain.prices[:, chain.contracts.index(old_contract)]
            reason = "futures_roll"
        else:
            # The replay closes at the NEW contract's price when the old one
            # is no longer listed, and records that in the reason.
            old_price = chain.prices[:, column]
            reason = "futures_roll_missing_old_contract"
        reasons = np.full(qty.size, reason)
        self._execute(ledger, -qty, old_price, old_contract, chain.multiplier, d, day,
                      np.full(qty.size, "roll_close"), reasons, trades, costs)
        self._execute(ledger, qty, chain.prices[:, column], code, chain.multiplier, d, day,
                      np.full(qty.size, "roll_open"), reasons, trades, costs)

    def _execute(self, ledger, quantity_delta, price, contract, multiplier, d, day,
                 trade_type, reason, trades, costs) -> None:
        """Book a per-path trade and its cost, one log row per trading path."""
        for i in np.flatnonzero(np.abs(quantity_delta) > POSITION_EPS):
            i = int(i)
            qty, px = float(quantity_delta[i]), float(price[i])
            notional = abs(qty * px * float(multiplier))
            cost = float(self.config.transaction_cost_model.calculate_cost(
                quantity=qty, price=px, notional=notional,
                instrument_type="futures", trade_type=str(trade_type[i]),
            ))
            costs[i] += cost
            trades.append({
                "path": i, "day": d, "date": day, "trade_type": str(trade_type[i]),
                "contract": contract, "quantity": qty, "price": px,
                "multiplier": float(multiplier), "notional": notional,
                "transaction_cost": cost, "reason": str(reason[i]),
            })
        ledger.trade(quantity_delta, price, contract, multiplier)

    # -- recording -----------------------------------------------------

    def _record(self, cube, d, *, spot, vol, rate, q_T, implied_q, basis, futures_price,
                pre_hedge, contracts, product_mtm, receivable, cashflows, costs, product_pnl,
                hedge_mtm, total_pnl, cash, portfolio_value, delta, gamma, flags) -> None:
        """Write the day's column of the state cube."""
        columns = {
            "portfolio_value": portfolio_value, "product_mtm": product_mtm, "hedge_mtm": hedge_mtm,
            "cash": cash, "cashflows": cashflows, "transaction_costs": costs,
            "product_pnl": product_pnl, "hedge_pnl": hedge_mtm, "total_pnl": total_pnl,
            "spot": spot, "volatility": vol, "rate": rate, "pricing_q": q_T,
            "implied_q": implied_q, "basis_yield": basis, "futures_price": futures_price,
            "futures_contracts": contracts, "pre_hedge_contracts": pre_hedge,
            "delta": delta, "gamma": gamma, "pending_receivable_pv": receivable,
        }
        for name, values in columns.items():
            getattr(cube, name)[:, d] = values
        for name in BOOL_COLUMNS:
            getattr(cube, name)[:, d] = flags[name]

    def _manifest(self, paths, pricers, cache, dividend_builds, lifecycle, started, days_run):
        return {
            "path_fingerprint": paths.fingerprint(),
            "days_run": int(days_run),
            "path_meta": dict(paths.meta),
            "engine_fingerprint": pricers[0].fingerprint(),
            "product_fingerprints": [p.product_fingerprint for p in pricers],
            "provider": "repricing",
            "mode": "exact",
            "gate": pricers[0].verify(
                empty_states(0, paths.dates[0]), self.config.pricing.gate
            ).as_dict(),
            "engine_calls": sum(p.stats()["engine_calls"] for p in pricers),
            "cache": cache.stats().as_dict(),
            "dividend_builds": dividend_builds,
            "data_end_paths": int(np.count_nonzero(~lifecycle.settled.all(axis=0))),
            "underlying": self.config.underlying,
            "seconds": time.perf_counter() - started,
            "metadata": dict(self.config.metadata),
        }
