"""The replay engine as the simulation's oracle (spec 10).

One path, both engines, the same settings.  In exact repricing mode the
two make the same engine calls on the same aged product in the same
environment, so the target is equality: the lifecycle flags, the active
contract, the hedge size and every trade must match exactly, and the
marked columns to float rounding.  A gap is reported, never absorbed.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from quantark.backtest.replay import ReplayBacktestConfig, ReplayBacktestEngine
from quantark.util.exceptions import ValidationError

from .config import EnsembleConfig
from .dataset import to_market_dataset
from .engine import EnsembleBacktestEngine
from .paths.market_path import MarketPath

#: Columns that must agree exactly, whatever the tolerances.
LIFECYCLE_COLUMNS = ("alive", "knocked_in", "knocked_out", "matured", "active_contract")
#: ...plus the hedge size, exact unless a ``contracts_tolerance`` is stated
#: (an approximate delta can round to a different hand).
EXACT_COLUMNS = LIFECYCLE_COLUMNS + ("futures_contracts",)
#: Columns compared against the tolerances.
NUMERIC_COLUMNS = ("product_mtm", "total_pnl")


@dataclass(frozen=True)
class OracleReport:
    """What the two engines agreed and disagreed about on one path."""

    path_index: int
    days: int
    exact_columns_match: bool
    max_pv_gap: float
    max_delta_gap: float
    max_total_pnl_gap: float
    trade_mismatches: int
    first_mismatch: Optional[str]
    passed: bool
    max_contracts_gap: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "path_index": self.path_index, "days": self.days,
            "exact_columns_match": self.exact_columns_match, "max_pv_gap": self.max_pv_gap,
            "max_delta_gap": self.max_delta_gap, "max_total_pnl_gap": self.max_total_pnl_gap,
            "max_contracts_gap": self.max_contracts_gap,
            "trade_mismatches": self.trade_mismatches, "first_mismatch": self.first_mismatch,
            "passed": self.passed,
        }

    def summary(self) -> str:
        verdict = "matches" if self.passed else "DIFFERS FROM"
        lines = [
            f"simulated path {self.path_index} {verdict} the replay engine over {self.days} days",
            f"  max product mark gap : {self.max_pv_gap:.6g}",
            f"  max delta gap        : {self.max_delta_gap:.6g}",
            f"  max total P&L gap    : {self.max_total_pnl_gap:.6g}",
            f"  max contracts gap    : {self.max_contracts_gap:.6g}",
            f"  trade mismatches     : {self.trade_mismatches}",
        ]
        if self.first_mismatch:
            lines.append(f"  first mismatch       : {self.first_mismatch}")
        return "\n".join(lines)


def run_oracle(
    config: EnsembleConfig,
    paths: MarketPath,
    path_index: int,
    *,
    pv_tolerance: float = 0.0,
    delta_tolerance: float = 0.0,
    contracts_tolerance: float = 0.0,
    _replay_strategy: Any = None,
) -> OracleReport:
    """Compare the ensemble engine with the replay engine on one path.

    The lifecycle flags and the active contract must always agree exactly.
    ``futures_contracts`` and the trade sizes are exact too unless
    ``contracts_tolerance`` (hands) is stated, for an approximate provider
    whose delta can round to a different hand.  ``_replay_strategy``
    overrides the strategy on the replay side only; it exists so a test can
    introduce a known disagreement and check that this function reports it.
    """
    if not 0 <= path_index < paths.n_paths:
        raise ValidationError(f"path index {path_index} out of range for {paths.n_paths} paths")

    ensemble = EnsembleBacktestEngine(config).run(paths)
    simulated = ensemble.path_states(path_index)
    sim_trades = ensemble.path_trades(path_index)

    replay = ReplayBacktestEngine(
        ReplayBacktestConfig(
            products=list(config.products),
            market_data=to_market_dataset(paths, path_index, multiplier=config.hedge.multiplier),
            hedge=config.hedge,
            engine_config=config.engine_config,
            strategy=_replay_strategy or config.strategy,
            transaction_cost_model=config.transaction_cost_model,
            underlying=config.underlying,
            delta_bump_size=config.delta_bump_size,
            gamma_bump_size=config.gamma_bump_size,
            calculate_surfaces=False,
            calculate_event_probabilities=False,
            terminate_on_lifecycle_end=True,
        )
    ).run()
    expected = replay.states_df()
    expected_greeks = replay.greeks_df()

    days = min(len(simulated), len(expected))
    mismatches: List[str] = []
    exact_ok = True
    columns = EXACT_COLUMNS if contracts_tolerance == 0.0 else LIFECYCLE_COLUMNS
    for name in columns:
        left = simulated[name].to_numpy()[:days]
        right = expected[name].to_numpy()[:days]
        bad = np.flatnonzero(left != right)
        if bad.size:
            exact_ok = False
            d = int(bad[0])
            mismatches.append(f"day {d}: {name} {left[d]!r} vs {right[d]!r}")
    if len(simulated) != len(expected):
        exact_ok = False
        mismatches.append(f"day count {len(simulated)} vs {len(expected)}")

    def gap(name: str, frame: pd.DataFrame) -> float:
        if not days:
            return 0.0
        return float(np.max(np.abs(simulated[name].to_numpy()[:days] - frame[name].to_numpy()[:days])))

    max_pv_gap = gap("product_mtm", expected)
    max_total_pnl_gap = gap("total_pnl", expected)
    max_delta_gap = gap("delta", expected_greeks)
    max_contracts_gap = 0.0 if contracts_tolerance == 0.0 else gap("futures_contracts", expected)

    trade_mismatches = _compare_trades(sim_trades, replay.trades_df(), mismatches,
                                       contracts_tolerance=contracts_tolerance)

    passed = (
        exact_ok
        and trade_mismatches == 0
        and max_pv_gap <= pv_tolerance
        and max_delta_gap <= delta_tolerance
        and max_total_pnl_gap <= pv_tolerance
        and max_contracts_gap <= contracts_tolerance
    )
    return OracleReport(
        path_index=int(path_index), days=days, exact_columns_match=exact_ok,
        max_pv_gap=max_pv_gap, max_delta_gap=max_delta_gap, max_total_pnl_gap=max_total_pnl_gap,
        trade_mismatches=trade_mismatches,
        first_mismatch=mismatches[0] if mismatches else None, passed=passed,
        max_contracts_gap=max_contracts_gap,
    )


def _compare_trades(simulated: pd.DataFrame, expected: pd.DataFrame,
                    mismatches: List[str], *, contracts_tolerance: float = 0.0) -> int:
    """Trade for trade: same day, type, contract, size and price.

    With a ``contracts_tolerance`` the size may differ by up to that many
    hands (a rounding difference is a size difference, not a missing trade).
    """
    fields = ("trade_type", "contract", "quantity", "price")

    def rows(frame: pd.DataFrame) -> list:
        if frame.empty:
            return []
        return [
            (pd.Timestamp(r["date"]).normalize(), *(r[f] for f in fields))
            for _, r in frame.iterrows()
        ]

    left, right = rows(simulated), rows(expected)
    bad = 0
    for n, (a, b) in enumerate(zip(left, right)):
        if contracts_tolerance == 0.0:
            same = a == b
        else:
            same = a[0] == b[0] and a[1] == b[1] and a[2] == b[2] and abs(a[3] - b[3]) <= contracts_tolerance
        if not same:
            bad += 1
            if len(mismatches) < 5:
                mismatches.append(f"trade {n}: {a} vs {b}")
    if len(left) != len(right):
        bad += abs(len(left) - len(right))
        mismatches.append(f"trade count {len(left)} vs {len(right)}")
    return bad


def main(argv: Optional[List[str]] = None) -> int:
    """CLI: check one path of a run built by an entry point module."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", type=int, required=True, help="path index to check")
    parser.add_argument(
        "--builder", required=True,
        help="import path of a callable returning (EnsembleConfig, MarketPath), "
             "e.g. mypkg.cells:desk_a",
    )
    args = parser.parse_args(argv)
    module_name, _, attribute = args.builder.partition(":")
    if not attribute:
        raise ValidationError("--builder must look like 'module:callable'")
    module = __import__(module_name, fromlist=[attribute])
    config, paths = getattr(module, attribute)()
    report = run_oracle(config, paths, args.path)
    print(report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
