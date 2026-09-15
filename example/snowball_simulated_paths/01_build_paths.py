"""Stage 01: the realised history into simulated batches.

    .venv/bin/python example/snowball_simulated_paths/01_build_paths.py            # 2,000 x 275
    .venv/bin/python example/snowball_simulated_paths/01_build_paths.py --quick    # 40 paths

Reads the CSI 1000 spot, the IM chain and the admitted IV surfaces from
``example/mo_volmodels/data/history`` (local, untracked), builds the joint
daily history, bootstraps forward paths from its last day, adds the stress
set, and writes ``<out>/paths/{bootstrap,stress}.npz`` with a manifest.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _sim_common as C  # noqa: E402

from quantark.backtest.simulation import (  # noqa: E402
    DEFAULT_TENOR_GRID, MarketPath, PathHistory, StationaryBlockBootstrap, trading_calendar,
)
from quantark.backtest.simulation.results import jsonable  # noqa: E402


def build_history(spot: pd.DataFrame, vol: pd.DataFrame, futures: pd.DataFrame, *, rate: float) -> PathHistory:
    """The joint daily state (ln S, vol, rate, carry curve) on the default tenor grid."""
    return PathHistory.from_frames(spot=spot, vol=vol, futures=futures, rate=float(rate), tenor_grid=DEFAULT_TENOR_GRID)


def cut_history_frames(spot: pd.DataFrame, futures: pd.DataFrame, history_end: date) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """The spot and futures frames through ``history_end`` inclusive.

    Cutting today's cache at 2026-09-09 reproduces the banked batch's
    history, bootstrap and stress fingerprints exactly (checked 2026-09-15),
    so a larger batch can start from the same state as a banked one.
    """
    end = pd.Timestamp(history_end)
    cut_spot = spot[pd.to_datetime(spot["date"]) <= end].reset_index(drop=True)
    cut_futures = futures[pd.to_datetime(futures["date"]) <= end].reset_index(drop=True)
    if cut_spot.empty:
        raise C.Q.StudyDataError(f"no history on or before {end.date()}")
    return cut_spot, cut_futures


def load_real_history(history_dir, *, rate: float, vol_tenor: float, history_end: Optional[date] = None) -> PathHistory:
    """The study's inputs through the q study's fail-closed loaders, optionally cut at ``history_end``."""
    frames = C.Q.load_history(history_dir)
    spot, futures = frames.spot, frames.futures
    if history_end is not None:
        spot, futures = cut_history_frames(spot, futures, history_end)
    surfaces = C.Q.surface_history(history_dir)
    vol = C.Q.atm_vol_channel([pd.Timestamp(d) for d in spot["date"]], surfaces, vol_tenor)
    return build_history(spot, vol, futures, rate=rate)


def build_paths(
    history: PathHistory, *, n_paths: int, n_days: int, seed: int, mean_block_days: int,
    annual_drift: float, vol_floor: float, carry_mode: str,
) -> Tuple[MarketPath, MarketPath]:
    """The bootstrap batch and the stress set, both from the last day's state on a forward weekday calendar."""
    start = history.snapshot()
    first = (pd.Timestamp(history.dates[-1]) + timedelta(days=1)).date()
    calendar = trading_calendar(first, n_days)
    generator = StationaryBlockBootstrap(
        history, mean_block_days=mean_block_days, demean_returns=True, annual_drift=annual_drift,
        vol_floor=vol_floor, carry_mode=carry_mode, start=start, calendar=calendar,
    )
    bootstrap = generator.generate(n_paths, n_days, seed=seed)
    stress = C.stress_paths(start, calendar, history.tenor_grid)
    return bootstrap, stress


def write_paths(out_dir, bootstrap: MarketPath, stress: MarketPath, *, history: PathHistory,
                history_end: Optional[date] = None) -> Dict[str, Any]:
    out = Path(out_dir) / "paths"
    out.mkdir(parents=True, exist_ok=True)
    bootstrap.to_npz(out / "bootstrap.npz")
    stress.to_npz(out / "stress.npz")
    record = {
        "bootstrap_fingerprint": bootstrap.fingerprint(), "stress_fingerprint": stress.fingerprint(),
        "history_fingerprint": history.source_fingerprint,
        "history_first_day": str(history.dates[0].date()), "history_last_day": str(history.dates[-1].date()),
        "n_paths": bootstrap.n_paths, "n_days": bootstrap.n_days,
        "calendar_first_day": str(bootstrap.dates[0].date()), "calendar_last_day": str(bootstrap.dates[-1].date()),
        "bootstrap_meta": bootstrap.meta, "stress_meta": stress.meta,
    }
    if history_end is not None:
        record["history_end"] = str(history_end)
    manifest = jsonable(record)         # JSON-safe now, so it equals what load_paths reads back
    C.write_json(out / "manifest.json", manifest)
    return manifest


def load_paths(out_dir) -> Tuple[MarketPath, MarketPath, Dict[str, Any]]:
    out = Path(out_dir) / "paths"
    for name in ("bootstrap.npz", "stress.npz", "manifest.json"):
        if not (out / name).exists():
            raise C.Q.StudyDataError(f"missing {out / name}; run 01_build_paths.py first")
    return (MarketPath.from_npz(out / "bootstrap.npz"), MarketPath.from_npz(out / "stress.npz"),
            C.read_json(out / "manifest.json"))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--history-dir", type=Path, default=C.Q.DEFAULT_HISTORY_DIR)
    parser.add_argument("--out-dir", type=Path, default=C.DEFAULT_OUT_DIR)
    parser.add_argument("--n-paths", type=int, default=C.N_PATHS)
    parser.add_argument("--n-days", type=int, default=C.N_DAYS)
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--mean-block-days", type=int, default=C.MEAN_BLOCK_DAYS)
    parser.add_argument("--vol-floor", type=float, default=C.VOL_FLOOR)
    parser.add_argument("--rate", type=float, default=C.Q.FLAT_RATE)
    parser.add_argument("--vol-tenor", type=float, default=C.Q.ATM_VOL_TENOR_YEARS)
    parser.add_argument("--history-end", type=date.fromisoformat, default=None,
                        help="cut the history at this day inclusive (YYYY-MM-DD)")
    parser.add_argument("--quick", action="store_true", help=f"{C.QUICK_PATHS} paths")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    n_paths = C.QUICK_PATHS if args.quick else args.n_paths
    history = load_real_history(args.history_dir, rate=args.rate, vol_tenor=args.vol_tenor,
                                history_end=args.history_end)
    bootstrap, stress = build_paths(
        history, n_paths=n_paths, n_days=args.n_days, seed=args.seed, mean_block_days=args.mean_block_days,
        annual_drift=C.ANNUAL_DRIFT, vol_floor=args.vol_floor, carry_mode=C.CARRY_MODE,
    )
    manifest = write_paths(args.out_dir, bootstrap, stress, history=history, history_end=args.history_end)
    print(f"history {manifest['history_first_day']}..{manifest['history_last_day']} ({history.n_days} days), "
          f"{bootstrap.n_paths} bootstrap paths x {bootstrap.n_days} days from {manifest['calendar_first_day']}, "
          f"{stress.n_paths} stress paths, vol floor hits {bootstrap.meta.get('vol_floor_hits')}"
          f"{f', history cut at {args.history_end}' if args.history_end else ''}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
