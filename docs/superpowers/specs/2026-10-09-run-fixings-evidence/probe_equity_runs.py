"""E1-E3: a dated range accrual and a dated Asian through the equity backtest, the equity dynamic scenario and the
replay; the numeric twins alongside.

For every run day k the run's own value is compared with the SAME run truncated at day k whose contract has every
fixing dated on/before day k recorded BY HAND at its own day's spot (``recorded_by_hand``). The difference is the
P&L the run invents by re-deciding a passed fixing at day k's spot. Run on branch w5b-fills-unbumped (b376cc18).
"""
import logging
import sys
import warnings
from datetime import timedelta

import pandas as pd

import _fixtures as F
from quantark.asset.equity.engine.analytical import AsianOptionAnalyticalEngine, RangeAccrualAnalyticalEngine
from quantark.backtest import BacktestConfig, BacktestEngine, ZeroCostModel
from quantark.backtest.strategy import DeltaNeutralStrategy
from quantark.dynamicscenario import DynamicScenarioConfig, DynamicScenarioEngine
from quantark.dynamicscenario.path.day_path import DayPath, DayStep, ParameterChange
from quantark.portfolio import Position
from quantark.portfolio.equity.portfolio import EquityPortfolio
from quantark.stresstest.stress.stress_types import StressType

ENGINES = {"ra": RangeAccrualAnalyticalEngine, "asian": AsianOptionAnalyticalEngine}
MAKERS = {("ra", "dated"): F.ra_dated, ("ra", "numeric"): F.ra_numeric,
          ("asian", "dated"): F.asian_dated, ("asian", "numeric"): F.asian_numeric}
SHOW = [0, F.FIX_DAY - 1, F.FIX_DAY, F.FIX_DAY + 1, 10, 14, F.FIX2_DAY, 20, 24]


def day_of_date(d):
    return (d - F.START).days


# ---------------------------------------------------------------- equity backtest
def backtest(product, kind, n):
    adapter = F.DailyAdapter(n)
    cfg = BacktestConfig(
        strategy=DeltaNeutralStrategy(delta_threshold=1e12),
        start_date=F.START, end_date=adapter.dates[-1].to_pydatetime(), underlying="EQUITY_INDEX",
        initial_positions=[Position(product=product, quantity=1.0, entry_price=0.0, underlying="EQUITY_INDEX",
                                    engine=ENGINES[kind](), entry_timestamp=F.START)],
        market_data_adapter=adapter, transaction_cost_model=ZeroCostModel(), handle_lifecycle_events=False,
    )
    return BacktestEngine(cfg).run().get_value_series().tolist()


# ---------------------------------------------------------------- equity dynamic scenario
def dynamic(product, kind, n):
    pf = EquityPortfolio(portfolio_name="p", pricing_environments={"EQUITY_INDEX": F.eq_env(0)},
                         creation_date=F.START)
    pf.add_position(product=product, quantity=1.0, entry_price=0.0, underlying="EQUITY_INDEX",
                    engine=ENGINES[kind](), entry_timestamp=F.START)
    steps = [DayStep(day_index=k, changes=[ParameterChange("spot", StressType.VALUE, F.spot_on(k))])
             for k in range(n)]
    path = DayPath(name="fix", steps=steps, start_date=F.START)
    cfg = DynamicScenarioConfig(calculate_greeks=True, handle_lifecycle_events=False)
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        res = DynamicScenarioEngine(cfg).run(pf, path)
    return [d.portfolio_value for d in res.day_results]


# ---------------------------------------------------------------- replay (analytical route: the Asian only)
def replay(product, kind, n, first=0):
    from replay_golden import fixtures
    from quantark.backtest.replay.config import AutocallableEngineConfig
    from quantark.backtest.replay.engine import ReplayBacktestEngine
    from quantark.backtest.replay.market import AutocallableMarketDataSet
    from quantark.util.enum.engine_enums import EngineType

    dates = pd.date_range(F.START + timedelta(days=first), periods=n, freq="D")
    spots = [F.spot_on(first + k) for k in range(n)]
    fut = []
    for d, s in zip(dates, spots):
        fut.append({"date": d, "contract": "IF2403", "futures_price": s * 1.004,
                    "expiry_date": pd.Timestamp("2024-03-15"), "multiplier": 300.0})
    md = AutocallableMarketDataSet.from_dataframes(
        spot_data=pd.DataFrame({"date": dates, "spot": spots}),
        vol_data=pd.DataFrame({"date": dates, "volatility": [F.VOL] * n}),
        rate_data=pd.DataFrame({"date": dates, "rate": [F.RATE] * n}),
        futures_data=pd.DataFrame(fut),
    )
    cfg = fixtures.make_book_config()
    cfg.market_data = md
    cfg.products = [fixtures.BookProduct(product=product, quantity=1.0, position_id=1, has_lifecycle=False)]
    cfg.calculate_event_probabilities = False
    cfg.engine_config = AutocallableEngineConfig(pricing_engine_type=EngineType.ANALYTICAL)
    return ReplayBacktestEngine(cfg).run().states_df()["product_mtm"].tolist()


def reference(run, kind, k):
    """Day k's value of the contract with the fixings dated STRICTLY before day k recorded by hand at their own
    day's spot (a fixing dated day k is live: t = 0, decided at day k's spot by the engine itself).

    Backtest and dynamic scenario build exactly ``F.eq_env(k)``, so the engine prices it directly there. The replay
    builds its own environment (futures-implied carry), so it is replayed over a window that STARTS on day k."""
    product = F.recorded_by_hand(kind, k - 1, day_of_date) if k > 0 else MAKERS[(kind, "dated")]()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if run is replay:
            return replay(product, kind, 2, first=k)[0]
        return ENGINES[kind]().price(product, F.eq_env(k))


def report(name, run, kinds):
    print(f"\n=== {name} ===")
    for kind in kinds:
        dated, n_dated = F.count_missing(run, MAKERS[(kind, "dated")](), kind, F.DAYS)
        numeric, n_num = F.count_missing(run, MAKERS[(kind, "numeric")](), kind, F.DAYS)
        print(f"[{kind}] MissingFixingWarning count over the run: dated {n_dated}, numeric {n_num}")
        print(f"{'day':>4} {'spot':>6} {'run dated':>12} {'by hand':>12} {'run-hand':>12} {'run numeric':>12}")
        for k in SHOW:
            ref = reference(run, kind, k)
            print(f"{k:>4} {F.spot_on(k):>6.1f} {dated[k]:>12.6f} {ref:>12.6f} {dated[k] - ref:>12.6f} "
                  f"{numeric[k]:>12.6f}")


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    which = sys.argv[1:] or ["backtest", "dynamic", "replay"]
    if "backtest" in which:
        report("E1 equity BacktestEngine (daily calendar path, value = position MTM)", backtest, ["ra", "asian"])
    if "dynamic" in which:
        report("E2 equity DynamicScenarioEngine (dated DayPath)", dynamic, ["ra", "asian"])
    if "replay" in which:
        report("E3 ReplayBacktestEngine (analytical route, product_mtm)", replay, ["asian"])
