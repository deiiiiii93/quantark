"""
Implied futures carry demo: futures marks -> implied q(T) -> bucket deltas ->
hedge hands -> post-hedge diagnostics. Deterministic synthetic data.

Two scenarios:
  1. Contango marks (negative implied q), quarterly KO observations —
     risk concentrates in the last bucket (flat-extrapolated tail).
  2. Backwardation marks (positive implied q, the typical IC discount),
     monthly KO observations, futures tenors spanning the observation
     range — risk spreads across all tenors.

Each scenario ends with a cross-engine check: the same buckets computed on
the QUAD, PDE, and MC snowball engines. The MC legs are seeded (common
random numbers) and use larger bumps (20 futures points, 1% carry) — the
discontinuous KO/KI payoff makes small-bump finite differences noisy even
with a fixed seed.

Run: python example/equity_futures_delta_buckets_demo.py
"""
from datetime import datetime

from quantark.asset.equity.engine.analytical.deltaone_engine import DeltaOneEngine
from quantark.asset.equity.engine.mc import SnowballMCEngine
from quantark.asset.equity.engine.pde_engine import PDEEngine
from quantark.asset.equity.engine.quad import SnowballQuadEngine
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.market import IndexFuturesCurve, IndexFuturesQuote
from quantark.asset.equity.product.deltaone.futures import Futures
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.asset.equity.riskmeasures.greeks_calculator import GreeksCalculator
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.portfolio.equity import (
    aggregate_futures_delta_buckets,
    aggregate_futures_rhoq_buckets,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType


def make_snowball(spot, ko_observation_dates, maturity):
    # barriers are absolute levels: KO 103%, KI 75% of initial
    return SnowballOption(
        initial_price=spot,
        strike=spot,
        barrier_config=BarrierConfig(
            ko_barrier=1.03 * spot,
            ko_rate=0.15,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=list(ko_observation_dates),
            ki_barrier=0.75 * spot,
            ki_observation_type=ObservationType.CONTINUOUS,
        ),
        payoff_config=None,
        contract_multiplier=1.0,
        maturity=maturity,
        is_reverse=False,
    )


def run_scenario(title, spot, quotes, ko_observation_dates, maturity):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")

    curve = IndexFuturesCurve(underlying="IC", spot=spot, quotes=list(quotes))
    env = PricingEnvironment(
        rate_curve=FlatRateCurve(0.03),
        valuation_date=datetime(2026, 7, 3),
        spot_quote=SpotQuote(spot),
        vol_surface=FlatVolSurface(0.20),
    )
    implied_div = curve.to_dividend_yield_curve(env.rate_curve)
    env.div_yield = implied_div
    print("implied q(T) nodes:")
    for t, y in zip(implied_div.times, implied_div.yields):
        print(f"  T={t:5.2f}  q={y:+.4%}")

    snowball = make_snowball(spot, ko_observation_dates, maturity)
    engine = SnowballQuadEngine()
    calc = GreeksCalculator()
    pv = engine.price(snowball, env)
    print(f"\nsnowball PV (implied carry): {pv:.6f}")
    print(
        f"KO observations: {len(ko_observation_dates)} dates in "
        f"[{ko_observation_dates[0]:.3f}, {ko_observation_dates[-1]:.3f}], "
        f"maturity {maturity}"
    )

    delta_rows = calc.calculate_futures_delta_buckets(snowball, env, engine, curve)
    print("\ncontract  F        dPV/dF_i     per-hand  hedge hands   extrapolated")
    print("          (hedge_hands_i = -delta_bucket_i / multiplier_i)")
    for r in delta_rows:
        print(
            f"{r['contract']}   {r['future_price']:8.1f} {r['delta_bucket']:+11.6f}"
            f"  {r['delta_per_hand']:8.1f}  {r['hedge_hands']:+11.6f}"
            f"   {r['extrapolated_tail']}"
        )

    rhoq_rows = calc.calculate_futures_rhoq_buckets(snowball, env, engine, curve)
    print("\ncontract  rhoq_bucket (per +1% carry)")
    for r in rhoq_rows:
        print(f"{r['contract']}   {r['rhoq_bucket']:+.6f}")

    deltaone = DeltaOneEngine()
    print("\ncontract  net delta bucket  option rhoq  hedge rhoq   net rhoq")
    hedge_rows = []
    for rd in delta_rows:
        hands = rd["hedge_hands"]
        fut = Futures(underlying="IC", multiplier=1.0, maturity=rd["maturity"])
        per_hand_rhoq = (
            deltaone.calculate_greeks(fut, env)["dividend_rho"]
            * curve.get_quote(rd["contract"]).multiplier
        )
        hedge_rows.append(
            {
                "contract": rd["contract"],
                "maturity": rd["maturity"],
                "future_price": rd["future_price"],
                "delta_per_hand": rd["delta_per_hand"],
                "delta_bucket": hands * rd["delta_per_hand"],
                "hedge_hands": 0.0,
                "rhoq_bucket": hands * per_hand_rhoq,
            }
        )
    net_delta = aggregate_futures_delta_buckets(
        {"snowball": delta_rows, "hedge": hedge_rows}
    )
    net_rhoq = aggregate_futures_rhoq_buckets(
        {"snowball": rhoq_rows, "hedge": hedge_rows}
    )
    for nd, nq, rq in zip(net_delta, net_rhoq, rhoq_rows):
        hedge_rhoq = nq["rhoq_bucket"] - rq["rhoq_bucket"]
        print(
            f"{nd['contract']}   {nd['delta_bucket']:+16.10f}"
            f"  {rq['rhoq_bucket']:+11.6f}  {hedge_rhoq:+11.6f}"
            f"  {nq['rhoq_bucket']:+11.6f}"
        )
    if maturity > quotes[-1].maturity:
        print(
            f"\nnote: snowball maturity {maturity} > last futures node "
            f"{quotes[-1].maturity} — the {quotes[-1].contract} bucket includes "
            "flat-extrapolated tail carry (extrapolated_tail=True)."
        )

    # cross-engine check: the same buckets on all three engine families.
    # MC rhoq uses a 50bp bump — the per-1% scaling (x 0.01/div_bump) would
    # otherwise amplify barrier-flip FD noise on the discontinuous payoff,
    # even with common random numbers.
    print("\ncross-engine check (same product, same implied curve):")
    contracts = "  ".join(f"{q.contract:>10s}" for q in curve.quotes)
    print(f"  {'engine':6s} {'greek':5s} {contracts}")
    families = [
        ("QUAD", SnowballQuadEngine()),
        ("PDE", PDEEngine()),
        ("MC", SnowballMCEngine(MCParams(seed=42, num_paths=100_000))),
    ]
    for name, family_engine in families:
        # MC needs stronger finite-difference signals than the deterministic
        # engines: a 1-point bump on F~5000 is only a few bp of carry, which
        # drowns in barrier-flip noise even with a fixed seed. 20 points is
        # still just 0.4% of the mark.
        bump = 20.0 if name == "MC" else 1.0
        d = (
            delta_rows
            if name == "QUAD"
            else calc.calculate_futures_delta_buckets(
                snowball, env, family_engine, curve, price_bump=bump
            )
        )
        q = calc.calculate_futures_rhoq_buckets(
            snowball, env, family_engine, curve,
            div_bump=0.01 if name == "MC" else 0.005,
        )
        print(
            f"  {name:6s} delta " + "  ".join(f"{r['delta_bucket']:+10.6f}" for r in d)
        )
        print(
            f"  {name:6s} rhoq  " + "  ".join(f"{r['rhoq_bucket']:+10.6f}" for r in q)
        )


def main() -> None:
    spot = 5000.0

    # Scenario 1: contango marks (negative implied q), quarterly KO dates.
    # The first observation (0.25) reads the curve only between the 0.18 and
    # 0.32 nodes, so IC00/IC01 buckets are exactly zero and the risk
    # concentrates in the extrapolated last bucket.
    run_scenario(
        "Scenario 1: contango, quarterly KO observations",
        spot,
        [
            IndexFuturesQuote("IC00", maturity=0.03, price=5008.0, multiplier=200.0),
            IndexFuturesQuote("IC01", maturity=0.10, price=5020.0, multiplier=200.0),
            IndexFuturesQuote("IC02", maturity=0.18, price=5036.0, multiplier=200.0),
            IndexFuturesQuote("IC03", maturity=0.32, price=5064.0, multiplier=200.0),
        ],
        ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
        maturity=1.0,
    )

    # Scenario 2: backwardation marks (IC futures typically trade at a
    # discount => positive implied q), monthly KO dates, and futures tenors
    # spanning the observation range (当月/次月/当季/次季) so every bucket
    # receives a genuine multi-tenor allocation.
    run_scenario(
        "Scenario 2: backwardation, monthly KO observations",
        spot,
        [
            IndexFuturesQuote("IC00", maturity=0.08, price=4985.0, multiplier=200.0),
            IndexFuturesQuote("IC01", maturity=0.17, price=4955.0, multiplier=200.0),
            IndexFuturesQuote("IC02", maturity=0.42, price=4900.0, multiplier=200.0),
            IndexFuturesQuote("IC03", maturity=0.67, price=4835.0, multiplier=200.0),
        ],
        ko_observation_dates=[round(i / 12.0, 6) for i in range(1, 13)],
        maturity=1.0,
    )


if __name__ == "__main__":
    main()
