"""
Implied futures carry demo: futures marks -> implied q(T) -> bucket deltas ->
hedge hands -> post-hedge diagnostics. Deterministic synthetic data.

Run: python example/equity_futures_delta_buckets_demo.py
"""
from datetime import datetime

from quantark.asset.equity.engine.analytical.deltaone_engine import DeltaOneEngine
from quantark.asset.equity.engine.quad import SnowballQuadEngine
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


def main() -> None:
    spot = 5000.0
    # 1. index futures curve (000905 / IC contracts, synthetic marks)
    curve = IndexFuturesCurve(
        underlying="IC",
        spot=spot,
        quotes=[
            IndexFuturesQuote("IC00", maturity=0.03, price=5008.0, multiplier=200.0),
            IndexFuturesQuote("IC01", maturity=0.10, price=5020.0, multiplier=200.0),
            IndexFuturesQuote("IC02", maturity=0.18, price=5036.0, multiplier=200.0),
            IndexFuturesQuote("IC03", maturity=0.32, price=5064.0, multiplier=200.0),
        ],
    )

    env = PricingEnvironment(
        rate_curve=FlatRateCurve(0.03),
        valuation_date=datetime(2026, 7, 3),
        spot_quote=SpotQuote(spot),
        vol_surface=FlatVolSurface(0.20),
    )
    # 2./3. implied dividend curve attached to the environment
    implied_div = curve.to_dividend_yield_curve(env.rate_curve)
    env.div_yield = implied_div
    print("implied q(T) nodes:")
    for t, y in zip(implied_div.times, implied_div.yields):
        print(f"  T={t:5.2f}  q={y:+.4%}")

    # 4. price a 000905 snowball on the implied-carry environment
    # (barriers are absolute levels: KO 103%, KI 75% of initial)
    snowball = SnowballOption(
        initial_price=spot,
        strike=spot,
        barrier_config=BarrierConfig(
            ko_barrier=1.03 * spot,
            ko_rate=0.15,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=0.75 * spot,
            ki_observation_type=ObservationType.CONTINUOUS,
        ),
        payoff_config=None,
        contract_multiplier=1.0,
        maturity=1.0,
        is_reverse=False,
    )
    engine = SnowballQuadEngine()
    calc = GreeksCalculator()
    pv = engine.price(snowball, env)
    print(f"\nsnowball PV (implied carry): {pv:.6f}")

    # 5. futures-tenor delta buckets -> hedge hands
    delta_rows = calc.calculate_futures_delta_buckets(snowball, env, engine, curve)
    print("\ncontract  F        dPV/dF_i     per-hand  hedge hands   extrapolated")
    print("          (hedge_hands_i = -delta_bucket_i / multiplier_i)")
    for r in delta_rows:
        print(
            f"{r['contract']}   {r['future_price']:8.1f} {r['delta_bucket']:+11.6f}"
            f"  {r['delta_per_hand']:8.1f}  {r['hedge_hands']:+11.6f}"
            f"   {r['extrapolated_tail']}"
        )

    # 6. bucketed rhoq diagnostics
    rhoq_rows = calc.calculate_futures_rhoq_buckets(snowball, env, engine, curve)
    print("\ncontract  rhoq_bucket (per +1% carry)")
    for r in rhoq_rows:
        print(f"{r['contract']}   {r['rhoq_bucket']:+.6f}")

    # 7./8. option + futures hedge portfolio: post-hedge diagnostics
    deltaone = DeltaOneEngine()
    print("\ncontract  net delta bucket  option rhoq  hedge rhoq   net rhoq")
    hedge_rows = []
    for rd in delta_rows:
        hands = rd["hedge_hands"]
        fut = Futures(
            underlying="IC",
            multiplier=1.0,
            maturity=rd["maturity"],
        )
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
    print(
        "\nnote: snowball maturity 1.0 > last futures node 0.32 — the IC03 "
        "bucket includes flat-extrapolated tail carry (extrapolated_tail=True)."
    )


if __name__ == "__main__":
    main()
