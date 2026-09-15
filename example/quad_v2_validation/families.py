"""Independent exact-GBM RQMC checks for the newly compiled product families.

Contractual schedules/payoff amounts come from product APIs. Path transitions,
coupon memory, termination and bridge sampling are independent of V2's compiler.
Intervals quantify sampling error; they do not certify the tighter deterministic
PV/Greek targets by themselves.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import math

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc, t as student_t

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from quantark.asset.equity.engine.quad import (
    PhoenixQuadEngineV2,
    KOResetSnowballQuadEngineV2,
)
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.snowball_config import (
    BarrierConfig,
    PayoffConfig,
)
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option import create_ko_reset_snowball
from quantark.asset.equity.param import QuadV2Params
from quantark.param import (
    SpotQuote,
    FlatVolSurface,
    FlatRateCurve,
    ContinuousDividendYield,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import CouponPayType, ObservationType


def environment():
    return PricingEnvironment(
        spot_quote=SpotQuote(100.0),
        vol_surface=FlatVolSurface(0.3),
        rate_curve=FlatRateCurve(0.03),
        div_yield=ContinuousDividendYield(0.01),
        valuation_date=datetime(2026, 9, 11),
    )


def inputs():
    for reverse, memory, expiry, continuous in [
        (False, False, False, False),
        (False, True, False, False),
        (False, True, True, False),
        (True, True, False, False),
        (False, True, True, True),
    ]:
        p = PhoenixOption(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            contract_multiplier=1.0,
            is_reverse=reverse,
            barrier_config=BarrierConfig(
                ko_barrier=97.0 if reverse else 103.0,
                ko_rate=0.03,
                ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
                ki_barrier=120.0 if reverse else 80.0,
                ki_observation_dates=[0.25, 0.5, 0.75, 1.0],
                ki_continuous=continuous,
                ki_observation_type=ObservationType.CONTINUOUS
                if continuous
                else ObservationType.DISCRETE,
            ),
            coupon_config=CouponBarrierConfig(
                coupon_barrier=110.0 if reverse else 90.0,
                coupon_rate=0.08,
                memory_coupon=memory,
                coupon_pay_type=CouponPayType.EXPIRY
                if expiry
                else CouponPayType.INSTANT,
            ),
            payoff_config=PayoffConfig(rebate_rate=0.0, include_principal=True),
        )
        yield f"phoenix_reverse{reverse}_memory{memory}_expiry{expiry}_continuous{continuous}", p, PhoenixQuadEngineV2
    for continuous in (False, True):
        p = create_ko_reset_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity_pre=1.0,
            maturity_post=1.5,
            pre_frequency="quarterly",
            post_frequency="quarterly",
            ki_frequency="quarterly",
            ki_continuous=continuous,
            pre_ko_barrier=103.0,
            post_ko_barrier=95.0,
            ki_barrier=80.0,
            pre_ko_rate=0.12,
            post_ko_rate=0.04,
        )
        yield f"reset_continuous{continuous}", p, KOResetSnowballQuadEngineV2


def monte_carlo(product, env, power=16, replicates=8):
    phoenix = isinstance(product, PhoenixOption)
    if phoenix:
        records = product.resolve_ko_observations(env)
        pre = {
            float(r.observation_time): (
                float(r.barrier),
                float(r.payoff),
                float(r.settlement_time),
                i,
            )
            for i, r in enumerate(records)
        }
        post = pre
        pre_maturity = maturity = product.get_maturity(env)
        fractions = product.get_coupon_period_year_fractions(list(pre))
        coupons = [
            product.get_coupon_payoff(i, year_fraction=f)
            for i, f in enumerate(fractions)
        ]
    else:

        def resolve(config):
            recs, rates, originals = product._resolve_ko_schedule(config, env)
            principal = 100.0 if product.payoff_config.include_principal else 0.0
            return {
                float(r.observation_time): (
                    float(r.barrier),
                    principal
                    + 100
                    * rate
                    * product.compute_ko_accrual_factor(r.observation_time, o, env),
                    float(r.settlement_time),
                    i,
                )
                for i, (r, rate, o) in enumerate(zip(recs, rates, originals))
            }

        pre, post = resolve(product.barrier_config), resolve(
            product.post_barrier_config
        )
        pre_maturity = product.get_pre_maturity_time(env)
        maturity = product.get_maturity(env)
    continuous = product.barrier_config.ki_continuous
    ki = (
        {}
        if continuous
        else {
            float(r.observation_time): float(r.barrier)
            for r in product.resolve_ki_observations(env)
        }
    )
    times = sorted(set(pre) | set(post) | set(ki) | {pre_maturity, maturity})
    samples = []
    for rep in range(replicates):
        u = qmc.Sobol(
            len(times) * (2 if continuous else 1), scramble=True, seed=9173 + rep * 101
        ).random_base2(power)
        z = ndtri(u[:, : len(times)])
        s = np.full(len(u), env.spot)
        active = np.ones(len(u), bool)
        hit = np.zeros(len(u), bool)
        missed = np.zeros(len(u))
        accrued = np.zeros(len(u))
        cash = np.zeros(len(u))
        previous = 0.0
        for j, time in enumerate(times):
            dt = time - previous
            old_s = s.copy()
            previous = time
            s *= np.exp(
                (0.03 - 0.01 - 0.5 * 0.3**2) * dt + 0.3 * math.sqrt(dt) * z[:, j]
            )
            if continuous:
                b = float(product.barrier_config.ki_barrier)
                a = np.log(old_s / b)
                end = np.log(s / b)
                safe = (
                    (a < 0) & (end < 0) if product.is_reverse else (a > 0) & (end > 0)
                )
                prob = np.ones(len(u))
                prob[safe] = np.exp(-2 * a[safe] * end[safe] / (0.3**2 * dt))
                hit |= active & (u[:, len(times) + j] < prob)
            incoming = hit.copy()
            new_ki = (
                np.zeros(len(u), bool)
                if time not in ki
                else (s >= ki[time] if product.is_reverse else s <= ki[time])
            )
            if phoenix:
                rec = pre.get(time)
                hit |= active & new_ki
                if rec is not None:
                    barrier, pay, payment, index = rec
                    ko = (s <= barrier if product.is_reverse else s >= barrier) & active
                    if product.barrier_config.disable_ko_after_ki:
                        ko &= ~hit
                    level = product.get_coupon_barrier_at(index)
                    coupon_hit = (
                        s <= level if product.is_reverse else s >= level
                    ) & active
                    earned = np.where(coupon_hit, coupons[index] + missed, 0.0)
                    ko_coupon = missed + np.where(coupon_hit, coupons[index], 0.0)
                    if product.coupon_config.coupon_pay_type == CouponPayType.EXPIRY:
                        cash[ko] += (pay + ko_coupon[ko] + accrued[ko]) * math.exp(
                            -0.03 * payment
                        )
                        accrued[active & ~ko] += earned[active & ~ko]
                    else:
                        cash[ko] += (pay + ko_coupon[ko]) * math.exp(-0.03 * payment)
                        cash[active & ~ko] += earned[active & ~ko] * math.exp(
                            -0.03 * payment
                        )
                    if product.has_memory_coupon:
                        missed = np.where(coupon_hit, 0.0, missed + coupons[index])
                    active &= ~ko
            else:
                pre_ko = np.zeros(len(u), bool)
                if time in pre:
                    barrier, pay, payment, _ = pre[time]
                    pre_ko = active & ~incoming & (s >= barrier)
                    cash[pre_ko] += pay * math.exp(-0.03 * payment)
                    active &= ~pre_ko
                hit |= active & new_ki
                if time in post:
                    barrier, pay, payment, _ = post[time]
                    post_ko = active & hit & (s >= barrier)
                    cash[post_ko] += pay * math.exp(-0.03 * payment)
                    active &= ~post_ko
            terminate = active & (
                ((time == pre_maturity) & ~hit) | ((time == maturity) & hit)
            )
            # Vectorize the known affine downside; the alive callback supplies
            # the fixed contractual rebate/principal at this state maturity.
            principal = 100.0 if product.payoff_config.include_principal else 0.0
            downside = (
                np.maximum(s - 100.0, 0.0)
                if product.is_reverse
                else np.maximum(100.0 - s, 0.0)
            )
            terminal = np.where(
                hit,
                principal - downside,
                product.get_maturity_payoff_v0(100.0, pricing_env=env),
            )
            if (
                phoenix
                and product.coupon_config.coupon_pay_type == CouponPayType.EXPIRY
            ):
                terminal += accrued
            cash[terminate] += terminal[terminate] * math.exp(-0.03 * time)
            active &= ~terminate
        assert not active.any()
        samples.append(float(cash.mean()))
    mean = float(np.mean(samples))
    se = float(np.std(samples, ddof=1) / math.sqrt(replicates))
    half = float(student_t.ppf(0.975, replicates - 1) * se)
    return dict(
        mean=mean,
        half95=half,
        lo95=mean - half,
        hi95=mean + half,
        paths=replicates * 2**power,
        replicates=samples,
    )


def main():
    rows = []
    sources = [Path(__file__)] + sorted(
        (ROOT / "quantark/asset/equity/engine/quad/v2").glob("*.py")
    )
    sources.append(ROOT / "quantark/asset/equity/param/quad_v2_params.py")
    metadata = dict(
        date=datetime.now().astimezone().isoformat(),
        python=sys.version,
        numpy=np.__version__,
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        scope="Independent path transitions using QuantArk contractual schedule and payoff amounts; intervals are across eight independent scrambles.",
    )
    for name, p, cls in inputs():
        env = environment()
        mc = monte_carlo(p, env)
        standard = cls().calculate_point_greeks(p, env)
        fine = cls(QuadV2Params(order=10, cells_per_sd=3.0)).calculate_point_greeks(
            p, env
        )
        row = dict(
            name=name,
            mc=mc,
            standard=standard,
            fine=fine,
            inside95=mc["lo95"] <= fine["price"] <= mc["hi95"],
            refinement_gap=fine["price"] - standard["price"],
        )
        rows.append(row)
        print(name, row["inside95"], row["refinement_gap"], flush=True)
        Path(__file__).with_name("family_results.json").write_text(
            json.dumps(dict(metadata=metadata, results=rows), indent=2) + "\n"
        )
    assert all(r["inside95"] for r in rows), rows


if __name__ == "__main__":
    main()
