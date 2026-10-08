"""Item 10: DCN, FX TARN and cashleg keep their own accrual code. Is each right for an aged contract?

1. DCN: the coupon / KO accrual of every surviving observation, built at inception (2023-01-03) and again aged
   (valuation 2023-07-03), on the test fixture DCN_A (30/360 per month, lock 3 months).
2. FX TARN: twelve monthly fixings at k/12. Aged by ``time_shift(0.125)`` (the theta path) versus the same contract
   rebuilt at the aged date with its remaining fixing times (k/12 - 0.125) and default accrual factors.
3. cashleg AccrualLeg: the same leg with ``day_count`` ACT/365 and ACT/360: the field is not read.

Run from the repo root:  env PYTHONPATH=$PWD:$PWD/test <venv>/bin/python <this file>
"""
import copy
from datetime import datetime

import numpy as np

from dcn_fixtures import DCN_A, make_dcn
from quantark.asset.equity.product.option.dcn_grid import build_dcn_grid_context
from quantark.asset.fx.product.option.fx_target_redemption_note import FxTargetRedemptionNote
from quantark.cashleg import AccrualLeg, BaseAmount, BaseAmountMode, LegSchedule
from quantark.cashleg.accrual_leg import KOBehavior, PaymentConvention, SurvivalBasis
from quantark.cashleg.base import LegDirection
from quantark.cashleg.event_distribution import EventDistribution
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar.day_counter import DayCountConvention


def dcn():
    print("1. DCN_A accruals per surviving observation (coupon, KO)")
    fresh = build_dcn_grid_context(make_dcn(DCN_A))
    aged_valuation = datetime(2023, 7, 3)
    aged = build_dcn_grid_context(make_dcn(DCN_A, valuation_date=aged_valuation))
    p_fresh, p_aged = make_dcn(DCN_A), make_dcn(DCN_A, valuation_date=aged_valuation)
    obs_f = {m.observation_date: (c, k) for m, c, k in zip(p_fresh.schedule.monthly, fresh.coupon_accruals,
                                                            fresh.ko_accruals)}
    rows = [(m.observation_date, c, k, obs_f[m.observation_date]) for m, c, k in
            zip(p_aged.schedule.monthly, aged.coupon_accruals, aged.ko_accruals)]
    moved = [r for r in rows if (r[1], r[2]) != r[3]]
    print(f"  observations at inception {len(fresh.coupon_accruals)}, aged {len(rows)}; "
          f"accrual per period {p_fresh.accrual_per_period:.10f}; surviving rows that differ: {len(moved)}")
    for d, c, k, (cf, kf) in rows[:3]:
        print(f"  {d:%Y-%m-%d} aged ({c:.6f}, {k:.6f}) inception ({cf:.6f}, {kf:.6f})")


def tarn():
    print("2. FX TARN accrual factors of the remaining fixings")
    fix = [k / 12 for k in range(1, 13)]
    note = FxTargetRedemptionNote(fixing_times=fix, coupon_rate=0.12, strike=1.0, notional=100.0)
    shifted = copy.deepcopy(note)
    shifted.time_shift(0.125)
    keep = [i for i, t in enumerate(shifted.fixing_times) if t > 0]
    rebuilt = FxTargetRedemptionNote(fixing_times=[shifted.fixing_times[i] for i in keep], coupon_rate=0.12,
                                     strike=1.0, notional=100.0)
    print(f"  inception      first 3: {np.round(note.accrual_factors[:3], 6).tolist()}")
    print(f"  time_shift     fixings {np.round(shifted.fixing_times[:3], 6).tolist()} "
          f"accruals {np.round(shifted.accrual_factors[:3], 6).tolist()}")
    print(f"  rebuilt aged   fixings {np.round(rebuilt.fixing_times[:2], 6).tolist()} "
          f"accruals {np.round(rebuilt.accrual_factors[:2], 6).tolist()}  "
          f"(contractual {round(1 / 12, 6)}; first coupon {12 * rebuilt.accrual_factors[0]:.4f} "
          f"vs {12 * note.accrual_factors[1]:.4f} on notional 100 at 12%)")
    print(f"  time_shift keeps the past fixing (t={shifted.fixing_times[0]:.6f}) in the list: "
          f"{len(shifted.fixing_times)} fixings")


def cashleg():
    print("3. cashleg AccrualLeg, quarterly 1y, 4% on 1,000,000, no KO")
    env = PricingEnvironment(spot_quote=SpotQuote(spot=100.0), vol_surface=FlatVolSurface(volatility=0.2),
                             rate_curve=FlatRateCurve(rate=0.05), div_yield=ContinuousDividendYield(div_yield=0.0),
                             valuation_date=datetime(2026, 1, 1))
    sched = LegSchedule(period_starts=np.array([0.0, 0.25, 0.5, 0.75]), period_ends=np.array([0.25, 0.5, 0.75, 1.0]),
                        payment_times=np.array([0.25, 0.5, 0.75, 1.0]))
    for dc in (DayCountConvention.ACT_365, DayCountConvention.ACT_360, DayCountConvention.THIRTY_360_US):
        leg = AccrualLeg(rate=0.04, base=BaseAmount(value=1_000_000.0, mode=BaseAmountMode.ABSOLUTE), schedule=sched,
                         day_count=dc, payment_convention=PaymentConvention.AT_PERIOD_END,
                         ko_behavior=KOBehavior.TRUNCATE_AT_KO, survival_basis=SurvivalBasis.ENTER_PERIOD,
                         direction=LegDirection.BUYER_RECEIVES)
        pv = leg.value(EventDistribution.trivial(maturity=1.0), env, 0.0)
        print(f"  day_count={dc.name:<15} PV={pv:.6f}")


if __name__ == "__main__":
    dcn()
    tarn()
    cashleg()
