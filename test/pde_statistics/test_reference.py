"""The exact reference (spec 3B §4.1, §5.7): reproduces its banked values, its errors are what it claims,
and it refuses what it does not implement."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from golden_compare import assert_close
from pde_statistics.cases import CASES, REFERENCE_CASES, REFERENCE_KWARGS
from pde_statistics.reference import exact_event_stats, exact_with_error

BANK = json.loads((Path(__file__).parent / "reference_values.json").read_text())["cases"]


@pytest.mark.parametrize("name", [pytest.param(n, marks=pytest.mark.slow) if n == "sb_daily" else n
                                  for n in REFERENCE_CASES])   # sb_daily: ~75 s (252 knock-in dates)
def test_reference_reproduces_its_bank(name):
    product, env = CASES[name][0](), CASES[name][1]()
    got = exact_with_error(product, env, **REFERENCE_KWARGS.get(name, {}))
    want = BANK[name]["values"]
    for key in ("ko_probability", "expected_discounted_ko_cashflow", "ki_ever_probability",
                "ki_survive_knocked_in_probability", "expected_discounted_maturity_cashflow", "pv"):
        if key in want:
            assert_close(np.asarray(got[key], float).tolist() if np.ndim(got[key]) else float(got[key]),
                         want[key], msg=f"{name}.{key}")
    assert max(got["quadrature_error"].values()) <= 1e-11
    assert abs(got["mass_lost"]) <= 1e-10 and abs(got["kie_mass_lost"]) <= 1e-10


def test_far_spot_knocks_out_on_the_first_date():
    product, _ = CASES["sb_cont"][0](), None
    from pde_3a.products import env
    out = exact_event_stats(product, env(spot=150.0, vol=0.01))
    assert abs(out["ko_probability"][0] - 1.0) <= 1e-12


def test_reverse_and_valuation_date_products_are_refused():
    from quantark.modelvalidation.builders.equity_snowball import make_snowball
    from pde_3a.products import SNOWBALL, env
    with pytest.raises(AssertionError, match="reverse"):
        exact_event_stats(make_snowball(dict(SNOWBALL, is_reverse=True, ko_barrier=97.0, ki_barrier=115.0)), env())
    with pytest.raises(AssertionError, match="memory"):
        exact_event_stats(CASES["ph_memory"][0](), CASES["ph_memory"][1]())


def test_an_observation_at_the_valuation_date_is_refused():
    from datetime import datetime

    from intraday.conftest import dated_snowball, flat_env
    from quantark.util.calendar import CalendarType, create_calendar
    p = dated_snowball(create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028)), datetime(2026, 3, 16))
    first = p.barrier_config.ko_observation_schedule.records[0].observation_date
    with pytest.raises(AssertionError, match="an observation at the valuation date is not implemented"):
        exact_event_stats(p, flat_env(first))


def test_coupon_cash_follows_the_records_settlement():
    """Evidence p12: lag every settlement by ten business days; each coupon's cash is the unlagged cash
    rediscounted from the observation to the lagged payment, DF(pay_lag) / DF(pay)."""
    from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit
    product, env = CASES["ph_cont"][0](), CASES["ph_cont"][1]()
    lagged = copy.deepcopy(product)
    names = [a for a in vars(lagged) if "settlement_convention" in a]
    assert names
    for a in names:
        setattr(lagged, a, SettlementConvention(lag=10.0 / 252.0, lag_unit=SettlementLagUnit.YEAR_FRACTION))
    pay = np.array([r.settlement_time for r in product.resolve_ko_observations(env)])
    pay_lag = np.array([r.settlement_time for r in lagged.resolve_ko_observations(env)])
    assert np.all(pay_lag > pay)
    base = np.asarray(exact_event_stats(product, env)["expected_discounted_coupon_cashflow"], float)
    got = np.asarray(exact_event_stats(lagged, env)["expected_discounted_coupon_cashflow"], float)
    df_ratio = np.array([env.rate_curve.get_discount_factor(t) / env.rate_curve.get_discount_factor(s)
                         for t, s in zip(pay_lag, pay)])
    np.testing.assert_allclose(got, base * df_ratio, rtol=0.0, atol=1e-12)
