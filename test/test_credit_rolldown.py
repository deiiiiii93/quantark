"""
Tests for seasoned ("as-of") CDS pricing and roll-down / carry.

* A dated CDS prices identically to the legacy tenor-based CDS at inception.
* As the valuation date advances the dated CDS rolls down (remaining maturity
  shrinks); a 2y-seasoned 5y trade matches a fresh 3y trade on flat curves.
* A fully matured CDS prices to zero.
* ``cds_coupon_schedule_asof`` drops settled coupons, re-expresses payment times
  relative to the as-of date, and keeps full contractual accrual fractions.
* The dynamic-scenario / backtest cash ledger books settled coupons so reported
  P&L is a carry-preserving total return (no jump on coupon dates).
"""
from datetime import datetime, timedelta

import pytest

from quantark.asset.credit.engine.analytical import CDSReducedFormEngine
from quantark.asset.credit.engine.schedule import (
    cds_coupon_schedule,
    cds_coupon_schedule_asof,
    contractual_coupon_dates,
)
from quantark.asset.credit.product import CDS, ProtectionSide
from quantark.dynamicscenario import CreditDynamicScenarioEngine
from quantark.dynamicscenario.path.day_path import DayPath, DayStep
from quantark.param import FlatRateCurve
from quantark.param.credit import FlatHazardCurve
from quantark.portfolio import CreditPortfolio
from quantark.priceenv import CreditPricingEnvironment
from quantark.util.exceptions import ValidationError

_EFF = datetime(2026, 6, 13)


def _env(date=_EFF, rate=0.03, hazard=0.02):
    return CreditPricingEnvironment(
        valuation_date=date,
        discount_curve=FlatRateCurve(rate=rate),
        hazard_curve=FlatHazardCurve(hazard_rate=hazard),
    )


# --------------------------------------------------------------------------- #
# Product / calendar
# --------------------------------------------------------------------------- #
def test_maturity_date_derived_from_effective_and_tenor():
    cds = CDS(notional=1e7, maturity=5.0, recovery_rate=0.4, effective_date=_EFF)
    assert cds.is_dated
    assert cds.maturity_date == _EFF + timedelta(days=5.0 * 365.0)


def test_undated_cds_is_not_dated():
    cds = CDS(notional=1e7, maturity=5.0, recovery_rate=0.4)
    assert not cds.is_dated
    assert cds.maturity_date is None


def test_maturity_date_without_effective_date_rejected():
    with pytest.raises(ValidationError):
        CDS(notional=1e7, maturity=5.0, recovery_rate=0.4,
            maturity_date=_EFF + timedelta(days=1000))


def test_maturity_date_before_effective_date_rejected():
    with pytest.raises(ValidationError):
        CDS(notional=1e7, maturity=5.0, recovery_rate=0.4,
            effective_date=_EFF, maturity_date=_EFF - timedelta(days=1))


# --------------------------------------------------------------------------- #
# As-of pricing
# --------------------------------------------------------------------------- #
def test_dated_cds_matches_legacy_at_inception():
    eng = CDSReducedFormEngine()
    legacy = CDS(notional=1e7, maturity=5.0, recovery_rate=0.4, coupon_spread=0.01)
    dated = CDS(notional=1e7, maturity=5.0, recovery_rate=0.4, coupon_spread=0.01,
                effective_date=_EFF)
    assert eng.price(dated, _env(_EFF)) == pytest.approx(eng.price(legacy, _env(_EFF)))


def test_dated_cds_rolls_down_to_remaining_tenor():
    eng = CDSReducedFormEngine()
    dated = CDS(notional=1e7, maturity=5.0, recovery_rate=0.4, coupon_spread=0.01,
                effective_date=_EFF)
    # 2 years (a whole number of quarterly coupons) into a 5y trade equals a
    # fresh 3y trade on flat curves: same remaining life, aligned coupon phase.
    as_of = _EFF + timedelta(days=2 * 365)
    fresh3 = CDS(notional=1e7, maturity=3.0, recovery_rate=0.4, coupon_spread=0.01)
    assert eng.price(dated, _env(as_of)) == pytest.approx(
        eng.price(fresh3, _env(as_of)), rel=1e-9
    )


def test_dated_cds_pv_decays_with_seasoning():
    eng = CDSReducedFormEngine()
    dated = CDS(notional=1e7, maturity=5.0, recovery_rate=0.4, coupon_spread=0.01,
                effective_date=_EFF, side=ProtectionSide.BUY)
    pv0 = eng.price(dated, _env(_EFF))
    pv3 = eng.price(dated, _env(_EFF + timedelta(days=3 * 365)))
    assert 0 < pv3 < pv0  # less remaining protection -> smaller positive PV


def test_matured_cds_prices_to_zero():
    eng = CDSReducedFormEngine()
    dated = CDS(notional=1e7, maturity=5.0, recovery_rate=0.4, coupon_spread=0.01,
                effective_date=_EFF)
    after = _EFF + timedelta(days=5 * 365 + 30)
    assert eng.price(dated, _env(after)) == 0.0


def test_forward_starting_cds_discounts_lead_period():
    # Valued one year before protection begins. On flat curves a forward-starting
    # CDS scales the spot value by exp(-(r + lambda) * lead): protection and
    # premium only run over [effective, maturity], never the lead period. (The
    # earlier clamp-to-zero bug priced it identically to a spot-start trade.)
    import math

    r, lam, lead = 0.03, 0.02, 1.0
    eng = CDSReducedFormEngine()
    spot = CDS(notional=1e7, maturity=5.0, recovery_rate=0.4, coupon_spread=0.01)
    fwd = CDS(notional=1e7, maturity=5.0, recovery_rate=0.4, coupon_spread=0.01,
              effective_date=_EFF + timedelta(days=int(lead * 365)))
    pv_spot = eng.price(spot, _env(_EFF, rate=r, hazard=lam))
    pv_fwd = eng.price(fwd, _env(_EFF, rate=r, hazard=lam))
    assert pv_fwd == pytest.approx(pv_spot * math.exp(-(r + lam) * lead), rel=1e-6)
    assert abs(pv_fwd - pv_spot) > 1000  # genuinely different from a spot trade


# --------------------------------------------------------------------------- #
# Schedule slicing
# --------------------------------------------------------------------------- #
def test_asof_schedule_drops_settled_coupons():
    mat = _EFF + timedelta(days=5 * 365)
    full = cds_coupon_schedule(5.0, 4)            # 20 quarterly coupons
    as_of = _EFF + timedelta(days=2 * 365)        # 2y in -> 8 coupons settled
    remaining = cds_coupon_schedule_asof(_EFF, mat, 4, as_of)
    assert len(remaining) == len(full) - 8
    # Payment times are measured from the as-of date and stay positive.
    assert all(t > 0 for t, _ in remaining)
    # First remaining coupon pays ~one quarter out, with a full-quarter accrual.
    first_t, first_accr = remaining[0]
    assert first_t == pytest.approx(0.25, abs=0.01)
    assert first_accr == pytest.approx(0.25, abs=1e-9)


def test_asof_schedule_full_at_inception_matches_legacy():
    mat = _EFF + timedelta(days=5 * 365)
    legacy = cds_coupon_schedule(5.0, 4)
    asof = cds_coupon_schedule_asof(_EFF, mat, 4, _EFF)
    assert len(asof) == len(legacy)
    for (t1, a1), (t2, a2) in zip(asof, legacy):
        assert t1 == pytest.approx(t2, abs=1e-6)
        assert a1 == pytest.approx(a2, abs=1e-9)


def test_contractual_coupon_dates_span_full_life():
    mat = _EFF + timedelta(days=5 * 365)
    dates = contractual_coupon_dates(_EFF, mat, 4)
    assert len(dates) == 20
    assert all(pd > _EFF for pd, _ in dates)
    # The final coupon falls on (within a day of) the contractual maturity date.
    assert abs((dates[-1][0] - mat).total_seconds()) < 86_400


# --------------------------------------------------------------------------- #
# Total-return cash ledger (integration)
# --------------------------------------------------------------------------- #
def _dated_portfolio(side=ProtectionSide.BUY):
    pf = CreditPortfolio(portfolio_name="cr", pricing_environments={"ACME": _env(_EFF)})
    pf.add_position(
        product=CDS(notional=10_000_000, maturity=5.0, recovery_rate=0.4,
                    coupon_spread=0.01, side=side, effective_date=_EFF),
        quantity=1.0, entry_price=0.0, reference_entity="ACME",
        engine=CDSReducedFormEngine(),
    )
    return pf


def test_total_return_ledger_absorbs_coupon_jump():
    # Flat (no-shock) 120-day walk crossing the first quarterly coupon (~day 91).
    # The settled coupon ($10M * 100bp * 0.25 = $25k) leaves the PV but is booked
    # as cash, so reported total-return daily P&L stays smooth (pure carry) with
    # no ~$25k spike on the coupon date.
    path = DayPath(name="flat", steps=[DayStep(day_index=i) for i in range(120)],
                   start_date=_EFF)
    res = CreditDynamicScenarioEngine().run(_dated_portfolio(), path)
    raw_coupon = 10_000_000 * 0.01 * 0.25
    max_daily = max(abs(d.daily_pnl) for d in res.day_results)
    assert max_daily < raw_coupon / 10  # no coupon-sized discontinuity
    # Sum of daily P&L reconciles with the cumulative total return.
    assert sum(d.daily_pnl for d in res.day_results) == pytest.approx(
        res.total_pnl, rel=1e-6
    )


def test_realized_coupon_cash_sign_and_size():
    pos = next(iter(_dated_portfolio(side=ProtectionSide.BUY).positions.values()))
    # Window spanning exactly the first quarterly coupon date.
    start = _EFF + timedelta(days=80)
    end = _EFF + timedelta(days=100)
    cash = CreditDynamicScenarioEngine._realized_coupon_cash(pos, start, end)
    # Protection buyer pays the coupon -> negative cash of ~one quarter's premium.
    assert cash == pytest.approx(-10_000_000 * 0.01 * 0.25, rel=1e-6)


def test_realized_coupon_cash_zero_for_undated():
    pf = CreditPortfolio(portfolio_name="cr", pricing_environments={"ACME": _env(_EFF)})
    pf.add_position(
        product=CDS(notional=1e7, maturity=5.0, recovery_rate=0.4, coupon_spread=0.01),
        quantity=1.0, entry_price=0.0, reference_entity="ACME",
        engine=CDSReducedFormEngine(),
    )
    pos = next(iter(pf.positions.values()))
    cash = CreditDynamicScenarioEngine._realized_coupon_cash(
        pos, _EFF, _EFF + timedelta(days=400)
    )
    assert cash == 0.0
