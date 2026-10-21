"""Final coupon on conversion at maturity: KEEP (default) pays parity + coupon, FORFEIT pays max(parity, face + coupon)."""
import dataclasses
import math
from datetime import datetime

import pytest

from quantark.asset.bond.engine.pde.convertible import (
    ConvertibleBondJumpDiffusionEngine,
    ConvertibleBondPDEParams,
    ConvertibleBondTFEngine,
)
from quantark.asset.bond.engine.tree.convertible import (
    ConvertibleBondBinomialEngine,
    ConvertibleBondTFTreeEngine,
    ConvertibleBondTreeParams,
    ConvertibleBondTrinomialEngine,
)
from quantark.asset.bond.product.convertible import ConvertibleBond
from quantark.param.quote import SpotQuote
from quantark.param.rrf import FlatRateCurve
from quantark.param.vol import FlatVolSurface
from quantark.priceenv import PricingEnvironment
from quantark.asset.bond.engine.tree.convertible.tree_params import ConvertibleBondTrinomialVolScheme
from quantark.asset.bond.product.convertible import PutScheduleEntry
from quantark.util.enum import FinalCouponOnConversion
from quantark.util.exceptions import ValidationError

VALUATION = datetime(2024, 6, 1)
MATURITY = datetime(2029, 1, 1)
RATE, VOL = 0.05, 0.30


def _bond(final_coupon=FinalCouponOnConversion.KEEP):
    return ConvertibleBond(
        issue_date=datetime(2024, 1, 1),
        maturity_date=MATURITY,
        face_value=100.0,
        coupon_rate=0.025,
        conversion_ratio=10.0,
        credit_spread=0.02,
        hazard_rate=0.01,
        recovery_rate=0.4,
        final_coupon_on_conversion=final_coupon,
    )


def _env(spot):
    return PricingEnvironment(
        valuation_date=VALUATION,
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=VOL),
        rate_curve=FlatRateCurve(rate=RATE),
    )


def _final_coupon(bond):
    """Coupon part of the cashflow paid at maturity, read off the schedule."""
    last = [cf for cf in bond.get_all_cashflows() if cf.payment_date >= bond.maturity_date]
    return sum(cf.amount for cf in last) - bond.face_value


ENGINES = {
    "gs": lambda env: ConvertibleBondBinomialEngine(env, ConvertibleBondTreeParams(num_steps=200)),
    "trinomial": lambda env: ConvertibleBondTrinomialEngine(env, ConvertibleBondTreeParams(num_steps=200)),
    "tf-tree": lambda env: ConvertibleBondTFTreeEngine(env, ConvertibleBondTreeParams(num_steps=500)),
    "tf-pde": lambda env: ConvertibleBondTFEngine(env, ConvertibleBondPDEParams(num_space_steps=200, num_time_steps=400)),
    "jump-diffusion": lambda env: ConvertibleBondJumpDiffusionEngine(
        env, ConvertibleBondPDEParams(num_space_steps=200, num_time_steps=400)
    ),
}


# --------------------------------------------------------------------------- product


def test_default_keeps_the_final_coupon():
    assert ConvertibleBond(
        issue_date=datetime(2024, 1, 1), maturity_date=MATURITY, face_value=100.0, coupon_rate=0.025, conversion_ratio=10.0
    ).final_coupon_on_conversion is FinalCouponOnConversion.KEEP


def test_maturity_redemption_is_face_when_keeping_and_face_plus_coupon_when_forfeiting():
    keep, forfeit = _bond(), _bond(FinalCouponOnConversion.FORFEIT)
    assert _final_coupon(keep) > 1.0
    assert keep.maturity_redemption == keep.face_value
    assert forfeit.maturity_redemption == pytest.approx(forfeit.face_value + _final_coupon(forfeit), abs=1e-12)


@pytest.mark.parametrize("text, member", [("keep", FinalCouponOnConversion.KEEP), ("FORFEIT", FinalCouponOnConversion.FORFEIT)])
def test_setting_given_as_text_becomes_the_enum(text, member):
    assert _bond(text).final_coupon_on_conversion is member


@pytest.mark.parametrize("bad", ["forfeited", True, None])
def test_unknown_setting_is_rejected(bad):
    with pytest.raises(ValidationError):
        _bond(bad)


def test_forfeit_moves_the_final_coupon_out_of_the_coupon_stream():
    keep, forfeit = _bond(), _bond(FinalCouponOnConversion.FORFEIT)
    for k_cf, f_cf in zip(keep.get_all_cashflows(), forfeit.get_all_cashflows()):
        if k_cf.payment_date >= keep.maturity_date:
            assert keep.coupon_amount(k_cf) == k_cf.amount - keep.face_value
            assert forfeit.coupon_amount(f_cf) == 0.0
        else:
            assert keep.coupon_amount(k_cf) == k_cf.amount
            assert forfeit.coupon_amount(f_cf) == f_cf.amount


# --------------------------------------------------------------------------- engines


TRINOMIAL_EARLY_CONVERSION = pytest.mark.xfail(
    strict=True,
    reason=(
        "Open item: the constant-vol trinomial converts a riskless q = 0 bond at once (dirty 300.57, P 0.997 "
        "at parity 300, where GS and the TF tree hold to 311.8), so almost no path reaches maturity unconverted"
    ),
)


@pytest.mark.parametrize(
    "engine",
    [pytest.param(name, marks=TRINOMIAL_EARLY_CONVERSION) if name == "trinomial" else name for name in ENGINES],
)
def test_deep_in_the_money_forfeit_costs_one_discounted_coupon_per_conversion(engine):
    # Riskless, so holding to maturity is optimal (with credit risk GS, the hazard engines and the
    # trinomial convert deep in-the-money bonds early, and the final coupon is never at stake).
    # Parity 300 against redemption ~101: forfeiting takes c * exp(-rT) from every path converting at T.
    riskless = {"credit_spread": 0.0, "hazard_rate": 0.0}
    env = _env(30.0)
    keep = ENGINES[engine](env).price(dataclasses.replace(_bond(), **riskless))
    forfeit = ENGINES[engine](env).price(dataclasses.replace(_bond(FinalCouponOnConversion.FORFEIT), **riskless))
    discounted_coupon = _final_coupon(_bond()) * math.exp(-RATE * _bond().time_to_maturity(VALUATION))
    assert 0.85 * discounted_coupon <= keep - forfeit <= 1.01 * discounted_coupon


@pytest.mark.parametrize("engine", list(ENGINES))
def test_far_out_of_the_money_the_setting_does_not_matter(engine):
    # Parity 5: no path converts, the final coupon is paid either way. The trinomial survival-weights
    # the folded coupon over the last step (about c * hazard * dt); everything else agrees to rounding.
    env = _env(0.5)
    keep = ENGINES[engine](env).price(_bond())
    forfeit = ENGINES[engine](env).price(_bond(FinalCouponOnConversion.FORFEIT))
    assert keep == pytest.approx(forfeit, abs=1e-3)


@pytest.mark.parametrize("scheme", list(ConvertibleBondTrinomialVolScheme))
def test_trinomial_forfeit_costs_the_coupon_on_paths_converting_at_maturity(scheme):
    # At the money and riskless the trinomial mostly holds to maturity, so every vol scheme must see
    # the forfeited coupon on the paths that convert there (an engine ignoring FORFEIT shows 0). The
    # bound is loose because the constant-vol scheme converts early on many paths (known issue): 0.09.
    riskless = {"credit_spread": 0.0, "hazard_rate": 0.0}
    engine = ConvertibleBondTrinomialEngine(_env(10.0), ConvertibleBondTreeParams(num_steps=200, trinomial_vol_scheme=scheme))
    keep = engine.price(dataclasses.replace(_bond(), **riskless))
    forfeit = engine.price(dataclasses.replace(_bond(FinalCouponOnConversion.FORFEIT), **riskless))
    discounted_coupon = _final_coupon(_bond()) * math.exp(-RATE * _bond().time_to_maturity(VALUATION))
    assert 0.05 * discounted_coupon <= keep - forfeit <= 1.01 * discounted_coupon


@pytest.mark.parametrize("engine", list(ENGINES))
def test_forfeit_put_in_the_last_step_at_or_below_face_plus_coupon_never_binds(engine):
    # Under FORFEIT the unconverted holder already gets face + coupon (~101.26) at maturity,
    # so a put at 101 two days before is worthless; the price is that of the bond without it.
    put = [PutScheduleEntry(datetime(2028, 12, 30), 101.0)]
    env = _env(0.5)
    with_put = ENGINES[engine](env).price(dataclasses.replace(_bond(FinalCouponOnConversion.FORFEIT), put_schedule=put))
    without = ENGINES[engine](env).price(_bond(FinalCouponOnConversion.FORFEIT))
    assert with_put == without


ENGINE_PATHS = {
    **{name: make for name, make in ENGINES.items() if name != "trinomial"},
    **{
        f"trinomial-{scheme.value}": (
            lambda env, s=scheme: ConvertibleBondTrinomialEngine(
                env, ConvertibleBondTreeParams(num_steps=200, trinomial_vol_scheme=s)
            )
        )
        for scheme in ConvertibleBondTrinomialVolScheme
    },
}


@pytest.mark.parametrize("spot", [10.5, 30.0])
@pytest.mark.parametrize("engine", list(ENGINE_PATHS))
def test_conversion_window_closed_before_maturity_makes_the_setting_irrelevant(engine, spot):
    # Conversion ends seven months before maturity: at T only redemption is possible, so the
    # holder gets face + final coupon under either convention.
    closed = {"conversion_end_date": datetime(2028, 6, 1)}
    env = _env(spot)
    keep = ENGINE_PATHS[engine](env).price(dataclasses.replace(_bond(), **closed))
    forfeit = ENGINE_PATHS[engine](env).price(dataclasses.replace(_bond(FinalCouponOnConversion.FORFEIT), **closed))
    assert keep == pytest.approx(forfeit, abs=1e-3)


@pytest.mark.parametrize(
    "late_payment",
    [
        {  # Saturday maturity, paid Monday
            "issue_date": datetime(2024, 6, 2),
            "maturity_date": datetime(2028, 12, 2),
            "conversion_start_date": datetime(2024, 6, 2),
            "conversion_end_date": datetime(2028, 12, 2),
        },
        {"settlement_days": 2},
    ],
    ids=["weekend-maturity", "settlement-days"],
)
@pytest.mark.parametrize("engine", list(ENGINE_PATHS))
def test_final_coupon_paid_after_maturity_date_is_still_paid(engine, late_payment):
    # The maturity cashflow is paid after maturity_date. Far out of the money nobody converts, so
    # KEEP (coupon in the coupon stream) and FORFEIT (coupon in the redemption) must both pay it.
    bond = dataclasses.replace(_bond(), **late_payment)
    assert max(cf.payment_date for cf in bond.get_all_cashflows()) > bond.maturity_date
    env = _env(0.5)
    keep = ENGINE_PATHS[engine](env).price(bond)
    forfeit = ENGINE_PATHS[engine](env).price(dataclasses.replace(bond, final_coupon_on_conversion=FinalCouponOnConversion.FORFEIT))
    assert keep == pytest.approx(forfeit, abs=1e-3)


def test_tf_tree_agrees_with_tf_pde_when_forfeiting():
    # The AC7 cross-check on demo-simple, under FORFEIT
    env, bond = _env(12.0), _bond(FinalCouponOnConversion.FORFEIT)
    T = bond.time_to_maturity(VALUATION)
    pde = ConvertibleBondTFEngine(
        env,
        ConvertibleBondPDEParams(
            num_space_steps=800, num_time_steps=2000, max_stock_multiple=max(5.0, math.exp(6.0 * VOL * math.sqrt(T)))
        ),
    ).price_with_details(bond)
    tree = ConvertibleBondTFTreeEngine(env, ConvertibleBondTreeParams(num_steps=2000)).price_with_details(bond)
    assert abs(tree.dirty_price - pde.dirty_price) <= 0.05


def test_gs_forfeiting_lands_on_quantlib():
    # QuantLib 1.44 BinomialConvertibleEngine<CRR>, 500 steps, on this bond at spot 12 (dirty NPV,
    # 2026-10-21 side-by-side). With the final coupon forfeited, what remains between GS and QuantLib
    # is O(dt): drift-matched vs martingale p, simple vs exponential discounting, per-child vs parent
    # blended rate, coupons snapped to nodes; together under 0.02 at 500 steps.
    quantlib_500 = 136.3232
    gs = ConvertibleBondBinomialEngine(_env(12.0), ConvertibleBondTreeParams(num_steps=500))
    assert gs.price_with_details(_bond(FinalCouponOnConversion.FORFEIT)).dirty_price == pytest.approx(quantlib_500, abs=0.02)
