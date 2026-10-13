import numpy as np
import pytest
from datetime import datetime
from quantark.param import GridVolSurface, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import FxPricingEnvironment
from quantark.asset.fx.product.option.fx_vanilla_option import FxVanillaOption
from quantark.asset.fx.engine.analytical import GarmanKohlhagenEngine
from quantark.asset.fx.engine.mc.local_vol_mc_engine import FxLocalVolMCEngine
from quantark.asset.fx.engine.pde.local_vol_pde_solver import FxLocalVolPDESolver
from quantark.util.enum import OptionType
from quantark.util.exceptions import PricingError


def _fx_env(vol=0.1, rd=0.03, rf=0.01):
    strikes = list(1.20 * np.exp(np.linspace(-0.4, 0.4, 7)))
    mats = list(np.linspace(0.1, 2.0, 6))
    surf = GridVolSurface(strikes, mats, np.full((6, 7), vol))
    return FxPricingEnvironment(
        valuation_date=datetime(2026, 1, 1), spot_quote=SpotQuote(spot=1.20),
        domestic_curve=FlatRateCurve(rd), foreign_curve=FlatRateCurve(rf), vol_surface=surf,
    )


def _opt(strike=1.20, T=1.0, notional=1_000_000.0, participation=1.0):
    return FxVanillaOption(strike=strike, option_type=OptionType.CALL, maturity=T,
                           notional_foreign=notional, participation_rate=participation)


def test_fx_lv_pde_matches_gk_flat_surface():
    env = _fx_env(0.1, 0.03, 0.01)
    opt = _opt()
    gk = GarmanKohlhagenEngine().price(opt, env)
    pde = FxLocalVolPDESolver(grid_size=500, time_steps=200).price(opt, env)
    assert pde == pytest.approx(gk, rel=2e-3)


def test_fx_lv_mc_matches_gk_flat_surface():
    env = _fx_env(0.1, 0.03, 0.01)
    opt = _opt()
    gk = GarmanKohlhagenEngine().price(opt, env)
    mc = FxLocalVolMCEngine(num_paths=300_000, time_steps=200, seed=5).price(opt, env)
    # per-unit MC error ~ notional * 1e-4; scale tolerance to notional
    assert mc == pytest.approx(gk, rel=5e-3)


def test_fx_lv_sizing_scales_with_notional_and_participation():
    env = _fx_env()
    base = FxLocalVolPDESolver(grid_size=400, time_steps=150).price(_opt(notional=1e6), env)
    dbl = FxLocalVolPDESolver(grid_size=400, time_steps=150).price(_opt(notional=2e6), env)
    part = FxLocalVolPDESolver(grid_size=400, time_steps=150).price(
        _opt(notional=1e6, participation=2.0), env)
    assert dbl == pytest.approx(2 * base, rel=1e-9)
    assert part == pytest.approx(2 * base, rel=1e-9)


def test_fx_lv_rejects_market_forward_override():
    env = _fx_env()
    env.market_forward = 1.25
    with pytest.raises(PricingError):
        FxLocalVolPDESolver().price(_opt(), env)


def test_fx_lv_requires_grid_vol_surface():
    env = _fx_env()
    env.vol_surface = FlatVolSurface(0.1)
    with pytest.raises(PricingError):
        FxLocalVolPDESolver().price(_opt(), env)


def test_fx_lv_greeks_exclude_vega():
    env = _fx_env()
    g = FxLocalVolPDESolver(grid_size=300, time_steps=120).calculate_greeks(_opt(), env)
    assert set(["price", "delta", "gamma", "rho_dom", "rho_for"]).issubset(g.keys())
    assert "vega" not in g


def test_fx_lv_greeks_match_gk_magnitude():
    env = _fx_env(0.1, 0.03, 0.01)
    opt = _opt()
    gk = GarmanKohlhagenEngine().calculate_greeks(opt, env)
    g = FxLocalVolPDESolver(grid_size=600, time_steps=250).calculate_greeks(opt, env)
    assert "theta" in g
    assert g["rho_dom"] == pytest.approx(gk["rho_dom"], rel=2e-2)
    assert g["rho_for"] == pytest.approx(gk["rho_for"], rel=2e-2)
    assert g["delta"] == pytest.approx(gk["delta"], rel=3e-2)  # FD-on-PDE delta carries grid noise


def test_fx_lv_rejects_expiry_delivery_mismatch():
    env = _fx_env()
    opt = FxVanillaOption(strike=1.20, option_type=OptionType.CALL, maturity=1.0,
                          delivery=1.05, notional_foreign=1e6)
    with pytest.raises(PricingError):
        FxLocalVolPDESolver().price(opt, env)


def test_fx_lv_theta_with_explicit_equal_delivery():
    # Explicit delivery == maturity must survive the theta maturity-shrink (both reduced).
    env = _fx_env()
    opt = FxVanillaOption(strike=1.20, option_type=OptionType.CALL, maturity=1.0,
                          delivery=1.0, notional_foreign=1e6)
    g = FxLocalVolPDESolver(grid_size=300, time_steps=120).calculate_greeks(opt, env)
    assert "theta" in g and g["theta"] < 0


# --- I8 / AC12: LV and Heston FX numeric theta steps by the environment's year fraction of theta_days -------------
from quantark.asset.fx.engine.localvol_greeks import fx_local_vol_model_greeks  # noqa: E402
from quantark.asset.fx.engine.pde import FxHestonPDESolver, FxHestonSLVPDESolver  # noqa: E402
from quantark.util.calendar.business_calendar import CalendarType, create_calendar  # noqa: E402
from quantark.util.calendar.day_counter import DayCountConvention as DC  # noqa: E402
from quantark.util.enum.engine_enums import LeverageCalibrationMethod  # noqa: E402
from quantark.util.exceptions import ValidationError  # noqa: E402
from quantark.volmodels.heston import HestonParams  # noqa: E402
from quantark.volmodels.localvol import build_dupire_local_vol  # noqa: E402
from quantark.volmodels.slv import calibrate_leverage_surface  # noqa: E402

_FRI = datetime(2026, 6, 12)  # a Friday
_HP = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.2, rho=-0.5)
_GK = GarmanKohlhagenEngine()


def _theta_env(dc, cal=None):
    return FxPricingEnvironment(
        valuation_date=_FRI, spot_quote=SpotQuote(spot=1.20), domestic_curve=FlatRateCurve(0.03),
        foreign_curve=FlatRateCurve(0.01), vol_surface=FlatVolSurface(0.1), day_count_convention=dc, calendar=cal)


def _theta_opt(T=1.0):
    return FxVanillaOption(strike=1.20, option_type=OptionType.CALL, maturity=T, notional_foreign=1e6)


def _lv_theta(env, opt, theta_days=1):
    return fx_local_vol_model_greeks(lambda p, e, s: _GK.price(p, e), opt, env, None, theta_days=theta_days)["theta"]


def _gk_at(env, T):
    return _GK.price(_theta_opt(T), env)


_LEV = None


def _lev_surface():
    global _LEV
    if _LEV is None:
        spot = 1.20
        strikes = list(spot * np.exp(np.linspace(-0.6, 0.6, 9)))
        mats = list(np.linspace(0.1, 2.0, 6))
        lv = build_dupire_local_vol(GridVolSurface(strikes, mats, np.full((6, 9), 0.2)), spot=spot,
                                    rate_curve=FlatRateCurve(0.0), div_yield=lambda t: 0.0)
        M = 20
        _LEV = calibrate_leverage_surface(spot, _HP, lv, np.full(M, 1.0 / M), np.zeros(M), np.zeros(M), eta=1.0,
                                          method=LeverageCalibrationMethod.MC_BINNING, num_paths=4000, num_bins=8,
                                          seed=5)
    return _LEV


def _heston_solver():
    return FxHestonPDESolver(_HP, n_x=40, n_v=20, n_t=20)


def _slv_solver():
    return FxHestonSLVPDESolver(_HP, _lev_surface(), eta=1.0, n_x=40, n_v=20, n_t=20)


_ROUTES = {
    "lv": lambda env, opt: _lv_theta(env, opt),
    "heston": lambda env, opt: _heston_solver().calculate_greeks(opt, env)["theta"],
    "slv": lambda env, opt: _slv_solver().calculate_greeks(opt, env)["theta"],
}
# Full-precision values captured on the code BEFORE the I8 change (ACT_365 and CALENDAR_DAYS must reproduce them).
_OLD_THETA = {
    "lv": "-0x1.8435e36bca800p+6",
    "heston": "0x1.b2a4bd3298500p+8",
    "slv": "0x1.ced6089f88c00p+8",
}


def _spy(solver):
    """Wrap solver._price; returns the list of (maturity, price) it was asked for."""
    calls = []
    real = solver._price

    def wrapped(product, env):
        v = real(product, env)
        calls.append((float(product.maturity), v))
        return v

    solver._price = wrapped
    return calls


@pytest.mark.parametrize("route", list(_ROUTES))
@pytest.mark.parametrize("dc", [DC.ACT_365, DC.CALENDAR_DAYS])
def test_i8_calendar_day_theta_is_bitwise_unchanged(route, dc):
    got = _ROUTES[route](_theta_env(dc), _theta_opt())
    assert got.hex() == _OLD_THETA[route]


@pytest.mark.parametrize("route", list(_ROUTES))
def test_i8_business_days_friday_theta_is_zero(route):
    env = _theta_env(DC.BUSINESS_DAYS, create_calendar(CalendarType.TARGET, year_range=(2026, 2028)))
    assert _ROUTES[route](env, _theta_opt()) == 0.0


@pytest.mark.parametrize("route", list(_ROUTES))
def test_i8_business_days_without_calendar_raises(route):
    with pytest.raises(ValidationError):
        _ROUTES[route](_theta_env(DC.BUSINESS_DAYS), _theta_opt())


def test_i8_lv_act_360_steps_one_360th():
    env = _theta_env(DC.ACT_360)
    got = _lv_theta(env, _theta_opt())
    base = _GK.price(_theta_opt(), env)
    assert got == pytest.approx((_gk_at(env, 1.0 - 1.0 / 360.0) - base) / 1.0, rel=1e-12)
    # the legacy 1/365 step gives a different number (the step really moved)
    assert got != pytest.approx((_gk_at(env, 1.0 - 1.0 / 365.0) - base) / 1.0, rel=1e-9)


def test_i8_lv_multi_day_step_is_per_calendar_day():
    env = _theta_env(DC.ACT_360)
    got = _lv_theta(env, _theta_opt(), theta_days=3)
    base = _GK.price(_theta_opt(), env)
    dt = 3.0 / 360.0
    assert got == pytest.approx((_gk_at(env, 1.0 - dt) - base) / 3.0, rel=1e-12)


def test_i8_lv_cap_keeps_half_maturity_and_rescales():
    env = _theta_env(DC.ACT_360)
    opt = _theta_opt(T=1.0 / 720.0)  # eff = 0.5 * T < 1/360
    got = _lv_theta(env, opt)
    base = _GK.price(opt, env)
    eff = 0.5 * opt.maturity
    expected = (_gk_at(env, opt.maturity - eff) - base) * (1.0 / 360.0) / (eff * 1.0)
    assert got == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("route", ["heston", "slv"])
def test_i8_heston_routes_act_360_step_one_360th(route):
    env = _theta_env(DC.ACT_360)
    opt = _theta_opt()
    solver = _heston_solver() if route == "heston" else _slv_solver()
    calls = _spy(solver)
    g = solver.calculate_greeks(opt, env)
    shifted_T, future = calls[-1]
    assert shifted_T == pytest.approx(1.0 - 1.0 / 360.0, rel=1e-15)
    assert g["theta"] == pytest.approx((future - g["price"]) / 1.0, rel=1e-12)
