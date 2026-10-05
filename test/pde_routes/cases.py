"""Cases of the 3C references and tests: the flat market, the two local-vol markets, the vanilla and
barrier products, Black-Scholes, the DCN fixture contracts, the kinked term curves and the observation
schedules of review 3.

Banked from the 3C evidence `l_lab.py` (lines 41, 145-243, 270-291), `lab3c.py` (`dcn_case`),
`x6_term_structure.py` (`kinked`, its curves) and `w3b_valuation_event.py` (`schedule`) (spec 3C §7.2);
not production code.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
from scipy.special import ndtr

from quantark.asset.equity.product.option import BarrierOption, EuropeanVanillaOption
from quantark.asset.equity.product.option.observation_schedule import ObservationSchedule
from quantark.param import FlatRateCurve, FlatVolSurface, GridVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.param.div.dividend_yield import TermStructureDividendYield
from quantark.param.rrf.rate_curve import LinearRateCurve
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import BarrierType, ObservationType, OptionType
from quantark.volmodels.localvol import LocalVolSurface, build_dupire_local_vol

VAL_DATE = datetime(2026, 1, 1)


# --- markets -------------------------------------------------------------------------------------

CONST = dict(vol=0.22, rate=0.03, div=0.01)


def env_const(spot=100.0, **over):
    m = dict(CONST, **over)
    return PricingEnvironment(rate_curve=FlatRateCurve(m["rate"]), valuation_date=VAL_DATE,
                              spot_quote=SpotQuote(spot=spot), vol_surface=FlatVolSurface(volatility=m["vol"]),
                              div_yield=ContinuousDividendYield(m["div"]))


def surface_const(vol=CONST["vol"]):
    return LocalVolSurface(strike_grid=np.array([1.0, 1.0e4]), time_grid=np.array([1.0]),
                           lv_grid=np.full((1, 2), float(vol)))


def env_book(spot=100.0):
    """The byte book's ``env_grid_vol`` (test/pde_bitwise/capture_book.py), verbatim."""
    strikes = list(100.0 * np.exp(np.linspace(-0.6, 0.6, 9)))
    smile = [0.22 + 0.02 * abs(k - 4) / 4 for k in range(9)]
    surf = GridVolSurface(strikes, list(np.linspace(0.1, 2.0, 6)), np.array([smile] * 6))
    return PricingEnvironment(rate_curve=FlatRateCurve(0.03), valuation_date=datetime(2026, 1, 1),
                              spot_quote=SpotQuote(spot=spot), vol_surface=surf,
                              div_yield=ContinuousDividendYield(0.01))


def env_steep(spot=100.0):
    """A steep equity skew with term decay: iv = 0.22 - a(T) k + 0.10 k^2, k = ln(K/100), a = 0.18 / sqrt(max(T, .25))^.5."""
    k = np.linspace(-0.6, 0.6, 13)
    times = np.array([0.1, 0.25, 0.5, 1.0, 1.5, 2.0])
    a = 0.18 / np.maximum(times, 0.25) ** 0.25
    iv = 0.22 - a[:, None] * k[None, :] + 0.10 * k[None, :] ** 2
    surf = GridVolSurface(list(100.0 * np.exp(k)), list(times), iv)
    return PricingEnvironment(rate_curve=FlatRateCurve(0.03), valuation_date=datetime(2026, 1, 1),
                              spot_quote=SpotQuote(spot=spot), vol_surface=surf,
                              div_yield=ContinuousDividendYield(0.01))


def dupire(env):
    return build_dupire_local_vol(env.vol_surface, spot=env.spot, rate_curve=env.rate_curve,
                                  div_yield=env.get_div_yield)


_MARKETS = {}


def market(name):
    """(env at spot 100, LocalVolSurface built once at spot 100 and held fixed)."""
    if name not in _MARKETS:
        if name == "const":
            _MARKETS[name] = (env_const, surface_const())
        elif name == "book":
            _MARKETS[name] = (env_book, dupire(env_book()))
        elif name == "steep":
            _MARKETS[name] = (env_steep, dupire(env_steep()))
        else:
            raise KeyError(name)
    make, surface = _MARKETS[name]
    return make, surface


# --- products ------------------------------------------------------------------------------------

def vanilla(call=True, strike=100.0, T=1.0):
    return EuropeanVanillaOption(strike=strike, option_type=OptionType.CALL if call else OptionType.PUT, maturity=T)


OBS = {"monthly": 12, "weekly": 52, "daily": 252}


def obs_dates(freq, T=1.0):
    n = int(round(OBS[freq] * T))
    return [T * k / n for k in range(1, n + 1)]


def barrier(kind, call, level, *, strike=100.0, T=1.0, obs="cont", rebate=0.0, at_hit=False, participation=1.0):
    """kind in UO / UI / DO / DI; obs in cont / expiry / monthly / weekly / daily."""
    bt = {"UO": BarrierType.UP_OUT, "UI": BarrierType.UP_IN, "DO": BarrierType.DOWN_OUT, "DI": BarrierType.DOWN_IN}[kind]
    kw = dict(strike=strike, option_type=OptionType.CALL if call else OptionType.PUT, barrier=level, barrier_type=bt,
              maturity=T, rebate=rebate, pay_at_hit=at_hit, participation_rate=participation)
    if obs == "cont":
        kw["observation_type"] = ObservationType.CONTINUOUS
    elif obs == "expiry":
        kw["observation_type"] = ObservationType.EXPIRY
    else:
        kw.update(observation_type=ObservationType.DISCRETE, observation_dates=obs_dates(obs, T))
    return BarrierOption(**kw)


def as_single_date_discrete(product):
    """EXPIRY monitoring for the prototype: the core barrier family refuses EXPIRY, so the
    prototype prices it as discrete monitoring with the maturity as its only observation."""
    if product.observation_type != ObservationType.EXPIRY:
        return product
    return BarrierOption(strike=product.strike, option_type=product.option_type, barrier=product.barrier,
                         barrier_type=product.barrier_type, maturity=product.maturity, rebate=product.rebate,
                         pay_at_hit=product.pay_at_hit, participation_rate=product.participation_rate,
                         observation_type=ObservationType.DISCRETE, observation_dates=[float(product.maturity)])




# --- closed forms --------------------------------------------------------------------------------

def _npdf(z):
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


def black_scholes(call, spot, strike, T, vol=CONST["vol"], r=CONST["rate"], q=CONST["div"]):
    """(price, delta, gamma)."""
    sd = vol * np.sqrt(T)
    d1 = (np.log(spot / strike) + (r - q + 0.5 * vol * vol) * T) / sd
    d2 = d1 - sd
    dq, dr = np.exp(-q * T), np.exp(-r * T)
    gamma = dq * _npdf(d1) / (spot * sd)
    if call:
        return spot * dq * ndtr(d1) - strike * dr * ndtr(d2), dq * ndtr(d1), gamma
    return strike * dr * ndtr(-d2) - spot * dq * ndtr(-d1), -dq * ndtr(-d1), gamma


def analytic_barrier(product, env):
    from quantark.asset.equity.engine.analytical.barrier_analytical_engine import BarrierAnalyticalEngine

    return float(BarrierAnalyticalEngine().price(product, env))




# --- DCN cases (lab3c.py) ------------------------------------------------------------------------

def dcn_case(name: str, **overrides):
    """(product, flat env) of a fixture contract: 'A', 'B', 'SA', 'SB' (test/dcn_fixtures.py)."""
    import dcn_fixtures as fx  # test/ is on sys.path

    contract = {"A": fx.DCN_A, "B": fx.DCN_B, "SA": fx.SAMPLE_DCN_A, "SB": fx.SAMPLE_DCN_B}[name]
    market = dict(fx.FLAT)
    spot = overrides.pop("spot", contract["initial_price"])
    for k in ("r", "q", "sigma"):
        if k in overrides:
            market[k] = overrides.pop(k)
    return fx.make_dcn(contract, **overrides), fx.flat_env(**market, spot=spot)


# --- the local-vol DCN surfaces ($E/d6_localvol.py:20-26, verbatim) -----------------------------

def dcn_flat_surface(s0=6000.0):
    strikes, times = s0 * np.linspace(0.2, 3.0, 141), np.array([0.0, 1.0, 2.0, 3.0])
    return LocalVolSurface(strikes, times, np.full((times.size, strikes.size), 0.184))


def dcn_skew_surface(s0=6000.0):
    """0.184 S0 / S, clipped to 8 % - 60 % (§4.2)."""
    strikes, times = s0 * np.linspace(0.2, 3.0, 141), np.array([0.0, 1.0, 2.0, 3.0])
    return LocalVolSurface(strikes, times, np.tile(np.clip(0.184 * s0 / strikes, 0.08, 0.60), (times.size, 1)))


# --- kinked term curves (x6_term_structure.py; test/term_structure_benchmarks.py "kinked") ---------

VOL = CONST["vol"]
TIMES = [0.25, 0.5, 1.0, 2.0]
RATES = [0.020, 0.035, 0.025, 0.030]            # test/term_structure_benchmarks.py "kinked"
YIELDS = [-0.015, 0.020, -0.005, 0.010]
D_RATES = [0.030, 0.040, 0.033, 0.037]
D_YIELDS = [0.120, 0.155, 0.135, 0.145]


def kinked(spot, vol=VOL, rates=RATES, yields=YIELDS, valuation_date=VAL_DATE):
    return PricingEnvironment(rate_curve=LinearRateCurve(list(zip(TIMES, rates))), valuation_date=valuation_date,
                              spot_quote=SpotQuote(spot=spot), vol_surface=FlatVolSurface(volatility=vol),
                              div_yield=TermStructureDividendYield(times=list(TIMES), yields=list(yields)))


# --- the valuation-date record product of test_valuation_decision.py (spec 3C §4.11) --------------------------

def record_product():
    """A record on the valuation date at 120 paying 7 at 0.25, then records at 0.5 and 1 on the defaults."""
    from quantark.asset.equity.product.option.observation_schedule import ObservationRecord
    from quantark.util.enum import ObservationAggregation

    sch = ObservationSchedule(records=[ObservationRecord(observation_time=0.0, barrier=120.0, payoff=7.0, settlement_time=0.25),
                                       ObservationRecord(observation_time=0.5), ObservationRecord(observation_time=1.0)],
                              aggregation_mode=ObservationAggregation.STOP_FIRST_HIT)
    return BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=130.0, barrier_type=BarrierType.UP_OUT,
                         maturity=1.0, rebate=5.0, pay_at_hit=True, observation_type=ObservationType.DISCRETE,
                         observation_schedule=sch)


# --- observation schedules (w3b_valuation_event.py, levels as arguments) -------------------------

def schedule(times, mode, *, double=False, barrier=130.0, payoff=5.0, lower=80.0):
    kw = dict(upper_barrier=barrier, lower_barrier=lower) if double else {}
    return ObservationSchedule.from_legacy(observation_dates=list(times), default_barrier=None if double else barrier,
                                           default_payoff=payoff, aggregation_mode=mode, **kw)
