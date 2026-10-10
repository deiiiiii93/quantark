"""W13a bit-identity harness: price a broad grid with the UNPATCHED EuropeanMCEngine (and the test's barrier MC cells)
and print repr(price) or the refusal. Run once on the 0d1e4133 library code (05_bit_identity_before.out) and once
after the statistical check (05_bit_identity_after.out); `diff` must show only RAISED -> price lines.

Usage: python 05_bit_identity.py > 05_bit_identity_<tag>.out
"""
import warnings

from _common import make_env, make_option
from quantark.asset.equity.engine.mc.euro_mc_engine import EuropeanMCEngine
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit
from quantark.util.enum import OptionType
from quantark.util.enum.engine_enums import MonteCarloMethod

CONFIGS = (
    ("plain256", MonteCarloMethod.PSEUDO, dict(num_paths=256)),
    ("anti256", MonteCarloMethod.PSEUDO, dict(num_paths=256, use_antithetic=True)),
    ("qmc1024", MonteCarloMethod.QUASI, dict(num_paths=1024)),
    ("rqmc256", MonteCarloMethod.RANDOMIZED_QUASI, dict(num_paths=256)),
)


def price_or_raise(engine, product, env):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return repr(float(engine.price(product, env)))
    except Exception as exc:  # noqa: BLE001
        return f"RAISED {type(exc).__name__}: {exc}"


for term in (False, True):
    for spot in (75.0, 90.0, 100.0, 110.0, 125.0):
        env = make_env(spot, term_curve=term)
        for opt in (OptionType.PUT, OptionType.CALL):
            for days in (1, 5, 30, 365):
                for lag in (0, 2):
                    for mult in (1.0, 100.0):
                        kw = dict(strike=100.0, option_type=opt, maturity=days / 365, contract_multiplier=mult)
                        if lag:
                            kw["settlement_convention"] = SettlementConvention(
                                lag=lag / 365.0, lag_unit=SettlementLagUnit.YEAR_FRACTION)
                        product = EuropeanVanillaOption(**kw)
                        for name, method, extra in CONFIGS:
                            engine = EuropeanMCEngine(params=MCParams(seed=7, **extra), method=method)
                            cell = (f"{'term' if term else 'flat'} S{spot:g} {opt.name.lower()} T={days}d "
                                    f"lag{lag}d mult{mult:g} {name}")
                            print(f"{cell:48s} {price_or_raise(engine, product, env)}", flush=True)

# the carried knock-in test cells (BarrierOptionMCEngine -> EuropeanMCEngine), day before and exercise date
from test_expiry_carried_knock_in import BARRIER_PRODUCTS, BEFORE, VALUATION, _mc, env as test_env  # noqa: E402
from quantark.asset.equity.engine.mc import BarrierOptionMCEngine  # noqa: E402

for kind, build, _, spots in BARRIER_PRODUCTS:
    for form in ("dated", "numeric"):
        for carried in (True, False):
            for valuation, tag in ((BEFORE, "before"), (VALUATION, "exercise")):
                for spot in spots:
                    try:
                        product = build(form, carried, valuation)
                    except Exception as exc:  # noqa: BLE001
                        print(f"barrier {kind} {form} carried={carried} {tag} S{spot:g}: BUILD {exc}")
                        continue
                    res = price_or_raise(BarrierOptionMCEngine(params=_mc()), product, test_env(spot, valuation))
                    print(f"barrier {kind} {form} carried={carried} {tag} S{spot:g}: {res}", flush=True)
