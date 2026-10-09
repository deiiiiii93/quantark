"""A03: what the defect-A fix moves. Before / after in one process, back to back, compared as float.hex().

"Before" re-binds ``SnowballPDESolver._set_boundary_conditions_v0_beside_v1_rows`` to the plain
``_set_boundary_conditions_v0`` (exactly what the 2D hooks called before the fix: the copy row reads 0.0).
Rows: 2D Heston / SLV snowball and Heston Phoenix with continuous KI (standard and reverse), the byte book's own 2D
rows (test/pde_bitwise/capture_book.py vol_rows: continuous-KI snowball, n = (80, 32, 40)), delta / gamma of one
snowball through the solver's own Greeks, and a KI sitting just inside the grid's lower edge (the only place the row
can reach the price). Discrete KI never takes the new path (guarded on ``_ki_continuous``).
"""
import numpy as np

from common import HP, env, snowball, solver
from quantark.asset.equity.engine.pde.phoenix_vol_pde_solvers import HestonPhoenixPDESolver
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import HestonSLVSnowballPDESolver
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig, CouponPayType
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.snowball_config import PayoffConfig
from quantark.util.calendar.day_counter import DayCountConvention
from quantark.volmodels.slv.leverage import LeverageSurface
from common import barrier, S0, T

FIXED = SnowballPDESolver._set_boundary_conditions_v0_beside_v1_rows


def _before(self, grid, x_vec, s_vec, t_idx, tau, product, pricing_env):
    self._set_boundary_conditions_v0(grid, x_vec, s_vec, t_idx, tau, product, pricing_env)


def phoenix(reverse=False):
    ko, ki = (97.0, 125.0) if reverse else (103.0, 75.0)
    return PhoenixOption(
        initial_price=S0, strike=S0, maturity=T, contract_multiplier=1.0, is_reverse=reverse,
        barrier_config=barrier(ko, ki, True),
        coupon_config=CouponBarrierConfig(coupon_barrier=ki, coupon_rate=0.01, memory_coupon=False,
                                          coupon_pay_type=CouponPayType.INSTANT,
                                          day_count_convention=DayCountConvention.ACT_365),
        payoff_config=PayoffConfig(rebate_rate=0.0, include_principal=True))


def rows():
    ks = np.array(list(100.0 * np.exp(np.linspace(-1.2, 1.2, 15))))
    lev = LeverageSurface(time_grid=np.linspace(0, 2, 6), strike_grid=ks, leverage_grid=np.ones((6, ks.size)))
    std = dict(ko=103.0, ki=75.0, continuous=True, rebate_rate=0.15)
    rev = dict(ko=97.0, ki=125.0, continuous=True, rebate_rate=0.15, reverse=True)
    out = {}
    for n in ((96, 24, 48), (192, 48, 192)):
        out[f"heston snowball std {n}"] = lambda n=n: solver(HP, *n).price(snowball(**std), env())
        out[f"heston snowball rev {n}"] = lambda n=n: solver(HP, *n).price(snowball(**rev), env())
        out[f"slv snowball std {n}"] = lambda n=n: HestonSLVSnowballPDESolver(HP, lev, n_x=n[0], n_v=n[1],
                                                                              n_t=n[2]).price(snowball(**std), env())
        out[f"heston phoenix std {n}"] = lambda n=n: HestonPhoenixPDESolver(HP, n_x=n[0], n_v=n[1],
                                                                            n_t=n[2]).price(phoenix(), env())
        out[f"heston phoenix rev {n}"] = lambda n=n: HestonPhoenixPDESolver(HP, n_x=n[0], n_v=n[1],
                                                                            n_t=n[2]).price(phoenix(True), env())
    for ki in (10.5, 11.0, 12.0, 15.0):   # the grid's lower edge is 10.2 at n_x=96
        out[f"heston snowball std KI={ki} (96,24,48)"] = (
            lambda ki=ki: solver(HP, 96, 24, 48).price(snowball(**dict(std, ki=ki)), env()))

    def greeks():
        from quantark.asset.equity.riskmeasures import GreeksCalculator
        g = GreeksCalculator().calculate(snowball(**std), env(), solver(HP, 96, 24, 48), greeks=["delta", "gamma"])
        return (float(g["delta"]), float(g["gamma"]))
    out["heston snowball std delta,gamma (96,24,48)"] = greeks
    return out


def run_all():
    res = {}
    for k, f in rows().items():
        try:
            v = f()
            res[k] = tuple(float(x).hex() for x in (v if isinstance(v, tuple) else (v,)))
            res[k + " value"] = v
        except Exception as exc:  # noqa: BLE001 - probe
            res[k] = (f"{type(exc).__name__}: {exc}",)
            res[k + " value"] = None
    return res


if __name__ == "__main__":
    SnowballPDESolver._set_boundary_conditions_v0_beside_v1_rows = _before
    before = run_all()
    SnowballPDESolver._set_boundary_conditions_v0_beside_v1_rows = FIXED
    after = run_all()
    for k in rows():
        same = before[k] == after[k]
        b, a = before[k + " value"], after[k + " value"]
        if same:
            print(f"{k:<46} bit-identical  {b}")
        else:
            d = (np.asarray(a) - np.asarray(b)) if b is not None and a is not None else None
            print(f"{k:<46} MOVED  before={b} after={a} diff={d}")
