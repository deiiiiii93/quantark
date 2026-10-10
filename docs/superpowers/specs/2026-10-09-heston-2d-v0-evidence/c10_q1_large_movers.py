"""C10: the two large Q1 movers in c09 -- are they the v = 0 column fix converging, or something else?

    env PYTHONPATH=<library root>:<wt>/test <venv>/bin/python c10_q1_large_movers.py pde <out.json>   (main and branch)
    env PYTHONPATH=<wt>:<wt>/test        <venv>/bin/python c10_q1_large_movers.py mc  <out.json>   (library MC, once)

1. The reverse 2D Heston Phoenix (KO 97 / continuous KI 125 / coupon barrier 125, 1% monthly INSTANT coupon, W3's
   Heston): c09 moved it -5.76e-3 at (96, 24, 48) and -0.1059 at (192, 48, 192). Refinement ladder, plus (branch only)
   the Q3 half-cell floor measured at the barrier (C02) and the standard orientation. Reference: the library's
   HestonPhoenixMCEngine (QE, continuous-KI bridge; no MC enters a PDE path).
2. The 2D Heston DCN_A under the Feller-violating calibration of test_dcn_vol_pde_solvers.py: c09 moved it -934.7 on
   (151, 41). Ladder (151, 41), (251, 81), (501, 201); reference: HestonDCNMCEngine QE-M as that test, 4x its paths.
"""
import json
import sys

from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver

if not hasattr(SnowballPDESolver, "_set_boundary_conditions_v0_beside_v1_rows"):   # as c09: a03 imports on main
    SnowballPDESolver._set_boundary_conditions_v0_beside_v1_rows = None

from a03_moved_values import phoenix  # noqa: E402
from common import HP, env  # noqa: E402

FELLER = dict(v0=0.0249, kappa=9.92, theta=0.045, sigma=1.583, rho=-0.025)
PHX_GRIDS = ((96, 24, 48), (192, 48, 192), (192, 96, 192), (288, 72, 288), (384, 96, 384))
DCN_GRIDS = ((151, 41), (251, 81), (501, 201))


def pde():
    from dcn_fixtures import DCN_A, FLAT, flat_env, make_dcn
    from quantark.asset.equity.engine.pde.dcn_vol_pde_solvers import HestonDCNPDESolver
    from quantark.asset.equity.engine.pde.phoenix_vol_pde_solvers import HestonPhoenixPDESolver
    from quantark.volmodels.heston import HestonParams

    out = {}

    def phx(n, reverse=True):
        return HestonPhoenixPDESolver(HP, n_x=n[0], n_v=n[1], n_t=n[2]).price(phoenix(reverse), env())

    for n in PHX_GRIDS:
        out[f"phoenix rev {n}"] = phx(n)
        out[f"phoenix std {n}"] = phx(n, reverse=False)
    try:
        from c02_q3_common import barrier_cell_floor   # the branch's evidence helpers; harmless on main
        with barrier_cell_floor():
            for n in PHX_GRIDS:
                out[f"phoenix rev {n} + Q3 floor"] = phx(n)
    except Exception as exc:  # noqa: BLE001 - probe
        out["floor arm"] = f"{type(exc).__name__}: {exc}"
    p, e = make_dcn(DCN_A), flat_env(**FLAT)
    out["dcn notional"] = float(p.notional)
    for n in DCN_GRIDS:
        out[f"dcn {n}"] = HestonDCNPDESolver(HestonParams(**FELLER), n_x=n[0], n_v=n[1]).price(p, e)
    return out


def mc():
    from dcn_fixtures import DCN_A, FLAT, flat_env, make_dcn
    from quantark.asset.equity.engine.mc.dcn_vol_mc_engines import HestonDCNMCEngine
    from quantark.asset.equity.engine.mc.phoenix_vol_mc_engines import HestonPhoenixMCEngine
    from quantark.asset.equity.param.engine_params import MCParams
    from quantark.util.enum.engine_enums import HestonMCScheme
    from quantark.volmodels.heston import HestonParams

    out = {}
    r = HestonDCNMCEngine(model_params=HestonParams(**FELLER), scheme=HestonMCScheme.QUADEXP_M,
                          substeps_per_interval=2, num_paths=4 * 2 ** 14, seed=42, use_sobol=True,
                          num_batches=8).price_detailed(make_dcn(DCN_A), flat_env(**FLAT))
    out["dcn MC QE-M"] = [float(r.pv), float(r.std_error)]
    print(out, flush=True)
    for reverse in (True, False):
        eng = HestonPhoenixMCEngine(HP, params=MCParams(num_paths=200_000, seed=20261010, use_antithetic=True,
                                                         use_business_day_grid=True),
                                    scheme=HestonMCScheme.QUADEXP, num_batches=8, substeps_per_interval=4)
        pv = eng.price(phoenix(reverse), env())
        out[f"phoenix {'rev' if reverse else 'std'} MC QE sub 4"] = [float(pv), float(eng.get_last_std_error())]
        print(out, flush=True)
    return out


if __name__ == "__main__":
    import quantark
    print("library:", quantark.__file__, flush=True)
    res = pde() if sys.argv[1] == "pde" else mc()
    for k, v in res.items():
        print(f"{k:<40} {v!r}", flush=True)
    json.dump(res, open(sys.argv[2], "w"), indent=1)
