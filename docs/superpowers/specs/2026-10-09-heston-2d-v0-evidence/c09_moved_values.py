"""C09: the values W7 moves, for moved_values.txt. Run once against main's library and once against the branch.

    env PYTHONPATH=<library root>:<wt>/test <venv>/bin/python c09_moved_values.py <out.json>

``<library root>`` is a directory holding ``quantark/`` (``git archive 0d1e4133 quantark`` for main; the worktree for
the branch). Every value is written as repr and float.hex(); ``c09_diff.py`` compares two outputs. Rows: the a03
desk-like set (2D Heston / SLV snowball, Heston Phoenix, continuous KI; edge-KI; delta / gamma), the byte book's 2D
rows (a07), the certain-V0 controls at (96, 24, 48) (standard KO 500, reverse KO 20 and KO 2; flat / annualized;
default damping and every damping step off), a 2D Heston DCN on its defaults, and the 1D Phoenix continuous-KI set (a05).
"""
import json
import sys

from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver

# a03 / a07 read the defect-A helper at import (their own before/after arms, unused here). On main the helper does not
# exist and main's 2D hooks never call it: a placeholder lets the row definitions import, and moves nothing.
if not hasattr(SnowballPDESolver, "_set_boundary_conditions_v0_beside_v1_rows"):
    SnowballPDESolver._set_boundary_conditions_v0_beside_v1_rows = None

import a03_moved_values as a03  # noqa: E402
import a07_byte_book_2d_rows as a07  # noqa: E402
from a05_phoenix_1d_copy_row import phoenix as phoenix_1d
from common import HP, certain_v0, closed_form, env, solver
from quantark.asset.equity.engine.pde.core.components import CrankNicolson
from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.param.engine_params import PDEParams


def rows():
    out = {}
    for k, f in a03.rows().items():
        out[f"desk: {k}"] = f
    for k, f in a07.rows().items():
        out[f"byte book 2D: {k}"] = f
    e = env()
    for label, st in (("default damping", None),
                      ("damping off", CrankNicolson(event_damping_steps=0, terminal_damping_steps=0))):
        for rev, ko in ((False, None), (True, 20.0), (True, 2.0)):
            for ann in (False, True):
                def f(rev=rev, ko=ko, ann=ann, st=st):
                    p = certain_v0(0.10, ann, reverse=rev, ko=ko)
                    v = solver(HP, 96, 24, 48, params=PDEParams(stepper=st) if st else None).price(p, e)
                    return (v, v - closed_form(p, e, 0.10, ann))
                out[f"control: {'rev' if rev else 'std'} KO {ko or 500.0:g} {'ann' if ann else 'flat'} {label}"] = f

    def dcn():
        from dcn_fixtures import DCN_A, FLAT, flat_env, make_dcn
        from quantark.asset.equity.engine.pde.dcn_vol_pde_solvers import HestonDCNPDESolver
        from quantark.volmodels.heston import HestonParams
        hp = HestonParams(v0=0.0249, kappa=9.92, theta=0.045, sigma=1.583, rho=-0.025)
        return HestonDCNPDESolver(hp, n_x=151, n_v=41).price(make_dcn(DCN_A), flat_env(**FLAT))
    out["dcn: Heston DCN_A (151, 41) defaults"] = dcn
    for T in (2.0, 0.25):
        for reverse in (False, True):
            for memory in (False, True):
                for ki in (None,) + ((60.0, 50.0) if not reverse else (150.0, 200.0)):
                    p = phoenix_1d(reverse, memory, T, ki)
                    tag = (f"1D phoenix: T={T} {'rev' if reverse else 'std'} {'memory' if memory else 'plain'} "
                           f"KI={p.barrier_config.ki_barrier}")
                    out[tag] = lambda p=p: PhoenixPDESolver().price(p, env())
    return out


if __name__ == "__main__":
    import quantark
    print("library:", quantark.__file__, flush=True)
    res = {}
    for k, f in rows().items():
        try:
            v = f()
            vals = list(v) if isinstance(v, tuple) else [v]
            res[k] = {"repr": [repr(float(x)) for x in vals], "hex": [float(x).hex() for x in vals]}
        except Exception as exc:  # noqa: BLE001 - probe
            res[k] = {"error": f"{type(exc).__name__}: {exc}"}
        print(k, res[k].get("repr", res[k].get("error")), flush=True)
    with open(sys.argv[1], "w") as fh:
        json.dump(res, fh, indent=1)
