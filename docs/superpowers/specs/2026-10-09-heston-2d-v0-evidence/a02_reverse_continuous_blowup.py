"""A02: side finding of a01 -- the 2D Heston reverse continuous-KI snowball (KO 97 / KI 125) diverges at
n = (384, 48, 384) (price ~ -3.8e211), with or without the copy-row patch. Bisect one axis at a time; also the
discrete-KI twin, the standard orientation and the BSM limit at the same grid. Diagnosis only, no fix.
"""
import math

from common import HP, env, snowball, solver
from quantark.volmodels.heston import HestonParams

BSM_LIMIT = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=1e-4, rho=-0.5)


def px(n, hp=HP, **kw):
    try:
        v = solver(hp, *n).price(snowball(**kw), env())
    except Exception as exc:  # noqa: BLE001 - probe
        return f"{type(exc).__name__}: {exc}"
    return f"{v:.6f}" if math.isfinite(v) and abs(v) < 1e6 else f"DIVERGED {v:.3e}"


if __name__ == "__main__":
    rev_c = dict(ko=97.0, ki=125.0, continuous=True, rebate_rate=0.15, reverse=True)
    rev_d = dict(ko=97.0, ki=125.0, continuous=False, rebate_rate=0.15, reverse=True)
    std_c = dict(ko=103.0, ki=75.0, continuous=True, rebate_rate=0.15)
    for n in ((192, 48, 192), (384, 48, 192), (192, 48, 384), (384, 48, 384), (384, 96, 384), (288, 48, 288),
              (384, 24, 384)):
        print(f"n={n}: rev cont {px(n, **rev_c)} | rev disc {px(n, **rev_d)} | std cont {px(n, **std_c)} | "
              f"rev cont BSM-limit {px(n, BSM_LIMIT, **rev_c)}", flush=True)
