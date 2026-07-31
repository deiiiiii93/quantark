"""Is degenerate_pde ALONE enough at real UNCONSTRAINED (Feller-violating) fits?

Settles the last assumption in spec 7A: the +0.33/+0.55% residual that justified
enforce_feller=True was measured on ONE synthetic control.  This runs the real
production configuration -- the gate's own product, env, engines and tolerance --
at both calibration policies on the same dates, so the comparison is paired.

Usage: unconstrained_pde_vs_mc.py <out.json> <date> [<date> ...]
"""
import copy, importlib.util, json, math, sys, time
from datetime import date as _date
from pathlib import Path

REPO = Path("/Users/fuxinyao/quant-ark")
sys.path.insert(0, str(REPO / "example" / "mo_volmodels"))

spec = importlib.util.spec_from_file_location(
    "gate", REPO / "example" / "mo_volmodels" / "11_pde_convergence_gate.py")
gate = importlib.util.module_from_spec(spec)
sys.modules["gate"] = gate  # @dataclass resolves cls.__module__ via sys.modules
spec.loader.exec_module(gate)

from quantark.asset.equity.param import PDEParams
from quantark.param.vol.surface_history import IvSurfaceArtifact
from quantark.volmodels.calibration import HESTON_PRESETS, VolModelCalibrator
from quantark.backtest.replay.config import VolModelCalibrationConfig

HIST = REPO / "example" / "mo_volmodels" / "data" / "history"
IV = HIST / "iv_surface"


def _calibrate(art, enforce):
    preset = copy.deepcopy(HESTON_PRESETS["mo_frozen"])
    preset["enforce_feller"] = enforce
    HESTON_PRESETS["mo_frozen"] = preset
    return VolModelCalibrator(VolModelCalibrationConfig()).calibrate("heston", art)


def run_date(iso):
    art = IvSurfaceArtifact.from_file(IV / f"mo_iv_surface_{iso.replace('-','')}.json")
    inception = art.trade_date
    calendar = gate.TradingCalendar.from_spot_csv(HIST / "csi1000_spot.csv")
    terms = gate.build_snowball_terms(inception, calendar)
    env = gate.build_pricing_env(art, gate.FLAT_RATE)
    product = gate.build_snowball_product(terms, float(art.s0))
    notional = float(art.s0)
    T = terms.maturity_years
    n_x, n_v = gate.LADDER_MEDIUM
    n_t = math.ceil(400.0 * T)

    out = {"date": iso, "T": T, "notional": notional, "grid": [n_x, n_v, n_t]}
    for label, enforce in (("soft", False), ("hard", True)):
        model = _calibrate(art, enforce)
        p = model.heston_params
        cell = {"ratio": 2.0 * p.kappa * p.theta / p.sigma**2,
                "kappa": p.kappa, "theta": p.theta, "sigma": p.sigma,
                "v0": p.v0, "rho": p.rho,
                "rmse_iv": model.record["overall_rmse_iv"]}

        t0 = time.perf_counter()
        pde = gate._make_pde_engine("heston", model, PDEParams(), (n_x, n_v, n_t))
        cell["pde"] = float(pde.price(product, env))
        cell["pde_secs"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        mcp = gate._make_mc_params(gate.MC_FULL, gate.SEED)
        mc = gate._make_mc_engine(
            "heston", model, mcp, gate.MC_FULL["substeps_per_interval"])
        cell["mc"] = float(mc.price(product, env))
        cell["mc_se"] = float(mc.get_last_std_error())
        cell["mc_secs"] = time.perf_counter() - t0

        cell["diff_pct"] = 100.0 * (cell["pde"] - cell["mc"]) / notional
        cell["mc_se_pct"] = 100.0 * cell["mc_se"] / notional
        cell["tol_pct"] = gate.gate_tolerance_pct(cell["mc_se_pct"])
        cell["passed"] = gate.gate_cell_passed(cell["diff_pct"], cell["mc_se_pct"])
        out[label] = cell
        print(f"  {iso} {label}: ratio={cell['ratio']:.4f} "
              f"diff={cell['diff_pct']:+.3f}% tol={cell['tol_pct']:.3f}% "
              f"{'PASS' if cell['passed'] else 'FAIL'} "
              f"(pde {cell['pde_secs']:.0f}s / mc {cell['mc_secs']:.0f}s)", flush=True)
    return out


if __name__ == "__main__":
    dest, dates = sys.argv[1], sys.argv[2:]
    rows = []
    for iso in dates:
        rows.append(run_date(iso))
        Path(dest).write_text(json.dumps(rows, indent=1))
