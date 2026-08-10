"""Scheme/n_v movement probe for heston/low_feller (P1.1 of the 2026-08-10 plan).

The schema-11 banked evidence for this cell was lost in the 2026-08-10 crash, so
unlike probe_delta_attribution.py this driver has no bitwise fidelity gate. It
measures the MOVEMENT of the production PDE delta under scheme and grid variants,
which is exact and reference-free; the recorded schema-11 gap for this cell
(-0.107 +/- 0.031 contracts, 2026-08-07 analysis) anchors interpretation:
if a variant moves delta by ~+0.107/scale, the bias was discretization; if delta
barely moves (the standing prediction), the low_feller bias has another origin.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location(
    "cert16", ROOT / "example" / "mo_volmodels" / "16_adi_greek_certification.py"
)
assert _spec is not None and _spec.loader is not None
cert = importlib.util.module_from_spec(_spec)
sys.modules["cert16"] = cert
_spec.loader.exec_module(cert)

HEDGE_INCEPTION_SPOT = 4532.52  # recovered anchor, scale-verified bitwise
RECORDED_GAP_CONTRACTS = -0.107  # schema-11 heston/low_feller, 2026-08-07 analysis

ROWS = [
    ("target", 300, 135, 3200, "adaptive_upwind"),
    ("centered", 300, 135, 3200, "centered"),
    ("nv270", 300, 270, 3200, "adaptive_upwind"),
    ("joint2x", 600, 270, 6400, "adaptive_upwind"),
]


def make_engine(case, n_x: int, n_v: int, n_t: int, scheme: str):
    controls = dict(cert.PRODUCTION_ENGINE_CONTROLS)
    controls.update(
        barrier_greek_steps_per_tick=0,
        greek_min_n_x=0,
        greek_min_n_v=0,
        greek_min_steps_per_year=0,
        barrier_greek_min_n_x=0,
        v_drift_scheme=scheme,
    )
    return cert.HestonSnowballPDESolver(
        case.params,
        n_x=n_x,
        n_v=n_v,
        n_t=n_t,
        params=cert.PDEParams(cache_enabled=False),
        **controls,
    )


def main() -> None:
    case = next(c for c in cert.certification_cases(quick=False) if c.name == "low_feller")
    product = cert.make_snowball(case, dense_ki=True)
    import math

    env = cert.make_environment(case.spot, math.sqrt(max(case.params.v0, case.params.theta)))
    scale = cert.EconomicGreekScale(
        model_spot=case.spot,
        hedge_inception_spot=HEDGE_INCEPTION_SPOT,
        study_notional=cert.STUDY_NOTIONAL,
        hedge_multiplier=cert.HEDGE_MULTIPLIER,
    )
    out = Path("output/delta_bias_attribution/heston_low_feller_movement.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    target_delta = None
    for name, n_x, n_v, n_t, scheme in ROWS:
        engine = make_engine(case, n_x, n_v, n_t, scheme)
        started = time.perf_counter()
        values = cert.central_bump_greeks(engine, product, env, cert.SPOT_BUMP)
        seconds = time.perf_counter() - started
        if name == "target":
            target_delta = values["delta"]
        move = scale.delta_contracts(values["delta"] - target_delta)
        record = {
            "variant": "heston",
            "case": "low_feller",
            "row": name,
            "n_x": n_x,
            "n_v": n_v,
            "n_t": n_t,
            "scheme": scheme,
            "delta": values["delta"],
            "gamma": values["gamma"],
            "price": values["price"],
            "seconds": round(seconds, 2),
            "delta_move_vs_target_contracts": move,
            "implied_gap_contracts_if_recorded_anchor": RECORDED_GAP_CONTRACTS + move,
        }
        with out.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
