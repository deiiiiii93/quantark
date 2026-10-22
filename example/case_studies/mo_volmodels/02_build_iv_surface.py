"""Stage 02 — put-call parity + OTM filter + Black-IV inversion -> GridVolSurface.

Run: .venv/bin/python example/mo_volmodels/02_build_iv_surface.py [--snapshot sample|latest|YYYYMMDD]

Recovers r(T), forward F(T), carry q(T) per expiry from parity, selects OTM quotes,
inverts them to Black implied vols, and assembles a rectangular strike x maturity grid.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _mo_common as mc  # noqa: E402

MIN_STRIKES = 5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="latest",
                    help="sample | latest | a YYYYMMDD stamp (default: latest)")
    args = ap.parse_args()
    snap_path = HERE / f"data/mo_snapshot_{args.snapshot}.json"
    snap = mc.load_snapshot(snap_path)
    s0 = float(snap["underlying"]["spot"])
    if not snap.get("market_open", True):
        print("WARNING: snapshot taken while market closed — IVs from last-price mids.")
    print(f"underlying 000852.SH spot = {s0:.2f}  ({snap['fetched_at']})")

    per_expiry, smile_rows = [], []
    for sl in mc.iter_expiries(snap):
        try:
            par = mc.imply_forward_and_rate(sl, s0)
        except ValueError as e:
            print(f"  skip {sl.expiry_date}: {e}")
            continue
        pts = []
        for oq in mc.select_otm(sl, par.forward):
            iv = mc.otm_implied_vol(oq, s0, par.r, par.q, par.forward, par.discount_factor, sl.T)
            if iv is None or not (0.0 < iv < 2.0):
                continue
            pts.append((oq.strike, iv))
        if len(pts) < MIN_STRIKES:
            print(f"  skip {sl.expiry_date}: only {len(pts)} usable strikes (<{MIN_STRIKES})")
            continue
        pts.sort()
        per_expiry.append({
            "expiry_date": sl.expiry_date, "T": sl.T, "r": par.r, "q": par.q,
            "forward": par.forward, "df": par.discount_factor, "points": pts,
        })
        print(f"  T={sl.T:.3f}  r={par.r*100:6.2f}%  q={par.q*100:6.2f}%  "
              f"F={par.forward:8.1f}  {len(pts)} OTM strikes")

    if len(per_expiry) < 2:
        sys.exit("need >=2 usable expiries to build a surface")

    # Exchange data arrives unsorted by maturity — GridVolSurface needs strictly increasing T.
    per_expiry.sort(key=lambda pe: pe["T"])
    for i in range(len(per_expiry) - 1):
        if per_expiry[i + 1]["T"] <= per_expiry[i]["T"]:
            sys.exit(f"duplicate maturity {per_expiry[i]['T']} — dedupe upstream")

    # Common rectangular strike grid: strikes present (near-exactly) in >=2 expiries;
    # each expiry's row is filled by linear interpolation of its own smile (flat at wings).
    all_strikes = sorted({k for pe in per_expiry for k, _ in pe["points"]})

    def _count(k):
        return sum(any(abs(k - kk) < 1e-6 for kk, _ in pe["points"]) for pe in per_expiry)

    strikes = [k for k in all_strikes if _count(k) >= 2]
    maturities = [pe["T"] for pe in per_expiry]
    grid = np.empty((len(maturities), len(strikes)))
    for i, pe in enumerate(per_expiry):
        ks = np.array([k for k, _ in pe["points"]])
        vs = np.array([v for _, v in pe["points"]])
        grid[i] = np.interp(strikes, ks, vs)  # flat extrapolation past this expiry's wings
        smile_rows.append((f"T={pe['T']:.2f}", [k for k, _ in pe["points"]], [v for _, v in pe["points"]]))

    out = {"s0": s0, "fetched_at": snap["fetched_at"], "strikes": strikes,
           "maturities": maturities, "iv_grid": grid.tolist(), "per_expiry": per_expiry}
    # Artifacts are keyed by the snapshot tag so the deterministic 'sample' pipeline
    # (tests) and the live 'latest' pipeline (lecture) never clobber each other.
    (HERE / f"data/mo_iv_surface_{args.snapshot}.json").write_text(json.dumps(out, indent=2))
    mc.plot_smiles(smile_rows, HERE / f"data/plots/02_smiles_{args.snapshot}.png",
                   title=f"MO (000852.SH) implied-vol smiles — spot {s0:.0f}")
    print(f"surface: {len(maturities)} maturities x {len(strikes)} strikes "
          f"-> data/mo_iv_surface_{args.snapshot}.json")


if __name__ == "__main__":
    main()
