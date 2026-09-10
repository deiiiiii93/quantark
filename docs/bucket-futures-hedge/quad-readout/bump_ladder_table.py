"""The bump ladder, both readouts, from whatever runs are on disk."""
import csv
import json
import pathlib
import statistics

ROOT = pathlib.Path(
    "/Users/fuxinyao/quant-ark/.claude/worktrees/bucket-futures-hedge/"
    "example/snowball_q_term_structure/data/bucket_hedge_v2"
)
CELL = "runs/20230504/term_flat_q__front"

print(f"{'run':22s} {'readout':12s} {'bump':>8s} {'fail':>5s} {'pass':>5s} "
      f"{'mean|id|':>10s} {'max|id|':>10s} {'mean id':>11s}")
for run in sorted(p.name for p in ROOT.iterdir() if p.is_dir()):
    d = ROOT / run / CELL
    att, cfgp = d / "hedge_attribution.csv", d / "run_config.json"
    if not att.exists() or not cfgp.exists():
        continue
    cfg = json.loads(cfgp.read_text())
    bump = cfg["risk"]["audit_spot_bump_rel"]
    summ = json.loads((d / "audit_summary.json").read_text())["by_status"]
    vals = []
    for r in csv.DictReader(open(att)):
        try:
            vals.append(float(r["identity_residual_hands"]))
        except (TypeError, ValueError):
            pass
    a = [abs(v) for v in vals]
    print(f"{run:22s} {str(cfg.get('quad_readout')):12s} {str(bump):>8s} "
          f"{summ.get('fail', 0):5d} {summ.get('pass', 0):5d} "
          f"{statistics.fmean(a):10.6f} {max(a):10.6f} "
          f"{statistics.fmean(vals):11.2e}")
