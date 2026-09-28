"""Did the whole-cell fix clear the fleet audit?

Compares the re-run against the banked Gate E values state by state. The
banked column IS the option-off case, and the harness was validated against
it first: re-running 242 states with the option off reproduced every banked
residual to 4.8e-10 hands, against a 0.01 budget.

Reports in both units. A state is one (inception, model, date); the fleet
recorded it once per hedge policy, seven times, because the identity is
holdings-free.
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SCRATCH = Path("/private/tmp/claude-501/-Users-fuxinyao-quant-ark--claude-worktrees"
               "-hedge")
HERE = Path(__file__).parent
RUNS = Path("/Users/fuxinyao/quant-ark/.claude/worktrees/bucket-futures-hedge"
            "/example/snowball_q_term_structure/data/bucket_hedge_v2/gate_e/runs")
BUDGET = 0.01
CELLS_PER_STATE = 7
#: the contract shape: KI at 75% and KO at 103% of the inception spot, so the
#: alignment target changes hands at their geometric mean
CROSS_RATIO = (0.75 * 1.03) ** 0.5

frame = pd.read_csv(HERE / (sys.argv[1] if len(sys.argv) > 1 else "audit_on.csv"))
print(f"{len(frame)} states re-run, {(frame.status == 'error').sum()} errors")
frame = frame[frame.status != "error"].copy()

frame["gate_on"] = frame.residual.abs() + frame.refinement
frame["gate_off"] = frame.banked_residual.abs() + frame.banked_refinement
frame["fail_on"] = frame.gate_on > BUDGET
frame["fail_off"] = frame.gate_off > BUDGET

print("\n=== the gate, |R| + E against 0.01 hands ===")
print(f"{'':22} {'option OFF (banked)':>21} {'option ON':>14}")
print(f"{'states':22} {len(frame):21d} {len(frame):14d}")
print(f"{'breaches':22} {int(frame.fail_off.sum()):21d} "
      f"{int(frame.fail_on.sum()):14d}")
print(f"{'implied cell-rows':22} {int(frame.fail_off.sum()) * CELLS_PER_STATE:21d} "
      f"{int(frame.fail_on.sum()) * CELLS_PER_STATE:14d}")
print(f"{'worst |R|+E':22} {frame.gate_off.max():21.6f} {frame.gate_on.max():14.6f}")
print(f"{'margin vs budget':22} {BUDGET / frame.gate_off.max():20.2f}x "
      f"{BUDGET / frame.gate_on.max():13.2f}x")
print(f"{'worst |R|':22} {frame.banked_residual.abs().max():21.6f} "
      f"{frame.residual.abs().max():14.6f}")
print(f"{'median |R|':22} {frame.banked_residual.abs().median():21.3e} "
      f"{frame.residual.abs().median():14.3e}")

print("\n=== the states that breached before ===")
was = frame[frame.fail_off]
if len(was):
    cols = ["inception", "model", "date", "spot", "gate_off", "gate_on", "fail_on"]
    print(was[cols].to_string(index=False))

print("\n=== any state that breaches now ===")
now = frame[frame.fail_on]
print("none" if not len(now) else
      now[["inception", "model", "date", "spot", "gate_off", "gate_on"]].to_string(index=False))

print("\n=== worst ten states after the fix ===")
worst = frame.nlargest(10, "gate_on")
print(worst[["inception", "model", "date", "spot", "gate_off", "gate_on"]].to_string(index=False))

# ---- did it help everywhere, or only at the crossover? --------------------
s0 = {}
for inception in sorted(p for p in RUNS.iterdir() if p.is_dir()):
    cell = next(p for p in inception.iterdir() if p.is_dir())
    s0[inception.name] = json.loads((cell / "run_summary.json").read_text())["s0"]

frame["cross"] = frame.inception.astype(str).map(lambda k: s0[k] * CROSS_RATIO)
frame["dist_bumps"] = (frame.spot / frame.cross - 1.0).abs() / 0.00025
live = frame[~frame.knocked_in]

print(f"\n=== by distance from the alignment crossover ({len(live)} pre-knock-in states) ===")
print(f"{'distance':22} {'states':>7} {'worst |R| off':>14} {'worst |R| on':>13} "
      f"{'breaches off':>13} {'breaches on':>12}")
for lo, hi in ((0, 2), (2, 10), (10, 50), (50, 200), (200, 1e9)):
    band = live[(live.dist_bumps >= lo) & (live.dist_bumps < hi)]
    if band.empty:
        continue
    label = f"{lo:.0f}-{hi:.0f} bumps" if hi < 1e9 else f"over {lo:.0f} bumps"
    print(f"{label:22} {len(band):7d} {band.banked_residual.abs().max():14.6f} "
          f"{band.residual.abs().max():13.6f} {int(band.fail_off.sum()):13d} "
          f"{int(band.fail_on.sum()):12d}")

print("\n=== did the fix cost anything away from the crossover? ===")
far = live[live.dist_bumps >= 50]
worse = far[far.gate_on > far.gate_off]
print(f"states 50+ bumps away: {len(far)}   worse with the option on: {len(worse)} "
      f"({len(worse) / max(len(far), 1):.1%})")
print(f"  worst increase there: {(far.gate_on - far.gate_off).max():.3e} hands")
print(f"  mean |R| off {far.banked_residual.abs().mean():.3e} -> "
      f"on {far.residual.abs().mean():.3e}")
