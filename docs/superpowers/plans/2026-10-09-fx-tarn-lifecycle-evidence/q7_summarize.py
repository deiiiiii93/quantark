"""Summarize the Q7 plugin logs into moved_values.txt lines (one per distinct test / engine / product / step)."""
import glob
import json
import os
import sys

rows = {}
for path in glob.glob(os.path.join(sys.argv[1], "*.jsonl")):
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            key = (r["nodeid"], r["engine"], r["product"], r["convention"], r["valuation_date"], r["new_dt"])
            rows[key] = r
print(f"# Q7 numeric FX theta moves: {len(rows)} distinct calls (old step theta_days/365 -> env year fraction)")
for key in sorted(rows):
    r = rows[key]
    print(f"Q7 | {r['nodeid']} | {r['engine']} / {r['product']} | {r['convention']} @ {r['valuation_date']} "
          f"| dt {r['old_dt']!r} -> {r['new_dt']!r} | theta {r['old_theta']!r} -> {r['new_theta']!r}")
