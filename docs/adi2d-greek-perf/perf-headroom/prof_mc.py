import cProfile, pstats, sys
sys.path.insert(0, 'docs/mc-reference-convergence')
from demo_common import batch_deltas
pr = cProfile.Profile(); pr.enable()
batch_deltas("ordinary_decayed", "heston_slv", batches=2, seed=7, bridge_dimensions=1, workers=1)
pr.disable()
st = pstats.Stats(pr); st.sort_stats("tottime")
rows = [(tt, nc, f"{f[0].split('/')[-1]}:{f[1]}({f[2]})")
        for f, (cc, nc, tt, ct, callers) in st.stats.items()]
rows.sort(reverse=True)
print(f"{'tottime':>8} {'ncalls':>9}  where")
for tt, nc, name in rows[:14]:
    print(f"{tt:8.3f} {nc:9d}  {name}")
