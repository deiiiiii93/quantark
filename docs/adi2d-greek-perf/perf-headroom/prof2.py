import cProfile, pstats, sys
sys.path.insert(0, 'test')
from test_adi_semi_lagrangian import _european_put
pr = cProfile.Profile(); pr.enable(); _european_put(120, "semi_lagrangian"); pr.disable()
st = pstats.Stats(pr)
st.sort_stats("tottime")
rows = []
for func, (cc, nc, tt, ct, callers) in st.stats.items():
    rows.append((tt, nc, f"{func[0].split('/')[-1]}:{func[1]}({func[2]})"))
rows.sort(reverse=True)
print(f"{'tottime':>8} {'ncalls':>8}  where")
for tt, nc, name in rows[:14]:
    print(f"{tt:8.3f} {nc:8d}  {name}")
