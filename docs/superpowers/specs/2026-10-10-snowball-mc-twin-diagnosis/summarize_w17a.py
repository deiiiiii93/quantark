"""W17a moved values: diff measure_w17a.py on c02296cb (w17a_base_c02296cb.txt) against the branch
(w17a_after_branch.txt). Prints every moved row, old -> new, the pinning test, and whether the move is expected (a
row whose c02296cb grid held a sub-tolerance step, recomputed here from the W17 grid scan's rule)."""
import sys


def load(path):
    rows = {}
    for line in open(path):
        if line.startswith("#") or " | " not in line:
            continue
        cell, form, value = (s.strip() for s in line.split(" | ", 2))
        rows[(cell, form)] = value
    return rows


#: c02296cb grids with a sub-tolerance step (W17 grid_scan.txt; KO-reset from the same scan; the twin routes share
#: the snowball / Phoenix grid builders)
COLLIDED = {
    "snowball": {"dated": {14, 53, 61, 105, 144, 209}, "numeric": {41, 61, 92, 170, 183, 209, 248, 261}},
    "phoenix": {"dated": {14, 53, 61, 105, 144, 209}, "numeric": {41, 61, 92, 170, 183, 209, 248, 261}},
    "ko_reset_continuous_ki": {"dated": {26}, "numeric": {26, 65, 78, 82}},
}


def family(cell):
    for name in ("ko_reset_continuous_ki", "snowball", "phoenix"):
        if name in cell:
            return name
    return None


def pinning_test(cell):
    if cell.startswith("twin:bs_ko_reset"):
        return "test_mc_time_node_merge.py::test_ko_reset_continuous_ki_twin_on_collision_days"
    if cell.startswith("twin:"):
        return "test_mc_time_node_merge.py::test_dated_twin_equals_numeric_on_collision_days"
    if cell.startswith("event_stats:"):
        return "test_ko_reset_snowball_mc_engine.py::test_ko_reset_event_stats_shape[Rebased] (shape only)"
    return "none (measured only)"


if __name__ == "__main__":
    base, after = load(sys.argv[1]), load(sys.argv[2])
    assert base.keys() == after.keys(), "row sets differ"
    moved = unexpected = 0
    print(f"rows {len(base)}")
    for key in base:
        if base[key] == after[key]:
            continue
        moved += 1
        cell, form = key
        fam = family(cell)
        day = int(cell.rsplit("-d", 1)[1]) if "-d" in cell else None
        expected = (fam is not None and day is not None and day in COLLIDED[fam].get(form, set())) or \
            cell.startswith("event_stats:")
        unexpected += not expected
        try:
            delta = f"{float(after[key]) - float(base[key]):+.6f}"
        except ValueError:
            delta = "n/a"
        print(f"{cell} | {form} | {base[key]} -> {after[key]} | delta {delta} | "
              f"{'expected (collision grid)' if expected else 'UNEXPECTED'} | {pinning_test(cell)}")
    print(f"moved {moved} of {len(base)}; unexpected {unexpected}")

    # the twin gap |dated - numeric| per family, before and after
    for label, rows in (("c02296cb", base), ("branch", after)):
        gaps = {}
        for (cell, form), value in rows.items():
            if form != "dated" or (cell, "numeric") not in rows:
                continue
            try:
                g = abs(float(value) - float(rows[(cell, "numeric")]))
            except ValueError:
                continue
            fam = cell.split("-d")[0]
            worst = gaps.get(fam, (0.0, None, 0))
            gaps[fam] = (max(worst[0], g), cell if g >= worst[0] else worst[1],
                         worst[2] + (g > 1e-10 * max(1.0, abs(float(value)))))
        print(f"twin gaps on {label}:")
        for fam, (g, cell, n) in sorted(gaps.items()):
            print(f"   {fam:40s} worst |dated - numeric| {g:.3e} ({cell}); pairs beyond rel 1e-10: {n}")
