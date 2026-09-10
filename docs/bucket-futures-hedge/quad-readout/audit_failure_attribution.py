"""Which of the carry audit's four checks actually breaches?

The audit fails a date if ANY of four checks breaches: the net delta against
the held book, the coordinate identity, the parallel rhoq, or any nodal
rhoq.  Only the identity involves no holdings at all --

    identity_residual = (delta_q - delta_f_direct - sum_i (F_i/S) B_i) / m_ref

-- so it must be the same number for every hedge policy sharing a q-model,
while the other three move with the book.  Splitting the failures says
whether the bucket hedge is implicated at all.

Usage:  python audit_failure_attribution.py [run_dir ...]
        default: the subset and subset_transition runs of bucket_hedge_v2.
"""
from __future__ import annotations

import csv
import pathlib
import sys

DEFAULT_ROOTS = (
    "example/snowball_q_term_structure/data/bucket_hedge_v2/subset",
    "example/snowball_q_term_structure/data/bucket_hedge_v2/subset_transition",
)
DELTA_TOL_HANDS = 0.01
RHOQ_TOL_BP = 0.01


def _f(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def attribute(cell_dir: pathlib.Path) -> dict[str, int] | None:
    attribution = cell_dir / "hedge_attribution.csv"
    if not attribution.exists():
        return None

    nodal: dict[str, list[float]] = {}
    legs = cell_dir / "hedge_legs.csv"
    if legs.exists():
        for row in csv.DictReader(open(legs)):
            error = _f(row.get("rhoq_audit_error_bp"))
            if error is not None:
                nodal.setdefault(row["date"], []).append(abs(error))

    tally = {
        "measured": 0, "failed": 0, "identity": 0, "net delta": 0,
        "parallel rhoq": 0, "nodal rhoq": 0, "identity alone": 0,
    }
    for row in csv.DictReader(open(attribution)):
        identity = _f(row["identity_residual_hands"])
        if identity is None:
            continue
        tally["measured"] += 1
        net_delta = _f(row["net_delta_audit_error_hands"])
        parallel = _f(row["parallel_rhoq_audit_error_bp"])

        breached = set()
        if abs(identity) > DELTA_TOL_HANDS:
            breached.add("identity")
        if net_delta is not None and abs(net_delta) > DELTA_TOL_HANDS:
            breached.add("net delta")
        if parallel is not None and abs(parallel) > RHOQ_TOL_BP:
            breached.add("parallel rhoq")
        if any(v > RHOQ_TOL_BP for v in nodal.get(row["date"], ())):
            breached.add("nodal rhoq")

        for name in breached:
            tally[name] += 1
        if breached:
            tally["failed"] += 1
        if breached == {"identity"}:
            tally["identity alone"] += 1
    return tally


def main(argv: list[str]) -> None:
    roots = [pathlib.Path(a) for a in argv] or [pathlib.Path(r) for r in DEFAULT_ROOTS]
    header = (f"{'run / cell':58s} {'meas':>5s} {'fail':>5s} {'ident':>6s} "
              f"{'delta':>6s} {'par':>5s} {'nodal':>6s} {'ident alone':>12s}")
    print(header)
    print("-" * len(header))
    for root in roots:
        for cell in sorted(root.glob("runs/*/*")):
            tally = attribute(cell)
            if tally is None:
                continue
            label = f"{root.name}/{cell.name}"
            print(f"{label:58s} {tally['measured']:5d} {tally['failed']:5d} "
                  f"{tally['identity']:6d} {tally['net delta']:6d} "
                  f"{tally['parallel rhoq']:5d} {tally['nodal rhoq']:6d} "
                  f"{tally['identity alone']:12d}")


if __name__ == "__main__":
    main(sys.argv[1:])
