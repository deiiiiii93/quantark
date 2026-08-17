"""Does the published aggregate gate discard banked over-allocation?

Stage-16's aggregate delta-bias gate takes the mean across the seven regime
cells *within* each scramble, which correctly preserves the cross-case
covariance before the outer standard error is measured.  It aligns the cells by
TRUNCATING each one to the common scramble count::

    row["batch_difference_contracts"]["delta"][:common_batches]

Two cells in ``output/p17_fixed`` were deliberately allocated more than the
common count, because the pilot-frozen Neyman allocation identified them as
variance-dominant: ``heston_slv/low_feller`` banked 512 against a common 128,
and ``heston/near_ki`` banked 2048 against a common 1024.  Truncation means the
extra rows -- 384 of them on the single most expensive cell in the fleet, which
cost 24.5 hours -- never reach the aggregate gate at all.

Stage-17 aligns the same cohorts by POOLING instead (``_group_delta_rows``
reshapes to the output count and averages), which keeps every banked row and
lowers the over-allocated cell's contribution to the row variance.  Both
estimators are unbiased and both keep the 128 outer rows independent, so the
outer standard error stays valid either way; pooling simply measures a
lower-variance estimator.

This probe answers, from banked evidence and with no re-run, how much the
aggregate margin would widen under pooling.  It first REPRODUCES the published
gate exactly -- if that reproduction does not match to the last bit, nothing
below it is trustworthy and the probe fails.

Usage:
    PYTHONPATH=$PWD .venv/bin/python \
      docs/adi2d-greek-perf/probes/probe_aggregate_pooling_headroom.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[3]
STAGE16_RELATIVE = "example/mo_volmodels/16_adi_greek_certification.py"
DEFAULT_EVIDENCE = ROOT / "output" / "p17_fixed" / "adi_greek_certification.json"

# The published common scramble count per variant, read back from the payload
# rather than assumed, so a different parent cannot silently change the meaning.
VARIANTS = ("heston", "heston_slv")


def load_stage16():
    """Import the stage-16 harness so its own gate function does the arithmetic.

    Registering the module in ``sys.modules`` before executing it is required:
    its frozen dataclasses resolve their own module during ``@dataclass``.
    """
    spec = importlib.util.spec_from_file_location(
        "s16_pooling_probe", ROOT / STAGE16_RELATIVE
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cell_rows(cell: dict) -> tuple[np.ndarray, np.ndarray]:
    delta = np.asarray(cell["batch_difference_contracts"]["delta"], dtype=float)
    substep = np.asarray(
        cell["certifications"]["delta"]["reference_substep_batch_contracts"],
        dtype=float,
    )
    if delta.shape != substep.shape:
        raise ValueError("delta and substep row counts disagree for a cell")
    return delta, substep


def align_truncate(rows: np.ndarray, common: int) -> np.ndarray:
    """Stage-16's alignment: keep the leading ``common`` rows, drop the rest."""
    return rows[:common]


def align_pool(rows: np.ndarray, common: int) -> np.ndarray:
    """Stage-17's alignment: average consecutive groups down to ``common`` rows.

    Groups are disjoint, so the ``common`` output rows stay mutually
    independent and the outer standard error remains valid. The estimator is
    unbiased for the same expectation; its variance is lower because an
    over-allocated cell contributes the mean of ``g`` scrambles instead of one.
    """
    if rows.size % common:
        raise ValueError(f"{rows.size} banked rows do not divide into {common} groups")
    return rows.reshape(common, rows.size // common).mean(axis=1)


def evaluate_gate(stage16, delta_by_case, substep_by_case, *, pde_envelope, label):
    """Run stage-16's own signed-bias gate over an aligned cohort."""
    confidence = stage16.STOCHASTIC_COMPONENT_CONFIDENCE
    aggregate_delta = np.mean(
        np.asarray([delta_by_case[name] for name in delta_by_case], dtype=float), axis=0
    )
    aggregate_substep = np.mean(
        np.asarray([substep_by_case[name] for name in substep_by_case], dtype=float),
        axis=0,
    )
    substep_mean = float(np.mean(aggregate_substep))
    substep_se = float(
        np.std(aggregate_substep, ddof=1) / np.sqrt(aggregate_substep.size)
    )
    substep_half_width = float(
        student_t.ppf(0.5 + 0.5 * confidence, aggregate_substep.size - 1) * substep_se
    )
    reference_bias_envelope = abs(substep_mean) + substep_half_width
    return stage16.certify_signed_bias_from_batches(
        aggregate_delta,
        stage16.DELTA_BIAS_BOUND_CONTRACTS,
        pde_discretization_envelope=pde_envelope,
        reference_bias_envelope=reference_bias_envelope,
        confidence=confidence,
        label=label,
    ).as_dict()


def describe(bound: float, gate: dict) -> str:
    low, high = gate["interval"]
    margin = min(bound - abs(low), bound - abs(high))
    return (
        f"est {gate['estimate_difference']:+.6f}  unc {gate['total_uncertainty']:.6f}  "
        f"interval [{low:+.6f}, {high:+.6f}]  margin {margin:.6f} "
        f"({100.0 * margin / bound:.1f}% of bound)  {gate['status']}"
    )


def margin_of(bound: float, gate: dict) -> float:
    low, high = gate["interval"]
    return min(bound - abs(low), bound - abs(high))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    args = parser.parse_args(argv)

    evidence = json.loads(Path(args.evidence).read_text())
    stage16 = load_stage16()
    bound = stage16.DELTA_BIAS_BOUND_CONTRACTS
    print(f"evidence      {args.evidence}")
    print(f"schema        {evidence.get('schema_version')}")
    print(f"bound         +/-{bound}  component confidence "
          f"{stage16.STOCHASTIC_COMPONENT_CONFIDENCE}")

    failures = []
    summary = {}
    for variant in VARIANTS:
        published = evidence["decisions"][variant]["delta_bias"]
        common = int(published["aggregate_common_scrambles"])
        pde_envelope = float(published["pde_discretization_envelope"])
        cells = {
            cell["case"]["name"]: cell
            for cell in evidence["cells"]
            if cell["variant"] == variant
        }
        banked = {name: cell_rows(cell) for name, cell in cells.items()}
        over = {
            name: rows[0].size
            for name, rows in banked.items()
            if rows[0].size > common
        }
        print(f"\n=== {variant}   common scrambles {common} ===")
        if over:
            for name, size in sorted(over.items()):
                print(
                    f"  over-allocated {name}: {size} banked rows, "
                    f"{size - common} unused under truncation "
                    f"({100.0 * (size - common) / size:.0f}% of the cell)"
                )
        else:
            print("  no cell is over-allocated; the two alignments coincide")

        aligned = {}
        for label, align in (("truncated", align_truncate), ("pooled", align_pool)):
            aligned[label] = evaluate_gate(
                stage16,
                {name: align(rows[0], common) for name, rows in banked.items()},
                {name: align(rows[1], common) for name, rows in banked.items()},
                pde_envelope=pde_envelope,
                label=f"{variant}/mean_signed_delta_bias",
            )

        # The reproduction gate: truncation must rebuild the published numbers.
        reproduced = all(
            abs(aligned["truncated"][key] - published[key]) <= 0.0
            for key in ("estimate_difference", "total_uncertainty")
        )
        print(f"  reproduces published gate exactly: {reproduced}")
        if not reproduced:
            for key in ("estimate_difference", "total_uncertainty"):
                print(
                    f"    {key}: rebuilt {aligned['truncated'][key]!r} "
                    f"published {published[key]!r}"
                )
            failures.append(variant)
            continue

        for label in ("truncated", "pooled"):
            print(f"  {label:10s} {describe(bound, aligned[label])}")
        truncated_margin = margin_of(bound, aligned["truncated"])
        pooled_margin = margin_of(bound, aligned["pooled"])
        widening = (
            pooled_margin / truncated_margin if truncated_margin > 0.0 else float("inf")
        )
        print(
            f"  margin {truncated_margin:.6f} -> {pooled_margin:.6f}  "
            f"({widening:.2f}x)   statistical half-width "
            f"{aligned['truncated']['reference_half_width']:.6f} -> "
            f"{aligned['pooled']['reference_half_width']:.6f}"
        )
        summary[variant] = {
            "common_scrambles": common,
            "over_allocated_rows": over,
            "truncated": aligned["truncated"],
            "pooled": aligned["pooled"],
            "truncated_margin": truncated_margin,
            "pooled_margin": pooled_margin,
            "margin_widening": widening,
        }

    if failures:
        print(f"\nFAILED to reproduce the published gate for: {', '.join(failures)}")
        return 1

    out = ROOT / "output" / "aggregate_pooling_headroom"
    out.mkdir(parents=True, exist_ok=True)
    (out / "headroom.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True, default=float) + "\n"
    )
    print(f"\nwrote {out / 'headroom.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
