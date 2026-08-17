"""Is cross-cell CRN coupling material, and does strided grouping fix pooling's SE?

Follow-on to ``probe_aggregate_pooling_headroom.py``.  That probe showed the
full-evidence (pooled) aggregate flips ``heston_slv`` to INCONCLUSIVE.  Its
pooling groups an over-allocated cell's rows CONSECUTIVELY::

    rows.reshape(common, g).mean(axis=1)   # outer row j <- scrambles gj..gj+g-1

Under the CRN premise that justified truncation in the first place -- rows at
the SAME scramble index are correlated ACROSS cells -- consecutive grouping
has a defect: outer row j carries the common cells at scramble j, but the
over-allocated cell's scramble-j row lands in outer row floor(j/g).  The
cross-cell covariance therefore straddles outer-row boundaries, the outer rows
are not independent, and (to leading order in 1/common) the empirical standard
error estimates the aggregate variance AS IF the over-cell cross covariance
were zero.

Grouping by STRIDE instead::

    rows.reshape(g, common).mean(axis=0)   # outer row j <- scrambles j, j+m, ...

keeps every same-scramble coupling inside one outer row: the outer rows are
genuinely i.i.d., and their empirical variance estimates the full-evidence
estimator's true variance INCLUDING the partial CRN overlap -- the "unequal
allocation aggregate" the stage-16 result note asks for.  Both groupings share
the identical point estimate; only the standard error can differ, and the
predicted difference is exactly the over-cell cross-covariance term

    (2 / (49 * n_over)) * sum_{c != over} gamma(c, over)   per over-allocated cell

where ``gamma`` is the per-scramble cross-cell covariance, measured here
directly from the banked shared-prefix rows.

The probe reports, per variant:

1. pairwise cross-cell covariance/correlation (is CRN coupling material?),
2. a plug-in decomposition of the full-evidence aggregate variance
   (within-cell / cross-cell / over-cell share),
3. empirical aggregate variances under consecutive and strided alignment
   against the plug-in prediction,
4. the truncated / consecutive / strided gate verdicts side by side.

It first REPRODUCES the published (truncated) gate bit-for-bit and fails if it
cannot; every other number is reporting, not a verdict.

Usage:
    PYTHONPATH=$PWD .venv/bin/python \
      docs/adi2d-greek-perf/probes/probe_crn_strided_alignment.py
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

VARIANTS = ("heston", "heston_slv")

# Point estimates of consecutive and strided grouping are algebraically equal
# (both are the mean of all banked rows); this bounds their float-order drift.
ESTIMATE_AGREEMENT_RTOL = 1e-12


def load_stage16():
    """Import the stage-16 harness so its own gate function does the arithmetic.

    Registering the module in ``sys.modules`` before executing it is required:
    its frozen dataclasses resolve their own module during ``@dataclass``.
    """
    spec = importlib.util.spec_from_file_location(
        "s16_strided_probe", ROOT / STAGE16_RELATIVE
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


def group_factor(rows: np.ndarray, common: int) -> int:
    if rows.size % common:
        raise ValueError(f"{rows.size} banked rows do not divide into {common} groups")
    return rows.size // common


def align_truncate(rows: np.ndarray, common: int) -> np.ndarray:
    """Stage-16's alignment: keep the leading ``common`` rows, drop the rest."""
    return rows[:common]


def align_consecutive(rows: np.ndarray, common: int) -> np.ndarray:
    """Stage-17's pooling: outer row j averages scrambles gj..gj+g-1."""
    g = group_factor(rows, common)
    return rows.reshape(common, g).mean(axis=1)


def align_strided(rows: np.ndarray, common: int) -> np.ndarray:
    """CRN-preserving pooling: outer row j averages scrambles j, j+m, j+2m, ...

    Scramble j (the only index the other cells also hold) stays in outer row j,
    so every cross-cell same-scramble coupling is contained inside one outer
    row and the outer rows are mutually independent.
    """
    g = group_factor(rows, common)
    return rows.reshape(g, common).mean(axis=0)


ALIGNMENTS = (
    ("truncated", align_truncate),
    ("consecutive", align_consecutive),
    ("strided", align_strided),
)


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


def empirical_variance_of_mean(aligned_delta_by_case: dict) -> float:
    """Variance the outer-SE machinery assigns to the aggregate mean."""
    rows = np.mean(
        np.asarray(list(aligned_delta_by_case.values()), dtype=float), axis=0
    )
    return float(np.var(rows, ddof=1) / rows.size)


def crn_statistics(banked_delta: dict) -> tuple[dict, dict]:
    """Per-cell full-row variances and shared-prefix pairwise covariances."""
    names = sorted(banked_delta)
    sigma2 = {
        name: float(np.var(banked_delta[name], ddof=1)) for name in names
    }
    pairs = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = min(banked_delta[a].size, banked_delta[b].size)
            xa, xb = banked_delta[a][:shared], banked_delta[b][:shared]
            gamma = float(np.cov(xa, xb, ddof=1)[0, 1])
            denom = float(np.std(xa, ddof=1) * np.std(xb, ddof=1))
            pairs[f"{a}|{b}"] = {
                "shared_rows": shared,
                "gamma": gamma,
                "correlation": gamma / denom if denom > 0.0 else float("nan"),
            }
    return sigma2, pairs


def plugin_decomposition(sigma2: dict, pairs: dict, sizes: dict) -> dict:
    """Plug-in variance of the equal-weighted mean of full cell means.

    Var = (1/k^2) [ sum_c sigma2_c / n_c
                    + 2 * sum_{c<c'} gamma_cc' / max(n_c, n_c') ]

    The ``max`` denominator is the partial-overlap algebra: two cell means
    sharing their leading min(n_c, n_c') scrambles have covariance
    gamma / max(n_c, n_c').
    """
    k = len(sigma2)
    within = sum(sigma2[name] / sizes[name] for name in sigma2) / (k * k)
    cross = 0.0
    over_cross = 0.0
    common_size = min(sizes.values())
    for key, stats in pairs.items():
        a, b = key.split("|")
        term = 2.0 * stats["gamma"] / max(sizes[a], sizes[b]) / (k * k)
        cross += term
        if sizes[a] > common_size or sizes[b] > common_size:
            over_cross += term
    return {
        "within": within,
        "cross": cross,
        "over_cell_cross": over_cross,
        "total": within + cross,
        # Consecutive grouping pushes the over-cell coupling across outer-row
        # boundaries; to leading order its empirical variance omits it.
        "predicted_consecutive": within + cross - over_cross,
        "predicted_strided": within + cross,
    }


def describe(bound: float, gate: dict) -> str:
    low, high = gate["interval"]
    margin = min(bound - abs(low), bound - abs(high))
    return (
        f"est {gate['estimate_difference']:+.6f}  unc {gate['total_uncertainty']:.6f}  "
        f"interval [{low:+.6f}, {high:+.6f}]  margin {margin:.6f} "
        f"({100.0 * margin / bound:.1f}% of bound)  {gate['status']}"
    )


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
        sizes = {name: rows[0].size for name, rows in banked.items()}
        over = {name: size // common for name, size in sizes.items() if size > common}
        print(f"\n=== {variant}   common scrambles {common} ===")
        for name, g in sorted(over.items()):
            print(f"  over-allocated {name}: {sizes[name]} banked rows (g={g})")

        aligned = {}
        gates = {}
        for label, align in ALIGNMENTS:
            aligned[label] = {
                name: align(rows[0], common) for name, rows in banked.items()
            }
            gates[label] = evaluate_gate(
                stage16,
                aligned[label],
                {name: align(rows[1], common) for name, rows in banked.items()},
                pde_envelope=pde_envelope,
                label=f"{variant}/mean_signed_delta_bias",
            )

        # Hard gate: truncation must rebuild the published numbers exactly.
        reproduced = all(
            abs(gates["truncated"][key] - published[key]) <= 0.0
            for key in ("estimate_difference", "total_uncertainty")
        )
        print(f"  reproduces published gate exactly: {reproduced}")
        if not reproduced:
            for key in ("estimate_difference", "total_uncertainty"):
                print(
                    f"    {key}: rebuilt {gates['truncated'][key]!r} "
                    f"published {published[key]!r}"
                )
            failures.append(variant)
            continue

        # Consecutive and strided are the same point estimate by construction.
        est_c = gates["consecutive"]["estimate_difference"]
        est_s = gates["strided"]["estimate_difference"]
        if abs(est_c - est_s) > ESTIMATE_AGREEMENT_RTOL * max(abs(est_c), abs(est_s)):
            print(f"  FAILED: pooled estimates disagree: {est_c!r} vs {est_s!r}")
            failures.append(variant)
            continue

        delta_full = {name: rows[0] for name, rows in banked.items()}
        sigma2, pairs = crn_statistics(delta_full)
        corr = np.asarray([p["correlation"] for p in pairs.values()])
        top_key = max(pairs, key=lambda key: abs(pairs[key]["correlation"]))
        print(
            f"  cross-cell CRN coupling over {len(pairs)} pairs: "
            f"mean corr {corr.mean():+.4f}  mean |corr| {np.abs(corr).mean():.4f}  "
            f"max |corr| {abs(pairs[top_key]['correlation']):.4f} ({top_key})"
        )
        for key in sorted(pairs):
            a, b = key.split("|")
            if a in over or b in over:
                stats = pairs[key]
                print(
                    f"    over-cell pair {key}: gamma {stats['gamma']:+.3e}  "
                    f"corr {stats['correlation']:+.4f}  "
                    f"(shared {stats['shared_rows']} rows)"
                )

        plugin = plugin_decomposition(sigma2, pairs, sizes)
        cross_share = plugin["cross"] / plugin["total"] if plugin["total"] else 0.0
        print(
            f"  plug-in variance of the full-evidence mean: {plugin['total']:.6e}\n"
            f"    within-cell {plugin['within']:.6e}   "
            f"cross-cell {plugin['cross']:+.6e} ({100.0 * cross_share:+.1f}%)   "
            f"of which over-cell pairs {plugin['over_cell_cross']:+.6e}"
        )
        empirical = {
            label: empirical_variance_of_mean(aligned[label])
            for label in ("consecutive", "strided")
        }
        for label in ("consecutive", "strided"):
            predicted = plugin[f"predicted_{label}"]
            print(
                f"  {label:11s} empirical var {empirical[label]:.6e}  "
                f"plug-in prediction {predicted:.6e}  "
                f"ratio {empirical[label] / predicted:.4f}"
            )

        for label, _ in ALIGNMENTS:
            print(f"  {label:11s} {describe(bound, gates[label])}")

        summary[variant] = {
            "common_scrambles": common,
            "sizes": sizes,
            "over_allocated": over,
            "cell_sigma2": sigma2,
            "pairwise_crn": pairs,
            "plugin_variance": plugin,
            "empirical_variance_of_mean": empirical,
            "gates": {label: gates[label] for label, _ in ALIGNMENTS},
        }

    if failures:
        print(f"\nFAILED for: {', '.join(failures)}")
        return 1

    out = ROOT / "output" / "crn_strided_alignment"
    out.mkdir(parents=True, exist_ok=True)
    (out / "crn_strided_alignment.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True, default=float) + "\n"
    )
    print(f"\nwrote {out / 'crn_strided_alignment.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
