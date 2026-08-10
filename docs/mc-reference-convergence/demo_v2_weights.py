"""V2 evidence: cross-fitted Heston-control weights on real cell batch means.

For each cell, generate matched-seed heston_slv (primary) and heston (control)
batch deltas via the harness estimator, feed them to cross_fitted_control with
an INDEPENDENT high-precision heston run as the control expectation, and
record the variance ratio plus the agreement against the frozen 0.7 weight.

Both estimators are unbiased for E[primary], so V2-G1 is an agreement check:
they must not disagree by more than their combined SE.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from demo_common import batch_deltas, case_fixture, case_scale  # noqa: E402

from quantark.montecarlo.control_weights import cross_fitted_control  # noqa: E402

CELLS = ("ordinary_full", "ordinary_decayed", "sigma_collapse")
FROZEN_WEIGHT = 0.7
PRIMARY_SEED = 20260810
EXPECTATION_SEED = 20260812
BATCHES = 32


def main() -> None:
    out = Path(__file__).resolve().parent / "logs" / "v2_weights.jsonl"
    out.parent.mkdir(exist_ok=True)
    for cell in CELLS:
        # Matched scrambles: the control must be correlated with the primary.
        primary, _ = batch_deltas(
            cell, "heston_slv", batches=BATCHES, seed=PRIMARY_SEED, bridge_dimensions=1
        )
        control, _ = batch_deltas(
            cell, "heston", batches=BATCHES, seed=PRIMARY_SEED, bridge_dimensions=1
        )
        # Independent run supplies E[control]; reusing the matched run would
        # cancel the control term entirely and prove nothing.
        expectation_sample, _ = batch_deltas(
            cell,
            "heston",
            batches=BATCHES,
            seed=EXPECTATION_SEED,
            bridge_dimensions=1,
        )
        expectation = float(expectation_sample.mean())

        fitted = cross_fitted_control(primary, control, control_expectation=expectation)
        fixed = primary - FROZEN_WEIGHT * (control - expectation)

        case, _, _, _ = case_fixture(cell, "heston_slv")
        scale_factor = abs(case_scale(case).delta_contracts(1.0))
        combined_se = float(
            np.sqrt(
                np.var(fitted.adjusted, ddof=1) / fitted.adjusted.size
                + np.var(fixed, ddof=1) / fixed.size
            )
        )
        difference = float(abs(fitted.adjusted.mean() - fixed.mean()))
        record = {
            "cell": cell,
            **fitted.as_dict(),
            "frozen_weight": FROZEN_WEIGHT,
            "adjusted_mean": float(fitted.adjusted.mean()),
            "fixed_07_mean": float(fixed.mean()),
            "difference_contracts": difference * scale_factor,
            "v2_g1_sigma": round(difference / max(combined_se, 1e-30), 2),
            "v2_g1_pass": bool(difference <= 2.0 * combined_se),
            "v2_g2_pass": bool(fitted.variance_ratio <= 1.0),
            "primary_sd_contracts": float(np.std(primary, ddof=1)) * scale_factor,
            "adjusted_sd_contracts": float(np.std(fitted.adjusted, ddof=1))
            * scale_factor,
        }
        with out.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
