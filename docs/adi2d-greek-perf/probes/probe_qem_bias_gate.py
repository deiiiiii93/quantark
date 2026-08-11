"""Gate 1 of 2 for smoothed QE-M: does the transition-law bias vanish under dt?

P5 established that gamma-QEM restores the RQMC rate (PV stderr slope -0.72 vs
-0.45) but detected a transition-law bias at full resolution: PV diff +0.0873
+/- 0.0281, z = +3.11. Matching two conditional moments does not match the law,
so higher-moment and tail differences survive at daily dt.

Both samplers converge to the SAME CIR transition as dt -> 0, so the difference
between them must vanish under refinement. That is the gate:

  * if |diff| falls like dt^p with p >= 1, the gap is discretization and
    gamma-QEM is a bias-CONTROLLED alternative -- refinement buys correctness,
    and the cost work is worth doing;
  * if |diff| is flat in dt, it is a persistent bias, no refinement rescues it,
    and the quantile-table engineering would be wasted.

Two design points that differ from P5 on purpose.

**Measured on the Greeks, not just PV.** The certification gate compares PDE
against MC on delta and gamma, never on PV. A transition-law bias that is smooth
in S0 largely cancels in a central difference, and P5's own numbers hint at
exactly that: gamma-QEM's FD-gamma sat ON the session reference while standard
QE-M's was 1.9 sigma off it, despite gamma-QEM being the biased one on PV. So
PV bias alone cannot condemn it, and PV bias alone cannot clear it either.

**Pseudo draws with common random numbers across the spot legs.** Bias is a
property of the transition law, not of the draw sequence, and Sobol at eight
substeps would need a 12,096-dimensional block per pricing. ``_draws`` reseeds
from ``self.seed`` deterministically, so each spot leg sees identical draws and
the finite difference stays CRN-paired.

Refinement note, learned the hard way: ``substeps_per_interval`` on the engine is
**inert** for these demo pricers.  The engine applies it in ``_refined_dt_array``
inside its own simulate path, while the demo pricers call ``_build_time_grid``
and then run their own loop over the raw ``dt_array`` -- so patching the engine
factory changed nothing and every substep count returned identical digits.  This
probe therefore refines the grid itself, exactly as the engine's mixin does
(``np.repeat(dt / n, n)``, term inputs rebuilt on the refined array, recorded
nodes sliced back to contractual with ``nodes[:, ::n]``), and prices through the
engine's own payoff kernel.  The OSS estimator cannot be used here at all: its
terminal closed form and barrier flags are welded to the contractual grid.  A
substeps=1 identity check guards the plumbing.

Usage:
    PYTHONPATH=$PWD python docs/adi2d-greek-perf/probes/probe_qem_bias_gate.py [--smoke]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEMOS = Path("/Users/fuxinyao/quant-ark/docs/mc2d-gamma-convergence/demos")
P5_PROBES = Path("/Users/fuxinyao/quant-ark/docs/cert-cost-reduction/probes")
OUTPUT_DIR = ROOT / "output" / "qem_bias_gate"

for path in (DEMOS, P5_PROBES):
    if not path.is_dir():
        raise SystemExit(f"required probe directory not found: {path}")
    sys.path.insert(0, str(path))

from common import HESTON, SPOT0, batch_seeds  # noqa: E402
from probe_p5_smoothed_qem import SamplerPricer  # noqa: E402

# The certification's own bump, not P5's 0.3%: a wider bump cuts FD noise and
# matches the quantity the gate actually certifies.
BUMP_FRACTION = 0.01


class RefinedSamplerPricer(SamplerPricer):
    """``mirror_price`` with the SDE grid refined n-for-1 per interval.

    Mirrors ``_SubstepRefinementMixin`` exactly: the fine grid is
    ``np.repeat(dt / n, n)``, term inputs are rebuilt on it (the engine refines
    before calling ``_term_inputs``), and recorded nodes are sliced back to the
    contractual grid so barrier monitoring and the payoff kernel are untouched.
    """

    def __init__(self, *args, substeps: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.substeps = int(substeps)
        if self.substeps < 1:
            raise ValueError("substeps must be a positive integer")

    def refined_mirror_price(self, spot: float) -> float:
        engine, env, T, all_times, dt_array, ko_idx, ki_idx, _term, _kop = self._setup(
            spot
        )
        n = self.substeps
        coarse = np.asarray(dt_array, dtype=float)
        fine = np.repeat(coarse / n, n) if n > 1 else coarse
        term = engine._term_inputs(T, fine)

        steps = int(fine.size)
        z_var, z_ind, u_var = self._draws(steps)
        params = HESTON
        log_s = np.full(self.num_paths, np.log(max(float(spot), 1e-12)))
        var = np.full(self.num_paths, max(float(params.v0), 0.0))
        nodes = np.empty((self.num_paths, steps + 1))
        nodes[:, 0] = np.exp(log_s)
        for index, dt in enumerate(fine):
            drift = float(term.rrf[index] - term.div[index])
            v_np, v_bar, mu_extra, s_cond = self._qe_step(
                var, z_var[:, index], u_var[:, index], params, dt, self.martingale
            )
            log_s = (
                log_s
                + (drift - 0.5 * v_bar) * dt
                + mu_extra
                + s_cond * z_ind[:, index]
            )
            var = v_np
            nodes[:, index + 1] = np.exp(log_s)

        contractual = nodes[:, ::n] if n > 1 else nodes
        rate = env.get_rate(T)
        vol = env.get_vol(self.product.strike, T)
        payoffs, settle, _stats = engine._compute_payoffs(
            self.product,
            env,
            contractual,
            all_times,
            ko_idx,
            ki_idx,
            rate,
            T,
            vol,
            rng_seed=self.seed + 1337,
        )
        return float((payoffs * engine._df(settle)).mean())


def assert_refinement_plumbing(paths: int, seed: int) -> None:
    """substeps=1 must reproduce the published mirror price bitwise.

    Without this the whole sweep can silently measure nothing -- the first
    version of this probe reported identical digits at every substep count
    because the refinement never took effect.
    """
    for sampler in ("qe", "gamma"):
        published = SamplerPricer(paths, seed, sampler=sampler, draws="pseudo")
        refined = RefinedSamplerPricer(
            paths, seed, sampler=sampler, draws="pseudo", substeps=1
        )
        baseline = published.mirror_price(SPOT0)
        mirrored = refined.refined_mirror_price(SPOT0)
        if baseline != mirrored:
            raise SystemExit(
                f"refinement plumbing broken for {sampler}: substeps=1 gives "
                f"{mirrored!r} but mirror_price gives {baseline!r}"
            )
    print("plumbing check: substeps=1 reproduces mirror_price bitwise for both samplers")


def _greeks(sampler: str, *, paths: int, seed: int, bump: float, substeps: int) -> tuple:
    """(pv, delta, gamma) from three CRN legs of one sampler."""

    def price(spot: float) -> float:
        pricer = RefinedSamplerPricer(
            paths, seed, sampler=sampler, draws="pseudo", substeps=substeps
        )
        return float(pricer.refined_mirror_price(spot))

    up = price(SPOT0 + bump)
    mid = price(SPOT0)
    down = price(SPOT0 - bump)
    return (
        mid,
        (up - down) / (2.0 * bump),
        (up - 2.0 * mid + down) / (bump * bump),
    )


def _summarize(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(
        array.std(ddof=1) / np.sqrt(array.size) if array.size > 1 else np.inf
    )


def measure(*, substeps: int, paths: int, seeds: Sequence[int], bump: float) -> dict:
    collected: dict[str, dict[str, list]] = {
        sampler: {"pv": [], "delta": [], "gamma": []} for sampler in ("qe", "gamma")
    }
    seconds = {"qe": 0.0, "gamma": 0.0}

    for seed in seeds:
        for sampler in ("qe", "gamma"):
            started = time.perf_counter()
            pv, delta, gamma = _greeks(
                sampler, paths=paths, seed=seed, bump=bump, substeps=substeps
            )
            seconds[sampler] += time.perf_counter() - started
            collected[sampler]["pv"].append(pv)
            collected[sampler]["delta"].append(delta)
            collected[sampler]["gamma"].append(gamma)

    row: dict = {
        "substeps": int(substeps),
        "paths": int(paths),
        "seeds": len(seeds),
        "seconds_per_pricing": {
            name: total / (len(seeds) * 3) for name, total in seconds.items()
        },
        "cost_ratio": (
            seconds["gamma"] / seconds["qe"] if seconds["qe"] > 0 else float("nan")
        ),
    }
    for quantity in ("pv", "delta", "gamma"):
        qe_mean, qe_se = _summarize(collected["qe"][quantity])
        ga_mean, ga_se = _summarize(collected["gamma"][quantity])
        # PAIRED difference: both samplers consume the same pseudo draws, so they
        # share the spot factor z_ind and the variance uniform u_var. Differencing
        # per seed before averaging cancels that shared randomness; the unpaired
        # sqrt(se_qe^2 + se_ga^2) throws the pairing away and can be several times
        # wider on the very quantity the gate is trying to resolve.
        paired = np.asarray(collected["gamma"][quantity], dtype=float) - np.asarray(
            collected["qe"][quantity], dtype=float
        )
        paired_mean, paired_se = _summarize(paired)
        unpaired_se = float(np.sqrt(qe_se**2 + ga_se**2))
        row[quantity] = {
            "qe": [qe_mean, qe_se],
            "gamma_qem": [ga_mean, ga_se],
            "diff": paired_mean,
            "joint_se": paired_se,
            "z": paired_mean / paired_se if paired_se > 0 else 0.0,
            "unpaired_se": unpaired_se,
            "pairing_gain": (
                unpaired_se / paired_se if paired_se > 0 else float("nan")
            ),
        }
    return row


def _convergence_order(rows: Sequence[dict], quantity: str) -> tuple:
    """Observed p in |diff| ~ dt^p, or a refusal and why.

    Refuses two ways, because a rate fitted through noise is worse than no rate:

    * the sign of ``diff`` is not constant along the ladder -- a quantity that
      crosses zero has no decay rate, and ``log|diff|`` would be fitting the
      shape of the noise;
    * no rung separates ``diff`` from zero (|z| < 2), so there is no signal whose
      decay could be measured.  This is the expected outcome when the effect is
      already below the noise floor at the coarsest rung, and it means the ladder
      is pointed the wrong way, not that p = 0.
    """
    entries = [row[quantity] for row in rows]
    differences = [entry["diff"] for entry in entries]
    if len(differences) < 3:
        return None, "need at least three rungs"
    if len({difference > 0.0 for difference in differences}) > 1:
        return None, "diff changes sign along the ladder: no decay rate exists"
    if max(abs(entry["z"]) for entry in entries) < 2.0:
        return None, "no rung separates diff from zero (all |z| < 2)"

    x = np.log([float(row["substeps"]) for row in rows])
    y = np.log([abs(difference) for difference in differences])
    # dt ~ 1/substeps, so the slope against log(substeps) is -p.
    return float(-np.polyfit(x, y, 1)[0]), "fitted"


def oss_pv_crosscheck(*, paths: int, seeds: Sequence[int]) -> dict:
    """Re-run P5's PV agreement gate on P5's own instrument, paired.

    The refined gate above finds no transition-law difference on the engine's
    payoff kernel, which contradicts P5's z = +3.11 on the OSS estimator. Either
    the bias is real and the engine kernel is blind to it, or it is specific to
    the OSS functional. Same fixture, same daily grid, same sampler pair -- only
    the estimator changes -- so this isolates which.
    """
    per_seed = {"qe": [], "gamma": []}
    for seed in seeds:
        for sampler in ("qe", "gamma"):
            pricer = SamplerPricer(paths, seed, sampler=sampler, draws="pseudo")
            per_seed[sampler].append(float(pricer.oss_price(SPOT0)))

    qe = np.asarray(per_seed["qe"], dtype=float)
    gamma_qem = np.asarray(per_seed["gamma"], dtype=float)
    paired_mean, paired_se = _summarize(gamma_qem - qe)
    qe_mean, qe_se = _summarize(qe)
    ga_mean, ga_se = _summarize(gamma_qem)
    unpaired_se = float(np.sqrt(qe_se**2 + ga_se**2))
    return {
        "estimator": "oss",
        "paths": int(paths),
        "seeds": len(seeds),
        "qe": [qe_mean, qe_se],
        "gamma_qem": [ga_mean, ga_se],
        "diff": paired_mean,
        "paired_se": paired_se,
        "paired_z": paired_mean / paired_se if paired_se > 0 else 0.0,
        "unpaired_se": unpaired_se,
        "unpaired_z": paired_mean / unpaired_se if unpaired_se > 0 else 0.0,
        # Per-seed differences, so the tail can be inspected rather than assumed
        # normal: a single outlier seed would make every t-style z here unsafe.
        "per_seed_diff": [float(value) for value in (gamma_qem - qe)],
        "seeds_used": [int(seed) for seed in seeds],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--oss-pv-crosscheck",
        action="store_true",
        help="only re-run P5's OSS PV agreement gate, paired",
    )
    parser.add_argument("--substeps", type=int, nargs="+", default=None)
    parser.add_argument("--paths", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument(
        "--seed-root",
        type=int,
        default=20_260_811,
        help="seed-set root; pass 24680 to replicate P5's exact seed set",
    )
    args = parser.parse_args(argv)

    if args.smoke:
        substep_ladder = args.substeps or (1, 2)
        paths = args.paths or 2048
        seed_count = args.seeds or 3
    else:
        substep_ladder = args.substeps or (1, 2, 4)
        paths = args.paths or 8192
        seed_count = args.seeds or 12

    bump = BUMP_FRACTION * SPOT0
    seeds = batch_seeds(int(args.seed_root), seed_count)

    if args.oss_pv_crosscheck:
        result = oss_pv_crosscheck(paths=paths, seeds=seeds)
        print(
            f"OSS PV cross-check ({result['seeds']} seeds x {result['paths']} paths, "
            f"daily grid)\n"
            f"  QE-M      {result['qe'][0]:+.5f} +/- {result['qe'][1]:.5f}\n"
            f"  gammaQEM  {result['gamma_qem'][0]:+.5f} +/- {result['gamma_qem'][1]:.5f}\n"
            f"  diff      {result['diff']:+.5f}   paired SE {result['paired_se']:.5f} "
            f"(z={result['paired_z']:+.2f})   unpaired SE {result['unpaired_se']:.5f} "
            f"(z={result['unpaired_z']:+.2f})"
        )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        target = OUTPUT_DIR / "oss_pv_crosscheck.json"
        target.write_text(json.dumps(result, indent=1, sort_keys=True))
        print(f"\nwrote {target}")
        return 0
    print(
        f"QE-M bias gate: substeps {list(substep_ladder)}, {paths} paths, "
        f"{seed_count} seeds, bump {BUMP_FRACTION:.1%}\n"
    )

    assert_refinement_plumbing(min(paths, 1024), int(seeds[0]))
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_DIR / "qem_bias_gate.json"

    rows = []
    for substeps in substep_ladder:
        row = measure(substeps=substeps, paths=paths, seeds=seeds, bump=bump)
        rows.append(row)
        # Checkpoint after every substep level: the ladder's top rungs are the
        # expensive ones, and a late failure must not discard the cheap ones.
        destination.write_text(
            json.dumps(
                {"bump_fraction": BUMP_FRACTION, "rows": rows, "complete": False},
                indent=1,
                sort_keys=True,
            )
        )
        print(f"--- substeps {substeps} (cost x{row['cost_ratio']:.2f}) ---")
        for quantity in ("pv", "delta", "gamma"):
            entry = row[quantity]
            print(
                f"  {quantity:<6} QE-M {entry['qe'][0]:+.5f}   "
                f"gammaQEM {entry['gamma_qem'][0]:+.5f}   "
                f"paired diff {entry['diff']:+.5f}+/-{entry['joint_se']:.5f} "
                f"(z={entry['z']:+.2f}, pairing x{entry['pairing_gain']:.1f})"
            )
        print(flush=True)

    print("=== observed convergence order p in |diff| ~ dt^p ===")
    orders = {}
    for quantity in ("pv", "delta", "gamma"):
        order, why = _convergence_order(rows, quantity)
        orders[quantity] = {"p": order, "note": why}
        rendered = f"{order:+.2f}" if order is not None else f"REFUSED -- {why}"
        print(f"  {quantity:<6} p = {rendered}")
    print(
        "\nreading: p >= 1 => discretization, refinement buys correctness "
        "(bias-controlled alternative). p ~ 0 => persistent bias, no refinement "
        "rescues it."
    )

    destination.write_text(
        json.dumps(
            {
                "bump_fraction": BUMP_FRACTION,
                "rows": rows,
                "convergence_order": orders,
                "complete": True,
            },
            indent=1,
            sort_keys=True,
        )
    )
    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
