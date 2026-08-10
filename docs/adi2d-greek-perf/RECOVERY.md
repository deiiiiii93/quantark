# Recovery Note — 2026-08-10

On 2026-08-10 (~02:00–09:30 CST) macOS crashed and rebooted, wiping `/private/tmp`.
That directory held the working copy `/private/tmp/quant-ark-adi-greek-certification`
(where the certification runs and this research package lived, uncommitted) and the
Claude session scratchpads. This package was reconstructed the same morning from
session transcripts. Codex independently recreated the source worktree at
`.worktrees/adi-greek-certification` (branch `codex/adi-greek-certification`, commit
`14234fd` — no source was lost).

## What was recovered, and from where

| Artifact | Provenance | Fidelity |
|---|---|---|
| `README.md`, `implementation-spec.md` | Write/Edit replay, Claude session `0b6e4fbe` | Exact final state (one errored edit correctly skipped) |
| `scripts/*` (22 demos + `thomas_kernel.c`) | Write replay, session `0b6e4fbe` scratchpad | Exact as-authored **scratchpad** versions — the packaged path-fixed copies are gone, so these still reference `/private/tmp/quant-ark-adi-greek-certification`; substitute the repo root and run with `PYTHONPATH=<repo-root>` (see README §15) |
| `probes/probe_delta_attribution.py` | Write/Edit replay, Claude session `3b8441f4` | Exact |
| `recovered/schema11_partial_values.json` | Probe fidelity record (session `3b8441f4`, 2026-08-10T01:51) + Codex rollout dumps | Exact for the fields present; **partial** — see below |

## What was permanently lost

- **Banked Stage-16 MC reference checkpoints** (schema 9 + schema-11 replacement cohorts,
  `output/adi_greek_certification/checkpoints/`) — the fixture every PDE-only re-cert gate
  in `implementation-spec.md` assumed. These must be regenerated before any B-G1 / C-G6 /
  G1-G1 style re-certification can run.
- **Schema-11 parent evidence JSON + report** (`adi_greek_certification.json` / `.md`).
  The *verdicts* survive in transcripts (see below); the raw per-cell records mostly do not.
- Stage-17 pilot JSONs, allocation projections, production-run checkpoints (the production
  process itself, PID 6864, was killed by the reboot).
- This package's `logs/` directory (raw demo outputs). The measured numbers survive in
  `README.md`; only the raw logs are gone.
- The four in-flight delta-bias attribution probe results from session `3b8441f4`
  (relaunched 2026-08-10 from this package's `probes/` copy).

## Schema-11 verdicts (recovered from transcripts — evidence lost, conclusions safe)

- All 7 `heston` cells PASS delta+gamma; aggregate mean signed delta bias PASS
  (−0.0479, 97.5% interval [−0.0843, −0.0115], bound ±0.10) → **Heston PDE route admitted**.
- All 7 `heston_slv` cells PASS individually; aggregate INCONCLUSIVE
  (−0.0527, interval [−0.1394, +0.0340] overlaps the ±0.10 bound) → **Heston-SLV route
  still excluded**.
- Uncertainty budget of the INCONCLUSIVE verdict: 63% statistical (MC noise concentrated
  in the three cells that never got the variance-reduction treatment: ordinary_decayed
  37.5%, ordinary_full 36.4%, sigma_collapse 15.8%), 21% substep envelope, 16% PDE
  refinement envelope.
- The point estimate itself is structural: `sigma_collapse` (heston −0.112 ± 0.010, 11σ)
  and `low_feller` (heston −0.107 ± 0.031) carry the bias in both families, and
  `sigma_collapse`'s n_v axis is not converged at production floors (first-order,
  refinement ratio ≈ 0.47).
- The frozen production allocation for the Stage-17 aggregate run (4096 primary / 256
  middle batches, Heston control weight 0.7, guarded interval [−0.099272, −0.038637])
  is committed and hash-pinned in source — the allocation-selection work does not need
  to be repeated.

## Exactly-recovered numeric anchors

`heston/sigma_collapse` at the schema-11 production grid (300×135×4800, adaptive_upwind,
1% central bump, dense-KI):

- PDE delta `0.32268601473516256`, PDE gamma `-0.033319329144351206`
- MC reference delta `0.3247230771781795`, SE `0.00018390682558800569`
- gap −0.11235816074815635 contracts; `hedge_inception_spot = 4532.52`
  (scale relation re-verified bitwise on 2026-08-10)

`heston_slv/ordinary_full` delta: PDE `0.208782811`, reference `0.209217109`,
SE `1.713e-03` (Codex rollout, 2026-08-07T14:52).

## Policy change to prevent recurrence

Run all long-lived work — worktrees, certification outputs, banked references — under
`/Users` (e.g. the repo's `output/`, which is gitignored but persistent), never under
`/private/tmp`. After any certification run completes, archive the checkpoint directory
as a dated tarball before starting dependent work.
