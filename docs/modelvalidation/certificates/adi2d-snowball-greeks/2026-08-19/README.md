# ADI 2D snowball Greeks — imported certification

This certificate was **not produced by `python -m quantark.modelvalidation run`.**
It is a translation of evidence banked by the certification harness that
predates this module, archived here so the result is discoverable, reviewable
and CI-guarded alongside the studies the module ran itself.

| | |
|---|---|
| Study | `adi2d-snowball-greeks` |
| Candidates | `HestonSnowballPDESolver`, `HestonSLVSnowballPDESolver` (2D ADI) |
| Quantities | spot delta, spot gamma |
| Cases | 7 variance regimes × 2 engines = 14 cells, 28 certified quantities |
| Decision | **both ADMITTED** |
| Heston margin | signed-bias interval `[-0.0590, -0.0053]` vs `±0.10` — **41.0 %** to spare |
| Heston-SLV margin | simultaneous interval `[-0.0634, +0.0034]` vs `±0.10` — **36.6 %** to spare |
| Certificate digest | `b6c99a6c96ce6f1bdcc221f35da8f325d5f754fe290649a82ea4364add411446` |

## Why it was imported rather than re-run

The benchmark is not a paired-RQMC average. It is a multilevel control-variate
telescope on independent seed families, with exact conditional integration of
the spot factor, a pilot-frozen cost-weighted Neyman allocation across cells,
and a substep-refinement bias envelope. The Heston-SLV arm alone consumed
**28.6 hours** of held-out production sampling (seeds 20260811/12, 1.16 × 10⁹
conditional path valuations) under an allocation frozen before those seeds were
opened.

Re-running it inside this module would cost that again and would break the
evidence chain the result is banked on — the amendment is byte-linked to its
parent by digest, and a re-run produces a different parent.

## What is live, and what is archived

**Live.** The candidate arm. `example/modelvalidation/adi2d_snowball_greeks.yaml`
defines the two deterministic engines exactly, and `anchors.json` pins the
values the evidence describes. `test/modelvalidation/test_banked_certificates.py`
re-runs both solvers over all fourteen cells on every commit — about six minutes
— and fails the moment they stop producing those numbers.

The anchor values were **independently re-derived**: the study's YAML and
builders reconstruct the configuration from a declarative description, and every
one of the 28 values came back bit-for-bit equal to the harness's. Two
independent expressions of the same configuration agreeing exactly is what makes
the anchor a tripwire rather than a restatement.

**Archived.** The stochastic arm. `evidence/` holds the original payloads
verbatim, digests intact:

| file | schema | what it is |
|---|---|---|
| `evidence/stage16_greek_certification.json` | 13 | the full certification (`p18_strided`), evidence `5c2bd579…` |
| `evidence/stage16_decision.json` | — | its decision record |
| `evidence/stage17_slv_aggregate_amendment.json` | 12 | the SLV aggregate amendment on held-out seeds, evidence `84a16fca…` |
| `evidence/stage17_decision.json` | — | its decision record |

Produced by `example/mo_volmodels/16_adi_greek_certification.py` and
`17_adi_slv_aggregate_certification.py` at commit `258fd7ec`.

## Where this certificate's arithmetic differs from the module's

The `imported.gate_differences` block in `certificate.json` carries this in
machine-readable form. In prose:

1. **Gamma's economic scale.** The harness converts a gamma error to hedge
   contracts through *each case's own model spot*; `HedgeContractScale` uses one
   hedge inception spot for every case. For `near_ki` (spot 75.5) that is a 25 %
   difference in the conversion. The banked cells therefore carry the harness's
   own already-converted numbers, and nothing here re-derives them. The
   `economic_scale` block in the YAML is recorded for the record, not applied.

2. **The interval is wider than `|err| + 2·SE`.** Each cell's `interval_c` is
   the conservative edge of the harness's own interval: Student-t on the
   reference with 1023 degrees of freedom, Bonferroni-split at 97.5 % across two
   stochastic components, then widened by the PDE refinement envelope and by an
   upper bound on the reference's own substep bias. Re-gating with this module's
   simpler arithmetic would report a narrower interval than was actually
   defended.

3. **Booleans.** `passed`, `se_budget_met` and `interval_within_bound` restate
   the harness's banked `PASS`, whose declared reason is that the comparison
   interval sits wholly inside the economic bound. `envelope_within_bound` is
   genuinely recomputed here, from the banked envelope against half the cell
   bound. None is invented.

4. **The aggregate is an interval, not a mean ± SE.** `mean_signed_bias_c` is
   its centre and `se_of_mean_c` its **half-width**, so the reported edge equals
   the banked edge. Heston pools *strided* across common scrambles (output row
   *j* averages scrambles *j*, *j+m*, …), which is the alignment whose standard
   error is exactly right under cross-cell common-random-number coupling —
   measured at corr −0.43 on `low_feller × near_ki`. Heston-SLV's is the
   stage-17 telescope, combined by Welch-t across independent cohorts.

5. **No gamma aggregate.** The certification declared a signed-bias gate for
   delta only, so none is reported for gamma.

## Continuous knock-in: why this evidence survived the 2026-08-19 engine fixes

Main landed six commits that changed how a **continuously** monitored knock-in
is priced — a first-passage transfer in the PDE, made barrier-local for the
vol-model solvers, and a Brownian bridge in the MC that now uses the variance
the paths actually accumulated. Four of the files they touch sit inside this
certification's fail-closed implementation digest.

Every cell here monitors its KI **discretely**, at 252 observations per year, so
none of that machinery is ever constructed for them. That was checked three
ways, not assumed
(`docs/adi2d-greek-perf/probes/probe_merge_ki_invariance.py`):

- **Structurally** — after pricing this study's own product, every PDE solver
  has no first-passage state and every MC engine has its bridge disarmed, with
  positive controls under continuous monitoring proving the assertions have
  teeth.
- **Numerically** — all 14 cells reproduce their banked candidate values
  **bit-for-bit** on the merged tree.
- **By contrast** — the harness's four cross-engine deterministic anchors *do*
  price a continuously monitored KI, and three of the four moved (1–6 %, price
  down in all three, the direction the correction predicts). All three still
  pass their cross-engine tolerances, so the degeneracy property they assert —
  Heston at zero vol-of-vol reduces to flat BSM, unit-leverage SLV reduces to
  Heston — survived the correction. Those anchors are *not* the ones banked
  here; `anchors.json` pins the fourteen discretely-monitored certified cells.

## Regenerating this directory

```bash
PYTHONPATH=$PWD .venv/bin/python \
    docs/adi2d-greek-perf/probes/archive_to_modelvalidation.py
```

The tool is deterministic: same banked evidence in, byte-identical directory
out. It reads `output/p18_strided` and `output/p18_slv_amendment`, which are the
run outputs the `evidence/` copies here came from.
