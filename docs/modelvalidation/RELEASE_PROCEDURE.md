# Engine Release Procedure

How a complex pricing engine (PDE, quadrature, or any other deterministic
method) gets released in this repository: a statistically controlled stochastic
benchmark certifies it, the run banks a schema-versioned evidence package, and
cheap deterministic anchors guard the certified behaviour afterwards.

The machinery is `quantark/modelvalidation/`. This document is the procedure
that uses it.

---

## 1. Which path does your change need?

| Situation | What to run | Why |
|---|---|---|
| New engine, or a new numerical method inside an existing engine | **Full certification** (`run`) | Nothing banked describes this engine's numbers. |
| A deliberate numerics change expressed in **configuration**: a different scheme, a changed default grid, a new engine-params field | **Amendment** (`amend`) | The parts you did not touch stay valid; re-running them wastes hours and breaks the evidence chain. The changed configuration moves the identity hash, which is what tells `amend` what to re-run. |
| A deliberate numerics change expressed in **code**: a fixed discretization bug, a repaired readout, a corrected stencil | **Full certification** (`run`) | `amend` decides what to re-run from the identity hash, and a code fix moves no hash. It would carry forward exactly the cells the fix invalidated. See section 5. |
| A refactor proven bitwise-identical (byte-compare on a detached worktree) | **Anchors only** | The numbers did not move, so the banked evidence still describes the engine. The anchor test proves that claim. |
| Performance work that changes results at all | Full certification or amendment | "Faster and slightly different" is a numerics change, not a refactor. |

If you are unsure between an amendment and a full certification, run the full
certification. An amendment narrows what is re-measured; guessing wrong means
banking evidence for a configuration you never actually tested.

## 2. Running a certification

```bash
# Wiring check first -- seconds, and explicitly not bankable evidence.
python -m quantark.modelvalidation run example/modelvalidation/<study>.yaml --quick

# The real run. Hours-scale for a realistic study.
python -m quantark.modelvalidation run example/modelvalidation/<study>.yaml \
    --out output/modelvalidation

# Interrupted? Resume reuses every checkpoint whose configuration still matches.
python -m quantark.modelvalidation run example/modelvalidation/<study>.yaml \
    --out output/modelvalidation --resume
```

Quick mode shrinks sampling so the standard error will usually miss its budget,
leaving cells `UNRESOLVED` and the decision `INCONCLUSIVE`. That is correct
behaviour, not a failure — quick mode proves the plumbing runs, nothing more.

A run writes four things under `<out>/<study>/`:

- `certificate.json` — the machine record, with its projected SHA-256.
- `report.md` — the terminal- and diff-friendly report.
- `report.html` — the review copy: a single self-contained file (no scripts, no
  external requests) that shows how much of each bound every measurement
  consumed. Open it in a browser, attach it to a review, print it to PDF.
- `checkpoints/` — resume state. **Never banked**; it is scratch, not evidence.

All three reports are written together from the same validated payload, so a
banked directory cannot hold a certificate whose reports describe something
else. Both reports are pure functions of the evidence — no timestamps — so
re-rendering an unchanged certificate produces byte-identical files.

### Why the HTML report matters for review

A cell that consumed 4% of its bound and one that consumed 96% both print as
`PASS`. The HTML report puts a gauge on every cell and every aggregate showing
the fraction of the budget actually used. Scan that column first: a study full
of passes at 90%+ is one small change away from failing, and it is worth knowing
that *before* the change lands rather than after.

It also carries an **Engine configuration** section: the grid each candidate
actually ran on, and the benchmark's own settings, side by side. Certification
without the grid is half a record — "the PDE engine" at 400 spatial points and
"the PDE engine" at 800 are different engines as far as the numbers go.

Those settings are recorded **resolved, not named**. A candidate that declares
`accuracy: standard` also records what that profile expanded to (400 points,
4 steps/day, `eps_crit` 0.003, and the rest). This matters for more than
readability: the resolved configuration feeds the candidate's identity hash, so
if a future release redefines a profile, the identity changes and stale
checkpoints and anchors are correctly rejected instead of silently reused.

The recorded values are what was *requested*. Achieved geometry — the node count
a grid settled on after alignment, or whether a step cap bit — is not exposed by
the engines through a public API and is deliberately not guessed at.

## 3. Reading the result

Three decisions are possible per candidate engine:

- **ADMITTED** — every cell passed against a benchmark that met its
  standard-error budget, and no aggregate bias gate tripped.
- **REJECTED** — confident evidence of disagreement: a failed cell whose
  benchmark met budget, or a measured systematic tilt.
- **INCONCLUSIVE** — the evidence cannot say. An errored cell, an unresolved
  cell, or a tilt that the sampling could not resolve.

`INCONCLUSIVE` is a real answer, not a soft failure. It means "we did not
measure this well enough to decide", and the fix is more sampling or a fixed
engine — never a loosened bound. **Never widen a bound to turn a REJECTED or
INCONCLUSIVE into an ADMITTED.** Bounds encode what a desk can tolerate; moving
them to fit a result inverts the whole point of the exercise. If a bound is
genuinely wrong, change it as its own reviewed commit, with the reasoning
written down, and re-certify from scratch.

## 4. Banking the evidence

Committed evidence lives at:

```
docs/modelvalidation/certificates/<study>/<YYYY-MM-DD>/
├── certificate.json
├── report.md
├── report.html
└── anchors.json
```

When two certifications for the same study land on one calendar day -- an
amendment on top of that morning's certificate, say -- the later one takes a
numeric suffix: `2026-08-19-2`. **Never overwrite the parent directory.** A
child records its parent's digest, and a chain whose parent has been replaced
cannot be verified; the CI guard globs `*/*/anchors.json`, so both directories
keep being checked.

Bank through the command, never by hand:

```bash
python -m quantark.modelvalidation bank output/modelvalidation/<study>
```

It validates the certificate's digest, refuses a quick run, copies
`certificate.json`, `report.md` and `report.html` (never `checkpoints/`), extracts
`anchors.json`, and prints the directory it created. On a same-day repeat it takes
the next suffix itself instead of touching the existing directory.

Reference the banked certificate from the release notes or PR description by
its digest, so a reader can tell which evidence backs which release.

### Retiring a superseded certificate

A numerics change retires every certificate banked before it: those engines no
longer compute what that evidence describes, so re-running their anchors can
only fail. The directory still may not be overwritten or deleted — a child
records its parent's digest, and a chain whose parent is gone cannot be
verified. Retire it in place instead, by adding two keys to its `anchors.json`:

```json
  "superseded_by": "snowball-flat-bsm/2026-09-11",
  "superseded_reason": "Retired by 25d4f7d6, which repaired three PDE near-barrier readout defects. ..."
```

The marker goes in `anchors.json` because that file carries no self-digest.
`certificate.json` is covered by its own `projected_sha256` and must not be
touched; the identity guard reads the marker from the anchor file beside it.

Both banked-evidence guards then skip that directory with a reason, and
`resolve_supersession` refuses the skip unless the successor is really banked
and anchors every `(candidate, case, quantity)` the retired file did. Scope may
grow across a supersession; it may never shrink. A separate test requires every
study to keep at least one live directory, so a study cannot be retired out of
existence by marking all of them — genuinely dropping a study's coverage means
removing its directories, which is visible in a diff.

`superseded_reason` is not decoration. As with an amendment's `--reason`, it is
the only part of the record that says why the numbers moved.

**New files under `docs/` are excluded from this repository's index**
(`.git/info/exclude`), so a freshly banked directory needs `git add -f`.
Forgetting it fails loudly rather than silently: the retired directories name a
successor that is not there, and `resolve_supersession` raises.

## 5. Anchors in CI

Anchors are the cheap residue of an expensive certification: the deterministic
engine's own outputs at pinned configurations. Re-running only the deterministic
side takes seconds, so every commit can check that the certified engine still
produces the numbers the evidence describes.

Wire one up with a single assertion:

```python
from quantark.modelvalidation import assert_anchors

def test_snowball_pde_matches_its_certification():
    assert_anchors("docs/modelvalidation/certificates/snowball-flat-bsm/2026-08-14/anchors.json")
```

**Tolerance policy.** On the machine that banked the evidence (matching
architecture fingerprint) the comparison is exact — any drift there is a real
change. On a different architecture it uses a relative tolerance (`rel_tol`,
default 1e-12, with a small absolute floor). This repository's CI runs x86_64
Linux while evidence is typically banked on ARM64 macOS, and IEEE results
legitimately differ in the last ULP or two across instruction sets. The same
constraint governs `test/golden_compare.py`.

**Quantities that difference prices.** A desk Greek or a roll theta is a finite
difference of prices, `Q = sum_i w_i V_i`, so it carries the prices' cross-arch
noise amplified by its stencil: about `rel_tol * |V| * sum_i |w_i|`. Near an event
that dwarfs `rel_tol * |Q|` -- a one-second theta divides a price difference by
1/3600 of an hour, and on the first CI run of the intraday certificate it moved
2.2e-8 relative while every PV it differences agreed within 1e-9. A candidate may
declare `anchor_noise_weights(case)`: per quantity, the L1 weight of the stencil
it is formed from (the intraday snowball candidates declare the runtime's own desk
bump and theta step, and those as a floor for the engines' point stencils). Off the
banking machine such a quantity is compared at
`max(rel_tol * |Q|, rel_tol * |anchored PV| * weight) + abs_tol`
(`anchors.anchor_tolerance`): exactly what the prices would admit, carried through
the stencil, and no more. On the intraday certificate that is at most 4e-8 of a
spot Greek's column scale and 2e-6 of desk theta's. Same-machine comparison stays
exact, and a candidate that declares nothing keeps the relative tolerance.

When an anchor test fails, the banked certificate no longer describes the
engine. Re-certify or amend — do not update the anchor file to match the new
numbers, which would silently relabel a numerics change as a no-op. Once the
successor is banked, retire the old directory as in section 4 rather than
leaving a failing guard in the suite.

**A code change can move the numbers without moving a single identity hash.**
The identity covers configuration; a repaired discretization is neither a
config change nor anything `candidate_identity` can see. That is why both
guards exist, and it is also why `amend` is the wrong tool for such a fix: its
carry-forward rule keys on identity alone, so it would carry forward exactly
the cells the repair invalidated, while re-pricing the ones it did not touch.
Run the full certification for a code-level numerics change.

## 6. Amendments

```bash
python -m quantark.modelvalidation amend example/modelvalidation/<study>.yaml \
    --parent docs/modelvalidation/certificates/<study>/<date>/certificate.json \
    --reason "TR-BDF2 replaces Craig-Sneyd for the variance sweep" \
    --out output/modelvalidation
```

Rules the tool enforces:

- The parent is fully validated (schema, structure, digest) before any pricing.
- A cell is carried forward only when **both** its candidate identity and its
  benchmark identity still match. A changed benchmark moves the comparison
  target, so even an untouched engine is re-gated against it.
- Scope may grow but never shrink. Dropping a case or a candidate is a new
  certification, because a shrunken amendment would read as though the missing
  coverage had passed.
- `--reason` is mandatory and lands in the payload. It is the only part of the
  record that explains *why* the numbers moved.

## 7. Adding a new engine family

1. **Write builders** (a few dozen lines) in
   `quantark/modelvalidation/builders/<family>.py`: a product spec validator, a
   reference builder wrapping the family's Monte Carlo engine with paired seeds,
   and one candidate evaluator per deterministic engine. Copy
   `equity_snowball.py`; it shows both a solver that returns Greeks directly and
   one that needs central differences.
2. **Register them** by importing the module in `builders/__init__.py`.
3. **Write the study YAML** in `example/modelvalidation/`. Choose cases that
   stress the payoff (barriers, short maturity, low vol), not just the easy
   middle.
4. **Calibrate the bounds honestly.** Measure the achievable standard error at
   your sampling budget first, then set bounds from what the desk needs — and
   check the two are compatible. If the desk bound is tighter than your
   benchmark can resolve, you need more sampling, not a looser bound.
5. `run --quick`, then the full run, then bank, then extract anchors, then wire
   the anchor test.

## 8. Reviewer sign-off checklist

Before a certification is accepted as backing a release:

- [ ] The decision in the report matches the decision in `certificate.json`.
- [ ] No `UNRESOLVED` cells behind an `ADMITTED` claim — every benchmark met its
      standard-error budget.
- [ ] Margin gauges in `report.html` reviewed: note any cell or aggregate above
      ~80% of its bound, since those pass without room to spare.
- [ ] Engine configuration section matches the engines being released — grid
      sizes, step densities, and scheme switches are the ones intended to ship.
- [ ] No `ERROR` cells, or each one is explained and the decision reflects it.
- [ ] Envelope column is populated for grid-based engines (a blank envelope
      means no refinement ladder ran, so the engine's own discretization error
      is unbounded).
- [ ] The `runtime` block matches the machine the run is claimed to have used.
- [ ] The study YAML embedded in the certificate is the one under review.
- [ ] Bounds were not changed in the same commit as the result they judge.
- [ ] `certificate.json`, `report.md`, and `anchors.json` are committed together.

## 9. Studies that ship with the module

| Study | Purpose | Cost |
|---|---|---|
| `european_selftest.yaml` | The framework's own calibration check: the candidate is closed-form Black-Scholes, so the framework **must** admit it. Runs in CI on every commit. | ~3 s |
| `snowball_flat_bsm.yaml` | The demonstration study: PDE and quadrature snowball engines against one paired-RQMC benchmark, five scenarios, PV and both spot Greeks. | minutes |
| `adi2d_snowball_greeks.yaml` | **Imported, not runnable** (see §10): the 2D ADI Heston and Heston-SLV solvers, spot Greeks, seven variance regimes. Its candidate arm is live and anchored. | anchors ~6 min |
| `snowball_intraday_daily_ki_bsm.yaml` | **Schema 2, deterministic reference.** The daily-KI snowball on the intraday clock: QUAD V2 and PDE intraday routes against the engine-independent Gaussian-transition solver (a five-level nested ladder; every radius declares itself analytical or a calibrated estimate; qualified case by case by paired RQMC) on PV, desk spot Greeks, desk theta and point spot Greeks, 23 cases. | about 4.6 h on one worker; anchors 5.5 min |

If `european_selftest` ever fails, suspect the certification machinery before
suspecting the engine — that study exists precisely to make that distinction
possible.

## 10. Imported certifications

Some evidence was produced before this module existed, or by a benchmark this
module's reference protocol cannot express. `adi2d-snowball-greeks` is both: its
benchmark is a multilevel control-variate telescope on independent seed
families, and its production arm cost 28.6 hours of held-out sampling. Re-running
such a study to obtain a native certificate is not a reasonable price for a
change of file format, and it would break the digest chain the original evidence
is banked on.

The rule for importing one:

- **Translate, never re-gate.** Carry the producing harness's own numbers into
  the certificate's numeric fields. If you re-derive verdicts with this module's
  arithmetic you will report verdicts nobody earned — the bounds may match while
  the interval construction does not.
- **Declare every difference** in an `imported` block on the payload:
  `reason`, the harness paths, the source digests, and a `gate_differences` list
  naming each place the arithmetic diverges. A reader comparing this certificate
  to a native one must be able to see where the comparison stops holding.
- **Booleans are exact or recomputed, never invented.** Restate the harness's
  own status where it means the same thing; recompute where the inputs are
  banked; leave a field out rather than guess it.
- **Publish the originals' identity, not their bytes.** The translation is a
  convenience; the original is the record — but a multi-megabyte dump of Monte
  Carlo rows is an artifact, and this repository carries code, tests and
  research docs. Record every digest the producing harness computed
  (evidence, decision, parent, implementation, numerical implementation, run
  configuration, schema) in `imported.source_digests`, name the payload layout
  in `imported.evidence_files`, and put the whole-file checksums in the
  certificate's `README.md`. A holder can then prove their payload is the one
  certified; nobody can substitute a different run.
  Prefer a chain that is checkable **without** the payloads: recording the
  parent's evidence digest both as the parent's own and as the child's declared
  parent makes the byte-link provable from the certificate alone.
- **Tests that need the rows must skip, not soften.** Mark them
  `requires_banked_evidence` (or equivalent) so they report `SKIPPED` with a
  reason naming the directory to drop the payloads into. Never weaken such a
  test into one that passes vacuously when the evidence is missing — a green
  run must mean the check ran.
- **The candidate arm must be live.** Write real builders and a real study YAML,
  so `anchors.json` re-runs the deterministic engines and compares them to the
  banked values. An import whose engines are not anchored is a document, not a
  certification — the anchor is the only part that keeps describing the engine
  as the code moves.
- **Prove the anchors independently.** The builders should reconstruct the
  configuration from the YAML rather than importing the harness, and every
  anchored value should come back bit-for-bit. Two independent expressions of
  one configuration agreeing exactly is the evidence that the YAML really
  describes what was certified.
- **The reference builder should refuse to run.** Declare the external
  benchmark, record its configuration, and raise. A simplified stand-in would
  make `run` appear to work while certifying against something the evidence does
  not describe.

## 11. Schema 2 and the intraday studies

Schema 2 keeps everything in sections 1-8 and adds: quantities from a shared catalogue
(`point_delta`, `desk_gamma`, `desk_theta`, ...), a budget per quantity
(`quantity_bounds`: absolute floor and relative term on the notional-normalized value), a
`context` block (valuation instant, phase, variance profile, calendar, history, checkpoint)
with per-case overrides, semantic assertions (`expect:`) for cells where no number exists,
independent per-case random substreams, and convergence evidence with at least three levels
per axis. Gate values are fractions of each cell's own budget, so a schema-2 study declares
`bounds: {cell: 1.0, mean_signed_bias: 0.2}`. Schema-1 studies and their banked evidence keep
their format and their verdict rule exactly.

Verdicts are three-way, on cells and on the aggregate mean signed error alike. Writing `d`
for the disagreement and `R` for the reference's uncertainty radius: PASS when
`d + R <= budget`, FAIL when `d - R > budget` (or a refinement envelope exceeds its share),
UNRESOLVED when the interval straddles the budget, the reference is not sharp enough, or an
axis shows fewer than three levels. A reference that declares no estimator for a quantity
leaves those cells UNRESOLVED by construction; they are reported, listed in the certificate
as `uncertified_quantities`, and kept out of the candidate decision.

**Reference selection belongs to each study.** A study declares one primary reference
builder, its estimands, its error policy and an optional qualification policy; all are
frozen before any candidate is compared and are part of the certification contract. Schema 2
supports two kinds of primary reference through the same builder, evidence and gate
protocols, and there is no global default: RQMC stays the reference of every study it can
resolve.

A *stochastic* reference states replicate standard errors: `R = interval_k x SE`, sharp
enough when `SE <= se_budget_fraction x budget`, and cell errors add in quadrature in the
aggregate. A *deterministic* reference (`reference_kind = "deterministic"`) states a
*radius*: `R` is the radius itself (`interval_k` never multiplies it), sharp enough when
`radius <= radius_budget_fraction x budget`, and radii add **linearly** in the aggregate,
because discretization errors may share a sign. Its record is typed (`kind`, `values`,
`radii`, `radius_basis`, `undefined`, `evidence`) and is never written as batches with a
zero standard error; a gate under it has `se_c: null` and a `radius_c`.

**A deterministic method does not, by itself, establish an error bound.** Every radius
declares what it is: `analytical` (a named exactness basis or a proved bound) or
`calibrated_estimate` (a numerical estimate with its calibration evidence in the
certificate). A radius that does not say is refused. Equal values across a refinement
ladder do not establish exactness: a builder must validate that every level refines every
discretization it uses, and must treat unexplained stagnation as unresolved. The executed
error policy is one structured, versioned object that also serializes into the `contract`,
so the certificate describes the policy that ran and a changed parameter is a changed
contract.

Use a deterministic reference only when a study revision shows the stochastic one cannot
resolve the budgets (the daily-KI study's is
`docs/superpowers/specs/2026-09-18-intraday-deterministic-reference-revision.md`). When the
study qualifies it, `reference_qualification` names a stochastic arm that simulates every
case; the deterministic value must sit within `max_z` of its standard errors plus the
radius. Qualification is an **eligibility condition**: a case that fails (or whose arm
errored, or had nothing to compare) has no decision-eligible gate, so its cells are
UNRESOLVED and feed no aggregate, whatever the candidate says -- nobody knows which arm is
wrong. The qualifier's sampling, seed and method are frozen in the contract; weakening them
is a new study revision, never an amendment. When reviewing such a certificate, read the
"Reference qualification" table and each case's `radius_basis` before the cells: a z near
`max_z` deserves a rerun of the qualifier with more paths before it deserves a conclusion.
Qualification at the qualifier's precision checks the formulation; it cannot vouch for the
radius at the certification budget, which is what the calibration evidence is for.

A certificate constrains the release, never the code. `quantark.intraday` reads no
certificate and carries no certification status; researchers exercise any engine in
development, test and standalone studies. A production release ships an intraday engine
only for the studies that admit its shipped configuration, and the exclusion of a REJECTED,
INCONCLUSIVE or not-yet-studied engine is recorded in the release notes
(`docs/modelvalidation/legacy/intraday-gate-c/2026-09-18/INVENTORY.md` lists what is pending).

An amendment to a schema-2 certificate needs a schema-2 parent with an identical `contract`
block: estimands, budgets, gate policy, scale, reference targets, seed scheme, the
convergence rule, and under a deterministic reference its kind, its structured error
policy and the whole qualification policy (threshold, targets, sampling, seed, method). Case context and semantic expectations live in the cell identity, so
changing them re-evaluates exactly those cells. A solved reference and its qualification are
carried together or not at all. The implementation digest in every intraday identity covers
the whole dependency tree, so an edit to the runtime, an engine family, the products, the
calendar data or the builders re-runs the affected work.
