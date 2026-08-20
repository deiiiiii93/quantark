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
| A deliberate numerics change: different scheme, changed default grid, a fixed discretization bug | **Amendment** (`amend`) | The parts you did not touch stay valid; re-running them wastes hours and breaks the evidence chain. |
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

Copy in `certificate.json`, `report.md`, and `report.html` (never
`checkpoints/`), then extract the anchors:

```bash
python -m quantark.modelvalidation anchors \
    docs/modelvalidation/certificates/<study>/<date>/certificate.json
```

Reference the banked certificate from the release notes or PR description by
its digest, so a reader can tell which evidence backs which release.

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

When an anchor test fails, the banked certificate no longer describes the
engine. Re-certify or amend — do not update the anchor file to match the new
numbers, which would silently relabel a numerics change as a no-op.

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

If `european_selftest` ever fails, suspect the certification machinery before
suspecting the engine — that study exists precisely to make that distinction
possible.
