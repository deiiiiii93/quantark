# MC Reference-Stack Convergence — Demos & Decision Matrix

Spec: `docs/superpowers/specs/2026-08-10-mc-reference-convergence-design.md`
Plan: `docs/superpowers/plans/2026-08-10-mc-reference-convergence.md`

Rows are measured by `demo_cell.py` using the stage-16 harness's own case,
product, environment, engine, and paired-RQMC estimator (loaded by path, so
the treatment under test is the production estimator with one knob changed).
Raw outputs live in `logs/`. **Numbers here are copied only from completed
runs** — no projections.

## What the rows mean

- `baseline` — the cell's current production profile (bridge dimensions 1),
  the SD and cost anchor.
- `bridge8` — the V1 treatment candidate: 8 leading residual bridge
  coordinates, the profile `near_ki` already uses.
- `unbias` — `bridge8` again on an independent seed, giving the V1-G1
  agreement check against `baseline`.

Gates (spec §4): **V1-G1** |baseline − unbias| ≤ 2σ combined; **V1-G2** SD
factor ≥ 4×; **V1-G3** cost recorded so the decision uses SE²·seconds, not SD
alone.

## Decision matrix — V1 treatments (`heston_slv`)

| Cell | Row | batch SD (c) | sec/batch | peak RSS (GiB) | SD factor | SE²·sec factor | V1-G1 |
|---|---|---|---|---|---|---|---|
| _pending Task 4_ | | | | | | | |

## V2 — cross-fitted control weights

| Cell | weights (out-of-fold) | variance ratio | V2-G1 agreement σ |
|---|---|---|---|
| _pending Task 6_ | | | |

## User decisions (Task 7 checkpoint)

- [ ] `ordinary_full` treatment: _pending_
- [ ] `ordinary_decayed` treatment: _pending_
- [ ] `sigma_collapse` treatment: _pending_
- [ ] V2 cross-fitted weights adopted where a control ships: _pending_

Cells left at `baseline` are not failures — WS-S allocation absorbs them by
sending more batches their way.

## Reproduction

```bash
cd .worktrees/adi-greek-certification
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
    docs/mc-reference-convergence/demo_cell.py --cell ordinary_full
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
    docs/mc-reference-convergence/demo_v2_weights.py
```

`PYTHONPATH` is required: the editable install's compat `.pth` binds
`quantark` at interpreter startup, before any in-script `sys.path` edit.
