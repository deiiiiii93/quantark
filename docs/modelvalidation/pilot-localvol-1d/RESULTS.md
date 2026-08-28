# Pilot results — snowball-localvol-1d certification

Gates the certification run for `LocalVolSnowballPDESolver`. Nothing is banked
until every control here passes.

Machine: `arm64` / macOS — Python 3.11.8, NumPy 2.4.6.
Spec: `docs/superpowers/specs/2026-08-28-localvol-1d-pde-certification-design.md` §7.

---

## Controls 3 and 5 — 2026-08-28

```
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  docs/modelvalidation/pilot-localvol-1d/probe_scale_and_flat.py
```

### Control 5 — economic scale on both surfaces

The two surfaces sit at different index levels while `economic_scale` is a
single study-level block. Uncorrected, every calm-surface error would be
overstated by `6207.268 / 4993.105 = 1.243` — which *inflates* a measured error
and so risks a false `REJECTED`, not a merely conservative pass.

```
delta_quantum = 1.000000000   (must be 1.0)

           surface         s0  contract_mult   reported       true     ratio
------------------------------------------------------------------------------
  crash 2024-02-08   4993.105       1.000000   1.000000   1.000000  1.000000
  calm  2023-11-15   6207.268       0.804397   0.804397   0.804397  1.000000
```

**PASS.** The computed `contract_multiplier = REFERENCE_SPOT / s0` converts a
known raw delta to the intended contract count on both surfaces, to 1e-6. The
correction is structural — derived in `resolve_product_spec` from the artifact's
own `s0` — so it cannot be mistyped in the study YAML.

### Control 3 — flat-surface collapse

Flatten the crash surface to its longest ATM pillar and the local-vol PDE must
collapse onto the flat-BSM PDE. This separates "the input is wrong" from "the
formula is wrong", and is the control that settled the original diagnosis.

```
flat vol = 0.294545   (longest ATM pillar)

quantity               LV              BSM          rel
--------------------------------------------------------
   price    4788.76486857    4788.76486857    1.083e-13
   delta       0.56359637       0.56359637    2.486e-13
   gamma      -0.00067065      -0.00067065    7.106e-12
```

**PASS, and far more strongly than the criterion required.** The pass bound was
`delta rel < 1e-3`, allowing for the two solvers aligning their grids
differently. The measured agreement is **1e-13** — float round-off. On a flat
Dupire surface the two code paths produce the same numbers to machine
precision, so the local-vol machinery contributes no discretization difference
of its own; it reuses the identical grid and time-stepping.

Worth recording because of what it rules out: any residual disagreement on a
*real* surface cannot come from the solver skeleton. It has to come from the
surface read or from the reference — which is exactly where
FINDING-2026-08-26 found it.
