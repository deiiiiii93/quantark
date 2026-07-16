# Execution framework Phase 2 benchmark — 2026-07-16

Host: macOS-26.5.2-arm64-arm-64bit, 14 logical CPUs, Python 3.11.8, reps=5 post-warm-up, medians reported. (Developer machine, NOT the controlled release host.)

## Gate 4 — CRN reuse (10-point spot ladder, serial)

| engine | reuse median s | disabled median s | speedup |
|---|---|---|---|
| DCNMCEngine 2^17x8 | 5.473 (IQR 0.020) | 11.318 (IQR 0.037) | **2.07x** |
| LocalVolDCNMCEngine 2^16x8 | 12.374 (IQR 0.122) | 15.637 (IQR 0.098) | 1.26x |

Attribution: the LV ladder is simulation-dominated (per-step Dupire interpolation), and each distinct spot needs its own Dupire surface, so only draw generation is reusable there; the gate workload is the draw-dominated GBM ladder.

## Gate 1 support — serial overhead (LocalVolDCNMCEngine, 2^16 paths, 8 batches)

direct median 1.207s, session-serial median 1.211s, overhead **+0.4%**

(Session-serial WARM includes Dupire+draw reuse across repetitions; the cold first-call overhead is covered by the 3% gate on the controlled host.)

## Gate 3 — thread scaling (DCNMCEngine, 131072 paths, 16 batches)

| workers | cold median s | cold speedup | warm median s | warm speedup |
|---|---|---|---|---|
| 1 | 1.182 | 1.00x | 0.500 | 1.00x |
| 2 | 0.729 | 1.62x | 0.294 | 1.70x |
| 4 | 0.501 | 2.36x | 0.235 | 2.13x |
| 8 | 0.530 | 2.23x | 0.362 | 1.38x |

## Framework vs legacy engine-internal threading (DCNMCEngine, 2^17 paths, 16 batches, cold draws)

| workers | framework session s | legacy num_workers s |
|---|---|---|
| 1 | 1.200 | 1.223 |
| 4 | 0.538 | 0.515 |
| 8 | 0.583 | 0.592 |

## Gate verdicts (this host)

- Gate 3 @4 workers (cold): 2.36x (PASS vs 1.5x)
- Gate 3 @8 workers (cold): 2.23x vs required 2.5x — HOST-LIMITED: the direct legacy num_workers=8 path is no faster on this machine (framework-vs-legacy table above); the 2.5x@8 gate requires the controlled release host (spec section 20)
- Framework threads vs legacy threads: worst ratio 1.05 (<= 1.05 expected; the framework adds no threading overhead)
- Gate 4 CRN reuse: 2.07x (PASS vs 2x)
- Serial overhead (warm, informational): +0.4%

## Reproducibility notes (2026-07-16 investigation)

- Gate 4 across 3 independent cool-machine runs: 2.03x / 2.07x / 2.05x
  (cached ~5.6s, uncached ~11.5s). Running the CRN ladder AFTER the
  thread-scaling section depresses the ratio to ~1.98-1.99x (thermal
  state); the script therefore runs serial gates first.
- Gate 3 on the LV engine for attribution: warm-cache LV scaling peaks at
  ~1.5x@2 and DEGRADES at 8 workers, identically on the direct legacy
  ``num_workers`` path (measured 1.51x@2 / 0.88x@8 direct) — the residual
  per-step interpolation loop is GIL/memory-bound. The framework matches
  legacy threading at every worker count (parity table above).
- The 2.5x@8 threshold was not reachable by ANY configuration of this
  engine family on this 14-core Apple-Silicon laptop (best observed
  ~2.4x@4, saturating), including the pre-existing legacy path; per spec
  section 20 the production-sized gates are required on the scheduled
  controlled host, where this row must be re-measured before release.
