# QUAD V2 feasibility experiment

This is a numerical experiment supporting the [proposed design](../../docs/plans/2026-09-11-quad-v2-design.md). It is not a new production engine.

The confirmed V2 speed priority is price plus Greeks and scenario curves, with accuracy gates across all supported products. The design benchmarks cold preparation, reusable 101-spot curves and complete market-risk/scenario batches separately; scalar RFQ speed is secondary. This probe measures only price plus delta and backend equivalence, so it does not establish performance for the complete target workload or the proposed twofold throughput improvement.

`backend_probe.py` compares direct banded and FFT application of the exact same kernels from the frozen Gaussian reference in `../gaussian_quad_comparison/quad_reference_snapshot.py`. It uses linear convolution, explicit boundary extension, suitable FFT padding and per-solve transformed-kernel reuse. The temporary module substitution is restored in `finally` and is intended only for this single-threaded experiment.

It checks nine operator shapes, five full Snowball pricing cases and 27 analytical vanilla Greek cases. It includes a cancellation control that removes the analytically integrable cash-and-asset component before evaluating gamma. A discovered raw gamma cancellation failure is retained in the output as `raw_gamma_error`.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python example/quad_v2_feasibility/backend_probe.py
```

The checked results are in [results.json](results.json). The measured daily-KI price-plus-delta call fell from approximately 88 ms to 57 ms; small-case timings were essentially unchanged. Both backends still share the source reference's contract coverage and numerical domain assumptions. Equality between them is an algebraic correctness check, not independent validation of those assumptions.
