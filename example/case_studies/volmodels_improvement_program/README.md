# Vol-model improvement program: research artifacts

Validation scripts behind the Heston / local-vol / SLV improvement work. Each prints its own verdict.

- `adi_cs_convergence.py` — Craig-Sneyd time self-convergence for the Heston ADI
- `fp_krylov_benchmark.py`, `fp_positivity_diagnostic.py`, `fp_trbdf2_convergence.py` — Fokker-Planck march variants
- `heston_calibration_benchmark.py` — calibration benchmark
- `volmodels_phase5_*.py` — QMC convergence, QE-M martingale, LV Rannacher, degenerate boundary, concentrated grid
