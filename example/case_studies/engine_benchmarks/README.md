# Engine benchmarks

Timing and convergence checks for the pricing engines. Each script prints its own table.

- `cash_leg_folding_benchmark.py` — cash-leg folding on QUAD / PDE / MC
- `snowball_dask_benchmark.py`, `snowball_pde_perf_demo.py`, `pde_term_structure_benchmark.py` — PDE and parallel throughput
- `qmc_mc_convergence_gbm.py` — QMC vs MC convergence on GBM
