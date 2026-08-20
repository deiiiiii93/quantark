# Certification report: adi2d-snowball-greeks

Evidence digest: `b6c99a6c96ce6f1bdcc221f35da8f325d5f754fe290649a82ea4364add411446`

Machine: `arm64` / macOS-26.6-arm64-arm-64bit - Python 3.11.8, NumPy 2.4.6, quantark `258fd7ec39416335a00e7fad70822c15c8c1294a`

## Decisions

| candidate | decision |
|---|---|
| equity.snowball.heston_pde | ADMITTED |
| equity.snowball.heston_slv_pde | ADMITTED |

Bounds: cell 0.5 c, mean signed bias 0.1 c, standard-error budget 0.25 x cell, interval k 2.

## Engine configuration

Resolved rather than named: a profile such as `standard` is an indirection whose meaning can change between releases. These are the requested settings.

| engine | setting | value |
|---|---|---|
| equity.snowball.heston_pde | controls.barrier_greek_min_n_x | 0 |
| equity.snowball.heston_pde | controls.barrier_greek_steps_per_tick | 0 |
| equity.snowball.heston_pde | controls.greek_min_n_v | 0 |
| equity.snowball.heston_pde | controls.greek_min_n_x | 0 |
| equity.snowball.heston_pde | controls.greek_min_steps_per_year | 0 |
| equity.snowball.heston_pde | controls.grid_style | concentrated |
| equity.snowball.heston_pde | controls.v0_boundary | degenerate_pde |
| equity.snowball.heston_pde | controls.v_drift_scheme | auto |
| equity.snowball.heston_pde | controls.variance_grid_mode | auto |
| equity.snowball.heston_pde | engine | HestonSnowballPDESolver |
| equity.snowball.heston_pde | greeks | central finite bump on a frozen spatial domain |
| equity.snowball.heston_pde | grid_policy.dense_ki_stencil.n_x | 600 |
| equity.snowball.heston_pde | grid_policy.dense_ki_stencil.steps_per_ki_tick | 16 |
| equity.snowball.heston_pde | grid_policy.min_n_t | 180 |
| equity.snowball.heston_pde | grid_policy.n_v | 135 |
| equity.snowball.heston_pde | grid_policy.n_x | 300 |
| equity.snowball.heston_pde | grid_policy.steps_per_year | 1600 |
| equity.snowball.heston_pde | spot_bump | 0.01 |
| equity.snowball.heston_slv_pde | controls.barrier_greek_min_n_x | 0 |
| equity.snowball.heston_slv_pde | controls.barrier_greek_steps_per_tick | 0 |
| equity.snowball.heston_slv_pde | controls.greek_min_n_v | 0 |
| equity.snowball.heston_slv_pde | controls.greek_min_n_x | 0 |
| equity.snowball.heston_slv_pde | controls.greek_min_steps_per_year | 0 |
| equity.snowball.heston_slv_pde | controls.grid_style | concentrated |
| equity.snowball.heston_slv_pde | controls.v0_boundary | degenerate_pde |
| equity.snowball.heston_slv_pde | controls.v_drift_scheme | auto |
| equity.snowball.heston_slv_pde | controls.variance_grid_mode | auto |
| equity.snowball.heston_slv_pde | engine | HestonSLVSnowballPDESolver |
| equity.snowball.heston_slv_pde | greeks | central finite bump on a frozen spatial domain |
| equity.snowball.heston_slv_pde | grid_policy.dense_ki_stencil.n_x | 600 |
| equity.snowball.heston_slv_pde | grid_policy.dense_ki_stencil.steps_per_ki_tick | 16 |
| equity.snowball.heston_slv_pde | grid_policy.min_n_t | 180 |
| equity.snowball.heston_slv_pde | grid_policy.n_v | 135 |
| equity.snowball.heston_slv_pde | grid_policy.n_x | 300 |
| equity.snowball.heston_slv_pde | grid_policy.steps_per_year | 1600 |
| equity.snowball.heston_slv_pde | spot_bump | 0.01 |
| (benchmark) | allocation | pilot-frozen cost-weighted Neyman, no optional stopping |
| (benchmark) | engine | QESnowballMCEngine / HestonSLVQESnowballMCEngine |
| (benchmark) | estimator | multilevel control-variate telescope with exact conditional integration of the spot factor |
| (benchmark) | external | True |
| (benchmark) | greeks | paired central difference (common random numbers) |
| (benchmark) | harness | example/mo_volmodels/16_adi_greek_certification.py |
| (benchmark) | method | randomized_quasi (scrambled Sobol, Brownian bridge) |
| (benchmark) | variance_scheme | QE-M (martingale-corrected quadratic exponential) |

## Benchmark sampling

| case | batches | stopped because | standard errors (raw) |
|---|---|---|---|
| low_feller | 1024 | declared_allocation_exhausted | delta: 0.000559, gamma: 0.00021 |
| near_expiry | 1024 | declared_allocation_exhausted | delta: 2.79e-05, gamma: 6.7e-06 |
| near_ki | 2048 | declared_allocation_exhausted | delta: 0.000293, gamma: 0.000778 |
| near_ko | 1024 | declared_allocation_exhausted | delta: 0.000117, gamma: 5.29e-05 |
| ordinary_decayed | 1024 | declared_allocation_exhausted | delta: 0.000197, gamma: 0.000123 |
| ordinary_full | 1024 | declared_allocation_exhausted | delta: 0.000296, gamma: 0.000203 |
| sigma_collapse | 1024 | declared_allocation_exhausted | delta: 0.000184, gamma: 8.72e-05 |

Sampling policy: 8192 paths/batch, 1024-2048 batches, seed 20260808, bump 0.01.

## Cells

| candidate | case | quantity | reference | SE | candidate | err (c) | interval (c) | envelope (c) | verdict |
|---|---|---|---|---|---|---|---|---|---|
| equity.snowball.heston_pde | ordinary_full | delta | 0.202601 | 0.000296 | 0.201905 | -0.03839 | 0.08125 | 0.003725 | PASS |
| equity.snowball.heston_pde | ordinary_full | gamma | -0.0232394 | 0.000203 | -0.022928 | 0.01718 | 0.04977 | 0.00336 | PASS |
| equity.snowball.heston_pde | ordinary_decayed | delta | 0.585689 | 0.000197 | 0.585459 | -0.0127 | 0.04493 | 0.006403 | PASS |
| equity.snowball.heston_pde | ordinary_decayed | gamma | -0.0592196 | 0.000123 | -0.0592281 | -0.000431 | 0.01894 | 0.001014 | PASS |
| equity.snowball.heston_pde | near_ko | delta | 0.102891 | 0.000117 | 0.102576 | -0.01741 | 0.04347 | 0.01111 | PASS |
| equity.snowball.heston_pde | near_ko | gamma | -0.0141373 | 5.29e-05 | -0.0141965 | -0.003353 | 0.01291 | 0.001828 | PASS |
| equity.snowball.heston_pde | near_ki | delta | 2.05623 | 0.000293 | 2.05573 | -0.02736 | 0.1778 | 0.09499 | PASS |
| equity.snowball.heston_pde | near_ki | gamma | 0.201439 | 0.000778 | 0.205393 | 0.1646 | 0.3457 | 0.03013 | PASS |
| equity.snowball.heston_pde | low_feller | delta | -0.0108225 | 0.000559 | -0.0127549 | -0.1066 | 0.2109 | 0.01244 | PASS |
| equity.snowball.heston_pde | low_feller | gamma | 0.0200765 | 0.00021 | 0.0211488 | 0.05915 | 0.125 | 0.01464 | PASS |
| equity.snowball.heston_pde | sigma_collapse | delta | 0.324729 | 0.000184 | 0.324769 | 0.002223 | 0.04568 | 0.02061 | PASS |
| equity.snowball.heston_pde | sigma_collapse | gamma | -0.0331118 | 8.72e-05 | -0.0332175 | -0.00583 | 0.02504 | 0.008219 | PASS |
| equity.snowball.heston_pde | near_expiry | delta | -0.363598 | 2.79e-05 | -0.36405 | -0.02496 | 0.04419 | 0.01557 | PASS |
| equity.snowball.heston_pde | near_expiry | gamma | -0.01767 | 6.7e-06 | -0.0174568 | 0.01176 | 0.02981 | 0.01714 | PASS |
| equity.snowball.heston_slv_pde | ordinary_full | delta | 0.208715 | 0.0011 | 0.208783 | 0.00373 | 0.2106 | 0.003572 | PASS |
| equity.snowball.heston_slv_pde | ordinary_full | gamma | -0.0224045 | 0.00126 | -0.0232229 | -0.04514 | 0.3993 | 0.003388 | PASS |
| equity.snowball.heston_slv_pde | ordinary_decayed | delta | 0.597527 | 0.000755 | 0.595836 | -0.09323 | 0.268 | 0.006291 | PASS |
| equity.snowball.heston_slv_pde | ordinary_decayed | gamma | -0.058192 | 0.00143 | -0.0582344 | -0.002148 | 0.4163 | 0.001009 | PASS |
| equity.snowball.heston_slv_pde | near_ko | delta | 0.111806 | 0.000521 | 0.112076 | 0.01491 | 0.1265 | 0.01101 | PASS |
| equity.snowball.heston_slv_pde | near_ko | gamma | -0.0148174 | 0.00083 | -0.0151528 | -0.019 | 0.2814 | 0.001862 | PASS |
| equity.snowball.heston_slv_pde | near_ki | delta | 1.99678 | 0.000302 | 1.99617 | -0.03372 | 0.2067 | 0.08764 | PASS |
| equity.snowball.heston_slv_pde | near_ki | gamma | 0.188255 | 0.00117 | 0.191573 | 0.1382 | 0.3786 | 0.02128 | PASS |
| equity.snowball.heston_slv_pde | low_feller | delta | 0.000280896 | 0.000631 | -0.00260352 | -0.1591 | 0.2981 | 0.01219 | PASS |
| equity.snowball.heston_slv_pde | low_feller | gamma | 0.0196499 | 0.000529 | 0.020368 | 0.03961 | 0.2031 | 0.0147 | PASS |
| equity.snowball.heston_slv_pde | sigma_collapse | delta | 0.340288 | 0.00068 | 0.339755 | -0.02942 | 0.1884 | 0.01983 | PASS |
| equity.snowball.heston_slv_pde | sigma_collapse | gamma | -0.0347779 | 0.0014 | -0.0332299 | 0.08538 | 0.4128 | 0.008219 | PASS |
| equity.snowball.heston_slv_pde | near_expiry | delta | -0.358612 | 0.000281 | -0.358903 | -0.01608 | 0.09342 | 0.01612 | PASS |
| equity.snowball.heston_slv_pde | near_expiry | gamma | -0.0202751 | 0.000772 | -0.0189231 | 0.07457 | 0.2718 | 0.01802 | PASS |

## Aggregate bias

| candidate | quantity | cells | mean bias (c) | SE (c) | passed |
|---|---|---|---|---|---|
| equity.snowball.heston_pde | delta | 7 | -0.03217 | 0.0269 | yes |
| equity.snowball.heston_slv_pde | delta | 7 | -0.02999 | 0.0334 | yes |
