# Certification report: adi2d-snowball-greeks

Evidence digest: `cd5b2d4724cb7bf47f834300f69e4c0fbe16457f6510a71a5a0e21585e7d30cb`

Machine: `arm64` / macOS-27.0-arm64-arm-64bit - Python 3.11.8, NumPy 2.4.6, quantark `d2a7976af2a37f4f640e339ce73316cd7c11fe67`

## Amendment

| field | value |
|---|---|
| parent | docs/modelvalidation/certificates/adi2d-snowball-greeks/2026-10-10/certificate.json |
| parent digest | `d8bf666d9446510bbabcb08c154b40419cd729cc251d27a3d677297e3dc50eb3` |
| reason | 79938a1c moved 2D Heston/SLV snowball numerics: deterministic discounting integrated exponentially outside the ADI generator, and a fixed-weight characteristic variance transport so added outcome cash reconciles with its indicator. Its linear weights were first order and moved the sigma_collapse delta 11 reference SE off the MC reference; the transport now uses the core's fixed 4-point Lagrange weights unclipped (still linear in the claim, third order). Re-run every candidate cell against the banked reference. The harness moved under e267d78c (example/ to example/case_studies/), recorded as amendment.harness_relocated. |
| re-priced | 28 cell(s) |
| carried forward | 0 cell(s) |

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
| (benchmark) | harness | example/case_studies/mo_volmodels/16_adi_greek_certification.py |
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
| equity.snowball.heston_pde | ordinary_full | delta | 0.202601 | 0.000296 | 0.201906 | -0.03834 | 0.0812 | 0.003725 | PASS |
| equity.snowball.heston_pde | ordinary_full | gamma | -0.0232394 | 0.000203 | -0.022927 | 0.01723 | 0.04983 | 0.00336 | PASS |
| equity.snowball.heston_pde | ordinary_decayed | delta | 0.585689 | 0.000197 | 0.585448 | -0.01332 | 0.04555 | 0.006403 | PASS |
| equity.snowball.heston_pde | ordinary_decayed | gamma | -0.0592196 | 0.000123 | -0.0592291 | -0.0004838 | 0.01899 | 0.001014 | PASS |
| equity.snowball.heston_pde | near_ko | delta | 0.102891 | 0.000117 | 0.102567 | -0.01788 | 0.04394 | 0.01111 | PASS |
| equity.snowball.heston_pde | near_ko | gamma | -0.0141373 | 5.29e-05 | -0.0141948 | -0.003258 | 0.01281 | 0.001828 | PASS |
| equity.snowball.heston_pde | near_ki | delta | 2.05623 | 0.000293 | 2.05573 | -0.02741 | 0.1779 | 0.09499 | PASS |
| equity.snowball.heston_pde | near_ki | gamma | 0.201439 | 0.000778 | 0.205393 | 0.1646 | 0.3457 | 0.03013 | PASS |
| equity.snowball.heston_pde | low_feller | delta | -0.0108225 | 0.000559 | -0.0124946 | -0.09223 | 0.1966 | 0.01244 | PASS |
| equity.snowball.heston_pde | low_feller | gamma | 0.0200765 | 0.00021 | 0.0211045 | 0.0567 | 0.1226 | 0.01464 | PASS |
| equity.snowball.heston_pde | sigma_collapse | delta | 0.324729 | 0.000184 | 0.32474 | 0.0006465 | 0.04411 | 0.02061 | PASS |
| equity.snowball.heston_pde | sigma_collapse | gamma | -0.0331118 | 8.72e-05 | -0.0332177 | -0.005842 | 0.02505 | 0.008219 | PASS |
| equity.snowball.heston_pde | near_expiry | delta | -0.363598 | 2.79e-05 | -0.364081 | -0.02665 | 0.04588 | 0.01557 | PASS |
| equity.snowball.heston_pde | near_expiry | gamma | -0.01767 | 6.7e-06 | -0.0174555 | 0.01183 | 0.02988 | 0.01714 | PASS |
| equity.snowball.heston_slv_pde | ordinary_full | delta | 0.208715 | 0.0011 | 0.208784 | 0.003788 | 0.2106 | 0.003572 | PASS |
| equity.snowball.heston_slv_pde | ordinary_full | gamma | -0.0224045 | 0.00126 | -0.0232219 | -0.04508 | 0.3993 | 0.003388 | PASS |
| equity.snowball.heston_slv_pde | ordinary_decayed | delta | 0.597527 | 0.000755 | 0.595825 | -0.09384 | 0.2686 | 0.006291 | PASS |
| equity.snowball.heston_slv_pde | ordinary_decayed | gamma | -0.058192 | 0.00143 | -0.0582354 | -0.002199 | 0.4163 | 0.001009 | PASS |
| equity.snowball.heston_slv_pde | near_ko | delta | 0.111806 | 0.000521 | 0.112068 | 0.01445 | 0.1261 | 0.01101 | PASS |
| equity.snowball.heston_slv_pde | near_ko | gamma | -0.0148174 | 0.00083 | -0.0151511 | -0.0189 | 0.2813 | 0.001862 | PASS |
| equity.snowball.heston_slv_pde | near_ki | delta | 1.99678 | 0.000302 | 1.99617 | -0.03377 | 0.2067 | 0.08764 | PASS |
| equity.snowball.heston_slv_pde | near_ki | gamma | 0.188255 | 0.00117 | 0.191573 | 0.1382 | 0.3786 | 0.02128 | PASS |
| equity.snowball.heston_slv_pde | low_feller | delta | 0.000280896 | 0.000631 | -0.00235178 | -0.1452 | 0.2842 | 0.01219 | PASS |
| equity.snowball.heston_slv_pde | low_feller | gamma | 0.0196499 | 0.000529 | 0.0203242 | 0.03719 | 0.2007 | 0.0147 | PASS |
| equity.snowball.heston_slv_pde | sigma_collapse | delta | 0.340288 | 0.00068 | 0.339726 | -0.031 | 0.19 | 0.01983 | PASS |
| equity.snowball.heston_slv_pde | sigma_collapse | gamma | -0.0347779 | 0.0014 | -0.0332302 | 0.08537 | 0.4128 | 0.008219 | PASS |
| equity.snowball.heston_slv_pde | near_expiry | delta | -0.358612 | 0.000281 | -0.358934 | -0.01778 | 0.09512 | 0.01612 | PASS |
| equity.snowball.heston_slv_pde | near_expiry | gamma | -0.0202751 | 0.000772 | -0.0189218 | 0.07464 | 0.2719 | 0.01802 | PASS |

## Aggregate bias

| candidate | quantity | cells | mean bias (c) | SE (c) | passed |
|---|---|---|---|---|---|
| equity.snowball.heston_pde | delta | 7 | -0.03217 | 0.0269 | yes |
| equity.snowball.heston_slv_pde | delta | 7 | -0.02999 | 0.0334 | yes |
