# Certification report: snowball-localvol-1d

Evidence digest: `c0d62f9f4c0986a93ff0f850a32cba4577422ae99d48deab6142ff3673b79a6c`

Machine: `arm64` / macOS-27.0-arm64-arm-64bit - Python 3.11.8, NumPy 2.4.6, quantark `ff27bb1dcbba72ec8fe757e6676a450992c3ddcc`

## Decisions

| candidate | decision |
|---|---|
| equity.snowball.localvol_pde | ADMITTED |

Bounds: cell 0.5 c, mean signed bias 0.1 c, standard-error budget 0.25 x cell, interval k 2.

## Engine configuration

Resolved rather than named: a profile such as `standard` is an indirection whose meaning can change between releases. These are the requested settings.

| engine | setting | value |
|---|---|---|
| equity.snowball.localvol_pde | accuracy | standard |
| equity.snowball.localvol_pde | engine | LocalVolSnowballPDESolver |
| equity.snowball.localvol_pde | grid.bounds | [None, None] |
| equity.snowball.localvol_pde | grid.day_count | 252 |
| equity.snowball.localvol_pde | grid.eps_crit | 0.003 |
| equity.snowball.localvol_pde | grid.event_damping_steps | 2 |
| equity.snowball.localvol_pde | grid.max_points | 2000 |
| equity.snowball.localvol_pde | grid.max_steps | 5000 |
| equity.snowball.localvol_pde | grid.num_std | 4 |
| equity.snowball.localvol_pde | grid.points | 400 |
| equity.snowball.localvol_pde | grid.steps_per_day | 4 |
| equity.snowball.localvol_pde | grid.terminal_damping_steps | 1 |
| (benchmark) | engine | LocalVolSnowballMCEngine |
| (benchmark) | estimator | plain |
| (benchmark) | greeks | paired central difference (common random numbers) |
| (benchmark) | lv_time_sampling | integrated |
| (benchmark) | method | randomized_quasi |
| (benchmark) | paths_per_batch | 65536 |
| (benchmark) | substeps_per_interval | 8 |

## Benchmark sampling

| case | batches | stopped because | standard errors (raw) |
|---|---|---|---|
| calm_discrete_ki | 4 | se_budget_met | delta: 0.00661, gamma: 7.79e-05, pv: 1.19 |
| calm_european_ki | 4 | se_budget_met | delta: 0.00713, gamma: 0.00014, pv: 0.968 |
| calm_inside_listed_grid | 4 | se_budget_met | delta: 0.00489, gamma: 7.32e-05, pv: 0.806 |
| calm_near_expiry | 4 | se_budget_met | delta: 0.00544, gamma: 0.000151, pv: 0.0986 |
| calm_near_ki | 4 | se_budget_met | delta: 0.0027, gamma: 0.000134, pv: 0.549 |
| calm_near_ko | 4 | se_budget_met | delta: 0.00348, gamma: 0.000239, pv: 0.51 |
| calm_ordinary | 4 | se_budget_met | delta: 0.00504, gamma: 0.000133, pv: 0.422 |
| calm_stepdown_ko | 4 | se_budget_met | delta: 0.00648, gamma: 0.000138, pv: 0.298 |
| crash_discrete_ki | 4 | se_budget_met | delta: 0.00246, gamma: 0.000238, pv: 0.614 |
| crash_european_ki | 4 | se_budget_met | delta: 0.00308, gamma: 0.00012, pv: 1.24 |
| crash_inside_listed_grid | 4 | se_budget_met | delta: 0.00463, gamma: 0.000188, pv: 0.261 |
| crash_near_expiry | 4 | se_budget_met | delta: 0.00181, gamma: 0.000226, pv: 0.197 |
| crash_near_ki | 4 | se_budget_met | delta: 0.00156, gamma: 0.000215, pv: 0.835 |
| crash_near_ko | 4 | se_budget_met | delta: 0.00705, gamma: 0.00018, pv: 0.666 |
| crash_ordinary | 4 | se_budget_met | delta: 0.00487, gamma: 0.00014, pv: 0.864 |
| crash_stepdown_ko | 4 | se_budget_met | delta: 0.00299, gamma: 0.000294, pv: 1.55 |

Sampling policy: 65536 paths/batch, 4-32 batches, seed 20260828, bump 0.01.

## Cells

| candidate | case | quantity | reference | SE | candidate | err (c) | interval (c) | envelope (c) | verdict |
|---|---|---|---|---|---|---|---|---|---|
| equity.snowball.localvol_pde | crash_ordinary | pv | 4570.2 | 0.864 | 4570.59 | 0.007682 | 0.04229 | 0.000852 | PASS |
| equity.snowball.localvol_pde | crash_ordinary | delta | 0.597314 | 0.00487 | 0.593787 | -0.003528 | 0.01327 | 3.207e-05 | PASS |
| equity.snowball.localvol_pde | crash_ordinary | gamma | -0.00046755 | 0.00014 | -0.000703484 | -0.01178 | 0.02578 | 3.768e-06 | PASS |
| equity.snowball.localvol_pde | crash_inside_listed_grid | pv | 4578.34 | 0.261 | 4579.26 | 0.01854 | 0.02901 | 0.0009604 | PASS |
| equity.snowball.localvol_pde | crash_inside_listed_grid | delta | 0.584974 | 0.00463 | 0.586331 | 0.001358 | 0.01062 | 2.524e-05 | PASS |
| equity.snowball.localvol_pde | crash_inside_listed_grid | gamma | -0.000673772 | 0.000188 | -0.000700818 | -0.00135 | 0.02017 | 3.531e-06 | PASS |
| equity.snowball.localvol_pde | crash_near_ko | pv | 4639.39 | 0.666 | 4639.18 | -0.004127 | 0.0308 | 0.0007771 | PASS |
| equity.snowball.localvol_pde | crash_near_ko | delta | 0.507707 | 0.00705 | 0.50496 | -0.002747 | 0.01685 | 7.134e-05 | PASS |
| equity.snowball.localvol_pde | crash_near_ko | gamma | -0.000874733 | 0.00018 | -0.000703742 | 0.008538 | 0.02653 | 1.862e-05 | PASS |
| equity.snowball.localvol_pde | crash_near_ki | pv | 4034.54 | 0.835 | 4034.8 | 0.005286 | 0.03875 | 7.356e-05 | PASS |
| equity.snowball.localvol_pde | crash_near_ki | delta | 0.876515 | 0.00156 | 0.874168 | -0.002347 | 0.005469 | 4.438e-05 | PASS |
| equity.snowball.localvol_pde | crash_near_ki | gamma | -0.000364237 | 0.000215 | -0.00016881 | 0.009758 | 0.03122 | 1.435e-05 | PASS |
| equity.snowball.localvol_pde | crash_discrete_ki | pv | 4581.18 | 0.614 | 4579.13 | -0.04118 | 0.06577 | 0.0006875 | PASS |
| equity.snowball.localvol_pde | crash_discrete_ki | delta | 0.60551 | 0.00246 | 0.590481 | -0.01503 | 0.01996 | 3.405e-05 | PASS |
| equity.snowball.localvol_pde | crash_discrete_ki | gamma | -0.00124101 | 0.000238 | -0.000713949 | 0.02632 | 0.05004 | 4.535e-06 | PASS |
| equity.snowball.localvol_pde | crash_european_ki | pv | 4666.66 | 1.24 | 4671.21 | 0.09099 | 0.1407 | 0.004719 | PASS |
| equity.snowball.localvol_pde | crash_european_ki | delta | 0.526647 | 0.00308 | 0.499726 | -0.02692 | 0.03307 | 0.0003774 | PASS |
| equity.snowball.localvol_pde | crash_european_ki | gamma | -0.000976611 | 0.00012 | -0.000658987 | 0.01586 | 0.02784 | 4.943e-05 | PASS |
| equity.snowball.localvol_pde | crash_stepdown_ko | pv | 4600.92 | 1.55 | 4600.73 | -0.003831 | 0.06596 | 0.0007467 | PASS |
| equity.snowball.localvol_pde | crash_stepdown_ko | delta | 0.571464 | 0.00299 | 0.572083 | 0.0006185 | 0.0066 | 7.625e-05 | PASS |
| equity.snowball.localvol_pde | crash_stepdown_ko | gamma | -0.000706261 | 0.000294 | -0.000713626 | -0.0003677 | 0.02974 | 6.111e-06 | PASS |
| equity.snowball.localvol_pde | crash_near_expiry | pv | 4707.05 | 0.197 | 4706.15 | -0.01818 | 0.02608 | 0.005083 | PASS |
| equity.snowball.localvol_pde | crash_near_expiry | delta | 0.553424 | 0.00181 | 0.554559 | 0.001135 | 0.004754 | 2.46e-05 | PASS |
| equity.snowball.localvol_pde | crash_near_expiry | gamma | -0.000724003 | 0.000226 | -0.000894699 | -0.008523 | 0.03105 | 3.117e-05 | PASS |
| equity.snowball.localvol_pde | calm_ordinary | pv | 4889.97 | 0.422 | 4888.77 | -0.02395 | 0.04086 | 0.002221 | PASS |
| equity.snowball.localvol_pde | calm_ordinary | delta | 0.579795 | 0.00504 | 0.586501 | 0.006706 | 0.01678 | 0.0001488 | PASS |
| equity.snowball.localvol_pde | calm_ordinary | gamma | -0.00131317 | 0.000133 | -0.00141773 | -0.005221 | 0.01854 | 3.638e-05 | PASS |
| equity.snowball.localvol_pde | calm_inside_listed_grid | pv | 4978.21 | 0.806 | 4978.8 | 0.01169 | 0.04399 | 0.0009868 | PASS |
| equity.snowball.localvol_pde | calm_inside_listed_grid | delta | 0.457397 | 0.00489 | 0.460185 | 0.002789 | 0.01257 | 6.524e-06 | PASS |
| equity.snowball.localvol_pde | calm_inside_listed_grid | gamma | -0.00167567 | 7.32e-05 | -0.00184639 | -0.008524 | 0.01584 | 2.687e-05 | PASS |
| equity.snowball.localvol_pde | calm_near_ko | pv | 4963.43 | 0.51 | 4962.79 | -0.01283 | 0.03326 | 0.0006194 | PASS |
| equity.snowball.localvol_pde | calm_near_ko | delta | 0.369924 | 0.00348 | 0.373968 | 0.004044 | 0.01101 | 5.054e-05 | PASS |
| equity.snowball.localvol_pde | calm_near_ko | gamma | -0.00116247 | 0.000239 | -0.00126321 | -0.00503 | 0.02891 | 8.446e-06 | PASS |
| equity.snowball.localvol_pde | calm_near_ki | pv | 4069.22 | 0.549 | 4068.72 | -0.01009 | 0.03208 | 0.005108 | PASS |
| equity.snowball.localvol_pde | calm_near_ki | delta | 1.04989 | 0.0027 | 1.0315 | -0.01838 | 0.02377 | 0.0005266 | PASS |
| equity.snowball.localvol_pde | calm_near_ki | gamma | -0.000176712 | 0.000134 | 0.00104599 | 0.06105 | 0.07444 | 4.73e-05 | PASS |
| equity.snowball.localvol_pde | calm_discrete_ki | pv | 4949.12 | 1.19 | 4950.21 | 0.02178 | 0.06932 | 0.000283 | PASS |
| equity.snowball.localvol_pde | calm_discrete_ki | delta | 0.47076 | 0.00661 | 0.473734 | 0.002973 | 0.01619 | 1.66e-05 | PASS |
| equity.snowball.localvol_pde | calm_discrete_ki | gamma | -0.00135738 | 7.79e-05 | -0.00155371 | -0.009803 | 0.01758 | 7.653e-06 | PASS |
| equity.snowball.localvol_pde | calm_european_ki | pv | 5045.68 | 0.968 | 5053.93 | 0.1653 | 0.2041 | 0.00152 | PASS |
| equity.snowball.localvol_pde | calm_european_ki | delta | 0.245769 | 0.00713 | 0.207569 | -0.0382 | 0.05245 | 0.0002493 | PASS |
| equity.snowball.localvol_pde | calm_european_ki | gamma | -0.0013305 | 0.00014 | -0.00125905 | 0.003567 | 0.01758 | 1.608e-05 | PASS |
| equity.snowball.localvol_pde | calm_stepdown_ko | pv | 4896.98 | 0.298 | 4895.92 | -0.02128 | 0.03322 | 0.0001599 | PASS |
| equity.snowball.localvol_pde | calm_stepdown_ko | delta | 0.555371 | 0.00648 | 0.553516 | -0.001855 | 0.01482 | 4.347e-05 | PASS |
| equity.snowball.localvol_pde | calm_stepdown_ko | gamma | -0.00109511 | 0.000138 | -0.00130676 | -0.01057 | 0.02439 | 8.549e-07 | PASS |
| equity.snowball.localvol_pde | calm_near_expiry | pv | 5376.42 | 0.0986 | 5375.44 | -0.01955 | 0.0235 | 0.005366 | PASS |
| equity.snowball.localvol_pde | calm_near_expiry | delta | -0.154595 | 0.00544 | -0.155308 | -0.0007133 | 0.01158 | 0.001084 | PASS |
| equity.snowball.localvol_pde | calm_near_expiry | gamma | -0.00372323 | 0.000151 | -0.00358311 | 0.006996 | 0.02212 | 0.0002811 | PASS |

## Aggregate bias

| candidate | quantity | cells | mean bias (c) | SE (c) | passed |
|---|---|---|---|---|---|
| equity.snowball.localvol_pde | pv | 16 | 0.01039 | 0.00399 | yes |
| equity.snowball.localvol_pde | delta | 16 | -0.005631 | 0.00119 | yes |
| equity.snowball.localvol_pde | gamma | 16 | 0.004432 | 0.00222 | yes |
