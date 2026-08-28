# Certification report: snowball-localvol-1d

Evidence digest: `931345b9ae5e684910b1e85be5f3376522af2b19ecc4548ee918764631eae06c`

Machine: `arm64` / macOS-26.6-arm64-arm-64bit - Python 3.11.8, NumPy 2.4.6, quantark `94b8e294567903893406ec8fb4a1fb747bf73d39`

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
| equity.snowball.localvol_pde | crash_ordinary | delta | 0.597314 | 0.00487 | 0.591502 | -0.005812 | 0.01555 | 0.005503 | PASS |
| equity.snowball.localvol_pde | crash_ordinary | gamma | -0.00046755 | 0.00014 | -0.000705555 | -0.01188 | 0.02588 | 0.0002542 | PASS |
| equity.snowball.localvol_pde | crash_inside_listed_grid | pv | 4578.34 | 0.261 | 4579.26 | 0.01854 | 0.02901 | 0.0009604 | PASS |
| equity.snowball.localvol_pde | crash_inside_listed_grid | delta | 0.584974 | 0.00463 | 0.585929 | 0.0009555 | 0.01021 | 0.003764 | PASS |
| equity.snowball.localvol_pde | crash_inside_listed_grid | gamma | -0.000673772 | 0.000188 | -0.000701153 | -0.001367 | 0.02019 | 0.000172 | PASS |
| equity.snowball.localvol_pde | crash_near_ko | pv | 4639.39 | 0.666 | 4639.18 | -0.004127 | 0.0308 | 0.0007771 | PASS |
| equity.snowball.localvol_pde | crash_near_ko | delta | 0.507707 | 0.00705 | 0.505443 | -0.002264 | 0.01637 | 0.001952 | PASS |
| equity.snowball.localvol_pde | crash_near_ko | gamma | -0.000874733 | 0.00018 | -0.0007042 | 0.008515 | 0.02651 | 2.919e-05 | PASS |
| equity.snowball.localvol_pde | crash_near_ki | pv | 4034.54 | 0.835 | 4034.8 | 0.005286 | 0.03875 | 7.356e-05 | PASS |
| equity.snowball.localvol_pde | crash_near_ki | delta | 0.876515 | 0.00156 | 0.87404 | -0.002475 | 0.005597 | 0.0002553 | PASS |
| equity.snowball.localvol_pde | crash_near_ki | gamma | -0.000364237 | 0.000215 | -0.000169123 | 0.009742 | 0.0312 | 5.615e-05 | PASS |
| equity.snowball.localvol_pde | crash_discrete_ki | pv | 4581.18 | 0.614 | 4579.13 | -0.04118 | 0.06577 | 0.0006875 | PASS |
| equity.snowball.localvol_pde | crash_discrete_ki | delta | 0.60551 | 0.00246 | 0.588164 | -0.01735 | 0.02227 | 0.005587 | PASS |
| equity.snowball.localvol_pde | crash_discrete_ki | gamma | -0.00124101 | 0.000238 | -0.000715948 | 0.02622 | 0.04994 | 0.000246 | PASS |
| equity.snowball.localvol_pde | crash_european_ki | pv | 4666.66 | 1.24 | 4671.21 | 0.09099 | 0.1407 | 0.004719 | PASS |
| equity.snowball.localvol_pde | crash_european_ki | delta | 0.526647 | 0.00308 | 0.497592 | -0.02905 | 0.03521 | 0.004744 | PASS |
| equity.snowball.localvol_pde | crash_european_ki | gamma | -0.000976611 | 0.00012 | -0.000660391 | 0.01579 | 0.02777 | 0.0002181 | PASS |
| equity.snowball.localvol_pde | crash_stepdown_ko | pv | 4600.92 | 1.55 | 4600.73 | -0.003831 | 0.06596 | 0.0007467 | PASS |
| equity.snowball.localvol_pde | crash_stepdown_ko | delta | 0.571464 | 0.00299 | 0.573569 | 0.002105 | 0.008086 | 0.001274 | PASS |
| equity.snowball.localvol_pde | crash_stepdown_ko | gamma | -0.000706261 | 0.000294 | -0.000712652 | -0.0003191 | 0.02969 | 4.903e-05 | PASS |
| equity.snowball.localvol_pde | crash_near_expiry | pv | 4707.05 | 0.197 | 4706.15 | -0.01818 | 0.02608 | 0.005083 | PASS |
| equity.snowball.localvol_pde | crash_near_expiry | delta | 0.553424 | 0.00181 | 0.548433 | -0.00499 | 0.00861 | 0.003797 | PASS |
| equity.snowball.localvol_pde | crash_near_expiry | gamma | -0.000724003 | 0.000226 | -0.000897094 | -0.008643 | 0.03117 | 4.364e-05 | PASS |
| equity.snowball.localvol_pde | calm_ordinary | pv | 4889.97 | 0.422 | 4888.77 | -0.02395 | 0.04086 | 0.002221 | PASS |
| equity.snowball.localvol_pde | calm_ordinary | delta | 0.579795 | 0.00504 | 0.583075 | 0.00328 | 0.01336 | 0.001862 | PASS |
| equity.snowball.localvol_pde | calm_ordinary | gamma | -0.00131317 | 0.000133 | -0.00141891 | -0.00528 | 0.0186 | 2.543e-05 | PASS |
| equity.snowball.localvol_pde | calm_inside_listed_grid | pv | 4978.21 | 0.806 | 4978.8 | 0.01169 | 0.04399 | 0.0009868 | PASS |
| equity.snowball.localvol_pde | calm_inside_listed_grid | delta | 0.457397 | 0.00489 | 0.444558 | -0.01284 | 0.02262 | 0.004761 | PASS |
| equity.snowball.localvol_pde | calm_inside_listed_grid | gamma | -0.00167567 | 7.32e-05 | -0.00183039 | -0.007725 | 0.01504 | 0.0002921 | PASS |
| equity.snowball.localvol_pde | calm_near_ko | pv | 4963.43 | 0.51 | 4962.79 | -0.01283 | 0.03326 | 0.0006194 | PASS |
| equity.snowball.localvol_pde | calm_near_ko | delta | 0.369924 | 0.00348 | 0.385287 | 0.01536 | 0.02233 | 0.01787 | PASS |
| equity.snowball.localvol_pde | calm_near_ko | gamma | -0.00116247 | 0.000239 | -0.00128121 | -0.005929 | 0.02981 | 0.001422 | PASS |
| equity.snowball.localvol_pde | calm_near_ki | pv | 4069.22 | 0.549 | 4068.72 | -0.01009 | 0.03208 | 0.005108 | PASS |
| equity.snowball.localvol_pde | calm_near_ki | delta | 1.04989 | 0.0027 | 1.02717 | -0.02272 | 0.02811 | 0.009737 | PASS |
| equity.snowball.localvol_pde | calm_near_ki | gamma | -0.000176712 | 0.000134 | 0.00113328 | 0.06541 | 0.0788 | 0.01549 | PASS |
| equity.snowball.localvol_pde | calm_discrete_ki | pv | 4949.12 | 1.19 | 4950.21 | 0.02178 | 0.06932 | 0.000283 | PASS |
| equity.snowball.localvol_pde | calm_discrete_ki | delta | 0.47076 | 0.00661 | 0.470044 | -0.0007163 | 0.01394 | 0.001875 | PASS |
| equity.snowball.localvol_pde | calm_discrete_ki | gamma | -0.00135738 | 7.79e-05 | -0.00155234 | -0.009734 | 0.01752 | 7.367e-05 | PASS |
| equity.snowball.localvol_pde | calm_european_ki | pv | 5045.68 | 0.968 | 5053.93 | 0.1653 | 0.2041 | 0.00152 | PASS |
| equity.snowball.localvol_pde | calm_european_ki | delta | 0.245769 | 0.00713 | 0.204609 | -0.04116 | 0.05541 | 0.001787 | PASS |
| equity.snowball.localvol_pde | calm_european_ki | gamma | -0.0013305 | 0.00014 | -0.0012567 | 0.003685 | 0.0177 | 0.0001175 | PASS |
| equity.snowball.localvol_pde | calm_stepdown_ko | pv | 4896.98 | 0.298 | 4895.92 | -0.02128 | 0.03322 | 0.0001599 | PASS |
| equity.snowball.localvol_pde | calm_stepdown_ko | delta | 0.555371 | 0.00648 | 0.557647 | 0.002276 | 0.01524 | 0.002052 | PASS |
| equity.snowball.localvol_pde | calm_stepdown_ko | gamma | -0.00109511 | 0.000138 | -0.00130648 | -0.01055 | 0.02438 | 3.485e-06 | PASS |
| equity.snowball.localvol_pde | calm_near_expiry | pv | 5376.42 | 0.0986 | 5375.44 | -0.01955 | 0.0235 | 0.005366 | PASS |
| equity.snowball.localvol_pde | calm_near_expiry | delta | -0.154595 | 0.00544 | -0.172648 | -0.01805 | 0.02892 | 0.003348 | PASS |
| equity.snowball.localvol_pde | calm_near_expiry | gamma | -0.00372323 | 0.000151 | -0.00352168 | 0.01006 | 0.02518 | 0.0006671 | PASS |

## Aggregate bias

| candidate | quantity | cells | mean bias (c) | SE (c) | passed |
|---|---|---|---|---|---|
| equity.snowball.localvol_pde | pv | 16 | 0.01039 | 0.00399 | yes |
| equity.snowball.localvol_pde | delta | 16 | -0.008341 | 0.00119 | yes |
| equity.snowball.localvol_pde | gamma | 16 | 0.004874 | 0.00222 | yes |
