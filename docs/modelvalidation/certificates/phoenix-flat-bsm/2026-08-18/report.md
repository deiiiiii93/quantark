# Certification report: phoenix-flat-bsm

Evidence digest: `ef6f1040db99998eb399110402df1151caf27b9d4dc1aa6484fcafa12615e3dc`

Machine: `arm64` / macOS-26.6-arm64-arm-64bit - Python 3.11.8, NumPy 2.4.6, quantark `5521168b5cec0dcfb4f7051dd409a32728353cf4`

## Decisions

| candidate | decision |
|---|---|
| equity.phoenix.pde | ADMITTED |
| equity.phoenix.quad | ADMITTED |

Bounds: cell 0.5 c, mean signed bias 0.1 c, standard-error budget 0.25 x cell, interval k 2.

## Engine configuration

Resolved rather than named: a profile such as `standard` is an indirection whose meaning can change between releases. These are the requested settings.

| engine | setting | value |
|---|---|---|
| equity.phoenix.pde | accuracy | standard |
| equity.phoenix.pde | engine | PhoenixPDESolver |
| equity.phoenix.pde | grid.bounds | [None, None] |
| equity.phoenix.pde | grid.day_count | 252 |
| equity.phoenix.pde | grid.eps_crit | 0.003 |
| equity.phoenix.pde | grid.event_damping_steps | 2 |
| equity.phoenix.pde | grid.max_points | 2000 |
| equity.phoenix.pde | grid.max_steps | 5000 |
| equity.phoenix.pde | grid.num_std | 4 |
| equity.phoenix.pde | grid.points | 400 |
| equity.phoenix.pde | grid.steps_per_day | 4 |
| equity.phoenix.pde | grid.terminal_damping_steps | 1 |
| equity.phoenix.quad | engine | PhoenixQuadEngine |
| equity.phoenix.quad | grid.align_priority | auto |
| equity.phoenix.quad | grid.barrier_reach_stddevs | -- |
| equity.phoenix.quad | grid.bgk_min_ki_observations | 100 |
| equity.phoenix.quad | grid.bus_days_in_year | 252 |
| equity.phoenix.quad | grid.event_projection | cell_average |
| equity.phoenix.quad | grid.event_smoothing_cells | 1 |
| equity.phoenix.quad | grid.event_smoothing_kernel | cosine |
| equity.phoenix.quad | grid.event_smoothing_log_width | 0.002 |
| equity.phoenix.quad | grid.event_smoothing_mode | fixed |
| equity.phoenix.quad | grid.fft_filter_alpha | 12 |
| equity.phoenix.quad | grid.fft_filter_power | 8 |
| equity.phoenix.quad | grid.fft_padding_factor | 2 |
| equity.phoenix.quad | grid.filter_unreachable_barriers | True |
| equity.phoenix.quad | grid.grid_points | 1001 |
| equity.phoenix.quad | grid.integration_rule | trapezoid |
| equity.phoenix.quad | grid.ki_monitoring_mode | exact_discrete |
| equity.phoenix.quad | grid.max_adaptive_grid_points | 5001 |
| equity.phoenix.quad | grid.min_diffusion_stddev_cells | 2.5 |
| equity.phoenix.quad | grid.num_std_devs | 10 |
| equity.phoenix.quad | grid.stability_preset | -- |
| equity.phoenix.quad | grid_points | 1001 |
| (benchmark) | engine | PhoenixMCEngine |
| (benchmark) | greeks | paired central difference (common random numbers) |
| (benchmark) | method | randomized_quasi |
| (benchmark) | paths_per_batch | 65536 |

## Benchmark sampling

| case | batches | stopped because | standard errors (raw) |
|---|---|---|---|
| low_vol | 4 | se_budget_met | delta: 0.00269, gamma: 0.00211, pv: 0.00105 |
| memory | 4 | se_budget_met | delta: 0.000903, gamma: 0.0044, pv: 0.00452 |
| near_coupon | 4 | se_budget_met | delta: 0.00425, gamma: 0.0216, pv: 0.00517 |
| near_expiry | 4 | se_budget_met | delta: 0.000659, gamma: 0.00317, pv: 0.00199 |
| near_ki | 4 | se_budget_met | delta: 0.00424, gamma: 0.00491, pv: 0.00146 |
| near_ko | 4 | se_budget_met | delta: 0.00292, gamma: 0.00687, pv: 0.00358 |
| ordinary | 4 | se_budget_met | delta: 0.000842, gamma: 0.00451, pv: 0.00424 |

Sampling policy: 65536 paths/batch, 4-32 batches, seed 20260818, bump 0.01.

## Cells

| candidate | case | quantity | reference | SE | candidate | err (c) | interval (c) | envelope (c) | verdict |
|---|---|---|---|---|---|---|---|---|---|
| equity.phoenix.pde | ordinary | pv | -3.42614 | 0.00424 | -3.377 | 0.04913 | 0.05762 | 0.02569 | PASS |
| equity.phoenix.pde | ordinary | delta | 0.488242 | 0.000842 | 0.4795 | -0.008742 | 0.01043 | 0.001017 | PASS |
| equity.phoenix.pde | ordinary | gamma | -0.0432388 | 0.00451 | -0.0353733 | 0.007865 | 0.01688 | 0.0001197 | PASS |
| equity.phoenix.pde | near_ko | pv | -2.31343 | 0.00358 | -2.27137 | 0.04206 | 0.04923 | 0.01038 | PASS |
| equity.phoenix.pde | near_ko | delta | 0.400328 | 0.00292 | 0.38753 | -0.0128 | 0.01864 | 0.001067 | PASS |
| equity.phoenix.pde | near_ko | gamma | -0.0493471 | 0.00687 | -0.0376711 | 0.01168 | 0.02543 | 8.99e-05 | PASS |
| equity.phoenix.pde | near_coupon | pv | -13.8608 | 0.00517 | -13.7118 | 0.149 | 0.1594 | 0.06581 | PASS |
| equity.phoenix.pde | near_coupon | delta | 0.948728 | 0.00425 | 0.950314 | 0.001586 | 0.01008 | 0.004063 | PASS |
| equity.phoenix.pde | near_coupon | gamma | -0.0437926 | 0.0216 | -0.0306571 | 0.01314 | 0.05629 | 0.0003097 | PASS |
| equity.phoenix.pde | near_ki | pv | -23.9184 | 0.00146 | -23.8499 | 0.06852 | 0.07144 | 0.002752 | PASS |
| equity.phoenix.pde | near_ki | delta | 1.13636 | 0.00424 | 1.14144 | 0.005078 | 0.01355 | 0.00027 | PASS |
| equity.phoenix.pde | near_ki | gamma | -0.00717388 | 0.00491 | -0.00759895 | -0.0004251 | 0.01025 | 6.958e-05 | PASS |
| equity.phoenix.pde | low_vol | pv | 0.526131 | 0.00105 | 0.555817 | 0.02969 | 0.03178 | 0.009772 | PASS |
| equity.phoenix.pde | low_vol | delta | 0.00547519 | 0.00269 | -0.00588449 | -0.01136 | 0.01675 | 0.009467 | PASS |
| equity.phoenix.pde | low_vol | gamma | -0.0266412 | 0.00211 | -0.0308233 | -0.004182 | 0.008395 | 0.002205 | PASS |
| equity.phoenix.pde | near_expiry | pv | 0.0923547 | 0.00199 | 0.107903 | 0.01555 | 0.01952 | 0.001316 | PASS |
| equity.phoenix.pde | near_expiry | delta | 0.0556837 | 0.000659 | 0.0489303 | -0.006753 | 0.008071 | 0.001045 | PASS |
| equity.phoenix.pde | near_expiry | gamma | -0.0205876 | 0.00317 | -0.0169273 | 0.00366 | 0.009995 | 0.0002263 | PASS |
| equity.phoenix.pde | memory | pv | -3.34946 | 0.00452 | -3.30045 | 0.04901 | 0.05806 | 0.02569 | PASS |
| equity.phoenix.pde | memory | delta | 0.477563 | 0.000903 | 0.468882 | -0.008681 | 0.01049 | 0.001064 | PASS |
| equity.phoenix.pde | memory | gamma | -0.0428127 | 0.0044 | -0.0344852 | 0.008327 | 0.01713 | 0.00012 | PASS |
| equity.phoenix.quad | ordinary | pv | -3.42614 | 0.00424 | -3.42618 | -3.764e-05 | 0.008526 | 0.004547 | PASS |
| equity.phoenix.quad | ordinary | delta | 0.488242 | 0.000842 | 0.487735 | -0.0005078 | 0.002192 | 0.003148 | PASS |
| equity.phoenix.quad | ordinary | gamma | -0.0432388 | 0.00451 | -0.0332451 | 0.009994 | 0.01901 | 0.0001285 | PASS |
| equity.phoenix.quad | near_ko | pv | -2.31343 | 0.00358 | -2.32044 | -0.007007 | 0.01418 | 0.007224 | PASS |
| equity.phoenix.quad | near_ko | delta | 0.400328 | 0.00292 | 0.395608 | -0.00472 | 0.01056 | 0.002238 | PASS |
| equity.phoenix.quad | near_ko | gamma | -0.0493471 | 0.00687 | -0.0379981 | 0.01135 | 0.0251 | 0.002753 | PASS |
| equity.phoenix.quad | near_coupon | pv | -13.8608 | 0.00517 | -13.8661 | -0.005296 | 0.01564 | 0.004035 | PASS |
| equity.phoenix.quad | near_coupon | delta | 0.948728 | 0.00425 | 0.947138 | -0.00159 | 0.01009 | 5.253e-06 | PASS |
| equity.phoenix.quad | near_coupon | gamma | -0.0437926 | 0.0216 | -0.0308271 | 0.01297 | 0.05612 | 0.00295 | PASS |
| equity.phoenix.quad | near_ki | pv | -23.9184 | 0.00146 | -23.9187 | -0.0002812 | 0.003205 | 0.0009991 | PASS |
| equity.phoenix.quad | near_ki | delta | 1.13636 | 0.00424 | 1.13168 | -0.004682 | 0.01315 | 0.0006387 | PASS |
| equity.phoenix.quad | near_ki | gamma | -0.00717388 | 0.00491 | -0.00610506 | 0.001069 | 0.01089 | 0.001399 | PASS |
| equity.phoenix.quad | low_vol | pv | 0.526131 | 0.00105 | 0.52733 | 0.001198 | 0.003292 | 0.004685 | PASS |
| equity.phoenix.quad | low_vol | delta | 0.00547519 | 0.00269 | 0.00550298 | 2.779e-05 | 0.005414 | 0.00173 | PASS |
| equity.phoenix.quad | low_vol | gamma | -0.0266412 | 0.00211 | -0.0319791 | -0.005338 | 0.009551 | 0.0002637 | PASS |
| equity.phoenix.quad | near_expiry | pv | 0.0923547 | 0.00199 | 0.0912468 | -0.001108 | 0.005079 | 0.0003531 | PASS |
| equity.phoenix.quad | near_expiry | delta | 0.0556837 | 0.000659 | 0.0554577 | -0.000226 | 0.001544 | 3.223e-06 | PASS |
| equity.phoenix.quad | near_expiry | gamma | -0.0205876 | 0.00317 | -0.0183734 | 0.002214 | 0.008549 | 0.0002962 | PASS |
| equity.phoenix.quad | memory | pv | -3.34946 | 0.00452 | -3.34957 | -0.0001063 | 0.009154 | 0.004358 | PASS |
| equity.phoenix.quad | memory | delta | 0.477563 | 0.000903 | 0.477033 | -0.0005301 | 0.002337 | 0.003084 | PASS |
| equity.phoenix.quad | memory | gamma | -0.0428127 | 0.0044 | -0.0324083 | 0.0104 | 0.01921 | 0.0001341 | PASS |

## Aggregate bias

| candidate | quantity | cells | mean bias (c) | SE (c) | passed |
|---|---|---|---|---|---|
| equity.phoenix.pde | pv | 7 | 0.05757 | 0.00132 | yes |
| equity.phoenix.pde | delta | 7 | -0.005953 | 0.00105 | yes |
| equity.phoenix.pde | gamma | 7 | 0.005723 | 0.00347 | yes |
| equity.phoenix.quad | pv | 7 | -0.001805 | 0.00132 | yes |
| equity.phoenix.quad | delta | 7 | -0.001747 | 0.00105 | yes |
| equity.phoenix.quad | gamma | 7 | 0.006094 | 0.00347 | yes |
