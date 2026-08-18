# Certification report: ko-reset-flat-bsm

Evidence digest: `33e1f27fdff63c2665b6342ed40150021dfaa445a86578f2571b3f03ce7be0d3`

Machine: `arm64` / macOS-26.6-arm64-arm-64bit - Python 3.11.8, NumPy 2.4.6, quantark `f470133f8b17729948189381a3684d9b7453218f`

## Decisions

| candidate | decision |
|---|---|
| equity.ko_reset_snowball.pde | ADMITTED |
| equity.ko_reset_snowball.quad | ADMITTED |

Bounds: cell 0.5 c, mean signed bias 0.1 c, standard-error budget 0.25 x cell, interval k 2.

## Engine configuration

Resolved rather than named: a profile such as `standard` is an indirection whose meaning can change between releases. These are the requested settings.

| engine | setting | value |
|---|---|---|
| equity.ko_reset_snowball.pde | accuracy | standard |
| equity.ko_reset_snowball.pde | engine | KOResetSnowballPDESolver |
| equity.ko_reset_snowball.pde | grid.bounds | [None, None] |
| equity.ko_reset_snowball.pde | grid.day_count | 252 |
| equity.ko_reset_snowball.pde | grid.eps_crit | 0.003 |
| equity.ko_reset_snowball.pde | grid.event_damping_steps | 2 |
| equity.ko_reset_snowball.pde | grid.max_points | 2000 |
| equity.ko_reset_snowball.pde | grid.max_steps | 5000 |
| equity.ko_reset_snowball.pde | grid.num_std | 4 |
| equity.ko_reset_snowball.pde | grid.points | 400 |
| equity.ko_reset_snowball.pde | grid.steps_per_day | 4 |
| equity.ko_reset_snowball.pde | grid.terminal_damping_steps | 1 |
| equity.ko_reset_snowball.quad | engine | KOResetSnowballQuadEngine |
| equity.ko_reset_snowball.quad | grid.align_priority | auto |
| equity.ko_reset_snowball.quad | grid.barrier_reach_stddevs | -- |
| equity.ko_reset_snowball.quad | grid.bgk_min_ki_observations | 100 |
| equity.ko_reset_snowball.quad | grid.bus_days_in_year | 252 |
| equity.ko_reset_snowball.quad | grid.event_projection | cell_average |
| equity.ko_reset_snowball.quad | grid.event_smoothing_cells | 1 |
| equity.ko_reset_snowball.quad | grid.event_smoothing_kernel | cosine |
| equity.ko_reset_snowball.quad | grid.event_smoothing_log_width | 0.002 |
| equity.ko_reset_snowball.quad | grid.event_smoothing_mode | fixed |
| equity.ko_reset_snowball.quad | grid.fft_filter_alpha | 12 |
| equity.ko_reset_snowball.quad | grid.fft_filter_power | 8 |
| equity.ko_reset_snowball.quad | grid.fft_padding_factor | 2 |
| equity.ko_reset_snowball.quad | grid.filter_unreachable_barriers | True |
| equity.ko_reset_snowball.quad | grid.grid_points | 1001 |
| equity.ko_reset_snowball.quad | grid.integration_rule | trapezoid |
| equity.ko_reset_snowball.quad | grid.ki_monitoring_mode | exact_discrete |
| equity.ko_reset_snowball.quad | grid.max_adaptive_grid_points | 5001 |
| equity.ko_reset_snowball.quad | grid.min_diffusion_stddev_cells | 2.5 |
| equity.ko_reset_snowball.quad | grid.num_std_devs | 10 |
| equity.ko_reset_snowball.quad | grid.stability_preset | -- |
| equity.ko_reset_snowball.quad | grid_points | 1001 |
| (benchmark) | engine | SnowballMCEngine |
| (benchmark) | greeks | paired central difference (common random numbers) |
| (benchmark) | method | randomized_quasi |
| (benchmark) | paths_per_batch | 65536 |

## Benchmark sampling

| case | batches | stopped because | standard errors (raw) |
|---|---|---|---|
| below_ki | 4 | se_budget_met | delta: 0.00477, gamma: 0.0123, pv: 0.0119 |
| discrete_ki | 4 | se_budget_met | delta: 0.00409, gamma: 0.00883, pv: 0.00999 |
| low_vol | 4 | se_budget_met | delta: 0.00292, gamma: 0.0118, pv: 0.0116 |
| near_expiry | 4 | se_budget_met | delta: 0.00117, gamma: 0.00329, pv: 0.00505 |
| near_ki | 4 | se_budget_met | delta: 0.00536, gamma: 0.0268, pv: 0.00618 |
| near_pre_ko | 4 | se_budget_met | delta: 0.00233, gamma: 0.0126, pv: 0.0132 |
| ordinary | 4 | se_budget_met | delta: 0.00868, gamma: 0.0297, pv: 0.011 |

Sampling policy: 65536 paths/batch, 4-32 batches, seed 20260818, bump 0.01.

## Cells

| candidate | case | quantity | reference | SE | candidate | err (c) | interval (c) | envelope (c) | verdict |
|---|---|---|---|---|---|---|---|---|---|
| equity.ko_reset_snowball.pde | ordinary | pv | 96.6613 | 0.011 | 96.704 | 0.04273 | 0.06475 | 0.01517 | PASS |
| equity.ko_reset_snowball.pde | ordinary | delta | 0.621389 | 0.00868 | 0.611252 | -0.01014 | 0.0275 | 0.005066 | PASS |
| equity.ko_reset_snowball.pde | ordinary | gamma | -0.0416238 | 0.0297 | -0.046614 | -0.00499 | 0.06439 | 0.0002392 | PASS |
| equity.ko_reset_snowball.pde | near_pre_ko | pv | 98.0475 | 0.0132 | 98.0694 | 0.02191 | 0.04835 | 0.002868 | PASS |
| equity.ko_reset_snowball.pde | near_pre_ko | delta | 0.487059 | 0.00233 | 0.485599 | -0.00146 | 0.006129 | 0.008528 | PASS |
| equity.ko_reset_snowball.pde | near_pre_ko | gamma | -0.034448 | 0.0126 | -0.050137 | -0.01569 | 0.04085 | 6.737e-05 | PASS |
| equity.ko_reset_snowball.pde | near_ki | pv | 79.0154 | 0.00618 | 79.0133 | -0.002097 | 0.01445 | 0.007806 | PASS |
| equity.ko_reset_snowball.pde | near_ki | delta | 1.06603 | 0.00536 | 1.04908 | -0.01695 | 0.02766 | 0.004301 | PASS |
| equity.ko_reset_snowball.pde | near_ki | gamma | -0.0382894 | 0.0268 | 0.00266685 | 0.04096 | 0.0946 | 0.000401 | PASS |
| equity.ko_reset_snowball.pde | below_ki | pv | 74.0373 | 0.0119 | 74.0367 | -0.0006474 | 0.02455 | 0.001612 | PASS |
| equity.ko_reset_snowball.pde | below_ki | delta | 1.10845 | 0.00477 | 1.1063 | -0.002142 | 0.01167 | 0.000501 | PASS |
| equity.ko_reset_snowball.pde | below_ki | gamma | -0.0131948 | 0.0123 | 0.00544677 | 0.01864 | 0.04328 | 1.701e-05 | PASS |
| equity.ko_reset_snowball.pde | low_vol | pv | 100.165 | 0.0116 | 100.193 | 0.02801 | 0.05123 | 0.02841 | PASS |
| equity.ko_reset_snowball.pde | low_vol | delta | 0.359471 | 0.00292 | 0.373215 | 0.01374 | 0.01959 | 0.01537 | PASS |
| equity.ko_reset_snowball.pde | low_vol | gamma | -0.0863362 | 0.0118 | -0.0956802 | -0.009344 | 0.03296 | 0.0008734 | PASS |
| equity.ko_reset_snowball.pde | near_expiry | pv | 99.5678 | 0.00505 | 99.6384 | 0.07065 | 0.08076 | 0.01055 | PASS |
| equity.ko_reset_snowball.pde | near_expiry | delta | 0.284584 | 0.00117 | 0.266508 | -0.01808 | 0.02041 | 0.007991 | PASS |
| equity.ko_reset_snowball.pde | near_expiry | gamma | -0.0472864 | 0.00329 | -0.0418617 | 0.005425 | 0.01201 | 0.0009397 | PASS |
| equity.ko_reset_snowball.pde | discrete_ki | pv | 96.7295 | 0.00999 | 96.7322 | 0.002688 | 0.02268 | 9.864e-06 | PASS |
| equity.ko_reset_snowball.pde | discrete_ki | delta | 0.612278 | 0.00409 | 0.608858 | -0.00342 | 0.0116 | 0.006534 | PASS |
| equity.ko_reset_snowball.pde | discrete_ki | gamma | -0.0366268 | 0.00883 | -0.046709 | -0.01008 | 0.02775 | 0.0002885 | PASS |
| equity.ko_reset_snowball.quad | ordinary | pv | 96.6613 | 0.011 | 96.6856 | 0.02433 | 0.04636 | 0.02263 | PASS |
| equity.ko_reset_snowball.quad | ordinary | delta | 0.621389 | 0.00868 | 0.611456 | -0.009933 | 0.02729 | 0.002288 | PASS |
| equity.ko_reset_snowball.quad | ordinary | gamma | -0.0416238 | 0.0297 | -0.0463172 | -0.004693 | 0.06409 | 0.03088 | PASS |
| equity.ko_reset_snowball.quad | near_pre_ko | pv | 98.0475 | 0.0132 | 98.0601 | 0.0126 | 0.03904 | 0.02145 | PASS |
| equity.ko_reset_snowball.quad | near_pre_ko | delta | 0.487059 | 0.00233 | 0.489166 | 0.002108 | 0.006777 | 0.003152 | PASS |
| equity.ko_reset_snowball.quad | near_pre_ko | gamma | -0.034448 | 0.0126 | -0.0499693 | -0.01552 | 0.04068 | 0.003377 | PASS |
| equity.ko_reset_snowball.quad | near_ki | pv | 79.0154 | 0.00618 | 79.0376 | 0.02219 | 0.03454 | 0.09065 | PASS |
| equity.ko_reset_snowball.quad | near_ki | delta | 1.06603 | 0.00536 | 1.05918 | -0.006857 | 0.01757 | 5.962e-06 | PASS |
| equity.ko_reset_snowball.quad | near_ki | gamma | -0.0382894 | 0.0268 | -0.0374854 | 0.000804 | 0.05445 | 0.005175 | PASS |
| equity.ko_reset_snowball.quad | below_ki | pv | 74.0373 | 0.0119 | 74.0455 | 0.008203 | 0.0321 | 0.07598 | PASS |
| equity.ko_reset_snowball.quad | below_ki | delta | 1.10845 | 0.00477 | 1.10667 | -0.001774 | 0.0113 | 0.004074 | PASS |
| equity.ko_reset_snowball.quad | below_ki | gamma | -0.0131948 | 0.0123 | 0.00547268 | 0.01867 | 0.04331 | 0.02009 | PASS |
| equity.ko_reset_snowball.quad | low_vol | pv | 100.165 | 0.0116 | 100.169 | 0.003378 | 0.0266 | 0.01508 | PASS |
| equity.ko_reset_snowball.quad | low_vol | delta | 0.359471 | 0.00292 | 0.368805 | 0.009333 | 0.01518 | 0.0003847 | PASS |
| equity.ko_reset_snowball.quad | low_vol | gamma | -0.0863362 | 0.0118 | -0.105346 | -0.01901 | 0.04262 | 0.03172 | PASS |
| equity.ko_reset_snowball.quad | near_expiry | pv | 99.5678 | 0.00505 | 99.57 | 0.00224 | 0.01235 | 0.0004124 | PASS |
| equity.ko_reset_snowball.quad | near_expiry | delta | 0.284584 | 0.00117 | 0.285801 | 0.001217 | 0.003552 | 0.000649 | PASS |
| equity.ko_reset_snowball.quad | near_expiry | gamma | -0.0472864 | 0.00329 | -0.0448217 | 0.002465 | 0.009045 | 0.001984 | PASS |
| equity.ko_reset_snowball.quad | discrete_ki | pv | 96.7295 | 0.00999 | 96.7344 | 0.004878 | 0.02487 | 0 | PASS |
| equity.ko_reset_snowball.quad | discrete_ki | delta | 0.612278 | 0.00409 | 0.605935 | -0.006343 | 0.01452 | 0 | PASS |
| equity.ko_reset_snowball.quad | discrete_ki | gamma | -0.0366268 | 0.00883 | -0.0463895 | -0.009763 | 0.02743 | 0 | PASS |

## Aggregate bias

| candidate | quantity | cells | mean bias (c) | SE (c) | passed |
|---|---|---|---|---|---|
| equity.ko_reset_snowball.pde | pv | 7 | 0.02332 | 0.00388 | yes |
| equity.ko_reset_snowball.pde | delta | 7 | -0.005492 | 0.0018 | yes |
| equity.ko_reset_snowball.pde | gamma | 7 | 0.00356 | 0.00661 | yes |
| equity.ko_reset_snowball.quad | pv | 7 | 0.01112 | 0.00388 | yes |
| equity.ko_reset_snowball.quad | delta | 7 | -0.00175 | 0.0018 | yes |
| equity.ko_reset_snowball.quad | gamma | 7 | -0.003864 | 0.00661 | yes |
