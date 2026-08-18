# Certification report: phoenix-flat-bsm

Evidence digest: `d1789cb2e8211dffd1a3f3e4389673cbb9ff2cff979e251888f2972171d13343`

Machine: `arm64` / macOS-26.6-arm64-arm-64bit - Python 3.11.8, NumPy 2.4.6, quantark `3655050ca8a42f2f559c5128fc63f4bcc8b95636`

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
| equity.phoenix.quad | ordinary | pv | -3.42614 | 0.00424 | -3.42722 | -0.001085 | 0.009574 | 0.004553 | PASS |
| equity.phoenix.quad | ordinary | delta | 0.488242 | 0.000842 | 0.487954 | -0.0002887 | 0.001973 | 0.00315 | PASS |
| equity.phoenix.quad | ordinary | gamma | -0.0432388 | 0.00451 | -0.03328 | 0.009959 | 0.01897 | 0.0001272 | PASS |
| equity.phoenix.quad | near_ko | pv | -2.31343 | 0.00358 | -2.32105 | -0.007613 | 0.01478 | 0.00723 | PASS |
| equity.phoenix.quad | near_ko | delta | 0.400328 | 0.00292 | 0.395748 | -0.004581 | 0.01042 | 0.002239 | PASS |
| equity.phoenix.quad | near_ko | gamma | -0.0493471 | 0.00687 | -0.0380249 | 0.01132 | 0.02507 | 0.002756 | PASS |
| equity.phoenix.quad | near_coupon | pv | -13.8608 | 0.00517 | -13.8766 | -0.01575 | 0.02609 | 0.004036 | PASS |
| equity.phoenix.quad | near_coupon | delta | 0.948728 | 0.00425 | 0.948334 | -0.0003942 | 0.008892 | 7.258e-06 | PASS |
| equity.phoenix.quad | near_coupon | gamma | -0.0437926 | 0.0216 | -0.0309079 | 0.01288 | 0.05604 | 0.00296 | PASS |
| equity.phoenix.quad | near_ki | pv | -23.9184 | 0.00146 | -23.944 | -0.02563 | 0.02855 | 0.001108 | PASS |
| equity.phoenix.quad | near_ki | delta | 1.13636 | 0.00424 | 1.13402 | -0.00234 | 0.01081 | 0.0006175 | PASS |
| equity.phoenix.quad | near_ki | gamma | -0.00717388 | 0.00491 | -0.00654924 | 0.0006246 | 0.01045 | 0.00133 | PASS |
| equity.phoenix.quad | low_vol | pv | 0.526131 | 0.00105 | 0.527809 | 0.001678 | 0.003771 | 0.004681 | PASS |
| equity.phoenix.quad | low_vol | delta | 0.00547519 | 0.00269 | 0.00537737 | -9.782e-05 | 0.005484 | 0.001729 | PASS |
| equity.phoenix.quad | low_vol | gamma | -0.0266412 | 0.00211 | -0.031955 | -0.005314 | 0.009527 | 0.0002634 | PASS |
| equity.phoenix.quad | near_expiry | pv | 0.0923547 | 0.00199 | 0.0915404 | -0.0008144 | 0.004785 | 0.0003528 | PASS |
| equity.phoenix.quad | near_expiry | delta | 0.0556837 | 0.000659 | 0.055379 | -0.0003047 | 0.001622 | 3.208e-06 | PASS |
| equity.phoenix.quad | near_expiry | gamma | -0.0205876 | 0.00317 | -0.0183541 | 0.002233 | 0.008568 | 0.0002959 | PASS |
| equity.phoenix.quad | memory | pv | -3.34946 | 0.00452 | -3.35014 | -0.0006835 | 0.009731 | 0.004371 | PASS |
| equity.phoenix.quad | memory | delta | 0.477563 | 0.000903 | 0.477248 | -0.0003148 | 0.002122 | 0.003086 | PASS |
| equity.phoenix.quad | memory | gamma | -0.0428127 | 0.0044 | -0.0324591 | 0.01035 | 0.01916 | 0.0001318 | PASS |

## Aggregate bias

| candidate | quantity | cells | mean bias (c) | SE (c) | passed |
|---|---|---|---|---|---|
| equity.phoenix.pde | pv | 7 | 0.05757 | 0.00132 | yes |
| equity.phoenix.pde | delta | 7 | -0.005953 | 0.00105 | yes |
| equity.phoenix.pde | gamma | 7 | 0.005723 | 0.00347 | yes |
| equity.phoenix.quad | pv | 7 | -0.007128 | 0.00132 | yes |
| equity.phoenix.quad | delta | 7 | -0.001189 | 0.00105 | yes |
| equity.phoenix.quad | gamma | 7 | 0.006009 | 0.00347 | yes |
