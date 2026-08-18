# Certification report: snowball-flat-bsm

Evidence digest: `a3591aa83b0dbc08c2e6202a2d61a6b3bdd3c2b069c2464281384625ee6bfbe4`

Machine: `arm64` / macOS-26.6-arm64-arm-64bit - Python 3.11.8, NumPy 2.4.6, quantark `40ca3b01fa7d8e3489688a97cb52e39580d69df1`

## Decisions

| candidate | decision |
|---|---|
| equity.snowball.pde | ADMITTED |
| equity.snowball.quad | ADMITTED |

Bounds: cell 0.5 c, mean signed bias 0.1 c, standard-error budget 0.25 x cell, interval k 2.

## Engine configuration

Resolved rather than named: a profile such as `standard` is an indirection whose meaning can change between releases. These are the requested settings.

| engine | setting | value |
|---|---|---|
| equity.snowball.pde | accuracy | standard |
| equity.snowball.pde | engine | SnowballPDESolver |
| equity.snowball.pde | grid.bounds | [None, None] |
| equity.snowball.pde | grid.day_count | 252 |
| equity.snowball.pde | grid.eps_crit | 0.003 |
| equity.snowball.pde | grid.event_damping_steps | 2 |
| equity.snowball.pde | grid.max_points | 2000 |
| equity.snowball.pde | grid.max_steps | 5000 |
| equity.snowball.pde | grid.num_std | 4 |
| equity.snowball.pde | grid.points | 400 |
| equity.snowball.pde | grid.steps_per_day | 4 |
| equity.snowball.pde | grid.terminal_damping_steps | 1 |
| equity.snowball.quad | engine | SnowballQuadEngine |
| equity.snowball.quad | grid.align_priority | auto |
| equity.snowball.quad | grid.barrier_reach_stddevs | -- |
| equity.snowball.quad | grid.bgk_min_ki_observations | 100 |
| equity.snowball.quad | grid.bus_days_in_year | 252 |
| equity.snowball.quad | grid.event_projection | cell_average |
| equity.snowball.quad | grid.event_smoothing_cells | 1 |
| equity.snowball.quad | grid.event_smoothing_kernel | cosine |
| equity.snowball.quad | grid.event_smoothing_log_width | 0.002 |
| equity.snowball.quad | grid.event_smoothing_mode | fixed |
| equity.snowball.quad | grid.fft_filter_alpha | 12 |
| equity.snowball.quad | grid.fft_filter_power | 8 |
| equity.snowball.quad | grid.fft_padding_factor | 2 |
| equity.snowball.quad | grid.filter_unreachable_barriers | True |
| equity.snowball.quad | grid.grid_points | 1001 |
| equity.snowball.quad | grid.integration_rule | trapezoid |
| equity.snowball.quad | grid.ki_monitoring_mode | exact_discrete |
| equity.snowball.quad | grid.max_adaptive_grid_points | 5001 |
| equity.snowball.quad | grid.min_diffusion_stddev_cells | 2.5 |
| equity.snowball.quad | grid.num_std_devs | 10 |
| equity.snowball.quad | grid.stability_preset | -- |
| equity.snowball.quad | grid_points | 1001 |
| (benchmark) | engine | SnowballMCEngine |
| (benchmark) | greeks | paired central difference (common random numbers) |
| (benchmark) | method | randomized_quasi |
| (benchmark) | paths_per_batch | 65536 |

## Benchmark sampling

| case | batches | stopped because | standard errors (raw) |
|---|---|---|---|
| low_vol | 4 | se_budget_met | delta: 0.00449, gamma: 0.00785, pv: 0.00392 |
| near_expiry | 4 | se_budget_met | delta: 0.005, gamma: 0.0177, pv: 0.00783 |
| near_ki | 4 | se_budget_met | delta: 0.00588, gamma: 0.0111, pv: 0.00342 |
| near_ko | 4 | se_budget_met | delta: 0.0043, gamma: 0.0128, pv: 0.00502 |
| ordinary | 4 | se_budget_met | delta: 0.00447, gamma: 0.00663, pv: 0.00863 |

Sampling policy: 65536 paths/batch, 4-32 batches, seed 20260814, bump 0.01.

## Cells

| candidate | case | quantity | reference | SE | candidate | err (c) | interval (c) | envelope (c) | verdict |
|---|---|---|---|---|---|---|---|---|---|
| equity.snowball.pde | ordinary | pv | 96.4619 | 0.00863 | 96.4961 | 0.03417 | 0.05142 | 0.00998 | PASS |
| equity.snowball.pde | ordinary | delta | 0.636873 | 0.00447 | 0.634152 | -0.002721 | 0.01165 | 0.002331 | PASS |
| equity.snowball.pde | ordinary | gamma | -0.0350358 | 0.00663 | -0.0474505 | -0.01241 | 0.02568 | 9.622e-05 | PASS |
| equity.snowball.pde | near_ko | pv | 97.8973 | 0.00502 | 97.9186 | 0.02129 | 0.03132 | 0.01042 | PASS |
| equity.snowball.pde | near_ko | delta | 0.517695 | 0.0043 | 0.510521 | -0.007174 | 0.01578 | 0.00543 | PASS |
| equity.snowball.pde | near_ko | gamma | -0.0394148 | 0.0128 | -0.0515764 | -0.01216 | 0.03772 | 4.389e-05 | PASS |
| equity.snowball.pde | near_ki | pv | 84.2677 | 0.00342 | 84.2835 | 0.01579 | 0.02263 | 0.0002684 | PASS |
| equity.snowball.pde | near_ki | delta | 1.01836 | 0.00588 | 1.0203 | 0.001939 | 0.01369 | 0.0001534 | PASS |
| equity.snowball.pde | near_ki | gamma | -0.013149 | 0.0111 | -0.0103243 | 0.002825 | 0.02495 | 2.98e-06 | PASS |
| equity.snowball.pde | low_vol | pv | 101.737 | 0.00392 | 101.866 | 0.1295 | 0.1373 | 0.03348 | PASS |
| equity.snowball.pde | low_vol | delta | 0.187698 | 0.00449 | 0.144697 | -0.043 | 0.05198 | 0.007844 | PASS |
| equity.snowball.pde | low_vol | gamma | -0.124929 | 0.00785 | -0.135065 | -0.01014 | 0.02583 | 0.002265 | PASS |
| equity.snowball.pde | near_expiry | pv | 103.261 | 0.00783 | 103.413 | 0.1518 | 0.1675 | 0.05294 | PASS |
| equity.snowball.pde | near_expiry | delta | 0.120079 | 0.005 | 0.0789251 | -0.04115 | 0.05114 | 0.006403 | PASS |
| equity.snowball.pde | near_expiry | gamma | -0.10491 | 0.0177 | -0.108376 | -0.003466 | 0.03888 | 0.0002647 | PASS |
| equity.snowball.quad | ordinary | pv | 96.4619 | 0.00863 | 96.459 | -0.002912 | 0.02016 | 0.005653 | PASS |
| equity.snowball.quad | ordinary | delta | 0.636873 | 0.00447 | 0.637121 | 0.0002477 | 0.009178 | 0.004398 | PASS |
| equity.snowball.quad | ordinary | gamma | -0.0350358 | 0.00663 | -0.0442877 | -0.009252 | 0.02251 | 0.0002628 | PASS |
| equity.snowball.quad | near_ko | pv | 97.8973 | 0.00502 | 97.8992 | 0.001843 | 0.01188 | 0.009546 | PASS |
| equity.snowball.quad | near_ko | delta | 0.517695 | 0.0043 | 0.51289 | -0.004806 | 0.01341 | 0.003052 | PASS |
| equity.snowball.quad | near_ko | gamma | -0.0394148 | 0.0128 | -0.0514942 | -0.01208 | 0.03764 | 0.003823 | PASS |
| equity.snowball.quad | near_ki | pv | 84.2677 | 0.00342 | 84.2672 | -0.0005286 | 0.00737 | 0.001969 | PASS |
| equity.snowball.quad | near_ki | delta | 1.01836 | 0.00588 | 1.01636 | -0.002 | 0.01375 | 0.0001584 | PASS |
| equity.snowball.quad | near_ki | gamma | -0.013149 | 0.0111 | -0.00968071 | 0.003468 | 0.0256 | 2.196e-05 | PASS |
| equity.snowball.quad | low_vol | pv | 101.737 | 0.00392 | 101.733 | -0.003648 | 0.01148 | 0.01411 | PASS |
| equity.snowball.quad | low_vol | delta | 0.187698 | 0.00449 | 0.186587 | -0.001111 | 0.01009 | 0.004846 | PASS |
| equity.snowball.quad | low_vol | gamma | -0.124929 | 0.00785 | -0.134778 | -0.009848 | 0.02554 | 0.0001344 | PASS |
| equity.snowball.quad | near_expiry | pv | 103.261 | 0.00783 | 103.263 | 0.001067 | 0.01673 | 0.003381 | PASS |
| equity.snowball.quad | near_expiry | delta | 0.120079 | 0.005 | 0.115268 | -0.004811 | 0.0148 | 0.0001723 | PASS |
| equity.snowball.quad | near_expiry | gamma | -0.10491 | 0.0177 | -0.112663 | -0.007752 | 0.04317 | 0.001919 | PASS |

## Aggregate bias

| candidate | quantity | cells | mean bias (c) | SE (c) | passed |
|---|---|---|---|---|---|
| equity.snowball.pde | pv | 5 | 0.07051 | 0.00274 | yes |
| equity.snowball.pde | delta | 5 | -0.01842 | 0.00217 | yes |
| equity.snowball.pde | gamma | 5 | -0.007071 | 0.00531 | yes |
| equity.snowball.quad | pv | 5 | -0.0008358 | 0.00274 | yes |
| equity.snowball.quad | delta | 5 | -0.002496 | 0.00217 | yes |
| equity.snowball.quad | gamma | 5 | -0.007093 | 0.00531 | yes |
