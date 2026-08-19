# Certification report: snowball-flat-bsm

Evidence digest: `0b0eb96318e7bf6f80027c5267bc39404bf8a774ad052c4e967edef7d7ac360b`

Machine: `arm64` / macOS-26.6-arm64-arm-64bit - Python 3.11.8, NumPy 2.4.6, quantark `c22a4307085ee10959b6d7d048a456dd59925379`

## Amendment

| field | value |
|---|---|
| parent | docs/modelvalidation/certificates/snowball-flat-bsm/2026-08-19/certificate.json |
| parent digest | `3454f3074cb39a8c49e7adc1bae9ac1f6648fc2a8ca69fb85f575bfca189c04a` |
| reason | Certify the discrete-KI, European-KI, step-down KO and parachute product variants the original scope omitted |
| re-priced | 36 cell(s) |
| carried forward | 30 cell(s) |

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
| discrete_ki | 4 | se_budget_met | delta: 0.00551, gamma: 0.0102, pv: 0.00376 |
| european_ki | 4 | se_budget_met | delta: 0.00596, gamma: 0.0111, pv: 0.0032 |
| low_vol | 4 | se_budget_met | delta: 0.00449, gamma: 0.00785, pv: 0.00392 |
| near_expiry | 4 | se_budget_met | delta: 0.005, gamma: 0.0177, pv: 0.00783 |
| near_ki | 4 | se_budget_met | delta: 0.00588, gamma: 0.0111, pv: 0.00342 |
| near_ko | 4 | se_budget_met | delta: 0.0043, gamma: 0.0128, pv: 0.00502 |
| ordinary | 4 | se_budget_met | delta: 0.00447, gamma: 0.00663, pv: 0.00863 |
| parachute | 4 | se_budget_met | delta: 0.00352, gamma: 0.00547, pv: 0.00387 |
| parachute_near_ki | 4 | se_budget_met | delta: 0.00419, gamma: 0.00214, pv: 0.00532 |
| stepdown_ko | 4 | se_budget_met | delta: 0.00611, gamma: 0.0102, pv: 0.00272 |
| stepdown_near_last_ko | 4 | se_budget_met | delta: 0.00352, gamma: 0.0158, pv: 0.00547 |

Sampling policy: 65536 paths/batch, 4-32 batches, seed 20260814, bump 0.01.

## Cells

| candidate | case | quantity | reference | SE | candidate | err (c) | interval (c) | envelope (c) | verdict |
|---|---|---|---|---|---|---|---|---|---|
| equity.snowball.pde | ordinary | pv | 96.4619 | 0.00863 | 96.4607 | -0.001291 | 0.01854 | 0.001934 | PASS |
| equity.snowball.pde | ordinary | delta | 0.636873 | 0.00447 | 0.637166 | 0.0002924 | 0.009223 | 0.00301 | PASS |
| equity.snowball.pde | ordinary | gamma | -0.0350358 | 0.00663 | -0.0472021 | -0.01217 | 0.02543 | 0.0001555 | PASS |
| equity.snowball.pde | near_ko | pv | 97.8973 | 0.00502 | 97.8994 | 0.002057 | 0.01209 | 0.002202 | PASS |
| equity.snowball.pde | near_ko | delta | 0.517695 | 0.0043 | 0.512836 | -0.00486 | 0.01347 | 0.004432 | PASS |
| equity.snowball.pde | near_ko | gamma | -0.0394148 | 0.0128 | -0.0515796 | -0.01216 | 0.03772 | 5.578e-05 | PASS |
| equity.snowball.pde | near_ki | pv | 84.2677 | 0.00342 | 84.2663 | -0.001404 | 0.008246 | 0.0001292 | PASS |
| equity.snowball.pde | near_ki | delta | 1.01836 | 0.00588 | 1.0145 | -0.003851 | 0.0156 | 2.785e-05 | PASS |
| equity.snowball.pde | near_ki | gamma | -0.013149 | 0.0111 | -0.0059712 | 0.007178 | 0.02931 | 0.0001 | PASS |
| equity.snowball.pde | low_vol | pv | 101.737 | 0.00392 | 101.737 | -0.0003949 | 0.008226 | 0.003265 | PASS |
| equity.snowball.pde | low_vol | delta | 0.187698 | 0.00449 | 0.167407 | -0.02029 | 0.02927 | 0.01432 | PASS |
| equity.snowball.pde | low_vol | gamma | -0.124929 | 0.00785 | -0.135347 | -0.01042 | 0.02611 | 0.002255 | PASS |
| equity.snowball.pde | near_expiry | pv | 103.261 | 0.00783 | 103.254 | -0.00756 | 0.02323 | 0.0139 | PASS |
| equity.snowball.pde | near_expiry | delta | 0.120079 | 0.005 | 0.100767 | -0.01931 | 0.0293 | 0.0009136 | PASS |
| equity.snowball.pde | near_expiry | gamma | -0.10491 | 0.0177 | -0.10969 | -0.00478 | 0.0402 | 9.773e-05 | PASS |
| equity.snowball.pde | discrete_ki | pv | 96.909 | 0.00376 | 96.9068 | -0.002211 | 0.009724 | 0.0003263 | PASS |
| equity.snowball.pde | discrete_ki | delta | 0.600136 | 0.00551 | 0.595941 | -0.004195 | 0.01521 | 0.003287 | PASS |
| equity.snowball.pde | discrete_ki | gamma | -0.031217 | 0.0102 | -0.0493195 | -0.0181 | 0.03856 | 8.658e-05 | PASS |
| equity.snowball.pde | european_ki | pv | 98.5502 | 0.0032 | 98.5492 | -0.001031 | 0.007425 | 0.0001073 | PASS |
| equity.snowball.pde | european_ki | delta | 0.401999 | 0.00596 | 0.399892 | -0.002107 | 0.01402 | 0.002899 | PASS |
| equity.snowball.pde | european_ki | gamma | -0.039412 | 0.0111 | -0.0423105 | -0.002899 | 0.02512 | 6.852e-05 | PASS |
| equity.snowball.pde | stepdown_ko | pv | 97.1182 | 0.00272 | 97.1197 | 0.001479 | 0.006922 | 0.001092 | PASS |
| equity.snowball.pde | stepdown_ko | delta | 0.56238 | 0.00611 | 0.570609 | 0.00823 | 0.02045 | 0.0007206 | PASS |
| equity.snowball.pde | stepdown_ko | gamma | -0.0469975 | 0.0102 | -0.0473838 | -0.0003863 | 0.02082 | 2.636e-05 | PASS |
| equity.snowball.pde | stepdown_near_last_ko | pv | 95.2078 | 0.00547 | 95.2066 | -0.00123 | 0.01216 | 0.001547 | PASS |
| equity.snowball.pde | stepdown_near_last_ko | delta | 0.705585 | 0.00352 | 0.699043 | -0.006542 | 0.01359 | 0.002764 | PASS |
| equity.snowball.pde | stepdown_near_last_ko | gamma | -0.0608893 | 0.0158 | -0.0417128 | 0.01918 | 0.05081 | 0.0001411 | PASS |
| equity.snowball.pde | parachute | pv | 98.5509 | 0.00387 | 98.5448 | -0.006095 | 0.01384 | 0.0003141 | PASS |
| equity.snowball.pde | parachute | delta | 0.402179 | 0.00352 | 0.40078 | -0.001399 | 0.008449 | 0.002836 | PASS |
| equity.snowball.pde | parachute | gamma | -0.0426591 | 0.00547 | -0.0424774 | 0.0001817 | 0.01113 | 5.973e-05 | PASS |
| equity.snowball.pde | parachute_near_ki | pv | 88.8357 | 0.00532 | 88.8353 | -0.0004172 | 0.01105 | 0.000546 | PASS |
| equity.snowball.pde | parachute_near_ki | delta | 0.962974 | 0.00419 | 0.956487 | -0.006488 | 0.01486 | 0.0001619 | PASS |
| equity.snowball.pde | parachute_near_ki | gamma | -0.0306684 | 0.00214 | -0.0324436 | -0.001775 | 0.006056 | 6.461e-06 | PASS |
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
| equity.snowball.quad | discrete_ki | pv | 96.909 | 0.00376 | 96.9037 | -0.005329 | 0.01284 | 0.007819 | PASS |
| equity.snowball.quad | discrete_ki | delta | 0.600136 | 0.00551 | 0.596296 | -0.00384 | 0.01486 | 0.004236 | PASS |
| equity.snowball.quad | discrete_ki | gamma | -0.031217 | 0.0102 | -0.046264 | -0.01505 | 0.03551 | 4.74e-05 | PASS |
| equity.snowball.quad | european_ki | pv | 98.5502 | 0.0032 | 98.5421 | -0.008079 | 0.01447 | 0.007023 | PASS |
| equity.snowball.quad | european_ki | delta | 0.401999 | 0.00596 | 0.401494 | -0.0005045 | 0.01242 | 0.003349 | PASS |
| equity.snowball.quad | european_ki | gamma | -0.039412 | 0.0111 | -0.0398281 | -0.0004162 | 0.02263 | 0.0005247 | PASS |
| equity.snowball.quad | stepdown_ko | pv | 97.1182 | 0.00272 | 97.1201 | 0.001919 | 0.007362 | 0.003003 | PASS |
| equity.snowball.quad | stepdown_ko | delta | 0.56238 | 0.00611 | 0.568675 | 0.006295 | 0.01851 | 0.0004737 | PASS |
| equity.snowball.quad | stepdown_ko | gamma | -0.0469975 | 0.0102 | -0.0476869 | -0.0006894 | 0.02112 | 0.0007087 | PASS |
| equity.snowball.quad | stepdown_near_last_ko | pv | 95.2078 | 0.00547 | 95.2064 | -0.001468 | 0.0124 | 0.007586 | PASS |
| equity.snowball.quad | stepdown_near_last_ko | delta | 0.705585 | 0.00352 | 0.703351 | -0.002234 | 0.009283 | 8.849e-05 | PASS |
| equity.snowball.quad | stepdown_near_last_ko | gamma | -0.0608893 | 0.0158 | -0.0407004 | 0.02019 | 0.05182 | 0.007502 | PASS |
| equity.snowball.quad | parachute | pv | 98.5509 | 0.00387 | 98.5421 | -0.008821 | 0.01657 | 0.007023 | PASS |
| equity.snowball.quad | parachute | delta | 0.402179 | 0.00352 | 0.401494 | -0.0006849 | 0.007734 | 0.003349 | PASS |
| equity.snowball.quad | parachute | gamma | -0.0426591 | 0.00547 | -0.0398281 | 0.002831 | 0.01378 | 0.0005247 | PASS |
| equity.snowball.quad | parachute_near_ki | pv | 88.8357 | 0.00532 | 88.8347 | -0.0009839 | 0.01162 | 0.001337 | PASS |
| equity.snowball.quad | parachute_near_ki | delta | 0.962974 | 0.00419 | 0.959708 | -0.003266 | 0.01164 | 0.0009966 | PASS |
| equity.snowball.quad | parachute_near_ki | gamma | -0.0306684 | 0.00214 | -0.0328135 | -0.002145 | 0.006426 | 0.002567 | PASS |

## Aggregate bias

| candidate | quantity | cells | mean bias (c) | SE (c) | passed |
|---|---|---|---|---|---|
| equity.snowball.pde | pv | 11 | -0.001645 | 0.00156 | yes |
| equity.snowball.pde | delta | 11 | -0.005502 | 0.00148 | yes |
| equity.snowball.pde | gamma | 11 | -0.003287 | 0.00331 | yes |
| equity.snowball.quad | pv | 11 | -0.002449 | 0.00156 | yes |
| equity.snowball.quad | delta | 11 | -0.001519 | 0.00148 | yes |
| equity.snowball.quad | gamma | 11 | -0.002795 | 0.00331 | yes |
