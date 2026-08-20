# Certification report: snowball-flat-bsm

Evidence digest: `40ae8280f233729d6eb7427a62c89c3f253948e5c189e0753517760fd89de563`

Machine: `arm64` / macOS-26.6-arm64-arm-64bit - Python 3.11.8, NumPy 2.4.6, quantark `886526b6d64f9980fd1b74d856934e4482e07d28`

## Amendment

| field | value |
|---|---|
| parent | docs/modelvalidation/certificates/snowball-flat-bsm/2026-08-19-2/certificate.json |
| parent digest | `0b0eb96318e7bf6f80027c5267bc39404bf8a774ad052c4e967edef7d7ac360b` |
| reason | Certify the remaining product feature surface: stepping KI barriers, reverse, airbag, protection, participation, call rebate, disable_ko_after_ki, expiry coupons, non-annualized accrual and per-observation KO rates |
| re-priced | 72 cell(s) |
| carried forward | 66 cell(s) |

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
| airbag | 4 | se_budget_met | delta: 0.00267, gamma: 0.00556, pv: 0.00749 |
| call_rebate | 4 | se_budget_met | delta: 0.004, gamma: 0.0046, pv: 0.0061 |
| coupon_at_expiry | 4 | se_budget_met | delta: 0.00401, gamma: 0.00577, pv: 0.00863 |
| disable_ko_after_ki | 4 | se_budget_met | delta: 0.00634, gamma: 0.0119, pv: 0.0051 |
| disable_ko_after_ki_knocked_in | 4 | se_budget_met | delta: 1.3e-07, gamma: 4.91e-08, pv: 1.02e-05 |
| discrete_ki | 4 | se_budget_met | delta: 0.00551, gamma: 0.0102, pv: 0.00376 |
| european_ki | 4 | se_budget_met | delta: 0.00596, gamma: 0.0111, pv: 0.0032 |
| ki_stepdown | 4 | se_budget_met | delta: 0.00638, gamma: 0.00645, pv: 0.00526 |
| ko_rate_step | 4 | se_budget_met | delta: 0.00427, gamma: 0.00609, pv: 0.00951 |
| low_vol | 4 | se_budget_met | delta: 0.00449, gamma: 0.00785, pv: 0.00392 |
| near_expiry | 4 | se_budget_met | delta: 0.005, gamma: 0.0177, pv: 0.00783 |
| near_ki | 4 | se_budget_met | delta: 0.00588, gamma: 0.0111, pv: 0.00342 |
| near_ko | 4 | se_budget_met | delta: 0.0043, gamma: 0.0128, pv: 0.00502 |
| not_annualized | 4 | se_budget_met | delta: 0.00736, gamma: 0.0126, pv: 0.00886 |
| ordinary | 4 | se_budget_met | delta: 0.00447, gamma: 0.00663, pv: 0.00863 |
| parachute | 4 | se_budget_met | delta: 0.00352, gamma: 0.00547, pv: 0.00387 |
| parachute_near_ki | 4 | se_budget_met | delta: 0.00419, gamma: 0.00214, pv: 0.00532 |
| participation | 4 | se_budget_met | delta: 0.00264, gamma: 0.00475, pv: 0.00697 |
| protection_full | 4 | se_budget_met | delta: 0.000962, gamma: 0.00421, pv: 0.00533 |
| protection_partial | 4 | se_budget_met | delta: 0.00444, gamma: 0.00665, pv: 0.00861 |
| reverse | 4 | se_budget_met | delta: 0.00817, gamma: 0.0106, pv: 0.0133 |
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
| equity.snowball.pde | ki_stepdown | pv | 97.793 | 0.00526 | 97.7938 | 0.000813 | 0.01134 | 0.0009674 | PASS |
| equity.snowball.pde | ki_stepdown | delta | 0.518504 | 0.00638 | 0.510636 | -0.007868 | 0.02063 | 0.002173 | PASS |
| equity.snowball.pde | ki_stepdown | gamma | -0.0485229 | 0.00645 | -0.0541776 | -0.005655 | 0.01855 | 3.53e-05 | PASS |
| equity.snowball.pde | reverse | pv | 96.9637 | 0.0133 | 96.9646 | 0.0008862 | 0.02754 | 0.0001675 | PASS |
| equity.snowball.pde | reverse | delta | -0.620071 | 0.00817 | -0.617462 | 0.00261 | 0.01895 | 0.002565 | PASS |
| equity.snowball.pde | reverse | gamma | -0.0223816 | 0.0106 | -0.0450885 | -0.02271 | 0.04391 | 0.0001947 | PASS |
| equity.snowball.pde | airbag | pv | 98.9502 | 0.00749 | 98.9473 | -0.002904 | 0.01788 | 0.004055 | PASS |
| equity.snowball.pde | airbag | delta | 0.326244 | 0.00267 | 0.32634 | 9.566e-05 | 0.005443 | 0.002638 | PASS |
| equity.snowball.pde | airbag | gamma | -0.0215876 | 0.00556 | -0.0315316 | -0.009944 | 0.02106 | 9.473e-06 | PASS |
| equity.snowball.pde | protection_partial | pv | 96.4653 | 0.00861 | 96.464 | -0.001278 | 0.0185 | 0.001856 | PASS |
| equity.snowball.pde | protection_partial | delta | 0.636258 | 0.00444 | 0.636553 | 0.0002949 | 0.009179 | 0.003015 | PASS |
| equity.snowball.pde | protection_partial | gamma | -0.034932 | 0.00665 | -0.0471075 | -0.01218 | 0.02548 | 0.0001551 | PASS |
| equity.snowball.pde | protection_full | pv | 101.701 | 0.00533 | 101.703 | 0.001744 | 0.0124 | 0.00139 | PASS |
| equity.snowball.pde | protection_full | delta | -0.0145239 | 0.000962 | -0.0150842 | -0.0005602 | 0.002484 | 0.0009254 | PASS |
| equity.snowball.pde | protection_full | gamma | -0.00949786 | 0.00421 | -0.0152155 | -0.005718 | 0.01414 | 0.0001751 | PASS |
| equity.snowball.pde | participation | pv | 99.0814 | 0.00697 | 99.0816 | 0.0002267 | 0.01417 | 0.001662 | PASS |
| equity.snowball.pde | participation | delta | 0.311175 | 0.00264 | 0.311041 | -0.0001339 | 0.005407 | 0.001968 | PASS |
| equity.snowball.pde | participation | gamma | -0.0222669 | 0.00475 | -0.0312088 | -0.008942 | 0.01844 | 9.81e-06 | PASS |
| equity.snowball.pde | call_rebate | pv | 96.3056 | 0.0061 | 96.3003 | -0.005379 | 0.01757 | 0.0005794 | PASS |
| equity.snowball.pde | call_rebate | delta | 0.64663 | 0.004 | 0.648697 | 0.002066 | 0.01007 | 0.003003 | PASS |
| equity.snowball.pde | call_rebate | gamma | -0.0418866 | 0.0046 | -0.045466 | -0.003579 | 0.01279 | 0.0002016 | PASS |
| equity.snowball.pde | disable_ko_after_ki | pv | 95.6655 | 0.0051 | 95.6656 | 3.629e-05 | 0.01024 | 0.004698 | PASS |
| equity.snowball.pde | disable_ko_after_ki | delta | 0.747742 | 0.00634 | 0.74508 | -0.002662 | 0.01534 | 0.003201 | PASS |
| equity.snowball.pde | disable_ko_after_ki | gamma | -0.0391323 | 0.0119 | -0.0551541 | -0.01602 | 0.03991 | 0.0001866 | PASS |
| equity.snowball.pde | disable_ko_after_ki_knocked_in | pv | 76.1557 | 1.02e-05 | 76.1557 | 2.988e-06 | 2.333e-05 | 0.0002187 | PASS |
| equity.snowball.pde | disable_ko_after_ki_knocked_in | delta | 0.798655 | 1.3e-07 | 0.799285 | 0.0006298 | 0.00063 | 0.0009897 | PASS |
| equity.snowball.pde | disable_ko_after_ki_knocked_in | gamma | -0.0143123 | 4.91e-08 | -0.0142905 | 2.179e-05 | 2.189e-05 | 3.751e-05 | PASS |
| equity.snowball.pde | coupon_at_expiry | pv | 95.0884 | 0.00863 | 95.0876 | -0.0008066 | 0.01806 | 0.001953 | PASS |
| equity.snowball.pde | coupon_at_expiry | delta | 0.539489 | 0.00401 | 0.539455 | -3.383e-05 | 0.008058 | 0.002971 | PASS |
| equity.snowball.pde | coupon_at_expiry | gamma | -0.0345602 | 0.00577 | -0.0465172 | -0.01196 | 0.02351 | 5.713e-05 | PASS |
| equity.snowball.pde | not_annualized | pv | 104.568 | 0.00886 | 104.564 | -0.004105 | 0.02183 | 0.001819 | PASS |
| equity.snowball.pde | not_annualized | delta | 1.22899 | 0.00736 | 1.23124 | 0.002243 | 0.01697 | 0.003154 | PASS |
| equity.snowball.pde | not_annualized | gamma | -0.0362235 | 0.0126 | -0.0500555 | -0.01383 | 0.03908 | 0.0007634 | PASS |
| equity.snowball.pde | ko_rate_step | pv | 96.8561 | 0.00951 | 96.8544 | -0.001666 | 0.0207 | 0.001952 | PASS |
| equity.snowball.pde | ko_rate_step | delta | 0.60049 | 0.00427 | 0.601047 | 0.0005569 | 0.009105 | 0.003117 | PASS |
| equity.snowball.pde | ko_rate_step | gamma | -0.0368043 | 0.00609 | -0.0487514 | -0.01195 | 0.02413 | 9.715e-05 | PASS |
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
| equity.snowball.quad | ki_stepdown | pv | 97.793 | 0.00526 | 97.7908 | -0.002167 | 0.01269 | 0.009488 | PASS |
| equity.snowball.quad | ki_stepdown | delta | 0.518504 | 0.00638 | 0.51689 | -0.001614 | 0.01438 | 0.004211 | PASS |
| equity.snowball.quad | ki_stepdown | gamma | -0.0485229 | 0.00645 | -0.0509338 | -0.002411 | 0.01531 | 0.0006969 | PASS |
| equity.snowball.quad | reverse | pv | 96.9637 | 0.0133 | 96.9629 | -0.0007814 | 0.02744 | 0.009245 | PASS |
| equity.snowball.quad | reverse | delta | -0.620071 | 0.00817 | -0.618304 | 0.001768 | 0.0181 | 0.003634 | PASS |
| equity.snowball.quad | reverse | gamma | -0.0223816 | 0.0106 | -0.0441859 | -0.0218 | 0.043 | 0.003441 | PASS |
| equity.snowball.quad | airbag | pv | 98.9502 | 0.00749 | 98.9575 | 0.007298 | 0.02227 | 0.01364 | PASS |
| equity.snowball.quad | airbag | delta | 0.326244 | 0.00267 | 0.325325 | -0.0009196 | 0.006267 | 0.001658 | PASS |
| equity.snowball.quad | airbag | gamma | -0.0215876 | 0.00556 | -0.0295168 | -0.007929 | 0.01905 | 0.0001869 | PASS |
| equity.snowball.quad | protection_partial | pv | 96.4653 | 0.00861 | 96.4624 | -0.002904 | 0.02012 | 0.005649 | PASS |
| equity.snowball.quad | protection_partial | delta | 0.636258 | 0.00444 | 0.636506 | 0.000248 | 0.009132 | 0.00439 | PASS |
| equity.snowball.quad | protection_partial | gamma | -0.034932 | 0.00665 | -0.0441987 | -0.009267 | 0.02257 | 0.0002663 | PASS |
| equity.snowball.quad | protection_full | pv | 101.701 | 0.00533 | 101.702 | 0.001073 | 0.01173 | 0.003141 | PASS |
| equity.snowball.quad | protection_full | delta | -0.0145239 | 0.000962 | -0.0144294 | 9.454e-05 | 0.002018 | 0.0008882 | PASS |
| equity.snowball.quad | protection_full | gamma | -0.00949786 | 0.00421 | -0.0141829 | -0.004685 | 0.01311 | 0.0007743 | PASS |
| equity.snowball.quad | participation | pv | 99.0814 | 0.00697 | 99.0805 | -0.0009197 | 0.01486 | 0.004397 | PASS |
| equity.snowball.quad | participation | delta | 0.311175 | 0.00264 | 0.311346 | 0.0001711 | 0.005444 | 0.002643 | PASS |
| equity.snowball.quad | participation | gamma | -0.0222669 | 0.00475 | -0.0292353 | -0.006968 | 0.01647 | 0.0002557 | PASS |
| equity.snowball.quad | call_rebate | pv | 96.3056 | 0.0061 | 96.2985 | -0.007184 | 0.01938 | 0.004867 | PASS |
| equity.snowball.quad | call_rebate | delta | 0.64663 | 0.004 | 0.648552 | 0.001921 | 0.009921 | 0.004364 | PASS |
| equity.snowball.quad | call_rebate | gamma | -0.0418866 | 0.0046 | -0.0426646 | -0.0007779 | 0.009985 | 0.0004105 | PASS |
| equity.snowball.quad | disable_ko_after_ki | pv | 95.6655 | 0.0051 | 95.6652 | -0.0003466 | 0.01055 | 0.007307 | PASS |
| equity.snowball.quad | disable_ko_after_ki | delta | 0.747742 | 0.00634 | 0.74488 | -0.002862 | 0.01554 | 0.005019 | PASS |
| equity.snowball.quad | disable_ko_after_ki | gamma | -0.0391323 | 0.0119 | -0.0517577 | -0.01263 | 0.03651 | 0.0002793 | PASS |
| equity.snowball.quad | disable_ko_after_ki_knocked_in | pv | 76.1557 | 1.02e-05 | 76.1556 | -0.0001132 | 0.0001336 | 0.0009363 | PASS |
| equity.snowball.quad | disable_ko_after_ki_knocked_in | delta | 0.798655 | 1.3e-07 | 0.798614 | -4.049e-05 | 4.075e-05 | 0.000179 | PASS |
| equity.snowball.quad | disable_ko_after_ki_knocked_in | gamma | -0.0143123 | 4.91e-08 | -0.0140451 | 0.0002672 | 0.0002673 | 0.0002608 | PASS |
| equity.snowball.quad | coupon_at_expiry | pv | 95.0884 | 0.00863 | 95.0857 | -0.002643 | 0.0199 | 0.006253 | PASS |
| equity.snowball.quad | coupon_at_expiry | delta | 0.539489 | 0.00401 | 0.539718 | 0.0002293 | 0.008254 | 0.004088 | PASS |
| equity.snowball.quad | coupon_at_expiry | gamma | -0.0345602 | 0.00577 | -0.0436161 | -0.009056 | 0.0206 | 0.0001014 | PASS |
| equity.snowball.quad | not_annualized | pv | 104.568 | 0.00886 | 104.563 | -0.004336 | 0.02206 | 0.001784 | PASS |
| equity.snowball.quad | not_annualized | delta | 1.22899 | 0.00736 | 1.22927 | 0.0002801 | 0.01501 | 0.006196 | PASS |
| equity.snowball.quad | not_annualized | gamma | -0.0362235 | 0.0126 | -0.0471522 | -0.01093 | 0.03617 | 0.002531 | PASS |
| equity.snowball.quad | ko_rate_step | pv | 96.8561 | 0.00951 | 96.8526 | -0.003482 | 0.02251 | 0.006195 | PASS |
| equity.snowball.quad | ko_rate_step | delta | 0.60049 | 0.00427 | 0.601203 | 0.0007133 | 0.009261 | 0.004389 | PASS |
| equity.snowball.quad | ko_rate_step | gamma | -0.0368043 | 0.00609 | -0.0457248 | -0.00892 | 0.0211 | 3.257e-05 | PASS |

## Aggregate bias

| candidate | quantity | cells | mean bias (c) | SE (c) | passed |
|---|---|---|---|---|---|
| equity.snowball.pde | pv | 23 | -0.001327 | 0.00138 | yes |
| equity.snowball.pde | delta | 23 | -0.002751 | 0.00102 | yes |
| equity.snowball.pde | gamma | 23 | -0.006896 | 0.00194 | yes |
| equity.snowball.quad | pv | 23 | -0.001889 | 0.00138 | yes |
| equity.snowball.quad | delta | 23 | -0.0007272 | 0.00102 | yes |
| equity.snowball.quad | gamma | 23 | -0.005472 | 0.00194 | yes |
