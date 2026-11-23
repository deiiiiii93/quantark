# Certification report: snowball-flat-bsm

Evidence digest: `ca9c0a8e9a914c878924b11596491646056a430ae84b8b89d90bf61be63bd925`

Machine: `arm64` / macOS-27.2.1-arm64-arm-64bit - Python 3.11.8, NumPy 2.4.6, quantark `e17e458f3e370634c7d63e66fd2931feece7e792`

## Amendment

| field | value |
|---|---|
| parent | docs/modelvalidation/certificates/snowball-flat-bsm/2026-11-15/certificate.json |
| parent digest | `e0e6e025f4cb0ec43f3b63870a16a468dece2c4187ea13eef38259eceed434dc` |
| reason | QUAD sub-resolution run (owner-signed 2026-11-23): QUAD V1 prices the valuation-to-first-event step exactly at the spot (Gaussian quadrature on the pre-event surfaces, exact barrier limits) whenever it spans fewer than 13 cells or is unresolved, instead of the lattice readout, whose error there reached 1.1e-3 (1.1% just above the old refusal threshold); QUAD anchors move (pv <= 3.7e-5 rel, gamma 2-12% toward a 16x finer lattice in 30/36 checked cells). Engine code change: QUAD candidates re-run against the carried benchmark. |
| re-priced | 138 cell(s) |
| carried forward | 69 cell(s) |

## Decisions

| candidate | decision |
|---|---|
| equity.snowball.pde | ADMITTED |
| equity.snowball.quad | ADMITTED |
| equity.snowball.quad.stretch_0.02 | ADMITTED |

Bounds: cell 0.5 c, mean signed bias 0.1 c, standard-error budget 0.25 x cell, interval k 2.

## Engine configuration

Resolved rather than named: a profile such as `standard` is an indirection whose meaning can change between releases. These are the requested settings.

| engine | setting | value |
|---|---|---|
| equity.snowball.pde | accuracy | standard |
| equity.snowball.pde | engine | SnowballPDESolver |
| equity.snowball.pde | grid.bounds | (None, None) |
| equity.snowball.pde | grid.day_count | 252 |
| equity.snowball.pde | grid.discrete_monitoring_damping | ['half_step', 1, 1] |
| equity.snowball.pde | grid.eps_crit | 0.003 |
| equity.snowball.pde | grid.event_damping_steps | 2 |
| equity.snowball.pde | grid.max_points | 2000 |
| equity.snowball.pde | grid.max_steps | 12000 |
| equity.snowball.pde | grid.num_std | 4 |
| equity.snowball.pde | grid.points | 400 |
| equity.snowball.pde | grid.steps_per_day | 4 |
| equity.snowball.pde | grid.terminal_damping_steps | 1 |
| equity.snowball.quad | engine | SnowballQuadEngine |
| equity.snowball.quad | grid.align_cell_stretch | -- |
| equity.snowball.quad | grid.align_priority | auto |
| equity.snowball.quad | grid.barrier_reach_stddevs | -- |
| equity.snowball.quad | grid.bgk_min_ki_observations | 100 |
| equity.snowball.quad | grid.bus_days_in_year | 252 |
| equity.snowball.quad | grid.continuous_steps_per_year | 0 |
| equity.snowball.quad | grid.continuous_term_structure | piecewise_constant |
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
| equity.snowball.quad | grid.readout | legacy_linear |
| equity.snowball.quad | grid.stability_preset | -- |
| equity.snowball.quad | grid_points | 1001 |
| equity.snowball.quad.stretch_0.02 | align_cell_stretch | 0.02 |
| equity.snowball.quad.stretch_0.02 | engine | SnowballQuadEngine |
| equity.snowball.quad.stretch_0.02 | grid.align_cell_stretch | 0.02 |
| equity.snowball.quad.stretch_0.02 | grid.align_priority | auto |
| equity.snowball.quad.stretch_0.02 | grid.barrier_reach_stddevs | -- |
| equity.snowball.quad.stretch_0.02 | grid.bgk_min_ki_observations | 100 |
| equity.snowball.quad.stretch_0.02 | grid.bus_days_in_year | 252 |
| equity.snowball.quad.stretch_0.02 | grid.continuous_steps_per_year | 0 |
| equity.snowball.quad.stretch_0.02 | grid.continuous_term_structure | piecewise_constant |
| equity.snowball.quad.stretch_0.02 | grid.event_projection | cell_average |
| equity.snowball.quad.stretch_0.02 | grid.event_smoothing_cells | 1 |
| equity.snowball.quad.stretch_0.02 | grid.event_smoothing_kernel | cosine |
| equity.snowball.quad.stretch_0.02 | grid.event_smoothing_log_width | 0.002 |
| equity.snowball.quad.stretch_0.02 | grid.event_smoothing_mode | fixed |
| equity.snowball.quad.stretch_0.02 | grid.fft_filter_alpha | 12 |
| equity.snowball.quad.stretch_0.02 | grid.fft_filter_power | 8 |
| equity.snowball.quad.stretch_0.02 | grid.fft_padding_factor | 2 |
| equity.snowball.quad.stretch_0.02 | grid.filter_unreachable_barriers | True |
| equity.snowball.quad.stretch_0.02 | grid.grid_points | 1001 |
| equity.snowball.quad.stretch_0.02 | grid.integration_rule | trapezoid |
| equity.snowball.quad.stretch_0.02 | grid.ki_monitoring_mode | exact_discrete |
| equity.snowball.quad.stretch_0.02 | grid.max_adaptive_grid_points | 5001 |
| equity.snowball.quad.stretch_0.02 | grid.min_diffusion_stddev_cells | 2.5 |
| equity.snowball.quad.stretch_0.02 | grid.num_std_devs | 10 |
| equity.snowball.quad.stretch_0.02 | grid.readout | legacy_linear |
| equity.snowball.quad.stretch_0.02 | grid.stability_preset | -- |
| equity.snowball.quad.stretch_0.02 | grid_points | 1001 |
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
| equity.snowball.pde | ordinary | delta | 0.636873 | 0.00447 | 0.637506 | 0.0006323 | 0.009563 | 0.0001818 | PASS |
| equity.snowball.pde | ordinary | gamma | -0.0350358 | 0.00663 | -0.0471861 | -0.01215 | 0.02541 | 1.754e-05 | PASS |
| equity.snowball.pde | near_ko | pv | 97.8973 | 0.00502 | 97.8994 | 0.002057 | 0.01209 | 0.002202 | PASS |
| equity.snowball.pde | near_ko | delta | 0.517695 | 0.0043 | 0.513305 | -0.004391 | 0.013 | 0.000232 | PASS |
| equity.snowball.pde | near_ko | gamma | -0.0394148 | 0.0128 | -0.0515722 | -0.01216 | 0.03771 | 5.954e-06 | PASS |
| equity.snowball.pde | near_ki | pv | 84.2677 | 0.00342 | 84.2663 | -0.001404 | 0.008246 | 0.0001292 | PASS |
| equity.snowball.pde | near_ki | delta | 1.01836 | 0.00588 | 1.01524 | -0.003111 | 0.01486 | 3.342e-06 | PASS |
| equity.snowball.pde | near_ki | gamma | -0.013149 | 0.0111 | -0.00348842 | 0.009661 | 0.03179 | 6.161e-06 | PASS |
| equity.snowball.pde | low_vol | pv | 101.737 | 0.00392 | 101.737 | -0.0003949 | 0.008226 | 0.003265 | PASS |
| equity.snowball.pde | low_vol | delta | 0.187698 | 0.00449 | 0.184098 | -0.0036 | 0.01258 | 0.0006418 | PASS |
| equity.snowball.pde | low_vol | gamma | -0.124929 | 0.00785 | -0.138081 | -0.01315 | 0.02885 | 1.819e-06 | PASS |
| equity.snowball.pde | near_expiry | pv | 103.261 | 0.00783 | 103.254 | -0.00756 | 0.02323 | 0.0139 | PASS |
| equity.snowball.pde | near_expiry | delta | 0.120079 | 0.005 | 0.113057 | -0.007022 | 0.01701 | 0.001744 | PASS |
| equity.snowball.pde | near_expiry | gamma | -0.10491 | 0.0177 | -0.111931 | -0.007021 | 0.04244 | 5.529e-05 | PASS |
| equity.snowball.pde | discrete_ki | pv | 96.909 | 0.00376 | 96.9067 | -0.002283 | 0.009796 | 0.0003262 | PASS |
| equity.snowball.pde | discrete_ki | delta | 0.600136 | 0.00551 | 0.596275 | -0.00386 | 0.01488 | 4.906e-05 | PASS |
| equity.snowball.pde | discrete_ki | gamma | -0.031217 | 0.0102 | -0.0493106 | -0.01809 | 0.03855 | 4.036e-06 | PASS |
| equity.snowball.pde | european_ki | pv | 98.5502 | 0.0032 | 98.5448 | -0.005423 | 0.01182 | 0.0003028 | PASS |
| equity.snowball.pde | european_ki | delta | 0.401999 | 0.00596 | 0.401049 | -0.00095 | 0.01287 | 3.705e-05 | PASS |
| equity.snowball.pde | european_ki | gamma | -0.039412 | 0.0111 | -0.0424812 | -0.003069 | 0.02529 | 2.869e-06 | PASS |
| equity.snowball.pde | stepdown_ko | pv | 97.1182 | 0.00272 | 97.1197 | 0.001479 | 0.006922 | 0.001092 | PASS |
| equity.snowball.pde | stepdown_ko | delta | 0.56238 | 0.00611 | 0.569086 | 0.006706 | 0.01893 | 2.858e-05 | PASS |
| equity.snowball.pde | stepdown_ko | gamma | -0.0469975 | 0.0102 | -0.0474281 | -0.0004305 | 0.02086 | 6.393e-06 | PASS |
| equity.snowball.pde | stepdown_near_last_ko | pv | 95.2078 | 0.00547 | 95.2066 | -0.00123 | 0.01216 | 0.001547 | PASS |
| equity.snowball.pde | stepdown_near_last_ko | delta | 0.705585 | 0.00352 | 0.703201 | -0.002383 | 0.009433 | 1.071e-05 | PASS |
| equity.snowball.pde | stepdown_near_last_ko | gamma | -0.0608893 | 0.0158 | -0.0414785 | 0.01941 | 0.05104 | 1.939e-05 | PASS |
| equity.snowball.pde | parachute | pv | 98.5509 | 0.00387 | 98.5448 | -0.006095 | 0.01384 | 0.0003141 | PASS |
| equity.snowball.pde | parachute | delta | 0.402179 | 0.00352 | 0.401048 | -0.001131 | 0.008181 | 3.787e-05 | PASS |
| equity.snowball.pde | parachute | gamma | -0.0426591 | 0.00547 | -0.0424824 | 0.0001767 | 0.01113 | 2.991e-06 | PASS |
| equity.snowball.pde | parachute_near_ki | pv | 88.8357 | 0.00532 | 88.8353 | -0.0004172 | 0.01105 | 0.000546 | PASS |
| equity.snowball.pde | parachute_near_ki | delta | 0.962974 | 0.00419 | 0.960131 | -0.002843 | 0.01122 | 1.367e-05 | PASS |
| equity.snowball.pde | parachute_near_ki | gamma | -0.0306684 | 0.00214 | -0.0322586 | -0.00159 | 0.005871 | 3.716e-06 | PASS |
| equity.snowball.pde | ki_stepdown | pv | 97.793 | 0.00526 | 97.7936 | 0.0006887 | 0.01122 | 0.0009674 | PASS |
| equity.snowball.pde | ki_stepdown | delta | 0.518504 | 0.00638 | 0.516309 | -0.002195 | 0.01496 | 0.0001029 | PASS |
| equity.snowball.pde | ki_stepdown | gamma | -0.0485229 | 0.00645 | -0.0543054 | -0.005783 | 0.01868 | 1.264e-05 | PASS |
| equity.snowball.pde | reverse | pv | 96.9637 | 0.0133 | 96.9646 | 0.0008862 | 0.02754 | 0.0001675 | PASS |
| equity.snowball.pde | reverse | delta | -0.620071 | 0.00817 | -0.618856 | 0.001215 | 0.01755 | 6.449e-06 | PASS |
| equity.snowball.pde | reverse | gamma | -0.0223816 | 0.0106 | -0.0449833 | -0.0226 | 0.0438 | 3.642e-07 | PASS |
| equity.snowball.pde | airbag | pv | 98.9502 | 0.00749 | 98.9473 | -0.002904 | 0.01788 | 0.004055 | PASS |
| equity.snowball.pde | airbag | delta | 0.326244 | 0.00267 | 0.32655 | 0.0003058 | 0.005654 | 0.0005071 | PASS |
| equity.snowball.pde | airbag | gamma | -0.0215876 | 0.00556 | -0.0315296 | -0.009942 | 0.02106 | 2.836e-05 | PASS |
| equity.snowball.pde | protection_partial | pv | 96.4653 | 0.00861 | 96.464 | -0.001278 | 0.0185 | 0.001856 | PASS |
| equity.snowball.pde | protection_partial | delta | 0.636258 | 0.00444 | 0.636892 | 0.0006344 | 0.009519 | 0.0001698 | PASS |
| equity.snowball.pde | protection_partial | gamma | -0.034932 | 0.00665 | -0.0470914 | -0.01216 | 0.02546 | 1.888e-05 | PASS |
| equity.snowball.pde | protection_full | pv | 101.701 | 0.00533 | 101.703 | 0.001744 | 0.0124 | 0.00139 | PASS |
| equity.snowball.pde | protection_full | delta | -0.0145239 | 0.000962 | -0.0150133 | -0.0004894 | 0.002413 | 0.0001038 | PASS |
| equity.snowball.pde | protection_full | gamma | -0.00949786 | 0.00421 | -0.0152301 | -0.005732 | 0.01416 | 1.188e-05 | PASS |
| equity.snowball.pde | participation | pv | 99.0814 | 0.00697 | 99.0816 | 0.0002267 | 0.01417 | 0.001662 | PASS |
| equity.snowball.pde | participation | delta | 0.311175 | 0.00264 | 0.311246 | 7.148e-05 | 0.005345 | 0.0001428 | PASS |
| equity.snowball.pde | participation | gamma | -0.0222669 | 0.00475 | -0.0312081 | -0.008941 | 0.01844 | 1.471e-05 | PASS |
| equity.snowball.pde | call_rebate | pv | 96.3056 | 0.0061 | 96.3003 | -0.005379 | 0.01757 | 0.0005794 | PASS |
| equity.snowball.pde | call_rebate | delta | 0.64663 | 0.004 | 0.649032 | 0.002401 | 0.0104 | 7.136e-05 | PASS |
| equity.snowball.pde | call_rebate | gamma | -0.0418866 | 0.0046 | -0.0454467 | -0.00356 | 0.01277 | 7.014e-06 | PASS |
| equity.snowball.pde | disable_ko_after_ki | pv | 95.6655 | 0.0051 | 95.6656 | 3.629e-05 | 0.01024 | 0.004698 | PASS |
| equity.snowball.pde | disable_ko_after_ki | delta | 0.747742 | 0.00634 | 0.745475 | -0.002267 | 0.01495 | 0.00053 | PASS |
| equity.snowball.pde | disable_ko_after_ki | gamma | -0.0391323 | 0.0119 | -0.0551367 | -0.016 | 0.03989 | 1.472e-06 | PASS |
| equity.snowball.pde | disable_ko_after_ki_knocked_in | pv | 76.1557 | 1.02e-05 | 76.1557 | 2.988e-06 | 2.333e-05 | 0.0002187 | PASS |
| equity.snowball.pde | disable_ko_after_ki_knocked_in | delta | 0.798655 | 1.3e-07 | 0.79872 | 6.539e-05 | 6.565e-05 | 1.498e-06 | PASS |
| equity.snowball.pde | disable_ko_after_ki_knocked_in | gamma | -0.0143123 | 4.91e-08 | -0.0143141 | -1.742e-06 | 1.84e-06 | 2.008e-06 | PASS |
| equity.snowball.pde | coupon_at_expiry | pv | 95.0884 | 0.00863 | 95.0876 | -0.0008066 | 0.01806 | 0.001953 | PASS |
| equity.snowball.pde | coupon_at_expiry | delta | 0.539489 | 0.00401 | 0.539773 | 0.0002843 | 0.008309 | 0.0001744 | PASS |
| equity.snowball.pde | coupon_at_expiry | gamma | -0.0345602 | 0.00577 | -0.04651 | -0.01195 | 0.0235 | 1.824e-05 | PASS |
| equity.snowball.pde | not_annualized | pv | 104.568 | 0.00886 | 104.564 | -0.004105 | 0.02183 | 0.001819 | PASS |
| equity.snowball.pde | not_annualized | delta | 1.22899 | 0.00736 | 1.2317 | 0.002708 | 0.01744 | 0.0002285 | PASS |
| equity.snowball.pde | not_annualized | gamma | -0.0362235 | 0.0126 | -0.0499846 | -0.01376 | 0.039 | 1.311e-05 | PASS |
| equity.snowball.pde | ko_rate_step | pv | 96.8561 | 0.00951 | 96.8544 | -0.001666 | 0.0207 | 0.001952 | PASS |
| equity.snowball.pde | ko_rate_step | delta | 0.60049 | 0.00427 | 0.601386 | 0.0008967 | 0.009445 | 0.0001796 | PASS |
| equity.snowball.pde | ko_rate_step | gamma | -0.0368043 | 0.00609 | -0.0487406 | -0.01194 | 0.02412 | 1.782e-05 | PASS |
| equity.snowball.quad | ordinary | pv | 96.4619 | 0.00863 | 96.4613 | -0.0005956 | 0.01785 | 9.492e-05 | PASS |
| equity.snowball.quad | ordinary | delta | 0.636873 | 0.00447 | 0.636956 | 8.245e-05 | 0.009013 | 0.0001057 | PASS |
| equity.snowball.quad | ordinary | gamma | -0.0350358 | 0.00663 | -0.0471424 | -0.01211 | 0.02537 | 2.024e-05 | PASS |
| equity.snowball.quad | near_ko | pv | 97.8973 | 0.00502 | 97.9016 | 0.004283 | 0.01432 | 0.0001056 | PASS |
| equity.snowball.quad | near_ko | delta | 0.517695 | 0.0043 | 0.51286 | -0.004835 | 0.01344 | 5.528e-05 | PASS |
| equity.snowball.quad | near_ko | gamma | -0.0394148 | 0.0128 | -0.0514852 | -0.01207 | 0.03763 | 1.925e-05 | PASS |
| equity.snowball.quad | near_ki | pv | 84.2677 | 0.00342 | 84.2671 | -0.0005972 | 0.007439 | 0.001821 | PASS |
| equity.snowball.quad | near_ki | delta | 1.01836 | 0.00588 | 1.01642 | -0.001932 | 0.01368 | 0.0001328 | PASS |
| equity.snowball.quad | near_ki | gamma | -0.013149 | 0.0111 | -0.00967896 | 0.00347 | 0.0256 | 1.255e-05 | PASS |
| equity.snowball.quad | low_vol | pv | 101.737 | 0.00392 | 101.737 | -0.0003286 | 0.00816 | 0.003074 | PASS |
| equity.snowball.quad | low_vol | delta | 0.187698 | 0.00449 | 0.187709 | 1.185e-05 | 0.008992 | 0.0002103 | PASS |
| equity.snowball.quad | low_vol | gamma | -0.124929 | 0.00785 | -0.137324 | -0.0124 | 0.02809 | 0.0001446 | PASS |
| equity.snowball.quad | near_expiry | pv | 103.261 | 0.00783 | 103.263 | 0.001067 | 0.01673 | 0.0009573 | PASS |
| equity.snowball.quad | near_expiry | delta | 0.120079 | 0.005 | 0.115268 | -0.004811 | 0.0148 | 0.0003333 | PASS |
| equity.snowball.quad | near_expiry | gamma | -0.10491 | 0.0177 | -0.112663 | -0.007752 | 0.04317 | 0.0009786 | PASS |
| equity.snowball.quad | discrete_ki | pv | 96.909 | 0.00376 | 96.9061 | -0.00287 | 0.01038 | 0.001668 | PASS |
| equity.snowball.quad | discrete_ki | delta | 0.600136 | 0.00551 | 0.596068 | -0.004068 | 0.01509 | 5.753e-05 | PASS |
| equity.snowball.quad | discrete_ki | gamma | -0.031217 | 0.0102 | -0.0492621 | -0.01805 | 0.03851 | 2.987e-05 | PASS |
| equity.snowball.quad | european_ki | pv | 98.5502 | 0.0032 | 98.5443 | -0.00589 | 0.01228 | 0.001499 | PASS |
| equity.snowball.quad | european_ki | delta | 0.401999 | 0.00596 | 0.401216 | -0.000783 | 0.0127 | 8.584e-05 | PASS |
| equity.snowball.quad | european_ki | gamma | -0.039412 | 0.0111 | -0.0424429 | -0.003031 | 0.02525 | 1.755e-05 | PASS |
| equity.snowball.quad | stepdown_ko | pv | 97.1182 | 0.00272 | 97.1206 | 0.002376 | 0.007819 | 0.00156 | PASS |
| equity.snowball.quad | stepdown_ko | delta | 0.56238 | 0.00611 | 0.568747 | 0.006367 | 0.01859 | 6.46e-05 | PASS |
| equity.snowball.quad | stepdown_ko | gamma | -0.0469975 | 0.0102 | -0.0480257 | -0.001028 | 0.02146 | 0.000315 | PASS |
| equity.snowball.quad | stepdown_near_last_ko | pv | 95.2078 | 0.00547 | 95.2079 | 7.195e-05 | 0.011 | 0.001936 | PASS |
| equity.snowball.quad | stepdown_near_last_ko | delta | 0.705585 | 0.00352 | 0.702727 | -0.002858 | 0.009907 | 0.000676 | PASS |
| equity.snowball.quad | stepdown_near_last_ko | gamma | -0.0608893 | 0.0158 | -0.0415105 | 0.01938 | 0.05101 | 0.001343 | PASS |
| equity.snowball.quad | parachute | pv | 98.5509 | 0.00387 | 98.5443 | -0.006632 | 0.01438 | 0.001499 | PASS |
| equity.snowball.quad | parachute | delta | 0.402179 | 0.00352 | 0.401216 | -0.0009634 | 0.008013 | 8.585e-05 | PASS |
| equity.snowball.quad | parachute | gamma | -0.0426591 | 0.00547 | -0.0424429 | 0.0002162 | 0.01116 | 1.754e-05 | PASS |
| equity.snowball.quad | parachute_near_ki | pv | 88.8357 | 0.00532 | 88.8352 | -0.0004992 | 0.01113 | 0.0007693 | PASS |
| equity.snowball.quad | parachute_near_ki | delta | 0.962974 | 0.00419 | 0.959867 | -0.003107 | 0.01148 | 0.000137 | PASS |
| equity.snowball.quad | parachute_near_ki | gamma | -0.0306684 | 0.00214 | -0.032248 | -0.00158 | 0.005861 | 3.527e-06 | PASS |
| equity.snowball.quad | ki_stepdown | pv | 97.793 | 0.00526 | 97.7936 | 0.0006117 | 0.01114 | 0.002445 | PASS |
| equity.snowball.quad | ki_stepdown | delta | 0.518504 | 0.00638 | 0.516526 | -0.001977 | 0.01474 | 0.0001335 | PASS |
| equity.snowball.quad | ki_stepdown | gamma | -0.0485229 | 0.00645 | -0.0542652 | -0.005742 | 0.01864 | 3.471e-05 | PASS |
| equity.snowball.quad | reverse | pv | 96.9637 | 0.0133 | 96.9654 | 0.001673 | 0.02833 | 0.0004437 | PASS |
| equity.snowball.quad | reverse | delta | -0.620071 | 0.00817 | -0.618166 | 0.001906 | 0.01824 | 5.312e-05 | PASS |
| equity.snowball.quad | reverse | gamma | -0.0223816 | 0.0106 | -0.0449621 | -0.02258 | 0.04378 | 2.263e-05 | PASS |
| equity.snowball.quad | airbag | pv | 98.9502 | 0.00749 | 98.9591 | 0.008953 | 0.02393 | 0.009462 | PASS |
| equity.snowball.quad | airbag | delta | 0.326244 | 0.00267 | 0.325143 | -0.001101 | 0.006449 | 0.0009927 | PASS |
| equity.snowball.quad | airbag | gamma | -0.0215876 | 0.00556 | -0.0314635 | -0.009876 | 0.02099 | 2.184e-06 | PASS |
| equity.snowball.quad | protection_partial | pv | 96.4653 | 0.00861 | 96.4647 | -0.0005921 | 0.01781 | 8.749e-05 | PASS |
| equity.snowball.quad | protection_partial | delta | 0.636258 | 0.00444 | 0.636341 | 8.367e-05 | 0.008968 | 0.0001045 | PASS |
| equity.snowball.quad | protection_partial | gamma | -0.034932 | 0.00665 | -0.0470477 | -0.01212 | 0.02542 | 2.008e-05 | PASS |
| equity.snowball.quad | protection_full | pv | 101.701 | 0.00533 | 101.703 | 0.002049 | 0.0127 | 0.0005742 | PASS |
| equity.snowball.quad | protection_full | delta | -0.0145239 | 0.000962 | -0.0146408 | -0.0001169 | 0.002041 | 8.274e-06 | PASS |
| equity.snowball.quad | protection_full | gamma | -0.00949786 | 0.00421 | -0.0151908 | -0.005693 | 0.01412 | 1.606e-05 | PASS |
| equity.snowball.quad | participation | pv | 99.0814 | 0.00697 | 99.0821 | 0.0007267 | 0.01467 | 0.0002396 | PASS |
| equity.snowball.quad | participation | delta | 0.311175 | 0.00264 | 0.311158 | -1.722e-05 | 0.00529 | 4.872e-05 | PASS |
| equity.snowball.quad | participation | gamma | -0.0222669 | 0.00475 | -0.0311666 | -0.0089 | 0.0184 | 1.815e-05 | PASS |
| equity.snowball.quad | call_rebate | pv | 96.3056 | 0.0061 | 96.3007 | -0.004966 | 0.01716 | 0.0006221 | PASS |
| equity.snowball.quad | call_rebate | delta | 0.64663 | 0.004 | 0.648418 | 0.001788 | 0.009787 | 0.0001454 | PASS |
| equity.snowball.quad | call_rebate | gamma | -0.0418866 | 0.0046 | -0.0454045 | -0.003518 | 0.01273 | 1.469e-05 | PASS |
| equity.snowball.quad | disable_ko_after_ki | pv | 95.6655 | 0.0051 | 95.6679 | 0.002326 | 0.01253 | 0.0006949 | PASS |
| equity.snowball.quad | disable_ko_after_ki | delta | 0.747742 | 0.00634 | 0.744687 | -0.003055 | 0.01573 | 1.265e-05 | PASS |
| equity.snowball.quad | disable_ko_after_ki | gamma | -0.0391323 | 0.0119 | -0.0550846 | -0.01595 | 0.03984 | 1.563e-05 | PASS |
| equity.snowball.quad | disable_ko_after_ki_knocked_in | pv | 76.1557 | 1.02e-05 | 76.1557 | 2.417e-05 | 4.451e-05 | 0.0005746 | PASS |
| equity.snowball.quad | disable_ko_after_ki_knocked_in | delta | 0.798655 | 1.3e-07 | 0.798656 | 7.622e-07 | 1.023e-06 | 3.704e-05 | PASS |
| equity.snowball.quad | disable_ko_after_ki_knocked_in | gamma | -0.0143123 | 4.91e-08 | -0.0143124 | -7.195e-08 | 1.702e-07 | 6.543e-08 | PASS |
| equity.snowball.quad | coupon_at_expiry | pv | 95.0884 | 0.00863 | 95.0881 | -0.0002814 | 0.01753 | 0.0003404 | PASS |
| equity.snowball.quad | coupon_at_expiry | delta | 0.539489 | 0.00401 | 0.539491 | 1.863e-06 | 0.008026 | 6.137e-05 | PASS |
| equity.snowball.quad | coupon_at_expiry | gamma | -0.0345602 | 0.00577 | -0.0464604 | -0.0119 | 0.02345 | 2.224e-05 | PASS |
| equity.snowball.quad | not_annualized | pv | 104.568 | 0.00886 | 104.565 | -0.002385 | 0.02011 | 0.002715 | PASS |
| equity.snowball.quad | not_annualized | delta | 1.22899 | 0.00736 | 1.2295 | 0.0005095 | 0.01524 | 0.00037 | PASS |
| equity.snowball.quad | not_annualized | gamma | -0.0362235 | 0.0126 | -0.0499813 | -0.01376 | 0.039 | 7.462e-06 | PASS |
| equity.snowball.quad | ko_rate_step | pv | 96.8561 | 0.00951 | 96.855 | -0.001049 | 0.02008 | 0.0001275 | PASS |
| equity.snowball.quad | ko_rate_step | delta | 0.60049 | 0.00427 | 0.60099 | 0.0005008 | 0.009049 | 8.963e-05 | PASS |
| equity.snowball.quad | ko_rate_step | gamma | -0.0368043 | 0.00609 | -0.0486918 | -0.01189 | 0.02407 | 2.318e-05 | PASS |
| equity.snowball.quad.stretch_0.02 | ordinary | pv | 96.4619 | 0.00863 | 96.4613 | -0.0005956 | 0.01785 | 9.492e-05 | PASS |
| equity.snowball.quad.stretch_0.02 | ordinary | delta | 0.636873 | 0.00447 | 0.636956 | 8.245e-05 | 0.009013 | 0.0001057 | PASS |
| equity.snowball.quad.stretch_0.02 | ordinary | gamma | -0.0350358 | 0.00663 | -0.0471424 | -0.01211 | 0.02537 | 2.024e-05 | PASS |
| equity.snowball.quad.stretch_0.02 | near_ko | pv | 97.8973 | 0.00502 | 97.9016 | 0.004283 | 0.01432 | 0.0001056 | PASS |
| equity.snowball.quad.stretch_0.02 | near_ko | delta | 0.517695 | 0.0043 | 0.51286 | -0.004835 | 0.01344 | 5.528e-05 | PASS |
| equity.snowball.quad.stretch_0.02 | near_ko | gamma | -0.0394148 | 0.0128 | -0.0514852 | -0.01207 | 0.03763 | 1.925e-05 | PASS |
| equity.snowball.quad.stretch_0.02 | near_ki | pv | 84.2677 | 0.00342 | 84.2671 | -0.0005972 | 0.007439 | 0.001821 | PASS |
| equity.snowball.quad.stretch_0.02 | near_ki | delta | 1.01836 | 0.00588 | 1.01642 | -0.001932 | 0.01368 | 0.0001328 | PASS |
| equity.snowball.quad.stretch_0.02 | near_ki | gamma | -0.013149 | 0.0111 | -0.00967896 | 0.00347 | 0.0256 | 1.255e-05 | PASS |
| equity.snowball.quad.stretch_0.02 | low_vol | pv | 101.737 | 0.00392 | 101.735 | -0.001932 | 0.009763 | 0.001471 | PASS |
| equity.snowball.quad.stretch_0.02 | low_vol | delta | 0.187698 | 0.00449 | 0.187942 | 0.0002442 | 0.009224 | 2.201e-05 | PASS |
| equity.snowball.quad.stretch_0.02 | low_vol | gamma | -0.124929 | 0.00785 | -0.137304 | -0.01237 | 0.02807 | 0.0001241 | PASS |
| equity.snowball.quad.stretch_0.02 | near_expiry | pv | 103.261 | 0.00783 | 103.262 | 0.0001701 | 0.01584 | 0.001601 | PASS |
| equity.snowball.quad.stretch_0.02 | near_expiry | delta | 0.120079 | 0.005 | 0.115434 | -0.004646 | 0.01464 | 0.000186 | PASS |
| equity.snowball.quad.stretch_0.02 | near_expiry | gamma | -0.10491 | 0.0177 | -0.111829 | -0.006919 | 0.04234 | 0.0001699 | PASS |
| equity.snowball.quad.stretch_0.02 | discrete_ki | pv | 96.909 | 0.00376 | 96.9061 | -0.00287 | 0.01038 | 0.001668 | PASS |
| equity.snowball.quad.stretch_0.02 | discrete_ki | delta | 0.600136 | 0.00551 | 0.596068 | -0.004068 | 0.01509 | 5.753e-05 | PASS |
| equity.snowball.quad.stretch_0.02 | discrete_ki | gamma | -0.031217 | 0.0102 | -0.0492621 | -0.01805 | 0.03851 | 2.987e-05 | PASS |
| equity.snowball.quad.stretch_0.02 | european_ki | pv | 98.5502 | 0.0032 | 98.5443 | -0.00589 | 0.01228 | 0.001499 | PASS |
| equity.snowball.quad.stretch_0.02 | european_ki | delta | 0.401999 | 0.00596 | 0.401216 | -0.000783 | 0.0127 | 8.584e-05 | PASS |
| equity.snowball.quad.stretch_0.02 | european_ki | gamma | -0.039412 | 0.0111 | -0.0424429 | -0.003031 | 0.02525 | 1.755e-05 | PASS |
| equity.snowball.quad.stretch_0.02 | stepdown_ko | pv | 97.1182 | 0.00272 | 97.1206 | 0.002376 | 0.007819 | 0.001601 | PASS |
| equity.snowball.quad.stretch_0.02 | stepdown_ko | delta | 0.56238 | 0.00611 | 0.568747 | 0.006367 | 0.01859 | 3.567e-05 | PASS |
| equity.snowball.quad.stretch_0.02 | stepdown_ko | gamma | -0.0469975 | 0.0102 | -0.0480257 | -0.001028 | 0.02146 | 0.0003585 | PASS |
| equity.snowball.quad.stretch_0.02 | stepdown_near_last_ko | pv | 95.2078 | 0.00547 | 95.2079 | 7.195e-05 | 0.011 | 0.001906 | PASS |
| equity.snowball.quad.stretch_0.02 | stepdown_near_last_ko | delta | 0.705585 | 0.00352 | 0.702727 | -0.002858 | 0.009907 | 0.0006852 | PASS |
| equity.snowball.quad.stretch_0.02 | stepdown_near_last_ko | gamma | -0.0608893 | 0.0158 | -0.0415105 | 0.01938 | 0.05101 | 0.001359 | PASS |
| equity.snowball.quad.stretch_0.02 | parachute | pv | 98.5509 | 0.00387 | 98.5443 | -0.006632 | 0.01438 | 0.001499 | PASS |
| equity.snowball.quad.stretch_0.02 | parachute | delta | 0.402179 | 0.00352 | 0.401216 | -0.0009634 | 0.008013 | 8.585e-05 | PASS |
| equity.snowball.quad.stretch_0.02 | parachute | gamma | -0.0426591 | 0.00547 | -0.0424429 | 0.0002162 | 0.01116 | 1.754e-05 | PASS |
| equity.snowball.quad.stretch_0.02 | parachute_near_ki | pv | 88.8357 | 0.00532 | 88.8352 | -0.0004992 | 0.01113 | 0.0007693 | PASS |
| equity.snowball.quad.stretch_0.02 | parachute_near_ki | delta | 0.962974 | 0.00419 | 0.959867 | -0.003107 | 0.01148 | 0.000137 | PASS |
| equity.snowball.quad.stretch_0.02 | parachute_near_ki | gamma | -0.0306684 | 0.00214 | -0.032248 | -0.00158 | 0.005861 | 3.527e-06 | PASS |
| equity.snowball.quad.stretch_0.02 | ki_stepdown | pv | 97.793 | 0.00526 | 97.7936 | 0.0006117 | 0.01114 | 0.002445 | PASS |
| equity.snowball.quad.stretch_0.02 | ki_stepdown | delta | 0.518504 | 0.00638 | 0.516526 | -0.001977 | 0.01474 | 0.0001335 | PASS |
| equity.snowball.quad.stretch_0.02 | ki_stepdown | gamma | -0.0485229 | 0.00645 | -0.0542652 | -0.005742 | 0.01864 | 3.471e-05 | PASS |
| equity.snowball.quad.stretch_0.02 | reverse | pv | 96.9637 | 0.0133 | 96.9653 | 0.001589 | 0.02825 | 0.0004864 | PASS |
| equity.snowball.quad.stretch_0.02 | reverse | delta | -0.620071 | 0.00817 | -0.618171 | 0.0019 | 0.01824 | 5.522e-05 | PASS |
| equity.snowball.quad.stretch_0.02 | reverse | gamma | -0.0223816 | 0.0106 | -0.044961 | -0.02258 | 0.04378 | 2.442e-05 | PASS |
| equity.snowball.quad.stretch_0.02 | airbag | pv | 98.9502 | 0.00749 | 98.9591 | 0.008953 | 0.02393 | 0.009462 | PASS |
| equity.snowball.quad.stretch_0.02 | airbag | delta | 0.326244 | 0.00267 | 0.325143 | -0.001101 | 0.006449 | 0.0009927 | PASS |
| equity.snowball.quad.stretch_0.02 | airbag | gamma | -0.0215876 | 0.00556 | -0.0314635 | -0.009876 | 0.02099 | 2.184e-06 | PASS |
| equity.snowball.quad.stretch_0.02 | protection_partial | pv | 96.4653 | 0.00861 | 96.4647 | -0.0005921 | 0.01781 | 8.749e-05 | PASS |
| equity.snowball.quad.stretch_0.02 | protection_partial | delta | 0.636258 | 0.00444 | 0.636341 | 8.367e-05 | 0.008968 | 0.0001045 | PASS |
| equity.snowball.quad.stretch_0.02 | protection_partial | gamma | -0.034932 | 0.00665 | -0.0470477 | -0.01212 | 0.02542 | 2.008e-05 | PASS |
| equity.snowball.quad.stretch_0.02 | protection_full | pv | 101.701 | 0.00533 | 101.703 | 0.002049 | 0.0127 | 0.0005742 | PASS |
| equity.snowball.quad.stretch_0.02 | protection_full | delta | -0.0145239 | 0.000962 | -0.0146408 | -0.0001169 | 0.002041 | 8.274e-06 | PASS |
| equity.snowball.quad.stretch_0.02 | protection_full | gamma | -0.00949786 | 0.00421 | -0.0151908 | -0.005693 | 0.01412 | 1.606e-05 | PASS |
| equity.snowball.quad.stretch_0.02 | participation | pv | 99.0814 | 0.00697 | 99.0821 | 0.0007267 | 0.01467 | 0.0002396 | PASS |
| equity.snowball.quad.stretch_0.02 | participation | delta | 0.311175 | 0.00264 | 0.311158 | -1.722e-05 | 0.00529 | 4.872e-05 | PASS |
| equity.snowball.quad.stretch_0.02 | participation | gamma | -0.0222669 | 0.00475 | -0.0311666 | -0.0089 | 0.0184 | 1.815e-05 | PASS |
| equity.snowball.quad.stretch_0.02 | call_rebate | pv | 96.3056 | 0.0061 | 96.3007 | -0.004966 | 0.01716 | 0.0006221 | PASS |
| equity.snowball.quad.stretch_0.02 | call_rebate | delta | 0.64663 | 0.004 | 0.648418 | 0.001788 | 0.009787 | 0.0001454 | PASS |
| equity.snowball.quad.stretch_0.02 | call_rebate | gamma | -0.0418866 | 0.0046 | -0.0454045 | -0.003518 | 0.01273 | 1.469e-05 | PASS |
| equity.snowball.quad.stretch_0.02 | disable_ko_after_ki | pv | 95.6655 | 0.0051 | 95.6679 | 0.002326 | 0.01253 | 0.0006949 | PASS |
| equity.snowball.quad.stretch_0.02 | disable_ko_after_ki | delta | 0.747742 | 0.00634 | 0.744687 | -0.003055 | 0.01573 | 1.265e-05 | PASS |
| equity.snowball.quad.stretch_0.02 | disable_ko_after_ki | gamma | -0.0391323 | 0.0119 | -0.0550846 | -0.01595 | 0.03984 | 1.563e-05 | PASS |
| equity.snowball.quad.stretch_0.02 | disable_ko_after_ki_knocked_in | pv | 76.1557 | 1.02e-05 | 76.1557 | 2.417e-05 | 4.451e-05 | 0.0005746 | PASS |
| equity.snowball.quad.stretch_0.02 | disable_ko_after_ki_knocked_in | delta | 0.798655 | 1.3e-07 | 0.798656 | 7.622e-07 | 1.023e-06 | 3.704e-05 | PASS |
| equity.snowball.quad.stretch_0.02 | disable_ko_after_ki_knocked_in | gamma | -0.0143123 | 4.91e-08 | -0.0143124 | -7.195e-08 | 1.702e-07 | 6.543e-08 | PASS |
| equity.snowball.quad.stretch_0.02 | coupon_at_expiry | pv | 95.0884 | 0.00863 | 95.0881 | -0.0002814 | 0.01753 | 0.0003404 | PASS |
| equity.snowball.quad.stretch_0.02 | coupon_at_expiry | delta | 0.539489 | 0.00401 | 0.539491 | 1.863e-06 | 0.008026 | 6.137e-05 | PASS |
| equity.snowball.quad.stretch_0.02 | coupon_at_expiry | gamma | -0.0345602 | 0.00577 | -0.0464604 | -0.0119 | 0.02345 | 2.224e-05 | PASS |
| equity.snowball.quad.stretch_0.02 | not_annualized | pv | 104.568 | 0.00886 | 104.565 | -0.002385 | 0.02011 | 0.002715 | PASS |
| equity.snowball.quad.stretch_0.02 | not_annualized | delta | 1.22899 | 0.00736 | 1.2295 | 0.0005095 | 0.01524 | 0.00037 | PASS |
| equity.snowball.quad.stretch_0.02 | not_annualized | gamma | -0.0362235 | 0.0126 | -0.0499813 | -0.01376 | 0.039 | 7.462e-06 | PASS |
| equity.snowball.quad.stretch_0.02 | ko_rate_step | pv | 96.8561 | 0.00951 | 96.855 | -0.001049 | 0.02008 | 0.0001275 | PASS |
| equity.snowball.quad.stretch_0.02 | ko_rate_step | delta | 0.60049 | 0.00427 | 0.60099 | 0.0005008 | 0.009049 | 8.963e-05 | PASS |
| equity.snowball.quad.stretch_0.02 | ko_rate_step | gamma | -0.0368043 | 0.00609 | -0.0486918 | -0.01189 | 0.02407 | 2.318e-05 | PASS |

## Aggregate bias

| candidate | quantity | cells | mean bias (c) | SE (c) | passed |
|---|---|---|---|---|---|
| equity.snowball.pde | pv | 23 | -0.001527 | 0.00138 | yes |
| equity.snowball.pde | delta | 23 | -0.0007966 | 0.00102 | yes |
| equity.snowball.pde | gamma | 23 | -0.006991 | 0.00194 | yes |
| equity.snowball.quad | pv | 23 | -0.0001098 | 0.00138 | yes |
| equity.snowball.quad | delta | 23 | -0.0007989 | 0.00102 | yes |
| equity.snowball.quad | gamma | 23 | -0.007255 | 0.00194 | yes |
| equity.snowball.quad.stretch_0.02 | pv | 23 | -0.0002222 | 0.00138 | yes |
| equity.snowball.quad.stretch_0.02 | delta | 23 | -0.0007818 | 0.00102 | yes |
| equity.snowball.quad.stretch_0.02 | gamma | 23 | -0.007218 | 0.00194 | yes |
