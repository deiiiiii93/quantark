# Certification report: ko-reset-flat-bsm

Evidence digest: `2913b3744b166bda41656dfbf948e4bf3912ffa28685a39d50988704dc62f3ca`

Machine: `arm64` / macOS-26.6-arm64-arm-64bit - Python 3.11.8, NumPy 2.4.6, quantark `9c5c20ec90bdd0438f503be1b732ccbabdd84c16`

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
| equity.ko_reset_snowball.quad | grid.readout | legacy_linear |
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
| disable_ko_after_ki | 4 | se_budget_met | delta: 0.00814, gamma: 0.0233, pv: 0.0101 |
| discrete_ki | 4 | se_budget_met | delta: 0.00409, gamma: 0.00883, pv: 0.00999 |
| european_ki | 4 | se_budget_met | delta: 0.00303, gamma: 0.0201, pv: 0.00975 |
| ki_stepdown | 4 | se_budget_met | delta: 0.00362, gamma: 0.0264, pv: 0.0132 |
| low_vol | 4 | se_budget_met | delta: 0.00292, gamma: 0.0118, pv: 0.0116 |
| near_expiry | 4 | se_budget_met | delta: 0.00117, gamma: 0.00329, pv: 0.00505 |
| near_ki | 4 | se_budget_met | delta: 0.00536, gamma: 0.0268, pv: 0.00618 |
| near_pre_ko | 4 | se_budget_met | delta: 0.00233, gamma: 0.0126, pv: 0.0132 |
| ordinary | 4 | se_budget_met | delta: 0.00868, gamma: 0.0297, pv: 0.011 |
| parachute | 4 | se_budget_met | delta: 0.0047, gamma: 0.0274, pv: 0.00947 |
| parachute_near_ki | 4 | se_budget_met | delta: 0.00552, gamma: 0.0271, pv: 0.00666 |
| stepdown | 4 | se_budget_met | delta: 0.00639, gamma: 0.0153, pv: 0.0175 |
| stepdown_near_last_pre_ko | 4 | se_budget_met | delta: 0.00784, gamma: 0.00291, pv: 0.0104 |

Sampling policy: 65536 paths/batch, 4-32 batches, seed 20260818, bump 0.01.

## Cells

| candidate | case | quantity | reference | SE | candidate | err (c) | interval (c) | envelope (c) | verdict |
|---|---|---|---|---|---|---|---|---|---|
| equity.ko_reset_snowball.pde | ordinary | pv | 96.6613 | 0.011 | 96.6702 | 0.008933 | 0.03096 | 0.001207 | PASS |
| equity.ko_reset_snowball.pde | ordinary | delta | 0.621389 | 0.00868 | 0.612941 | -0.008448 | 0.02581 | 0.0001998 | PASS |
| equity.ko_reset_snowball.pde | ordinary | gamma | -0.0416238 | 0.0297 | -0.0464266 | -0.004803 | 0.0642 | 0.0004916 | PASS |
| equity.ko_reset_snowball.pde | near_pre_ko | pv | 98.0475 | 0.0132 | 98.0527 | 0.005218 | 0.03166 | 0.001094 | PASS |
| equity.ko_reset_snowball.pde | near_pre_ko | delta | 0.487059 | 0.00233 | 0.491112 | 0.004053 | 0.008722 | 0.0002483 | PASS |
| equity.ko_reset_snowball.pde | near_pre_ko | gamma | -0.034448 | 0.0126 | -0.0502831 | -0.01584 | 0.041 | 0.0003092 | PASS |
| equity.ko_reset_snowball.pde | near_ki | pv | 79.0154 | 0.00618 | 79.0229 | 0.007533 | 0.01989 | 0.002191 | PASS |
| equity.ko_reset_snowball.pde | near_ki | delta | 1.06603 | 0.00536 | 1.05978 | -0.006256 | 0.01697 | 0.0001146 | PASS |
| equity.ko_reset_snowball.pde | near_ki | gamma | -0.0382894 | 0.0268 | -0.0593725 | -0.02108 | 0.07472 | 0.001165 | PASS |
| equity.ko_reset_snowball.pde | below_ki | pv | 74.0373 | 0.0119 | 74.0367 | -0.0006474 | 0.02455 | 0.001612 | PASS |
| equity.ko_reset_snowball.pde | below_ki | delta | 1.10845 | 0.00477 | 1.10635 | -0.002095 | 0.01163 | 2.458e-05 | PASS |
| equity.ko_reset_snowball.pde | below_ki | gamma | -0.0131948 | 0.0123 | 0.00544322 | 0.01864 | 0.04328 | 1.854e-05 | PASS |
| equity.ko_reset_snowball.pde | low_vol | pv | 100.165 | 0.0116 | 100.164 | -0.0009706 | 0.02419 | 0.00343 | PASS |
| equity.ko_reset_snowball.pde | low_vol | delta | 0.359471 | 0.00292 | 0.36796 | 0.008488 | 0.01434 | 0.000816 | PASS |
| equity.ko_reset_snowball.pde | low_vol | gamma | -0.0863362 | 0.0118 | -0.0957119 | -0.009376 | 0.03299 | 0.0001622 | PASS |
| equity.ko_reset_snowball.pde | near_expiry | pv | 99.5678 | 0.00505 | 99.5693 | 0.001497 | 0.01161 | 0.001347 | PASS |
| equity.ko_reset_snowball.pde | near_expiry | delta | 0.284584 | 0.00117 | 0.284919 | 0.0003354 | 0.002671 | 0.0001339 | PASS |
| equity.ko_reset_snowball.pde | near_expiry | gamma | -0.0472864 | 0.00329 | -0.043639 | 0.003647 | 0.01023 | 0.0004911 | PASS |
| equity.ko_reset_snowball.pde | discrete_ki | pv | 96.7295 | 0.00999 | 96.7322 | 0.002688 | 0.02268 | 9.864e-06 | PASS |
| equity.ko_reset_snowball.pde | discrete_ki | delta | 0.612278 | 0.00409 | 0.607343 | -0.004935 | 0.01311 | 8.497e-05 | PASS |
| equity.ko_reset_snowball.pde | discrete_ki | gamma | -0.0366268 | 0.00883 | -0.0466655 | -0.01004 | 0.02771 | 0.0004785 | PASS |
| equity.ko_reset_snowball.pde | european_ki | pv | 97.6357 | 0.00975 | 97.6496 | 0.01384 | 0.03335 | 7.589e-05 | PASS |
| equity.ko_reset_snowball.pde | european_ki | delta | 0.503157 | 0.00303 | 0.498078 | -0.005079 | 0.01114 | 7.359e-05 | PASS |
| equity.ko_reset_snowball.pde | european_ki | gamma | -0.040584 | 0.0201 | -0.0430475 | -0.002463 | 0.04258 | 0.0002829 | PASS |
| equity.ko_reset_snowball.pde | stepdown | pv | 98.2196 | 0.0175 | 98.2098 | -0.009763 | 0.04476 | 0.0008175 | PASS |
| equity.ko_reset_snowball.pde | stepdown | delta | 0.44371 | 0.00639 | 0.440571 | -0.003139 | 0.01592 | 0.0002294 | PASS |
| equity.ko_reset_snowball.pde | stepdown | gamma | -0.0722007 | 0.0153 | -0.0435208 | 0.02868 | 0.05921 | 3.315e-05 | PASS |
| equity.ko_reset_snowball.pde | stepdown_near_last_pre_ko | pv | 96.7042 | 0.0104 | 96.6924 | -0.0118 | 0.03251 | 0.0004656 | PASS |
| equity.ko_reset_snowball.pde | stepdown_near_last_pre_ko | delta | 0.574327 | 0.00784 | 0.570512 | -0.003816 | 0.01949 | 0.0002187 | PASS |
| equity.ko_reset_snowball.pde | stepdown_near_last_pre_ko | gamma | -0.058302 | 0.00291 | -0.0424848 | 0.01582 | 0.02165 | 3.732e-05 | PASS |
| equity.ko_reset_snowball.pde | parachute | pv | 97.4197 | 0.00947 | 97.4293 | 0.009548 | 0.02849 | 0.003274 | PASS |
| equity.ko_reset_snowball.pde | parachute | delta | 0.550792 | 0.0047 | 0.543561 | -0.007232 | 0.01664 | 0.0003841 | PASS |
| equity.ko_reset_snowball.pde | parachute | gamma | -0.043298 | 0.0274 | -0.0502378 | -0.00694 | 0.06179 | 0.0002156 | PASS |
| equity.ko_reset_snowball.pde | parachute_near_ki | pv | 79.0793 | 0.00666 | 79.0904 | 0.01109 | 0.02442 | 0.000817 | PASS |
| equity.ko_reset_snowball.pde | parachute_near_ki | delta | 1.17078 | 0.00552 | 1.16264 | -0.008146 | 0.01919 | 0.001597 | PASS |
| equity.ko_reset_snowball.pde | parachute_near_ki | gamma | 0.0247501 | 0.0271 | 0.033174 | 0.008424 | 0.06267 | 0.0005128 | PASS |
| equity.ko_reset_snowball.pde | ki_stepdown | pv | 97.7019 | 0.0132 | 97.7119 | 0.009986 | 0.03631 | 0.002712 | PASS |
| equity.ko_reset_snowball.pde | ki_stepdown | delta | 0.505667 | 0.00362 | 0.505498 | -0.0001694 | 0.007405 | 7.321e-05 | PASS |
| equity.ko_reset_snowball.pde | ki_stepdown | gamma | -0.0474391 | 0.0264 | -0.0478426 | -0.0004034 | 0.05321 | 4.737e-05 | PASS |
| equity.ko_reset_snowball.pde | disable_ko_after_ki | pv | 95.3788 | 0.0101 | 95.3706 | -0.008212 | 0.02841 | 0.003169 | PASS |
| equity.ko_reset_snowball.pde | disable_ko_after_ki | delta | 0.786006 | 0.00814 | 0.782318 | -0.003688 | 0.01996 | 0.0004683 | PASS |
| equity.ko_reset_snowball.pde | disable_ko_after_ki | gamma | -0.0704925 | 0.0233 | -0.0569868 | 0.01351 | 0.06007 | 0.0006517 | PASS |
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
| equity.ko_reset_snowball.quad | european_ki | pv | 97.6357 | 0.00975 | 97.6571 | 0.02136 | 0.04087 | 0.00899 | PASS |
| equity.ko_reset_snowball.quad | european_ki | delta | 0.503157 | 0.00303 | 0.497717 | -0.00544 | 0.0115 | 0.0008737 | PASS |
| equity.ko_reset_snowball.quad | european_ki | gamma | -0.040584 | 0.0201 | -0.0429305 | -0.002346 | 0.04246 | 0.02941 | PASS |
| equity.ko_reset_snowball.quad | stepdown | pv | 98.2196 | 0.0175 | 98.2224 | 0.002795 | 0.03779 | 0.04126 | PASS |
| equity.ko_reset_snowball.quad | stepdown | delta | 0.44371 | 0.00639 | 0.439574 | -0.004136 | 0.01692 | 0.003078 | PASS |
| equity.ko_reset_snowball.quad | stepdown | gamma | -0.0722007 | 0.0153 | -0.0434782 | 0.02872 | 0.05925 | 0.00726 | PASS |
| equity.ko_reset_snowball.quad | stepdown_near_last_pre_ko | pv | 96.7042 | 0.0104 | 96.7036 | -0.0005741 | 0.02128 | 0.0391 | PASS |
| equity.ko_reset_snowball.quad | stepdown_near_last_pre_ko | delta | 0.574327 | 0.00784 | 0.571444 | -0.002884 | 0.01856 | 0.00261 | PASS |
| equity.ko_reset_snowball.quad | stepdown_near_last_pre_ko | gamma | -0.058302 | 0.00291 | -0.0384921 | 0.01981 | 0.02564 | 0.009001 | PASS |
| equity.ko_reset_snowball.quad | parachute | pv | 97.4197 | 0.00947 | 97.4422 | 0.0225 | 0.04145 | 0.005514 | PASS |
| equity.ko_reset_snowball.quad | parachute | delta | 0.550792 | 0.0047 | 0.542663 | -0.008129 | 0.01754 | 0.0009742 | PASS |
| equity.ko_reset_snowball.quad | parachute | gamma | -0.043298 | 0.0274 | -0.0500342 | -0.006736 | 0.06158 | 0.03475 | PASS |
| equity.ko_reset_snowball.quad | parachute_near_ki | pv | 79.0793 | 0.00666 | 79.1024 | 0.02309 | 0.03642 | 0.08927 | PASS |
| equity.ko_reset_snowball.quad | parachute_near_ki | delta | 1.17078 | 0.00552 | 1.1637 | -0.007085 | 0.01813 | 0.002219 | PASS |
| equity.ko_reset_snowball.quad | parachute_near_ki | gamma | 0.0247501 | 0.0271 | 0.0222077 | -0.002542 | 0.05679 | 0.006441 | PASS |
| equity.ko_reset_snowball.quad | ki_stepdown | pv | 97.7019 | 0.0132 | 97.7195 | 0.01756 | 0.04388 | 0.006053 | PASS |
| equity.ko_reset_snowball.quad | ki_stepdown | delta | 0.505667 | 0.00362 | 0.505269 | -0.0003986 | 0.007634 | 0.0008806 | PASS |
| equity.ko_reset_snowball.quad | ki_stepdown | gamma | -0.0474391 | 0.0264 | -0.0477298 | -0.0002907 | 0.0531 | 0.03329 | PASS |
| equity.ko_reset_snowball.quad | disable_ko_after_ki | pv | 95.3788 | 0.0101 | 95.3845 | 0.005679 | 0.02588 | 0.01119 | PASS |
| equity.ko_reset_snowball.quad | disable_ko_after_ki | delta | 0.786006 | 0.00814 | 0.781059 | -0.004947 | 0.02122 | 0.0007447 | PASS |
| equity.ko_reset_snowball.quad | disable_ko_after_ki | gamma | -0.0704925 | 0.0233 | -0.0569152 | 0.01358 | 0.06014 | 0.0376 | PASS |

## Aggregate bias

| candidate | quantity | cells | mean bias (c) | SE (c) | passed |
|---|---|---|---|---|---|
| equity.ko_reset_snowball.pde | pv | 14 | 0.002781 | 0.00291 | yes |
| equity.ko_reset_snowball.pde | delta | 14 | -0.002866 | 0.00143 | yes |
| equity.ko_reset_snowball.pde | gamma | 14 | 0.001269 | 0.0053 | yes |
| equity.ko_reset_snowball.quad | pv | 14 | 0.01216 | 0.00291 | yes |
| equity.ko_reset_snowball.quad | delta | 14 | -0.003233 | 0.00143 | yes |
| equity.ko_reset_snowball.quad | gamma | 14 | 0.001653 | 0.0053 | yes |
