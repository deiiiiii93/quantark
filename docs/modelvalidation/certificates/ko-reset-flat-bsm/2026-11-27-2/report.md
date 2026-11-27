# Certification report: ko-reset-flat-bsm

Evidence digest: `f594b4f61192e992356a32f704f1c2a9705d08a9cb534f421de79839c8e97c16`

Machine: `arm64` / macOS-27.2.1-arm64-arm-64bit - Python 3.11.8, NumPy 2.4.6, quantark `fb464de185c49445642d3cf934e1adec4bca6dfc`

## Amendment

| field | value |
|---|---|
| parent | docs/modelvalidation/certificates/ko-reset-flat-bsm/2026-11-27/certificate.json |
| parent digest | `c9b2bbf1f8869a391c1c7345a9133d161a62108e0c0d6524d326afecb47cca44` |
| reason | 0.16.0 run 5 T5 (owner-signed 2026-11-27, 'approved'): QUAD V1 linear convolutions run on the smallest even 5-smooth FFT length at or above the legacy length (real FFTs; the adjoint transposes on the same length). Round-off moves only (anchors <= 1.5e-11 rel, prices ~1e-15 rel). QUAD candidates re-run against the carried benchmarks; other candidates unchanged. |
| re-priced | 84 cell(s) |
| carried forward | 42 cell(s) |

## Decisions

| candidate | decision |
|---|---|
| equity.ko_reset_snowball.pde | ADMITTED |
| equity.ko_reset_snowball.quad | ADMITTED |
| equity.ko_reset_snowball.quad.stretch_0.02 | ADMITTED |

Bounds: cell 0.5 c, mean signed bias 0.1 c, standard-error budget 0.25 x cell, interval k 2.

## Engine configuration

Resolved rather than named: a profile such as `standard` is an indirection whose meaning can change between releases. These are the requested settings.

| engine | setting | value |
|---|---|---|
| equity.ko_reset_snowball.pde | accuracy | standard |
| equity.ko_reset_snowball.pde | engine | KOResetSnowballPDESolver |
| equity.ko_reset_snowball.pde | grid.bounds | (None, None) |
| equity.ko_reset_snowball.pde | grid.day_count | 252 |
| equity.ko_reset_snowball.pde | grid.discrete_monitoring_damping | ['half_step', 1, 1] |
| equity.ko_reset_snowball.pde | grid.eps_crit | 0.003 |
| equity.ko_reset_snowball.pde | grid.event_damping_steps | 2 |
| equity.ko_reset_snowball.pde | grid.max_points | 2000 |
| equity.ko_reset_snowball.pde | grid.max_steps | 12000 |
| equity.ko_reset_snowball.pde | grid.num_std | 4 |
| equity.ko_reset_snowball.pde | grid.points | 400 |
| equity.ko_reset_snowball.pde | grid.steps_per_day | 4 |
| equity.ko_reset_snowball.pde | grid.terminal_damping_steps | 1 |
| equity.ko_reset_snowball.quad | engine | KOResetSnowballQuadEngine |
| equity.ko_reset_snowball.quad | grid.align_cell_stretch | -- |
| equity.ko_reset_snowball.quad | grid.align_priority | auto |
| equity.ko_reset_snowball.quad | grid.barrier_reach_stddevs | -- |
| equity.ko_reset_snowball.quad | grid.bgk_min_ki_observations | 100 |
| equity.ko_reset_snowball.quad | grid.bus_days_in_year | 252 |
| equity.ko_reset_snowball.quad | grid.continuous_steps_per_year | 0 |
| equity.ko_reset_snowball.quad | grid.continuous_term_structure | piecewise_constant |
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
| equity.ko_reset_snowball.quad.stretch_0.02 | align_cell_stretch | 0.02 |
| equity.ko_reset_snowball.quad.stretch_0.02 | engine | KOResetSnowballQuadEngine |
| equity.ko_reset_snowball.quad.stretch_0.02 | grid.align_cell_stretch | 0.02 |
| equity.ko_reset_snowball.quad.stretch_0.02 | grid.align_priority | auto |
| equity.ko_reset_snowball.quad.stretch_0.02 | grid.barrier_reach_stddevs | -- |
| equity.ko_reset_snowball.quad.stretch_0.02 | grid.bgk_min_ki_observations | 100 |
| equity.ko_reset_snowball.quad.stretch_0.02 | grid.bus_days_in_year | 252 |
| equity.ko_reset_snowball.quad.stretch_0.02 | grid.continuous_steps_per_year | 0 |
| equity.ko_reset_snowball.quad.stretch_0.02 | grid.continuous_term_structure | piecewise_constant |
| equity.ko_reset_snowball.quad.stretch_0.02 | grid.event_projection | cell_average |
| equity.ko_reset_snowball.quad.stretch_0.02 | grid.event_smoothing_cells | 1 |
| equity.ko_reset_snowball.quad.stretch_0.02 | grid.event_smoothing_kernel | cosine |
| equity.ko_reset_snowball.quad.stretch_0.02 | grid.event_smoothing_log_width | 0.002 |
| equity.ko_reset_snowball.quad.stretch_0.02 | grid.event_smoothing_mode | fixed |
| equity.ko_reset_snowball.quad.stretch_0.02 | grid.fft_filter_alpha | 12 |
| equity.ko_reset_snowball.quad.stretch_0.02 | grid.fft_filter_power | 8 |
| equity.ko_reset_snowball.quad.stretch_0.02 | grid.fft_padding_factor | 2 |
| equity.ko_reset_snowball.quad.stretch_0.02 | grid.filter_unreachable_barriers | True |
| equity.ko_reset_snowball.quad.stretch_0.02 | grid.grid_points | 1001 |
| equity.ko_reset_snowball.quad.stretch_0.02 | grid.integration_rule | trapezoid |
| equity.ko_reset_snowball.quad.stretch_0.02 | grid.ki_monitoring_mode | exact_discrete |
| equity.ko_reset_snowball.quad.stretch_0.02 | grid.max_adaptive_grid_points | 5001 |
| equity.ko_reset_snowball.quad.stretch_0.02 | grid.min_diffusion_stddev_cells | 2.5 |
| equity.ko_reset_snowball.quad.stretch_0.02 | grid.num_std_devs | 10 |
| equity.ko_reset_snowball.quad.stretch_0.02 | grid.readout | legacy_linear |
| equity.ko_reset_snowball.quad.stretch_0.02 | grid.stability_preset | -- |
| equity.ko_reset_snowball.quad.stretch_0.02 | grid_points | 1001 |
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
| equity.ko_reset_snowball.pde | ordinary | delta | 0.621389 | 0.00868 | 0.612968 | -0.008421 | 0.02578 | 0.0001829 | PASS |
| equity.ko_reset_snowball.pde | ordinary | gamma | -0.0416238 | 0.0297 | -0.0465414 | -0.004918 | 0.06432 | 8.491e-06 | PASS |
| equity.ko_reset_snowball.pde | near_pre_ko | pv | 98.0475 | 0.0132 | 98.0527 | 0.005218 | 0.03166 | 0.001094 | PASS |
| equity.ko_reset_snowball.pde | near_pre_ko | delta | 0.487059 | 0.00233 | 0.491121 | 0.004062 | 0.008731 | 0.0002426 | PASS |
| equity.ko_reset_snowball.pde | near_pre_ko | gamma | -0.034448 | 0.0126 | -0.0501512 | -0.0157 | 0.04086 | 2.643e-06 | PASS |
| equity.ko_reset_snowball.pde | near_ki | pv | 79.0154 | 0.00618 | 79.0229 | 0.007533 | 0.01989 | 0.002191 | PASS |
| equity.ko_reset_snowball.pde | near_ki | delta | 1.06603 | 0.00536 | 1.05887 | -0.007162 | 0.01787 | 0.0002489 | PASS |
| equity.ko_reset_snowball.pde | near_ki | gamma | -0.0382894 | 0.0268 | -0.0531229 | -0.01483 | 0.06847 | 0.002163 | PASS |
| equity.ko_reset_snowball.pde | below_ki | pv | 74.0373 | 0.0119 | 74.0367 | -0.0006474 | 0.02455 | 0.001612 | PASS |
| equity.ko_reset_snowball.pde | below_ki | delta | 1.10845 | 0.00477 | 1.10635 | -0.002097 | 0.01163 | 2.519e-05 | PASS |
| equity.ko_reset_snowball.pde | below_ki | gamma | -0.0131948 | 0.0123 | 0.00544551 | 0.01864 | 0.04328 | 3.188e-06 | PASS |
| equity.ko_reset_snowball.pde | low_vol | pv | 100.165 | 0.0116 | 100.164 | -0.0009706 | 0.02419 | 0.00343 | PASS |
| equity.ko_reset_snowball.pde | low_vol | delta | 0.359471 | 0.00292 | 0.367943 | 0.008472 | 0.01432 | 0.0008092 | PASS |
| equity.ko_reset_snowball.pde | low_vol | gamma | -0.0863362 | 0.0118 | -0.095418 | -0.009082 | 0.03269 | 9.846e-05 | PASS |
| equity.ko_reset_snowball.pde | near_expiry | pv | 99.5678 | 0.00505 | 99.5693 | 0.001497 | 0.01161 | 0.001347 | PASS |
| equity.ko_reset_snowball.pde | near_expiry | delta | 0.284584 | 0.00117 | 0.284906 | 0.0003219 | 0.002657 | 0.000149 | PASS |
| equity.ko_reset_snowball.pde | near_expiry | gamma | -0.0472864 | 0.00329 | -0.0441565 | 0.00313 | 0.00971 | 2.648e-05 | PASS |
| equity.ko_reset_snowball.pde | discrete_ki | pv | 96.7295 | 0.00999 | 96.7307 | 0.001171 | 0.02116 | 8.904e-06 | PASS |
| equity.ko_reset_snowball.pde | discrete_ki | delta | 0.612278 | 0.00409 | 0.607257 | -0.005022 | 0.0132 | 6.877e-05 | PASS |
| equity.ko_reset_snowball.pde | discrete_ki | gamma | -0.0366268 | 0.00883 | -0.0467742 | -0.01015 | 0.02782 | 2.775e-06 | PASS |
| equity.ko_reset_snowball.pde | european_ki | pv | 97.6357 | 0.00975 | 97.6496 | 0.01383 | 0.03333 | 7.667e-05 | PASS |
| equity.ko_reset_snowball.pde | european_ki | delta | 0.503157 | 0.00303 | 0.498086 | -0.005071 | 0.01113 | 6.424e-05 | PASS |
| equity.ko_reset_snowball.pde | european_ki | gamma | -0.040584 | 0.0201 | -0.0431122 | -0.002528 | 0.04264 | 1.593e-06 | PASS |
| equity.ko_reset_snowball.pde | stepdown | pv | 98.2196 | 0.0175 | 98.2098 | -0.009763 | 0.04476 | 0.0008175 | PASS |
| equity.ko_reset_snowball.pde | stepdown | delta | 0.44371 | 0.00639 | 0.440582 | -0.003127 | 0.01591 | 0.0002216 | PASS |
| equity.ko_reset_snowball.pde | stepdown | gamma | -0.0722007 | 0.0153 | -0.0435385 | 0.02866 | 0.05919 | 1.391e-07 | PASS |
| equity.ko_reset_snowball.pde | stepdown_near_last_pre_ko | pv | 96.7042 | 0.0104 | 96.6924 | -0.0118 | 0.03251 | 0.0004656 | PASS |
| equity.ko_reset_snowball.pde | stepdown_near_last_pre_ko | delta | 0.574327 | 0.00784 | 0.570531 | -0.003796 | 0.01947 | 0.0002109 | PASS |
| equity.ko_reset_snowball.pde | stepdown_near_last_pre_ko | gamma | -0.058302 | 0.00291 | -0.0425504 | 0.01575 | 0.02158 | 8.51e-06 | PASS |
| equity.ko_reset_snowball.pde | parachute | pv | 97.4197 | 0.00947 | 97.4293 | 0.009548 | 0.02849 | 0.003274 | PASS |
| equity.ko_reset_snowball.pde | parachute | delta | 0.550792 | 0.0047 | 0.543573 | -0.007219 | 0.01663 | 0.0003765 | PASS |
| equity.ko_reset_snowball.pde | parachute | gamma | -0.043298 | 0.0274 | -0.0502894 | -0.006991 | 0.06184 | 1.266e-05 | PASS |
| equity.ko_reset_snowball.pde | parachute_near_ki | pv | 79.0793 | 0.00666 | 79.0904 | 0.01109 | 0.02442 | 0.000817 | PASS |
| equity.ko_reset_snowball.pde | parachute_near_ki | delta | 1.17078 | 0.00552 | 1.16313 | -0.007658 | 0.0187 | 0.001411 | PASS |
| equity.ko_reset_snowball.pde | parachute_near_ki | gamma | 0.0247501 | 0.0271 | 0.029806 | 0.005056 | 0.0593 | 0.0001616 | PASS |
| equity.ko_reset_snowball.pde | ki_stepdown | pv | 97.7019 | 0.0132 | 97.7119 | 0.009944 | 0.03627 | 0.002712 | PASS |
| equity.ko_reset_snowball.pde | ki_stepdown | delta | 0.505667 | 0.00362 | 0.505506 | -0.000161 | 0.007397 | 7.641e-05 | PASS |
| equity.ko_reset_snowball.pde | ki_stepdown | gamma | -0.0474391 | 0.0264 | -0.0479424 | -0.0005033 | 0.05331 | 2.22e-06 | PASS |
| equity.ko_reset_snowball.pde | disable_ko_after_ki | pv | 95.3788 | 0.0101 | 95.3706 | -0.008212 | 0.02841 | 0.003169 | PASS |
| equity.ko_reset_snowball.pde | disable_ko_after_ki | delta | 0.786006 | 0.00814 | 0.782353 | -0.003653 | 0.01992 | 0.0004463 | PASS |
| equity.ko_reset_snowball.pde | disable_ko_after_ki | gamma | -0.0704925 | 0.0233 | -0.0571363 | 0.01336 | 0.05992 | 2.141e-06 | PASS |
| equity.ko_reset_snowball.quad | ordinary | pv | 96.6613 | 0.011 | 96.6726 | 0.01129 | 0.03332 | 0.001992 | PASS |
| equity.ko_reset_snowball.quad | ordinary | delta | 0.621389 | 0.00868 | 0.612284 | -0.009105 | 0.02646 | 0.0004686 | PASS |
| equity.ko_reset_snowball.quad | ordinary | gamma | -0.0416238 | 0.0297 | -0.0464988 | -0.004875 | 0.06427 | 5.531e-05 | PASS |
| equity.ko_reset_snowball.quad | near_pre_ko | pv | 98.0475 | 0.0132 | 98.0539 | 0.006445 | 0.03289 | 0.001001 | PASS |
| equity.ko_reset_snowball.quad | near_pre_ko | delta | 0.487059 | 0.00233 | 0.490823 | 0.003764 | 0.008433 | 0.0003217 | PASS |
| equity.ko_reset_snowball.quad | near_pre_ko | gamma | -0.034448 | 0.0126 | -0.0500608 | -0.01561 | 0.04077 | 5.971e-05 | PASS |
| equity.ko_reset_snowball.quad | near_ki | pv | 79.0154 | 0.00618 | 79.0234 | 0.007949 | 0.0203 | 0.003969 | PASS |
| equity.ko_reset_snowball.quad | near_ki | delta | 1.06603 | 0.00536 | 1.05865 | -0.007386 | 0.0181 | 0.0001438 | PASS |
| equity.ko_reset_snowball.quad | near_ki | gamma | -0.0382894 | 0.0268 | -0.0375098 | 0.0007796 | 0.05442 | 0.0002688 | PASS |
| equity.ko_reset_snowball.quad | below_ki | pv | 74.0373 | 0.0119 | 74.0354 | -0.001872 | 0.02577 | 0.002905 | PASS |
| equity.ko_reset_snowball.quad | below_ki | delta | 1.10845 | 0.00477 | 1.10624 | -0.002208 | 0.01174 | 0.0002526 | PASS |
| equity.ko_reset_snowball.quad | below_ki | gamma | -0.0131948 | 0.0123 | 0.00544244 | 0.01864 | 0.04328 | 1.96e-05 | PASS |
| equity.ko_reset_snowball.quad | low_vol | pv | 100.165 | 0.0116 | 100.162 | -0.003072 | 0.02629 | 0.00646 | PASS |
| equity.ko_reset_snowball.quad | low_vol | delta | 0.359471 | 0.00292 | 0.369191 | 0.009719 | 0.01557 | 0.000559 | PASS |
| equity.ko_reset_snowball.quad | low_vol | gamma | -0.0863362 | 0.0118 | -0.095049 | -0.008713 | 0.03232 | 0.0001694 | PASS |
| equity.ko_reset_snowball.quad | near_expiry | pv | 99.5678 | 0.00505 | 99.5696 | 0.001831 | 0.01194 | 0.00213 | PASS |
| equity.ko_reset_snowball.quad | near_expiry | delta | 0.284584 | 0.00117 | 0.285802 | 0.001218 | 0.003554 | 0.000376 | PASS |
| equity.ko_reset_snowball.quad | near_expiry | gamma | -0.0472864 | 0.00329 | -0.0448181 | 0.002468 | 0.009049 | 0.0006648 | PASS |
| equity.ko_reset_snowball.quad | discrete_ki | pv | 96.7295 | 0.00999 | 96.7308 | 0.001291 | 0.02128 | 0 | PASS |
| equity.ko_reset_snowball.quad | discrete_ki | delta | 0.612278 | 0.00409 | 0.60671 | -0.005568 | 0.01374 | 0 | PASS |
| equity.ko_reset_snowball.quad | discrete_ki | gamma | -0.0366268 | 0.00883 | -0.0467454 | -0.01012 | 0.02779 | 0 | PASS |
| equity.ko_reset_snowball.quad | european_ki | pv | 97.6357 | 0.00975 | 97.6501 | 0.01434 | 0.03385 | 0.0005739 | PASS |
| equity.ko_reset_snowball.quad | european_ki | delta | 0.503157 | 0.00303 | 0.497852 | -0.005306 | 0.01137 | 0.0001553 | PASS |
| equity.ko_reset_snowball.quad | european_ki | gamma | -0.040584 | 0.0201 | -0.0430705 | -0.002487 | 0.0426 | 4.695e-05 | PASS |
| equity.ko_reset_snowball.quad | stepdown | pv | 98.2196 | 0.0175 | 98.2082 | -0.01134 | 0.04634 | 0.005247 | PASS |
| equity.ko_reset_snowball.quad | stepdown | delta | 0.44371 | 0.00639 | 0.440734 | -0.002975 | 0.01576 | 0.0003277 | PASS |
| equity.ko_reset_snowball.quad | stepdown | gamma | -0.0722007 | 0.0153 | -0.0434546 | 0.02875 | 0.05927 | 0.001073 | PASS |
| equity.ko_reset_snowball.quad | stepdown_near_last_pre_ko | pv | 96.7042 | 0.0104 | 96.6911 | -0.01307 | 0.03378 | 0.007567 | PASS |
| equity.ko_reset_snowball.quad | stepdown_near_last_pre_ko | delta | 0.574327 | 0.00784 | 0.570205 | -0.004123 | 0.0198 | 0.0008923 | PASS |
| equity.ko_reset_snowball.quad | stepdown_near_last_pre_ko | gamma | -0.058302 | 0.00291 | -0.0429138 | 0.01539 | 0.02122 | 0.001518 | PASS |
| equity.ko_reset_snowball.quad | parachute | pv | 97.4197 | 0.00947 | 97.4331 | 0.01336 | 0.03231 | 0.003584 | PASS |
| equity.ko_reset_snowball.quad | parachute | delta | 0.550792 | 0.0047 | 0.543081 | -0.007711 | 0.01712 | 6.439e-05 | PASS |
| equity.ko_reset_snowball.quad | parachute | gamma | -0.043298 | 0.0274 | -0.0502425 | -0.006945 | 0.06179 | 7.992e-05 | PASS |
| equity.ko_reset_snowball.quad | parachute_near_ki | pv | 79.0793 | 0.00666 | 79.0885 | 0.009219 | 0.02255 | 0.004397 | PASS |
| equity.ko_reset_snowball.quad | parachute_near_ki | delta | 1.17078 | 0.00552 | 1.16394 | -0.006843 | 0.01789 | 0.0005494 | PASS |
| equity.ko_reset_snowball.quad | parachute_near_ki | gamma | 0.0247501 | 0.0271 | 0.0229607 | -0.001789 | 0.05604 | 0.0001302 | PASS |
| equity.ko_reset_snowball.quad | ki_stepdown | pv | 97.7019 | 0.0132 | 97.7123 | 0.01042 | 0.03674 | 0.0008935 | PASS |
| equity.ko_reset_snowball.quad | ki_stepdown | delta | 0.505667 | 0.00362 | 0.505432 | -0.0002353 | 0.007471 | 0.0001659 | PASS |
| equity.ko_reset_snowball.quad | ki_stepdown | gamma | -0.0474391 | 0.0264 | -0.0479056 | -0.0004664 | 0.05327 | 6.42e-05 | PASS |
| equity.ko_reset_snowball.quad | disable_ko_after_ki | pv | 95.3788 | 0.0101 | 95.3759 | -0.002881 | 0.02308 | 0.002128 | PASS |
| equity.ko_reset_snowball.quad | disable_ko_after_ki | delta | 0.786006 | 0.00814 | 0.781163 | -0.004842 | 0.02111 | 1.681e-05 | PASS |
| equity.ko_reset_snowball.quad | disable_ko_after_ki | gamma | -0.0704925 | 0.0233 | -0.0570776 | 0.01341 | 0.05998 | 3.37e-05 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | ordinary | pv | 96.6613 | 0.011 | 96.6726 | 0.01129 | 0.03332 | 0.001992 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | ordinary | delta | 0.621389 | 0.00868 | 0.612284 | -0.009105 | 0.02646 | 0.0004686 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | ordinary | gamma | -0.0416238 | 0.0297 | -0.0464988 | -0.004875 | 0.06427 | 5.531e-05 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | near_pre_ko | pv | 98.0475 | 0.0132 | 98.0539 | 0.006445 | 0.03289 | 0.001001 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | near_pre_ko | delta | 0.487059 | 0.00233 | 0.490823 | 0.003764 | 0.008433 | 0.0003217 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | near_pre_ko | gamma | -0.034448 | 0.0126 | -0.0500608 | -0.01561 | 0.04077 | 5.971e-05 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | near_ki | pv | 79.0154 | 0.00618 | 79.0234 | 0.007949 | 0.0203 | 0.003969 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | near_ki | delta | 1.06603 | 0.00536 | 1.05865 | -0.007386 | 0.0181 | 0.0001438 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | near_ki | gamma | -0.0382894 | 0.0268 | -0.0375098 | 0.0007796 | 0.05442 | 0.0002688 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | below_ki | pv | 74.0373 | 0.0119 | 74.0354 | -0.001872 | 0.02577 | 0.002905 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | below_ki | delta | 1.10845 | 0.00477 | 1.10624 | -0.002208 | 0.01174 | 0.0002526 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | below_ki | gamma | -0.0131948 | 0.0123 | 0.00544244 | 0.01864 | 0.04328 | 1.96e-05 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | low_vol | pv | 100.165 | 0.0116 | 100.162 | -0.003454 | 0.02668 | 0.006736 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | low_vol | delta | 0.359471 | 0.00292 | 0.369256 | 0.009785 | 0.01563 | 0.0005986 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | low_vol | gamma | -0.0863362 | 0.0118 | -0.0950491 | -0.008713 | 0.03233 | 0.0001732 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | near_expiry | pv | 99.5678 | 0.00505 | 99.5697 | 0.001903 | 0.01201 | 0.001264 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | near_expiry | delta | 0.284584 | 0.00117 | 0.285789 | 0.001205 | 0.00354 | 0.0001228 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | near_expiry | gamma | -0.0472864 | 0.00329 | -0.0457481 | 0.001538 | 0.008119 | 0.001561 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | discrete_ki | pv | 96.7295 | 0.00999 | 96.7293 | -0.0001348 | 0.02012 | 0 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | discrete_ki | delta | 0.612278 | 0.00409 | 0.606844 | -0.005435 | 0.01361 | 0 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | discrete_ki | gamma | -0.0366268 | 0.00883 | -0.0467403 | -0.01011 | 0.02778 | 0 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | european_ki | pv | 97.6357 | 0.00975 | 97.6501 | 0.01434 | 0.03385 | 0.0005739 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | european_ki | delta | 0.503157 | 0.00303 | 0.497852 | -0.005306 | 0.01137 | 0.0001553 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | european_ki | gamma | -0.040584 | 0.0201 | -0.0430705 | -0.002487 | 0.0426 | 4.695e-05 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | stepdown | pv | 98.2196 | 0.0175 | 98.2083 | -0.01133 | 0.04632 | 0.005256 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | stepdown | delta | 0.44371 | 0.00639 | 0.440731 | -0.002978 | 0.01576 | 0.000327 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | stepdown | gamma | -0.0722007 | 0.0153 | -0.0434662 | 0.02873 | 0.05926 | 0.00111 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | stepdown_near_last_pre_ko | pv | 96.7042 | 0.0104 | 96.6911 | -0.01308 | 0.03378 | 0.007577 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | stepdown_near_last_pre_ko | delta | 0.574327 | 0.00784 | 0.570205 | -0.004122 | 0.0198 | 0.0009087 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | stepdown_near_last_pre_ko | gamma | -0.058302 | 0.00291 | -0.0429138 | 0.01539 | 0.02122 | 0.001548 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | parachute | pv | 97.4197 | 0.00947 | 97.4331 | 0.01336 | 0.03231 | 0.003584 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | parachute | delta | 0.550792 | 0.0047 | 0.543081 | -0.007711 | 0.01712 | 6.439e-05 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | parachute | gamma | -0.043298 | 0.0274 | -0.0502425 | -0.006945 | 0.06179 | 7.992e-05 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | parachute_near_ki | pv | 79.0793 | 0.00666 | 79.0885 | 0.009219 | 0.02255 | 0.004397 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | parachute_near_ki | delta | 1.17078 | 0.00552 | 1.16394 | -0.006843 | 0.01789 | 0.0005494 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | parachute_near_ki | gamma | 0.0247501 | 0.0271 | 0.0229607 | -0.001789 | 0.05604 | 0.0001302 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | ki_stepdown | pv | 97.7019 | 0.0132 | 97.7121 | 0.01014 | 0.03647 | 0.0004653 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | ki_stepdown | delta | 0.505667 | 0.00362 | 0.50546 | -0.0002073 | 0.007443 | 0.0002159 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | ki_stepdown | gamma | -0.0474391 | 0.0264 | -0.0479054 | -0.0004662 | 0.05327 | 6.57e-05 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | disable_ko_after_ki | pv | 95.3788 | 0.0101 | 95.3759 | -0.002881 | 0.02308 | 0.002128 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | disable_ko_after_ki | delta | 0.786006 | 0.00814 | 0.781163 | -0.004842 | 0.02111 | 1.681e-05 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | disable_ko_after_ki | gamma | -0.0704925 | 0.0233 | -0.0570776 | 0.01341 | 0.05998 | 3.37e-05 | PASS |

## Aggregate bias

| candidate | quantity | cells | mean bias (c) | SE (c) | passed |
|---|---|---|---|---|---|
| equity.ko_reset_snowball.pde | pv | 14 | 0.002669 | 0.00291 | yes |
| equity.ko_reset_snowball.pde | delta | 14 | -0.002895 | 0.00143 | yes |
| equity.ko_reset_snowball.pde | gamma | 14 | 0.001421 | 0.0053 | yes |
| equity.ko_reset_snowball.quad | pv | 14 | 0.003137 | 0.00291 | yes |
| equity.ko_reset_snowball.quad | delta | 14 | -0.002972 | 0.00143 | yes |
| equity.ko_reset_snowball.quad | gamma | 14 | 0.002031 | 0.0053 | yes |
| equity.ko_reset_snowball.quad.stretch_0.02 | pv | 14 | 0.002994 | 0.00291 | yes |
| equity.ko_reset_snowball.quad.stretch_0.02 | delta | 14 | -0.002957 | 0.00143 | yes |
| equity.ko_reset_snowball.quad.stretch_0.02 | gamma | 14 | 0.001964 | 0.0053 | yes |
