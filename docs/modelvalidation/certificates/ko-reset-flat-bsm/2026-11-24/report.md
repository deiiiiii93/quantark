# Certification report: ko-reset-flat-bsm

Evidence digest: `1625d586211cdf3d93c25edb6d0fb0a5c1e31f621b0aff2bc0149b5b2a327b30`

Machine: `arm64` / macOS-27.2.1-arm64-arm-64bit - Python 3.11.8, NumPy 2.4.6, quantark `e17e458f3e370634c7d63e66fd2931feece7e792`

## Amendment

| field | value |
|---|---|
| parent | docs/modelvalidation/certificates/ko-reset-flat-bsm/2026-11-15/certificate.json |
| parent digest | `fbfea7eb48716cc14004416302bc63c513f7b66587e36c46a40cd55259f23f75` |
| reason | QUAD sub-resolution run (owner-signed 2026-11-23): QUAD V1 prices the valuation-to-first-event step exactly at the spot (Gaussian quadrature on the pre-event surfaces, exact barrier limits) whenever it spans fewer than 13 cells or is unresolved, instead of the lattice readout, whose error there reached 1.1e-3 (1.1% just above the old refusal threshold); QUAD anchors move (pv <= 3.7e-5 rel, gamma 2-12% toward a 16x finer lattice in 30/36 checked cells). Engine code change: QUAD candidates re-run against the carried benchmark. |
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
| equity.ko_reset_snowball.quad | ordinary | pv | 96.6613 | 0.011 | 96.685 | 0.0237 | 0.04572 | 0.03169 | PASS |
| equity.ko_reset_snowball.quad | ordinary | delta | 0.621389 | 0.00868 | 0.611136 | -0.01025 | 0.02761 | 0.002896 | PASS |
| equity.ko_reset_snowball.quad | ordinary | gamma | -0.0416238 | 0.0297 | -0.0465304 | -0.004907 | 0.06431 | 0.0001093 | PASS |
| equity.ko_reset_snowball.quad | near_pre_ko | pv | 98.0475 | 0.0132 | 98.0634 | 0.01593 | 0.04237 | 0.02425 | PASS |
| equity.ko_reset_snowball.quad | near_pre_ko | delta | 0.487059 | 0.00233 | 0.489663 | 0.002604 | 0.007273 | 0.002971 | PASS |
| equity.ko_reset_snowball.quad | near_pre_ko | gamma | -0.034448 | 0.0126 | -0.0500379 | -0.01559 | 0.04075 | 4.963e-05 | PASS |
| equity.ko_reset_snowball.quad | near_ki | pv | 79.0154 | 0.00618 | 79.036 | 0.02057 | 0.03292 | 0.08599 | PASS |
| equity.ko_reset_snowball.quad | near_ki | delta | 1.06603 | 0.00536 | 1.05946 | -0.006577 | 0.01729 | 0.0004789 | PASS |
| equity.ko_reset_snowball.quad | near_ki | gamma | -0.0382894 | 0.0268 | -0.0373736 | 0.0009158 | 0.05456 | 0.002088 | PASS |
| equity.ko_reset_snowball.quad | below_ki | pv | 74.0373 | 0.0119 | 74.0453 | 0.007992 | 0.03189 | 0.06978 | PASS |
| equity.ko_reset_snowball.quad | below_ki | delta | 1.10845 | 0.00477 | 1.10683 | -0.001613 | 0.01114 | 0.004149 | PASS |
| equity.ko_reset_snowball.quad | below_ki | gamma | -0.0131948 | 0.0123 | 0.0054461 | 0.01864 | 0.04328 | 3.746e-05 | PASS |
| equity.ko_reset_snowball.quad | low_vol | pv | 100.165 | 0.0116 | 100.169 | 0.003796 | 0.02702 | 0.01547 | PASS |
| equity.ko_reset_snowball.quad | low_vol | delta | 0.359471 | 0.00292 | 0.368507 | 0.009036 | 0.01488 | 0.001636 | PASS |
| equity.ko_reset_snowball.quad | low_vol | gamma | -0.0863362 | 0.0118 | -0.0952239 | -0.008888 | 0.0325 | 0.0003742 | PASS |
| equity.ko_reset_snowball.quad | near_expiry | pv | 99.5678 | 0.00505 | 99.57 | 0.00224 | 0.01235 | 0.002467 | PASS |
| equity.ko_reset_snowball.quad | near_expiry | delta | 0.284584 | 0.00117 | 0.285801 | 0.001217 | 0.003552 | 0.0004398 | PASS |
| equity.ko_reset_snowball.quad | near_expiry | gamma | -0.0472864 | 0.00329 | -0.0448217 | 0.002465 | 0.009045 | 0.0006695 | PASS |
| equity.ko_reset_snowball.quad | discrete_ki | pv | 96.7295 | 0.00999 | 96.7357 | 0.006261 | 0.02625 | 0 | PASS |
| equity.ko_reset_snowball.quad | discrete_ki | delta | 0.612278 | 0.00409 | 0.606421 | -0.005858 | 0.01403 | 0 | PASS |
| equity.ko_reset_snowball.quad | discrete_ki | gamma | -0.0366268 | 0.00883 | -0.0467716 | -0.01014 | 0.02781 | 0 | PASS |
| equity.ko_reset_snowball.quad | european_ki | pv | 97.6357 | 0.00975 | 97.6571 | 0.02134 | 0.04084 | 0.01934 | PASS |
| equity.ko_reset_snowball.quad | european_ki | delta | 0.503157 | 0.00303 | 0.497318 | -0.00584 | 0.0119 | 0.00158 | PASS |
| equity.ko_reset_snowball.quad | european_ki | gamma | -0.040584 | 0.0201 | -0.0431204 | -0.002536 | 0.04265 | 0.0001203 | PASS |
| equity.ko_reset_snowball.quad | stepdown | pv | 98.2196 | 0.0175 | 98.2211 | 0.001542 | 0.03654 | 0.03232 | PASS |
| equity.ko_reset_snowball.quad | stepdown | delta | 0.44371 | 0.00639 | 0.439504 | -0.004205 | 0.01699 | 0.003392 | PASS |
| equity.ko_reset_snowball.quad | stepdown | gamma | -0.0722007 | 0.0153 | -0.0435117 | 0.02869 | 0.05922 | 0.0001961 | PASS |
| equity.ko_reset_snowball.quad | stepdown_near_last_pre_ko | pv | 96.7042 | 0.0104 | 96.7058 | 0.00164 | 0.02235 | 0.04423 | PASS |
| equity.ko_reset_snowball.quad | stepdown_near_last_pre_ko | delta | 0.574327 | 0.00784 | 0.56999 | -0.004338 | 0.02002 | 0.004384 | PASS |
| equity.ko_reset_snowball.quad | stepdown_near_last_pre_ko | gamma | -0.058302 | 0.00291 | -0.0414473 | 0.01685 | 0.02268 | 0.003387 | PASS |
| equity.ko_reset_snowball.quad | parachute | pv | 97.4197 | 0.00947 | 97.4425 | 0.02284 | 0.04178 | 0.01873 | PASS |
| equity.ko_reset_snowball.quad | parachute | delta | 0.550792 | 0.0047 | 0.542152 | -0.00864 | 0.01805 | 0.001828 | PASS |
| equity.ko_reset_snowball.quad | parachute | gamma | -0.043298 | 0.0274 | -0.0502495 | -0.006952 | 0.0618 | 1.849e-05 | PASS |
| equity.ko_reset_snowball.quad | parachute_near_ki | pv | 79.0793 | 0.00666 | 79.1007 | 0.0214 | 0.03473 | 0.08462 | PASS |
| equity.ko_reset_snowball.quad | parachute_near_ki | delta | 1.17078 | 0.00552 | 1.16404 | -0.006749 | 0.01779 | 0.001742 | PASS |
| equity.ko_reset_snowball.quad | parachute_near_ki | gamma | 0.0247501 | 0.0271 | 0.022684 | -0.002066 | 0.05631 | 0.00337 | PASS |
| equity.ko_reset_snowball.quad | ki_stepdown | pv | 97.7019 | 0.0132 | 97.7199 | 0.01795 | 0.04427 | 0.01901 | PASS |
| equity.ko_reset_snowball.quad | ki_stepdown | delta | 0.505667 | 0.00362 | 0.504759 | -0.0009081 | 0.008144 | 0.001738 | PASS |
| equity.ko_reset_snowball.quad | ki_stepdown | gamma | -0.0474391 | 0.0264 | -0.0479287 | -0.0004896 | 0.05329 | 5.652e-05 | PASS |
| equity.ko_reset_snowball.quad | disable_ko_after_ki | pv | 95.3788 | 0.0101 | 95.3835 | 0.00468 | 0.02488 | 0.0216 | PASS |
| equity.ko_reset_snowball.quad | disable_ko_after_ki | delta | 0.786006 | 0.00814 | 0.780711 | -0.005294 | 0.02156 | 0.001396 | PASS |
| equity.ko_reset_snowball.quad | disable_ko_after_ki | gamma | -0.0704925 | 0.0233 | -0.0571652 | 0.01333 | 0.05989 | 0.000242 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | ordinary | pv | 96.6613 | 0.011 | 96.685 | 0.0237 | 0.04572 | 0.03169 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | ordinary | delta | 0.621389 | 0.00868 | 0.611136 | -0.01025 | 0.02761 | 0.002896 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | ordinary | gamma | -0.0416238 | 0.0297 | -0.0465304 | -0.004907 | 0.06431 | 0.0001093 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | near_pre_ko | pv | 98.0475 | 0.0132 | 98.0634 | 0.01593 | 0.04237 | 0.02425 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | near_pre_ko | delta | 0.487059 | 0.00233 | 0.489663 | 0.002604 | 0.007273 | 0.002971 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | near_pre_ko | gamma | -0.034448 | 0.0126 | -0.0500379 | -0.01559 | 0.04075 | 4.963e-05 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | near_ki | pv | 79.0154 | 0.00618 | 79.036 | 0.02057 | 0.03292 | 0.08599 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | near_ki | delta | 1.06603 | 0.00536 | 1.05946 | -0.006577 | 0.01729 | 0.0004789 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | near_ki | gamma | -0.0382894 | 0.0268 | -0.0373736 | 0.0009158 | 0.05456 | 0.002088 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | below_ki | pv | 74.0373 | 0.0119 | 74.0453 | 0.007992 | 0.03189 | 0.06978 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | below_ki | delta | 1.10845 | 0.00477 | 1.10683 | -0.001613 | 0.01114 | 0.004149 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | below_ki | gamma | -0.0131948 | 0.0123 | 0.0054461 | 0.01864 | 0.04328 | 3.746e-05 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | low_vol | pv | 100.165 | 0.0116 | 100.169 | 0.00346 | 0.02668 | 0.01517 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | low_vol | delta | 0.359471 | 0.00292 | 0.368564 | 0.009093 | 0.01494 | 0.00158 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | low_vol | gamma | -0.0863362 | 0.0118 | -0.0952235 | -0.008887 | 0.0325 | 0.0003753 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | near_expiry | pv | 99.5678 | 0.00505 | 99.57 | 0.002235 | 0.01234 | 0.0008134 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | near_expiry | delta | 0.284584 | 0.00117 | 0.285809 | 0.001225 | 0.003561 | 3.361e-05 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | near_expiry | gamma | -0.0472864 | 0.00329 | -0.0457564 | 0.00153 | 0.00811 | 0.001571 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | discrete_ki | pv | 96.7295 | 0.00999 | 96.7331 | 0.003599 | 0.02359 | 0 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | discrete_ki | delta | 0.612278 | 0.00409 | 0.606722 | -0.005556 | 0.01373 | 0 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | discrete_ki | gamma | -0.0366268 | 0.00883 | -0.0467773 | -0.01015 | 0.02782 | 0 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | european_ki | pv | 97.6357 | 0.00975 | 97.6571 | 0.02134 | 0.04084 | 0.01934 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | european_ki | delta | 0.503157 | 0.00303 | 0.497318 | -0.00584 | 0.0119 | 0.00158 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | european_ki | gamma | -0.040584 | 0.0201 | -0.0431204 | -0.002536 | 0.04265 | 0.0001203 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | stepdown | pv | 98.2196 | 0.0175 | 98.2211 | 0.001543 | 0.03654 | 0.03236 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | stepdown | delta | 0.44371 | 0.00639 | 0.439502 | -0.004207 | 0.01699 | 0.003394 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | stepdown | gamma | -0.0722007 | 0.0153 | -0.0435088 | 0.02869 | 0.05922 | 0.0002229 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | stepdown_near_last_pre_ko | pv | 96.7042 | 0.0104 | 96.7058 | 0.00166 | 0.02237 | 0.04424 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | stepdown_near_last_pre_ko | delta | 0.574327 | 0.00784 | 0.569982 | -0.004345 | 0.02002 | 0.004371 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | stepdown_near_last_pre_ko | gamma | -0.058302 | 0.00291 | -0.0414581 | 0.01684 | 0.02267 | 0.003364 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | parachute | pv | 97.4197 | 0.00947 | 97.4425 | 0.02284 | 0.04178 | 0.01873 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | parachute | delta | 0.550792 | 0.0047 | 0.542152 | -0.00864 | 0.01805 | 0.001828 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | parachute | gamma | -0.043298 | 0.0274 | -0.0502495 | -0.006952 | 0.0618 | 1.849e-05 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | parachute_near_ki | pv | 79.0793 | 0.00666 | 79.1007 | 0.0214 | 0.03473 | 0.08462 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | parachute_near_ki | delta | 1.17078 | 0.00552 | 1.16404 | -0.006749 | 0.01779 | 0.001742 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | parachute_near_ki | gamma | 0.0247501 | 0.0271 | 0.022684 | -0.002066 | 0.05631 | 0.00337 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | ki_stepdown | pv | 97.7019 | 0.0132 | 97.7198 | 0.01786 | 0.04418 | 0.01968 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | ki_stepdown | delta | 0.505667 | 0.00362 | 0.504764 | -0.000903 | 0.008139 | 0.001814 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | ki_stepdown | gamma | -0.0474391 | 0.0264 | -0.0479272 | -0.0004881 | 0.05329 | 5.444e-05 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | disable_ko_after_ki | pv | 95.3788 | 0.0101 | 95.3835 | 0.00468 | 0.02488 | 0.0216 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | disable_ko_after_ki | delta | 0.786006 | 0.00814 | 0.780711 | -0.005294 | 0.02156 | 0.001396 | PASS |
| equity.ko_reset_snowball.quad.stretch_0.02 | disable_ko_after_ki | gamma | -0.0704925 | 0.0233 | -0.0571652 | 0.01333 | 0.05989 | 0.000242 | PASS |

## Aggregate bias

| candidate | quantity | cells | mean bias (c) | SE (c) | passed |
|---|---|---|---|---|---|
| equity.ko_reset_snowball.pde | pv | 14 | 0.002669 | 0.00291 | yes |
| equity.ko_reset_snowball.pde | delta | 14 | -0.002895 | 0.00143 | yes |
| equity.ko_reset_snowball.pde | gamma | 14 | 0.001421 | 0.0053 | yes |
| equity.ko_reset_snowball.quad | pv | 14 | 0.01228 | 0.00291 | yes |
| equity.ko_reset_snowball.quad | delta | 14 | -0.003387 | 0.00143 | yes |
| equity.ko_reset_snowball.quad | gamma | 14 | 0.002094 | 0.0053 | yes |
| equity.ko_reset_snowball.quad.stretch_0.02 | pv | 14 | 0.01206 | 0.00291 | yes |
| equity.ko_reset_snowball.quad.stretch_0.02 | delta | 14 | -0.003361 | 0.00143 | yes |
| equity.ko_reset_snowball.quad.stretch_0.02 | gamma | 14 | 0.002027 | 0.0053 | yes |
