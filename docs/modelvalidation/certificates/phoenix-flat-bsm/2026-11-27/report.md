# Certification report: phoenix-flat-bsm

Evidence digest: `a8210bf716ebd7138e196c75234156ab7b664b34b19570374aae162832808923`

Machine: `arm64` / macOS-27.2.1-arm64-arm-64bit - Python 3.11.8, NumPy 2.4.6, quantark `fb464de185c49445642d3cf934e1adec4bca6dfc`

## Amendment

| field | value |
|---|---|
| parent | docs/modelvalidation/certificates/phoenix-flat-bsm/2026-11-24/certificate.json |
| parent digest | `b6a83c52a9fee7b5690817ed03c593e06f90794b1cdfec40899b6c168d4faa69` |
| reason | 0.16.0 run 5 T5 (owner-signed 2026-11-27, 'approved'): QUAD V1 linear convolutions run on the smallest even 5-smooth FFT length at or above the legacy length (real FFTs; the adjoint transposes on the same length). Round-off moves only (anchors <= 1.5e-11 rel, prices ~1e-15 rel). QUAD candidates re-run against the carried benchmarks; other candidates unchanged. |
| re-priced | 84 cell(s) |
| carried forward | 42 cell(s) |

## Decisions

| candidate | decision |
|---|---|
| equity.phoenix.pde | ADMITTED |
| equity.phoenix.quad | ADMITTED |
| equity.phoenix.quad.stretch_0.02 | ADMITTED |

Bounds: cell 0.5 c, mean signed bias 0.1 c, standard-error budget 0.25 x cell, interval k 2.

## Engine configuration

Resolved rather than named: a profile such as `standard` is an indirection whose meaning can change between releases. These are the requested settings.

| engine | setting | value |
|---|---|---|
| equity.phoenix.pde | accuracy | standard |
| equity.phoenix.pde | engine | PhoenixPDESolver |
| equity.phoenix.pde | grid.bounds | (None, None) |
| equity.phoenix.pde | grid.day_count | 252 |
| equity.phoenix.pde | grid.discrete_monitoring_damping | ['half_step', 1, 1] |
| equity.phoenix.pde | grid.eps_crit | 0.003 |
| equity.phoenix.pde | grid.event_damping_steps | 2 |
| equity.phoenix.pde | grid.max_points | 2000 |
| equity.phoenix.pde | grid.max_steps | 12000 |
| equity.phoenix.pde | grid.num_std | 4 |
| equity.phoenix.pde | grid.points | 400 |
| equity.phoenix.pde | grid.steps_per_day | 4 |
| equity.phoenix.pde | grid.terminal_damping_steps | 1 |
| equity.phoenix.quad | engine | PhoenixQuadEngine |
| equity.phoenix.quad | grid.align_cell_stretch | -- |
| equity.phoenix.quad | grid.align_priority | auto |
| equity.phoenix.quad | grid.barrier_reach_stddevs | -- |
| equity.phoenix.quad | grid.bgk_min_ki_observations | 100 |
| equity.phoenix.quad | grid.bus_days_in_year | 252 |
| equity.phoenix.quad | grid.continuous_steps_per_year | 0 |
| equity.phoenix.quad | grid.continuous_term_structure | piecewise_constant |
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
| equity.phoenix.quad | grid.readout | legacy_linear |
| equity.phoenix.quad | grid.stability_preset | -- |
| equity.phoenix.quad | grid_points | 1001 |
| equity.phoenix.quad.stretch_0.02 | align_cell_stretch | 0.02 |
| equity.phoenix.quad.stretch_0.02 | engine | PhoenixQuadEngine |
| equity.phoenix.quad.stretch_0.02 | grid.align_cell_stretch | 0.02 |
| equity.phoenix.quad.stretch_0.02 | grid.align_priority | auto |
| equity.phoenix.quad.stretch_0.02 | grid.barrier_reach_stddevs | -- |
| equity.phoenix.quad.stretch_0.02 | grid.bgk_min_ki_observations | 100 |
| equity.phoenix.quad.stretch_0.02 | grid.bus_days_in_year | 252 |
| equity.phoenix.quad.stretch_0.02 | grid.continuous_steps_per_year | 0 |
| equity.phoenix.quad.stretch_0.02 | grid.continuous_term_structure | piecewise_constant |
| equity.phoenix.quad.stretch_0.02 | grid.event_projection | cell_average |
| equity.phoenix.quad.stretch_0.02 | grid.event_smoothing_cells | 1 |
| equity.phoenix.quad.stretch_0.02 | grid.event_smoothing_kernel | cosine |
| equity.phoenix.quad.stretch_0.02 | grid.event_smoothing_log_width | 0.002 |
| equity.phoenix.quad.stretch_0.02 | grid.event_smoothing_mode | fixed |
| equity.phoenix.quad.stretch_0.02 | grid.fft_filter_alpha | 12 |
| equity.phoenix.quad.stretch_0.02 | grid.fft_filter_power | 8 |
| equity.phoenix.quad.stretch_0.02 | grid.fft_padding_factor | 2 |
| equity.phoenix.quad.stretch_0.02 | grid.filter_unreachable_barriers | True |
| equity.phoenix.quad.stretch_0.02 | grid.grid_points | 1001 |
| equity.phoenix.quad.stretch_0.02 | grid.integration_rule | trapezoid |
| equity.phoenix.quad.stretch_0.02 | grid.ki_monitoring_mode | exact_discrete |
| equity.phoenix.quad.stretch_0.02 | grid.max_adaptive_grid_points | 5001 |
| equity.phoenix.quad.stretch_0.02 | grid.min_diffusion_stddev_cells | 2.5 |
| equity.phoenix.quad.stretch_0.02 | grid.num_std_devs | 10 |
| equity.phoenix.quad.stretch_0.02 | grid.readout | legacy_linear |
| equity.phoenix.quad.stretch_0.02 | grid.stability_preset | -- |
| equity.phoenix.quad.stretch_0.02 | grid_points | 1001 |
| (benchmark) | engine | PhoenixMCEngine |
| (benchmark) | greeks | paired central difference (common random numbers) |
| (benchmark) | method | randomized_quasi |
| (benchmark) | paths_per_batch | 65536 |

## Benchmark sampling

| case | batches | stopped because | standard errors (raw) |
|---|---|---|---|
| coupon_at_expiry | 4 | se_budget_met | delta: 0.000844, gamma: 0.00451, pv: 0.00425 |
| coupon_at_expiry_memory | 4 | se_budget_met | delta: 0.000904, gamma: 0.0044, pv: 0.00453 |
| disable_ko_after_ki | 4 | se_budget_met | delta: 0.00105, gamma: 0.00415, pv: 0.00402 |
| discrete_ki | 4 | se_budget_met | delta: 0.00246, gamma: 0.00682, pv: 0.00314 |
| ki_stepdown | 4 | se_budget_met | delta: 0.00141, gamma: 0.00623, pv: 0.00407 |
| low_vol | 4 | se_budget_met | delta: 0.00269, gamma: 0.00211, pv: 0.00105 |
| memory | 4 | se_budget_met | delta: 0.000903, gamma: 0.0044, pv: 0.00452 |
| near_coupon | 4 | se_budget_met | delta: 0.00425, gamma: 0.0216, pv: 0.00517 |
| near_expiry | 4 | se_budget_met | delta: 0.000659, gamma: 0.00317, pv: 0.00199 |
| near_ki | 4 | se_budget_met | delta: 0.00424, gamma: 0.00491, pv: 0.00146 |
| near_ko | 4 | se_budget_met | delta: 0.00292, gamma: 0.00687, pv: 0.00358 |
| ordinary | 4 | se_budget_met | delta: 0.000842, gamma: 0.00451, pv: 0.00424 |
| reverse | 4 | se_budget_met | delta: 0.0042, gamma: 0.00925, pv: 0.0111 |
| stepdown | 4 | se_budget_met | delta: 0.00108, gamma: 0.0103, pv: 0.00389 |

Sampling policy: 65536 paths/batch, 4-32 batches, seed 20260818, bump 0.01.

## Cells

| candidate | case | quantity | reference | SE | candidate | err (c) | interval (c) | envelope (c) | verdict |
|---|---|---|---|---|---|---|---|---|---|
| equity.phoenix.pde | ordinary | pv | -3.42614 | 0.00424 | -3.42484 | 0.001297 | 0.009786 | 0.0006088 | PASS |
| equity.phoenix.pde | ordinary | delta | 0.488242 | 0.000842 | 0.487884 | -0.0003586 | 0.002043 | 0.0001499 | PASS |
| equity.phoenix.pde | ordinary | gamma | -0.0432388 | 0.00451 | -0.0353616 | 0.007877 | 0.01689 | 1.223e-06 | PASS |
| equity.phoenix.pde | near_ko | pv | -2.31343 | 0.00358 | -2.31872 | -0.005292 | 0.01246 | 0.0004943 | PASS |
| equity.phoenix.pde | near_ko | delta | 0.400328 | 0.00292 | 0.395742 | -0.004587 | 0.01043 | 1.464e-05 | PASS |
| equity.phoenix.pde | near_ko | gamma | -0.0493471 | 0.00687 | -0.0380321 | 0.01131 | 0.02506 | 5.992e-06 | PASS |
| equity.phoenix.pde | near_coupon | pv | -13.8608 | 0.00517 | -13.8715 | -0.01062 | 0.02097 | 0.003183 | PASS |
| equity.phoenix.pde | near_coupon | delta | 0.948728 | 0.00425 | 0.947558 | -0.00117 | 0.009667 | 8.764e-05 | PASS |
| equity.phoenix.pde | near_coupon | gamma | -0.0437926 | 0.0216 | -0.0296067 | 0.01419 | 0.05735 | 3.344e-05 | PASS |
| equity.phoenix.pde | near_ki | pv | -23.9184 | 0.00146 | -23.9199 | -0.001497 | 0.004421 | 0.002681 | PASS |
| equity.phoenix.pde | near_ki | delta | 1.13636 | 0.00424 | 1.12755 | -0.008814 | 0.01728 | 0.0005062 | PASS |
| equity.phoenix.pde | near_ki | gamma | -0.00717388 | 0.00491 | 0.0163062 | 0.02348 | 0.0333 | 0.0006064 | PASS |
| equity.phoenix.pde | low_vol | pv | 0.526131 | 0.00105 | 0.52683 | 0.0006989 | 0.002792 | 0.00272 | PASS |
| equity.phoenix.pde | low_vol | delta | 0.00547519 | 0.00269 | 0.0046174 | -0.0008578 | 0.006244 | 0.0006747 | PASS |
| equity.phoenix.pde | low_vol | gamma | -0.0266412 | 0.00211 | -0.0327922 | -0.006151 | 0.01036 | 0.0001202 | PASS |
| equity.phoenix.pde | near_expiry | pv | 0.0923547 | 0.00199 | 0.0910755 | -0.001279 | 0.00525 | 0.0008889 | PASS |
| equity.phoenix.pde | near_expiry | delta | 0.0556837 | 0.000659 | 0.0547747 | -0.000909 | 0.002227 | 0.0001831 | PASS |
| equity.phoenix.pde | near_expiry | gamma | -0.0205876 | 0.00317 | -0.0181782 | 0.002409 | 0.008744 | 2.816e-05 | PASS |
| equity.phoenix.pde | memory | pv | -3.34946 | 0.00452 | -3.34829 | 0.001174 | 0.01022 | 0.0006083 | PASS |
| equity.phoenix.pde | memory | delta | 0.477563 | 0.000903 | 0.477191 | -0.0003726 | 0.002179 | 0.0001481 | PASS |
| equity.phoenix.pde | memory | gamma | -0.0428127 | 0.0044 | -0.0344727 | 0.00834 | 0.01715 | 1.371e-06 | PASS |
| equity.phoenix.pde | stepdown | pv | -3.26049 | 0.00389 | -3.25703 | 0.003461 | 0.01125 | 0.001951 | PASS |
| equity.phoenix.pde | stepdown | delta | 0.470443 | 0.00108 | 0.472843 | 0.0024 | 0.004556 | 1.843e-05 | PASS |
| equity.phoenix.pde | stepdown | gamma | -0.0345018 | 0.0103 | -0.0359934 | -0.001492 | 0.02216 | 6.143e-06 | PASS |
| equity.phoenix.pde | discrete_ki | pv | -2.90513 | 0.00314 | -2.89927 | 0.005853 | 0.01214 | 6.014e-05 | PASS |
| equity.phoenix.pde | discrete_ki | delta | 0.422553 | 0.00246 | 0.427055 | 0.004502 | 0.009422 | 6.039e-05 | PASS |
| equity.phoenix.pde | discrete_ki | gamma | -0.0337775 | 0.00682 | -0.0338167 | -3.917e-05 | 0.01368 | 1.742e-06 | PASS |
| equity.phoenix.pde | ki_stepdown | pv | -1.90855 | 0.00407 | -1.90772 | 0.0008356 | 0.008976 | 0.0006134 | PASS |
| equity.phoenix.pde | ki_stepdown | delta | 0.309466 | 0.00141 | 0.312939 | 0.003473 | 0.006295 | 0.0001509 | PASS |
| equity.phoenix.pde | ki_stepdown | gamma | -0.040774 | 0.00623 | -0.0312053 | 0.009569 | 0.02203 | 1.226e-05 | PASS |
| equity.phoenix.pde | disable_ko_after_ki | pv | -3.42908 | 0.00402 | -3.42754 | 0.001532 | 0.009576 | 0.0006189 | PASS |
| equity.phoenix.pde | disable_ko_after_ki | delta | 0.489054 | 0.00105 | 0.488377 | -0.000677 | 0.002785 | 0.0001517 | PASS |
| equity.phoenix.pde | disable_ko_after_ki | gamma | -0.0427926 | 0.00415 | -0.0354387 | 0.007354 | 0.01565 | 1.45e-06 | PASS |
| equity.phoenix.pde | reverse | pv | -4.00024 | 0.0111 | -4.00604 | -0.005806 | 0.02803 | 0.002822 | PASS |
| equity.phoenix.pde | reverse | delta | -0.593336 | 0.0042 | -0.591655 | 0.00168 | 0.01009 | 0.0002602 | PASS |
| equity.phoenix.pde | reverse | gamma | -0.0290847 | 0.00925 | -0.0352051 | -0.00612 | 0.02462 | 2.489e-06 | PASS |
| equity.phoenix.pde | coupon_at_expiry | pv | -3.43198 | 0.00425 | -3.43068 | 0.0013 | 0.009804 | 0.0006079 | PASS |
| equity.phoenix.pde | coupon_at_expiry | delta | 0.488815 | 0.000844 | 0.488463 | -0.0003515 | 0.00204 | 0.0001499 | PASS |
| equity.phoenix.pde | coupon_at_expiry | gamma | -0.0432232 | 0.00451 | -0.0353475 | 0.007876 | 0.01689 | 1.229e-06 | PASS |
| equity.phoenix.pde | coupon_at_expiry_memory | pv | -3.35582 | 0.00453 | -3.35464 | 0.001179 | 0.01024 | 0.0006074 | PASS |
| equity.phoenix.pde | coupon_at_expiry_memory | delta | 0.478215 | 0.000904 | 0.47785 | -0.0003649 | 0.002174 | 0.0001482 | PASS |
| equity.phoenix.pde | coupon_at_expiry_memory | gamma | -0.042803 | 0.0044 | -0.0344673 | 0.008336 | 0.01714 | 1.375e-06 | PASS |
| equity.phoenix.quad | ordinary | pv | -3.42614 | 0.00424 | -3.42454 | 0.001596 | 0.01008 | 0.0005154 | PASS |
| equity.phoenix.quad | ordinary | delta | 0.488242 | 0.000842 | 0.487609 | -0.0006334 | 0.002317 | 2.434e-05 | PASS |
| equity.phoenix.quad | ordinary | gamma | -0.0432388 | 0.00451 | -0.0353513 | 0.007888 | 0.0169 | 4.212e-06 | PASS |
| equity.phoenix.quad | near_ko | pv | -2.31343 | 0.00358 | -2.3187 | -0.005265 | 0.01243 | 0.0004417 | PASS |
| equity.phoenix.quad | near_ko | delta | 0.400328 | 0.00292 | 0.395609 | -0.004719 | 0.01056 | 3.416e-05 | PASS |
| equity.phoenix.quad | near_ko | gamma | -0.0493471 | 0.00687 | -0.0379837 | 0.01136 | 0.02511 | 3.344e-06 | PASS |
| equity.phoenix.quad | near_coupon | pv | -13.8608 | 0.00517 | -13.8659 | -0.005042 | 0.01538 | 0.00131 | PASS |
| equity.phoenix.quad | near_coupon | delta | 0.948728 | 0.00425 | 0.947255 | -0.001474 | 0.009971 | 7.268e-05 | PASS |
| equity.phoenix.quad | near_coupon | gamma | -0.0437926 | 0.0216 | -0.0296125 | 0.01418 | 0.05734 | 1.791e-05 | PASS |
| equity.phoenix.quad | near_ki | pv | -23.9184 | 0.00146 | -23.9187 | -0.0003245 | 0.003249 | 0.001052 | PASS |
| equity.phoenix.quad | near_ki | delta | 1.13636 | 0.00424 | 1.13178 | -0.004582 | 0.01305 | 0.0004565 | PASS |
| equity.phoenix.quad | near_ki | gamma | -0.00717388 | 0.00491 | -0.00677687 | 0.000397 | 0.01022 | 5.397e-06 | PASS |
| equity.phoenix.quad | low_vol | pv | 0.526131 | 0.00105 | 0.528104 | 0.001973 | 0.004067 | 0.002066 | PASS |
| equity.phoenix.quad | low_vol | delta | 0.00547519 | 0.00269 | 0.00572835 | 0.0002532 | 0.005639 | 0.0003748 | PASS |
| equity.phoenix.quad | low_vol | gamma | -0.0266412 | 0.00211 | -0.0326412 | -0.006 | 0.01021 | 1.081e-05 | PASS |
| equity.phoenix.quad | near_expiry | pv | 0.0923547 | 0.00199 | 0.0912468 | -0.001108 | 0.005079 | 0.0003217 | PASS |
| equity.phoenix.quad | near_expiry | delta | 0.0556837 | 0.000659 | 0.0554577 | -0.000226 | 0.001544 | 0.0001014 | PASS |
| equity.phoenix.quad | near_expiry | gamma | -0.0205876 | 0.00317 | -0.0183734 | 0.002214 | 0.008549 | 0.0001735 | PASS |
| equity.phoenix.quad | memory | pv | -3.34946 | 0.00452 | -3.34798 | 0.001483 | 0.01053 | 0.0004399 | PASS |
| equity.phoenix.quad | memory | delta | 0.477563 | 0.000903 | 0.476913 | -0.0006503 | 0.002457 | 1.764e-05 | PASS |
| equity.phoenix.quad | memory | gamma | -0.0428127 | 0.0044 | -0.0344619 | 0.008351 | 0.01716 | 3.736e-06 | PASS |
| equity.phoenix.quad | stepdown | pv | -3.26049 | 0.00389 | -3.25636 | 0.004133 | 0.01192 | 0.001615 | PASS |
| equity.phoenix.quad | stepdown | delta | 0.470443 | 0.00108 | 0.472692 | 0.002249 | 0.004405 | 0.0007199 | PASS |
| equity.phoenix.quad | stepdown | gamma | -0.0345018 | 0.0103 | -0.0369341 | -0.002432 | 0.0231 | 0.0004456 | PASS |
| equity.phoenix.quad | discrete_ki | pv | -2.90513 | 0.00314 | -2.89981 | 0.00532 | 0.01161 | 0.0001747 | PASS |
| equity.phoenix.quad | discrete_ki | delta | 0.422553 | 0.00246 | 0.427038 | 0.004485 | 0.009405 | 2.332e-05 | PASS |
| equity.phoenix.quad | discrete_ki | gamma | -0.0337775 | 0.00682 | -0.0338144 | -3.692e-05 | 0.01368 | 9.115e-06 | PASS |
| equity.phoenix.quad | ki_stepdown | pv | -1.90855 | 0.00407 | -1.90802 | 0.000536 | 0.008676 | 0.001525 | PASS |
| equity.phoenix.quad | ki_stepdown | delta | 0.309466 | 0.00141 | 0.313173 | 0.003708 | 0.00653 | 0.0001636 | PASS |
| equity.phoenix.quad | ki_stepdown | gamma | -0.040774 | 0.00623 | -0.0312137 | 0.00956 | 0.02202 | 2.872e-06 | PASS |
| equity.phoenix.quad | disable_ko_after_ki | pv | -3.42908 | 0.00402 | -3.42724 | 0.00184 | 0.009883 | 0.0005354 | PASS |
| equity.phoenix.quad | disable_ko_after_ki | delta | 0.489054 | 0.00105 | 0.488103 | -0.0009514 | 0.00306 | 2.771e-05 | PASS |
| equity.phoenix.quad | disable_ko_after_ki | gamma | -0.0427926 | 0.00415 | -0.0354283 | 0.007364 | 0.01566 | 3.762e-06 | PASS |
| equity.phoenix.quad | reverse | pv | -4.00024 | 0.0111 | -4.00341 | -0.003175 | 0.0254 | 0.001319 | PASS |
| equity.phoenix.quad | reverse | delta | -0.593336 | 0.0042 | -0.590711 | 0.002625 | 0.01103 | 9.812e-05 | PASS |
| equity.phoenix.quad | reverse | gamma | -0.0290847 | 0.00925 | -0.0352227 | -0.006138 | 0.02464 | 9.603e-06 | PASS |
| equity.phoenix.quad | coupon_at_expiry | pv | -3.43198 | 0.00425 | -3.43038 | 0.001601 | 0.01011 | 0.0005061 | PASS |
| equity.phoenix.quad | coupon_at_expiry | delta | 0.488815 | 0.000844 | 0.488186 | -0.000629 | 0.002318 | 2.348e-05 | PASS |
| equity.phoenix.quad | coupon_at_expiry | gamma | -0.0432232 | 0.00451 | -0.0353372 | 0.007886 | 0.0169 | 4.157e-06 | PASS |
| equity.phoenix.quad | coupon_at_expiry_memory | pv | -3.35582 | 0.00453 | -3.35433 | 0.001491 | 0.01055 | 0.0004312 | PASS |
| equity.phoenix.quad | coupon_at_expiry_memory | delta | 0.478215 | 0.000904 | 0.47757 | -0.0006452 | 0.002454 | 1.684e-05 | PASS |
| equity.phoenix.quad | coupon_at_expiry_memory | gamma | -0.042803 | 0.0044 | -0.0344566 | 0.008346 | 0.01716 | 3.688e-06 | PASS |
| equity.phoenix.quad.stretch_0.02 | ordinary | pv | -3.42614 | 0.00424 | -3.42492 | 0.001221 | 0.00971 | 0.0001407 | PASS |
| equity.phoenix.quad.stretch_0.02 | ordinary | delta | 0.488242 | 0.000842 | 0.487647 | -0.0005953 | 0.002279 | 1.38e-05 | PASS |
| equity.phoenix.quad.stretch_0.02 | ordinary | gamma | -0.0432388 | 0.00451 | -0.0353508 | 0.007888 | 0.0169 | 3.69e-06 | PASS |
| equity.phoenix.quad.stretch_0.02 | near_ko | pv | -2.31343 | 0.00358 | -2.31898 | -0.005544 | 0.01271 | 0.0001622 | PASS |
| equity.phoenix.quad.stretch_0.02 | near_ko | delta | 0.400328 | 0.00292 | 0.395646 | -0.004682 | 0.01052 | 2.835e-06 | PASS |
| equity.phoenix.quad.stretch_0.02 | near_ko | gamma | -0.0493471 | 0.00687 | -0.037985 | 0.01136 | 0.02511 | 4.725e-06 | PASS |
| equity.phoenix.quad.stretch_0.02 | near_coupon | pv | -13.8608 | 0.00517 | -13.8661 | -0.00522 | 0.01556 | 0.001132 | PASS |
| equity.phoenix.quad.stretch_0.02 | near_coupon | delta | 0.948728 | 0.00425 | 0.947239 | -0.001489 | 0.009987 | 5.718e-05 | PASS |
| equity.phoenix.quad.stretch_0.02 | near_coupon | gamma | -0.0437926 | 0.0216 | -0.0296122 | 0.01418 | 0.05734 | 1.761e-05 | PASS |
| equity.phoenix.quad.stretch_0.02 | near_ki | pv | -23.9184 | 0.00146 | -23.9187 | -0.0002514 | 0.003176 | 0.001125 | PASS |
| equity.phoenix.quad.stretch_0.02 | near_ki | delta | 1.13636 | 0.00424 | 1.13177 | -0.004585 | 0.01306 | 0.0004529 | PASS |
| equity.phoenix.quad.stretch_0.02 | near_ki | gamma | -0.00717388 | 0.00491 | -0.00677643 | 0.0003974 | 0.01022 | 5.835e-06 | PASS |
| equity.phoenix.quad.stretch_0.02 | low_vol | pv | 0.526131 | 0.00105 | 0.527794 | 0.001663 | 0.003756 | 0.001755 | PASS |
| equity.phoenix.quad.stretch_0.02 | low_vol | delta | 0.00547519 | 0.00269 | 0.00579281 | 0.0003176 | 0.005703 | 0.0003104 | PASS |
| equity.phoenix.quad.stretch_0.02 | low_vol | gamma | -0.0266412 | 0.00211 | -0.0326471 | -0.006006 | 0.01022 | 5.002e-06 | PASS |
| equity.phoenix.quad.stretch_0.02 | near_expiry | pv | 0.0923547 | 0.00199 | 0.0912468 | -0.001108 | 0.005079 | 0.000302 | PASS |
| equity.phoenix.quad.stretch_0.02 | near_expiry | delta | 0.0556837 | 0.000659 | 0.0554577 | -0.000226 | 0.001544 | 2.341e-05 | PASS |
| equity.phoenix.quad.stretch_0.02 | near_expiry | gamma | -0.0205876 | 0.00317 | -0.0183734 | 0.002214 | 0.008549 | 0.0001541 | PASS |
| equity.phoenix.quad.stretch_0.02 | memory | pv | -3.34946 | 0.00452 | -3.34837 | 0.001086 | 0.01013 | 4.232e-05 | PASS |
| equity.phoenix.quad.stretch_0.02 | memory | delta | 0.477563 | 0.000903 | 0.476954 | -0.0006095 | 0.002416 | 2.313e-05 | PASS |
| equity.phoenix.quad.stretch_0.02 | memory | gamma | -0.0428127 | 0.0044 | -0.0344615 | 0.008351 | 0.01716 | 3.258e-06 | PASS |
| equity.phoenix.quad.stretch_0.02 | stepdown | pv | -3.26049 | 0.00389 | -3.25636 | 0.004133 | 0.01192 | 0.002286 | PASS |
| equity.phoenix.quad.stretch_0.02 | stepdown | delta | 0.470443 | 0.00108 | 0.472692 | 0.002249 | 0.004405 | 0.0005841 | PASS |
| equity.phoenix.quad.stretch_0.02 | stepdown | gamma | -0.0345018 | 0.0103 | -0.0369341 | -0.002432 | 0.0231 | 0.001473 | PASS |
| equity.phoenix.quad.stretch_0.02 | discrete_ki | pv | -2.90513 | 0.00314 | -2.9 | 0.005133 | 0.01142 | 1.281e-05 | PASS |
| equity.phoenix.quad.stretch_0.02 | discrete_ki | delta | 0.422553 | 0.00246 | 0.42706 | 0.004507 | 0.009427 | 4.527e-05 | PASS |
| equity.phoenix.quad.stretch_0.02 | discrete_ki | gamma | -0.0337775 | 0.00682 | -0.0338151 | -3.762e-05 | 0.01368 | 9.815e-06 | PASS |
| equity.phoenix.quad.stretch_0.02 | ki_stepdown | pv | -1.90855 | 0.00407 | -1.90785 | 0.0007006 | 0.008841 | 0.002389 | PASS |
| equity.phoenix.quad.stretch_0.02 | ki_stepdown | delta | 0.309466 | 0.00141 | 0.31316 | 0.003695 | 0.006517 | 0.0002466 | PASS |
| equity.phoenix.quad.stretch_0.02 | ki_stepdown | gamma | -0.040774 | 0.00623 | -0.0312154 | 0.009559 | 0.02202 | 1.598e-07 | PASS |
| equity.phoenix.quad.stretch_0.02 | disable_ko_after_ki | pv | -3.42908 | 0.00402 | -3.42761 | 0.001465 | 0.009509 | 0.0001608 | PASS |
| equity.phoenix.quad.stretch_0.02 | disable_ko_after_ki | delta | 0.489054 | 0.00105 | 0.488141 | -0.0009133 | 0.003021 | 1.041e-05 | PASS |
| equity.phoenix.quad.stretch_0.02 | disable_ko_after_ki | gamma | -0.0427926 | 0.00415 | -0.0354278 | 0.007365 | 0.01566 | 3.238e-06 | PASS |
| equity.phoenix.quad.stretch_0.02 | reverse | pv | -4.00024 | 0.0111 | -4.00374 | -0.003501 | 0.02573 | 0.00116 | PASS |
| equity.phoenix.quad.stretch_0.02 | reverse | delta | -0.593336 | 0.0042 | -0.590743 | 0.002592 | 0.011 | 8.228e-05 | PASS |
| equity.phoenix.quad.stretch_0.02 | reverse | gamma | -0.0290847 | 0.00925 | -0.035221 | -0.006136 | 0.02464 | 8.486e-06 | PASS |
| equity.phoenix.quad.stretch_0.02 | coupon_at_expiry | pv | -3.43198 | 0.00425 | -3.43075 | 0.001226 | 0.00973 | 0.0001311 | PASS |
| equity.phoenix.quad.stretch_0.02 | coupon_at_expiry | delta | 0.488815 | 0.000844 | 0.488224 | -0.0005909 | 0.00228 | 1.467e-05 | PASS |
| equity.phoenix.quad.stretch_0.02 | coupon_at_expiry | gamma | -0.0432232 | 0.00451 | -0.0353367 | 0.007886 | 0.0169 | 3.634e-06 | PASS |
| equity.phoenix.quad.stretch_0.02 | coupon_at_expiry_memory | pv | -3.35582 | 0.00453 | -3.35473 | 0.001093 | 0.01015 | 3.367e-05 | PASS |
| equity.phoenix.quad.stretch_0.02 | coupon_at_expiry_memory | delta | 0.478215 | 0.000904 | 0.477611 | -0.0006044 | 0.002413 | 2.392e-05 | PASS |
| equity.phoenix.quad.stretch_0.02 | coupon_at_expiry_memory | gamma | -0.042803 | 0.0044 | -0.0344561 | 0.008347 | 0.01716 | 3.208e-06 | PASS |

## Aggregate bias

| candidate | quantity | cells | mean bias (c) | SE (c) | passed |
|---|---|---|---|---|---|
| equity.phoenix.pde | pv | 14 | -0.000512 | 0.00125 | yes |
| equity.phoenix.pde | delta | 14 | -0.0004576 | 0.000652 | yes |
| equity.phoenix.pde | gamma | 14 | 0.00621 | 0.00217 | yes |
| equity.phoenix.quad | pv | 14 | 0.0003614 | 0.00125 | yes |
| equity.phoenix.quad | delta | 14 | -8.5e-05 | 0.000652 | yes |
| equity.phoenix.quad | gamma | 14 | 0.004496 | 0.00217 | yes |
| equity.phoenix.quad.stretch_0.02 | pv | 14 | 0.0001498 | 0.00125 | yes |
| equity.phoenix.quad.stretch_0.02 | delta | 14 | -6.681e-05 | 0.000652 | yes |
| equity.phoenix.quad.stretch_0.02 | gamma | 14 | 0.004496 | 0.00217 | yes |
