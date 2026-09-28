# Certification report: phoenix-flat-bsm

Evidence digest: `2d35741043195f9a73fd2c6b9f5581a564f20e13c13b605922190771c5e2c699`

Machine: `arm64` / macOS-26.6-arm64-arm-64bit - Python 3.11.8, NumPy 2.4.6, quantark `42926c9b75d4344a956fc61b92d73f534fd1f686`

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
| equity.phoenix.quad | grid.align_cell_stretch | -- |
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
| equity.phoenix.pde | ordinary | delta | 0.488242 | 0.000842 | 0.487866 | -0.0003767 | 0.002061 | 0.0001456 | PASS |
| equity.phoenix.pde | ordinary | gamma | -0.0432388 | 0.00451 | -0.0355594 | 0.007679 | 0.01669 | 0.0001215 | PASS |
| equity.phoenix.pde | near_ko | pv | -2.31343 | 0.00358 | -2.31872 | -0.005292 | 0.01246 | 0.0004943 | PASS |
| equity.phoenix.pde | near_ko | delta | 0.400328 | 0.00292 | 0.395726 | -0.004602 | 0.01044 | 2.135e-05 | PASS |
| equity.phoenix.pde | near_ko | gamma | -0.0493471 | 0.00687 | -0.0381013 | 0.01125 | 0.025 | 1.546e-05 | PASS |
| equity.phoenix.pde | near_coupon | pv | -13.8608 | 0.00517 | -13.8715 | -0.01062 | 0.02097 | 0.003183 | PASS |
| equity.phoenix.pde | near_coupon | delta | 0.948728 | 0.00425 | 0.947548 | -0.001181 | 0.009678 | 8.751e-05 | PASS |
| equity.phoenix.pde | near_coupon | gamma | -0.0437926 | 0.0216 | -0.0294236 | 0.01437 | 0.05753 | 0.0002596 | PASS |
| equity.phoenix.pde | near_ki | pv | -23.9184 | 0.00146 | -23.9199 | -0.001497 | 0.004421 | 0.002681 | PASS |
| equity.phoenix.pde | near_ki | delta | 1.13636 | 0.00424 | 1.127 | -0.009357 | 0.01783 | 0.0003492 | PASS |
| equity.phoenix.pde | near_ki | gamma | -0.00717388 | 0.00491 | 0.0175587 | 0.02473 | 0.03455 | 0.001589 | PASS |
| equity.phoenix.pde | low_vol | pv | 0.526131 | 0.00105 | 0.52683 | 0.0006989 | 0.002792 | 0.00272 | PASS |
| equity.phoenix.pde | low_vol | delta | 0.00547519 | 0.00269 | 0.00463642 | -0.0008388 | 0.006225 | 0.0006627 | PASS |
| equity.phoenix.pde | low_vol | gamma | -0.0266412 | 0.00211 | -0.031867 | -0.005226 | 0.009439 | 0.001594 | PASS |
| equity.phoenix.pde | near_expiry | pv | 0.0923547 | 0.00199 | 0.0910755 | -0.001279 | 0.00525 | 0.0008889 | PASS |
| equity.phoenix.pde | near_expiry | delta | 0.0556837 | 0.000659 | 0.0547821 | -0.0009016 | 0.002219 | 0.0002016 | PASS |
| equity.phoenix.pde | near_expiry | gamma | -0.0205876 | 0.00317 | -0.0177815 | 0.002806 | 0.009141 | 0.0001855 | PASS |
| equity.phoenix.pde | memory | pv | -3.34946 | 0.00452 | -3.34829 | 0.001174 | 0.01022 | 0.0006083 | PASS |
| equity.phoenix.pde | memory | delta | 0.477563 | 0.000903 | 0.477173 | -0.0003906 | 0.002197 | 0.0001439 | PASS |
| equity.phoenix.pde | memory | gamma | -0.0428127 | 0.0044 | -0.0346691 | 0.008144 | 0.01695 | 0.0001205 | PASS |
| equity.phoenix.pde | stepdown | pv | -3.26049 | 0.00389 | -3.25703 | 0.003461 | 0.01125 | 0.001951 | PASS |
| equity.phoenix.pde | stepdown | delta | 0.470443 | 0.00108 | 0.472819 | 0.002376 | 0.004531 | 2.748e-06 | PASS |
| equity.phoenix.pde | stepdown | gamma | -0.0345018 | 0.0103 | -0.0359311 | -0.001429 | 0.0221 | 7.195e-05 | PASS |
| equity.phoenix.pde | discrete_ki | pv | -2.90513 | 0.00314 | -2.89923 | 0.005897 | 0.01218 | 6.012e-05 | PASS |
| equity.phoenix.pde | discrete_ki | delta | 0.422553 | 0.00246 | 0.427045 | 0.004492 | 0.009412 | 5.773e-05 | PASS |
| equity.phoenix.pde | discrete_ki | gamma | -0.0337775 | 0.00682 | -0.0339405 | -0.000163 | 0.01381 | 7.93e-05 | PASS |
| equity.phoenix.pde | ki_stepdown | pv | -1.90855 | 0.00407 | -1.90768 | 0.0008703 | 0.009011 | 0.0006134 | PASS |
| equity.phoenix.pde | ki_stepdown | delta | 0.309466 | 0.00141 | 0.312938 | 0.003472 | 0.006294 | 0.0001529 | PASS |
| equity.phoenix.pde | ki_stepdown | gamma | -0.040774 | 0.00623 | -0.0312366 | 0.009537 | 0.022 | 3.86e-05 | PASS |
| equity.phoenix.pde | disable_ko_after_ki | pv | -3.42908 | 0.00402 | -3.42754 | 0.001532 | 0.009576 | 0.0006189 | PASS |
| equity.phoenix.pde | disable_ko_after_ki | delta | 0.489054 | 0.00105 | 0.488359 | -0.000695 | 0.002803 | 0.0001475 | PASS |
| equity.phoenix.pde | disable_ko_after_ki | gamma | -0.0427926 | 0.00415 | -0.0356357 | 0.007157 | 0.01545 | 0.0001208 | PASS |
| equity.phoenix.pde | reverse | pv | -4.00024 | 0.0111 | -4.00604 | -0.005806 | 0.02803 | 0.002822 | PASS |
| equity.phoenix.pde | reverse | delta | -0.593336 | 0.0042 | -0.591622 | 0.001714 | 0.01012 | 0.0002446 | PASS |
| equity.phoenix.pde | reverse | gamma | -0.0290847 | 0.00925 | -0.0351713 | -0.006087 | 0.02459 | 0.0001798 | PASS |
| equity.phoenix.pde | coupon_at_expiry | pv | -3.43198 | 0.00425 | -3.43068 | 0.0013 | 0.009804 | 0.0006079 | PASS |
| equity.phoenix.pde | coupon_at_expiry | delta | 0.488815 | 0.000844 | 0.488445 | -0.0003697 | 0.002059 | 0.0001456 | PASS |
| equity.phoenix.pde | coupon_at_expiry | gamma | -0.0432232 | 0.00451 | -0.0355465 | 0.007677 | 0.01669 | 0.0001222 | PASS |
| equity.phoenix.pde | coupon_at_expiry_memory | pv | -3.35582 | 0.00453 | -3.35464 | 0.001179 | 0.01024 | 0.0006074 | PASS |
| equity.phoenix.pde | coupon_at_expiry_memory | delta | 0.478215 | 0.000904 | 0.477832 | -0.000383 | 0.002192 | 0.0001439 | PASS |
| equity.phoenix.pde | coupon_at_expiry_memory | gamma | -0.042803 | 0.0044 | -0.0346649 | 0.008138 | 0.01695 | 0.0001212 | PASS |
| equity.phoenix.quad | ordinary | pv | -3.42614 | 0.00424 | -3.42618 | -3.764e-05 | 0.008526 | 0.004547 | PASS |
| equity.phoenix.quad | ordinary | delta | 0.488242 | 0.000842 | 0.487735 | -0.0005078 | 0.002192 | 0.003148 | PASS |
| equity.phoenix.quad | ordinary | gamma | -0.0432388 | 0.00451 | -0.0332451 | 0.009994 | 0.01901 | 0.0001285 | PASS |
| equity.phoenix.quad | near_ko | pv | -2.31343 | 0.00358 | -2.32044 | -0.007007 | 0.01418 | 0.007224 | PASS |
| equity.phoenix.quad | near_ko | delta | 0.400328 | 0.00292 | 0.395608 | -0.00472 | 0.01056 | 0.002238 | PASS |
| equity.phoenix.quad | near_ko | gamma | -0.0493471 | 0.00687 | -0.0379981 | 0.01135 | 0.0251 | 0.002753 | PASS |
| equity.phoenix.quad | near_coupon | pv | -13.8608 | 0.00517 | -13.8661 | -0.005296 | 0.01564 | 0.004035 | PASS |
| equity.phoenix.quad | near_coupon | delta | 0.948728 | 0.00425 | 0.947138 | -0.00159 | 0.01009 | 5.253e-06 | PASS |
| equity.phoenix.quad | near_coupon | gamma | -0.0437926 | 0.0216 | -0.0308271 | 0.01297 | 0.05612 | 0.00295 | PASS |
| equity.phoenix.quad | near_ki | pv | -23.9184 | 0.00146 | -23.9187 | -0.0002812 | 0.003205 | 0.0009991 | PASS |
| equity.phoenix.quad | near_ki | delta | 1.13636 | 0.00424 | 1.13168 | -0.004682 | 0.01315 | 0.0006387 | PASS |
| equity.phoenix.quad | near_ki | gamma | -0.00717388 | 0.00491 | -0.00610506 | 0.001069 | 0.01089 | 0.001399 | PASS |
| equity.phoenix.quad | low_vol | pv | 0.526131 | 0.00105 | 0.52733 | 0.001198 | 0.003292 | 0.004685 | PASS |
| equity.phoenix.quad | low_vol | delta | 0.00547519 | 0.00269 | 0.00550298 | 2.779e-05 | 0.005414 | 0.00173 | PASS |
| equity.phoenix.quad | low_vol | gamma | -0.0266412 | 0.00211 | -0.0319791 | -0.005338 | 0.009551 | 0.0002637 | PASS |
| equity.phoenix.quad | near_expiry | pv | 0.0923547 | 0.00199 | 0.0912468 | -0.001108 | 0.005079 | 0.0003531 | PASS |
| equity.phoenix.quad | near_expiry | delta | 0.0556837 | 0.000659 | 0.0554577 | -0.000226 | 0.001544 | 3.223e-06 | PASS |
| equity.phoenix.quad | near_expiry | gamma | -0.0205876 | 0.00317 | -0.0183734 | 0.002214 | 0.008549 | 0.0002962 | PASS |
| equity.phoenix.quad | memory | pv | -3.34946 | 0.00452 | -3.34957 | -0.0001063 | 0.009154 | 0.004358 | PASS |
| equity.phoenix.quad | memory | delta | 0.477563 | 0.000903 | 0.477033 | -0.0005301 | 0.002337 | 0.003084 | PASS |
| equity.phoenix.quad | memory | gamma | -0.0428127 | 0.0044 | -0.0324083 | 0.0104 | 0.01921 | 0.0001341 | PASS |
| equity.phoenix.quad | stepdown | pv | -3.26049 | 0.00389 | -3.25671 | 0.003782 | 0.01157 | 0.002511 | PASS |
| equity.phoenix.quad | stepdown | delta | 0.470443 | 0.00108 | 0.472622 | 0.002179 | 0.004335 | 0.001386 | PASS |
| equity.phoenix.quad | stepdown | gamma | -0.0345018 | 0.0103 | -0.0364904 | -0.001989 | 0.02266 | 0.0005013 | PASS |
| equity.phoenix.quad | discrete_ki | pv | -2.90513 | 0.00314 | -2.90139 | 0.003737 | 0.01002 | 0.004105 | PASS |
| equity.phoenix.quad | discrete_ki | delta | 0.422553 | 0.00246 | 0.427189 | 0.004636 | 0.009556 | 0.002953 | PASS |
| equity.phoenix.quad | discrete_ki | gamma | -0.0337775 | 0.00682 | -0.0317914 | 0.001986 | 0.01563 | 6.266e-05 | PASS |
| equity.phoenix.quad | ki_stepdown | pv | -1.90855 | 0.00407 | -1.90952 | -0.0009667 | 0.009107 | 0.005307 | PASS |
| equity.phoenix.quad | ki_stepdown | delta | 0.309466 | 0.00141 | 0.313376 | 0.00391 | 0.006732 | 0.002324 | PASS |
| equity.phoenix.quad | ki_stepdown | gamma | -0.040774 | 0.00623 | -0.0293308 | 0.01144 | 0.02391 | 0.0004253 | PASS |
| equity.phoenix.quad | disable_ko_after_ki | pv | -3.42908 | 0.00402 | -3.42887 | 0.000203 | 0.008246 | 0.004577 | PASS |
| equity.phoenix.quad | disable_ko_after_ki | delta | 0.489054 | 0.00105 | 0.488229 | -0.000825 | 0.002933 | 0.00315 | PASS |
| equity.phoenix.quad | disable_ko_after_ki | gamma | -0.0427926 | 0.00415 | -0.0333175 | 0.009475 | 0.01777 | 0.0001262 | PASS |
| equity.phoenix.quad | reverse | pv | -4.00024 | 0.0111 | -4.0053 | -0.005065 | 0.02729 | 0.008171 | PASS |
| equity.phoenix.quad | reverse | delta | -0.593336 | 0.0042 | -0.59075 | 0.002586 | 0.011 | 0.003026 | PASS |
| equity.phoenix.quad | reverse | gamma | -0.0290847 | 0.00925 | -0.0345928 | -0.005508 | 0.02401 | 0.002526 | PASS |
| equity.phoenix.quad | coupon_at_expiry | pv | -3.43198 | 0.00425 | -3.43201 | -3.102e-05 | 0.008535 | 0.004534 | PASS |
| equity.phoenix.quad | coupon_at_expiry | delta | 0.488815 | 0.000844 | 0.488311 | -0.0005041 | 0.002193 | 0.003149 | PASS |
| equity.phoenix.quad | coupon_at_expiry | gamma | -0.0432232 | 0.00451 | -0.033232 | 0.009991 | 0.01901 | 0.0001319 | PASS |
| equity.phoenix.quad | coupon_at_expiry_memory | pv | -3.35582 | 0.00453 | -3.35592 | -9.83e-05 | 0.00916 | 0.004348 | PASS |
| equity.phoenix.quad | coupon_at_expiry_memory | delta | 0.478215 | 0.000904 | 0.47769 | -0.0005256 | 0.002334 | 0.003086 | PASS |
| equity.phoenix.quad | coupon_at_expiry_memory | gamma | -0.042803 | 0.0044 | -0.0324035 | 0.0104 | 0.01921 | 0.0001374 | PASS |
| equity.phoenix.quad.stretch_0.02 | ordinary | pv | -3.42614 | 0.00424 | -3.42658 | -0.0004398 | 0.008929 | 0.004145 | PASS |
| equity.phoenix.quad.stretch_0.02 | ordinary | delta | 0.488242 | 0.000842 | 0.487758 | -0.0004844 | 0.002168 | 0.003171 | PASS |
| equity.phoenix.quad.stretch_0.02 | ordinary | gamma | -0.0432388 | 0.00451 | -0.0330108 | 0.01023 | 0.01924 | 0.0003628 | PASS |
| equity.phoenix.quad.stretch_0.02 | near_ko | pv | -2.31343 | 0.00358 | -2.32074 | -0.007311 | 0.01448 | 0.00692 | PASS |
| equity.phoenix.quad.stretch_0.02 | near_ko | delta | 0.400328 | 0.00292 | 0.395625 | -0.004703 | 0.01055 | 0.002256 | PASS |
| equity.phoenix.quad.stretch_0.02 | near_ko | gamma | -0.0493471 | 0.00687 | -0.0379668 | 0.01138 | 0.02513 | 0.002722 | PASS |
| equity.phoenix.quad.stretch_0.02 | near_coupon | pv | -13.8608 | 0.00517 | -13.8663 | -0.005495 | 0.01584 | 0.003836 | PASS |
| equity.phoenix.quad.stretch_0.02 | near_coupon | delta | 0.948728 | 0.00425 | 0.94712 | -0.001608 | 0.01011 | 1.258e-05 | PASS |
| equity.phoenix.quad.stretch_0.02 | near_coupon | gamma | -0.0437926 | 0.0216 | -0.0308144 | 0.01298 | 0.05614 | 0.002937 | PASS |
| equity.phoenix.quad.stretch_0.02 | near_ki | pv | -23.9184 | 0.00146 | -23.9186 | -0.0002074 | 0.003132 | 0.001073 | PASS |
| equity.phoenix.quad.stretch_0.02 | near_ki | delta | 1.13636 | 0.00424 | 1.13168 | -0.004684 | 0.01315 | 0.0006371 | PASS |
| equity.phoenix.quad.stretch_0.02 | near_ki | gamma | -0.00717388 | 0.00491 | -0.006073 | 0.001101 | 0.01092 | 0.001367 | PASS |
| equity.phoenix.quad.stretch_0.02 | low_vol | pv | 0.526131 | 0.00105 | 0.526915 | 0.0007838 | 0.002877 | 0.00427 | PASS |
| equity.phoenix.quad.stretch_0.02 | low_vol | delta | 0.00547519 | 0.00269 | 0.00574661 | 0.0002714 | 0.005657 | 0.001487 | PASS |
| equity.phoenix.quad.stretch_0.02 | low_vol | gamma | -0.0266412 | 0.00211 | -0.0321305 | -0.005489 | 0.009702 | 0.0001124 | PASS |
| equity.phoenix.quad.stretch_0.02 | near_expiry | pv | 0.0923547 | 0.00199 | 0.0912468 | -0.001108 | 0.005079 | 0.0009771 | PASS |
| equity.phoenix.quad.stretch_0.02 | near_expiry | delta | 0.0556837 | 0.000659 | 0.0554577 | -0.000226 | 0.001544 | 7.388e-05 | PASS |
| equity.phoenix.quad.stretch_0.02 | near_expiry | gamma | -0.0205876 | 0.00317 | -0.0183734 | 0.002214 | 0.008549 | 0.0003211 | PASS |
| equity.phoenix.quad.stretch_0.02 | memory | pv | -3.34946 | 0.00452 | -3.34999 | -0.0005324 | 0.00958 | 0.003932 | PASS |
| equity.phoenix.quad.stretch_0.02 | memory | delta | 0.477563 | 0.000903 | 0.47706 | -0.0005029 | 0.00231 | 0.003111 | PASS |
| equity.phoenix.quad.stretch_0.02 | memory | gamma | -0.0428127 | 0.0044 | -0.0321803 | 0.01063 | 0.01944 | 0.0003621 | PASS |
| equity.phoenix.quad.stretch_0.02 | stepdown | pv | -3.26049 | 0.00389 | -3.25671 | 0.003782 | 0.01157 | 0.003133 | PASS |
| equity.phoenix.quad.stretch_0.02 | stepdown | delta | 0.470443 | 0.00108 | 0.472622 | 0.002179 | 0.004335 | 0.001296 | PASS |
| equity.phoenix.quad.stretch_0.02 | stepdown | gamma | -0.0345018 | 0.0103 | -0.0364904 | -0.001989 | 0.02266 | 0.00139 | PASS |
| equity.phoenix.quad.stretch_0.02 | discrete_ki | pv | -2.90513 | 0.00314 | -2.90161 | 0.003523 | 0.009809 | 0.00389 | PASS |
| equity.phoenix.quad.stretch_0.02 | discrete_ki | delta | 0.422553 | 0.00246 | 0.427194 | 0.004641 | 0.009561 | 0.002959 | PASS |
| equity.phoenix.quad.stretch_0.02 | discrete_ki | gamma | -0.0337775 | 0.00682 | -0.0315632 | 0.002214 | 0.01586 | 0.0001655 | PASS |
| equity.phoenix.quad.stretch_0.02 | ki_stepdown | pv | -1.90855 | 0.00407 | -1.90938 | -0.000826 | 0.008966 | 0.006085 | PASS |
| equity.phoenix.quad.stretch_0.02 | ki_stepdown | delta | 0.309466 | 0.00141 | 0.313345 | 0.003879 | 0.006701 | 0.002398 | PASS |
| equity.phoenix.quad.stretch_0.02 | ki_stepdown | gamma | -0.040774 | 0.00623 | -0.0291312 | 0.01164 | 0.02411 | 0.0003264 | PASS |
| equity.phoenix.quad.stretch_0.02 | disable_ko_after_ki | pv | -3.42908 | 0.00402 | -3.42928 | -0.0001991 | 0.008242 | 0.004175 | PASS |
| equity.phoenix.quad.stretch_0.02 | disable_ko_after_ki | delta | 0.489054 | 0.00105 | 0.488253 | -0.0008017 | 0.00291 | 0.003173 | PASS |
| equity.phoenix.quad.stretch_0.02 | disable_ko_after_ki | gamma | -0.0427926 | 0.00415 | -0.0330826 | 0.00971 | 0.018 | 0.0003611 | PASS |
| equity.phoenix.quad.stretch_0.02 | reverse | pv | -4.00024 | 0.0111 | -4.00581 | -0.005569 | 0.0278 | 0.007708 | PASS |
| equity.phoenix.quad.stretch_0.02 | reverse | delta | -0.593336 | 0.0042 | -0.590759 | 0.002577 | 0.01099 | 0.003772 | PASS |
| equity.phoenix.quad.stretch_0.02 | reverse | gamma | -0.0290847 | 0.00925 | -0.0339125 | -0.004828 | 0.02333 | 0.002221 | PASS |
| equity.phoenix.quad.stretch_0.02 | coupon_at_expiry | pv | -3.43198 | 0.00425 | -3.43241 | -0.0004334 | 0.008937 | 0.004132 | PASS |
| equity.phoenix.quad.stretch_0.02 | coupon_at_expiry | delta | 0.488815 | 0.000844 | 0.488334 | -0.0004805 | 0.002169 | 0.003173 | PASS |
| equity.phoenix.quad.stretch_0.02 | coupon_at_expiry | gamma | -0.0432232 | 0.00451 | -0.0329979 | 0.01023 | 0.01924 | 0.0003661 | PASS |
| equity.phoenix.quad.stretch_0.02 | coupon_at_expiry_memory | pv | -3.35582 | 0.00453 | -3.35635 | -0.0005244 | 0.009586 | 0.003921 | PASS |
| equity.phoenix.quad.stretch_0.02 | coupon_at_expiry_memory | delta | 0.478215 | 0.000904 | 0.477717 | -0.0004984 | 0.002307 | 0.003113 | PASS |
| equity.phoenix.quad.stretch_0.02 | coupon_at_expiry_memory | gamma | -0.042803 | 0.0044 | -0.0321756 | 0.01063 | 0.01944 | 0.0003653 | PASS |

## Aggregate bias

| candidate | quantity | cells | mean bias (c) | SE (c) | passed |
|---|---|---|---|---|---|
| equity.phoenix.pde | pv | 14 | -0.0005064 | 0.00125 | yes |
| equity.phoenix.pde | delta | 14 | -0.0005029 | 0.000652 | yes |
| equity.phoenix.pde | gamma | 14 | 0.006327 | 0.00217 | yes |
| equity.phoenix.quad | pv | 14 | -0.0007912 | 0.00125 | yes |
| equity.phoenix.quad | delta | 14 | -5.521e-05 | 0.000652 | yes |
| equity.phoenix.quad | gamma | 14 | 0.005604 | 0.00217 | yes |
| equity.phoenix.quad.stretch_0.02 | pv | 14 | -0.00104 | 0.00125 | yes |
| equity.phoenix.quad.stretch_0.02 | delta | 14 | -3.15e-05 | 0.000652 | yes |
| equity.phoenix.quad.stretch_0.02 | gamma | 14 | 0.005761 | 0.00217 | yes |
