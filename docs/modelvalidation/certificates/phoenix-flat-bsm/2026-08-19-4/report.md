# Certification report: phoenix-flat-bsm

Evidence digest: `35d6ee3a290cbaacd4cacf22b70405a1678dc84b48ec5d646fbf8737374c7542`

Machine: `arm64` / macOS-26.6-arm64-arm-64bit - Python 3.11.8, NumPy 2.4.6, quantark `10be095bf984de8b5c7aa7bd93de32dfe69cb0fb`

## Amendment

| field | value |
|---|---|
| parent | docs/modelvalidation/certificates/phoenix-flat-bsm/2026-08-19-3/certificate.json |
| parent digest | `1133341daafebd859cd89935883ab95dfb302b4b1de9889a5d70b4c312ce96cf` |
| reason | Certify coupon_pay_type=EXPIRY now that all three engines roll coupons up to the note's termination: the deterministic pair carry a termination-value surface and the benchmark pays the roll-up at the knock-out settlement |
| re-priced | 12 cell(s) |
| carried forward | 72 cell(s) |

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
| equity.phoenix.pde | ordinary | delta | 0.488242 | 0.000842 | 0.484888 | -0.003354 | 0.005038 | 0.001958 | PASS |
| equity.phoenix.pde | ordinary | gamma | -0.0432388 | 0.00451 | -0.0354742 | 0.007765 | 0.01678 | 6.963e-05 | PASS |
| equity.phoenix.pde | near_ko | pv | -2.31343 | 0.00358 | -2.31872 | -0.005292 | 0.01246 | 0.0004943 | PASS |
| equity.phoenix.pde | near_ko | delta | 0.400328 | 0.00292 | 0.394197 | -0.006132 | 0.01197 | 0.0004674 | PASS |
| equity.phoenix.pde | near_ko | gamma | -0.0493471 | 0.00687 | -0.038058 | 0.01129 | 0.02504 | 1.643e-06 | PASS |
| equity.phoenix.pde | near_coupon | pv | -13.8608 | 0.00517 | -13.8715 | -0.01062 | 0.02097 | 0.003183 | PASS |
| equity.phoenix.pde | near_coupon | delta | 0.948728 | 0.00425 | 0.949923 | 0.001195 | 0.009692 | 0.003883 | PASS |
| equity.phoenix.pde | near_coupon | gamma | -0.0437926 | 0.0216 | -0.0294966 | 0.0143 | 0.05746 | 0.0001431 | PASS |
| equity.phoenix.pde | near_ki | pv | -23.9184 | 0.00146 | -23.9199 | -0.001497 | 0.004421 | 0.002681 | PASS |
| equity.phoenix.pde | near_ki | delta | 1.13636 | 0.00424 | 1.1267 | -0.009661 | 0.01813 | 0.0005933 | PASS |
| equity.phoenix.pde | near_ki | gamma | -0.00717388 | 0.00491 | 0.017574 | 0.02475 | 0.03457 | 0.001601 | PASS |
| equity.phoenix.pde | low_vol | pv | 0.526131 | 0.00105 | 0.52683 | 0.0006989 | 0.002792 | 0.00272 | PASS |
| equity.phoenix.pde | low_vol | delta | 0.00547519 | 0.00269 | 0.000710208 | -0.004765 | 0.01015 | 0.006752 | PASS |
| equity.phoenix.pde | low_vol | gamma | -0.0266412 | 0.00211 | -0.0317493 | -0.005108 | 0.009321 | 0.001816 | PASS |
| equity.phoenix.pde | near_expiry | pv | 0.0923547 | 0.00199 | 0.0910755 | -0.001279 | 0.00525 | 0.0008889 | PASS |
| equity.phoenix.pde | near_expiry | delta | 0.0556837 | 0.000659 | 0.0527909 | -0.002893 | 0.00421 | 0.000975 | PASS |
| equity.phoenix.pde | near_expiry | gamma | -0.0205876 | 0.00317 | -0.0177224 | 0.002865 | 0.0092 | 0.0002085 | PASS |
| equity.phoenix.pde | memory | pv | -3.34946 | 0.00452 | -3.34829 | 0.001174 | 0.01022 | 0.0006083 | PASS |
| equity.phoenix.pde | memory | delta | 0.477563 | 0.000903 | 0.47427 | -0.003294 | 0.0051 | 0.001911 | PASS |
| equity.phoenix.pde | memory | gamma | -0.0428127 | 0.0044 | -0.0345861 | 0.008227 | 0.01703 | 6.997e-05 | PASS |
| equity.phoenix.pde | stepdown | pv | -3.26049 | 0.00389 | -3.25703 | 0.003461 | 0.01125 | 0.001951 | PASS |
| equity.phoenix.pde | stepdown | delta | 0.470443 | 0.00108 | 0.473905 | 0.003462 | 0.005617 | 0.001135 | PASS |
| equity.phoenix.pde | stepdown | gamma | -0.0345018 | 0.0103 | -0.0359623 | -0.00146 | 0.02213 | 3.948e-05 | PASS |
| equity.phoenix.pde | discrete_ki | pv | -2.90513 | 0.00314 | -2.89923 | 0.005897 | 0.01218 | 6.012e-05 | PASS |
| equity.phoenix.pde | discrete_ki | delta | 0.422553 | 0.00246 | 0.424203 | 0.00165 | 0.00657 | 0.001786 | PASS |
| equity.phoenix.pde | discrete_ki | gamma | -0.0337775 | 0.00682 | -0.0338588 | -8.129e-05 | 0.01372 | 2.965e-05 | PASS |
| equity.phoenix.pde | ki_stepdown | pv | -1.90855 | 0.00407 | -1.90768 | 0.0008703 | 0.009011 | 0.0006134 | PASS |
| equity.phoenix.pde | ki_stepdown | delta | 0.309466 | 0.00141 | 0.317173 | 0.007708 | 0.01053 | 0.003344 | PASS |
| equity.phoenix.pde | ki_stepdown | gamma | -0.040774 | 0.00623 | -0.0313595 | 0.009414 | 0.02188 | 0.0001401 | PASS |
| equity.phoenix.pde | disable_ko_after_ki | pv | -3.42908 | 0.00402 | -3.42754 | 0.001532 | 0.009576 | 0.0006189 | PASS |
| equity.phoenix.pde | disable_ko_after_ki | delta | 0.489054 | 0.00105 | 0.485375 | -0.003679 | 0.005787 | 0.001963 | PASS |
| equity.phoenix.pde | disable_ko_after_ki | gamma | -0.0427926 | 0.00415 | -0.0355503 | 0.007242 | 0.01554 | 6.883e-05 | PASS |
| equity.phoenix.pde | reverse | pv | -4.00024 | 0.0111 | -4.00604 | -0.005806 | 0.02803 | 0.002822 | PASS |
| equity.phoenix.pde | reverse | delta | -0.593336 | 0.0042 | -0.592098 | 0.001238 | 0.009647 | 0.002212 | PASS |
| equity.phoenix.pde | reverse | gamma | -0.0290847 | 0.00925 | -0.0351562 | -0.006071 | 0.02457 | 0.0002576 | PASS |
| equity.phoenix.pde | coupon_at_expiry | pv | -3.43198 | 0.00425 | -3.43068 | 0.0013 | 0.009804 | 0.0006079 | PASS |
| equity.phoenix.pde | coupon_at_expiry | delta | 0.488815 | 0.000844 | 0.485469 | -0.003346 | 0.005035 | 0.001957 | PASS |
| equity.phoenix.pde | coupon_at_expiry | gamma | -0.0432232 | 0.00451 | -0.0354613 | 0.007762 | 0.01678 | 7.035e-05 | PASS |
| equity.phoenix.pde | coupon_at_expiry_memory | pv | -3.35582 | 0.00453 | -3.35464 | 0.001179 | 0.01024 | 0.0006074 | PASS |
| equity.phoenix.pde | coupon_at_expiry_memory | delta | 0.478215 | 0.000904 | 0.474929 | -0.003286 | 0.005095 | 0.00191 | PASS |
| equity.phoenix.pde | coupon_at_expiry_memory | gamma | -0.042803 | 0.0044 | -0.0345819 | 0.008221 | 0.01703 | 7.066e-05 | PASS |
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

## Aggregate bias

| candidate | quantity | cells | mean bias (c) | SE (c) | passed |
|---|---|---|---|---|---|
| equity.phoenix.pde | pv | 14 | -0.0005064 | 0.00125 | yes |
| equity.phoenix.pde | delta | 14 | -0.001797 | 0.000652 | yes |
| equity.phoenix.pde | gamma | 14 | 0.006365 | 0.00217 | yes |
| equity.phoenix.quad | pv | 14 | -0.0007912 | 0.00125 | yes |
| equity.phoenix.quad | delta | 14 | -5.521e-05 | 0.000652 | yes |
| equity.phoenix.quad | gamma | 14 | 0.005604 | 0.00217 | yes |
