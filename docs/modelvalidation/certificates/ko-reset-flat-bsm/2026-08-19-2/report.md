# Certification report: ko-reset-flat-bsm

Evidence digest: `c3408743dcf9e4a93e2846e0eac244363c7a7c32c5b2006af0cfd8c973b6d927`

Machine: `arm64` / macOS-26.6-arm64-arm-64bit - Python 3.11.8, NumPy 2.4.6, quantark `5d7f22faecabdacf437bbd2c071025c576e1acfb`

## Amendment

| field | value |
|---|---|
| parent | docs/modelvalidation/certificates/ko-reset-flat-bsm/2026-08-19/certificate.json |
| parent digest | `4b5d2213a18ffd207556b0123699a05da60a576876d3776d2334e31ea5248ce1` |
| reason | Certify the European-KI, step-down (both schedules) and parachute product variants the original scope omitted |
| re-priced | 30 cell(s) |
| carried forward | 42 cell(s) |

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
| discrete_ki | 4 | se_budget_met | delta: 0.00409, gamma: 0.00883, pv: 0.00999 |
| european_ki | 4 | se_budget_met | delta: 0.00303, gamma: 0.0201, pv: 0.00975 |
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
| equity.ko_reset_snowball.pde | ordinary | delta | 0.621389 | 0.00868 | 0.614449 | -0.00694 | 0.0243 | 0.006387 | PASS |
| equity.ko_reset_snowball.pde | ordinary | gamma | -0.0416238 | 0.0297 | -0.0464698 | -0.004846 | 0.06424 | 0.0003026 | PASS |
| equity.ko_reset_snowball.pde | near_pre_ko | pv | 98.0475 | 0.0132 | 98.0527 | 0.005218 | 0.03166 | 0.001094 | PASS |
| equity.ko_reset_snowball.pde | near_pre_ko | delta | 0.487059 | 0.00233 | 0.487729 | 0.0006702 | 0.005339 | 0.008306 | PASS |
| equity.ko_reset_snowball.pde | near_pre_ko | gamma | -0.034448 | 0.0126 | -0.0501873 | -0.01574 | 0.0409 | 8.086e-05 | PASS |
| equity.ko_reset_snowball.pde | near_ki | pv | 79.0154 | 0.00618 | 79.0229 | 0.007533 | 0.01989 | 0.002191 | PASS |
| equity.ko_reset_snowball.pde | near_ki | delta | 1.06603 | 0.00536 | 1.06276 | -0.003275 | 0.01399 | 0.0002949 | PASS |
| equity.ko_reset_snowball.pde | near_ki | gamma | -0.0382894 | 0.0268 | -0.0594754 | -0.02119 | 0.07483 | 0.001151 | PASS |
| equity.ko_reset_snowball.pde | below_ki | pv | 74.0373 | 0.0119 | 74.0367 | -0.0006474 | 0.02455 | 0.001612 | PASS |
| equity.ko_reset_snowball.pde | below_ki | delta | 1.10845 | 0.00477 | 1.1063 | -0.002142 | 0.01167 | 0.000501 | PASS |
| equity.ko_reset_snowball.pde | below_ki | gamma | -0.0131948 | 0.0123 | 0.00544677 | 0.01864 | 0.04328 | 1.701e-05 | PASS |
| equity.ko_reset_snowball.pde | low_vol | pv | 100.165 | 0.0116 | 100.164 | -0.0009706 | 0.02419 | 0.00343 | PASS |
| equity.ko_reset_snowball.pde | low_vol | delta | 0.359471 | 0.00292 | 0.378962 | 0.01949 | 0.02534 | 0.009128 | PASS |
| equity.ko_reset_snowball.pde | low_vol | gamma | -0.0863362 | 0.0118 | -0.0960379 | -0.009702 | 0.03331 | 0.0004568 | PASS |
| equity.ko_reset_snowball.pde | near_expiry | pv | 99.5678 | 0.00505 | 99.5693 | 0.001497 | 0.01161 | 0.001347 | PASS |
| equity.ko_reset_snowball.pde | near_expiry | delta | 0.284584 | 0.00117 | 0.278687 | -0.005897 | 0.008232 | 0.006087 | PASS |
| equity.ko_reset_snowball.pde | near_expiry | gamma | -0.0472864 | 0.00329 | -0.0434562 | 0.00383 | 0.01041 | 0.0006736 | PASS |
| equity.ko_reset_snowball.pde | discrete_ki | pv | 96.7295 | 0.00999 | 96.7322 | 0.002688 | 0.02268 | 9.864e-06 | PASS |
| equity.ko_reset_snowball.pde | discrete_ki | delta | 0.612278 | 0.00409 | 0.608858 | -0.00342 | 0.0116 | 0.006534 | PASS |
| equity.ko_reset_snowball.pde | discrete_ki | gamma | -0.0366268 | 0.00883 | -0.046709 | -0.01008 | 0.02775 | 0.0002885 | PASS |
| equity.ko_reset_snowball.pde | european_ki | pv | 97.6357 | 0.00975 | 97.6496 | 0.01384 | 0.03335 | 7.589e-05 | PASS |
| equity.ko_reset_snowball.pde | european_ki | delta | 0.503157 | 0.00303 | 0.499476 | -0.003681 | 0.009741 | 0.006015 | PASS |
| equity.ko_reset_snowball.pde | european_ki | gamma | -0.040584 | 0.0201 | -0.0430878 | -0.002504 | 0.04262 | 0.0001073 | PASS |
| equity.ko_reset_snowball.pde | stepdown | pv | 98.2196 | 0.0175 | 98.2098 | -0.009763 | 0.04476 | 0.0008175 | PASS |
| equity.ko_reset_snowball.pde | stepdown | delta | 0.44371 | 0.00639 | 0.441396 | -0.002314 | 0.0151 | 0.001641 | PASS |
| equity.ko_reset_snowball.pde | stepdown | gamma | -0.0722007 | 0.0153 | -0.0435447 | 0.02866 | 0.05918 | 7.776e-06 | PASS |
| equity.ko_reset_snowball.pde | stepdown_near_last_pre_ko | pv | 96.7042 | 0.0104 | 96.6924 | -0.0118 | 0.03251 | 0.0004656 | PASS |
| equity.ko_reset_snowball.pde | stepdown_near_last_pre_ko | delta | 0.574327 | 0.00784 | 0.571872 | -0.002455 | 0.01813 | 0.0005325 | PASS |
| equity.ko_reset_snowball.pde | stepdown_near_last_pre_ko | gamma | -0.058302 | 0.00291 | -0.042525 | 0.01578 | 0.02161 | 1.515e-05 | PASS |
| equity.ko_reset_snowball.pde | parachute | pv | 97.4197 | 0.00947 | 97.4293 | 0.009548 | 0.02849 | 0.003274 | PASS |
| equity.ko_reset_snowball.pde | parachute | delta | 0.550792 | 0.0047 | 0.545192 | -0.0056 | 0.01501 | 0.006709 | PASS |
| equity.ko_reset_snowball.pde | parachute | gamma | -0.043298 | 0.0274 | -0.050285 | -0.006987 | 0.06183 | 1.047e-05 | PASS |
| equity.ko_reset_snowball.pde | parachute_near_ki | pv | 79.0793 | 0.00666 | 79.0904 | 0.01109 | 0.02442 | 0.000817 | PASS |
| equity.ko_reset_snowball.pde | parachute_near_ki | delta | 1.17078 | 0.00552 | 1.16097 | -0.009812 | 0.02086 | 0.0013 | PASS |
| equity.ko_reset_snowball.pde | parachute_near_ki | gamma | 0.0247501 | 0.0271 | 0.0332451 | 0.008495 | 0.06274 | 0.0005254 | PASS |
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

## Aggregate bias

| candidate | quantity | cells | mean bias (c) | SE (c) | passed |
|---|---|---|---|---|---|
| equity.ko_reset_snowball.pde | pv | 12 | 0.003097 | 0.0031 | yes |
| equity.ko_reset_snowball.pde | delta | 12 | -0.002115 | 0.0015 | yes |
| equity.ko_reset_snowball.pde | gamma | 12 | 0.0003628 | 0.00545 | yes |
| equity.ko_reset_snowball.quad | pv | 12 | 0.01225 | 0.0031 | yes |
| equity.ko_reset_snowball.quad | delta | 12 | -0.003327 | 0.0015 | yes |
| equity.ko_reset_snowball.quad | gamma | 12 | 0.0008214 | 0.00545 | yes |
