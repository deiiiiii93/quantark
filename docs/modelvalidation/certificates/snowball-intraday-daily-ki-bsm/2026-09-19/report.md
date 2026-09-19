# Certification report: snowball-intraday-daily-ki-bsm

Evidence digest: `a0569e97687cc01cc67df75b00612b3fc6a3c4e215ffe7bde5b3a2ede9a9893d`

Machine: `arm64` / macOS-27.0-arm64-arm-64bit - Python 3.11.8, NumPy 2.4.6, quantark `08fa9e705bf20d1e841f8fefe4529544f5409a65`

## Decisions

| candidate | decision |
|---|---|
| equity.snowball.intraday.pde | REJECTED |
| equity.snowball.intraday.quad_v2 | ADMITTED |

Bounds: cell 1 c, mean signed bias 0.2 c, reference radius at most 0.25 x cell (deterministic reference: a radius is consumed whole, with no interval multiplier).

Schema 2: gate values are a fraction of each cell's own budget (quantity_bounds below); the cell bound of 1 is that budget.

| quantity | abs floor | relative |
|---|---|---|
| desk_delta | 1e-05 | 0.0001 |
| desk_gamma | 0.0001 | 0.001 |
| desk_theta | 1e-06 | 0.0001 |
| point_delta | 1e-05 | 0.0001 |
| point_gamma | 0.0001 | 0.001 |
| pv | 1e-06 | 0 |

## Engine configuration

Resolved rather than named: a profile such as `standard` is an indirection whose meaning can change between releases. These are the requested settings.

| engine | setting | value |
|---|---|---|
| equity.snowball.intraday.quad_v2 | cells_per_sd | 2 |
| equity.snowball.intraday.quad_v2 | convergence_axes.cells_per_sd | [2.0, 4.0, 8.0] |
| equity.snowball.intraday.quad_v2 | convergence_axes.domain_sd | [11.0, 13.0, 15.0] |
| equity.snowball.intraday.quad_v2 | convergence_axes.order | [8, 12, 16] |
| equity.snowball.intraday.quad_v2 | desk_bump | 0.01 |
| equity.snowball.intraday.quad_v2 | engine | SnowballQuadEngineV2 |
| equity.snowball.intraday.quad_v2 | grid.bump_config.div_bump | 0.0001 |
| equity.snowball.intraday.quad_v2 | grid.bump_config.gamma_spot_bump | -- |
| equity.snowball.intraday.quad_v2 | grid.bump_config.rate_bump | 0.0001 |
| equity.snowball.intraday.quad_v2 | grid.bump_config.spot_bump | 0.01 |
| equity.snowball.intraday.quad_v2 | grid.bump_config.time_bump_days | 1 |
| equity.snowball.intraday.quad_v2 | grid.bump_config.time_bump_mode | auto |
| equity.snowball.intraday.quad_v2 | grid.bump_config.vol_bump | 0.01 |
| equity.snowball.intraday.quad_v2 | grid.bump_size | -- |
| equity.snowball.intraday.quad_v2 | grid.bus_days_in_year | 252 |
| equity.snowball.intraday.quad_v2 | grid.cells_per_sd | 2 |
| equity.snowball.intraday.quad_v2 | grid.continuous_steps_per_year | 48 |
| equity.snowball.intraday.quad_v2 | grid.continuous_term_structure | exact |
| equity.snowball.intraday.quad_v2 | grid.domain_sd | 11 |
| equity.snowball.intraday.quad_v2 | grid.event_time_tolerance | 0 |
| equity.snowball.intraday.quad_v2 | grid.order | 8 |
| equity.snowball.intraday.quad_v2 | grid.readout_order | 32 |
| equity.snowball.intraday.quad_v2 | grid.tail_sd | 10 |
| equity.snowball.intraday.quad_v2 | order | 8 |
| equity.snowball.intraday.pde | accuracy | standard |
| equity.snowball.intraday.pde | convergence_axes.placement | achieved points + [1, 2, 3] |
| equity.snowball.intraday.pde | convergence_axes.space | achieved points x (1, 2, 4) |
| equity.snowball.intraday.pde | convergence_axes.time | achieved steps per day x (1, 2, 4) |
| equity.snowball.intraday.pde | desk_bump | 0.01 |
| equity.snowball.intraday.pde | engine | SnowballPDESolver |
| equity.snowball.intraday.pde | grid.bounds | [None, None] |
| equity.snowball.intraday.pde | grid.day_count | 252 |
| equity.snowball.intraday.pde | grid.eps_crit | 0.003 |
| equity.snowball.intraday.pde | grid.event_damping_steps | 2 |
| equity.snowball.intraday.pde | grid.max_points | 2000 |
| equity.snowball.intraday.pde | grid.max_steps | 5000 |
| equity.snowball.intraday.pde | grid.num_std | 4 |
| equity.snowball.intraday.pde | grid.points | 400 |
| equity.snowball.intraday.pde | grid.steps_per_day | 4 |
| equity.snowball.intraday.pde | grid.terminal_damping_steps | 1 |
| (benchmark) | calibration_controls | ['in-ladder calibration one level down, recorded per quantity', 'test_intraday_snowball_gaussian_reference.py: closed-form single-event coverage', 'reference_qualification: independent simulation of every case'] |
| (benchmark) | desk_bump | 0.01 |
| (benchmark) | engine | intraday_gaussian.solve_snowball |
| (benchmark) | exactness_bases.decided_at_valuation | the only remaining instant is the valuation instant: its events are decided at the known spot |
| (benchmark) | exactness_bases.single_instant_quadrature | one instant remains: value and point derivatives are single Gaussian integrals, taken by adaptive Gauss-Kronrod quadrature split at every jump and kink, with no grid |
| (benchmark) | exactness_bases.terminated | no contingent claim remains: the value is the pending receivables, discounted in closed form |
| (benchmark) | ladder_policy.branches.correction | order confirmed as for geometric, extrapolants do not contract; value: Richardson extrapolant; radius: the whole correction |E_n - v_n| |
| (benchmark) | ladder_policy.branches.exact | the last two differences are below the stagnation floor AND the solve names an exactness basis; value: finest level; radius: the larger difference |
| (benchmark) | ladder_policy.branches.geometric | the last two differences share a sign with observed order in [1.5, 2.5] and successive extrapolants contract to 0.5 or less; value: Richardson extrapolant (4 v_n - v_(n-1)) / 3; radius: the last extrapolant spread, never below 1/16.0 of the previous spread |
| (benchmark) | ladder_policy.branches.stagnant | the last two differences are below the stagnation floor and no exactness basis is named: equal values do not establish exactness; radius: infinite |
| (benchmark) | ladder_policy.branches.unbounded | the last difference grew; value: finest level; radius: infinite |
| (benchmark) | ladder_policy.branches.uncalibrated | the same rule applied one level down (without the finest level) did not cover this value with its own radius; radius: infinite |
| (benchmark) | ladder_policy.branches.unextrapolated | order not confirmed (outside the window, or alternating differences) and the last difference did not grow; value: finest level; radius: 3.0 x the previous difference |
| (benchmark) | ladder_policy.calibration | in-ladder: |value - value one level down| <= radius one level down, recorded per quantity; an infinite radius one level down is no evidence and does not calibrate |
| (benchmark) | ladder_policy.parameters.contraction | 0.5 |
| (benchmark) | ladder_policy.parameters.discretization_order | 2 |
| (benchmark) | ladder_policy.parameters.max_contraction | 16 |
| (benchmark) | ladder_policy.parameters.min_levels | 5 |
| (benchmark) | ladder_policy.parameters.order_window | [1.5, 2.5] |
| (benchmark) | ladder_policy.parameters.refinement_ratio | 2 |
| (benchmark) | ladder_policy.parameters.roundoff_factor | 16 |
| (benchmark) | ladder_policy.parameters.stagnation_floor | 1e-13 |
| (benchmark) | ladder_policy.parameters.unconfirmed_safety | 3 |
| (benchmark) | ladder_policy.radius_kind | calibrated_numerical_estimate |
| (benchmark) | ladder_policy.version | 2 |
| (benchmark) | levels | [2001, 4001, 8001, 16001, 32001] |
| (benchmark) | local_grid | (points - 1) / 4 + 1 uniform knots over +/- 15 first-interval standard deviations, nested |
| (benchmark) | method | backward Gaussian transitions of a piecewise-linear value function of ln S with exact barrier jumps; closed-form expectations; an exact last step with analytic first and second derivatives |
| (benchmark) | quick_ladder | False |
| (benchmark) | radius | analytical: the exactness basis's own error; calibrated_estimate: the ladder policy's radius plus the residual components, scaled to the quantity |
| (benchmark) | residual_components.roundoff | estimate: roundoff_factor x eps x value span x instants |
| (benchmark) | residual_components.saturation | analytical bound: 4 x Phi(-9) x value span x instants |
| (benchmark) | residual_components.scaling | pv 1; desk delta 1/(S h); desk gamma 4/(S h)^2; desk theta 2/step; point delta E|Z|/(S sd1); point gamma (E|Z^2-1|/sd1^2 + E|Z|/sd1)/S^2, sd1 the first interval's |
| (benchmark) | residual_components.truncation | analytical bound: value span x reflection-principle mass beyond the flat-extended grids |
| (benchmark) | validity_domain | deterministic Black-Scholes coefficients, discrete monitoring, snowball payoffs without continuous knock-in or Phoenix coupons; every level refines every discretization in play |
| (benchmark) | width_std | 8 |

## Deterministic reference

The reference is a deterministic solve. Its uncertainty is a declared radius, not a standard error: `analytical` where the solve names an exactness basis, otherwise a `calibrated_estimate` from a refinement ladder -- a numerical estimate with calibration evidence, not a proved bound. A radius consumes the budget whole (no interval multiplier) and radii add linearly across cells.

| error model | value |
|---|---|
| calibration_controls | in-ladder calibration one level down, recorded per quantity, test_intraday_snowball_gaussian_reference.py: closed-form single-event coverage, reference_qualification: independent simulation of every case |
| exactness_bases | decided_at_valuation: the only remaining instant is the valuation instant: its events are decided at the known spot; single_instant_quadrature: one instant remains: value and point derivatives are single Gaussian integrals, taken by adaptive Gauss-Kronrod quadrature split at every jump and kink, with no grid; terminated: no contingent claim remains: the value is the pending receivables, discounted in closed form |
| ladder_policy | branches: correction: order confirmed as for geometric, extrapolants do not contract; value: Richardson extrapolant; radius: the whole correction |E_n - v_n|; exact: the last two differences are below the stagnation floor AND the solve names an exactness basis; value: finest level; radius: the larger difference; geometric: the last two differences share a sign with observed order in [1.5, 2.5] and successive extrapolants contract to 0.5 or less; value: Richardson extrapolant (4 v_n - v_(n-1)) / 3; radius: the last extrapolant spread, never below 1/16.0 of the previous spread; stagnant: the last two differences are below the stagnation floor and no exactness basis is named: equal values do not establish exactness; radius: infinite; unbounded: the last difference grew; value: finest level; radius: infinite; uncalibrated: the same rule applied one level down (without the finest level) did not cover this value with its own radius; radius: infinite; unextrapolated: order not confirmed (outside the window, or alternating differences) and the last difference did not grow; value: finest level; radius: 3.0 x the previous difference; calibration: in-ladder: |value - value one level down| <= radius one level down, recorded per quantity; an infinite radius one level down is no evidence and does not calibrate; parameters: contraction: 0.5; discretization_order: 2; max_contraction: 16.0; min_levels: 5; order_window: 1.5, 2.5; refinement_ratio: 2; roundoff_factor: 16.0; stagnation_floor: 1e-13; unconfirmed_safety: 3.0; radius_kind: calibrated_numerical_estimate; version: 2 |
| levels | 2001, 4001, 8001, 16001, 32001 |
| local_grid | (points - 1) / 4 + 1 uniform knots over +/- 15 first-interval standard deviations, nested |
| method | backward Gaussian transitions of a piecewise-linear value function of ln S with exact barrier jumps; closed-form expectations; an exact last step with analytic first and second derivatives |
| radius | analytical: the exactness basis's own error; calibrated_estimate: the ladder policy's radius plus the residual components, scaled to the quantity |
| residual_components | roundoff: estimate: roundoff_factor x eps x value span x instants; saturation: analytical bound: 4 x Phi(-9) x value span x instants; scaling: pv 1; desk delta 1/(S h); desk gamma 4/(S h)^2; desk theta 2/step; point delta E|Z|/(S sd1); point gamma (E|Z^2-1|/sd1^2 + E|Z|/sd1)/S^2, sd1 the first interval's; truncation: analytical bound: value span x reflection-principle mass beyond the flat-extended grids |
| validity_domain | deterministic Black-Scholes coefficients, discrete monitoring, snowball payoffs without continuous knock-in or Phoenix coupons; every level refines every discretization in play |
| width_std | 8.0 |

| case | radii (raw) | undefined here |
|---|---|---|
| already_ki | desk_delta: 8.27e-09 (calibrated_estimate), desk_gamma: 1.23e-09 (calibrated_estimate), desk_theta: 1.93e-10 (calibrated_estimate), point_delta: 8.28e-09 (calibrated_estimate), point_gamma: 1.21e-09 (calibrated_estimate), pv: 3.17e-08 (calibrated_estimate) | none |
| carry | desk_delta: 1.01e-08 (calibrated_estimate), desk_gamma: 2.41e-08 (calibrated_estimate), desk_theta: 4.03e-09 (calibrated_estimate), point_delta: 1.04e-08 (calibrated_estimate), point_gamma: 2.62e-08 (calibrated_estimate), pv: 1.61e-07 (calibrated_estimate) | none |
| high_vol | desk_delta: 3.72e-09 (calibrated_estimate), desk_gamma: 9.24e-09 (calibrated_estimate), desk_theta: 4.55e-09 (calibrated_estimate), point_delta: 4.81e-09 (calibrated_estimate), point_gamma: 9.3e-09 (calibrated_estimate), pv: 3.56e-08 (calibrated_estimate) | none |
| holiday_eve | desk_delta: 1.32e-08 (calibrated_estimate), desk_gamma: 6.07e-07 (calibrated_estimate), desk_theta: 1.18e-07 (calibrated_estimate), point_delta: 7.86e-09 (calibrated_estimate), point_gamma: 2.68e-07 (calibrated_estimate), pv: 1.37e-05 (calibrated_estimate) | none |
| ko_level_on_ki_day | desk_delta: 3.22e-08 (calibrated_estimate), desk_gamma: 8.15e-09 (calibrated_estimate), desk_theta: 1.44e-09 (calibrated_estimate), point_delta: 3.28e-08 (calibrated_estimate), point_gamma: 8.8e-09 (calibrated_estimate), pv: 7.24e-08 (calibrated_estimate) | none |
| low_vol | desk_delta: 2.59e-07 (calibrated_estimate), desk_gamma: 3.68e-07 (calibrated_estimate), desk_theta: 7.11e-08 (calibrated_estimate), point_delta: 4.97e-07 (calibrated_estimate), point_gamma: 1.35e-06 (calibrated_estimate), pv: 3.38e-07 (calibrated_estimate) | none |
| lunch_break | desk_delta: 2.49e-08 (calibrated_estimate), desk_gamma: 5.84e-08 (calibrated_estimate), desk_theta: 7.35e-10 (calibrated_estimate), point_delta: 1.92e-08 (calibrated_estimate), point_gamma: 6.14e-08 (calibrated_estimate), pv: 1.32e-07 (calibrated_estimate) | none |
| lunch_break_zero_variance | desk_delta: 2.48e-08 (calibrated_estimate), desk_gamma: 5.06e-08 (calibrated_estimate), desk_theta: 1.33e-10 (calibrated_estimate), point_delta: 2.2e-08 (calibrated_estimate), point_gamma: 5.56e-08 (calibrated_estimate), pv: 1.29e-07 (calibrated_estimate) | none |
| maturity_day | desk_delta: 4.46e-12 (analytical), desk_gamma: 2.47e-11 (analytical), desk_theta: 7.9e-12 (analytical), point_delta: 2.34e-11 (analytical), point_gamma: 1.32e-11 (analytical), pv: 7.89e-12 (analytical) | none |
| near_ki_10s | desk_delta: 7.72e-08 (calibrated_estimate), desk_gamma: 1.5e-07 (calibrated_estimate), desk_theta: 3.03e-08 (calibrated_estimate), point_delta: 4.1e-08 (calibrated_estimate), point_gamma: 1.28e-07 (calibrated_estimate), pv: 1.26e-07 (calibrated_estimate) | none |
| near_ki_1s | desk_delta: 7.72e-08 (calibrated_estimate), desk_gamma: 1.5e-07 (calibrated_estimate), desk_theta: 2.67e-07 (calibrated_estimate), point_delta: 4.4e-08 (calibrated_estimate), point_gamma: 8.35e-07 (calibrated_estimate), pv: 1.26e-07 (calibrated_estimate) | none |
| near_ki_5m | desk_delta: 7.7e-08 (calibrated_estimate), desk_gamma: 1.36e-07 (calibrated_estimate), desk_theta: 4.99e-08 (calibrated_estimate), point_delta: 1.08e-07 (calibrated_estimate), point_gamma: 1.06e-06 (calibrated_estimate), pv: 1.22e-07 (calibrated_estimate) | none |
| near_ki_above | desk_delta: 6.94e-08 (calibrated_estimate), desk_gamma: 4.77e-08 (calibrated_estimate), desk_theta: 2.68e-08 (calibrated_estimate), point_delta: 9.95e-08 (calibrated_estimate), point_gamma: 9.4e-08 (calibrated_estimate), pv: 9.9e-08 (calibrated_estimate) | none |
| near_ki_below | desk_delta: 6.97e-08 (calibrated_estimate), desk_gamma: 4.04e-08 (calibrated_estimate), desk_theta: 4.12e-08 (calibrated_estimate), point_delta: 1.03e-07 (calibrated_estimate), point_gamma: 7.65e-08 (calibrated_estimate), pv: 6.71e-08 (calibrated_estimate) | none |
| near_ko_on_ko_day | desk_delta: 7.63e-09 (calibrated_estimate), desk_gamma: 1.51e-09 (calibrated_estimate), desk_theta: 4.08e-09 (calibrated_estimate), point_delta: 1.06e-08 (calibrated_estimate), point_gamma: 8.58e-10 (calibrated_estimate), pv: 7.7e-09 (calibrated_estimate) | none |
| near_ko_on_ko_day_1s | desk_delta: 8.02e-09 (calibrated_estimate), desk_gamma: 6.66e-09 (calibrated_estimate), desk_theta: 2.54e-07 (calibrated_estimate), point_delta: 7.48e-09 (calibrated_estimate), point_gamma: 4.07e-07 (calibrated_estimate), pv: 1.17e-08 (calibrated_estimate) | none |
| on_ki_barrier_at_close | desk_delta: 7.66e-08 (calibrated_estimate), desk_gamma: 2.01e-07 (calibrated_estimate), pv: 2.6e-08 (calibrated_estimate) | desk_theta (valuation is at an event instant: no roll or derivative exists inside the segment), point_delta (valuation is at an event instant: no roll or derivative exists inside the segment), point_gamma (valuation is at an event instant: no roll or derivative exists inside the segment) |
| ordinary | desk_delta: 1.04e-08 (calibrated_estimate), desk_gamma: 2.45e-08 (calibrated_estimate), desk_theta: 4.09e-09 (calibrated_estimate), point_delta: 1.06e-08 (calibrated_estimate), point_gamma: 2.66e-08 (calibrated_estimate), pv: 1.64e-07 (calibrated_estimate) | none |
| pre_open | desk_delta: 1.08e-08 (calibrated_estimate), desk_gamma: 1.69e-08 (calibrated_estimate), desk_theta: 4.43e-10 (calibrated_estimate), point_delta: 1.02e-08 (calibrated_estimate), point_gamma: 1.89e-08 (calibrated_estimate), pv: 1.55e-07 (calibrated_estimate) | none |
| provisional_ki_after_close | desk_delta: 2.59e-09 (calibrated_estimate), desk_gamma: 6.49e-10 (calibrated_estimate), desk_theta: 9.64e-11 (calibrated_estimate), point_delta: 2.66e-09 (calibrated_estimate), point_gamma: 3.22e-10 (calibrated_estimate), pv: 1.1e-05 (calibrated_estimate) | none |
| sessions_only_profile | desk_delta: 6.49e-08 (calibrated_estimate), desk_gamma: 3.35e-08 (calibrated_estimate), desk_theta: 2.97e-08 (calibrated_estimate), point_delta: 8.71e-08 (calibrated_estimate), point_gamma: 5.63e-08 (calibrated_estimate), pv: 9.61e-08 (calibrated_estimate) | none |
| terminated_pending_cash | desk_delta: 4.49e-15 (analytical), desk_gamma: 1.8e-14 (analytical), desk_theta: 8.99e-15 (analytical), point_delta: 0 (analytical), point_gamma: 0 (analytical), pv: 4.49e-15 (analytical) | none |
| uniform_profile | desk_delta: 7.66e-08 (calibrated_estimate), desk_gamma: 1.01e-07 (calibrated_estimate), desk_theta: 1.42e-08 (calibrated_estimate), point_delta: 1.41e-07 (calibrated_estimate), point_gamma: 5.55e-07 (calibrated_estimate), pv: 1.12e-07 (calibrated_estimate) | none |

## Reference qualification

An independent stochastic arm simulates every case. The deterministic value must sit within 4 of its standard errors plus the reference's radius; a case that does not is not qualified and its cells are unresolved.

| case | quantity | reference - qualifier | qualifier SE | z | within |
|---|---|---|---|---|---|
| already_ki | desk_delta | -8.01e-05 | 0.00172 | -0.0465 | yes |
| already_ki | desk_gamma | 0.000443 | 0.00441 | 0.1 | yes |
| already_ki | desk_theta | 0.000821 | 0.00367 | 0.224 | yes |
| already_ki | pv | 0.000717 | 0.00295 | 0.243 | yes |
| carry | desk_delta | -0.000365 | 0.00161 | -0.227 | yes |
| carry | desk_gamma | 0.00452 | 0.00504 | 0.896 | yes |
| carry | desk_theta | 0.00261 | 0.00478 | 0.547 | yes |
| carry | pv | -0.00153 | 0.00351 | -0.437 | yes |
| high_vol | desk_delta | -0.00189 | 0.0023 | -0.822 | yes |
| high_vol | desk_gamma | -0.00626 | 0.00663 | -0.945 | yes |
| high_vol | desk_theta | 0.00197 | 0.0046 | 0.427 | yes |
| high_vol | pv | 0.00091 | 0.003 | 0.304 | yes |
| holiday_eve | desk_delta | -0.00498 | 0.00483 | -1.03 | yes |
| holiday_eve | desk_gamma | 0.0207 | 0.0123 | 1.68 | yes |
| holiday_eve | desk_theta | 0.00858 | 0.00714 | 1.2 | yes |
| holiday_eve | pv | -0.00513 | 0.00388 | -1.32 | yes |
| ko_level_on_ki_day | desk_delta | 0.000499 | 0.00147 | 0.339 | yes |
| ko_level_on_ki_day | desk_gamma | 0.0042 | 0.00294 | 1.43 | yes |
| ko_level_on_ki_day | desk_theta | 0.00213 | 0.00363 | 0.587 | yes |
| ko_level_on_ki_day | pv | -0.000847 | 0.0024 | -0.352 | yes |
| low_vol | desk_delta | 0.00456 | 0.00371 | 1.23 | yes |
| low_vol | desk_gamma | 0.0149 | 0.0182 | 0.823 | yes |
| low_vol | desk_theta | -0.00473 | 0.00628 | -0.753 | yes |
| low_vol | pv | -0.000754 | 0.00496 | -0.152 | yes |
| lunch_break | desk_delta | -0.00229 | 0.00355 | -0.644 | yes |
| lunch_break | desk_gamma | 0.0279 | 0.0116 | 2.41 | yes |
| lunch_break | desk_theta | 0.00188 | 0.000702 | 2.68 | yes |
| lunch_break | pv | -0.0034 | 0.00392 | -0.867 | yes |
| lunch_break_zero_variance | desk_delta | -0.00322 | 0.00453 | -0.711 | yes |
| lunch_break_zero_variance | desk_gamma | 0.0114 | 0.0145 | 0.784 | yes |
| lunch_break_zero_variance | desk_theta | -0.000149 | 0.00113 | -0.132 | yes |
| lunch_break_zero_variance | pv | -0.00664 | 0.00465 | -1.43 | yes |
| maturity_day | desk_delta | 8.08e-14 | 0 | -- | yes |
| maturity_day | desk_gamma | 1.23e-12 | 0 | -- | yes |
| maturity_day | desk_theta | 3.57e-13 | 0 | -- | yes |
| maturity_day | pv | -3.57e-13 | 0 | -- | yes |
| near_ki_10s | desk_delta | 0.00253 | 0.0029 | 0.874 | yes |
| near_ki_10s | desk_gamma | -0.00505 | 0.0105 | -0.481 | yes |
| near_ki_10s | desk_theta | 2.07 | 1.57 | 1.32 | yes |
| near_ki_10s | pv | 0.00322 | 0.00346 | 0.931 | yes |
| near_ki_1s | desk_delta | 0.0047 | 0.00285 | 1.65 | yes |
| near_ki_1s | desk_gamma | 0.00942 | 0.00982 | 0.96 | yes |
| near_ki_1s | desk_theta | 14 | 16.9 | 0.831 | yes |
| near_ki_1s | pv | 0.00139 | 0.00277 | 0.502 | yes |
| near_ki_5m | desk_delta | -3.18e-05 | 0.00322 | -0.00987 | yes |
| near_ki_5m | desk_gamma | -0.00456 | 0.0136 | -0.337 | yes |
| near_ki_5m | desk_theta | -0.0235 | 0.0615 | -0.382 | yes |
| near_ki_5m | pv | 0.000654 | 0.00326 | 0.2 | yes |
| near_ki_above | desk_delta | 0.00373 | 0.00289 | 1.29 | yes |
| near_ki_above | desk_gamma | -0.00149 | 0.0124 | -0.121 | yes |
| near_ki_above | desk_theta | -0.012 | 0.00505 | -2.37 | yes |
| near_ki_above | pv | 0.00375 | 0.00359 | 1.04 | yes |
| near_ki_below | desk_delta | 0.00485 | 0.00243 | 1.99 | yes |
| near_ki_below | desk_gamma | -0.00283 | 0.00922 | -0.307 | yes |
| near_ki_below | desk_theta | -0.00528 | 0.00222 | -2.37 | yes |
| near_ki_below | pv | 0.00547 | 0.00217 | 2.52 | yes |
| near_ko_on_ko_day | desk_delta | 0.000269 | 0.00152 | 0.177 | yes |
| near_ko_on_ko_day | desk_gamma | 0.00323 | 0.00424 | 0.762 | yes |
| near_ko_on_ko_day | desk_theta | 0.00666 | 0.00341 | 1.96 | yes |
| near_ko_on_ko_day | pv | -0.00203 | 0.00238 | -0.853 | yes |
| near_ko_on_ko_day_1s | desk_delta | -0.000637 | 0.00117 | -0.547 | yes |
| near_ko_on_ko_day_1s | desk_gamma | 0.00105 | 0.00374 | 0.281 | yes |
| near_ko_on_ko_day_1s | desk_theta | 4.33 | 12.7 | 0.341 | yes |
| near_ko_on_ko_day_1s | pv | 9.94e-05 | 0.00227 | 0.0437 | yes |
| on_ki_barrier_at_close | desk_delta | -0.00139 | 0.00343 | -0.406 | yes |
| on_ki_barrier_at_close | desk_gamma | -0.00359 | 0.00899 | -0.399 | yes |
| on_ki_barrier_at_close | pv | 0.000897 | 0.000678 | 1.32 | yes |
| ordinary | desk_delta | 0.00286 | 0.00189 | 1.51 | yes |
| ordinary | desk_gamma | 0.00614 | 0.00417 | 1.47 | yes |
| ordinary | desk_theta | -0.000128 | 0.00407 | -0.0314 | yes |
| ordinary | pv | -0.00381 | 0.00313 | -1.22 | yes |
| pre_open | desk_delta | -0.000406 | 0.00187 | -0.217 | yes |
| pre_open | desk_gamma | 0.000623 | 0.00379 | 0.164 | yes |
| pre_open | desk_theta | 0.000293 | 0.000693 | 0.423 | yes |
| pre_open | pv | 0.000431 | 0.00324 | 0.133 | yes |
| provisional_ki_after_close | desk_delta | -0.000132 | 0.000387 | -0.342 | yes |
| provisional_ki_after_close | desk_gamma | -0.000659 | 0.00155 | -0.425 | yes |
| provisional_ki_after_close | desk_theta | -2.4e-06 | 6.59e-05 | -0.0364 | yes |
| provisional_ki_after_close | pv | 0.000452 | 0.000689 | 0.657 | yes |
| sessions_only_profile | desk_delta | -0.0066 | 0.00262 | -2.52 | yes |
| sessions_only_profile | desk_gamma | -0.000181 | 0.00965 | -0.0188 | yes |
| sessions_only_profile | desk_theta | 0.00433 | 0.00491 | 0.882 | yes |
| sessions_only_profile | pv | -0.00321 | 0.00271 | -1.18 | yes |
| terminated_pending_cash | desk_delta | 0 | 0 | -- | yes |
| terminated_pending_cash | desk_gamma | 0 | 0 | -- | yes |
| terminated_pending_cash | desk_theta | 0 | 0 | -- | yes |
| terminated_pending_cash | pv | 0 | 0 | -- | yes |
| uniform_profile | desk_delta | -0.00682 | 0.00258 | -2.64 | yes |
| uniform_profile | desk_gamma | -0.00763 | 0.0104 | -0.734 | yes |
| uniform_profile | desk_theta | -0.00247 | 0.00486 | -0.508 | yes |
| uniform_profile | pv | -0.00295 | 0.00293 | -1.01 | yes |

## Qualifying arm sampling

| case | batches | stopped because | standard errors (raw) |
|---|---|---|---|
| already_ki | 32 | max_batches | desk_delta: 0.00172, desk_gamma: 0.00441, desk_theta: 0.00367, point_delta: 0.00172, point_gamma: 0.00441, pv: 0.00295 |
| carry | 32 | max_batches | desk_delta: 0.00161, desk_gamma: 0.00504, desk_theta: 0.00478, point_delta: 0.00161, point_gamma: 0.00504, pv: 0.00351 |
| high_vol | 32 | max_batches | desk_delta: 0.0023, desk_gamma: 0.00663, desk_theta: 0.0046, point_delta: 0.0023, point_gamma: 0.00663, pv: 0.003 |
| holiday_eve | 32 | max_batches | desk_delta: 0.00483, desk_gamma: 0.0123, desk_theta: 0.00714, point_delta: 0.00483, point_gamma: 0.0123, pv: 0.00388 |
| ko_level_on_ki_day | 32 | max_batches | desk_delta: 0.00147, desk_gamma: 0.00294, desk_theta: 0.00363, point_delta: 0.00147, point_gamma: 0.00294, pv: 0.0024 |
| low_vol | 32 | max_batches | desk_delta: 0.00371, desk_gamma: 0.0182, desk_theta: 0.00628, point_delta: 0.00371, point_gamma: 0.0182, pv: 0.00496 |
| lunch_break | 32 | max_batches | desk_delta: 0.00355, desk_gamma: 0.0116, desk_theta: 0.000702, point_delta: 0.00355, point_gamma: 0.0116, pv: 0.00392 |
| lunch_break_zero_variance | 32 | max_batches | desk_delta: 0.00453, desk_gamma: 0.0145, desk_theta: 0.00113, point_delta: 0.00453, point_gamma: 0.0145, pv: 0.00465 |
| maturity_day | 32 | se_budget_met | desk_delta: 0, desk_gamma: 0, desk_theta: 0, point_delta: 0, point_gamma: 0, pv: 0 |
| near_ki_10s | 32 | max_batches | desk_delta: 0.0029, desk_gamma: 0.0105, desk_theta: 1.57, point_delta: 0.0029, point_gamma: 0.0105, pv: 0.00346 |
| near_ki_1s | 32 | max_batches | desk_delta: 0.00285, desk_gamma: 0.00982, desk_theta: 16.9, point_delta: 0.00285, point_gamma: 0.00982, pv: 0.00277 |
| near_ki_5m | 32 | max_batches | desk_delta: 0.00322, desk_gamma: 0.0136, desk_theta: 0.0615, point_delta: 0.00322, point_gamma: 0.0136, pv: 0.00326 |
| near_ki_above | 32 | max_batches | desk_delta: 0.00289, desk_gamma: 0.0124, desk_theta: 0.00505, point_delta: 0.00289, point_gamma: 0.0124, pv: 0.00359 |
| near_ki_below | 32 | max_batches | desk_delta: 0.00243, desk_gamma: 0.00922, desk_theta: 0.00222, point_delta: 0.00243, point_gamma: 0.00922, pv: 0.00217 |
| near_ko_on_ko_day | 32 | max_batches | desk_delta: 0.00152, desk_gamma: 0.00424, desk_theta: 0.00341, point_delta: 0.00152, point_gamma: 0.00424, pv: 0.00238 |
| near_ko_on_ko_day_1s | 32 | max_batches | desk_delta: 0.00117, desk_gamma: 0.00374, desk_theta: 12.7, point_delta: 0.00117, point_gamma: 0.00374, pv: 0.00227 |
| on_ki_barrier_at_close | 32 | max_batches | desk_delta: 0.00343, desk_gamma: 0.00899, desk_theta: 0, point_delta: 0.00343, point_gamma: 0.00899, pv: 0.000678 |
| ordinary | 32 | max_batches | desk_delta: 0.00189, desk_gamma: 0.00417, desk_theta: 0.00407, point_delta: 0.00189, point_gamma: 0.00417, pv: 0.00313 |
| pre_open | 32 | max_batches | desk_delta: 0.00187, desk_gamma: 0.00379, desk_theta: 0.000693, point_delta: 0.00187, point_gamma: 0.00379, pv: 0.00324 |
| provisional_ki_after_close | 32 | max_batches | desk_delta: 0.000387, desk_gamma: 0.00155, desk_theta: 6.59e-05, point_delta: 0.000387, point_gamma: 0.00155, pv: 0.000689 |
| sessions_only_profile | 32 | max_batches | desk_delta: 0.00262, desk_gamma: 0.00965, desk_theta: 0.00491, point_delta: 0.00262, point_gamma: 0.00965, pv: 0.00271 |
| terminated_pending_cash | 32 | se_budget_met | desk_delta: 0, desk_gamma: 0, desk_theta: 0, point_delta: 0, point_gamma: 0, pv: 0 |
| uniform_profile | 32 | max_batches | desk_delta: 0.00258, desk_gamma: 0.0104, desk_theta: 0.00486, point_delta: 0.00258, point_gamma: 0.0104, pv: 0.00293 |

Sampling policy: 32768 paths/batch, 32-32 batches, seed 20260918, bump 0.01.

## Cells

| candidate | case | quantity | reference | radius | candidate | err (c) | interval (c) | envelope (c) | verdict |
|---|---|---|---|---|---|---|---|---|---|
| equity.snowball.intraday.quad_v2 | ordinary | pv | 7.97428 | 1.64e-07 | 7.97428 | 0.000268 | 0.001908 | 3.988e-09 | PASS |
| equity.snowball.intraday.quad_v2 | ordinary | desk_delta | -0.164795 | 1.04e-08 | -0.164795 | -6.291e-05 | 0.0006913 | 2.425e-09 | PASS |
| equity.snowball.intraday.quad_v2 | ordinary | desk_gamma | -0.09634 | 2.45e-08 | -0.09634 | -5.139e-05 | 0.0003058 | 2.581e-10 | PASS |
| equity.snowball.intraday.quad_v2 | ordinary | desk_theta | 0.014282 | 4.09e-09 | 0.014282 | 7.785e-06 | 4.871e-05 | 1.252e-09 | PASS |
| equity.snowball.intraday.quad_v2 | ordinary | point_delta | -0.16652 | 1.06e-08 | -0.16652 | -6.353e-05 | 0.0006982 | 2.529e-09 | PASS |
| equity.snowball.intraday.quad_v2 | ordinary | point_gamma | -0.0977908 | 2.66e-08 | -0.0977908 | -5.546e-05 | 0.0003273 | 2.986e-10 | PASS |
| equity.snowball.intraday.quad_v2 | near_ki_above | pv | -22.4213 | 9.9e-08 | -22.4213 | -0.0003889 | 0.001379 | 1.023e-08 | PASS |
| equity.snowball.intraday.quad_v2 | near_ki_above | desk_delta | 2.61027 | 6.94e-08 | 2.61027 | -0.0001024 | 0.0003683 | 6.975e-10 | PASS |
| equity.snowball.intraday.quad_v2 | near_ki_above | desk_gamma | 0.516564 | 4.77e-08 | 0.516564 | 3.032e-05 | 0.0001226 | 3.77e-10 | PASS |
| equity.snowball.intraday.quad_v2 | near_ki_above | desk_theta | 0.224767 | 2.68e-08 | 0.224767 | -0.0001021 | 0.0003705 | 4.648e-07 | PASS |
| equity.snowball.intraday.quad_v2 | near_ki_above | point_delta | 2.99352 | 9.95e-08 | 2.99352 | -0.0001278 | 0.0004602 | 7.492e-10 | PASS |
| equity.snowball.intraday.quad_v2 | near_ki_above | point_gamma | 0.120793 | 9.4e-08 | 0.120793 | 0.0002703 | 0.001049 | 2.266e-09 | PASS |
| equity.snowball.intraday.quad_v2 | near_ki_below | pv | -23.2848 | 6.71e-08 | -23.2848 | -0.0002703 | 0.0009411 | 1.666e-08 | PASS |
| equity.snowball.intraday.quad_v2 | near_ki_below | desk_delta | 2.34313 | 6.97e-08 | 2.34313 | -0.0001132 | 0.0004106 | 2.498e-09 | PASS |
| equity.snowball.intraday.quad_v2 | near_ki_below | desk_gamma | 1.45744 | 4.04e-08 | 1.45744 | -1.089e-05 | 3.86e-05 | 7.255e-10 | PASS |
| equity.snowball.intraday.quad_v2 | near_ki_below | desk_theta | -0.613422 | 4.12e-08 | -0.613422 | 0.0001551 | 0.0005672 | 7.887e-09 | PASS |
| equity.snowball.intraday.quad_v2 | near_ki_below | point_delta | 2.66348 | 1.03e-07 | 2.66348 | -0.000146 | 0.0005325 | 4.425e-09 | PASS |
| equity.snowball.intraday.quad_v2 | near_ki_below | point_gamma | 2.05773 | 7.65e-08 | 2.05773 | -1.443e-05 | 5.163e-05 | 9.643e-10 | PASS |
| equity.snowball.intraday.quad_v2 | near_ki_1s | pv | -22.1965 | 1.26e-07 | -22.1965 | -0.000491 | 0.001749 | 1.036e-07 | PASS |
| equity.snowball.intraday.quad_v2 | near_ki_1s | desk_delta | 2.68629 | 7.72e-08 | 2.68629 | -0.0001101 | 0.0003974 | 1.237e-07 | PASS |
| equity.snowball.intraday.quad_v2 | near_ki_1s | desk_gamma | -0.48681 | 1.5e-07 | -0.48681 | 0.0001127 | 0.0004211 | 2.624e-07 | PASS |
| equity.snowball.intraday.quad_v2 | near_ki_1s | desk_theta | -0.0436783 | 2.67e-07 | -0.0436783 | -2.575e-05 | 0.0027 | 0.001262 | PASS |
| equity.snowball.intraday.quad_v2 | near_ki_1s | point_delta | 2.36023 | 4.4e-08 | 2.36023 | -6.55e-05 | 0.0002518 | 3.633e-06 | PASS |
| equity.snowball.intraday.quad_v2 | near_ki_1s | point_gamma | 0.532522 | 8.35e-07 | 0.532522 | 4.081e-05 | 0.00161 | 0.0005987 | PASS |
| equity.snowball.intraday.quad_v2 | near_ki_10s | pv | -22.1964 | 1.26e-07 | -22.1964 | -0.000491 | 0.001748 | 5.524e-08 | PASS |
| equity.snowball.intraday.quad_v2 | near_ki_10s | desk_delta | 2.6863 | 7.72e-08 | 2.6863 | -0.0001101 | 0.0003974 | 4.097e-08 | PASS |
| equity.snowball.intraday.quad_v2 | near_ki_10s | desk_gamma | -0.487153 | 1.5e-07 | -0.487153 | 0.0001126 | 0.0004207 | 1.06e-07 | PASS |
| equity.snowball.intraday.quad_v2 | near_ki_10s | desk_theta | -0.0436754 | 3.03e-08 | -0.0436754 | -1.724e-05 | 0.0003199 | 0.0001443 | PASS |
| equity.snowball.intraday.quad_v2 | near_ki_10s | point_delta | 2.36012 | 4.1e-08 | 2.36012 | -6.546e-05 | 0.0002391 | 2.529e-06 | PASS |
| equity.snowball.intraday.quad_v2 | near_ki_10s | point_gamma | 0.532451 | 1.28e-07 | 0.532451 | 3.877e-05 | 0.0002791 | 2.49e-05 | PASS |
| equity.snowball.intraday.quad_v2 | near_ki_5m | pv | -22.239 | 1.22e-07 | -22.239 | -0.0004757 | 0.001692 | 1.041e-08 | PASS |
| equity.snowball.intraday.quad_v2 | near_ki_5m | desk_delta | 2.68679 | 7.7e-08 | 2.68679 | -0.0001099 | 0.0003966 | 6.694e-10 | PASS |
| equity.snowball.intraday.quad_v2 | near_ki_5m | desk_gamma | -0.335102 | 1.36e-07 | -0.335102 | 0.0001481 | 0.0005543 | 5.25e-10 | PASS |
| equity.snowball.intraday.quad_v2 | near_ki_5m | desk_theta | 0.509623 | 4.99e-08 | 0.509623 | -0.0001837 | 0.0006827 | 5.555e-06 | PASS |
| equity.snowball.intraday.quad_v2 | near_ki_5m | point_delta | 3.19392 | 1.08e-07 | 3.19392 | -0.0001288 | 0.0004678 | 3.049e-09 | PASS |
| equity.snowball.intraday.quad_v2 | near_ki_5m | point_gamma | -11.7084 | 1.06e-06 | -11.7084 | 3.382e-05 | 0.0001244 | 4.808e-10 | PASS |
| equity.snowball.intraday.quad_v2 | near_ko_on_ko_day | pv | 7.22688 | 7.7e-09 | 7.22688 | 0.0001176 | 0.0001946 | 1.954e-09 | PASS |
| equity.snowball.intraday.quad_v2 | near_ko_on_ko_day | desk_delta | -0.972193 | 7.63e-09 | -0.972193 | -0.0001082 | 0.0001868 | 1.625e-09 | PASS |
| equity.snowball.intraday.quad_v2 | near_ko_on_ko_day | desk_gamma | -0.154872 | 1.51e-09 | -0.154872 | -8.896e-08 | 9.867e-06 | 2.165e-10 | PASS |
| equity.snowball.intraday.quad_v2 | near_ko_on_ko_day | desk_theta | 0.828925 | 4.08e-09 | 0.828925 | 7.329e-05 | 0.0001141 | 1.03e-09 | PASS |
| equity.snowball.intraday.quad_v2 | near_ko_on_ko_day | point_delta | -1.49252 | 1.06e-08 | -1.49252 | -0.0001037 | 0.000175 | 1.645e-09 | PASS |
| equity.snowball.intraday.quad_v2 | near_ko_on_ko_day | point_gamma | -0.370089 | 8.58e-10 | -0.370089 | -4.425e-06 | 6.743e-06 | 1.849e-10 | PASS |
| equity.snowball.intraday.quad_v2 | near_ko_on_ko_day_1s | pv | 8.0558 | 1.17e-08 | 8.0558 | 0.0001909 | 0.000308 | 3.233e-09 | PASS |
| equity.snowball.intraday.quad_v2 | near_ko_on_ko_day_1s | desk_delta | -1.03888 | 8.02e-09 | -1.03888 | -0.0001075 | 0.0001848 | 1.488e-09 | PASS |
| equity.snowball.intraday.quad_v2 | near_ko_on_ko_day_1s | desk_gamma | -1.77076 | 6.66e-09 | -1.77076 | -8.096e-06 | 1.186e-05 | 1.753e-10 | PASS |
| equity.snowball.intraday.quad_v2 | near_ko_on_ko_day_1s | desk_theta | 0.00175132 | 2.54e-07 | 0.00175132 | -2.174e-06 | 0.002545 | 4.924e-06 | PASS |
| equity.snowball.intraday.quad_v2 | near_ko_on_ko_day_1s | point_delta | -0.135401 | 7.48e-09 | -0.135401 | -0.0002773 | 0.0008296 | 3.198e-08 | PASS |
| equity.snowball.intraday.quad_v2 | near_ko_on_ko_day_1s | point_gamma | -0.0111397 | 4.07e-07 | -0.0111397 | 1.634e-05 | 0.03655 | 1.129e-06 | PASS |
| equity.snowball.intraday.quad_v2 | ko_level_on_ki_day | pv | 7.17628 | 7.24e-08 | 7.17628 | 0.0001126 | 0.0008364 | 2.114e-09 | PASS |
| equity.snowball.intraday.quad_v2 | ko_level_on_ki_day | desk_delta | -0.325387 | 3.22e-08 | -0.325387 | -0.000166 | 0.001154 | 1.153e-09 | PASS |
| equity.snowball.intraday.quad_v2 | ko_level_on_ki_day | desk_gamma | 0.00215328 | 8.15e-09 | 0.00215328 | 0.0008019 | 0.004588 | 1.208e-08 | PASS |
| equity.snowball.intraday.quad_v2 | ko_level_on_ki_day | desk_theta | -0.00032688 | 1.44e-09 | -0.00032688 | -3.014e-06 | 1.741e-05 | 2.451e-09 | PASS |
| equity.snowball.intraday.quad_v2 | ko_level_on_ki_day | point_delta | -0.333597 | 3.28e-08 | -0.333597 | -0.0001652 | 0.001147 | 1.213e-09 | PASS |
| equity.snowball.intraday.quad_v2 | ko_level_on_ki_day | point_gamma | 0.00259306 | 8.8e-09 | 0.00259306 | 0.0007327 | 0.004128 | 8.41e-09 | PASS |
| equity.snowball.intraday.quad_v2 | pre_open | pv | 7.92926 | 1.55e-07 | 7.92926 | 0.0002499 | 0.001795 | 3.757e-09 | PASS |
| equity.snowball.intraday.quad_v2 | pre_open | desk_delta | -0.159805 | 1.08e-08 | -0.159805 | -5.352e-05 | 0.0007291 | 2.14e-09 | PASS |
| equity.snowball.intraday.quad_v2 | pre_open | desk_gamma | -0.0887234 | 1.69e-08 | -0.0887234 | -4.346e-05 | 0.0002338 | 2.002e-10 | PASS |
| equity.snowball.intraday.quad_v2 | pre_open | desk_theta | 0.00106058 | 4.43e-10 | 0.00106058 | 7.943e-08 | 4.512e-06 | 2.185e-09 | PASS |
| equity.snowball.intraday.quad_v2 | pre_open | point_delta | -0.161987 | 1.02e-08 | -0.161987 | -6.255e-05 | 0.0006941 | 2.209e-09 | PASS |
| equity.snowball.intraday.quad_v2 | pre_open | point_gamma | -0.0899154 | 1.89e-08 | -0.0899154 | -4.379e-05 | 0.0002541 | 2.121e-10 | PASS |
| equity.snowball.intraday.quad_v2 | lunch_break | pv | -20.2071 | 1.32e-07 | -20.2071 | -0.0005264 | 0.00185 | 1.837e-08 | PASS |
| equity.snowball.intraday.quad_v2 | lunch_break | desk_delta | 2.66292 | 2.49e-08 | 2.66292 | -4.116e-05 | 0.0001348 | 4.569e-10 | PASS |
| equity.snowball.intraday.quad_v2 | lunch_break | desk_gamma | -0.137078 | 5.84e-08 | -0.137078 | 0.0001502 | 0.0005765 | 1.256e-09 | PASS |
| equity.snowball.intraday.quad_v2 | lunch_break | desk_theta | 0.00279031 | 7.35e-10 | 0.00279031 | -3.535e-06 | 1.089e-05 | 1.492e-09 | PASS |
| equity.snowball.intraday.quad_v2 | lunch_break | point_delta | 2.66675 | 1.92e-08 | 2.66675 | -3.406e-05 | 0.000106 | 7.061e-10 | PASS |
| equity.snowball.intraday.quad_v2 | lunch_break | point_gamma | -0.201527 | 6.14e-08 | -0.201527 | 0.0001109 | 0.0004157 | 1.102e-09 | PASS |
| equity.snowball.intraday.quad_v2 | lunch_break_zero_variance | pv | -20.2163 | 1.29e-07 | -20.2163 | -0.0005158 | 0.001808 | 1.41e-08 | PASS |
| equity.snowball.intraday.quad_v2 | lunch_break_zero_variance | desk_delta | 2.64135 | 2.48e-08 | 2.64135 | -4.124e-05 | 0.0001351 | 5.296e-10 | PASS |
| equity.snowball.intraday.quad_v2 | lunch_break_zero_variance | desk_gamma | -0.0645224 | 5.06e-08 | -0.0645223 | 0.0002777 | 0.001061 | 3.623e-09 | PASS |
| equity.snowball.intraday.quad_v2 | lunch_break_zero_variance | desk_theta | -0.000531722 | 1.33e-10 | -0.000531722 | -1.392e-07 | 1.474e-06 | 1.101e-09 | PASS |
| equity.snowball.intraday.quad_v2 | lunch_break_zero_variance | point_delta | 2.66537 | 2.2e-08 | 2.66537 | -3.716e-05 | 0.0001195 | 4.749e-10 | PASS |
| equity.snowball.intraday.quad_v2 | lunch_break_zero_variance | point_gamma | -0.127976 | 5.56e-08 | -0.127976 | 0.0001553 | 0.0005898 | 1.156e-09 | PASS |
| equity.snowball.intraday.quad_v2 | holiday_eve | pv | -20.0159 | 1.37e-05 | -20.0159 | -0.0002563 | 0.1376 | 2.416e-08 | PASS |
| equity.snowball.intraday.quad_v2 | holiday_eve | desk_delta | 2.8658 | 1.32e-08 | 2.8658 | -1.649e-05 | 6.247e-05 | 9.933e-10 | PASS |
| equity.snowball.intraday.quad_v2 | holiday_eve | desk_gamma | -0.304062 | 6.07e-07 | -0.304062 | 3.962e-05 | 0.002036 | 2.792e-09 | PASS |
| equity.snowball.intraday.quad_v2 | holiday_eve | desk_theta | -0.00170456 | 1.18e-07 | -0.00170456 | -6.188e-06 | 0.001188 | 1.451e-06 | PASS |
| equity.snowball.intraday.quad_v2 | holiday_eve | point_delta | 2.7757 | 7.86e-09 | 2.7757 | -8.077e-06 | 3.64e-05 | 6.912e-10 | PASS |
| equity.snowball.intraday.quad_v2 | holiday_eve | point_gamma | -0.181555 | 2.68e-07 | -0.181555 | 5.364e-05 | 0.001531 | 7.04e-09 | PASS |
| equity.snowball.intraday.quad_v2 | maturity_day | pv | 12 | 7.89e-12 | 12 | 3.588e-09 | 8.251e-08 | 0 | PASS |
| equity.snowball.intraday.quad_v2 | maturity_day | desk_delta | 8.08242e-14 | 4.46e-12 | 0 | -8.082e-09 | 4.539e-07 | 0 | PASS |
| equity.snowball.intraday.quad_v2 | maturity_day | desk_gamma | 1.23102e-12 | 2.47e-11 | 0 | -1.231e-06 | 2.593e-05 | 0 | PASS |
| equity.snowball.intraday.quad_v2 | maturity_day | desk_theta | 4.10958e-05 | 7.9e-12 | 4.10958e-05 | -3.588e-09 | 8.261e-08 | 0 | PASS |
| equity.snowball.intraday.quad_v2 | maturity_day | point_delta | 1.35417e-14 | 2.34e-11 | 2.17371e-15 | -1.137e-09 | 2.343e-06 | 0 | PASS |
| equity.snowball.intraday.quad_v2 | maturity_day | point_gamma | 1.48711e-12 | 1.32e-11 | -1.96332e-17 | -1.487e-06 | 1.466e-05 | 0 | PASS |
| equity.snowball.intraday.quad_v2 | already_ki | pv | -8.26798 | 3.17e-08 | -8.26798 | 0.0001942 | 0.0005111 | 7.141e-09 | PASS |
| equity.snowball.intraday.quad_v2 | already_ki | desk_delta | 1.04517 | 8.27e-09 | 1.04517 | 3.567e-05 | 0.0001148 | 6.82e-10 | PASS |
| equity.snowball.intraday.quad_v2 | already_ki | desk_gamma | -0.00409272 | 1.23e-09 | -0.00409272 | 8.425e-05 | 0.0003858 | 1.232e-08 | PASS |
| equity.snowball.intraday.quad_v2 | already_ki | desk_theta | 0.000226998 | 1.93e-10 | 0.000226998 | -4.081e-07 | 2.336e-06 | 3.73e-10 | PASS |
| equity.snowball.intraday.quad_v2 | already_ki | point_delta | 1.04544 | 8.28e-09 | 1.04544 | 3.563e-05 | 0.0001148 | 6.436e-10 | PASS |
| equity.snowball.intraday.quad_v2 | already_ki | point_gamma | -0.00407064 | 1.21e-09 | -0.00407064 | 8.468e-05 | 0.0003808 | 6.947e-09 | PASS |
| equity.snowball.intraday.quad_v2 | terminated_pending_cash | pv | 5.0605 | 4.49e-15 | 5.0605 | 0 | 4.495e-11 | 0 | PASS |
| equity.snowball.intraday.quad_v2 | terminated_pending_cash | desk_delta | 0 | 4.49e-15 | 0 | 0 | 4.495e-10 | 0 | PASS |
| equity.snowball.intraday.quad_v2 | terminated_pending_cash | desk_gamma | 0 | 1.8e-14 | 0 | 0 | 1.798e-08 | 0 | PASS |
| equity.snowball.intraday.quad_v2 | terminated_pending_cash | desk_theta | 1.73305e-05 | 8.99e-15 | 1.73305e-05 | 0 | 8.989e-11 | 0 | PASS |
| equity.snowball.intraday.quad_v2 | terminated_pending_cash | point_delta | 0 | 0 | 0 | 0 | 0 | 0 | PASS |
| equity.snowball.intraday.quad_v2 | terminated_pending_cash | point_gamma | 0 | 0 | 0 | 0 | 0 | 0 | PASS |
| equity.snowball.intraday.quad_v2 | provisional_ki_after_close | pv | -23.8982 | 1.1e-05 | -23.8982 | -6.74e-05 | 0.1097 | 1.073e-08 | PASS |
| equity.snowball.intraday.quad_v2 | provisional_ki_after_close | desk_delta | 1.00962 | 2.59e-09 | 1.00962 | 1.406e-06 | 2.702e-05 | 1.126e-09 | PASS |
| equity.snowball.intraday.quad_v2 | provisional_ki_after_close | desk_gamma | 0.00288312 | 6.49e-10 | 0.00288312 | 0.00022 | 0.000445 | 2.636e-08 | PASS |
| equity.snowball.intraday.quad_v2 | provisional_ki_after_close | desk_theta | -0.000272405 | 9.64e-11 | -0.000272405 | -1.532e-07 | 1.117e-06 | 1.03e-09 | PASS |
| equity.snowball.intraday.quad_v2 | provisional_ki_after_close | point_delta | 1.00959 | 2.66e-09 | 1.00959 | 1.569e-06 | 2.787e-05 | 1.203e-09 | PASS |
| equity.snowball.intraday.quad_v2 | provisional_ki_after_close | point_gamma | 0.00288347 | 3.22e-10 | 0.00288347 | 0.0002452 | 0.0003567 | 3.626e-09 | PASS |
| equity.snowball.intraday.quad_v2 | low_vol | pv | -21.6485 | 3.38e-07 | -21.6485 | -0.0002333 | 0.003612 | 3.379e-08 | PASS |
| equity.snowball.intraday.quad_v2 | low_vol | desk_delta | 4.25292 | 2.59e-07 | 4.25292 | -4.499e-05 | 0.0006547 | 2.7e-09 | PASS |
| equity.snowball.intraday.quad_v2 | low_vol | desk_gamma | 1.62662 | 3.68e-07 | 1.62662 | 1.619e-05 | 0.0002424 | 2.696e-09 | PASS |
| equity.snowball.intraday.quad_v2 | low_vol | desk_theta | 0.204715 | 7.11e-08 | 0.204715 | -5.05e-05 | 0.0007619 | 8.059e-07 | PASS |
| equity.snowball.intraday.quad_v2 | low_vol | point_delta | 5.54256 | 4.97e-07 | 5.54256 | -6.541e-05 | 0.0009618 | 2.014e-09 | PASS |
| equity.snowball.intraday.quad_v2 | low_vol | point_gamma | -1.96134 | 1.35e-06 | -1.96134 | 4.986e-05 | 0.0007397 | 5.293e-09 | PASS |
| equity.snowball.intraday.quad_v2 | high_vol | pv | -23.1511 | 3.56e-08 | -23.1511 | -0.0001191 | 0.0004755 | 9.699e-09 | PASS |
| equity.snowball.intraday.quad_v2 | high_vol | desk_delta | 1.40359 | 3.72e-09 | 1.40359 | -2.774e-05 | 5.423e-05 | 4.714e-10 | PASS |
| equity.snowball.intraday.quad_v2 | high_vol | desk_gamma | 0.0939431 | 9.24e-09 | 0.0939431 | 9.085e-06 | 0.0001075 | 1.07e-09 | PASS |
| equity.snowball.intraday.quad_v2 | high_vol | desk_theta | 0.114381 | 4.55e-09 | 0.114381 | -2.655e-05 | 7.207e-05 | 7.681e-08 | PASS |
| equity.snowball.intraday.quad_v2 | high_vol | point_delta | 1.44503 | 4.81e-09 | 1.44503 | -3.182e-05 | 6.509e-05 | 4.825e-10 | PASS |
| equity.snowball.intraday.quad_v2 | high_vol | point_gamma | 0.0834244 | 9.3e-09 | 0.0834244 | 1.399e-05 | 0.0001255 | 2.246e-09 | PASS |
| equity.snowball.intraday.quad_v2 | carry | pv | 7.92602 | 1.61e-07 | 7.92602 | 0.0002647 | 0.001879 | 3.872e-09 | PASS |
| equity.snowball.intraday.quad_v2 | carry | desk_delta | -0.158905 | 1.01e-08 | -0.158905 | -6.231e-05 | 0.0007008 | 2.236e-09 | PASS |
| equity.snowball.intraday.quad_v2 | carry | desk_gamma | -0.094928 | 2.41e-08 | -0.094928 | -5.13e-05 | 0.0003053 | 1.871e-10 | PASS |
| equity.snowball.intraday.quad_v2 | carry | desk_theta | 0.0140883 | 4.03e-09 | 0.0140883 | 7.656e-06 | 4.793e-05 | 9.948e-10 | PASS |
| equity.snowball.intraday.quad_v2 | carry | point_delta | -0.160595 | 1.04e-08 | -0.160595 | -6.293e-05 | 0.0007077 | 2.383e-09 | PASS |
| equity.snowball.intraday.quad_v2 | carry | point_gamma | -0.0963502 | 2.62e-08 | -0.0963502 | -5.536e-05 | 0.0003268 | 2.601e-10 | PASS |
| equity.snowball.intraday.quad_v2 | uniform_profile | pv | -22.3432 | 1.12e-07 | -22.3432 | -0.0004383 | 0.001555 | 9.415e-09 | PASS |
| equity.snowball.intraday.quad_v2 | uniform_profile | desk_delta | 2.686 | 7.66e-08 | 2.686 | -0.0001094 | 0.0003946 | 5.522e-10 | PASS |
| equity.snowball.intraday.quad_v2 | uniform_profile | desk_gamma | 0.042248 | 1.01e-07 | 0.0422481 | 0.0008647 | 0.003259 | 7.881e-09 | PASS |
| equity.snowball.intraday.quad_v2 | uniform_profile | desk_theta | 0.146642 | 1.42e-08 | 0.146642 | -5.275e-05 | 0.0001946 | 4.634e-07 | PASS |
| equity.snowball.intraday.quad_v2 | uniform_profile | point_delta | 3.54835 | 1.41e-07 | 3.54835 | -0.0001507 | 0.0005483 | 6.032e-10 | PASS |
| equity.snowball.intraday.quad_v2 | uniform_profile | point_gamma | -5.25104 | 5.55e-07 | -5.25104 | 3.934e-05 | 0.0001451 | 1.142e-10 | PASS |
| equity.snowball.intraday.quad_v2 | sessions_only_profile | pv | -22.4108 | 9.61e-08 | -22.4108 | -0.0003802 | 0.001341 | 9.734e-09 | PASS |
| equity.snowball.intraday.quad_v2 | sessions_only_profile | desk_delta | 2.56127 | 6.49e-08 | 2.56127 | -9.805e-05 | 0.0003514 | 4.803e-10 | PASS |
| equity.snowball.intraday.quad_v2 | sessions_only_profile | desk_gamma | 0.579227 | 3.35e-08 | 0.579227 | 1.953e-05 | 7.745e-05 | 3.688e-10 | PASS |
| equity.snowball.intraday.quad_v2 | sessions_only_profile | desk_theta | 0.214222 | 2.97e-08 | 0.214222 | -0.0001108 | 0.0004083 | 4.616e-07 | PASS |
| equity.snowball.intraday.quad_v2 | sessions_only_profile | point_delta | 2.8313 | 8.71e-08 | 2.8313 | -0.000118 | 0.0004257 | 4.376e-10 | PASS |
| equity.snowball.intraday.quad_v2 | sessions_only_profile | point_gamma | 0.431034 | 5.63e-08 | 0.431034 | 4.531e-05 | 0.0001759 | 4.908e-10 | PASS |
| equity.snowball.intraday.quad_v2 | on_ki_barrier_at_close | pv | -23.7972 | 2.6e-08 | -23.7972 | -0.0001151 | 0.0003749 | 1.03e-08 | PASS |
| equity.snowball.intraday.quad_v2 | on_ki_barrier_at_close | desk_delta | 2.47934 | 7.66e-08 | 2.47934 | -0.0001176 | 0.0004267 | 3.139e-07 | PASS |
| equity.snowball.intraday.quad_v2 | on_ki_barrier_at_close | desk_gamma | 3.9214 | 2.01e-07 | 3.9214 | -1.952e-05 | 7.072e-05 | 5.288e-08 | PASS |
| equity.snowball.intraday.quad_v2 | on_ki_barrier_at_close | desk_theta | -- | -- | -- | -- | -- | -- | PASS |
| equity.snowball.intraday.quad_v2 | on_ki_barrier_at_close | point_delta | -- | -- | -- | -- | -- | -- | PASS |
| equity.snowball.intraday.quad_v2 | on_ki_barrier_at_close | point_gamma | -- | -- | -- | -- | -- | -- | PASS |
| equity.snowball.intraday.pde | ordinary | pv | 7.97428 | 1.64e-07 | 7.9762 | 19.19 | 19.19 | 16.94 | FAIL |
| equity.snowball.intraday.pde | ordinary | desk_delta | -0.164795 | 1.04e-08 | -0.16473 | 3.977 | 3.978 | 7.87 | FAIL |
| equity.snowball.intraday.pde | ordinary | desk_gamma | -0.09634 | 2.45e-08 | -0.0967109 | -3.849 | 3.849 | 9.729 | FAIL |
| equity.snowball.intraday.pde | ordinary | desk_theta | 0.014282 | 4.09e-09 | 0.0117388 | -25.43 | 25.43 | 37.98 | FAIL |
| equity.snowball.intraday.pde | ordinary | point_delta | -0.16652 | 1.06e-08 | -0.166283 | 14.2 | 14.2 | 13.57 | FAIL |
| equity.snowball.intraday.pde | ordinary | point_gamma | -0.0977908 | 2.66e-08 | -0.0981266 | -3.435 | 3.435 | 3.004 | FAIL |
| equity.snowball.intraday.pde | near_ki_above | pv | -22.4213 | 9.9e-08 | -22.4177 | 36.35 | 36.35 | 38.12 | FAIL |
| equity.snowball.intraday.pde | near_ki_above | desk_delta | 2.61027 | 6.94e-08 | 2.61417 | 14.92 | 14.92 | 16.2 | FAIL |
| equity.snowball.intraday.pde | near_ki_above | desk_gamma | 0.516564 | 4.77e-08 | 0.522311 | 11.13 | 11.13 | 19.68 | FAIL |
| equity.snowball.intraday.pde | near_ki_above | desk_theta | 0.224767 | 2.68e-08 | 0.232079 | 73.12 | 73.12 | 297 | FAIL |
| equity.snowball.intraday.pde | near_ki_above | point_delta | 2.99352 | 9.95e-08 | 3.00242 | 29.74 | 29.74 | 30.87 | FAIL |
| equity.snowball.intraday.pde | near_ki_above | point_gamma | 0.120793 | 9.4e-08 | 0.117649 | -26.03 | 26.03 | 122 | FAIL |
| equity.snowball.intraday.pde | near_ki_below | pv | -23.2848 | 6.71e-08 | -23.2835 | 13.15 | 13.15 | 31.42 | FAIL |
| equity.snowball.intraday.pde | near_ki_below | desk_delta | 2.34313 | 6.97e-08 | 2.34631 | 13.58 | 13.58 | 12.68 | FAIL |
| equity.snowball.intraday.pde | near_ki_below | desk_gamma | 1.45744 | 4.04e-08 | 1.46719 | 6.692 | 6.692 | 9.71 | FAIL |
| equity.snowball.intraday.pde | near_ki_below | desk_theta | -0.613422 | 4.12e-08 | -0.614628 | -12.06 | 12.06 | 32.38 | FAIL |
| equity.snowball.intraday.pde | near_ki_below | point_delta | 2.66348 | 1.03e-07 | 2.66775 | 16.03 | 16.03 | 49.45 | FAIL |
| equity.snowball.intraday.pde | near_ki_below | point_gamma | 2.05773 | 7.65e-08 | 2.06958 | 5.756 | 5.756 | 12.62 | FAIL |
| equity.snowball.intraday.pde | near_ki_1s | pv | -22.1965 | 1.26e-07 | -22.1618 | 346.9 | 346.9 | 159.8 | FAIL |
| equity.snowball.intraday.pde | near_ki_1s | desk_delta | 2.68629 | 7.72e-08 | 2.76362 | 287.9 | 287.9 | 131.5 | FAIL |
| equity.snowball.intraday.pde | near_ki_1s | desk_gamma | -0.48681 | 1.5e-07 | -0.40438 | 169.3 | 169.3 | 76.3 | FAIL |
| equity.snowball.intraday.pde | near_ki_1s | desk_theta | -0.0436783 | 2.67e-07 | -85.5178 | -8.547e+05 | 8.547e+05 | 1.989e+06 | FAIL |
| equity.snowball.intraday.pde | near_ki_1s | point_delta | 2.36023 | 4.4e-08 | 2.50971 | 633.3 | 633.3 | 298.2 | FAIL |
| equity.snowball.intraday.pde | near_ki_1s | point_gamma | 0.532522 | 8.35e-07 | 0.535054 | 4.756 | 4.757 | 8.026 | FAIL |
| equity.snowball.intraday.pde | near_ki_10s | pv | -22.1964 | 1.26e-07 | -22.1777 | 187.1 | 187.1 | -- | UNRESOLVED |
| equity.snowball.intraday.pde | near_ki_10s | desk_delta | 2.6863 | 7.72e-08 | 2.72836 | 156.5 | 156.5 | -- | UNRESOLVED |
| equity.snowball.intraday.pde | near_ki_10s | desk_gamma | -0.487153 | 1.5e-07 | -0.442034 | 92.62 | 92.62 | -- | UNRESOLVED |
| equity.snowball.intraday.pde | near_ki_10s | desk_theta | -0.0436754 | 3.03e-08 | -2.83908 | -2.795e+04 | 2.795e+04 | -- | UNRESOLVED |
| equity.snowball.intraday.pde | near_ki_10s | point_delta | 2.36012 | 4.1e-08 | 2.43915 | 334.9 | 334.9 | -- | UNRESOLVED |
| equity.snowball.intraday.pde | near_ki_10s | point_gamma | 0.532451 | 1.28e-07 | 0.539307 | 12.88 | 12.88 | -- | UNRESOLVED |
| equity.snowball.intraday.pde | near_ki_5m | pv | -22.239 | 1.22e-07 | -22.2381 | 8.896 | 8.897 | 40 | FAIL |
| equity.snowball.intraday.pde | near_ki_5m | desk_delta | 2.68679 | 7.7e-08 | 2.6923 | 20.52 | 20.52 | 15.29 | FAIL |
| equity.snowball.intraday.pde | near_ki_5m | desk_gamma | -0.335102 | 1.36e-07 | -0.323608 | 34.3 | 34.3 | 34.43 | FAIL |
| equity.snowball.intraday.pde | near_ki_5m | desk_theta | 0.509623 | 4.99e-08 | 0.630305 | 1207 | 1207 | 3492 | FAIL |
| equity.snowball.intraday.pde | near_ki_5m | point_delta | 3.19392 | 1.08e-07 | 3.19679 | 8.973 | 8.974 | 46.85 | FAIL |
| equity.snowball.intraday.pde | near_ki_5m | point_gamma | -11.7084 | 1.06e-06 | -11.4717 | 20.21 | 20.21 | 18.95 | FAIL |
| equity.snowball.intraday.pde | near_ko_on_ko_day | pv | 7.22688 | 7.7e-09 | 7.22703 | 1.501 | 1.501 | 9.334 | FAIL |
| equity.snowball.intraday.pde | near_ko_on_ko_day | desk_delta | -0.972193 | 7.63e-09 | -0.969779 | 24.83 | 24.83 | 31.6 | FAIL |
| equity.snowball.intraday.pde | near_ko_on_ko_day | desk_gamma | -0.154872 | 1.51e-09 | -0.154001 | 5.625 | 5.625 | 14.58 | FAIL |
| equity.snowball.intraday.pde | near_ko_on_ko_day | desk_theta | 0.828925 | 4.08e-09 | 0.828525 | -3.999 | 3.999 | 8.149 | FAIL |
| equity.snowball.intraday.pde | near_ko_on_ko_day | point_delta | -1.49252 | 1.06e-08 | -1.49774 | -34.99 | 34.99 | 70.06 | FAIL |
| equity.snowball.intraday.pde | near_ko_on_ko_day | point_gamma | -0.370089 | 8.58e-10 | -0.369475 | 1.659 | 1.659 | 20.67 | FAIL |
| equity.snowball.intraday.pde | near_ko_on_ko_day_1s | pv | 8.0558 | 1.17e-08 | 8.0583 | 24.94 | 24.94 | 12.11 | FAIL |
| equity.snowball.intraday.pde | near_ko_on_ko_day_1s | desk_delta | -1.03888 | 8.02e-09 | -1.04205 | -30.48 | 30.48 | 14.72 | FAIL |
| equity.snowball.intraday.pde | near_ko_on_ko_day_1s | desk_gamma | -1.77076 | 6.66e-09 | -1.76932 | 0.8156 | 0.8156 | 0.3879 | PASS |
| equity.snowball.intraday.pde | near_ko_on_ko_day_1s | desk_theta | 0.00175132 | 2.54e-07 | -9.87632 | -9.878e+04 | 9.878e+04 | 1.461e+05 | FAIL |
| equity.snowball.intraday.pde | near_ko_on_ko_day_1s | point_delta | -0.135401 | 7.48e-09 | -0.139103 | -273.4 | 273.4 | 131.7 | FAIL |
| equity.snowball.intraday.pde | near_ko_on_ko_day_1s | point_gamma | -0.0111397 | 4.07e-07 | -0.0106186 | 46.77 | 46.81 | 22.66 | FAIL |
| equity.snowball.intraday.pde | ko_level_on_ki_day | pv | 7.17628 | 7.24e-08 | 7.17646 | 1.719 | 1.719 | 2.717 | FAIL |
| equity.snowball.intraday.pde | ko_level_on_ki_day | desk_delta | -0.325387 | 3.22e-08 | -0.326425 | -31.9 | 31.9 | 27.83 | FAIL |
| equity.snowball.intraday.pde | ko_level_on_ki_day | desk_gamma | 0.00215328 | 8.15e-09 | 0.00228437 | 60.88 | 60.88 | 108.4 | FAIL |
| equity.snowball.intraday.pde | ko_level_on_ki_day | desk_theta | -0.00032688 | 1.44e-09 | -0.000547474 | -2.206 | 2.206 | 3.307 | FAIL |
| equity.snowball.intraday.pde | ko_level_on_ki_day | point_delta | -0.333597 | 3.28e-08 | -0.334819 | -36.62 | 36.62 | 30.2 | FAIL |
| equity.snowball.intraday.pde | ko_level_on_ki_day | point_gamma | 0.00259306 | 8.8e-09 | 0.00266801 | 28.9 | 28.91 | 25.73 | FAIL |
| equity.snowball.intraday.pde | pre_open | pv | 7.92926 | 1.55e-07 | 7.93011 | 8.505 | 8.507 | 39.71 | FAIL |
| equity.snowball.intraday.pde | pre_open | desk_delta | -0.159805 | 1.08e-08 | -0.159537 | 16.75 | 16.75 | 14.86 | FAIL |
| equity.snowball.intraday.pde | pre_open | desk_gamma | -0.0887234 | 1.69e-08 | -0.0879464 | 8.757 | 8.757 | 51.44 | FAIL |
| equity.snowball.intraday.pde | pre_open | desk_theta | 0.00106058 | 4.43e-10 | 0.00355029 | 24.9 | 24.9 | 59.7 | FAIL |
| equity.snowball.intraday.pde | pre_open | point_delta | -0.161987 | 1.02e-08 | -0.161828 | 9.852 | 9.852 | 24.81 | FAIL |
| equity.snowball.intraday.pde | pre_open | point_gamma | -0.0899154 | 1.89e-08 | -0.0900202 | -1.165 | 1.165 | 6.436 | FAIL |
| equity.snowball.intraday.pde | lunch_break | pv | -20.2071 | 1.32e-07 | -20.1998 | 73.21 | 73.21 | 117.2 | FAIL |
| equity.snowball.intraday.pde | lunch_break | desk_delta | 2.66292 | 2.49e-08 | 2.66344 | 1.956 | 1.956 | 30.35 | FAIL |
| equity.snowball.intraday.pde | lunch_break | desk_gamma | -0.137078 | 5.84e-08 | -0.131152 | 43.23 | 43.23 | 182.8 | FAIL |
| equity.snowball.intraday.pde | lunch_break | desk_theta | 0.00279031 | 7.35e-10 | 0.00287159 | 0.8128 | 0.8128 | 49.17 | FAIL |
| equity.snowball.intraday.pde | lunch_break | point_delta | 2.66675 | 1.92e-08 | 2.66953 | 10.39 | 10.39 | 16.59 | FAIL |
| equity.snowball.intraday.pde | lunch_break | point_gamma | -0.201527 | 6.14e-08 | -0.196061 | 27.12 | 27.12 | 72.1 | FAIL |
| equity.snowball.intraday.pde | lunch_break_zero_variance | pv | -20.2163 | 1.29e-07 | -20.2027 | 135 | 135 | 147.4 | FAIL |
| equity.snowball.intraday.pde | lunch_break_zero_variance | desk_delta | 2.64135 | 2.48e-08 | 2.64683 | 20.77 | 20.77 | 30.65 | FAIL |
| equity.snowball.intraday.pde | lunch_break_zero_variance | desk_gamma | -0.0645224 | 5.06e-08 | -0.0604765 | 62.7 | 62.7 | 375.4 | FAIL |
| equity.snowball.intraday.pde | lunch_break_zero_variance | desk_theta | -0.000531722 | 1.33e-10 | -0.000616951 | -0.8523 | 0.8523 | 0.7722 | FAIL |
| equity.snowball.intraday.pde | lunch_break_zero_variance | point_delta | 2.66537 | 2.2e-08 | 2.67045 | 19.06 | 19.06 | 28.82 | FAIL |
| equity.snowball.intraday.pde | lunch_break_zero_variance | point_gamma | -0.127976 | 5.56e-08 | -0.12679 | 9.269 | 9.27 | 125.7 | FAIL |
| equity.snowball.intraday.pde | holiday_eve | pv | -20.0159 | 1.37e-05 | -20.0049 | 110.6 | 110.7 | 100.3 | FAIL |
| equity.snowball.intraday.pde | holiday_eve | desk_delta | 2.8658 | 1.32e-08 | 2.87033 | 15.8 | 15.8 | 16.34 | FAIL |
| equity.snowball.intraday.pde | holiday_eve | desk_gamma | -0.304062 | 6.07e-07 | -0.315698 | -38.27 | 38.27 | 60.43 | FAIL |
| equity.snowball.intraday.pde | holiday_eve | desk_theta | -0.00170456 | 1.18e-07 | -0.00611451 | -44.1 | 44.1 | 122.9 | FAIL |
| equity.snowball.intraday.pde | holiday_eve | point_delta | 2.7757 | 7.86e-09 | 2.78155 | 21.06 | 21.06 | 18.08 | FAIL |
| equity.snowball.intraday.pde | holiday_eve | point_gamma | -0.181555 | 2.68e-07 | -0.193154 | -63.89 | 63.89 | 63.99 | FAIL |
| equity.snowball.intraday.pde | maturity_day | pv | 12 | 7.89e-12 | 12 | 5.205e-09 | 8.412e-08 | 4.921e-09 | PASS |
| equity.snowball.intraday.pde | maturity_day | desk_delta | 8.08242e-14 | 4.46e-12 | -2.84217e-14 | -1.092e-08 | 4.568e-07 | 1.386e-08 | PASS |
| equity.snowball.intraday.pde | maturity_day | desk_gamma | 1.23102e-12 | 2.47e-11 | -2.27374e-13 | -1.458e-06 | 2.616e-05 | 2e-06 | PASS |
| equity.snowball.intraday.pde | maturity_day | desk_theta | 4.10958e-05 | 7.9e-12 | 4.10958e-05 | -5.205e-09 | 8.423e-08 | 4.921e-09 | PASS |
| equity.snowball.intraday.pde | maturity_day | point_delta | 1.35417e-14 | 2.34e-11 | 0 | -1.354e-09 | 2.343e-06 | 0 | PASS |
| equity.snowball.intraday.pde | maturity_day | point_gamma | 1.48711e-12 | 1.32e-11 | -0 | -1.487e-06 | 1.466e-05 | 0 | PASS |
| equity.snowball.intraday.pde | already_ki | pv | -8.26798 | 3.17e-08 | -8.26808 | -0.9991 | 0.9994 | -- | UNRESOLVED |
| equity.snowball.intraday.pde | already_ki | desk_delta | 1.04517 | 8.27e-09 | 1.04516 | -0.09503 | 0.09511 | -- | UNRESOLVED |
| equity.snowball.intraday.pde | already_ki | desk_gamma | -0.00409272 | 1.23e-09 | -0.00389176 | 49.1 | 49.1 | -- | UNRESOLVED |
| equity.snowball.intraday.pde | already_ki | desk_theta | 0.000226998 | 1.93e-10 | 0.000315191 | 0.8819 | 0.8819 | -- | UNRESOLVED |
| equity.snowball.intraday.pde | already_ki | point_delta | 1.04544 | 8.28e-09 | 1.04543 | -0.1023 | 0.1023 | -- | UNRESOLVED |
| equity.snowball.intraday.pde | already_ki | point_gamma | -0.00407064 | 1.21e-09 | -0.00407225 | -0.3955 | 0.3958 | -- | UNRESOLVED |
| equity.snowball.intraday.pde | terminated_pending_cash | pv | 5.0605 | 4.49e-15 | 5.0605 | 0 | 4.495e-11 | 0 | PASS |
| equity.snowball.intraday.pde | terminated_pending_cash | desk_delta | 0 | 4.49e-15 | 0 | 0 | 4.495e-10 | 0 | PASS |
| equity.snowball.intraday.pde | terminated_pending_cash | desk_gamma | 0 | 1.8e-14 | 0 | 0 | 1.798e-08 | 0 | PASS |
| equity.snowball.intraday.pde | terminated_pending_cash | desk_theta | 1.73305e-05 | 8.99e-15 | 1.73305e-05 | 0 | 8.989e-11 | 0 | PASS |
| equity.snowball.intraday.pde | terminated_pending_cash | point_delta | 0 | 0 | 0 | 0 | 0 | 0 | PASS |
| equity.snowball.intraday.pde | terminated_pending_cash | point_gamma | 0 | 0 | 0 | 0 | 0 | 0 | PASS |
| equity.snowball.intraday.pde | provisional_ki_after_close | pv | -23.8982 | 1.1e-05 | -23.8981 | 1.098 | 1.207 | 1.276 | FAIL |
| equity.snowball.intraday.pde | provisional_ki_after_close | desk_delta | 1.00962 | 2.59e-09 | 1.00959 | -0.3632 | 0.3633 | 1.264 | FAIL |
| equity.snowball.intraday.pde | provisional_ki_after_close | desk_gamma | 0.00288312 | 6.49e-10 | 0.00266529 | -75.55 | 75.55 | 265.3 | FAIL |
| equity.snowball.intraday.pde | provisional_ki_after_close | desk_theta | -0.000272405 | 9.64e-11 | -0.000288191 | -0.1579 | 0.1579 | 0.4806 | PASS |
| equity.snowball.intraday.pde | provisional_ki_after_close | point_delta | 1.00959 | 2.66e-09 | 1.00959 | -0.004346 | 0.004372 | 0.04276 | PASS |
| equity.snowball.intraday.pde | provisional_ki_after_close | point_gamma | 0.00288347 | 3.22e-10 | 0.00288265 | -0.2827 | 0.2828 | 0.2882 | PASS |
| equity.snowball.intraday.pde | low_vol | pv | -21.6485 | 3.38e-07 | -21.6404 | 81.29 | 81.29 | 92.91 | FAIL |
| equity.snowball.intraday.pde | low_vol | desk_delta | 4.25292 | 2.59e-07 | 4.25976 | 16.09 | 16.09 | 16.13 | FAIL |
| equity.snowball.intraday.pde | low_vol | desk_gamma | 1.62662 | 3.68e-07 | 1.62354 | -1.892 | 1.892 | 12.29 | FAIL |
| equity.snowball.intraday.pde | low_vol | desk_theta | 0.204715 | 7.11e-08 | 0.208243 | 35.28 | 35.28 | 231.2 | FAIL |
| equity.snowball.intraday.pde | low_vol | point_delta | 5.54256 | 4.97e-07 | 5.56075 | 32.82 | 32.82 | 42.47 | FAIL |
| equity.snowball.intraday.pde | low_vol | point_gamma | -1.96134 | 1.35e-06 | -2.01275 | -26.21 | 26.21 | 99.9 | FAIL |
| equity.snowball.intraday.pde | high_vol | pv | -23.1511 | 3.56e-08 | -23.1499 | 12.17 | 12.17 | 16.48 | FAIL |
| equity.snowball.intraday.pde | high_vol | desk_delta | 1.40359 | 3.72e-09 | 1.40596 | 16.93 | 16.93 | 18.39 | FAIL |
| equity.snowball.intraday.pde | high_vol | desk_gamma | 0.0939431 | 9.24e-09 | 0.0954027 | 15.54 | 15.54 | 76.52 | FAIL |
| equity.snowball.intraday.pde | high_vol | desk_theta | 0.114381 | 4.55e-09 | 0.114007 | -3.74 | 3.74 | 16.84 | FAIL |
| equity.snowball.intraday.pde | high_vol | point_delta | 1.44503 | 4.81e-09 | 1.44745 | 16.78 | 16.78 | 24.69 | FAIL |
| equity.snowball.intraday.pde | high_vol | point_gamma | 0.0834244 | 9.3e-09 | 0.0841009 | 8.11 | 8.11 | 27.31 | FAIL |
| equity.snowball.intraday.pde | carry | pv | 7.92602 | 1.61e-07 | 7.92791 | 18.88 | 18.88 | 16.68 | FAIL |
| equity.snowball.intraday.pde | carry | desk_delta | -0.158905 | 1.01e-08 | -0.158842 | 3.945 | 3.946 | 7.985 | FAIL |
| equity.snowball.intraday.pde | carry | desk_gamma | -0.094928 | 2.41e-08 | -0.0952907 | -3.82 | 3.821 | 9.734 | FAIL |
| equity.snowball.intraday.pde | carry | desk_theta | 0.0140883 | 4.03e-09 | 0.0115864 | -25.02 | 25.02 | 37.37 | FAIL |
| equity.snowball.intraday.pde | carry | point_delta | -0.160595 | 1.04e-08 | -0.160364 | 14.38 | 14.38 | 13.78 | FAIL |
| equity.snowball.intraday.pde | carry | point_gamma | -0.0963502 | 2.62e-08 | -0.0966788 | -3.411 | 3.411 | 2.985 | FAIL |
| equity.snowball.intraday.pde | uniform_profile | pv | -22.3432 | 1.12e-07 | -22.343 | 1.413 | 1.414 | 14.54 | FAIL |
| equity.snowball.intraday.pde | uniform_profile | desk_delta | 2.686 | 7.66e-08 | 2.68599 | -0.04421 | 0.0445 | 1.44 | FAIL |
| equity.snowball.intraday.pde | uniform_profile | desk_gamma | 0.042248 | 1.01e-07 | 0.0440757 | 43.26 | 43.26 | 141.8 | FAIL |
| equity.snowball.intraday.pde | uniform_profile | desk_theta | 0.146642 | 1.42e-08 | 0.157427 | 107.9 | 107.9 | 327.4 | FAIL |
| equity.snowball.intraday.pde | uniform_profile | point_delta | 3.54835 | 1.41e-07 | 3.5434 | -13.95 | 13.95 | 21.4 | FAIL |
| equity.snowball.intraday.pde | uniform_profile | point_gamma | -5.25104 | 5.55e-07 | -5.18243 | 13.07 | 13.07 | 22.38 | FAIL |
| equity.snowball.intraday.pde | sessions_only_profile | pv | -22.4108 | 9.61e-08 | -22.4024 | 83.96 | 83.97 | 78.94 | FAIL |
| equity.snowball.intraday.pde | sessions_only_profile | desk_delta | 2.56127 | 6.49e-08 | 2.56874 | 29.18 | 29.18 | 42.63 | FAIL |
| equity.snowball.intraday.pde | sessions_only_profile | desk_gamma | 0.579227 | 3.35e-08 | 0.585341 | 10.55 | 10.55 | 29.06 | FAIL |
| equity.snowball.intraday.pde | sessions_only_profile | desk_theta | 0.214222 | 2.97e-08 | 0.21715 | 29.28 | 29.28 | 236.3 | FAIL |
| equity.snowball.intraday.pde | sessions_only_profile | point_delta | 2.8313 | 8.71e-08 | 2.84556 | 50.38 | 50.38 | 56.28 | FAIL |
| equity.snowball.intraday.pde | sessions_only_profile | point_gamma | 0.431034 | 5.63e-08 | 0.435197 | 9.658 | 9.658 | 44.88 | FAIL |
| equity.snowball.intraday.pde | on_ki_barrier_at_close | pv | -23.7972 | 2.6e-08 | -23.7971 | 1.294 | 1.295 | 2.368 | FAIL |
| equity.snowball.intraday.pde | on_ki_barrier_at_close | desk_delta | 2.47934 | 7.66e-08 | 2.48543 | 24.56 | 24.56 | 52.01 | FAIL |
| equity.snowball.intraday.pde | on_ki_barrier_at_close | desk_gamma | 3.9214 | 2.01e-07 | 3.93724 | 4.038 | 4.038 | 8.374 | FAIL |
| equity.snowball.intraday.pde | on_ki_barrier_at_close | desk_theta | -- | -- | -- | -- | -- | -- | PASS |
| equity.snowball.intraday.pde | on_ki_barrier_at_close | point_delta | -- | -- | -- | -- | -- | -- | PASS |
| equity.snowball.intraday.pde | on_ki_barrier_at_close | point_gamma | -- | -- | -- | -- | -- | -- | PASS |

## Semantic, uncertified and unresolved cells

| candidate | case | quantity | kind | verdict | reason |
|---|---|---|---|---|---|
| equity.snowball.intraday.quad_v2 | on_ki_barrier_at_close | desk_theta | semantic | PASS | expected undefined, candidate reported undefined |
| equity.snowball.intraday.quad_v2 | on_ki_barrier_at_close | point_delta | semantic | PASS | expected undefined, candidate reported undefined |
| equity.snowball.intraday.quad_v2 | on_ki_barrier_at_close | point_gamma | semantic | PASS | expected undefined, candidate reported undefined |
| equity.snowball.intraday.pde | near_ki_10s | pv | numeric | UNRESOLVED | convergence evidence has fewer than 3 levels on: space, time |
| equity.snowball.intraday.pde | near_ki_10s | desk_delta | numeric | UNRESOLVED | convergence evidence has fewer than 3 levels on: space, time |
| equity.snowball.intraday.pde | near_ki_10s | desk_gamma | numeric | UNRESOLVED | convergence evidence has fewer than 3 levels on: space, time |
| equity.snowball.intraday.pde | near_ki_10s | desk_theta | numeric | UNRESOLVED | convergence evidence has fewer than 3 levels on: space, time |
| equity.snowball.intraday.pde | near_ki_10s | point_delta | numeric | UNRESOLVED | convergence evidence has fewer than 3 levels on: space, time |
| equity.snowball.intraday.pde | near_ki_10s | point_gamma | numeric | UNRESOLVED | convergence evidence has fewer than 3 levels on: space, time |
| equity.snowball.intraday.pde | already_ki | pv | numeric | UNRESOLVED | convergence evidence has fewer than 3 levels on: space, time |
| equity.snowball.intraday.pde | already_ki | desk_delta | numeric | UNRESOLVED | convergence evidence has fewer than 3 levels on: space, time |
| equity.snowball.intraday.pde | already_ki | desk_gamma | numeric | UNRESOLVED | convergence evidence has fewer than 3 levels on: space, time |
| equity.snowball.intraday.pde | already_ki | desk_theta | numeric | UNRESOLVED | convergence evidence has fewer than 3 levels on: space, time |
| equity.snowball.intraday.pde | already_ki | point_delta | numeric | UNRESOLVED | convergence evidence has fewer than 3 levels on: space, time |
| equity.snowball.intraday.pde | already_ki | point_gamma | numeric | UNRESOLVED | convergence evidence has fewer than 3 levels on: space, time |
| equity.snowball.intraday.pde | on_ki_barrier_at_close | desk_theta | semantic | PASS | expected undefined, candidate reported undefined |
| equity.snowball.intraday.pde | on_ki_barrier_at_close | point_delta | semantic | PASS | expected undefined, candidate reported undefined |
| equity.snowball.intraday.pde | on_ki_barrier_at_close | point_gamma | semantic | PASS | expected undefined, candidate reported undefined |

## Convergence

| candidate | case | quantity | envelope | observed order per axis | non-monotone axes |
|---|---|---|---|---|---|
| equity.snowball.intraday.quad_v2 | ordinary | pv | 3.99e-13 | cells_per_sd: 0.634, domain_sd: -5.61, order: 1.74 | domain_sd |
| equity.snowball.intraday.quad_v2 | ordinary | desk_delta | 4e-14 | cells_per_sd: 0.737, domain_sd: -12.7, order: 1.58 | domain_sd |
| equity.snowball.intraday.quad_v2 | ordinary | desk_gamma | 2.49e-14 | cells_per_sd: 2.32, domain_sd: 4.15, order: -0.71 | order |
| equity.snowball.intraday.quad_v2 | ordinary | desk_theta | 1.25e-13 | cells_per_sd: 0.507, domain_sd: -13.1, order: -0.29 | order, domain_sd |
| equity.snowball.intraday.quad_v2 | ordinary | point_delta | 4.21e-14 | cells_per_sd: 0.886, domain_sd: -10.5, order: 1.41 | domain_sd |
| equity.snowball.intraday.quad_v2 | ordinary | point_gamma | 2.92e-14 | cells_per_sd: 2.64, domain_sd: 6.37, order: -4.79 | order |
| equity.snowball.intraday.quad_v2 | near_ki_above | pv | 1.02e-12 | cells_per_sd: 0.497, domain_sd: -5.62, order: 0.249 | domain_sd |
| equity.snowball.intraday.quad_v2 | near_ki_above | desk_delta | 1.82e-13 | cells_per_sd: 1.66, domain_sd: -1.53, order: 0.959 | domain_sd |
| equity.snowball.intraday.quad_v2 | near_ki_above | desk_gamma | 1.95e-13 | cells_per_sd: 0.539, domain_sd: -6.51, order: 1.71 | domain_sd |
| equity.snowball.intraday.quad_v2 | near_ki_above | desk_theta | 4.65e-11 | cells_per_sd: 10.1, domain_sd: 20.2, order: 8.36 | -- |
| equity.snowball.intraday.quad_v2 | near_ki_above | point_delta | 2.24e-13 | cells_per_sd: 0.395, domain_sd: -1.64, order: 0.839 | domain_sd |
| equity.snowball.intraday.quad_v2 | near_ki_above | point_gamma | 2.74e-13 | cells_per_sd: 1.85, domain_sd: -1.35, order: 6.21 | domain_sd |
| equity.snowball.intraday.quad_v2 | near_ki_below | pv | 1.67e-12 | cells_per_sd: 6.51, domain_sd: -5.89, order: 1.75 | domain_sd |
| equity.snowball.intraday.quad_v2 | near_ki_below | desk_delta | 5.85e-13 | cells_per_sd: 4.12, domain_sd: -5.09, order: 2.68 | domain_sd |
| equity.snowball.intraday.quad_v2 | near_ki_below | desk_gamma | 1.06e-12 | cells_per_sd: 2.93, domain_sd: -3.27, order: 5.39 | domain_sd |
| equity.snowball.intraday.quad_v2 | near_ki_below | desk_theta | 7.89e-13 | cells_per_sd: --, domain_sd: -23.9, order: 8.77 | domain_sd |
| equity.snowball.intraday.quad_v2 | near_ki_below | point_delta | 1.18e-12 | cells_per_sd: 7.64, domain_sd: -4.74, order: 4.49 | domain_sd |
| equity.snowball.intraday.quad_v2 | near_ki_below | point_gamma | 1.98e-12 | cells_per_sd: 5.31, domain_sd: 1.26, order: 7.33 | -- |
| equity.snowball.intraday.quad_v2 | near_ki_1s | pv | 1.04e-11 | cells_per_sd: 6.75, domain_sd: 16.7, order: 5.12 | -- |
| equity.snowball.intraday.quad_v2 | near_ki_1s | desk_delta | 3.32e-11 | cells_per_sd: 8.2, domain_sd: 27.9, order: 11.9 | -- |
| equity.snowball.intraday.quad_v2 | near_ki_1s | desk_gamma | 1.28e-10 | cells_per_sd: 12.4, domain_sd: 28.1, order: 13.4 | -- |
| equity.snowball.intraday.quad_v2 | near_ki_1s | desk_theta | 1.26e-07 | cells_per_sd: -3.17, domain_sd: 32.1, order: -3.42 | cells_per_sd, order |
| equity.snowball.intraday.quad_v2 | near_ki_1s | point_delta | 8.57e-10 | cells_per_sd: 3.45, domain_sd: 30.8, order: 6.03 | -- |
| equity.snowball.intraday.quad_v2 | near_ki_1s | point_gamma | 3.19e-07 | cells_per_sd: 4.32, domain_sd: 30.6, order: 9.75 | -- |
| equity.snowball.intraday.quad_v2 | near_ki_10s | pv | 5.52e-12 | cells_per_sd: 6.55, domain_sd: 0.521, order: 4.11 | -- |
| equity.snowball.intraday.quad_v2 | near_ki_10s | desk_delta | 1.1e-11 | cells_per_sd: 6.32, domain_sd: 26.2, order: 5.98 | -- |
| equity.snowball.intraday.quad_v2 | near_ki_10s | desk_gamma | 5.16e-11 | cells_per_sd: 7.81, domain_sd: 28.3, order: 14.8 | -- |
| equity.snowball.intraday.quad_v2 | near_ki_10s | desk_theta | 1.44e-08 | cells_per_sd: 4.4, domain_sd: 29.1, order: 6.87 | -- |
| equity.snowball.intraday.quad_v2 | near_ki_10s | point_delta | 5.97e-10 | cells_per_sd: 6.25, domain_sd: 31.6, order: 8.68 | -- |
| equity.snowball.intraday.quad_v2 | near_ki_10s | point_gamma | 1.33e-08 | cells_per_sd: 7.4, domain_sd: 31.9, order: 16.3 | -- |
| equity.snowball.intraday.quad_v2 | near_ki_5m | pv | 1.04e-12 | cells_per_sd: 2.65, domain_sd: -7.35, order: 0.329 | domain_sd |
| equity.snowball.intraday.quad_v2 | near_ki_5m | desk_delta | 1.8e-13 | cells_per_sd: -0.635, domain_sd: -13.6, order: 3.13 | cells_per_sd, domain_sd |
| equity.snowball.intraday.quad_v2 | near_ki_5m | desk_gamma | 1.76e-13 | cells_per_sd: 2.81, domain_sd: -17.7, order: -2.5 | order, domain_sd |
| equity.snowball.intraday.quad_v2 | near_ki_5m | desk_theta | 5.56e-10 | cells_per_sd: 5.16, domain_sd: 23, order: 9.5 | -- |
| equity.snowball.intraday.quad_v2 | near_ki_5m | point_delta | 9.74e-13 | cells_per_sd: 5.35, domain_sd: -33.8, order: 12.3 | domain_sd |
| equity.snowball.intraday.quad_v2 | near_ki_5m | point_gamma | 5.63e-12 | cells_per_sd: 2.59, domain_sd: -40.1, order: 4.11 | domain_sd |
| equity.snowball.intraday.quad_v2 | near_ko_on_ko_day | pv | 1.95e-13 | cells_per_sd: 1, domain_sd: 0.23, order: 0.686 | -- |
| equity.snowball.intraday.quad_v2 | near_ko_on_ko_day | desk_delta | 1.58e-13 | cells_per_sd: 1.08, domain_sd: -0.114, order: 0.559 | domain_sd |
| equity.snowball.intraday.quad_v2 | near_ko_on_ko_day | desk_gamma | 3.35e-14 | cells_per_sd: 0, domain_sd: 1.72, order: 2.68 | -- |
| equity.snowball.intraday.quad_v2 | near_ko_on_ko_day | desk_theta | 1.03e-13 | cells_per_sd: 0.766, domain_sd: 1.59, order: 1.3 | -- |
| equity.snowball.intraday.quad_v2 | near_ko_on_ko_day | point_delta | 2.46e-13 | cells_per_sd: 1.04, domain_sd: 0.118, order: 0.666 | -- |
| equity.snowball.intraday.quad_v2 | near_ko_on_ko_day | point_gamma | 6.84e-14 | cells_per_sd: 1.27, domain_sd: 1.52, order: 1.39 | -- |
| equity.snowball.intraday.quad_v2 | near_ko_on_ko_day_1s | pv | 3.23e-13 | cells_per_sd: 4.68, domain_sd: -0.885, order: -0.76 | order, domain_sd |
| equity.snowball.intraday.quad_v2 | near_ko_on_ko_day_1s | desk_delta | 1.55e-13 | cells_per_sd: 3.35, domain_sd: -1.51, order: -1.25 | order, domain_sd |
| equity.snowball.intraday.quad_v2 | near_ko_on_ko_day_1s | desk_gamma | 3.1e-13 | cells_per_sd: 5.2, domain_sd: -0.248, order: -0.292 | order, domain_sd |
| equity.snowball.intraday.quad_v2 | near_ko_on_ko_day_1s | desk_theta | 4.92e-10 | cells_per_sd: 1.17, domain_sd: 10.2, order: 2.13 | -- |
| equity.snowball.intraday.quad_v2 | near_ko_on_ko_day_1s | point_delta | 4.33e-13 | cells_per_sd: -2.03, domain_sd: -2.15, order: -0.621 | cells_per_sd, order, domain_sd |
| equity.snowball.intraday.quad_v2 | near_ko_on_ko_day_1s | point_gamma | 1.26e-11 | cells_per_sd: -2.28, domain_sd: -1.67, order: -0.333 | cells_per_sd, order, domain_sd |
| equity.snowball.intraday.quad_v2 | ko_level_on_ki_day | pv | 2.11e-13 | cells_per_sd: 0.372, domain_sd: -5.02, order: 1.5 | domain_sd |
| equity.snowball.intraday.quad_v2 | ko_level_on_ki_day | desk_delta | 3.75e-14 | cells_per_sd: -2.05, domain_sd: -8.25, order: 1.7 | cells_per_sd, domain_sd |
| equity.snowball.intraday.quad_v2 | ko_level_on_ki_day | desk_gamma | 2.6e-14 | cells_per_sd: 1.14, domain_sd: -8.78, order: 2.71 | domain_sd |
| equity.snowball.intraday.quad_v2 | ko_level_on_ki_day | desk_theta | 2.45e-13 | cells_per_sd: -2.09, domain_sd: -2.45, order: -0.495 | cells_per_sd, order, domain_sd |
| equity.snowball.intraday.quad_v2 | ko_level_on_ki_day | point_delta | 4.05e-14 | cells_per_sd: -0.759, domain_sd: -7.6, order: 1.66 | cells_per_sd, domain_sd |
| equity.snowball.intraday.quad_v2 | ko_level_on_ki_day | point_gamma | 2.18e-14 | cells_per_sd: 1.68, domain_sd: -4.04, order: 2.21 | domain_sd |
| equity.snowball.intraday.quad_v2 | pre_open | pv | 3.76e-13 | cells_per_sd: 2.58, domain_sd: -4.87, order: 0.251 | domain_sd |
| equity.snowball.intraday.quad_v2 | pre_open | desk_delta | 3.42e-14 | cells_per_sd: -1.58, domain_sd: -7.57, order: 0.753 | cells_per_sd, domain_sd |
| equity.snowball.intraday.quad_v2 | pre_open | desk_gamma | 1.78e-14 | cells_per_sd: 1.58, domain_sd: 4.15, order: -- | -- |
| equity.snowball.intraday.quad_v2 | pre_open | desk_theta | 2.18e-13 | cells_per_sd: -2.86, domain_sd: -0.0584, order: 1.91 | cells_per_sd, domain_sd |
| equity.snowball.intraday.quad_v2 | pre_open | point_delta | 3.58e-14 | cells_per_sd: 2.46, domain_sd: -7.15, order: 0.521 | domain_sd |
| equity.snowball.intraday.quad_v2 | pre_open | point_gamma | 1.91e-14 | cells_per_sd: 7.14, domain_sd: 1.54, order: 0.925 | -- |
| equity.snowball.intraday.quad_v2 | lunch_break | pv | 1.84e-12 | cells_per_sd: 7.78, domain_sd: -1.2, order: -2.08 | order, domain_sd |
| equity.snowball.intraday.quad_v2 | lunch_break | desk_delta | 1.22e-13 | cells_per_sd: 1.58, domain_sd: -4.86, order: 0.601 | domain_sd |
| equity.snowball.intraday.quad_v2 | lunch_break | desk_gamma | 1.72e-13 | cells_per_sd: 0.00109, domain_sd: 6.29, order: -4.42 | order |
| equity.snowball.intraday.quad_v2 | lunch_break | desk_theta | 1.49e-13 | cells_per_sd: 1.58, domain_sd: -2.2, order: 1.21 | domain_sd |
| equity.snowball.intraday.quad_v2 | lunch_break | point_delta | 1.88e-13 | cells_per_sd: 1.23, domain_sd: 2.16, order: -1.81 | order |
| equity.snowball.intraday.quad_v2 | lunch_break | point_gamma | 2.22e-13 | cells_per_sd: 0.502, domain_sd: 10, order: 2.63 | -- |
| equity.snowball.intraday.quad_v2 | lunch_break_zero_variance | pv | 1.41e-12 | cells_per_sd: 4.18, domain_sd: -10.3, order: -1.87 | order, domain_sd |
| equity.snowball.intraday.quad_v2 | lunch_break_zero_variance | desk_delta | 1.4e-13 | cells_per_sd: -0.509, domain_sd: -6.96, order: 0.376 | cells_per_sd, domain_sd |
| equity.snowball.intraday.quad_v2 | lunch_break_zero_variance | desk_gamma | 2.34e-13 | cells_per_sd: -1.72, domain_sd: -1.16, order: -0.709 | cells_per_sd, order, domain_sd |
| equity.snowball.intraday.quad_v2 | lunch_break_zero_variance | desk_theta | 1.1e-13 | cells_per_sd: -1.89, domain_sd: 4.15, order: -1 | cells_per_sd, order |
| equity.snowball.intraday.quad_v2 | lunch_break_zero_variance | point_delta | 1.27e-13 | cells_per_sd: 1.47, domain_sd: -2.64, order: -0.456 | order, domain_sd |
| equity.snowball.intraday.quad_v2 | lunch_break_zero_variance | point_gamma | 1.48e-13 | cells_per_sd: 2.6, domain_sd: -16, order: 6.26 | domain_sd |
| equity.snowball.intraday.quad_v2 | holiday_eve | pv | 2.42e-12 | cells_per_sd: 1.25, domain_sd: 0.891, order: 0.0525 | -- |
| equity.snowball.intraday.quad_v2 | holiday_eve | desk_delta | 2.85e-13 | cells_per_sd: 0.729, domain_sd: -3.88, order: -2.46 | order, domain_sd |
| equity.snowball.intraday.quad_v2 | holiday_eve | desk_gamma | 8.49e-13 | cells_per_sd: 5.42, domain_sd: 2.48, order: 0.619 | -- |
| equity.snowball.intraday.quad_v2 | holiday_eve | desk_theta | 1.45e-10 | cells_per_sd: 8.9, domain_sd: 19.9, order: 22.1 | -- |
| equity.snowball.intraday.quad_v2 | holiday_eve | point_delta | 1.92e-13 | cells_per_sd: 0.227, domain_sd: -21, order: -0.219 | order, domain_sd |
| equity.snowball.intraday.quad_v2 | holiday_eve | point_gamma | 1.28e-12 | cells_per_sd: 5.02, domain_sd: -0.0805, order: 6.1 | domain_sd |
| equity.snowball.intraday.quad_v2 | maturity_day | pv | 0 | cells_per_sd: --, domain_sd: --, order: -- | -- |
| equity.snowball.intraday.quad_v2 | maturity_day | desk_delta | 0 | cells_per_sd: --, domain_sd: --, order: -- | -- |
| equity.snowball.intraday.quad_v2 | maturity_day | desk_gamma | 0 | cells_per_sd: --, domain_sd: --, order: -- | -- |
| equity.snowball.intraday.quad_v2 | maturity_day | desk_theta | 0 | cells_per_sd: --, domain_sd: --, order: -- | -- |
| equity.snowball.intraday.quad_v2 | maturity_day | point_delta | 0 | cells_per_sd: --, domain_sd: --, order: -- | -- |
| equity.snowball.intraday.quad_v2 | maturity_day | point_gamma | 0 | cells_per_sd: --, domain_sd: --, order: -- | -- |
| equity.snowball.intraday.quad_v2 | already_ki | pv | 7.14e-13 | cells_per_sd: 3.84, domain_sd: -5.49, order: 0.141 | domain_sd |
| equity.snowball.intraday.quad_v2 | already_ki | desk_delta | 7.13e-14 | cells_per_sd: 0.556, domain_sd: -2.27, order: 0.781 | domain_sd |
| equity.snowball.intraday.quad_v2 | already_ki | desk_gamma | 5.04e-14 | cells_per_sd: 0, domain_sd: -12.4, order: -0.785 | order, domain_sd |
| equity.snowball.intraday.quad_v2 | already_ki | desk_theta | 3.73e-14 | cells_per_sd: -3.17, domain_sd: -22.8, order: -5.74 | cells_per_sd, order, domain_sd |
| equity.snowball.intraday.quad_v2 | already_ki | point_delta | 6.73e-14 | cells_per_sd: 3.38, domain_sd: -2.54, order: 0.415 | domain_sd |
| equity.snowball.intraday.quad_v2 | already_ki | point_gamma | 2.83e-14 | cells_per_sd: 0.433, domain_sd: -7.46, order: -3.35 | order, domain_sd |
| equity.snowball.intraday.quad_v2 | terminated_pending_cash | pv | 0 | cells_per_sd: --, domain_sd: --, order: -- | -- |
| equity.snowball.intraday.quad_v2 | terminated_pending_cash | desk_delta | 0 | cells_per_sd: --, domain_sd: --, order: -- | -- |
| equity.snowball.intraday.quad_v2 | terminated_pending_cash | desk_gamma | 0 | cells_per_sd: --, domain_sd: --, order: -- | -- |
| equity.snowball.intraday.quad_v2 | terminated_pending_cash | desk_theta | 0 | cells_per_sd: --, domain_sd: --, order: -- | -- |
| equity.snowball.intraday.quad_v2 | terminated_pending_cash | point_delta | 0 | cells_per_sd: --, domain_sd: --, order: -- | -- |
| equity.snowball.intraday.quad_v2 | terminated_pending_cash | point_gamma | 0 | cells_per_sd: --, domain_sd: --, order: -- | -- |
| equity.snowball.intraday.quad_v2 | provisional_ki_after_close | pv | 1.07e-12 | cells_per_sd: 2, domain_sd: 4.96, order: 0.444 | -- |
| equity.snowball.intraday.quad_v2 | provisional_ki_after_close | desk_delta | 1.14e-13 | cells_per_sd: 1.9, domain_sd: 11.9, order: 2.53 | -- |
| equity.snowball.intraday.quad_v2 | provisional_ki_after_close | desk_gamma | 7.6e-14 | cells_per_sd: 1.32, domain_sd: 8.3, order: 1 | -- |
| equity.snowball.intraday.quad_v2 | provisional_ki_after_close | desk_theta | 1.03e-13 | cells_per_sd: 0.807, domain_sd: 0.923, order: 1.88 | -- |
| equity.snowball.intraday.quad_v2 | provisional_ki_after_close | point_delta | 1.21e-13 | cells_per_sd: 2.39, domain_sd: --, order: 0.145 | -- |
| equity.snowball.intraday.quad_v2 | provisional_ki_after_close | point_gamma | 1.05e-14 | cells_per_sd: 2.8, domain_sd: 2, order: -4.4 | order |
| equity.snowball.intraday.quad_v2 | low_vol | pv | 3.38e-12 | cells_per_sd: 4.92, domain_sd: -8.51, order: 3.11 | domain_sd |
| equity.snowball.intraday.quad_v2 | low_vol | desk_delta | 1.15e-12 | cells_per_sd: 6.47, domain_sd: -14, order: 3.77 | domain_sd |
| equity.snowball.intraday.quad_v2 | low_vol | desk_gamma | 4.39e-12 | cells_per_sd: 4.02, domain_sd: -18.2, order: 9.26 | domain_sd |
| equity.snowball.intraday.quad_v2 | low_vol | desk_theta | 8.06e-11 | cells_per_sd: 4.29, domain_sd: 24.5, order: 20.4 | -- |
| equity.snowball.intraday.quad_v2 | low_vol | point_delta | 1.12e-12 | cells_per_sd: 3.29, domain_sd: -4.91, order: 5.22 | domain_sd |
| equity.snowball.intraday.quad_v2 | low_vol | point_gamma | 1.04e-11 | cells_per_sd: 3.64, domain_sd: -21.7, order: 8.78 | domain_sd |
| equity.snowball.intraday.quad_v2 | high_vol | pv | 9.7e-13 | cells_per_sd: 2.32, domain_sd: 1.64, order: -0.399 | order |
| equity.snowball.intraday.quad_v2 | high_vol | desk_delta | 6.62e-14 | cells_per_sd: -1.54, domain_sd: -1.77, order: 0.845 | cells_per_sd, domain_sd |
| equity.snowball.intraday.quad_v2 | high_vol | desk_gamma | 1.01e-13 | cells_per_sd: 0, domain_sd: 2.43, order: 2.97 | -- |
| equity.snowball.intraday.quad_v2 | high_vol | desk_theta | 7.68e-12 | cells_per_sd: 4.2, domain_sd: 26.2, order: 8.99 | -- |
| equity.snowball.intraday.quad_v2 | high_vol | point_delta | 6.97e-14 | cells_per_sd: 1.95, domain_sd: -4.15, order: 0.464 | domain_sd |
| equity.snowball.intraday.quad_v2 | high_vol | point_gamma | 1.87e-13 | cells_per_sd: 6.14, domain_sd: 31.3, order: 10.4 | -- |
| equity.snowball.intraday.quad_v2 | carry | pv | 3.87e-13 | cells_per_sd: 0.678, domain_sd: -5.45, order: 0.792 | domain_sd |
| equity.snowball.intraday.quad_v2 | carry | desk_delta | 3.55e-14 | cells_per_sd: -1.18, domain_sd: -15.7, order: 0.956 | cells_per_sd, domain_sd |
| equity.snowball.intraday.quad_v2 | carry | desk_gamma | 1.78e-14 | cells_per_sd: --, domain_sd: -6.58, order: 1.71 | domain_sd |
| equity.snowball.intraday.quad_v2 | carry | desk_theta | 9.95e-14 | cells_per_sd: 0.312, domain_sd: -15.1, order: -2.77 | order, domain_sd |
| equity.snowball.intraday.quad_v2 | carry | point_delta | 3.83e-14 | cells_per_sd: -1.38, domain_sd: -12.3, order: 0.864 | cells_per_sd, domain_sd |
| equity.snowball.intraday.quad_v2 | carry | point_gamma | 2.51e-14 | cells_per_sd: 3.02, domain_sd: -0.221, order: -0.0232 | order, domain_sd |
| equity.snowball.intraday.quad_v2 | uniform_profile | pv | 9.41e-13 | cells_per_sd: 1.37, domain_sd: -7.83, order: -0.176 | order, domain_sd |
| equity.snowball.intraday.quad_v2 | uniform_profile | desk_delta | 1.48e-13 | cells_per_sd: --, domain_sd: -17.8, order: 1.76 | domain_sd |
| equity.snowball.intraday.quad_v2 | uniform_profile | desk_gamma | 3.33e-13 | cells_per_sd: 0.585, domain_sd: -29.8, order: 3.89 | domain_sd |
| equity.snowball.intraday.quad_v2 | uniform_profile | desk_theta | 4.63e-11 | cells_per_sd: 5.97, domain_sd: 22.3, order: 9.67 | -- |
| equity.snowball.intraday.quad_v2 | uniform_profile | point_delta | 2.14e-13 | cells_per_sd: 1.95, domain_sd: -23.2, order: -5.07 | order, domain_sd |
| equity.snowball.intraday.quad_v2 | uniform_profile | point_gamma | 6e-13 | cells_per_sd: -2.43, domain_sd: -15.6, order: 4.07 | cells_per_sd, domain_sd |
| equity.snowball.intraday.quad_v2 | sessions_only_profile | pv | 9.73e-13 | cells_per_sd: 4, domain_sd: -6.93, order: 0.186 | domain_sd |
| equity.snowball.intraday.quad_v2 | sessions_only_profile | desk_delta | 1.23e-13 | cells_per_sd: -2, domain_sd: -5.96, order: 2.08 | cells_per_sd, domain_sd |
| equity.snowball.intraday.quad_v2 | sessions_only_profile | desk_gamma | 2.14e-13 | cells_per_sd: 2.46, domain_sd: 5.5, order: 6.35 | -- |
| equity.snowball.intraday.quad_v2 | sessions_only_profile | desk_theta | 4.62e-11 | cells_per_sd: 7.69, domain_sd: 20.4, order: 10.3 | -- |
| equity.snowball.intraday.quad_v2 | sessions_only_profile | point_delta | 1.24e-13 | cells_per_sd: 0.212, domain_sd: -5.23, order: 0.154 | domain_sd |
| equity.snowball.intraday.quad_v2 | sessions_only_profile | point_gamma | 2.12e-13 | cells_per_sd: 1.63, domain_sd: 4.08, order: 3.71 | -- |
| equity.snowball.intraday.quad_v2 | on_ki_barrier_at_close | pv | 1.03e-12 | cells_per_sd: 0, domain_sd: 4.38, order: 0.644 | -- |
| equity.snowball.intraday.quad_v2 | on_ki_barrier_at_close | desk_delta | 7.78e-11 | cells_per_sd: 8.98, domain_sd: 29.5, order: 14.6 | -- |
| equity.snowball.intraday.quad_v2 | on_ki_barrier_at_close | desk_gamma | 2.07e-10 | cells_per_sd: --, domain_sd: 28.5, order: 15.3 | -- |
| equity.snowball.intraday.pde | ordinary | pv | 0.00169 | space: -2.53, time: 1.9 | space |
| equity.snowball.intraday.pde | ordinary | desk_delta | 0.00013 | space: 1.71, time: 1.9 | -- |
| equity.snowball.intraday.pde | ordinary | desk_gamma | 0.000937 | space: 0.0508, time: 1.89 | -- |
| equity.snowball.intraday.pde | ordinary | desk_theta | 0.0038 | space: 0.924, time: 2.52 | -- |
| equity.snowball.intraday.pde | ordinary | point_delta | 0.000226 | space: 1.67, time: 1.9 | -- |
| equity.snowball.intraday.pde | ordinary | point_gamma | 0.000294 | space: -0.0877, time: 1.89 | space |
| equity.snowball.intraday.pde | near_ki_above | pv | 0.00381 | space: -1.81, time: 1.87 | space |
| equity.snowball.intraday.pde | near_ki_above | desk_delta | 0.00423 | space: 1.75, time: 1.53 | -- |
| equity.snowball.intraday.pde | near_ki_above | desk_gamma | 0.0102 | space: 2.58, time: -1.43 | time |
| equity.snowball.intraday.pde | near_ki_above | desk_theta | 0.0297 | space: 3.7, time: 3.77 | -- |
| equity.snowball.intraday.pde | near_ki_above | point_delta | 0.00924 | space: -0.55, time: 1.9 | space |
| equity.snowball.intraday.pde | near_ki_above | point_gamma | 0.0147 | space: 2.76, time: 2 | -- |
| equity.snowball.intraday.pde | near_ki_below | pv | 0.00314 | space: -0.181, time: 1.84 | space |
| equity.snowball.intraday.pde | near_ki_below | desk_delta | 0.00297 | space: 2.08, time: 1.01 | -- |
| equity.snowball.intraday.pde | near_ki_below | desk_gamma | 0.0142 | space: 2.47, time: 0.0156 | -- |
| equity.snowball.intraday.pde | near_ki_below | desk_theta | 0.00324 | space: 0.183, time: 1.67 | -- |
| equity.snowball.intraday.pde | near_ki_below | point_delta | 0.0132 | space: 0.465, time: 1.88 | -- |
| equity.snowball.intraday.pde | near_ki_below | point_gamma | 0.026 | space: 0.875, time: 1.92 | -- |
| equity.snowball.intraday.pde | near_ki_1s | pv | 0.016 | space: 3.51, time: 0.365 | -- |
| equity.snowball.intraday.pde | near_ki_1s | desk_delta | 0.0353 | space: 2.61, time: 0.3 | -- |
| equity.snowball.intraday.pde | near_ki_1s | desk_gamma | 0.0371 | space: -2.54, time: 0.205 | space |
| equity.snowball.intraday.pde | near_ki_1s | desk_theta | 199 | space: 18.9, time: 20.1 | -- |
| equity.snowball.intraday.pde | near_ki_1s | point_delta | 0.0704 | space: 5.86, time: 0.278 | -- |
| equity.snowball.intraday.pde | near_ki_1s | point_gamma | 0.00427 | space: 0.304, time: 2.49 | -- |
| equity.snowball.intraday.pde | near_ki_10s | pv | -- | -- | -- |
| equity.snowball.intraday.pde | near_ki_10s | desk_delta | -- | -- | -- |
| equity.snowball.intraday.pde | near_ki_10s | desk_gamma | -- | -- | -- |
| equity.snowball.intraday.pde | near_ki_10s | desk_theta | -- | -- | -- |
| equity.snowball.intraday.pde | near_ki_10s | point_delta | -- | -- | -- |
| equity.snowball.intraday.pde | near_ki_10s | point_gamma | -- | -- | -- |
| equity.snowball.intraday.pde | near_ki_5m | pv | 0.004 | space: 3.11, time: 1.83 | -- |
| equity.snowball.intraday.pde | near_ki_5m | desk_delta | 0.00411 | space: 1.3, time: 1.88 | -- |
| equity.snowball.intraday.pde | near_ki_5m | desk_gamma | 0.0115 | space: 3.09, time: 1.98 | -- |
| equity.snowball.intraday.pde | near_ki_5m | desk_theta | 0.349 | space: 5.88, time: 12.8 | -- |
| equity.snowball.intraday.pde | near_ki_5m | point_delta | 0.015 | space: 2.04, time: 1.9 | -- |
| equity.snowball.intraday.pde | near_ki_5m | point_gamma | 0.222 | space: 2.2, time: 2.32 | -- |
| equity.snowball.intraday.pde | near_ko_on_ko_day | pv | 0.000933 | space: 0.614, time: 1.91 | -- |
| equity.snowball.intraday.pde | near_ko_on_ko_day | desk_delta | 0.00307 | space: 2.56, time: 3.03 | -- |
| equity.snowball.intraday.pde | near_ko_on_ko_day | desk_gamma | 0.00226 | space: 1.26, time: 5.33 | -- |
| equity.snowball.intraday.pde | near_ko_on_ko_day | desk_theta | 0.000815 | space: -0.437, time: 3.32 | space |
| equity.snowball.intraday.pde | near_ko_on_ko_day | point_delta | 0.0105 | space: 0.612, time: 1.95 | -- |
| equity.snowball.intraday.pde | near_ko_on_ko_day | point_gamma | 0.00765 | space: 1.21, time: 1.95 | -- |
| equity.snowball.intraday.pde | near_ko_on_ko_day_1s | pv | 0.00121 | space: 4.45, time: 0.466 | -- |
| equity.snowball.intraday.pde | near_ko_on_ko_day_1s | desk_delta | 0.00153 | space: 2.27, time: 0.481 | -- |
| equity.snowball.intraday.pde | near_ko_on_ko_day_1s | desk_gamma | 0.000687 | space: 3.77, time: 0.531 | -- |
| equity.snowball.intraday.pde | near_ko_on_ko_day_1s | desk_theta | 14.6 | space: 0.342, time: 1.09 | -- |
| equity.snowball.intraday.pde | near_ko_on_ko_day_1s | point_delta | 0.00178 | space: 2.79, time: 0.492 | -- |
| equity.snowball.intraday.pde | near_ko_on_ko_day_1s | point_gamma | 0.000252 | space: 3.7, time: 0.432 | -- |
| equity.snowball.intraday.pde | ko_level_on_ki_day | pv | 0.000272 | space: 2.36, time: 1.89 | -- |
| equity.snowball.intraday.pde | ko_level_on_ki_day | desk_delta | 0.000905 | space: 3.89, time: 1.88 | -- |
| equity.snowball.intraday.pde | ko_level_on_ki_day | desk_gamma | 0.000233 | space: 0.631, time: 1.44 | -- |
| equity.snowball.intraday.pde | ko_level_on_ki_day | desk_theta | 0.000331 | space: 1.43, time: 2.58 | -- |
| equity.snowball.intraday.pde | ko_level_on_ki_day | point_delta | 0.00101 | space: 0.0883, time: 1.9 | -- |
| equity.snowball.intraday.pde | ko_level_on_ki_day | point_gamma | 6.67e-05 | space: 2, time: 1.9 | -- |
| equity.snowball.intraday.pde | pre_open | pv | 0.00397 | space: 5.73, time: 1.9 | -- |
| equity.snowball.intraday.pde | pre_open | desk_delta | 0.000237 | space: 2.7, time: 1.9 | -- |
| equity.snowball.intraday.pde | pre_open | desk_gamma | 0.00456 | space: 4.26, time: 1.89 | -- |
| equity.snowball.intraday.pde | pre_open | desk_theta | 0.00597 | space: 7.55, time: 1.85 | -- |
| equity.snowball.intraday.pde | pre_open | point_delta | 0.000402 | space: 3.94, time: 1.91 | -- |
| equity.snowball.intraday.pde | pre_open | point_gamma | 0.000579 | space: 2.18, time: 1.89 | -- |
| equity.snowball.intraday.pde | lunch_break | pv | 0.0117 | space: 0.519, time: 1.89 | -- |
| equity.snowball.intraday.pde | lunch_break | desk_delta | 0.00808 | space: 5.22, time: 1.93 | -- |
| equity.snowball.intraday.pde | lunch_break | desk_gamma | 0.0251 | space: 3.79, time: 1.81 | -- |
| equity.snowball.intraday.pde | lunch_break | desk_theta | 0.00492 | space: -5.36, time: 1.71 | space |
| equity.snowball.intraday.pde | lunch_break | point_delta | 0.00442 | space: 0.805, time: 1.9 | -- |
| equity.snowball.intraday.pde | lunch_break | point_gamma | 0.0145 | space: 2.96, time: 1.91 | -- |
| equity.snowball.intraday.pde | lunch_break_zero_variance | pv | 0.0147 | space: 0.799, time: 1.88 | -- |
| equity.snowball.intraday.pde | lunch_break_zero_variance | desk_delta | 0.0081 | space: -1.91, time: 2.05 | space |
| equity.snowball.intraday.pde | lunch_break_zero_variance | desk_gamma | 0.0242 | space: 3.25, time: 1.16 | -- |
| equity.snowball.intraday.pde | lunch_break_zero_variance | desk_theta | 7.72e-05 | space: 2.92, time: 2.01 | -- |
| equity.snowball.intraday.pde | lunch_break_zero_variance | point_delta | 0.00768 | space: 1.1, time: 1.88 | -- |
| equity.snowball.intraday.pde | lunch_break_zero_variance | point_gamma | 0.0161 | space: 6.05, time: 1.92 | -- |
| equity.snowball.intraday.pde | holiday_eve | pv | 0.01 | space: 1.39, time: 1.84 | -- |
| equity.snowball.intraday.pde | holiday_eve | desk_delta | 0.00468 | space: 1.71, time: 1.87 | -- |
| equity.snowball.intraday.pde | holiday_eve | desk_gamma | 0.0184 | space: 4.83, time: 1.7 | -- |
| equity.snowball.intraday.pde | holiday_eve | desk_theta | 0.0123 | space: 1.02, time: 0.947 | -- |
| equity.snowball.intraday.pde | holiday_eve | point_delta | 0.00502 | space: 3.86, time: 1.84 | -- |
| equity.snowball.intraday.pde | holiday_eve | point_gamma | 0.0116 | space: 4.46, time: 1.71 | -- |
| equity.snowball.intraday.pde | maturity_day | pv | 4.92e-13 | space: 0.0656, time: -0.989 | time |
| equity.snowball.intraday.pde | maturity_day | desk_delta | 1.39e-13 | space: -4.72, time: 1.14 | space |
| equity.snowball.intraday.pde | maturity_day | desk_gamma | 2e-12 | space: 0.358, time: -1.46 | time |
| equity.snowball.intraday.pde | maturity_day | desk_theta | 4.92e-13 | space: 0.0656, time: -0.989 | time |
| equity.snowball.intraday.pde | maturity_day | point_delta | 0 | space: --, time: -- | -- |
| equity.snowball.intraday.pde | maturity_day | point_gamma | 0 | space: --, time: -- | -- |
| equity.snowball.intraday.pde | already_ki | pv | -- | -- | -- |
| equity.snowball.intraday.pde | already_ki | desk_delta | -- | -- | -- |
| equity.snowball.intraday.pde | already_ki | desk_gamma | -- | -- | -- |
| equity.snowball.intraday.pde | already_ki | desk_theta | -- | -- | -- |
| equity.snowball.intraday.pde | already_ki | point_delta | -- | -- | -- |
| equity.snowball.intraday.pde | already_ki | point_gamma | -- | -- | -- |
| equity.snowball.intraday.pde | terminated_pending_cash | pv | 0 | -- | -- |
| equity.snowball.intraday.pde | terminated_pending_cash | desk_delta | 0 | -- | -- |
| equity.snowball.intraday.pde | terminated_pending_cash | desk_gamma | 0 | -- | -- |
| equity.snowball.intraday.pde | terminated_pending_cash | desk_theta | 0 | -- | -- |
| equity.snowball.intraday.pde | terminated_pending_cash | point_delta | 0 | -- | -- |
| equity.snowball.intraday.pde | terminated_pending_cash | point_gamma | 0 | -- | -- |
| equity.snowball.intraday.pde | provisional_ki_after_close | pv | 0.000128 | space: 5.67, time: 2 | -- |
| equity.snowball.intraday.pde | provisional_ki_after_close | desk_delta | 0.000128 | space: 2.44, time: 11 | -- |
| equity.snowball.intraday.pde | provisional_ki_after_close | desk_gamma | 0.000765 | space: 1.71, time: 15.8 | -- |
| equity.snowball.intraday.pde | provisional_ki_after_close | desk_theta | 4.81e-05 | space: 4.63, time: 19.1 | -- |
| equity.snowball.intraday.pde | provisional_ki_after_close | point_delta | 4.32e-06 | space: -0.479, time: 2 | space |
| equity.snowball.intraday.pde | provisional_ki_after_close | point_gamma | 8.31e-07 | space: -0.77, time: 2 | space |
| equity.snowball.intraday.pde | low_vol | pv | 0.00929 | space: 2.81, time: 1.87 | -- |
| equity.snowball.intraday.pde | low_vol | desk_delta | 0.00686 | space: 4.51, time: 1.36 | -- |
| equity.snowball.intraday.pde | low_vol | desk_gamma | 0.02 | space: 4.12, time: 1.91 | -- |
| equity.snowball.intraday.pde | low_vol | desk_theta | 0.0231 | space: 3.23, time: 2.4 | -- |
| equity.snowball.intraday.pde | low_vol | point_delta | 0.0235 | space: 3.03, time: 1.9 | -- |
| equity.snowball.intraday.pde | low_vol | point_gamma | 0.196 | space: 2.45, time: 1.64 | -- |
| equity.snowball.intraday.pde | high_vol | pv | 0.00165 | space: -4.25, time: 1.85 | space |
| equity.snowball.intraday.pde | high_vol | desk_delta | 0.00258 | space: 2.09, time: 1.79 | -- |
| equity.snowball.intraday.pde | high_vol | desk_gamma | 0.00719 | space: 0.933, time: 1.22 | -- |
| equity.snowball.intraday.pde | high_vol | desk_theta | 0.00168 | space: 1.27, time: -2.7 | time |
| equity.snowball.intraday.pde | high_vol | point_delta | 0.00357 | space: -1.92, time: 1.9 | space |
| equity.snowball.intraday.pde | high_vol | point_gamma | 0.00228 | space: -0.251, time: 1.95 | space |
| equity.snowball.intraday.pde | carry | pv | 0.00167 | space: -2.59, time: 1.9 | space |
| equity.snowball.intraday.pde | carry | desk_delta | 0.000127 | space: 1.72, time: 1.9 | -- |
| equity.snowball.intraday.pde | carry | desk_gamma | 0.000924 | space: 0.0459, time: 1.89 | -- |
| equity.snowball.intraday.pde | carry | desk_theta | 0.00374 | space: 0.924, time: 2.52 | -- |
| equity.snowball.intraday.pde | carry | point_delta | 0.000221 | space: 1.66, time: 1.9 | -- |
| equity.snowball.intraday.pde | carry | point_gamma | 0.000288 | space: -0.0823, time: 1.89 | space |
| equity.snowball.intraday.pde | uniform_profile | pv | 0.00145 | space: 0.361, time: 1.99 | -- |
| equity.snowball.intraday.pde | uniform_profile | desk_delta | 0.000387 | space: 3.31, time: -2.6 | time |
| equity.snowball.intraday.pde | uniform_profile | desk_gamma | 0.00599 | space: 2.69, time: 6.53 | -- |
| equity.snowball.intraday.pde | uniform_profile | desk_theta | 0.0327 | space: 6.23, time: 9 | -- |
| equity.snowball.intraday.pde | uniform_profile | point_delta | 0.00759 | space: 2.16, time: 2.15 | -- |
| equity.snowball.intraday.pde | uniform_profile | point_gamma | 0.118 | space: 5.34, time: 1.98 | -- |
| equity.snowball.intraday.pde | sessions_only_profile | pv | 0.00789 | space: 2.94, time: 1.86 | -- |
| equity.snowball.intraday.pde | sessions_only_profile | desk_delta | 0.0109 | space: 2.39, time: 2.18 | -- |
| equity.snowball.intraday.pde | sessions_only_profile | desk_gamma | 0.0168 | space: 4.16, time: 2.94 | -- |
| equity.snowball.intraday.pde | sessions_only_profile | desk_theta | 0.0236 | space: 3, time: 1.93 | -- |
| equity.snowball.intraday.pde | sessions_only_profile | point_delta | 0.0159 | space: 4.42, time: 1.89 | -- |
| equity.snowball.intraday.pde | sessions_only_profile | point_gamma | 0.0193 | space: 2.51, time: 1.95 | -- |
| equity.snowball.intraday.pde | on_ki_barrier_at_close | pv | 0.000237 | space: 1.58, time: 2 | -- |
| equity.snowball.intraday.pde | on_ki_barrier_at_close | desk_delta | 0.0129 | space: 2.5, time: 6.13 | -- |
| equity.snowball.intraday.pde | on_ki_barrier_at_close | desk_gamma | 0.0328 | space: 2.53, time: 6.09 | -- |

## Aggregate bias

| candidate | quantity | cells | mean bias (c) | mean radius (c) | passed |
|---|---|---|---|---|---|
| equity.snowball.intraday.quad_v2 | pv | 23 | -0.0001466 | 0.0116 | yes |
| equity.snowball.intraday.quad_v2 | desk_delta | 23 | -6.808e-05 | 0.000275 | yes |
| equity.snowball.intraday.quad_v2 | desk_gamma | 23 | 0.0001174 | 0.000532 | yes |
| equity.snowball.intraday.quad_v2 | desk_theta | 22 | -1.55e-05 | 0.00042 | yes |
| equity.snowball.intraday.quad_v2 | point_delta | 22 | -7.621e-05 | 0.000312 | yes |
| equity.snowball.intraday.quad_v2 | point_gamma | 22 | 7.982e-05 | 0.00214 | yes |
| equity.snowball.intraday.pde | pv | 21 | 46.67 | 0.0126 | no |
| equity.snowball.intraday.pde | desk_delta | 21 | 21.37 | 0.000284 | no |
| equity.snowball.intraday.pde | desk_gamma | 21 | 16.83 | 0.000553 | no |
| equity.snowball.intraday.pde | desk_theta | 20 | -4.761e+04 | 0.000446 | no |
| equity.snowball.intraday.pde | point_delta | 20 | 25.9 | 0.00033 | no |
| equity.snowball.intraday.pde | point_gamma | 20 | 2.543 | 0.00233 | no |
