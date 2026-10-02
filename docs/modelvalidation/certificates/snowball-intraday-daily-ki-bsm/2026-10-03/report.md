# Certification report: snowball-intraday-daily-ki-bsm

Evidence digest: `5368aebd01d2fb052b5a1d7809eb9121f0a77576669845dbc752f24a2a216000`

Machine: `arm64` / macOS-27.0-arm64-arm-64bit - Python 3.11.8, NumPy 2.4.6, quantark `53731491b177cb01b530e6e9cafd31d4b4d50d37`

## Amendment

| field | value |
|---|---|
| parent | docs/modelvalidation/certificates/snowball-intraday-daily-ki-bsm/2026-09-19/certificate.json |
| parent digest | `a0569e97687cc01cc67df75b00612b3fc6a3c4e215ffe7bde5b3a2ede9a9893d` |
| reason | PDE 1D numerical policy (sub-project 3A, 0.8.0): grid day from the environment, max_steps 12,000 and no scaling, base-family projection, Snowball NODAL, Phoenix VALUE, half-step damping where monitoring is discrete, forward far-field rows; docs/superpowers/specs/2026-10-01-pde-1d-numerical-policy-design.md |
| re-priced | 276 cell(s) |
| carried forward | 0 cell(s) |

## Decisions

| candidate | decision |
|---|---|
| equity.snowball.intraday.pde | REJECTED |
| equity.snowball.intraday.quad_v2 | ADMITTED |

Bounds: cell 1 c, mean signed bias 0.2 c, reference radius at most 0.25 x cell (deterministic reference: a radius is consumed whole, with no interval multiplier).

Schema 2: gate values are a fraction of each cell's own budget (quantity_bounds below); the cell bound of 1 is that budget.

| quantity | abs floor | relative |
|---|---|---|
| pv | 1e-06 | 0 |
| desk_delta | 1e-05 | 0.0001 |
| desk_gamma | 0.0001 | 0.001 |
| desk_theta | 1e-06 | 0.0001 |
| point_delta | 1e-05 | 0.0001 |
| point_gamma | 0.0001 | 0.001 |

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
| equity.snowball.intraday.pde | grid.bounds | (None, None) |
| equity.snowball.intraday.pde | grid.day_count | 244 |
| equity.snowball.intraday.pde | grid.discrete_monitoring_damping | ['half_step', 1, 1] |
| equity.snowball.intraday.pde | grid.eps_crit | 0.003 |
| equity.snowball.intraday.pde | grid.event_damping_steps | 2 |
| equity.snowball.intraday.pde | grid.max_points | 2000 |
| equity.snowball.intraday.pde | grid.max_steps | 12000 |
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
| exactness_bases | terminated: no contingent claim remains: the value is the pending receivables, discounted in closed form; decided_at_valuation: the only remaining instant is the valuation instant: its events are decided at the known spot; single_instant_quadrature: one instant remains: value and point derivatives are single Gaussian integrals, taken by adaptive Gauss-Kronrod quadrature split at every jump and kink, with no grid |
| ladder_policy | version: 2; radius_kind: calibrated_numerical_estimate; parameters: discretization_order: 2; refinement_ratio: 2; min_levels: 5; order_window: 1.5, 2.5; contraction: 0.5; max_contraction: 16.0; unconfirmed_safety: 3.0; stagnation_floor: 1e-13; roundoff_factor: 16.0; branches: exact: the last two differences are below the stagnation floor AND the solve names an exactness basis; value: finest level; radius: the larger difference; stagnant: the last two differences are below the stagnation floor and no exactness basis is named: equal values do not establish exactness; radius: infinite; geometric: the last two differences share a sign with observed order in [1.5, 2.5] and successive extrapolants contract to 0.5 or less; value: Richardson extrapolant (4 v_n - v_(n-1)) / 3; radius: the last extrapolant spread, never below 1/16.0 of the previous spread; correction: order confirmed as for geometric, extrapolants do not contract; value: Richardson extrapolant; radius: the whole correction |E_n - v_n|; unextrapolated: order not confirmed (outside the window, or alternating differences) and the last difference did not grow; value: finest level; radius: 3.0 x the previous difference; unbounded: the last difference grew; value: finest level; radius: infinite; uncalibrated: the same rule applied one level down (without the finest level) did not cover this value with its own radius; radius: infinite; calibration: in-ladder: |value - value one level down| <= radius one level down, recorded per quantity; an infinite radius one level down is no evidence and does not calibrate |
| levels | 2001, 4001, 8001, 16001, 32001 |
| local_grid | (points - 1) / 4 + 1 uniform knots over +/- 15 first-interval standard deviations, nested |
| method | backward Gaussian transitions of a piecewise-linear value function of ln S with exact barrier jumps; closed-form expectations; an exact last step with analytic first and second derivatives |
| radius | analytical: the exactness basis's own error; calibrated_estimate: the ladder policy's radius plus the residual components, scaled to the quantity |
| residual_components | truncation: analytical bound: value span x reflection-principle mass beyond the flat-extended grids; saturation: analytical bound: 4 x Phi(-9) x value span x instants; roundoff: estimate: roundoff_factor x eps x value span x instants; scaling: pv 1; desk delta 1/(S h); desk gamma 4/(S h)^2; desk theta 2/step; point delta E|Z|/(S sd1); point gamma (E|Z^2-1|/sd1^2 + E|Z|/sd1)/S^2, sd1 the first interval's |
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
| already_ki | pv | 0.000717 | 0.00295 | 0.243 | yes |
| already_ki | desk_delta | -8.01e-05 | 0.00172 | -0.0465 | yes |
| already_ki | desk_gamma | 0.000443 | 0.00441 | 0.1 | yes |
| already_ki | desk_theta | 0.000821 | 0.00367 | 0.224 | yes |
| carry | pv | -0.00153 | 0.00351 | -0.437 | yes |
| carry | desk_delta | -0.000365 | 0.00161 | -0.227 | yes |
| carry | desk_gamma | 0.00452 | 0.00504 | 0.896 | yes |
| carry | desk_theta | 0.00261 | 0.00478 | 0.547 | yes |
| high_vol | pv | 0.00091 | 0.003 | 0.304 | yes |
| high_vol | desk_delta | -0.00189 | 0.0023 | -0.822 | yes |
| high_vol | desk_gamma | -0.00626 | 0.00663 | -0.945 | yes |
| high_vol | desk_theta | 0.00197 | 0.0046 | 0.427 | yes |
| holiday_eve | pv | -0.00513 | 0.00388 | -1.32 | yes |
| holiday_eve | desk_delta | -0.00498 | 0.00483 | -1.03 | yes |
| holiday_eve | desk_gamma | 0.0207 | 0.0123 | 1.68 | yes |
| holiday_eve | desk_theta | 0.00858 | 0.00714 | 1.2 | yes |
| ko_level_on_ki_day | pv | -0.000847 | 0.0024 | -0.352 | yes |
| ko_level_on_ki_day | desk_delta | 0.000499 | 0.00147 | 0.339 | yes |
| ko_level_on_ki_day | desk_gamma | 0.0042 | 0.00294 | 1.43 | yes |
| ko_level_on_ki_day | desk_theta | 0.00213 | 0.00363 | 0.587 | yes |
| low_vol | pv | -0.000754 | 0.00496 | -0.152 | yes |
| low_vol | desk_delta | 0.00456 | 0.00371 | 1.23 | yes |
| low_vol | desk_gamma | 0.0149 | 0.0182 | 0.823 | yes |
| low_vol | desk_theta | -0.00473 | 0.00628 | -0.753 | yes |
| lunch_break | pv | -0.0034 | 0.00392 | -0.867 | yes |
| lunch_break | desk_delta | -0.00229 | 0.00355 | -0.644 | yes |
| lunch_break | desk_gamma | 0.0279 | 0.0116 | 2.41 | yes |
| lunch_break | desk_theta | 0.00188 | 0.000702 | 2.68 | yes |
| lunch_break_zero_variance | pv | -0.00664 | 0.00465 | -1.43 | yes |
| lunch_break_zero_variance | desk_delta | -0.00322 | 0.00453 | -0.711 | yes |
| lunch_break_zero_variance | desk_gamma | 0.0114 | 0.0145 | 0.784 | yes |
| lunch_break_zero_variance | desk_theta | -0.000149 | 0.00113 | -0.132 | yes |
| maturity_day | pv | -3.57e-13 | 0 | -- | yes |
| maturity_day | desk_delta | 8.08e-14 | 0 | -- | yes |
| maturity_day | desk_gamma | 1.23e-12 | 0 | -- | yes |
| maturity_day | desk_theta | 3.57e-13 | 0 | -- | yes |
| near_ki_10s | pv | 0.00322 | 0.00346 | 0.931 | yes |
| near_ki_10s | desk_delta | 0.00253 | 0.0029 | 0.874 | yes |
| near_ki_10s | desk_gamma | -0.00505 | 0.0105 | -0.481 | yes |
| near_ki_10s | desk_theta | 2.07 | 1.57 | 1.32 | yes |
| near_ki_1s | pv | 0.00139 | 0.00277 | 0.502 | yes |
| near_ki_1s | desk_delta | 0.0047 | 0.00285 | 1.65 | yes |
| near_ki_1s | desk_gamma | 0.00942 | 0.00982 | 0.96 | yes |
| near_ki_1s | desk_theta | 14 | 16.9 | 0.831 | yes |
| near_ki_5m | pv | 0.000654 | 0.00326 | 0.2 | yes |
| near_ki_5m | desk_delta | -3.18e-05 | 0.00322 | -0.00987 | yes |
| near_ki_5m | desk_gamma | -0.00456 | 0.0136 | -0.337 | yes |
| near_ki_5m | desk_theta | -0.0235 | 0.0615 | -0.382 | yes |
| near_ki_above | pv | 0.00375 | 0.00359 | 1.04 | yes |
| near_ki_above | desk_delta | 0.00373 | 0.00289 | 1.29 | yes |
| near_ki_above | desk_gamma | -0.00149 | 0.0124 | -0.121 | yes |
| near_ki_above | desk_theta | -0.012 | 0.00505 | -2.37 | yes |
| near_ki_below | pv | 0.00547 | 0.00217 | 2.52 | yes |
| near_ki_below | desk_delta | 0.00485 | 0.00243 | 1.99 | yes |
| near_ki_below | desk_gamma | -0.00283 | 0.00922 | -0.307 | yes |
| near_ki_below | desk_theta | -0.00528 | 0.00222 | -2.37 | yes |
| near_ko_on_ko_day | pv | -0.00203 | 0.00238 | -0.853 | yes |
| near_ko_on_ko_day | desk_delta | 0.000269 | 0.00152 | 0.177 | yes |
| near_ko_on_ko_day | desk_gamma | 0.00323 | 0.00424 | 0.762 | yes |
| near_ko_on_ko_day | desk_theta | 0.00666 | 0.00341 | 1.96 | yes |
| near_ko_on_ko_day_1s | pv | 9.94e-05 | 0.00227 | 0.0437 | yes |
| near_ko_on_ko_day_1s | desk_delta | -0.000637 | 0.00117 | -0.547 | yes |
| near_ko_on_ko_day_1s | desk_gamma | 0.00105 | 0.00374 | 0.281 | yes |
| near_ko_on_ko_day_1s | desk_theta | 4.33 | 12.7 | 0.341 | yes |
| on_ki_barrier_at_close | pv | 0.000897 | 0.000678 | 1.32 | yes |
| on_ki_barrier_at_close | desk_delta | -0.00139 | 0.00343 | -0.406 | yes |
| on_ki_barrier_at_close | desk_gamma | -0.00359 | 0.00899 | -0.399 | yes |
| ordinary | pv | -0.00381 | 0.00313 | -1.22 | yes |
| ordinary | desk_delta | 0.00286 | 0.00189 | 1.51 | yes |
| ordinary | desk_gamma | 0.00614 | 0.00417 | 1.47 | yes |
| ordinary | desk_theta | -0.000128 | 0.00407 | -0.0314 | yes |
| pre_open | pv | 0.000431 | 0.00324 | 0.133 | yes |
| pre_open | desk_delta | -0.000406 | 0.00187 | -0.217 | yes |
| pre_open | desk_gamma | 0.000623 | 0.00379 | 0.164 | yes |
| pre_open | desk_theta | 0.000293 | 0.000693 | 0.423 | yes |
| provisional_ki_after_close | pv | 0.000452 | 0.000689 | 0.657 | yes |
| provisional_ki_after_close | desk_delta | -0.000132 | 0.000387 | -0.342 | yes |
| provisional_ki_after_close | desk_gamma | -0.000659 | 0.00155 | -0.425 | yes |
| provisional_ki_after_close | desk_theta | -2.4e-06 | 6.59e-05 | -0.0364 | yes |
| sessions_only_profile | pv | -0.00321 | 0.00271 | -1.18 | yes |
| sessions_only_profile | desk_delta | -0.0066 | 0.00262 | -2.52 | yes |
| sessions_only_profile | desk_gamma | -0.000181 | 0.00965 | -0.0188 | yes |
| sessions_only_profile | desk_theta | 0.00433 | 0.00491 | 0.882 | yes |
| terminated_pending_cash | pv | 0 | 0 | -- | yes |
| terminated_pending_cash | desk_delta | 0 | 0 | -- | yes |
| terminated_pending_cash | desk_gamma | 0 | 0 | -- | yes |
| terminated_pending_cash | desk_theta | 0 | 0 | -- | yes |
| uniform_profile | pv | -0.00295 | 0.00293 | -1.01 | yes |
| uniform_profile | desk_delta | -0.00682 | 0.00258 | -2.64 | yes |
| uniform_profile | desk_gamma | -0.00763 | 0.0104 | -0.734 | yes |
| uniform_profile | desk_theta | -0.00247 | 0.00486 | -0.508 | yes |

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
| equity.snowball.intraday.pde | ordinary | pv | 7.97428 | 1.64e-07 | 7.97475 | 4.652 | 4.654 | 6.326 | FAIL |
| equity.snowball.intraday.pde | ordinary | desk_delta | -0.164795 | 1.04e-08 | -0.164818 | -1.362 | 1.363 | 3.957 | FAIL |
| equity.snowball.intraday.pde | ordinary | desk_gamma | -0.09634 | 2.45e-08 | -0.0964589 | -1.234 | 1.234 | 7.804 | FAIL |
| equity.snowball.intraday.pde | ordinary | desk_theta | 0.014282 | 4.09e-09 | 0.0130905 | -11.92 | 11.92 | 27.37 | FAIL |
| equity.snowball.intraday.pde | ordinary | point_delta | -0.16652 | 1.06e-08 | -0.166428 | 5.484 | 5.485 | 7.186 | FAIL |
| equity.snowball.intraday.pde | ordinary | point_gamma | -0.0977908 | 2.66e-08 | -0.0978673 | -0.7825 | 0.7827 | 1.064 | FAIL |
| equity.snowball.intraday.pde | near_ki_above | pv | -22.4213 | 9.9e-08 | -22.42 | 12.79 | 12.79 | 20.96 | FAIL |
| equity.snowball.intraday.pde | near_ki_above | desk_delta | 2.61027 | 6.94e-08 | 2.61034 | 0.2415 | 0.2418 | 5.498 | FAIL |
| equity.snowball.intraday.pde | near_ki_above | desk_gamma | 0.516564 | 4.77e-08 | 0.520014 | 6.68 | 6.68 | 21.91 | FAIL |
| equity.snowball.intraday.pde | near_ki_above | desk_theta | 0.224767 | 2.68e-08 | 0.234279 | 95.12 | 95.12 | 313.6 | FAIL |
| equity.snowball.intraday.pde | near_ki_above | point_delta | 2.99352 | 9.95e-08 | 2.99708 | 11.91 | 11.91 | 17.81 | FAIL |
| equity.snowball.intraday.pde | near_ki_above | point_gamma | 0.120793 | 9.4e-08 | 0.116335 | -36.91 | 36.91 | 113.4 | FAIL |
| equity.snowball.intraday.pde | near_ki_below | pv | -23.2848 | 6.71e-08 | -23.2844 | 3.795 | 3.796 | 24.57 | FAIL |
| equity.snowball.intraday.pde | near_ki_below | desk_delta | 2.34313 | 6.97e-08 | 2.34324 | 0.4911 | 0.4914 | 8.595 | FAIL |
| equity.snowball.intraday.pde | near_ki_below | desk_gamma | 1.45744 | 4.04e-08 | 1.46211 | 3.207 | 3.207 | 10.43 | FAIL |
| equity.snowball.intraday.pde | near_ki_below | desk_theta | -0.613422 | 4.12e-08 | -0.613692 | -2.699 | 2.7 | 25.54 | FAIL |
| equity.snowball.intraday.pde | near_ki_below | point_delta | 2.66348 | 1.03e-07 | 2.66402 | 2.054 | 2.054 | 39.08 | FAIL |
| equity.snowball.intraday.pde | near_ki_below | point_gamma | 2.05773 | 7.65e-08 | 2.06076 | 1.47 | 1.47 | 9.421 | FAIL |
| equity.snowball.intraday.pde | near_ki_1s | pv | -22.1965 | 1.26e-07 | -22.1773 | 192.2 | 192.2 | 138.2 | FAIL |
| equity.snowball.intraday.pde | near_ki_1s | desk_delta | 2.68629 | 7.72e-08 | 2.72908 | 159.3 | 159.3 | 114.1 | FAIL |
| equity.snowball.intraday.pde | near_ki_1s | desk_gamma | -0.48681 | 1.5e-07 | -0.441265 | 93.56 | 93.56 | 66.71 | FAIL |
| equity.snowball.intraday.pde | near_ki_1s | desk_theta | -0.0436783 | 2.67e-07 | -30.3884 | -3.034e+05 | 3.034e+05 | 8.009e+05 | FAIL |
| equity.snowball.intraday.pde | near_ki_1s | point_delta | 2.36023 | 4.4e-08 | 2.43995 | 337.8 | 337.8 | 279.9 | FAIL |
| equity.snowball.intraday.pde | near_ki_1s | point_gamma | 0.532522 | 8.35e-07 | 0.53912 | 12.39 | 12.39 | 14.51 | FAIL |
| equity.snowball.intraday.pde | near_ki_10s | pv | -22.1964 | 1.26e-07 | -22.191 | 54.04 | 54.04 | -- | UNRESOLVED |
| equity.snowball.intraday.pde | near_ki_10s | desk_delta | 2.6863 | 7.72e-08 | 2.69847 | 45.29 | 45.29 | -- | UNRESOLVED |
| equity.snowball.intraday.pde | near_ki_10s | desk_gamma | -0.487153 | 1.5e-07 | -0.474185 | 26.62 | 26.62 | -- | UNRESOLVED |
| equity.snowball.intraday.pde | near_ki_10s | desk_theta | -0.0436754 | 3.03e-08 | 1.89552 | 1.939e+04 | 1.939e+04 | -- | UNRESOLVED |
| equity.snowball.intraday.pde | near_ki_10s | point_delta | 2.36012 | 4.1e-08 | 2.37379 | 57.92 | 57.92 | -- | UNRESOLVED |
| equity.snowball.intraday.pde | near_ki_10s | point_gamma | 0.532451 | 1.28e-07 | 0.546672 | 26.71 | 26.71 | -- | UNRESOLVED |
| equity.snowball.intraday.pde | near_ki_5m | pv | -22.239 | 1.22e-07 | -22.2401 | -10.97 | 10.97 | 25.78 | FAIL |
| equity.snowball.intraday.pde | near_ki_5m | desk_delta | 2.68679 | 7.7e-08 | 2.68805 | 4.694 | 4.694 | 3.771 | FAIL |
| equity.snowball.intraday.pde | near_ki_5m | desk_gamma | -0.335102 | 1.36e-07 | -0.327866 | 21.59 | 21.59 | 24.94 | FAIL |
| equity.snowball.intraday.pde | near_ki_5m | desk_theta | 0.509623 | 4.99e-08 | 0.652273 | 1427 | 1427 | 3980 | FAIL |
| equity.snowball.intraday.pde | near_ki_5m | point_delta | 3.19392 | 1.08e-07 | 3.19041 | -11.01 | 11.01 | 32.41 | FAIL |
| equity.snowball.intraday.pde | near_ki_5m | point_gamma | -11.7084 | 1.06e-06 | -11.4824 | 19.3 | 19.3 | 18.6 | FAIL |
| equity.snowball.intraday.pde | near_ko_on_ko_day | pv | 7.22688 | 7.7e-09 | 7.22684 | -0.3573 | 0.3574 | 7.953 | FAIL |
| equity.snowball.intraday.pde | near_ko_on_ko_day | desk_delta | -0.972193 | 7.63e-09 | -0.969629 | 26.37 | 26.37 | 30.49 | FAIL |
| equity.snowball.intraday.pde | near_ko_on_ko_day | desk_gamma | -0.154872 | 1.51e-09 | -0.153935 | 6.049 | 6.049 | 14.89 | FAIL |
| equity.snowball.intraday.pde | near_ko_on_ko_day | desk_theta | 0.828925 | 4.08e-09 | 0.828702 | -2.236 | 2.236 | 6.776 | FAIL |
| equity.snowball.intraday.pde | near_ko_on_ko_day | point_delta | -1.49252 | 1.06e-08 | -1.49683 | -28.91 | 28.91 | 65.41 | FAIL |
| equity.snowball.intraday.pde | near_ko_on_ko_day | point_gamma | -0.370089 | 8.58e-10 | -0.368484 | 4.337 | 4.337 | 19.01 | FAIL |
| equity.snowball.intraday.pde | near_ko_on_ko_day_1s | pv | 8.0558 | 1.17e-08 | 8.05724 | 14.32 | 14.32 | 10.97 | FAIL |
| equity.snowball.intraday.pde | near_ko_on_ko_day_1s | desk_delta | -1.03888 | 8.02e-09 | -1.04063 | -16.78 | 16.78 | 11.92 | FAIL |
| equity.snowball.intraday.pde | near_ko_on_ko_day_1s | desk_gamma | -1.77076 | 6.66e-09 | -1.77008 | 0.3853 | 0.3853 | 0.19 | PASS |
| equity.snowball.intraday.pde | near_ko_on_ko_day_1s | desk_theta | 0.00175132 | 2.54e-07 | -6.08783 | -6.09e+04 | 6.09e+04 | 1.432e+05 | FAIL |
| equity.snowball.intraday.pde | near_ko_on_ko_day_1s | point_delta | -0.135401 | 7.48e-09 | -0.137382 | -146.3 | 146.3 | 98.57 | FAIL |
| equity.snowball.intraday.pde | near_ko_on_ko_day_1s | point_gamma | -0.0111397 | 4.07e-07 | -0.0108557 | 25.5 | 25.53 | 17.2 | FAIL |
| equity.snowball.intraday.pde | ko_level_on_ki_day | pv | 7.17628 | 7.24e-08 | 7.17631 | 0.2663 | 0.267 | 1.66 | FAIL |
| equity.snowball.intraday.pde | ko_level_on_ki_day | desk_delta | -0.325387 | 3.22e-08 | -0.325627 | -7.372 | 7.373 | 9.891 | FAIL |
| equity.snowball.intraday.pde | ko_level_on_ki_day | desk_gamma | 0.00215328 | 8.15e-09 | 0.00222761 | 34.52 | 34.52 | 89.08 | FAIL |
| equity.snowball.intraday.pde | ko_level_on_ki_day | desk_theta | -0.00032688 | 1.44e-09 | -0.000411989 | -0.8511 | 0.8511 | 2.253 | FAIL |
| equity.snowball.intraday.pde | ko_level_on_ki_day | point_delta | -0.333597 | 3.28e-08 | -0.333932 | -10.05 | 10.05 | 10.76 | FAIL |
| equity.snowball.intraday.pde | ko_level_on_ki_day | point_gamma | 0.00259306 | 8.8e-09 | 0.00260899 | 6.144 | 6.148 | 9.092 | FAIL |
| equity.snowball.intraday.pde | pre_open | pv | 7.92926 | 1.55e-07 | 7.92877 | -4.939 | 4.941 | 29.81 | FAIL |
| equity.snowball.intraday.pde | pre_open | desk_delta | -0.159805 | 1.08e-08 | -0.159517 | 18.02 | 18.02 | 13.93 | FAIL |
| equity.snowball.intraday.pde | pre_open | desk_gamma | -0.0887234 | 1.69e-08 | -0.0877393 | 11.09 | 11.09 | 49.55 | FAIL |
| equity.snowball.intraday.pde | pre_open | desk_theta | 0.00106058 | 4.43e-10 | 0.00354038 | 24.8 | 24.8 | 59.42 | FAIL |
| equity.snowball.intraday.pde | pre_open | point_delta | -0.161987 | 1.02e-08 | -0.161842 | 8.952 | 8.952 | 24.03 | FAIL |
| equity.snowball.intraday.pde | pre_open | point_gamma | -0.0899154 | 1.89e-08 | -0.0898014 | 1.268 | 1.268 | 4.62 | FAIL |
| equity.snowball.intraday.pde | lunch_break | pv | -20.2071 | 1.32e-07 | -20.2055 | 16.32 | 16.32 | 75.26 | FAIL |
| equity.snowball.intraday.pde | lunch_break | desk_delta | 2.66292 | 2.49e-08 | 2.6607 | -8.334 | 8.335 | 22.65 | FAIL |
| equity.snowball.intraday.pde | lunch_break | desk_gamma | -0.137078 | 5.84e-08 | -0.128391 | 63.38 | 63.38 | 166.7 | FAIL |
| equity.snowball.intraday.pde | lunch_break | desk_theta | 0.00279031 | 7.35e-10 | 0.00284702 | 0.5671 | 0.5671 | 48.65 | FAIL |
| equity.snowball.intraday.pde | lunch_break | point_delta | 2.66675 | 1.92e-08 | 2.66679 | 0.1273 | 0.1274 | 9.072 | FAIL |
| equity.snowball.intraday.pde | lunch_break | point_gamma | -0.201527 | 6.14e-08 | -0.193096 | 41.84 | 41.84 | 61.39 | FAIL |
| equity.snowball.intraday.pde | lunch_break_zero_variance | pv | -20.2163 | 1.29e-07 | -20.2138 | 24.44 | 24.44 | 65.62 | FAIL |
| equity.snowball.intraday.pde | lunch_break_zero_variance | desk_delta | 2.64135 | 2.48e-08 | 2.64147 | 0.4577 | 0.4578 | 15.74 | FAIL |
| equity.snowball.intraday.pde | lunch_break_zero_variance | desk_gamma | -0.0645224 | 5.06e-08 | -0.0558556 | 134.3 | 134.3 | 342.5 | FAIL |
| equity.snowball.intraday.pde | lunch_break_zero_variance | desk_theta | -0.000531722 | 1.33e-10 | -0.000549903 | -0.1818 | 0.1818 | 0.2372 | PASS |
| equity.snowball.intraday.pde | lunch_break_zero_variance | point_delta | 2.66537 | 2.2e-08 | 2.66496 | -1.547 | 1.547 | 13.79 | FAIL |
| equity.snowball.intraday.pde | lunch_break_zero_variance | point_gamma | -0.127976 | 5.56e-08 | -0.121292 | 52.23 | 52.23 | 93.04 | FAIL |
| equity.snowball.intraday.pde | holiday_eve | pv | -20.0159 | 1.37e-05 | -20.0122 | 37.92 | 38.06 | 47.81 | FAIL |
| equity.snowball.intraday.pde | holiday_eve | desk_delta | 2.8658 | 1.32e-08 | 2.86685 | 3.641 | 3.641 | 7.522 | FAIL |
| equity.snowball.intraday.pde | holiday_eve | desk_gamma | -0.304062 | 6.07e-07 | -0.309607 | -18.24 | 18.24 | 46.09 | FAIL |
| equity.snowball.intraday.pde | holiday_eve | desk_theta | -0.00170456 | 1.18e-07 | 0.000557268 | 22.62 | 22.62 | 100.6 | FAIL |
| equity.snowball.intraday.pde | holiday_eve | point_delta | 2.7757 | 7.86e-09 | 2.77787 | 7.817 | 7.817 | 8.587 | FAIL |
| equity.snowball.intraday.pde | holiday_eve | point_gamma | -0.181555 | 2.68e-07 | -0.185359 | -20.95 | 20.95 | 34.43 | FAIL |
| equity.snowball.intraday.pde | maturity_day | pv | 12 | 7.89e-12 | 12 | 4.761e-09 | 8.368e-08 | 4.547e-09 | PASS |
| equity.snowball.intraday.pde | maturity_day | desk_delta | 8.08242e-14 | 4.46e-12 | -3.19744e-14 | -1.128e-08 | 4.571e-07 | 1.315e-08 | PASS |
| equity.snowball.intraday.pde | maturity_day | desk_gamma | 1.23102e-12 | 2.47e-11 | -2.0961e-13 | -1.441e-06 | 2.614e-05 | 1.631e-06 | PASS |
| equity.snowball.intraday.pde | maturity_day | desk_theta | 4.10958e-05 | 7.9e-12 | 4.10958e-05 | -4.761e-09 | 8.378e-08 | 4.547e-09 | PASS |
| equity.snowball.intraday.pde | maturity_day | point_delta | 1.35417e-14 | 2.34e-11 | 0 | -1.354e-09 | 2.343e-06 | 0 | PASS |
| equity.snowball.intraday.pde | maturity_day | point_gamma | 1.48711e-12 | 1.32e-11 | -0 | -1.487e-06 | 1.466e-05 | 0 | PASS |
| equity.snowball.intraday.pde | already_ki | pv | -8.26798 | 3.17e-08 | -8.26808 | -0.9928 | 0.9931 | -- | UNRESOLVED |
| equity.snowball.intraday.pde | already_ki | desk_delta | 1.04517 | 8.27e-09 | 1.04516 | -0.09439 | 0.09447 | -- | UNRESOLVED |
| equity.snowball.intraday.pde | already_ki | desk_gamma | -0.00409272 | 1.23e-09 | -0.00389176 | 49.1 | 49.1 | -- | UNRESOLVED |
| equity.snowball.intraday.pde | already_ki | desk_theta | 0.000226998 | 1.93e-10 | 0.000317075 | 0.9008 | 0.9008 | -- | UNRESOLVED |
| equity.snowball.intraday.pde | already_ki | point_delta | 1.04544 | 8.28e-09 | 1.04543 | -0.1016 | 0.1017 | -- | UNRESOLVED |
| equity.snowball.intraday.pde | already_ki | point_gamma | -0.00407064 | 1.21e-09 | -0.00407224 | -0.3936 | 0.3939 | -- | UNRESOLVED |
| equity.snowball.intraday.pde | terminated_pending_cash | pv | 5.0605 | 4.49e-15 | 5.0605 | 0 | 4.495e-11 | 0 | PASS |
| equity.snowball.intraday.pde | terminated_pending_cash | desk_delta | 0 | 4.49e-15 | 0 | 0 | 4.495e-10 | 0 | PASS |
| equity.snowball.intraday.pde | terminated_pending_cash | desk_gamma | 0 | 1.8e-14 | 0 | 0 | 1.798e-08 | 0 | PASS |
| equity.snowball.intraday.pde | terminated_pending_cash | desk_theta | 1.73305e-05 | 8.99e-15 | 1.73305e-05 | 0 | 8.989e-11 | 0 | PASS |
| equity.snowball.intraday.pde | terminated_pending_cash | point_delta | 0 | 0 | 0 | 0 | 0 | 0 | PASS |
| equity.snowball.intraday.pde | terminated_pending_cash | point_gamma | 0 | 0 | 0 | 0 | 0 | 0 | PASS |
| equity.snowball.intraday.pde | provisional_ki_after_close | pv | -23.8982 | 1.1e-05 | -23.8981 | 1.098 | 1.208 | 1.276 | FAIL |
| equity.snowball.intraday.pde | provisional_ki_after_close | desk_delta | 1.00962 | 2.59e-09 | 1.00959 | -0.3627 | 0.3627 | 1.263 | FAIL |
| equity.snowball.intraday.pde | provisional_ki_after_close | desk_gamma | 0.00288312 | 6.49e-10 | 0.00266531 | -75.55 | 75.55 | 265.3 | FAIL |
| equity.snowball.intraday.pde | provisional_ki_after_close | desk_theta | -0.000272405 | 9.64e-11 | -0.000288191 | -0.1579 | 0.1579 | 0.4806 | PASS |
| equity.snowball.intraday.pde | provisional_ki_after_close | point_delta | 1.00959 | 2.66e-09 | 1.00959 | -0.003829 | 0.003856 | 0.04237 | PASS |
| equity.snowball.intraday.pde | provisional_ki_after_close | point_gamma | 0.00288347 | 3.22e-10 | 0.00288267 | -0.2772 | 0.2773 | 0.2841 | PASS |
| equity.snowball.intraday.pde | low_vol | pv | -21.6485 | 3.38e-07 | -21.6445 | 40.27 | 40.28 | 62.9 | FAIL |
| equity.snowball.intraday.pde | low_vol | desk_delta | 4.25292 | 2.59e-07 | 4.25367 | 1.783 | 1.784 | 7.237 | FAIL |
| equity.snowball.intraday.pde | low_vol | desk_gamma | 1.62662 | 3.68e-07 | 1.62157 | -3.1 | 3.101 | 13.12 | FAIL |
| equity.snowball.intraday.pde | low_vol | desk_theta | 0.204715 | 7.11e-08 | 0.212051 | 73.36 | 73.36 | 260.8 | FAIL |
| equity.snowball.intraday.pde | low_vol | point_delta | 5.54256 | 4.97e-07 | 5.54891 | 11.45 | 11.45 | 26.81 | FAIL |
| equity.snowball.intraday.pde | low_vol | point_gamma | -1.96134 | 1.35e-06 | -2.00794 | -23.76 | 23.76 | 99.3 | FAIL |
| equity.snowball.intraday.pde | high_vol | pv | -23.1511 | 3.56e-08 | -23.1507 | 4.487 | 4.487 | 10.87 | FAIL |
| equity.snowball.intraday.pde | high_vol | desk_delta | 1.40359 | 3.72e-09 | 1.40462 | 7.333 | 7.333 | 11.33 | FAIL |
| equity.snowball.intraday.pde | high_vol | desk_gamma | 0.0939431 | 9.24e-09 | 0.0948074 | 9.2 | 9.2 | 72.85 | FAIL |
| equity.snowball.intraday.pde | high_vol | desk_theta | 0.114381 | 4.55e-09 | 0.114727 | 3.458 | 3.458 | 21.69 | FAIL |
| equity.snowball.intraday.pde | high_vol | point_delta | 1.44503 | 4.81e-09 | 1.44592 | 6.137 | 6.137 | 16.85 | FAIL |
| equity.snowball.intraday.pde | high_vol | point_gamma | 0.0834244 | 9.3e-09 | 0.08354 | 1.386 | 1.386 | 22.2 | FAIL |
| equity.snowball.intraday.pde | carry | pv | 7.92602 | 1.61e-07 | 7.92648 | 4.571 | 4.573 | 6.235 | FAIL |
| equity.snowball.intraday.pde | carry | desk_delta | -0.158905 | 1.01e-08 | -0.158927 | -1.433 | 1.434 | 4.043 | FAIL |
| equity.snowball.intraday.pde | carry | desk_gamma | -0.094928 | 2.41e-08 | -0.0950441 | -1.223 | 1.223 | 7.822 | FAIL |
| equity.snowball.intraday.pde | carry | desk_theta | 0.0140883 | 4.03e-09 | 0.0129163 | -11.72 | 11.72 | 26.93 | FAIL |
| equity.snowball.intraday.pde | carry | point_delta | -0.160595 | 1.04e-08 | -0.160505 | 5.567 | 5.567 | 7.328 | FAIL |
| equity.snowball.intraday.pde | carry | point_gamma | -0.0963502 | 2.62e-08 | -0.096425 | -0.7764 | 0.7767 | 1.058 | FAIL |
| equity.snowball.intraday.pde | uniform_profile | pv | -22.3432 | 1.12e-07 | -22.3433 | -0.8076 | 0.8087 | 12.83 | FAIL |
| equity.snowball.intraday.pde | uniform_profile | desk_delta | 2.686 | 7.66e-08 | 2.68574 | -0.9653 | 0.9656 | 2.058 | FAIL |
| equity.snowball.intraday.pde | uniform_profile | desk_gamma | 0.042248 | 1.01e-07 | 0.0441231 | 44.38 | 44.39 | 140.7 | FAIL |
| equity.snowball.intraday.pde | uniform_profile | desk_theta | 0.146642 | 1.42e-08 | 0.157512 | 108.7 | 108.7 | 327.8 | FAIL |
| equity.snowball.intraday.pde | uniform_profile | point_delta | 3.54835 | 1.41e-07 | 3.54341 | -13.92 | 13.92 | 21.36 | FAIL |
| equity.snowball.intraday.pde | uniform_profile | point_gamma | -5.25104 | 5.55e-07 | -5.17553 | 14.38 | 14.38 | 21.28 | FAIL |
| equity.snowball.intraday.pde | sessions_only_profile | pv | -22.4108 | 9.61e-08 | -22.4073 | 35.04 | 35.04 | 42.91 | FAIL |
| equity.snowball.intraday.pde | sessions_only_profile | desk_delta | 2.56127 | 6.49e-08 | 2.56141 | 0.5528 | 0.553 | 21.73 | FAIL |
| equity.snowball.intraday.pde | sessions_only_profile | desk_gamma | 0.579227 | 3.35e-08 | 0.581413 | 3.773 | 3.773 | 23.78 | FAIL |
| equity.snowball.intraday.pde | sessions_only_profile | desk_theta | 0.214222 | 2.97e-08 | 0.221731 | 75.09 | 75.09 | 271.2 | FAIL |
| equity.snowball.intraday.pde | sessions_only_profile | point_delta | 2.8313 | 8.71e-08 | 2.83612 | 17.04 | 17.04 | 31.74 | FAIL |
| equity.snowball.intraday.pde | sessions_only_profile | point_gamma | 0.431034 | 5.63e-08 | 0.430938 | -0.2244 | 0.2245 | 37.15 | FAIL |
| equity.snowball.intraday.pde | on_ki_barrier_at_close | pv | -23.7972 | 2.6e-08 | -23.7971 | 1.303 | 1.303 | 2.362 | FAIL |
| equity.snowball.intraday.pde | on_ki_barrier_at_close | desk_delta | 2.47934 | 7.66e-08 | 2.48517 | 23.49 | 23.49 | 51.18 | FAIL |
| equity.snowball.intraday.pde | on_ki_barrier_at_close | desk_gamma | 3.9214 | 2.01e-07 | 3.93653 | 3.857 | 3.857 | 8.233 | FAIL |
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
| equity.snowball.intraday.pde | ordinary | pv | 0.000633 | space: -2.44, time: 2.02 | space |
| equity.snowball.intraday.pde | ordinary | desk_delta | 6.52e-05 | space: 1.7, time: 2.07 | -- |
| equity.snowball.intraday.pde | ordinary | desk_gamma | 0.000752 | space: 0.0572, time: 2 | -- |
| equity.snowball.intraday.pde | ordinary | desk_theta | 0.00274 | space: 3.2, time: 3.62 | -- |
| equity.snowball.intraday.pde | ordinary | point_delta | 0.00012 | space: 1.68, time: 2.05 | -- |
| equity.snowball.intraday.pde | ordinary | point_gamma | 0.000104 | space: -0.183, time: 2 | space |
| equity.snowball.intraday.pde | near_ki_above | pv | 0.0021 | space: -1.8, time: 2.06 | space |
| equity.snowball.intraday.pde | near_ki_above | desk_delta | 0.00144 | space: 1.75, time: -1.54 | time |
| equity.snowball.intraday.pde | near_ki_above | desk_gamma | 0.0113 | space: 2.59, time: 3.14 | -- |
| equity.snowball.intraday.pde | near_ki_above | desk_theta | 0.0314 | space: 4.78, time: 6.39 | -- |
| equity.snowball.intraday.pde | near_ki_above | point_delta | 0.00533 | space: -0.549, time: 2.05 | space |
| equity.snowball.intraday.pde | near_ki_above | point_gamma | 0.0137 | space: 2.76, time: 2.02 | -- |
| equity.snowball.intraday.pde | near_ki_below | pv | 0.00246 | space: -0.18, time: 2.06 | space |
| equity.snowball.intraday.pde | near_ki_below | desk_delta | 0.00201 | space: 2.07, time: 1.95 | -- |
| equity.snowball.intraday.pde | near_ki_below | desk_gamma | 0.0152 | space: 2.47, time: 3.2 | -- |
| equity.snowball.intraday.pde | near_ki_below | desk_theta | 0.00255 | space: 0.178, time: 1.21 | -- |
| equity.snowball.intraday.pde | near_ki_below | point_delta | 0.0104 | space: 0.462, time: 2.06 | -- |
| equity.snowball.intraday.pde | near_ki_below | point_gamma | 0.0194 | space: 0.732, time: 2.06 | -- |
| equity.snowball.intraday.pde | near_ki_1s | pv | 0.0138 | space: 4.04, time: 2.2 | -- |
| equity.snowball.intraday.pde | near_ki_1s | desk_delta | 0.0307 | space: 2.56, time: 2.15 | -- |
| equity.snowball.intraday.pde | near_ki_1s | desk_gamma | 0.0325 | space: -1.66, time: 2.09 | space |
| equity.snowball.intraday.pde | near_ki_1s | desk_theta | 80.1 | space: 17.6, time: 15 | -- |
| equity.snowball.intraday.pde | near_ki_1s | point_delta | 0.0661 | space: 6.23, time: 3.56 | -- |
| equity.snowball.intraday.pde | near_ki_1s | point_gamma | 0.00773 | space: 1.49, time: -0.563 | time |
| equity.snowball.intraday.pde | near_ki_10s | pv | -- | -- | -- |
| equity.snowball.intraday.pde | near_ki_10s | desk_delta | -- | -- | -- |
| equity.snowball.intraday.pde | near_ki_10s | desk_gamma | -- | -- | -- |
| equity.snowball.intraday.pde | near_ki_10s | desk_theta | -- | -- | -- |
| equity.snowball.intraday.pde | near_ki_10s | point_delta | -- | -- | -- |
| equity.snowball.intraday.pde | near_ki_10s | point_gamma | -- | -- | -- |
| equity.snowball.intraday.pde | near_ki_5m | pv | 0.00258 | space: 3.11, time: 2.06 | -- |
| equity.snowball.intraday.pde | near_ki_5m | desk_delta | 0.00101 | space: 1.33, time: 2.02 | -- |
| equity.snowball.intraday.pde | near_ki_5m | desk_gamma | 0.00836 | space: 3.09, time: 1.94 | -- |
| equity.snowball.intraday.pde | near_ki_5m | desk_theta | 0.398 | space: 6.13, time: 12.5 | -- |
| equity.snowball.intraday.pde | near_ki_5m | point_delta | 0.0104 | space: 2.04, time: 2.08 | -- |
| equity.snowball.intraday.pde | near_ki_5m | point_gamma | 0.218 | space: 2.21, time: 2 | -- |
| equity.snowball.intraday.pde | near_ko_on_ko_day | pv | 0.000795 | space: 0.614, time: 2.06 | -- |
| equity.snowball.intraday.pde | near_ko_on_ko_day | desk_delta | 0.00296 | space: 2.56, time: 4.33 | -- |
| equity.snowball.intraday.pde | near_ko_on_ko_day | desk_gamma | 0.00231 | space: 1.25, time: 6.89 | -- |
| equity.snowball.intraday.pde | near_ko_on_ko_day | desk_theta | 0.000678 | space: -0.108, time: 5.44 | space |
| equity.snowball.intraday.pde | near_ko_on_ko_day | point_delta | 0.00976 | space: 0.612, time: 2.03 | -- |
| equity.snowball.intraday.pde | near_ko_on_ko_day | point_gamma | 0.00704 | space: 1.02, time: 2 | -- |
| equity.snowball.intraday.pde | near_ko_on_ko_day_1s | pv | 0.0011 | space: 5.58, time: 2.63 | -- |
| equity.snowball.intraday.pde | near_ko_on_ko_day_1s | desk_delta | 0.00124 | space: 2.1, time: 2.1 | -- |
| equity.snowball.intraday.pde | near_ko_on_ko_day_1s | desk_gamma | 0.000336 | space: 3.01, time: 0.561 | -- |
| equity.snowball.intraday.pde | near_ko_on_ko_day_1s | desk_theta | 14.3 | space: 16.7, time: 19.1 | -- |
| equity.snowball.intraday.pde | near_ko_on_ko_day_1s | point_delta | 0.00133 | space: 2.51, time: 1.8 | -- |
| equity.snowball.intraday.pde | near_ko_on_ko_day_1s | point_gamma | 0.000192 | space: 5.45, time: 1.8 | -- |
| equity.snowball.intraday.pde | ko_level_on_ki_day | pv | 0.000166 | space: 2.36, time: 2.06 | -- |
| equity.snowball.intraday.pde | ko_level_on_ki_day | desk_delta | 0.000322 | space: 3.9, time: 1.96 | -- |
| equity.snowball.intraday.pde | ko_level_on_ki_day | desk_gamma | 0.000192 | space: 0.604, time: -2.53 | time |
| equity.snowball.intraday.pde | ko_level_on_ki_day | desk_theta | 0.000225 | space: 4.02, time: 3.93 | -- |
| equity.snowball.intraday.pde | ko_level_on_ki_day | point_delta | 0.000359 | space: 0.086, time: 2.02 | -- |
| equity.snowball.intraday.pde | ko_level_on_ki_day | point_gamma | 2.36e-05 | space: 2.01, time: 2.01 | -- |
| equity.snowball.intraday.pde | pre_open | pv | 0.00298 | space: 5.74, time: 2.02 | -- |
| equity.snowball.intraday.pde | pre_open | desk_delta | 0.000223 | space: 2.71, time: 1.95 | -- |
| equity.snowball.intraday.pde | pre_open | desk_gamma | 0.0044 | space: 4.25, time: 2.01 | -- |
| equity.snowball.intraday.pde | pre_open | desk_theta | 0.00594 | space: 7.55, time: 2.09 | -- |
| equity.snowball.intraday.pde | pre_open | point_delta | 0.000389 | space: 3.89, time: 2.61 | -- |
| equity.snowball.intraday.pde | pre_open | point_gamma | 0.000415 | space: 2.18, time: 2.01 | -- |
| equity.snowball.intraday.pde | lunch_break | pv | 0.00753 | space: 0.516, time: 2.04 | -- |
| equity.snowball.intraday.pde | lunch_break | desk_delta | 0.00603 | space: 5.2, time: 2.13 | -- |
| equity.snowball.intraday.pde | lunch_break | desk_gamma | 0.0228 | space: 3.79, time: 1.59 | -- |
| equity.snowball.intraday.pde | lunch_break | desk_theta | 0.00487 | space: -3.98, time: 3.15 | space |
| equity.snowball.intraday.pde | lunch_break | point_delta | 0.00242 | space: 0.822, time: 2.01 | -- |
| equity.snowball.intraday.pde | lunch_break | point_gamma | 0.0124 | space: 2.97, time: 2.08 | -- |
| equity.snowball.intraday.pde | lunch_break_zero_variance | pv | 0.00656 | space: 0.744, time: 2.04 | -- |
| equity.snowball.intraday.pde | lunch_break_zero_variance | desk_delta | 0.00416 | space: -1.83, time: 2.55 | space |
| equity.snowball.intraday.pde | lunch_break_zero_variance | desk_gamma | 0.0221 | space: 3.24, time: 2.28 | -- |
| equity.snowball.intraday.pde | lunch_break_zero_variance | desk_theta | 2.37e-05 | space: 1.44, time: 2.01 | -- |
| equity.snowball.intraday.pde | lunch_break_zero_variance | point_delta | 0.00367 | space: 1.1, time: 2.02 | -- |
| equity.snowball.intraday.pde | lunch_break_zero_variance | point_gamma | 0.0119 | space: 6.09, time: 2.12 | -- |
| equity.snowball.intraday.pde | holiday_eve | pv | 0.00478 | space: 1.39, time: 2.04 | -- |
| equity.snowball.intraday.pde | holiday_eve | desk_delta | 0.00216 | space: 1.72, time: 2.07 | -- |
| equity.snowball.intraday.pde | holiday_eve | desk_gamma | 0.014 | space: 4.88, time: 1.94 | -- |
| equity.snowball.intraday.pde | holiday_eve | desk_theta | 0.0101 | space: 3.25, time: 1.83 | -- |
| equity.snowball.intraday.pde | holiday_eve | point_delta | 0.00238 | space: 3.7, time: 2 | -- |
| equity.snowball.intraday.pde | holiday_eve | point_gamma | 0.00625 | space: 4.49, time: 2.09 | -- |
| equity.snowball.intraday.pde | maturity_day | pv | 4.55e-13 | space: -2.33, time: -1.69 | space, time |
| equity.snowball.intraday.pde | maturity_day | desk_delta | 1.31e-13 | space: -4.74, time: 1.58 | space |
| equity.snowball.intraday.pde | maturity_day | desk_gamma | 1.63e-12 | space: -1.29, time: -1.68 | space, time |
| equity.snowball.intraday.pde | maturity_day | desk_theta | 4.55e-13 | space: -2.33, time: -1.69 | space, time |
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
| equity.snowball.intraday.pde | provisional_ki_after_close | pv | 0.000128 | space: 5.67, time: 1.99 | -- |
| equity.snowball.intraday.pde | provisional_ki_after_close | desk_delta | 0.000128 | space: 2.44, time: 12.7 | -- |
| equity.snowball.intraday.pde | provisional_ki_after_close | desk_gamma | 0.000765 | space: 1.71, time: 17.7 | -- |
| equity.snowball.intraday.pde | provisional_ki_after_close | desk_theta | 4.81e-05 | space: 4.63, time: 20 | -- |
| equity.snowball.intraday.pde | provisional_ki_after_close | point_delta | 4.28e-06 | space: -0.479, time: 2 | space |
| equity.snowball.intraday.pde | provisional_ki_after_close | point_gamma | 8.19e-07 | space: -0.77, time: 2.01 | space |
| equity.snowball.intraday.pde | low_vol | pv | 0.00629 | space: 2.81, time: 2.06 | -- |
| equity.snowball.intraday.pde | low_vol | desk_delta | 0.00308 | space: 4.54, time: -0.152 | time |
| equity.snowball.intraday.pde | low_vol | desk_gamma | 0.0213 | space: 4.08, time: 3.89 | -- |
| equity.snowball.intraday.pde | low_vol | desk_theta | 0.0261 | space: 6.88, time: 5.37 | -- |
| equity.snowball.intraday.pde | low_vol | point_delta | 0.0149 | space: 3.03, time: 2.05 | -- |
| equity.snowball.intraday.pde | low_vol | point_gamma | 0.195 | space: 2.46, time: 1.94 | -- |
| equity.snowball.intraday.pde | high_vol | pv | 0.00109 | space: -4.27, time: 2.07 | space |
| equity.snowball.intraday.pde | high_vol | desk_delta | 0.00159 | space: 2.09, time: 1.54 | -- |
| equity.snowball.intraday.pde | high_vol | desk_gamma | 0.00684 | space: 0.926, time: 0.93 | -- |
| equity.snowball.intraday.pde | high_vol | desk_theta | 0.00217 | space: 2.31, time: 3.84 | -- |
| equity.snowball.intraday.pde | high_vol | point_delta | 0.00243 | space: -1.93, time: 2.05 | space |
| equity.snowball.intraday.pde | high_vol | point_gamma | 0.00185 | space: -0.276, time: 2.04 | space |
| equity.snowball.intraday.pde | carry | pv | 0.000623 | space: -2.5, time: 2.02 | space |
| equity.snowball.intraday.pde | carry | desk_delta | 6.42e-05 | space: 1.7, time: 2.07 | -- |
| equity.snowball.intraday.pde | carry | desk_gamma | 0.000743 | space: 0.0523, time: 2 | -- |
| equity.snowball.intraday.pde | carry | desk_theta | 0.00269 | space: 3.2, time: 3.62 | -- |
| equity.snowball.intraday.pde | carry | point_delta | 0.000118 | space: 1.67, time: 2.05 | -- |
| equity.snowball.intraday.pde | carry | point_gamma | 0.000102 | space: -0.177, time: 2 | space |
| equity.snowball.intraday.pde | uniform_profile | pv | 0.00128 | space: 0.332, time: 2 | -- |
| equity.snowball.intraday.pde | uniform_profile | desk_delta | 0.000553 | space: 3.3, time: 4.04 | -- |
| equity.snowball.intraday.pde | uniform_profile | desk_gamma | 0.00594 | space: 2.67, time: 7.68 | -- |
| equity.snowball.intraday.pde | uniform_profile | desk_theta | 0.0328 | space: 6.22, time: 11.9 | -- |
| equity.snowball.intraday.pde | uniform_profile | point_delta | 0.00758 | space: 2.16, time: 2 | -- |
| equity.snowball.intraday.pde | uniform_profile | point_gamma | 0.112 | space: 5.28, time: 2 | -- |
| equity.snowball.intraday.pde | sessions_only_profile | pv | 0.00429 | space: 2.92, time: 2.04 | -- |
| equity.snowball.intraday.pde | sessions_only_profile | desk_delta | 0.00557 | space: 2.41, time: 3.06 | -- |
| equity.snowball.intraday.pde | sessions_only_profile | desk_gamma | 0.0138 | space: 4.21, time: 4.15 | -- |
| equity.snowball.intraday.pde | sessions_only_profile | desk_theta | 0.0271 | space: 4.94, time: 5.32 | -- |
| equity.snowball.intraday.pde | sessions_only_profile | point_delta | 0.00899 | space: 4.4, time: 2.05 | -- |
| equity.snowball.intraday.pde | sessions_only_profile | point_gamma | 0.016 | space: 2.55, time: 2.07 | -- |
| equity.snowball.intraday.pde | on_ki_barrier_at_close | pv | 0.000236 | space: 1.58, time: 2 | -- |
| equity.snowball.intraday.pde | on_ki_barrier_at_close | desk_delta | 0.0127 | space: 2.5, time: 8.49 | -- |
| equity.snowball.intraday.pde | on_ki_barrier_at_close | desk_gamma | 0.0323 | space: 2.53, time: 8.44 | -- |

## Aggregate bias

| candidate | quantity | cells | mean bias (c) | mean radius (c) | passed |
|---|---|---|---|---|---|
| equity.snowball.intraday.quad_v2 | pv | 23 | -0.0001466 | 0.0116 | yes |
| equity.snowball.intraday.quad_v2 | desk_delta | 23 | -6.808e-05 | 0.000275 | yes |
| equity.snowball.intraday.quad_v2 | desk_gamma | 23 | 0.0001174 | 0.000532 | yes |
| equity.snowball.intraday.quad_v2 | desk_theta | 22 | -1.55e-05 | 0.00042 | yes |
| equity.snowball.intraday.quad_v2 | point_delta | 22 | -7.621e-05 | 0.000312 | yes |
| equity.snowball.intraday.quad_v2 | point_gamma | 22 | 7.982e-05 | 0.00214 | yes |
| equity.snowball.intraday.pde | pv | 21 | 17.92 | 0.0126 | no |
| equity.snowball.intraday.pde | desk_delta | 21 | 9.988 | 0.000284 | no |
| equity.snowball.intraday.pde | desk_gamma | 21 | 16.03 | 0.000553 | no |
| equity.snowball.intraday.pde | desk_theta | 20 | -1.813e+04 | 0.000446 | no |
| equity.snowball.intraday.pde | point_delta | 20 | 10.13 | 0.00033 | no |
| equity.snowball.intraday.pde | point_gamma | 20 | 4.828 | 0.00233 | no |
