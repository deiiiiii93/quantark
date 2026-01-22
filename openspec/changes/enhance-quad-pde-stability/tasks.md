## 1. Quadrature: Event-step smoothing (kernel upgrade)
- [x] 1.1 Replace linear smoothing ramp with raised-cosine or tanh kernel around KO/KI/coupon barriers.
- [x] 1.2 Ensure smoothing is applied only to discrete event operators (no diffusion changes).
- [ ] 1.3 Add tests or diagnostics confirming smoothed transitions keep monotonicity across grid sizes.

## 2. Quadrature: Barrier alignment priority (reverse-aware)
- [x] 2.1 Add `QuadParams.align_priority` (enum/string: `ko|coupon|ki|auto`) with backward-compatible default (`auto`).
- [x] 2.2 Implement reverse-aware alignment: prefer KO if near spot, else coupon, else KI.
- [ ] 2.3 Add a Phoenix diagnostic case to verify alignment priority impacts oscillation.

## 3. Quadrature: Adaptive smoothing heuristic
- [x] 3.1 Implement adaptive `event_smoothing_cells` heuristic based on grid spacing and barrier spacing.
- [x] 3.2 Provide overrides to force `cells=0` (reverse) or `cells=1` (standard) as a config option.
- [ ] 3.3 Benchmark standard vs reverse Phoenix with heuristic enabled.

## 4. Quadrature: Padding/filter auto-tuning
- [x] 4.1 Expose optional tuning presets (e.g., `filter_alpha=18/24`, `padding=4`) via params or helper.
- [ ] 4.2 Run a small N-sweep to validate reduced oscillation for hard-discontinuity cases.

## 5. PDE: Event alignment enforcement (done)
- [x] 5.1 Confirm KO/KI/coupon mapping uses exact event indices (no argmin fallback).
- [x] 5.2 Verify ValidationError messaging guides users to event-aligned grids.

## 6. PDE: Higher-order boundary conditions
- [x] 6.1 Add optional far-field asymptotic boundary conditions for standard/reverse cases.
- [x] 6.2 Expand spatial bounds automatically when KO/KI barriers are in tails.
- [ ] 6.3 Add regression checks for barriered products with wide domains.

## 7. PDE: Variable-theta near events
- [x] 7.1 Apply theta=1.0 only for the step immediately before an event time.
- [x] 7.2 Revert to theta=0.5 (or 0.6) immediately after event steps.
- [ ] 7.3 Validate that event-time diffusion bias is reduced without instability.

## 8. PDE: Barrier-focused grid clustering
- [x] 8.1 Tighten log-space spacing around KO/KI/coupon barriers with a refinement rule.
- [x] 8.2 Expose parameters to control refinement width and density.
- [ ] 8.3 Compare PDE vs MC convergence on reverse Phoenix with refined grids.

## 9. Demo defaults and validation
- [x] 9.1 Update Phoenix engine compare demo defaults to tuned grids and RQMC.
- [ ] 9.2 Run Phoenix compare demo and capture before/after metrics.
- [ ] 9.3 Run relevant quad/PDE unit tests.
