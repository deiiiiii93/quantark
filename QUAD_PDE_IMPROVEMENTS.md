# QUAD and PDE Engine Improvements (Phoenix/Snowball)
Professional Report

Author: QuantArk engineering  
Date: 2026-01-21

## Abstract
Phoenix and reverse Phoenix products apply multiple discontinuous events (KO/KI/coupon)
at discrete observation dates. These discontinuities interact with diffusion and grid
discretization, often causing oscillations, ringing, and slow or non-monotonic convergence.
We implemented a targeted set of improvements in QUAD (FFT-based diffusion) and PDE engines
to address aliasing, event-time alignment, barrier misalignment, grid resolution near
barriers, and tail boundary bias.

## Methods

### Products and Numerical Setting
The focus is on Phoenix and reverse Phoenix structures (including memory, quarterly KO,
and step-down variants). These are sensitive to barrier placement and event-time accuracy.

### QUAD Engine Methodology
QUAD uses FFT-based convolution in log-price space. The following changes were introduced:

1) FFT padding and spectral filtering  
- Added zero-padding before convolution to mitigate wrap-around aliasing.  
- Applied a mild low-pass filter in Fourier space to damp Gibbs ringing.

2) Barrier-aligned grids with reverse-aware priority  
- Added grid alignment so key barriers land on grid nodes.  
- Implemented `align_priority` to select KO/coupon/KI based on product type.

3) Event-step smoothing with smooth kernels  
- Replaced linear ramps with raised-cosine or tanh transitions on discrete event operators.  
- Smoothing is applied only to KO/KI/coupon events, not to diffusion.

4) Adaptive smoothing strength  
- Added heuristic scaling for smoothing cells based on grid spacing, with overrides.

5) Expanded log-domain control  
- Added explicit `num_std_devs` to widen tails and reduce truncation bias.

### PDE Engine Methodology
The PDE solver evolves value backward between events. Improvements include:

1) Event-aligned time grids  
- Enforce exact event times in the time grid and apply operators exactly at those nodes.

2) Event-adjacent theta control  
- Use `event_theta=1.0` for steps immediately before events, then revert to standard theta.

3) Barrier-focused spatial refinement and domain expansion  
- Add log-space refinement layers around KO/KI/coupon barriers.  
- Expand domain when barriers lie in the tails.

4) Asymptotic boundary conditions  
- Added tail-aware boundary conditions and made `boundary_mode=asymptotic` the default.

### Validation Approach
We compared PDE output against MC (100k randomized QMC paths) using the Phoenix comparison
demo defaults. The primary diagnostic was PDE bias vs MC across multiple Phoenix variants.

## Results

### QUAD Stability
Grid-size oscillation and flip-flop behavior were reduced after applying padding, filtering,
barrier alignment, and smooth event-step kernels. Stability improved without over-biasing
event boundaries, especially in reverse variants.

### PDE Boundary Mode Impact
Switching from `boundary_mode=default` to `asymptotic` reduced PDE bias vs MC by roughly
0.08 to 0.20 percentage points for standard and step-down Phoenix cases. Reverse cases were
essentially unchanged.

### Demo Defaults (Baseline)
The demo baseline for reproducibility now uses:
- QUAD grid size = 1001
- PDE grid size = 1000, time steps = 400
- MC paths = 100,000 with randomized QMC

## Discussion

### Why QUAD Improvements Worked
FFT aliasing and Gibbs ringing are the dominant causes of oscillation. Padding removes
wrap-around contamination, filters attenuate high-frequency components, and barrier alignment
prevents repeated interpolation across kinks. Smooth kernels replace the sharpest step
functions with controlled transitions, improving convergence while preserving the economics.

### Why PDE Improvements Worked
Event alignment is essential for correct timing of KO/KI/coupon logic. Event-adjacent
Backward Euler steps reduce oscillations near discontinuities. Barrier refinement allocates
resolution where gradients are largest, and asymptotic boundary conditions prevent tail bias.

### Limitations and Future Work
Reverse Phoenix remains sensitive to tail truncation and upper barrier placement. Further
improvements may require more aggressive tail expansion, adaptive meshing, or enhanced
boundary models tailored to reverse structures.

## Practical Guidance (Recommended Settings)

### QUAD (Phoenix)
- `num_std_devs=10`
- `fft_padding_factor=2`
- `fft_filter_alpha=18`
- `event_smoothing_kernel="raised_cosine"`
- `align_priority="auto"` (or `"ko"` for reverse)

### PDE (Phoenix)
- `boundary_mode="asymptotic"`
- `event_theta=1.0`, `event_rannacher_steps=1`
- `barrier_refine_log_width=0.02` (product-dependent)

## Files Updated
- QUAD: `asset/equity/engine/quad/quad_math.py`,
  `asset/equity/engine/quad/phoenix_quad_engine.py`,
  `asset/equity/engine/quad/snowball_quad_engine.py`
- Params: `asset/equity/param/engine_params.py`
- PDE: `asset/equity/engine/pde/base_pde_solver.py`,
  `asset/equity/engine/pde/phoenix_pde_solver.py`,
  `asset/equity/engine/pde/snowball_pde_solver.py`
- Demo: `example/phoenix_engine_compare_demo.py`

