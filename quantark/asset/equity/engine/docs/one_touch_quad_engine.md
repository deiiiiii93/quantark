# OneTouch Quad Engine

## Overview
`OneTouchQuadEngine` prices discretely monitored one-touch and no-touch options
using the FFT-based quadrature recursion shared with barrier options. The method
maps barrier-hit rebates to the quadrature cash factors, so the recursion
naturally handles "stop first hit" aggregation.

## Supported Features
- Discrete or expiry-only monitoring
- Up/down barriers
- One-touch and no-touch (no-touch via parity)
- Pay-at-hit or pay-at-expiry rebates
- Observation schedules with per-observation barriers

## Method Notes
- The recursion operates on a fixed time grid and uses Simpson integration
  with FFT-based convolution.
- For pay-at-hit, the rebate is treated as a cash payoff at the observation
  grid time, with an optional settlement delay discount.
- For pay-at-expiry, the rebate value at each observation time is discounted
  from expiry back to that grid time, then the recursion handles discounting
  to valuation.

## Limitations
- Continuous monitoring is not supported (use `OneTouchAnalyticalEngine`).
- Extremely high volatilities or long maturities may require larger grids.
