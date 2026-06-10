# Cash Legs Module - Developer Guide

## Overview

`cashleg/` provides composable cash-flow primitives that price alongside an
equity option payoff. Each leg returns a signed PV from the buyer's perspective.
Legs attach to an `EquityPosition` via its `cash_legs` field.

## Quick Reference

| Leg type | When to use |
|---|---|
| `DeterministicLeg` | Premium or fixed fee with fixed amount and fixed timing |
| `AccrualLeg` | Periodic interest or rebate, optionally truncated by KO |
| `FixedPayoffLeg` | Single fixed amount paid only if a KO/KI/maturity event occurs |

## Architecture

- `CashLeg.value(event_dist, env, position_notional) -> float` returns signed PV.
- `EventDistribution` is emitted by `engine.price_with_events(...)`.
- `BaseEngine.price_with_events(...)` adapts existing `calculate_event_stats(...)`
  implementations when available, otherwise returns a trivial maturity-only
  distribution.

## Sign Convention

- `LegDirection.BUYER_RECEIVES` has positive sign.
- `LegDirection.BUYER_PAYS` has negative sign.
- `EquityPosition` multiplies per-unit product and leg PVs by `quantity`.

## Spec

See `docs/superpowers/specs/2026-05-18-equity-cash-legs-design.md`.
