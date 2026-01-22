# Engine Parameter Presets and Factories

This guide provides a batch-friendly way to configure QUAD and PDE engines using
named presets, factory helpers, and YAML/JSON configs.

## Quick Start

```python
from asset.equity.param import make_quad_params, make_pde_params

quad_params = make_quad_params(profile="barrier_sensitive")
pde_params = make_pde_params(profile="balanced")
```

## Preset Decision Table

| Use Case | Profile | Trade-offs |
| --- | --- | --- |
| Fast previews, wide sweeps | `fast` | Lowest cost, lower accuracy |
| Daily batch production | `balanced` | Default stability/accuracy |
| Benchmark or validation | `accurate` | Higher cost, higher accuracy |
| Barrier-heavy products | `barrier_sensitive` | More refinement near barriers |
| Reverse Phoenix / tail-sensitive | `reverse_sensitive` | Stronger tail focus |

## When Results Exceed Tolerance (Action Map)

Use this quick map when a case drifts beyond ~2% vs MC:

| Symptom | Likely cause | Suggested action |
| --- | --- | --- |
| PDE underprices with dense discrete KI | Insufficient event resolution | Use `barrier_sensitive` or increase `time_steps` and `event_steps_per_day` |
| PDE underprices near KO | Barrier gradient under-resolved | Increase `barrier_refine_levels` and lower `log_dx_target` |
| High vol cases drift | Grid too coarse in time/space | Switch to `accurate` or increase `grid_size` + `time_steps` |
| Reverse Phoenix drift | Tail sensitivity | Use `reverse_sensitive`, increase `barrier_domain_expand` and QUAD `num_std_devs` |
| Long maturities drift | Accumulated time-stepping error | Scale `time_steps` roughly with maturity (see below) |

## Factory Helpers

```python
from asset.equity.param import make_engine_params

params = make_engine_params("quad", profile="reverse_sensitive", reverse=True)
```

Notes:
- `product` and `reverse` trigger light auto-tuning.
- Explicit overrides always win.

## YAML / JSON Config (Batch Friendly)

Example YAML:
```yaml
engine: quad
profile: barrier_sensitive
overrides:
  num_std_devs: 10
  fft_filter_alpha: 18
```

Example JSON:
```json
{
  "engine": "pde",
  "profile": "balanced",
  "overrides": {
    "grid_size": 300,
    "time_steps": 150
  }
}
```

Load from config:
```python
from asset.equity.param import QuadParams, PDEParams

quad_params = QuadParams.from_config("path/to/engine_params.yaml")
pde_params = PDEParams.from_config("path/to/engine_params.json")
```

## Override Rules
- Profile values are applied first.
- Product hints may adjust a small number of safety parameters.
- `overrides` always take precedence.

## Auto-Tuning Rule of Thumb (Long Maturities)
For maturities beyond 1Y, scale time steps linearly with maturity to avoid drift:

- `time_steps ≈ base_steps * max(1, T / 1.0)`
- Example: if `balanced` uses 200 steps at 1Y, use ~400 at 2Y.

This is especially important for PDE with discrete event schedules.
