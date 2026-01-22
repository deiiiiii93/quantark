# Design: Engine Parameter Presets, Factories, and Config Loaders

## Goals
- Provide a simple, batch-friendly way to configure QUAD/PDE engines.
- Preserve full flexibility for expert users.
- Avoid breaking existing APIs or defaults.

## Proposed API

### Preset registry
- `ENGINE_PARAM_PRESETS` mapping keyed by engine type (`"quad"`, `"pde"`).
- Each engine provides profiles: `fast`, `balanced`, `accurate`,
  `barrier_sensitive`, `reverse_sensitive`.

### Factory helpers
```python
make_quad_params(profile="balanced", product=None, reverse=False, **overrides) -> QuadParams
make_pde_params(profile="balanced", product=None, reverse=False, **overrides) -> PDEParams
make_engine_params(engine="quad", profile="balanced", product=None, reverse=False, **overrides)
```

Notes:
- `product` and `reverse` provide light auto-tuning (alignment priority, smoothing).
- `overrides` always win over presets.

### Config loaders
- `QuadParams.from_config(config_or_path, profile=None, **overrides)`
- `PDEParams.from_config(config_or_path, profile=None, **overrides)`

Supported inputs:
- dict
- YAML file path
- JSON file path

Example YAML:
```yaml
engine: quad
profile: barrier_sensitive
overrides:
  num_std_devs: 10
  fft_filter_alpha: 18
```

## Placement
- New module: `asset/equity/param/engine_param_profiles.py`
  - Holds preset registry, factories, and config helpers.
- `engine_params.py` adds lightweight `from_config` classmethods.

## Validation
- Unknown profile raises `ValidationError` with allowed values.
- Unknown keys in `overrides` raise `ValidationError`.
- Config missing `engine` or `profile` uses defaults (`engine` provided by loader).

## Compatibility
- Defaults remain unchanged unless a user opts into presets/factories.
- No changes to pricing engines required beyond optional convenience calls.

