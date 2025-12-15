<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

## Numerical Utilities (IMPORTANT)

Always use `util/numerical/` for numerical operations:

```python
from util.numerical import (
    is_zero, is_close,                    # Float comparison
    safe_log, safe_exp, safe_sqrt,        # Safe math
    format_currency, format_percentage,    # Formatting
    validate_positive, is_valid_number     # Validation
)

# Use is_zero() instead of: if T < 1e-10
# Use safe_log() instead of: math.log()
# Use format_currency() instead of: f"${x:.2f}"
```