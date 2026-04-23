
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
