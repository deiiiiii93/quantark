# SIMM Module

ISDA Standard Initial Margin Model (SIMM) foundation module for QuantArk.

## Overview

This module provides the foundational data structures and utilities for ISDA SIMM v2.6 calculations. It establishes the core taxonomy, data models, and interchange formats that all subsequent SIMM components will build upon.

## Features

- **Risk Class Taxonomy**: Six risk classes (IR, CreditQ, CreditNQ, Equity, Commodity, FX)
- **Product Class Taxonomy**: Four product classes (RatesFX, Credit, Equity, Commodity)
- **Margin Types**: Delta, Vega, Curvature, and Base Correlation
- **Sensitivity Data Models**: Type-safe sensitivity dataclasses with protocols
- **CRIF Support**: Full CRIF (Common Risk Interchange Format) parsing and export
- **Configuration**: Flexible SIMM configuration with version support

## Quick Start

```python
from simm import (
    SIMMConfig,
    RiskClass,
    MarginType,
    IRDeltaSensitivity,
    SensitivityCollection,
    parse_crif_csv,
    crif_to_sensitivities,
)

# Create sensitivities programmatically
collection = SensitivityCollection()
collection.add(IRDeltaSensitivity(
    trade_id="TRADE001",
    amount=150000.0,
    amount_currency="USD",
    currency="USD",
    tenor=5.0,
))

# Or parse from CRIF file
records, warnings = parse_crif_csv("sensitivities.csv")
collection = crif_to_sensitivities(records)

# Group by risk class
ir_sensitivities = collection.by_risk_class(RiskClass.INTEREST_RATE)

# Configure SIMM calculation
config = SIMMConfig(
    calculation_currency="USD",
    calculate_delta=True,
    calculate_vega=True,
)
```

## Module Structure

```
simm/
├── __init__.py           # Public API exports
├── config.py             # SIMMConfig, SIMMVersion
├── taxonomy.py           # Enums, buckets, tenors
├── sensitivity.py        # Sensitivity dataclasses
└── crif/
    ├── __init__.py
    ├── models.py         # CRIFRecord, CRIFHeader
    └── parser.py         # CSV parsing, validation
```

## Risk Classes

| Risk Class | Code | Description |
|------------|------|-------------|
| Interest Rate | IR | Interest rate curve and volatility risk |
| Credit Qualifying | CreditQ | Investment grade and high yield credit |
| Credit Non-Qualifying | CreditNQ | Non-qualifying credit exposures |
| Equity | Equity | Equity spot and volatility risk |
| Commodity | Commodity | Commodity price and volatility risk |
| FX | FX | Foreign exchange spot and volatility risk |

## Bucket Definitions

Each risk class has specific bucket definitions:

- **IR**: Buckets by currency with volatility classification (Low/Regular/High)
- **Credit Qualifying**: 12 buckets + residual (by credit quality and sector)
- **Credit Non-Qualifying**: 2 buckets + residual
- **Equity**: 12 buckets + residual (by size, region, sector)
- **Commodity**: 17 buckets (by commodity type)
- **FX**: Single bucket

## CRIF Format

The module supports ISDA CRIF v2.x format for sensitivity exchange:

```csv
TradeID,ValuationDate,RiskType,Qualifier,Bucket,Label1,Label2,Amount,AmountCurrency
TRADE001,2024-01-15,Risk_IRCurve,USD,1,5y,OIS,150000,USD
TRADE002,2024-01-15,Risk_FX,EURUSD,,,50000,USD
```

## Dependencies

This is a foundational module with no internal dependencies. Future SIMM modules will depend on this foundation:

- `simm-calibration`: Risk weights and correlations
- `simm-sensitivity-engines`: Sensitivity calculation
- `simm-aggregation`: Margin aggregation formulas
- `simm-reporting`: Attribution and reporting

## Testing

```bash
# Run SIMM tests
python -m pytest test/test_simm_*.py -v
```
