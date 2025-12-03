# Tech Stack and Dependencies

## Core Python Version
- **Python**: 3.8 or higher (targeted for 3.10+)

## Core Scientific Computing Libraries
- **NumPy** (>=1.24.0): Numerical computing, array operations, linear algebra
- **SciPy** (>=1.10.0): Statistical functions, numerical integration, optimization
- **Pandas** (>=2.0.0): DataFrame manipulation, time series analysis

## Financial Data Handling
- **PyArrow** (>=12.0.0): Columnar data format for efficient data interchange
- **OpenPyXL** (>=3.0.0): Excel file reading and writing

## Visualization
- **Matplotlib** (>=3.7.0): Plotting and visualization
- **Seaborn** (>=0.12.0): Statistical data visualization
- **Plotly** (>=5.14.0): Interactive visualizations
- **Kaleido** (>=0.2.1): Static image export for Plotly

## Testing Framework
- **pytest** (>=7.0.0): Primary testing framework
- **Coverage**: Test coverage reporting (optional)

## Configuration
- **PyYAML** (>=6.0.0): YAML configuration file parsing

## Development Tools
- **Black**: Code formatting (recommended)
- **Flake8**: Linting (recommended)
- **mypy**: Static type checking (recommended)

## Virtual Environment
- **quantark/**: Pre-configured Python virtual environment
  - Located in project root
  - Activate with: `source quantark/bin/activate`
  - Use directly: `quantark/bin/python`, `quantark/bin/pip`

## Key Architectural Patterns

### Type System
- **Dataclasses**: Extensive use for data containers
- **Type Hints**: All public APIs use type annotations
- **Protocols**: For abstract interfaces (e.g., VaREngine protocol)
- **Enums**: For method selection and configuration (see EngineType pattern)

### Numerical Computing
- **Vectorized Operations**: NumPy array operations for performance
- **Matrix Operations**: Linear algebra via NumPy/SciPy
- **Statistical Functions**: SciPy for distributions, hypothesis testing
- **Time Series**: Pandas for time-based data manipulation

### Data Structures
- **DataFrames**: Market data representation (equity, FI)
- **TimeSeries**: DateTimeIndex for time-based data
- **Columnar Data**: PyArrow for efficient data interchange

### Performance Optimization
- **Numba**: Not currently used, but suitable for future optimization
- **Cython**: Potential for critical performance paths
- **GPU Computing**: Not currently used, potential future enhancement

## Operating System
- **Primary Platform**: macOS (Darwin)
- **Python Compatibility**: Cross-platform (Linux, Windows, macOS)

## External Integrations
- **No Market Data Feeds**: Library provides pricing logic only
- **Excel Integration**: OpenPyXL for data import/export
- **Database**: Not currently integrated (potential future feature)

## Development Workflow Tools
- **Git**: Version control
- **OpenSpec**: Specification-driven development workflow
- **pytest**: Testing framework
- **Virtual Environment**: Isolated Python environment

## Key Usage Patterns
- PricingEnvironment as central data container
- Engine-agnostic product definitions
- Strategy pattern for engine selection
- Factory pattern for instrument creation
