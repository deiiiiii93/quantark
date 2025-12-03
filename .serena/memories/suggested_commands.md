# Essential Commands for QuantArk Development

## Environment Setup

### Activate Virtual Environment
```bash
# Activate the pre-configured virtual environment
source quantark/bin/activate

# Verify activation (should show quantark path)
which python
which pip
```

### Install Dependencies
```bash
# Install all dependencies
pip install -r requirements.txt

# Install in development mode (if needed)
pip install -e .
```

## Testing Commands

### Run All Tests
```bash
# Run entire test suite
python -m pytest

# Run with verbose output
python -m pytest -v

# Run with coverage (if pytest-cov installed)
python -m pytest --cov=.
```

### Run Specific Tests
```bash
# Run specific test file
python -m pytest test/test_european_option.py

# Run tests matching pattern
python -m pytest -k "test_name_pattern"

# Run specific test method
python -m pytest test/test_european_option.py::TestEuropeanOption::test_call_option_price

# Run tests in directory
python -m pytest test/var/

# Run with markers
python -m pytest -m "not slow"
```

### Test Output Options
```bash
# Show local variables on failure
python -m pytest -l

# Stop on first failure
python -m pytest -x

# Show print statements
python -m pytest -s

# Capture mode (no/capture/final)
python -m pytest --capture=no
```

## Example Scripts

### Run All Examples
```bash
# European option demo
python example/european_option_demo.py

# American option demo
python example/american_option_demo.py

# Monte Carlo demo
python example/european_mc_demo.py

# PDE pricing demo
python example/pde_pricing_demo.py

# Bond pricing
python example/fixed_bond_demo.py
python example/bond_option_demo.py
python example/frn_demo.py
python example/irs_demo.py

# VaR calculations
python example/parametric_var_demo.py
python example/historical_var_demo.py
python example/monte_carlo_var_demo.py
python example/var_backtest_demo.py

# Portfolio and risk
python example/portfolio_demo.py
python example/dynamic_scenario_demo.py
python example/stress_test_demo.py
```

### List All Examples
```bash
# List all available example scripts
ls example/*.py
```

## Code Quality Tools

### Format Code (Recommended)
```bash
# Format Python code with Black
black .

# Format specific file
black src/file.py
```

### Lint Code (Recommended)
```bash
# Lint with flake8
flake8 .

# Lint specific file
flake8 src/file.py
```

### Type Checking (Recommended)
```bash
# Run mypy for type checking
mypy .

# Check specific module
mypy src/module
```

## OpenSpec Workflow (Specification-Driven Development)

### List Changes and Specs
```bash
# List active changes
openspec list

# List all specifications
openspec list --specs

# Show change details
openspec show <change-id>

# Show spec details
openspec show <spec-id> --type spec
```

### Validate and Archive Changes
```bash
# Validate a change (always use --strict)
openspec validate <change-id> --strict

# Archive a completed change
openspec archive <change-id --yes

# Update instruction files
openspec update
```

## Git Commands (Common Workflow)

### Basic Git Operations
```bash
# Check status
git status

# Add files
git add .
git add <file>

# Commit changes
git commit -m "Your commit message"

# Push to remote
git push

# Pull from remote
git pull
```

### View History
```bash
# View commit log
git log --oneline

# View changes
git diff
git diff HEAD~1
```

## Utility Commands

### File Operations
```bash
# List files in directory
ls -la
ls -la test/
ls -la var/engines/

# Find files
find . -name "*.py" -type f
find test -name "test_*.py"

# Search in files
grep -r "def " var/engines/
grep -r "class " var/
```

### Check Python Environment
```bash
# Check Python version
python --version

# Check installed packages
pip list

# Show package info
pip show numpy
pip show pandas
```

### Run Python Interactively
```bash
# Start Python REPL
python

# Run script
python script.py

# Run module as script
python -m module_name
```

## File Structure Navigation

### Key Directories
```bash
# Source code
ls -la asset/
ls -la param/
ls -la priceenv/
ls -la var/
ls -la portfolio/

# Tests
ls -la test/

# Examples
ls -la example/

# Documentation
ls -la docs/  # if exists
cat README.md
cat CLAUDE.md
```

## Performance and Profiling

### Run with Timing
```bash
# Time execution
time python example/parametric_var_demo.py

# Profile with cProfile
python -m cProfile -o profile.prof script.py

# View profiling results
python -c "import pstats; pstats.Stats('profile.prof').sort_stats('cumulative').print_stats(20)"
```

### Memory Usage
```bash
# Check memory (macOS)
top -pid $(pgrep -f "python.*demo")
```

## Debugging

### Print Debugging
```python
# In code, use print statements
print(f"Variable value: {variable}")

# Use logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.debug("Debug message")
```

### Exception Debugging
```bash
# Run with traceback on error
python -m pdb script.py

# Post-mortem debugging
# After exception occurs:
python -c "import pdb; pdb.pm()"
```

## Data File Operations

### View Data Files
```bash
# View CSV
head data/file.csv

# View Excel file info
python -c "import pandas as pd; print(pd.ExcelFile('file.xlsx').sheet_names)"
```

## Best Practices

### Before Running Tests
```bash
# Ensure virtual environment is activated
source quantark/bin/activate

# Run from project root
cd /Users/fuxinyao/quant-ark

# Clear pytest cache
python -m pytest --cache-clear
```

### Before Running Examples
```bash
# Check dependencies are installed
pip list | grep -E "numpy|scipy|pandas"

# Run from correct directory
cd /Users/fuxinyao/quant-ark
python example/parametric_var_demo.py
```

### System Information
```bash
# macOS system info
sw_vers

# Python installation location
which python
python -c "import sys; print(sys.executable)"

# Check Darwin compatibility
uname -a
```
