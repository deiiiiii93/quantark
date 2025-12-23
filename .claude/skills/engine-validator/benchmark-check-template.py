"""
Benchmark Check Template for Engine Validation
===============================================

This template compares analytical engine prices against Monte Carlo benchmark.
Copy and customize for specific engine validation.

Usage:
    1. Copy this file to: asset/<type>/engine/validation/script/benchmark_check_<engine_name>.py
    2. Replace placeholders with actual imports and implementations
    3. Configure test cases as needed
    4. Run: python asset/<type>/engine/validation/script/benchmark_check_<engine_name>.py
"""
import numpy as np
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import time

# Add project root to path
sys.path.insert(0, '.')

# ==============================================================================
# REPLACE THESE IMPORTS WITH ACTUAL ENGINE/PRODUCT IMPORTS
# ==============================================================================
# Analytical engine
# from asset.equity.product.option.<product> import <Product>
# from asset.equity.engine.analytical.<engine> import <AnalyticalEngine>

# Monte Carlo benchmark
# from asset.equity.engine.mc.<mc_engine> import <MCEngine>

from priceenv.pricing_environment import PricingEnvironment
from param.spot_quote import SpotQuote
from param.rate_curve import FlatRateCurve
from param.vol_surface import FlatVolSurface
from param.dividend import ContinuousDividendYield


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Default pass criteria: 5% relative error
DEFAULT_TOLERANCE = 0.05

# Monte Carlo parameters
MC_PATHS = 100000
MC_STEPS = 252  # Daily steps for 1 year


class TestStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass
class BenchmarkResult:
    case_name: str
    analytical: float
    mc: float
    mc_std_error: Optional[float]
    rel_error: float
    status: TestStatus
    params: Dict[str, Any] = field(default_factory=dict)


class BenchmarkResults:
    """Collects and reports benchmark check results."""
    
    def __init__(self, tolerance: float = DEFAULT_TOLERANCE):
        self.tolerance = tolerance
        self.results: List[BenchmarkResult] = []
        self.execution_time: float = 0.0
    
    def add_result(self, case_name: str, analytical: float, mc: float, 
                   mc_std_error: float = None, params: Dict = None):
        """Add a benchmark result."""
        if mc != 0:
            rel_error = abs(analytical - mc) / abs(mc)
        else:
            rel_error = abs(analytical - mc) if analytical != 0 else 0.0
        
        status = TestStatus.PASS if rel_error <= self.tolerance else TestStatus.FAIL
        
        self.results.append(BenchmarkResult(
            case_name=case_name,
            analytical=analytical,
            mc=mc,
            mc_std_error=mc_std_error,
            rel_error=rel_error,
            status=status,
            params=params or {}
        ))
    
    def add_skip(self, case_name: str, reason: str):
        """Add a skipped test case."""
        self.results.append(BenchmarkResult(
            case_name=case_name,
            analytical=0.0,
            mc=0.0,
            mc_std_error=None,
            rel_error=0.0,
            status=TestStatus.SKIP,
            params={'skip_reason': reason}
        ))
    
    @property
    def passed(self) -> List[BenchmarkResult]:
        return [r for r in self.results if r.status == TestStatus.PASS]
    
    @property
    def failed(self) -> List[BenchmarkResult]:
        return [r for r in self.results if r.status == TestStatus.FAIL]
    
    @property
    def skipped(self) -> List[BenchmarkResult]:
        return [r for r in self.results if r.status == TestStatus.SKIP]
    
    def summary(self) -> bool:
        """Print summary and return True if all tests passed."""
        total = len(self.results)
        n_pass = len(self.passed)
        n_fail = len(self.failed)
        n_skip = len(self.skipped)
        
        # Header
        print(f"\n{'='*90}")
        print(f"BENCHMARK CHECK SUMMARY")
        print(f"{'='*90}")
        print(f"Tolerance: {self.tolerance*100:.1f}%")
        print(f"MC Paths: {MC_PATHS:,}")
        print(f"MC Steps: {MC_STEPS}")
        print(f"Execution Time: {self.execution_time:.2f}s")
        print(f"{'='*90}")
        
        # Results table
        print(f"\n{'Case':<35} {'Analytical':>12} {'MC':>12} {'StdErr':>10} {'Error':>10} {'Status':>8}")
        print(f"{'-'*90}")
        
        for r in self.results:
            if r.status == TestStatus.SKIP:
                print(f"{r.case_name:<35} {'--':>12} {'--':>12} {'--':>10} {'--':>10} {'SKIP':>8}")
                print(f"  └─ Reason: {r.params.get('skip_reason', 'N/A')}")
            else:
                std_err_str = f"{r.mc_std_error:.4f}" if r.mc_std_error else "--"
                print(f"{r.case_name:<35} {r.analytical:>12.4f} {r.mc:>12.4f} {std_err_str:>10} "
                      f"{r.rel_error*100:>9.2f}% {r.status.value:>8}")
        
        print(f"{'-'*90}")
        
        # Summary statistics
        print(f"\nTotal Tests: {total}")
        print(f"  Passed:  {n_pass:3d} ({100*n_pass/total:.1f}%)" if total > 0 else "  Passed:  0")
        print(f"  Failed:  {n_fail:3d} ({100*n_fail/total:.1f}%)" if total > 0 else "  Failed:  0")
        print(f"  Skipped: {n_skip:3d}")
        
        # Error statistics for passed tests
        if self.passed:
            errors = [r.rel_error for r in self.passed]
            print(f"\nError Statistics (passed tests):")
            print(f"  Mean:   {np.mean(errors)*100:.2f}%")
            print(f"  Median: {np.median(errors)*100:.2f}%")
            print(f"  Max:    {np.max(errors)*100:.2f}%")
        
        # Failed tests details
        if self.failed:
            print(f"\n{'='*90}")
            print("FAILED TESTS:")
            print(f"{'='*90}")
            for r in self.failed:
                print(f"\n  [{r.case_name}]")
                print(f"    Analytical: {r.analytical:.6f}")
                print(f"    MC:         {r.mc:.6f}")
                print(f"    Error:      {r.rel_error*100:.2f}% (tolerance: {self.tolerance*100:.1f}%)")
                if r.params:
                    print(f"    Params:     {r.params}")
        
        print(f"\n{'='*90}")
        overall = "PASSED" if n_fail == 0 else "FAILED"
        print(f"OVERALL STATUS: {overall}")
        print(f"{'='*90}\n")
        
        return n_fail == 0


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def create_pricing_env(spot: float, rate: float, vol: float, div: float = 0.0) -> PricingEnvironment:
    """Create a pricing environment with flat parameters."""
    return PricingEnvironment(
        spot=SpotQuote(spot),
        rate_curve=FlatRateCurve(rate),
        vol_surface=FlatVolSurface(vol),
        dividend=ContinuousDividendYield(div)
    )


# ==============================================================================
# TEST CASE DEFINITIONS
# ==============================================================================

# Define test cases as a list of dictionaries
# Each test case should contain:
#   - name: descriptive name
#   - spot, strike, rate, vol, div, maturity: market parameters
#   - is_call: True for call, False for put
#   - Additional product-specific parameters

TEST_CASES = [
    # ATM cases
    {
        'name': 'ATM Call T=1Y',
        'spot': 100.0, 'strike': 100.0, 'rate': 0.05, 'vol': 0.2, 'div': 0.0,
        'maturity': 1.0, 'is_call': True
    },
    {
        'name': 'ATM Put T=1Y',
        'spot': 100.0, 'strike': 100.0, 'rate': 0.05, 'vol': 0.2, 'div': 0.0,
        'maturity': 1.0, 'is_call': False
    },
    # ITM cases
    {
        'name': 'ITM Call K=90',
        'spot': 100.0, 'strike': 90.0, 'rate': 0.05, 'vol': 0.2, 'div': 0.0,
        'maturity': 1.0, 'is_call': True
    },
    {
        'name': 'ITM Put K=110',
        'spot': 100.0, 'strike': 110.0, 'rate': 0.05, 'vol': 0.2, 'div': 0.0,
        'maturity': 1.0, 'is_call': False
    },
    # OTM cases
    {
        'name': 'OTM Call K=110',
        'spot': 100.0, 'strike': 110.0, 'rate': 0.05, 'vol': 0.2, 'div': 0.0,
        'maturity': 1.0, 'is_call': True
    },
    {
        'name': 'OTM Put K=90',
        'spot': 100.0, 'strike': 90.0, 'rate': 0.05, 'vol': 0.2, 'div': 0.0,
        'maturity': 1.0, 'is_call': False
    },
    # Different maturities
    {
        'name': 'ATM Call T=0.25Y',
        'spot': 100.0, 'strike': 100.0, 'rate': 0.05, 'vol': 0.2, 'div': 0.0,
        'maturity': 0.25, 'is_call': True
    },
    {
        'name': 'ATM Call T=2Y',
        'spot': 100.0, 'strike': 100.0, 'rate': 0.05, 'vol': 0.2, 'div': 0.0,
        'maturity': 2.0, 'is_call': True
    },
    # Different volatilities
    {
        'name': 'Low Vol σ=0.1',
        'spot': 100.0, 'strike': 100.0, 'rate': 0.05, 'vol': 0.1, 'div': 0.0,
        'maturity': 1.0, 'is_call': True
    },
    {
        'name': 'High Vol σ=0.4',
        'spot': 100.0, 'strike': 100.0, 'rate': 0.05, 'vol': 0.4, 'div': 0.0,
        'maturity': 1.0, 'is_call': True
    },
    # With dividend
    {
        'name': 'With Dividend q=3%',
        'spot': 100.0, 'strike': 100.0, 'rate': 0.05, 'vol': 0.2, 'div': 0.03,
        'maturity': 1.0, 'is_call': True
    },
    # Deep ITM/OTM
    {
        'name': 'Deep ITM Call K=70',
        'spot': 100.0, 'strike': 70.0, 'rate': 0.05, 'vol': 0.2, 'div': 0.0,
        'maturity': 1.0, 'is_call': True
    },
    {
        'name': 'Deep OTM Call K=130',
        'spot': 100.0, 'strike': 130.0, 'rate': 0.05, 'vol': 0.2, 'div': 0.0,
        'maturity': 1.0, 'is_call': True
    },
]


# ==============================================================================
# BENCHMARK RUNNER
# ==============================================================================

def run_benchmark(AnalyticalEngine, MCEngine, Product, 
                  test_cases: List[Dict] = None,
                  tolerance: float = DEFAULT_TOLERANCE,
                  mc_paths: int = MC_PATHS,
                  mc_steps: int = MC_STEPS) -> BenchmarkResults:
    """
    Run benchmark comparison between analytical and MC engines.
    
    Args:
        AnalyticalEngine: Analytical pricing engine class
        MCEngine: Monte Carlo pricing engine class
        Product: Product class
        test_cases: List of test case dictionaries
        tolerance: Pass/fail tolerance (default 5%)
        mc_paths: Number of MC paths
        mc_steps: Number of MC time steps
    
    Returns:
        BenchmarkResults object
    """
    results = BenchmarkResults(tolerance=tolerance)
    test_cases = test_cases or TEST_CASES
    
    print(f"Running Benchmark Checks...")
    print(f"Analytical Engine: {AnalyticalEngine.__name__}")
    print(f"MC Engine: {MCEngine.__name__}")
    print(f"Product: {Product.__name__}")
    print(f"Tolerance: {tolerance*100:.1f}%")
    print(f"MC Paths: {mc_paths:,}")
    print("=" * 70)
    
    start_time = time.time()
    
    analytical_engine = AnalyticalEngine()
    mc_engine = MCEngine(n_paths=mc_paths, n_steps=mc_steps)
    
    for i, case in enumerate(test_cases, 1):
        case_name = case.get('name', f'Case {i}')
        print(f"  [{i}/{len(test_cases)}] {case_name}...", end=" ", flush=True)
        
        try:
            # Create pricing environment
            env = create_pricing_env(
                spot=case['spot'],
                rate=case['rate'],
                vol=case['vol'],
                div=case.get('div', 0.0)
            )
            
            # Create product
            product = Product(
                strike=case['strike'],
                maturity=case['maturity'],
                is_call=case['is_call']
            )
            
            # Price with analytical engine
            analytical_price = analytical_engine.price(product, env)
            
            # Price with MC engine
            mc_price = mc_engine.price(product, env)
            
            # Get standard error if available
            mc_std_error = getattr(mc_engine, 'last_std_error', None)
            
            # Add result
            results.add_result(
                case_name=case_name,
                analytical=analytical_price,
                mc=mc_price,
                mc_std_error=mc_std_error,
                params=case
            )
            
            print("Done")
        
        except Exception as e:
            print(f"Error: {str(e)}")
            results.add_skip(case_name, f"Exception: {str(e)}")
    
    results.execution_time = time.time() - start_time
    
    return results


# ==============================================================================
# ADDITIONAL TEST GENERATORS
# ==============================================================================

def generate_grid_cases(spots: List[float], strikes: List[float], 
                        vols: List[float], maturities: List[float],
                        rate: float = 0.05, div: float = 0.0) -> List[Dict]:
    """Generate a grid of test cases."""
    cases = []
    for spot in spots:
        for strike in strikes:
            for vol in vols:
                for maturity in maturities:
                    for is_call in [True, False]:
                        option_type = "Call" if is_call else "Put"
                        moneyness = "ATM" if spot == strike else ("ITM" if (is_call and spot > strike) or (not is_call and spot < strike) else "OTM")
                        
                        cases.append({
                            'name': f'{option_type} {moneyness} S={spot} K={strike} σ={vol} T={maturity}',
                            'spot': spot, 'strike': strike, 'rate': rate, 'vol': vol, 'div': div,
                            'maturity': maturity, 'is_call': is_call
                        })
    return cases


# ==============================================================================
# EXPORT RESULTS
# ==============================================================================

def export_to_csv(results: BenchmarkResults, filepath: str):
    """Export results to CSV file."""
    import csv
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Case', 'Analytical', 'MC', 'Std Error', 'Rel Error (%)', 'Status'])
        
        for r in results.results:
            std_err = r.mc_std_error if r.mc_std_error else ''
            writer.writerow([
                r.case_name,
                f'{r.analytical:.6f}',
                f'{r.mc:.6f}',
                f'{std_err:.6f}' if std_err else '',
                f'{r.rel_error*100:.4f}',
                r.status.value
            ])
    
    print(f"Results exported to: {filepath}")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    # REPLACE WITH ACTUAL IMPORTS
    # from asset.equity.product.option.european_vanilla_option import EuropeanVanillaOption
    # from asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
    # from asset.equity.engine.mc.euro_mc_engine import EuroMCEngine
    
    # results = run_benchmark(
    #     AnalyticalEngine=BlackScholesEngine,
    #     MCEngine=EuroMCEngine,
    #     Product=EuropeanVanillaOption,
    #     tolerance=0.05
    # )
    # success = results.summary()
    
    print("This is a template file.")
    print("Please copy and customize for your specific engine validation.")
    print("\nUsage:")
    print("  1. Copy to: asset/<type>/engine/validation/script/benchmark_check_<engine>.py")
    print("  2. Update imports for your analytical engine, MC engine, and product")
    print("  3. Customize TEST_CASES as needed")
    print("  4. Run: python <path_to_script>")
    
    sys.exit(0)
