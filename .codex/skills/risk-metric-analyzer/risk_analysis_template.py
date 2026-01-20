"""
Unified Risk Analysis Script Template

This template demonstrates the proper use of dependency checking and
folder management for risk metric analysis tasks.

Usage:
    python risk_analysis_template.py --product european_call --analysis spot_vol
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add skill directory to path for imports
SKILL_DIR = Path(__file__).parent
sys.path.insert(0, str(SKILL_DIR))

from dependency_checker import DependencyChecker, format_dependency_table
from folder_manager import FolderManager, get_analysis_folder_name


def check_dependencies(python_path: str = None) -> bool:
    """
    Step 0: Validate all required dependencies.

    Returns:
        True if all dependencies satisfied, False otherwise
    """
    print("=" * 60)
    print("Step 0: Dependency Validation")
    print("=" * 60)

    checker = DependencyChecker(python_path=python_path)
    missing_core, missing_optional, available = checker.check_all_dependencies()

    # Print status
    print(checker.get_status_report())

    # If core dependencies missing, ask to install
    if missing_core:
        print(format_dependency_table(missing_core, missing_optional))
        print("\n⚠️  Core dependencies are missing and must be installed.")

        response = input("\n❓ Install missing core dependencies? [Y/n]: ").strip().lower()

        if response in ['y', 'yes', '']:
            for dep in missing_core:
                print(f"\n  Installing {dep.name}...")
                success, msg = checker.install_dependency(dep)
                if success:
                    print(f"  ✓ {msg}")
                else:
                    print(f"  ✗ {msg}")
                    print(f"\n⚠️  Could not install {dep.name}. Please install manually:")
                    print(f"   {checker.get_install_command(dep)}")
                    return False
            print("\n✓ All dependencies installed successfully!")
        else:
            print("⚠️  Analysis cancelled - missing required dependencies.")
            return False

    print()
    return True


def setup_output_folder(product_info: dict, python_path: str = None) -> tuple:
    """
    Step 0.5: Organize output into dedicated task folder.

    Returns:
        Tuple of (analysis_folder_path, paths_dict) or (None, None) if cancelled
    """
    print("=" * 60)
    print("Step 0.5: Output Folder Organization")
    print("=" * 60)

    folder_mgr = FolderManager()

    # Get folder name from user
    analysis_folder, should_proceed = folder_mgr.get_folder_name_from_user(product_info)

    if not should_proceed:
        print("⚠️  Analysis cancelled by user.")
        return None, None

    # Get path shortcuts
    paths = folder_mgr.get_output_paths(analysis_folder)

    print(f"\n✓ Output folder configured:")
    print(f"  Scripts:        {paths['scripts']}")
    print(f"  Data:           {paths['data']}")
    print(f"  Visualizations: {paths['visualizations']}")
    print(f"  Reports:        {paths['reports']}")
    print()

    return analysis_folder, paths


def run_analysis(product_info: dict, paths: dict):
    """
    Main analysis function - Replace with actual analysis code.

    Args:
        product_info: Product configuration
        paths: Output path shortcuts from folder manager
    """
    print("=" * 60)
    print("Running Analysis")
    print("=" * 60)

    # TODO: Replace with actual analysis code
    # Example structure:

    # 1. Import QuantArk modules
    # from asset.equity.product.option import EuropeanVanillaOption
    # from asset.equity.engine.analytical import BlackScholesEngine
    # from asset.equity.riskmeasures import GreeksCalculator

    # 2. Create product and pricing environment
    # option = EuropeanVanillaOption(...)

    # 3. Calculate risk metrics
    # calculator = GreeksCalculator()
    # greeks = calculator.calculate_analytical_greeks(option, pricing_env)

    # 4. Save results to appropriate paths
    # import pandas as pd
    # df = pd.DataFrame([greeks])
    # df.to_csv(paths['data'] / 'greeks.csv', index=False)

    # 5. Generate visualizations (if requested)
    # import matplotlib.pyplot as plt
    # # ... create plots ...
    # plt.savefig(paths['visualizations'] / 'delta_plot.png')

    # 6. Generate report
    # report_path = paths['reports'] / 'analysis_report.pdf'
    # create_pdf_report(greeks, report_path)

    # Demo placeholder
    print(f"Product: {product_info.get('product_type', 'unknown')}")
    print(f"Analysis: {product_info.get('analysis_type', 'basic')}")
    print(f"\n⚠️  TODO: Implement actual analysis logic")

    # Save a placeholder summary
    summary_path = paths['reports'] / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Risk Analysis Summary\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Product: {product_info.get('product_type', 'unknown')}\n")
        f.write(f"Analysis: {product_info.get('analysis_type', 'basic')}\n")
        f.write(f"\n⚠️  This is a placeholder. Implement actual analysis logic.\n")

    print(f"\n✓ Placeholder summary saved to: {summary_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Risk Metric Analysis')
    parser.add_argument('--product', type=str, help='Product type (e.g., european_call)')
    parser.add_argument('--analysis', type=str, default='basic', help='Analysis type (e.g., spot_vol, basic)')
    parser.add_argument('--strike', type=float, help='Strike price')
    parser.add_argument('--maturity', type=float, help='Time to maturity (years)')
    parser.add_argument('--python', type=str, default='quantark/bin/python', help='Python executable')

    args = parser.parse_args()

    # Prepare product info
    product_info = {
        'product_type': args.product or 'european_call',
        'analysis_type': args.analysis,
    }

    if args.strike:
        product_info['strike'] = args.strike
    if args.maturity:
        product_info['maturity'] = args.maturity

    # Step 0: Check dependencies
    if not check_dependencies(python_path=args.python):
        return 1

    # Step 0.5: Setup output folder
    analysis_folder, paths = setup_output_folder(product_info, python_path=args.python)

    if not analysis_folder:
        return 1

    # Run analysis
    run_analysis(product_info, paths)

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print(f"Results saved to: {analysis_folder}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
