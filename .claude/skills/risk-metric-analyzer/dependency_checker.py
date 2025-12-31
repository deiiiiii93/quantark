"""
Dependency Checker for Risk Metric Analyzer Skill

Validates that all required third-party libraries are available
before running risk analysis calculations.
"""
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional


class DependencyInfo:
    """Information about a required dependency."""

    def __init__(self, name: str, import_name: str, pip_name: str,
                 required_for: str, optional: bool = False):
        self.name = name  # Display name (e.g., "NumPy")
        self.import_name = import_name  # Import name for checking (e.g., "numpy")
        self.pip_name = pip_name  # Name for pip install (e.g., "numpy")
        self.required_for = required_for  # Description of what it's used for
        self.optional = optional  # Whether it's optional or required


# Core dependencies for risk metric analysis
CORE_DEPENDENCIES = [
    DependencyInfo(
        name="NumPy",
        import_name="numpy",
        pip_name="numpy",
        required_for="numerical calculations and array operations"
    ),
    DependencyInfo(
        name="Pandas",
        import_name="pandas",
        pip_name="pandas",
        required_for="data manipulation and CSV output"
    ),
    DependencyInfo(
        name="Matplotlib",
        import_name="matplotlib",
        pip_name="matplotlib",
        required_for="visualization and PDF report generation"
    ),
]

# Optional dependencies for enhanced features
OPTIONAL_DEPENDENCIES = [
    DependencyInfo(
        name="Seaborn",
        import_name="seaborn",
        pip_name="seaborn",
        required_for="enhanced statistical visualizations",
        optional=True
    ),
    DependencyInfo(
        name="ReportLab",
        import_name="reportlab",
        pip_name="reportlab",
        required_for="PDF generation via ReportLab (alternative to matplotlib)",
        optional=True
    ),
]


class DependencyChecker:
    """Checks for and manages dependency installation."""

    def __init__(self, python_path: Optional[str] = None):
        """
        Initialize the dependency checker.

        Args:
            python_path: Path to Python executable for package installation.
                         If None, uses sys.executable.
        """
        self.python_path = python_path or sys.executable
        self.missing_core = []
        self.missing_optional = []
        self.available_optional = []

    def check_import(self, import_name: str) -> bool:
        """Check if a module can be imported."""
        try:
            __import__(import_name)
            return True
        except ImportError:
            return False

    def check_all_dependencies(self) -> Tuple[List[DependencyInfo], List[DependencyInfo], List[DependencyInfo]]:
        """
        Check all dependencies.

        Returns:
            Tuple of (missing_core, missing_optional, available_optional)
        """
        # Check core dependencies
        self.missing_core = []
        for dep in CORE_DEPENDENCIES:
            if not self.check_import(dep.import_name):
                self.missing_core.append(dep)

        # Check optional dependencies
        self.missing_optional = []
        self.available_optional = []
        for dep in OPTIONAL_DEPENDENCIES:
            if self.check_import(dep.import_name):
                self.available_optional.append(dep)
            else:
                self.missing_optional.append(dep)

        return self.missing_core, self.missing_optional, self.available_optional

    def get_install_command(self, dependency: DependencyInfo) -> str:
        """Get the pip install command for a dependency."""
        return f"{self.python_path} -m pip install {dependency.pip_name}"

    def install_dependency(self, dependency: DependencyInfo) -> Tuple[bool, str]:
        """
        Attempt to install a dependency.

        Returns:
            Tuple of (success, message)
        """
        try:
            cmd = [self.python_path, "-m", "pip", "install", dependency.pip_name, "--quiet"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode == 0:
                # Verify it can be imported
                if self.check_import(dependency.import_name):
                    return True, f"{dependency.name} installed successfully"
                else:
                    return False, f"{dependency.name} installed but cannot be imported"
            else:
                error_msg = result.stderr.strip() or result.stdout.strip()
                return False, f"Installation failed: {error_msg[:200]}"

        except subprocess.TimeoutExpired:
            return False, "Installation timed out"
        except Exception as e:
            return False, f"Installation error: {str(e)}"

    def get_status_report(self) -> str:
        """Generate a human-readable status report."""
        missing_core, missing_optional, available = self.check_all_dependencies()

        lines = ["=" * 60]
        lines.append("Dependency Status Check")
        lines.append("=" * 60)

        # Core dependencies
        lines.append("\nCore Dependencies (Required):")
        if not missing_core:
            lines.append("  ✓ All core dependencies installed")
        else:
            lines.append(f"  ⚠ Missing {len(missing_core)} core dependencies:")
            for dep in missing_core:
                lines.append(f"    - {dep.name}: {dep.required_for}")

        # Optional dependencies
        lines.append("\nOptional Dependencies:")
        if available:
            lines.append(f"  ✓ {len(available)} optional dependencies available:")
            for dep in available:
                lines.append(f"    - {dep.name}: {dep.required_for}")

        if missing_optional:
            lines.append(f"  ⚠ {len(missing_optional)} optional dependencies not installed:")
            for dep in missing_optional:
                lines.append(f"    - {dep.name}: {dep.required_for}")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)

    def can_proceed_with_analysis(self, need_pdf: bool = True) -> Tuple[bool, str]:
        """
        Check if analysis can proceed.

        Args:
            need_pdf: Whether PDF generation is required

        Returns:
            Tuple of (can_proceed, message)
        """
        missing_core, missing_optional, _ = self.check_all_dependencies()

        if missing_core:
            missing_names = ", ".join([d.name for d in missing_core])
            return False, (
                f"Cannot proceed: Missing core dependencies: {missing_names}. "
                f"Please install them to continue."
            )

        # PDF generation requires matplotlib (which is a core dep)
        if need_pdf:
            # Matplotlib can generate PDFs, so we're good
            pass

        optional_info = ""
        if missing_optional:
            missing_names = ", ".join([d.name for d in missing_optional])
            optional_info = (
                f"\n\nNote: Optional dependencies not installed: {missing_names}. "
                f"Analysis will proceed with reduced functionality."
            )

        return True, f"All core dependencies satisfied.{optional_info}"


def format_dependency_table(missing_core: List[DependencyInfo],
                            missing_optional: List[DependencyInfo]) -> str:
    """Format dependencies into a table for user confirmation."""
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("Required Packages Installation Summary")
    lines.append("=" * 60)

    if missing_core:
        lines.append("\nCore Dependencies (REQUIRED for basic analysis):")
        lines.append(f"{'Package':<15} {'Used For':<40}")
        lines.append("-" * 60)
        for dep in missing_core:
            lines.append(f"{dep.name:<15} {dep.required_for:<40}")

    if missing_optional:
        lines.append("\nOptional Dependencies (Enhanced features):")
        lines.append(f"{'Package':<15} {'Used For':<40}")
        lines.append("-" * 60)
        for dep in missing_optional:
            lines.append(f"{dep.name:<15} {dep.required_for:<40}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def print_dependency_check_results():
    """Print dependency check results to console."""
    checker = DependencyChecker()
    print(checker.get_status_report())


def check_and_prompt_for_installation(python_path: Optional[str] = None) -> bool:
    """
    Check dependencies and prompt user for installation if needed.

    Returns:
        True if all dependencies are available or installation successful,
        False if critical dependencies remain missing.
    """
    checker = DependencyChecker(python_path)
    missing_core, missing_optional, available = checker.check_all_dependencies()

    # Print current status
    print(checker.get_status_report())

    # If everything is installed, return True
    if not missing_core and not missing_optional:
        print("\n✓ All dependencies satisfied. Proceeding with analysis...\n")
        return True

    # If core dependencies are missing, we MUST install them
    if missing_core:
        print(format_dependency_table(missing_core, missing_optional))
        print("\n⚠ Core dependencies are missing and must be installed.")
        return False  # Let caller handle the user prompt

    # Only optional dependencies missing - we can proceed
    if missing_optional and not missing_core:
        print("\n✓ Core dependencies satisfied. Proceeding with basic analysis...")
        print("  (Optional enhancements are not installed)\n")
        return True

    return True


if __name__ == "__main__":
    # Run dependency check when executed directly
    print_dependency_check_results()
