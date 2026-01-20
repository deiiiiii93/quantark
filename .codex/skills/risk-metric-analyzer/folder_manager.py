"""
Folder Manager for Risk Metric Analyzer Skill

Manages output folder organization with user-provided or auto-generated names,
handling conflicts and maintaining clean structure for multiple analysis runs.
"""
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List
import re


class FolderManager:
    """Manages output folder creation and organization for risk analysis."""

    def __init__(self, base_dir: Path = None):
        """
        Initialize the folder manager.

        Args:
            base_dir: Base directory for all risk analysis outputs.
                     Defaults to 'risk_metric_analysis/' in project root.
        """
        if base_dir is None:
            # Try to find project root
            current_path = Path.cwd()
            # Look for common project markers
            for parent in [current_path] + list(current_path.parents):
                if (parent / "asset").exists() or (parent / "CLAUDE.md").exists():
                    self.base_dir = parent / "risk_metric_analysis"
                    break
            else:
                self.base_dir = Path("risk_metric_analysis")
        else:
            self.base_dir = Path(base_dir)

        self.base_dir.mkdir(parents=True, exist_ok=True)

    def sanitize_folder_name(self, name: str) -> str:
        """
        Sanitize a user-provided folder name.

        Removes or replaces characters that are problematic in file names.

        Args:
            name: User-provided name

        Returns:
            Sanitized folder name
        """
        # Replace spaces with underscores
        name = name.strip().replace(" ", "_")
        # Remove special characters except alphanumeric, underscore, hyphen
        name = re.sub(r'[^\w\-]', '', name)
        # Limit length
        if len(name) > 50:
            name = name[:50]
        return name or "analysis"

    def generate_folder_name(self, product_info: dict, include_date: bool = True) -> str:
        """
        Auto-generate a folder name based on product/task information.

        Args:
            product_info: Dictionary containing product details
                - product_type: e.g., 'european_call', 'american_put'
                - strike: optional strike price
                - maturity: optional maturity
                - analysis_type: e.g., 'spot_vol', 'basic', 'full'
            include_date: Whether to append date to folder name

        Returns:
            Generated folder name
        """
        parts = []

        # Product type
        product_type = product_info.get('product_type', 'analysis')
        parts.append(product_type)

        # Add optional details
        if 'strike' in product_info:
            parts.append(f"K{product_info['strike']}")
        if 'maturity' in product_info:
            parts.append(f"T{product_info['maturity']}y")

        # Analysis type
        analysis_type = product_info.get('analysis_type', 'basic')
        parts.append(analysis_type)

        # Date
        if include_date:
            date_str = datetime.now().strftime("%Y%m%d")
            parts.append(date_str)

        return "_".join(parts)

    def generate_timestamp_name(self, base_name: str = "analysis") -> str:
        """
        Generate a folder name with full timestamp.

        Args:
            base_name: Base name for the folder

        Returns:
            Folder name with timestamp (e.g., 'european_call_20241230_143522')
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{timestamp}"

    def folder_exists(self, folder_name: str) -> Tuple[bool, Path]:
        """
        Check if a folder already exists.

        Args:
            folder_name: Name of the folder to check

        Returns:
            Tuple of (exists, full_path)
        """
        folder_path = self.base_dir / folder_name
        return folder_path.exists(), folder_path

    def list_existing_folders(self) -> List[str]:
        """List all existing analysis folders."""
        if not self.base_dir.exists():
            return []
        return [d.name for d in self.base_dir.iterdir() if d.is_dir() and d.name != '__pycache__']

    def count_existing_similar(self, pattern: str) -> int:
        """Count existing folders matching a pattern."""
        if not self.base_dir.exists():
            return 0
        try:
            return len(list(self.base_dir.glob(f"{pattern}*")))
        except:
            return 0

    def create_folder_structure(self, folder_name: str) -> Path:
        """
        Create the full folder structure for an analysis.

        Args:
            folder_name: Name of the analysis folder

        Returns:
            Path to the created analysis folder
        """
        analysis_dir = self.base_dir / folder_name

        # Create subdirectories
        subdirs = [
            "scripts",           # All generated scripts
            "data",              # CSV and data files
            "visualizations",    # All plots and charts
            "reports",           # Final reports (PDF, MD, etc.)
        ]

        for subdir in subdirs:
            (analysis_dir / subdir).mkdir(parents=True, exist_ok=True)

        # Create a README in the analysis folder
        readme_path = analysis_dir / "README.md"
        if not readme_path.exists():
            readme_content = f"""# {folder_name.replace('_', ' ').title()}

Risk analysis generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}

## Folder Structure

```
{folder_name}/
├── scripts/              # Generated calculation and visualization scripts
├── data/                 # CSV data files
├── visualizations/       # PNG plots and charts
├── reports/              # Final analysis reports
└── README.md             # This file
```

## Generated Files

"""
            readme_path.write_text(readme_content)

        return analysis_dir

    def suggest_folder_name(self, product_info: dict, existing_names: List[str] = None) -> str:
        """
        Suggest a unique folder name based on product info.

        Args:
            product_info: Product information dictionary
            existing_names: List of existing folder names (optional)

        Returns:
            Suggested unique folder name
        """
        if existing_names is None:
            existing_names = self.list_existing_folders()

        # Start with base generated name
        base_name = self.generate_folder_name(product_info)
        suggested = base_name

        # If it exists, add a counter
        counter = 1
        while suggested in existing_names:
            suggested = f"{base_name}_v{counter}"
            counter += 1

        return suggested

    def resolve_folder_conflict(self, folder_name: str) -> Tuple[str, bool]:
        """
        Resolve folder name conflicts with user input.

        Args:
            folder_name: Proposed folder name

        Returns:
            Tuple of (final_folder_name, should_create_new)
        """
        exists, folder_path = self.folder_exists(folder_name)

        if not exists:
            return folder_name, True

        # Conflict exists - provide options
        print(f"\n⚠️  Folder '{folder_name}' already exists at:")
        print(f"   {folder_path}")
        print()

        # Show what's in the existing folder
        existing_files = list(folder_path.rglob("*")) if folder_path.exists() else []
        file_count = len([f for f in existing_files if f.is_file()])

        if file_count > 0:
            print(f"   Contains {file_count} files from previous analysis")
            print()

        print("Options:")
        print("  1. Override - Delete existing folder and create new")
        print("  2. New name  - Create with a new name (suggested below)")
        print("  3. Cancel    - Cancel this operation")

        # Generate suggestion
        suggested = self.generate_timestamp_name(folder_name)
        print(f"\n  Suggested new name: {suggested}")

        while True:
            response = input("\n  Choose option [1/2/3] or provide custom name: ").strip()

            # Override
            if response in ['1', 'override']:
                confirm = input(f"  Confirm delete '{folder_name}'? [yes/NO]: ").strip().lower()
                if confirm in ['yes', 'y']:
                    # Delete existing folder
                    import shutil
                    shutil.rmtree(folder_path)
                    print(f"  ✓ Deleted existing folder")
                    return folder_name, True
                else:
                    print("  Override cancelled")
                    continue

            # New name
            elif response in ['2', 'new', '']:
                return suggested, True

            # Cancel
            elif response in ['3', 'cancel', 'c']:
                return "", False

            # Custom name provided
            elif response:
                custom_name = self.sanitize_folder_name(response)
                if custom_name != response:
                    print(f"  Note: Name sanitized to '{custom_name}'")
                return custom_name, True

            else:
                print("  Invalid option, please try again")

    def get_folder_name_from_user(self, product_info: dict = None,
                                  default_suggestion: str = None) -> Tuple[Path, bool]:
        """
        Interactively get a folder name from the user.

        Args:
            product_info: Product info for auto-generation
            default_suggestion: Pre-computed suggestion (optional)

        Returns:
            Tuple of (folder_path, should_proceed)
        """
        print("\n" + "=" * 60)
        print("Output Folder Configuration")
        print("=" * 60)

        # Generate default suggestion
        if default_suggestion is None:
            if product_info:
                default_suggestion = self.suggest_folder_name(product_info)
            else:
                default_suggestion = self.generate_timestamp_name()

        existing = self.list_existing_folders()
        if existing:
            print(f"\nExisting analysis folders:")
            for name in existing[-5:]:  # Show last 5
                print(f"  - {name}")
            if len(existing) > 5:
                print(f"  ... and {len(existing) - 5} more")

        print(f"\n📁 Base directory: {self.base_dir}")
        print(f"📝 Suggested name:  {default_suggestion}")

        response = input("\nFolder name [Press Enter for suggestion, or type custom name]: ").strip()

        if not response:
            folder_name = default_suggestion
        else:
            folder_name = self.sanitize_folder_name(response)

        # Check for conflicts
        exists, folder_path = self.folder_exists(folder_name)

        if exists:
            folder_name, should_proceed = self.resolve_folder_conflict(folder_name)
            if not should_proceed:
                return None, False
            folder_path = self.base_dir / folder_name
        else:
            should_proceed = True

        # Create the folder structure
        if should_proceed:
            created_path = self.create_folder_structure(folder_name)
            print(f"\n✓ Created output folder: {created_path}")
            return created_path, True

        return None, False

    def get_subfolder_path(self, analysis_folder: Path, file_type: str) -> Path:
        """
        Get the appropriate subfolder path for a file type.

        Args:
            analysis_folder: The main analysis folder
            file_type: Type of file ('script', 'data', 'visualization', 'report')

        Returns:
            Path to the appropriate subfolder
        """
        subfolder_map = {
            'script': 'scripts',
            'data': 'data',
            'visualization': 'visualizations',
            'report': 'reports',
        }

        subfolder = subfolder_map.get(file_type, 'reports')
        return analysis_folder / subfolder

    def get_output_paths(self, analysis_folder: Path) -> dict:
        """
        Get all output path shortcuts for an analysis.

        Args:
            analysis_folder: The main analysis folder

        Returns:
            Dictionary with path shortcuts
        """
        return {
            'base': analysis_folder,
            'scripts': analysis_folder / 'scripts',
            'data': analysis_folder / 'data',
            'visualizations': analysis_folder / 'visualizations',
            'reports': analysis_folder / 'reports',
        }


def get_analysis_folder_name(product_type: str, analysis_type: str = "basic",
                            strike: float = None, maturity: float = None,
                            include_date: bool = True) -> str:
    """
    Quick helper to generate a folder name.

    Args:
        product_type: Type of product (e.g., 'european_call')
        analysis_type: Type of analysis (e.g., 'spot_vol', 'basic')
        strike: Optional strike price
        maturity: Optional maturity in years
        include_date: Whether to include date in name

    Returns:
        Generated folder name
    """
    manager = FolderManager()
    product_info = {
        'product_type': product_type,
        'analysis_type': analysis_type,
    }
    if strike:
        product_info['strike'] = strike
    if maturity:
        product_info['maturity'] = maturity

    return manager.generate_folder_name(product_info, include_date)


if __name__ == "__main__":
    # Test the folder manager
    manager = FolderManager()

    print("=== Folder Manager Test ===")
    print(f"Base directory: {manager.base_dir}")
    print(f"\nExisting folders: {manager.list_existing_folders()}")

    # Example product info
    product_info = {
        'product_type': 'european_call',
        'strike': 100,
        'maturity': 1.0,
        'analysis_type': 'spot_vol'
    }

    suggested = manager.suggest_folder_name(product_info)
    print(f"\nSuggested folder: {suggested}")

    timestamp_name = manager.generate_timestamp_name("european_call")
    print(f"Timestamp folder: {timestamp_name}")
